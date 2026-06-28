"""Threshold-triggered per-env/per-term reward anomaly logging for FastSAC."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

if TYPE_CHECKING:
    from src.rl.adapters.mjlab_holosoma_env import MjlabHolosomaEnvAdapter
    from src.rl.config.fast_sac_cfg import HolosomaFastSACRunnerCfg

_ANOMALY_FILENAME = "reward_anomalies.jsonl"


@dataclass(frozen=True)
class RewardAnomalyCfg:
    """Configuration for reward anomaly JSONL logging."""

    enabled: bool = True
    abs_threshold: float = 1e4
    max_envs: int = 32
    log_non_finite: bool = True

    @classmethod
    def from_runner(cls, cfg: HolosomaFastSACRunnerCfg) -> RewardAnomalyCfg:
        return cls(
            enabled=cfg.reward_anomaly_log_enabled,
            abs_threshold=cfg.reward_anomaly_abs_threshold,
            max_envs=cfg.reward_anomaly_max_envs,
            log_non_finite=cfg.reward_anomaly_log_non_finite,
        )


def _should_trigger(
    rewards: torch.Tensor,
    threshold: float,
    log_non_finite: bool,
) -> bool:
    if log_non_finite and not rewards.isfinite().all():
        return True
    if rewards.abs().gt(threshold).any():
        return True
    if abs(rewards.mean().item()) > threshold:
        return True
    return False


def _select_env_ids(
    rewards: torch.Tensor,
    threshold: float,
    max_envs: int,
    log_non_finite: bool,
) -> torch.Tensor:
    r = rewards.float()
    finite = r.isfinite()
    per_env_bad = r.abs().gt(threshold) | (~finite if log_non_finite else torch.zeros_like(finite))
    bad_ids = per_env_bad.nonzero(as_tuple=False).squeeze(-1)
    if bad_ids.numel() == 0:
        # Mean-only trigger: log top-|reward| envs for context.
        k = min(max_envs, r.numel())
        _, top_idx = torch.topk(r.abs(), k=k, largest=True)
        return top_idx
    if bad_ids.dim() == 0:
        bad_ids = bad_ids.unsqueeze(0)
    order = torch.argsort(r[bad_ids].abs(), descending=True)
    bad_ids = bad_ids[order]
    return bad_ids[:max_envs]


def _build_record(
    step: int,
    rewards: torch.Tensor,
    adapter: MjlabHolosomaEnvAdapter,
    cfg: RewardAnomalyCfg,
    overflow_lines: int,
    rank: int,
) -> dict[str, Any]:
    r = rewards.float()
    env_ids = _select_env_ids(r, cfg.abs_threshold, cfg.max_envs, cfg.log_non_finite)
    term_names, term_values = adapter.get_step_reward_breakdown(env_ids)

    episode_lengths = adapter._env.episode_length_buf[env_ids].detach().cpu()
    dones = adapter.reset_buf[env_ids].detach().cpu()
    totals = r[env_ids].detach().cpu()
    terms_cpu = term_values.detach().cpu()

    env_records: list[dict[str, Any]] = []
    for row_idx, env_id in enumerate(env_ids.tolist()):
        term_map = {
            name: float(terms_cpu[row_idx, col_idx].item())
            for col_idx, name in enumerate(term_names)
        }
        total = float(totals[row_idx].item())
        env_records.append(
            {
                "env_id": int(env_id),
                "total_reward": total,
                "terms_sum": float(sum(term_map.values())),
                "episode_length": int(episode_lengths[row_idx].item()),
                "done": int(dones[row_idx].item()),
                "terms": term_map,
            }
        )

    per_env_bad = r.abs().gt(cfg.abs_threshold) | (
        ~r.isfinite() if cfg.log_non_finite else torch.zeros_like(r, dtype=torch.bool)
    )

    return {
        "step": int(step),
        "rank": int(rank),
        "threshold": float(cfg.abs_threshold),
        "mean_reward": float(r.mean().item()),
        "min_reward": float(r.min().item()),
        "max_reward": float(r.max().item()),
        "bad_count": int(per_env_bad.sum().item()),
        "num_envs": int(r.numel()),
        "overflow_lines": int(overflow_lines),
        "term_names": list(term_names),
        "envs": env_records,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def maybe_log_reward_anomaly(
    *,
    step: int,
    rewards: torch.Tensor,
    adapter: MjlabHolosomaEnvAdapter,
    log_dir: str | Path,
    cfg: RewardAnomalyCfg | None,
    overflow_lines: int = 0,
    is_main_process: bool = True,
    rank: int = 0,
) -> bool:
    """Log per-env/per-term reward breakdown when an anomaly is detected.

    Returns True if a record was written.
    """
    if cfg is None or not cfg.enabled:
        return False

    with torch.no_grad():
        r = rewards.double()
        if not _should_trigger(r, cfg.abs_threshold, cfg.log_non_finite):
            return False

        record = _build_record(step, r, adapter, cfg, overflow_lines, rank)

    logger.warning(
        f"step={step} reward anomaly: "
        f"min={record['min_reward']:.4g} max={record['max_reward']:.4g} "
        f"mean={record['mean_reward']:.4g} bad_count={record['bad_count']}"
    )
    if record["envs"]:
        first = record["envs"][0]
        logger.warning(
            f"top bad env_id={first['env_id']} total={first['total_reward']:.4g} "
            f"terms_sum={first['terms_sum']:.4g}"
        )

    if is_main_process:
        log_path = Path(log_dir) / _ANOMALY_FILENAME
        _append_jsonl(log_path, record)

    return True


def _legacy_reward_warning(step: int, rewards: torch.Tensor, threshold: float = 1e4) -> None:
    """Fallback when this module is unavailable from the holosoma agent."""
    with torch.no_grad():
        r = rewards.double()
        bad = r.abs() > threshold
        if bad.any() or not r.isfinite().all():
            logger.warning(
                f"step={step} reward stats: "
                f"min={r.min().item():.4g} max={r.max().item():.4g} "
                f"mean={r.mean().item():.4g} "
                f"bad_count={bad.sum().item()}"
            )
            idx = bad.nonzero(as_tuple=False).squeeze(-1)[:10]
            logger.warning(f"bad env ids (first 10): {idx.tolist()}")
            logger.warning(f"bad rewards: {r[idx].tolist()}")


def log_reward_anomaly_from_agent(
    *,
    step: int,
    rewards: torch.Tensor,
    agent: Any,
    overflow_lines: int = 0,
) -> None:
    """Entry point used by FastSACAgent; handles ImportError fallback internally."""
    cfg = getattr(agent, "reward_anomaly_cfg", None)
    rank = int(os.environ.get("RANK", "0"))
    try:
        maybe_log_reward_anomaly(
            step=step,
            rewards=rewards,
            adapter=agent.unwrapped_env,
            log_dir=agent.log_dir,
            cfg=cfg,
            overflow_lines=overflow_lines,
            is_main_process=getattr(agent, "is_main_process", True),
            rank=rank,
        )
    except Exception as exc:
        logger.warning(f"reward anomaly logging failed: {exc}")
        threshold = cfg.abs_threshold if cfg is not None else 1e4
        _legacy_reward_warning(step, rewards, threshold=threshold)
