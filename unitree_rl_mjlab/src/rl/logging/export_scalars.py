"""Export training scalars to JSONL for offline analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.rl.logging.training_metrics import append_training_metric
from src.rl.logging.wall_clock_reward import (
    append_reward_wall_clock,
    elapsed_wall_time_s,
)


def _to_json_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().cpu().mean().item())
    return float(value)


def expand_ppo_compatible_keys(scalars: dict[str, float]) -> dict[str, float]:
    """Add PPO-canonical aliases alongside FastSAC-native keys."""
    out: dict[str, float] = {}
    for key, value in scalars.items():
        out[key] = _to_json_scalar(value)

    for key, value in list(out.items()):
        if key.startswith("Env/"):
            canonical = key[len("Env/") :]
            out.setdefault(canonical, value)

        if key.startswith("Episode/rew_"):
            reward_name = key[len("Episode/rew_") :]
            out.setdefault(f"Episode_Reward/{reward_name}", value)

        if key == "Perf/collection_time":
            out.setdefault("Perf/collection time", value)
        elif key == "Perf/collection time":
            out.setdefault("Perf/collection_time", value)

        if key == "Train/num_samples":
            out.setdefault("Train/total_timesteps", value)
        elif key == "Train/total_timesteps":
            out.setdefault("Train/num_samples", value)

    return out


def append_metrics_jsonl(
    log_dir: str | Path,
    step: int,
    scalars: dict[str, float],
    *,
    filename: str = "metrics.jsonl",
    algo: str | None = None,
) -> None:
    """Append one JSON line of scalars to ``{log_dir}/{filename}``."""
    path = Path(log_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    expanded = expand_ppo_compatible_keys(scalars)
    algo_label = algo or os.environ.get("MJLAB_TRAIN_ALGO_LABEL")
    record: dict[str, Any] = {
        "step": step,
        "wall_time_s": elapsed_wall_time_s(),
        "scalars": expanded,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    append_reward_wall_clock(log_dir, step, expanded, algo=algo_label)
    append_training_metric(log_dir, step, expanded, algo=algo_label)
