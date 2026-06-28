"""Collect PPO/FastSAC training metrics and export structured JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MEAN_REWARD_KEY = "Train/mean_reward"
MEAN_EPISODE_LENGTH_KEY = "Train/mean_episode_length"
BUFFER_FILENAME = "training_metrics_buffer.jsonl"

ALGO_JSON_NAMES: dict[str, str] = {
    "ppo": "ppo_metrics.json",
    "fast_sac": "fastsac_metrics.json",
}


def _extract_metric(scalars: dict[str, float], key: str) -> float | None:
    if key not in scalars:
        return None
    return float(scalars[key])


def append_training_metric(
    log_dir: str | Path,
    step: int,
    scalars: dict[str, float],
    *,
    algo: str | None = None,
) -> None:
    """Append one wall-clock sample when episode metrics are available."""
    reward = _extract_metric(scalars, MEAN_REWARD_KEY)
    episode_length = _extract_metric(scalars, MEAN_EPISODE_LENGTH_KEY)
    if reward is None and episode_length is None:
        return

    from src.rl.logging.wall_clock_reward import elapsed_wall_time_s

    record: dict[str, Any] = {
        "step": int(step),
        "wall_time_h": elapsed_wall_time_s() / 3600.0,
    }
    if reward is not None:
        record["mean_reward"] = reward
    if episode_length is not None:
        record["episode_length"] = episode_length
    if algo is not None:
        record["algo"] = algo

    path = Path(log_dir) / BUFFER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def finalize_training_metrics(log_dir: str | Path, algo: str) -> Path | None:
    """Write ``{ppo,fastsac}_metrics.json`` from the training buffer."""
    log_path = Path(log_dir)
    buffer_path = log_path / BUFFER_FILENAME
    if not buffer_path.is_file():
        return None

    records: list[dict[str, Any]] = []
    with buffer_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        buffer_path.unlink(missing_ok=True)
        return None

    out_name = ALGO_JSON_NAMES.get(algo, f"{algo}_metrics.json")
    out_path = log_path / out_name
    payload = {"algo": algo, "records": records}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    buffer_path.unlink(missing_ok=True)
    return out_path


def load_training_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a finalized metrics JSON file."""
    metrics_path = Path(path)
    with metrics_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return list(data.get("records", []))
