"""Record reward vs wall-clock time and plot after training."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

_TRAINING_START: float | None = None
REWARD_WALL_CLOCK_FILE = "reward_wall_clock.jsonl"
PLOT_FILENAME = "reward_wall_clock.png"

from src.rl.logging.training_metrics import (
    MEAN_EPISODE_LENGTH_KEY,
    MEAN_REWARD_KEY,
)
COMPARE_PLOT_FILENAME = "reward_wall_clock_compare.png"
_COMPARE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def mark_training_wall_clock_start() -> None:
    """Reset wall-clock timer (call once before runner.learn())."""
    import time

    global _TRAINING_START
    _TRAINING_START = time.perf_counter()


def elapsed_wall_time_s() -> float:
    import time

    if _TRAINING_START is None:
        return 0.0
    return time.perf_counter() - _TRAINING_START


def extract_primary_reward(scalars: dict[str, float]) -> float | None:
    """Episode-mean reward for cross-algo wall-clock plots (PPO and FastSAC)."""
    if MEAN_REWARD_KEY not in scalars:
        return None
    return float(scalars[MEAN_REWARD_KEY])


def append_reward_wall_clock(
    log_dir: str | Path,
    step: int,
    scalars: dict[str, float],
    *,
    algo: str | None = None,
) -> None:
    """Append one reward sample with elapsed wall time (seconds)."""
    reward = extract_primary_reward(scalars)
    if reward is None:
        return
    record: dict[str, Any] = {
        "step": int(step),
        "wall_time_s": elapsed_wall_time_s(),
        "reward": reward,
    }
    if algo is not None:
        record["algo"] = algo
    path = Path(log_dir) / REWARD_WALL_CLOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_reward_wall_clock(log_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(log_dir) / REWARD_WALL_CLOCK_FILE
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_run_dir(path: str | Path) -> Path:
    """Accept a run directory or a ``metrics.jsonl`` path inside it."""
    run_path = Path(path)
    if run_path.is_file() and run_path.name == "metrics.jsonl":
        return run_path.parent
    return run_path


def infer_algo_label(log_dir: str | Path) -> str | None:
    name = Path(log_dir).name
    for prefix in ("PPO", "FastSAC"):
        if name.startswith(f"{prefix}_"):
            return prefix
    return None


def trim_rows_to_wall_time(
    rows: list[dict[str, Any]], max_wall_time_s: float
) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["wall_time_s"]) <= max_wall_time_s]


def load_mean_reward_wall_clock(log_dir: str | Path) -> list[dict[str, Any]]:
    """Load episode-mean reward vs wall time; skip steps without completed episodes."""
    log_path = Path(log_dir)
    metrics_path = log_path / "metrics.jsonl"
    if metrics_path.is_file():
        rows: list[dict[str, Any]] = []
        with metrics_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                reward = extract_primary_reward(record.get("scalars", {}))
                if reward is None:
                    continue
                row: dict[str, Any] = {
                    "step": int(record["step"]),
                    "wall_time_s": float(record["wall_time_s"]),
                    "reward": reward,
                }
                algo = record.get("algo")
                if algo is not None:
                    row["algo"] = algo
                rows.append(row)
        if rows:
            return rows

    return load_reward_wall_clock(log_dir)


def _plot_mean_reward_axes(
    ax: Any,
    *,
    wall_times: list[float],
    rewards: list[float],
    label: str,
    color: str,
) -> None:
    ax.plot(
        wall_times,
        rewards,
        label=f"{label} Train/mean_reward",
        color=color,
        alpha=0.9,
    )


def plot_compare_reward_wall_clock(
    runs: list[tuple[str | Path, str | None]],
    *,
    out: str | Path | None = None,
    out_name: str = COMPARE_PLOT_FILENAME,
) -> Path | None:
    """Plot PPO vs FastSAC (or more runs) on one wall-clock axis.

    Truncates every series to the shortest run's max wall time.
    """
    if len(runs) < 2:
        raise ValueError("plot_compare_reward_wall_clock requires at least two runs")

    series: list[tuple[Path, str, list[dict[str, Any]]]] = []
    for run_path, algo in runs:
        log_dir = resolve_run_dir(run_path)
        rows = load_mean_reward_wall_clock(log_dir)
        if not rows:
            print(f"[WARN] No Train/mean_reward data in {log_dir}")
            return None
        label = algo or infer_algo_label(log_dir) or log_dir.name
        series.append((log_dir, label, rows))

    wall_caps = [max(float(r["wall_time_s"]) for r in rows) for _, _, rows in series]
    wall_limit = min(wall_caps)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip reward wall-clock plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (_, label, rows) in enumerate(series):
        trimmed = trim_rows_to_wall_time(rows, wall_limit)
        color = _COMPARE_COLORS[idx % len(_COMPARE_COLORS)]
        _plot_mean_reward_axes(
            ax,
            wall_times=[float(r["wall_time_s"]) for r in trimmed],
            rewards=[float(r["reward"]) for r in trimmed],
            label=label,
            color=color,
        )

    labels = " vs ".join(label for _, label, _ in series)
    ax.set_xlim(0.0, wall_limit)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Train/mean_reward (episode mean)")
    ax.set_title(f"Episode mean reward vs wall-clock time ({labels})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out is not None:
        out_path = Path(out)
    else:
        parent = series[0][0].parent
        out_path = parent / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_reward_wall_clock(
    log_dir: str | Path,
    *,
    algo: str | None = None,
    out: str | Path | None = None,
    out_name: str = PLOT_FILENAME,
) -> Path | None:
    """Plot Train/mean_reward vs wall-clock time; returns output path or None if no data."""
    rows = load_mean_reward_wall_clock(log_dir)
    if not rows:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip reward wall-clock plot.")
        return None

    log_path = Path(log_dir)
    wall_times = [float(r["wall_time_s"]) for r in rows]
    rewards = [float(r["reward"]) for r in rows]
    label = algo or rows[-1].get("algo", "training")

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_mean_reward_axes(
        ax,
        wall_times=wall_times,
        rewards=rewards,
        label=label,
        color=_COMPARE_COLORS[0],
    )
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Train/mean_reward (episode mean)")
    ax.set_title(f"Episode mean reward vs wall-clock time ({label})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out) if out is not None else log_path / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def install_ppo_wall_clock_metrics(runner: Any, algo: str = "ppo") -> None:
    """Bridge PPO ``Logger.log()`` to metrics export hooks."""
    logger = getattr(runner, "logger", None)
    if logger is None or logger.log_dir is None or getattr(logger, "disable_logs", False):
        return

    from src.rl.logging.export_scalars import append_metrics_jsonl

    original_log = logger.log

    def wrapped_log(*args: Any, **kwargs: Any) -> None:
        original_log(*args, **kwargs)
        if logger.writer is None:
            return
        if getattr(runner, "is_distributed", False) and getattr(runner, "gpu_global_rank", 0) != 0:
            return

        it = kwargs.get("it")
        if it is None and args:
            it = args[0]
        if it is None:
            return

        scalars: dict[str, float] = {}
        rewbuffer = getattr(logger, "rewbuffer", None) or []
        lenbuffer = getattr(logger, "lenbuffer", None) or []
        if len(rewbuffer) > 0:
            scalars[MEAN_REWARD_KEY] = float(statistics.mean(rewbuffer))
        if len(lenbuffer) > 0:
            scalars[MEAN_EPISODE_LENGTH_KEY] = float(statistics.mean(lenbuffer))
        if not scalars:
            return
        append_metrics_jsonl(logger.log_dir, int(it), scalars, algo=algo)

    logger.log = wrapped_log
