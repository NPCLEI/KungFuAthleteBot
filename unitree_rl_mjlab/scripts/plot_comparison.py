"""Plot PPO vs FastSAC training metrics against wall-clock time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_SPECS = {
    "reward": ("mean_reward", "Mean Reward"),
    "episode": ("episode_length", "Episode Length"),
}

PLOT_STYLE = {
    "PPO": {"color": "#2563eb", "alpha_fill": 0.18},
    "FastSAC": {"color": "#111827", "alpha_fill": 0.12},
}


def load_training_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return list(data.get("records", []))


def _apply_plot_style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#d9d9d9", linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11, colors="black")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)


def _extract_series(records: list[dict], field: str) -> tuple[list[float], list[float]]:
    wall_times: list[float] = []
    values: list[float] = []
    for record in records:
        if field not in record:
            continue
        wall_times.append(float(record["wall_time_h"]))
        values.append(float(record[field]))
    return wall_times, values


def _filter_by_wall_time(
    records: list[dict], max_wall_h: float | None
) -> list[dict]:
    if max_wall_h is None:
        return records
    return [r for r in records if float(r["wall_time_h"]) <= max_wall_h]


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def first_reach(records: list[dict], threshold: float) -> dict | None:
    for record in records:
        if float(record.get("mean_reward", float("-inf"))) >= threshold:
            return record
    return None


def print_milestone_table(
    ppo_records: list[dict], sac_records: list[dict], thresholds: list[float]
) -> None:
    print("| Reward ≥ | PPO wall (h) | PPO step | FastSAC wall (h) | FastSAC step | Speedup |")
    print("|----------|--------------|----------|------------------|--------------|---------|")
    for threshold in thresholds:
        ppo_hit = first_reach(ppo_records, threshold)
        sac_hit = first_reach(sac_records, threshold)
        if ppo_hit is None or sac_hit is None:
            continue
        ppo_h = float(ppo_hit["wall_time_h"])
        sac_h = float(sac_hit["wall_time_h"])
        speedup = ppo_h / sac_h if sac_h > 0 else float("inf")
        print(
            f"| {threshold:.0f} | {ppo_h:.2f} | {int(ppo_hit['step'])} | "
            f"{sac_h:.2f} | {int(sac_hit['step'])} | {speedup:.2f}x |"
        )


def plot_comparison(
    *,
    ppo_json: Path,
    sac_json: Path,
    metric: str,
    task_name: str,
    out: Path | None = None,
    max_wall_h: float | None = None,
    smooth_window: int = 0,
    milestone_rewards: list[float] | None = None,
) -> Path:
    field, ylabel = METRIC_SPECS[metric]
    ppo_records = _filter_by_wall_time(load_training_metrics(ppo_json), max_wall_h)
    sac_records = _filter_by_wall_time(load_training_metrics(sac_json), max_wall_h)
    series = [
        ("PPO", ppo_records),
        ("FastSAC", sac_records),
    ]

    if milestone_rewards and metric == "reward":
        print_milestone_table(ppo_records, sac_records, milestone_rewards)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plot_comparison.py") from exc

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

    for label, records in series:
        wall_times, values = _extract_series(records, field)
        if not wall_times:
            continue
        if smooth_window > 1:
            values = _smooth(values, smooth_window)
        style = PLOT_STYLE[label]
        ax.plot(
            wall_times,
            values,
            label=label,
            color=style["color"],
            linewidth=2.8,
            solid_capstyle="round",
        )

    if milestone_rewards and metric == "reward":
        for threshold in milestone_rewards:
            ax.axhline(
                threshold,
                color="#9ca3af",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
            ax.text(
                0.01,
                threshold,
                f"  {threshold:.0f}",
                transform=ax.get_yaxis_transform(),
                fontsize=9,
                color="#6b7280",
                va="bottom",
            )

    title_suffix = f" (≤{max_wall_h:.1f}h)" if max_wall_h is not None else ""
    ax.set_title(f"G1 {task_name}{title_suffix}")
    ax.set_xlabel("Wall Time (h)")
    ax.set_ylabel(ylabel)
    if max_wall_h is not None:
        ax.set_xlim(0.0, max_wall_h)
    _apply_plot_style(ax)
    ax.legend(frameon=True, fontsize=11, loc="best")
    fig.tight_layout()

    if out is None:
        suffix = f"_le{max_wall_h:.0f}h" if max_wall_h is not None else ""
        out = Path(f"g1_{task_name.replace(' ', '_')}_{metric}{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PPO and FastSAC training curves by wall-clock time."
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SPECS),
        required=True,
        help="Metric to plot: episode length or mean reward.",
    )
    parser.add_argument("--ppo_json", type=Path, required=True, help="Path to ppo_metrics.json.")
    parser.add_argument("--sac_json", type=Path, required=True, help="Path to fastsac_metrics.json.")
    parser.add_argument("--task_name", type=str, required=True, help="Task label used in the plot title.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output PNG path.")
    parser.add_argument(
        "--max_wall_h",
        type=float,
        default=None,
        help="Optional x-axis cap (hours). Useful to compare within PPO total training time.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=0,
        help="Rolling mean window over logged points (0 = raw).",
    )
    parser.add_argument(
        "--milestone_rewards",
        type=float,
        nargs="*",
        default=None,
        help="Reward thresholds for horizontal guides and markdown table output.",
    )
    args = parser.parse_args()

    out_path = plot_comparison(
        ppo_json=args.ppo_json,
        sac_json=args.sac_json,
        metric=args.metric,
        task_name=args.task_name,
        out=args.out,
        max_wall_h=args.max_wall_h,
        smooth_window=args.smooth_window,
        milestone_rewards=args.milestone_rewards,
    )
    print(f"[INFO] Saved comparison plot to: {out_path}")


if __name__ == "__main__":
    main()
