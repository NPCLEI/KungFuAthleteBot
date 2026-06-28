"""MuJoCo stderr overflow tap and FastSAC divergence early-stop helpers."""

from __future__ import annotations

import math
import sys
from typing import Any, TextIO

_OVERFLOW_MARKERS = ("contact match overflow", "nefc overflow")

_tap_installed = False
_overflow_lines_this_step = 0


class _OverflowCountingStream:
    def __init__(self, underlying: TextIO) -> None:
        self._underlying = underlying

    def write(self, data: str) -> int:
        global _overflow_lines_this_step
        if data:
            lower = data.lower()
            if any(marker in lower for marker in _OVERFLOW_MARKERS):
                _overflow_lines_this_step += max(1, data.count("\n"))
        return self._underlying.write(data)

    def flush(self) -> None:
        self._underlying.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


def install_overflow_stderr_tap() -> None:
    """Wrap sys.stderr once to count MuJoCo overflow lines per training step."""
    global _tap_installed
    if _tap_installed:
        return
    sys.stderr = _OverflowCountingStream(sys.stderr)  # type: ignore[assignment]
    _tap_installed = True


def reset_overflow_step_counter() -> None:
    global _overflow_lines_this_step
    _overflow_lines_this_step = 0


def overflow_lines_this_step() -> int:
    return _overflow_lines_this_step


def _scalar_finite(value: Any) -> bool:
    if value is None:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return math.isfinite(v)


def check_divergence_early_stop(
    loss_dict: dict[str, float],
    *,
    enabled: bool,
    abs_reward_threshold: float,
    check_nan_loss: bool,
    overflow_lines_threshold: int,
) -> str | None:
    """Return early-stop reason string, or None if training should continue."""
    if not enabled:
        return None

    overflow_n = overflow_lines_this_step()
    if overflow_lines_threshold > 0 and overflow_n >= overflow_lines_threshold:
        return f"mujoco_overflow_lines={overflow_n}>={overflow_lines_threshold}"

    for key in ("actor_loss", "qf_loss", "alpha_loss"):
        if check_nan_loss and key in loss_dict and not _scalar_finite(loss_dict[key]):
            return f"non_finite_{key}={loss_dict[key]}"

    for key in ("env_rewards", "buffer_rewards"):
        if key in loss_dict:
            val = abs(float(loss_dict[key]))
            if val > abs_reward_threshold:
                return f"|{key}|={val}>threshold={abs_reward_threshold}"

    return None
