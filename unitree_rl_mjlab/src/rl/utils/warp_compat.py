"""Compatibility shims for mjlab + warp-lang."""

from __future__ import annotations


def ensure_warp_context() -> None:
    """Expose wp.context for mjlab sim when warp-lang omits the alias."""
    import warp as wp

    wp.init()
    if hasattr(wp, "context"):
        return
    import warp._src.context as wpc

    wp.context = wpc  # type: ignore[attr-defined]
