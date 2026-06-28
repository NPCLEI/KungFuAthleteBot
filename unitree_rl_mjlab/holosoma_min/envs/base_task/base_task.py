"""Minimal BaseTask stub for FastSAC type hints (mjlab adapter implements contract)."""
from __future__ import annotations

from typing import Any


class BaseTask:
    """Stub; real env is MjlabHolosomaEnvAdapter."""

    num_envs: int
    device: str
    robot_config: Any
    observation_manager: Any

    def reset_all(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, actor_state: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
        raise NotImplementedError
