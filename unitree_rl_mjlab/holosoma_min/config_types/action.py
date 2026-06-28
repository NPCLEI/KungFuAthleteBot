"""Configuration types for action manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ActionTermCfg:
    """Configuration for a single action-processing term."""

    func: str
    """Import path to the action term class (e.g. ``holosoma_min.managers.action.terms:JointPositionActionTerm``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to initialize the action term."""

    scale: float | tuple[float, ...] = 1.0
    """Scaling factor(s) applied to the raw action values before processing."""

    clip: tuple[float, float] | None = None
    """Optional min/max clamp applied to the raw action values."""


@dataclass(frozen=True)
class ActionManagerCfg:
    """Configuration for the action manager."""

    terms: dict[str, ActionTermCfg] = field(default_factory=dict)
    """Mapping of action term name to configuration."""
