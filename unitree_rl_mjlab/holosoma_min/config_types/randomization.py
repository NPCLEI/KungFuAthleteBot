"""Configuration types for randomization manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RandomizationTermCfg:
    """Configuration for a single randomization hook."""

    func: str
    """Import path of the randomization hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class RandomizationManagerCfg:
    """Configuration for the randomization manager."""

    setup_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, RandomizationTermCfg] = field(default_factory=dict)
    """Hooks invoked every simulation step."""

    ignore_unsupported: bool = False
    """Flag to ignore errors when randomizers are not implemented by a simulator."""
