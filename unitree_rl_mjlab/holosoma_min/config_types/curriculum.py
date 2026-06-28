"""Configuration types for the curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumTermCfg:
    """Configuration for a single curriculum hook."""

    func: str
    """Import path for the curriculum hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CurriculumManagerCfg:
    """Configuration for the curriculum manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across curriculum hooks."""

    setup_terms: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    """Hooks invoked every simulation step."""
