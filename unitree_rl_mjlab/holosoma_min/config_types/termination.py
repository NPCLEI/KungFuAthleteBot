"""Configuration types for termination manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TerminationTermCfg:
    """Configuration for a single termination term."""

    func: str
    """Import path of the termination hook."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""

    is_timeout: bool = False
    """Whether this term should be treated as a timeout condition."""


@dataclass(frozen=True)
class TerminationManagerCfg:
    """Configuration for the termination manager."""

    terms: dict[str, TerminationTermCfg] = field(default_factory=dict)
    """Mapping of termination term name to configuration."""
