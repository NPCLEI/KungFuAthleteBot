"""Observation manager for modular observation computation.

This module provides a manager-based system for computing observations in a modular,
configurable way while maintaining exact equivalence with the direct observation system.
"""

# Import from config_types for consistency with tyro migration
from holosoma_min.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

from .base import ObservationTermBase
from .manager import ObservationManager

__all__ = [
    "ObsGroupCfg",
    "ObsTermCfg",
    "ObservationManager",
    "ObservationManagerCfg",
    "ObservationTermBase",
]
