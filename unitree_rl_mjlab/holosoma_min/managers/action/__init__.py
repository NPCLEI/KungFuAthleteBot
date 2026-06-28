"""Action manager module for processing and applying actions.

This module provides the infrastructure for a manager-based action system,
allowing modular and configurable action processing.
"""

from .base import ActionTermBase
from .manager import ActionManager

__all__ = ["ActionManager", "ActionTermBase"]
