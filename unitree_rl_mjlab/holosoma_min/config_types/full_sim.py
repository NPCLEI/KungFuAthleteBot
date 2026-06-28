from __future__ import annotations

from pydantic.dataclasses import dataclass

from holosoma_min.config_types.experiment import TrainingConfig
from holosoma_min.config_types.logger import LoggerConfig
from holosoma_min.config_types.robot import RobotConfig
from holosoma_min.config_types.simulator import SimulatorInitConfig


@dataclass(frozen=True)
class FullSimConfig:
    """Collection of configs needed for constructing simulator classes."""

    simulator: SimulatorInitConfig
    robot: RobotConfig
    training: TrainingConfig
    logger: LoggerConfig
    """Logger configuration for video recording and output directories."""

    experiment_dir: str | None = None
    """Experiment directory path (computed from logger config in base_task)."""
