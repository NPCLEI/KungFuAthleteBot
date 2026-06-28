"""Scene configuration data classes.

This module provides typed configuration data classes for scene loading,
separating configuration structure from business logic and defining
protocols for scene interfaces across different simulators.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from holosoma_min.utils.safe_torch_import import torch


@runtime_checkable
class SceneInterface(Protocol):
    """Protocol defining the scene interface that all simulators must implement.

    This protocol ensures consistent scene management across different simulator
    implementations by defining required properties and methods.
    """

    @property
    def env_origins(self) -> torch.Tensor:
        """Environment origins for multi-environment setups.

        Returns
        -------
        torch.Tensor
            Shape [num_envs, 3] containing [x, y, z] origins for each environment.
            Must be float32 tensor on the same device as the simulator.
        """
        ...
