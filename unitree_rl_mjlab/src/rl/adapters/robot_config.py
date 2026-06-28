"""Lightweight robot_config for holosoma FastSAC action boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv


@dataclass
class _InitState:
    default_joint_angles: dict[str, float] = field(default_factory=dict)


@dataclass
class _Control:
    action_scale: float


@dataclass
class MjlabRobotConfig:
    """Subset of holosoma robot config used by FastSACEnv."""

    dof_names: list[str]
    dof_pos_lower_limit_list: list[float]
    dof_pos_upper_limit_list: list[float]
    init_state: _InitState
    control: _Control
    actions_dim: int


def build_robot_config_from_mjlab_env(env: ManagerBasedRlEnv) -> MjlabRobotConfig:
    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    default_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()
    soft_limits = robot.data.soft_joint_pos_limits[0].detach().cpu()
    lower = soft_limits[:, 0].tolist()
    upper = soft_limits[:, 1].tolist()

    default_angles = {
        name: float(default_pos[i]) for i, name in enumerate(joint_names)
    }

    action_term = env.action_manager.get_term("joint_pos")
    scale_val = 0.25
    if hasattr(action_term, "scale"):
        sc = action_term.scale
        if isinstance(sc, dict):
            scale_val = float(next(iter(sc.values())))
        elif isinstance(sc, (int, float)):
            scale_val = float(sc)
        elif isinstance(sc, torch.Tensor):
            scale_val = float(sc.flatten()[0].item())

    return MjlabRobotConfig(
        dof_names=joint_names,
        dof_pos_lower_limit_list=lower,
        dof_pos_upper_limit_list=upper,
        init_state=_InitState(default_joint_angles=default_angles),
        control=_Control(action_scale=scale_val),
        actions_dim=env.action_manager.total_action_dim,
    )
