"""Extended task registry with PPO + FastSAC configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks import registry as mjlab_registry

from src.rl.config import HolosomaFastSACRunnerCfg

_REGISTRY_EXT: dict[str, "_TrackingTaskCfg"] = {}


@dataclass
class _TrackingTaskCfg:
    env_cfg: ManagerBasedRlEnvCfg
    play_env_cfg: ManagerBasedRlEnvCfg
    rl_cfg_ppo: RslRlOnPolicyRunnerCfg
    rl_cfg_fastsac: HolosomaFastSACRunnerCfg
    runner_cls_ppo: type | None
    runner_cls_fastsac: type | None


def register_tracking_task(
    task_id: str,
    env_cfg: ManagerBasedRlEnvCfg,
    play_env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg_ppo: RslRlOnPolicyRunnerCfg,
    rl_cfg_fastsac: HolosomaFastSACRunnerCfg,
    runner_cls_ppo: type | None = None,
    runner_cls_fastsac: type | None = None,
) -> None:
    """Register tracking task with dual algorithm configs."""
    mjlab_registry.register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg_ppo,
        runner_cls=runner_cls_ppo,
    )
    if task_id in _REGISTRY_EXT:
        raise ValueError(f"Task '{task_id}' already registered in extended registry")
    _REGISTRY_EXT[task_id] = _TrackingTaskCfg(
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg_ppo=rl_cfg_ppo,
        rl_cfg_fastsac=rl_cfg_fastsac,
        runner_cls_ppo=runner_cls_ppo,
        runner_cls_fastsac=runner_cls_fastsac,
    )


def fastsac_task_ids() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY_EXT))


def assert_fastsac_task(task_id: str) -> None:
    if task_id in _REGISTRY_EXT:
        return
    allowed = ", ".join(fastsac_task_ids()) or "(none registered)"
    raise ValueError(
        f"FastSAC is only supported for: {allowed}. Got task '{task_id}'."
    )


def load_fastsac_rl_cfg(
    task_name: str, algo: Literal["ppo", "fast_sac"] = "fast_sac"
) -> HolosomaFastSACRunnerCfg:
    del algo
    assert_fastsac_task(task_name)
    return deepcopy(_REGISTRY_EXT[task_name].rl_cfg_fastsac)


def load_runner_cls_for_algo(
    task_name: str, algo: Literal["ppo", "fast_sac"] = "ppo"
) -> type | None:
    if algo == "fast_sac":
        assert_fastsac_task(task_name)
        return _REGISTRY_EXT[task_name].runner_cls_fastsac
    if task_name not in _REGISTRY_EXT:
        return mjlab_registry.load_runner_cls(task_name)
    return _REGISTRY_EXT[task_name].runner_cls_ppo
