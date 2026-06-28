from mjlab.tasks.registry import register_mjlab_task
from src.tasks.registry_ext import register_tracking_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from src.rl.runners import MotionTrackingFastSACRunner

from .env_cfgs import *
from .rl_cfg import (
  unitree_g1_tracking_fastsac_runner_cfg,
  unitree_g1_tracking_ppo_runner_cfg,
)

_FASTSAC = unitree_g1_tracking_fastsac_runner_cfg()
_PPO = unitree_g1_tracking_ppo_runner_cfg()


def _register_stage_task(task_id: str, env_cfg, play_env_cfg) -> None:
  register_tracking_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_env_cfg,
    rl_cfg_ppo=_PPO,
    rl_cfg_fastsac=_FASTSAC,
    runner_cls_ppo=MotionTrackingOnPolicyRunner,
    runner_cls_fastsac=MotionTrackingFastSACRunner,
  )


register_mjlab_task(
  task_id="Unitree-G1-Tracking",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Tracking-Standing",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

_register_stage_task(
  task_id="Unitree-G1-1307-Stage-I",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(play=True),
)

_register_stage_task(
  task_id="Unitree-G1-1307-Stage-II",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_II(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_II(play=True),
)

_register_stage_task(
  task_id="Unitree-G1-1307-Stage-III",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_III(),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_III(play=True),
)

register_mjlab_task(
  task_id="Unitree-G1-1307-Checkpoint",
  env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_standing_env_cfg_1307_stage_I(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
