"""Script to play RL agent with PPO or FastSAC."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.rl.config import HolosomaFastSACRunnerCfg
from src.rl.runners import HolosomaFastSACRunner
from src.tasks.registry_ext import load_fastsac_rl_cfg
from src.rl.utils.warp_compat import ensure_warp_context


@dataclass(frozen=True)
class PlayConfig:
  algo: Literal["ppo", "fast_sac"] = "ppo"
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  from_fall: bool = False
  wandb_run_path: str | None = None

  _demo_mode: tyro.conf.Suppress[bool] = False


def _build_fastsac_policy(
  env: ManagerBasedRlEnv, checkpoint: Path, device: str, task_id: str
):
  sac_cfg = cast(HolosomaFastSACRunnerCfg, load_fastsac_rl_cfg(task_id))
  runner = HolosomaFastSACRunner(env, sac_cfg, str(checkpoint.parent), device)
  runner.load(str(checkpoint))

  def policy(obs) -> torch.Tensor:
    actor_obs = obs["actor"]
    with torch.no_grad():
      if runner.agent.obs_normalization:
        actor_obs = runner.agent.obs_normalizer(actor_obs, update=False)
      return runner.agent.actor(actor_obs)[0]

  return policy


def run_play(task_id: str, cfg: PlayConfig):
  ensure_warp_context()
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  motion_cmd = env_cfg.commands.get("motion")
  is_tracking_task = motion_cmd is not None and hasattr(motion_cmd, "motion_file")

  if is_tracking_task and cfg._demo_mode:
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      raise ValueError("Tracking tasks require --motion-file for dummy agents.")

  if cfg.from_fall and is_tracking_task and hasattr(motion_cmd, "tracking_standing_weight"):
    motion_cmd.tracking_standing_weight = (0.0, 1.0)
    print("[INFO]: Recovery play — initializing from fallen states (tracking_standing_weight=(0, 1))")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    if cfg.algo == "fast_sac":
      exp_name = load_fastsac_rl_cfg(task_id).experiment_name
      log_root = (Path("logs") / "fastsac" / exp_name).resolve()
    else:
      agent_cfg = load_rl_cfg(task_id)
      log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()

    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` or `checkpoint-file` is required for trained play."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root, Path(cfg.wandb_run_path)
      )
      run_id = resume_path.parent.name
      print(
        f"[INFO]: Loading checkpoint: {resume_path.name} "
        f"(run: {run_id}, {'cached' if was_cached else 'downloaded'})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print("[WARN] Video recording with dummy agents is disabled.")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  if DUMMY_MODE:
    env_wrap = RslRlVecEnvWrapper(env, clip_actions=None)
    action_shape: tuple[int, ...] = env_wrap.unwrapped.action_space.shape

    class PolicyZero:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(action_shape, device=env_wrap.unwrapped.device)

    class PolicyRandom:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=env_wrap.unwrapped.device) - 1

    policy = PolicyZero() if cfg.agent == "zero" else PolicyRandom()
    play_env = env_wrap
  elif cfg.algo == "fast_sac":
    assert resume_path is not None
    policy = _build_fastsac_policy(env, resume_path, device, task_id)
    play_env = RslRlVecEnvWrapper(env, clip_actions=None)
  else:
    agent_cfg = load_rl_cfg(task_id)
    play_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(play_env, asdict(agent_cfg), device=device)
    assert resume_path is not None
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  num_steps = cfg.video_length if (TRAINED_MODE and cfg.video) else None
  if resolved_viewer == "native":
    NativeMujocoViewer(play_env, policy).run(num_steps=num_steps)
  elif resolved_viewer == "viser":
    ViserPlayViewer(play_env, policy).run(num_steps=num_steps)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  play_env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
