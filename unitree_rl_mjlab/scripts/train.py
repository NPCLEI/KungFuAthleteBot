"""Script to train RL agent with PPO (rsl_rl) or FastSAC (holosoma)."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import mjlab
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from src.rl.config import HolosomaFastSACRunnerCfg
from src.rl.logging.run_meta import write_run_meta
from src.rl.logging.training_metrics import finalize_training_metrics
from src.rl.logging.wall_clock_reward import (
  install_ppo_wall_clock_metrics,
  mark_training_wall_clock_start,
)
from src.rl.runners import HolosomaFastSACRunner
from src.rl.utils.distributed import (
  local_cuda_device,
  mujoco_egl_device_id,
)
from src.rl.utils.warp_compat import ensure_warp_context
from src.tasks.registry_ext import (
  assert_fastsac_task,
  load_fastsac_rl_cfg,
  load_runner_cls_for_algo,
)


def _setup_train_log_file(log_dir: Path, rank: int) -> None:
  if rank != 0:
    return
  log_dir.mkdir(parents=True, exist_ok=True)
  train_log = log_dir / "train.log"
  root = logging.getLogger()
  has_file = any(
    isinstance(h, logging.FileHandler)
    and getattr(h, "baseFilename", None) == str(train_log.resolve())
    for h in root.handlers
  )
  if has_file:
    return
  handler = logging.FileHandler(train_log, encoding="utf-8")
  handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  )
  root.addHandler(handler)
  if root.level == logging.NOTSET:
    root.setLevel(logging.INFO)


def _init_fastsac_wandb(
  cfg: TrainConfig,
  log_dir: Path,
  env_cfg_dict: dict,
  agent_cfg_dict: dict,
  rank: int,
) -> None:
  assert cfg.fast_sac_agent is not None
  if cfg.fast_sac_agent.logger != "wandb" or rank != 0:
    return
  try:
    import wandb
  except ImportError:
    print("[WARN] wandb not installed; skipping wandb init.")
    return
  wandb.init(
    project=getattr(cfg.fast_sac_agent, "wandb_project", "mjlab"),
    name=cfg.fast_sac_agent.run_name or log_dir.name,
    dir=str(log_dir),
    config={"env": env_cfg_dict, "agent": agent_cfg_dict, "algo": "fast_sac"},
  )


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  algo: Literal["ppo", "fast_sac"] = "ppo"
  agent: RslRlBaseRunnerCfg | None = None
  fast_sac_agent: HolosomaFastSACRunnerCfg | None = None
  max_iterations: int | None = None
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(
    task_id: str, algo: Literal["ppo", "fast_sac"] = "ppo"
  ) -> TrainConfig:
    env_cfg = load_env_cfg(task_id)
    ppo_cfg = load_rl_cfg(task_id)
    fast_sac_cfg: HolosomaFastSACRunnerCfg | None = None
    try:
      fast_sac_cfg = cast(
        HolosomaFastSACRunnerCfg, load_fastsac_rl_cfg(task_id, algo="fast_sac")
      )
    except ValueError:
      if algo == "fast_sac":
        raise
    return TrainConfig(
      env=env_cfg,
      algo=algo,
      agent=ppo_cfg,
      fast_sac_agent=fast_sac_cfg,
    )


def _apply_tracking_motion(cfg: TrainConfig, rank: int) -> str | None:
  if "motion" not in cfg.env.commands:
    return None
  motion_cmd = cfg.env.commands["motion"]
  if not hasattr(motion_cmd, "motion_file"):
    return None
  if not cfg.motion_file:
    raise ValueError("For tracking tasks, --motion-file must be set.")
  motion_path = Path(cfg.motion_file).expanduser().resolve()
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  motion_cmd.motion_file = str(motion_path)
  if rank == 0:
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")
  return motion_cmd.motion_file


def _experiment_name(cfg: TrainConfig) -> str:
  if cfg.algo == "fast_sac":
    assert cfg.fast_sac_agent is not None
    return cfg.fast_sac_agent.experiment_name
  assert cfg.agent is not None
  return cfg.agent.experiment_name


def _run_name(cfg: TrainConfig) -> str:
  if cfg.algo == "fast_sac":
    assert cfg.fast_sac_agent is not None
    return cfg.fast_sac_agent.run_name
  assert cfg.agent is not None
  return cfg.agent.run_name


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    base_seed = (
      cfg.fast_sac_agent.seed
      if cfg.algo == "fast_sac" and cfg.fast_sac_agent
      else cfg.agent.seed  # type: ignore[union-attr]
    )
    seed = base_seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(mujoco_egl_device_id(local_rank))
    device = local_cuda_device(local_rank)
    base_seed = (
      cfg.fast_sac_agent.seed
      if cfg.algo == "fast_sac" and cfg.fast_sac_agent
      else cfg.agent.seed  # type: ignore[union-attr]
    )
    seed = base_seed + local_rank

  if rank == 0:
    mark_training_wall_clock_start()
    os.environ["MJLAB_TRAIN_ALGO_LABEL"] = cfg.algo

  ensure_warp_context()
  configure_torch_backends()

  cfg.env = replace(cfg.env, seed=seed)
  if cfg.algo == "fast_sac" and cfg.fast_sac_agent is not None:
    cfg.fast_sac_agent = replace(cfg.fast_sac_agent, seed=seed)
  elif cfg.agent is not None:
    cfg.agent = replace(cfg.agent, seed=seed)

  print(
    f"[INFO] Training with: algo={cfg.algo}, device={device}, seed={seed}, rank={rank}"
  )

  if cfg.max_iterations is not None:
    if cfg.algo == "fast_sac" and cfg.fast_sac_agent is not None:
      cfg.fast_sac_agent = replace(
        cfg.fast_sac_agent, max_iterations=cfg.max_iterations
      )
    elif cfg.agent is not None:
      cfg.agent = replace(cfg.agent, max_iterations=cfg.max_iterations)

  motion_file = _apply_tracking_motion(cfg, rank)

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    if rank == 0:
      print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    _setup_train_log_file(log_dir, rank)
    steps_per_iter = (
      cfg.agent.num_steps_per_env if cfg.algo == "ppo" and cfg.agent else None
    )
    write_run_meta(
      log_dir,
      algo=cfg.algo,
      task_id=task_id,
      motion_file=motion_file,
      steps_per_iter=steps_per_iter,
    )

  env_cfg = cfg.env

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent
  resume_path: Path | None = None
  if cfg.algo == "fast_sac":
    sac_cfg = cfg.fast_sac_agent
    assert sac_cfg is not None
    if sac_cfg.resume:
      resume_path = get_checkpoint_path(
        log_root_path, sac_cfg.load_run, sac_cfg.load_checkpoint
      )
  else:
    ppo_cfg = cfg.agent
    assert ppo_cfg is not None
    if ppo_cfg.resume:
      resume_path = get_checkpoint_path(
        log_root_path, ppo_cfg.load_run, ppo_cfg.load_checkpoint
      )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env_cfg_dict = asdict(env_cfg)

  if cfg.algo == "fast_sac":
    assert cfg.fast_sac_agent is not None
    runner_cls = load_runner_cls_for_algo(task_id, "fast_sac") or HolosomaFastSACRunner
    runner = runner_cls(env, cfg.fast_sac_agent, str(log_dir), device)
    agent_cfg_dict = asdict(cfg.fast_sac_agent)
    runner.add_git_repo_to_log(__file__)
    _init_fastsac_wandb(cfg, log_dir, env_cfg_dict, agent_cfg_dict, rank)
    if resume_path is not None:
      print(f"[INFO]: Loading FastSAC checkpoint from: {resume_path}")
      runner.load(str(resume_path))
    if rank == 0:
      dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)
    if hasattr(runner, "store_git_state"):
      runner.store_git_state()
    runner.learn(num_learning_iterations=cfg.fast_sac_agent.max_iterations)
    if rank == 0:
      metrics_path = finalize_training_metrics(log_dir, cfg.algo)
      if metrics_path is not None:
        print(f"[INFO] Exported training metrics to: {metrics_path}")
  else:
    assert cfg.agent is not None
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    agent_cfg_dict = asdict(cfg.agent)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, agent_cfg_dict, str(log_dir), device)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
      print(f"[INFO]: Loading model checkpoint from: {resume_path}")
      runner.load(str(resume_path))
    if rank == 0:
      dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)
      install_ppo_wall_clock_metrics(runner, algo="ppo")
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations,
      init_at_random_ep_len=True,
    )
    if rank == 0:
      metrics_path = finalize_training_metrics(log_dir, cfg.algo)
      if metrics_path is not None:
        print(f"[INFO] Exported training metrics to: {metrics_path}")

  env.close()


def _resolve_log_dir(args: TrainConfig) -> Path:
  log_subdir = "fastsac" if args.algo == "fast_sac" else "rsl_rl"
  log_root_path = Path("logs") / log_subdir / _experiment_name(args)
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_name = _run_name(args)
  if run_name:
    log_dir_name += f"_{run_name}"
  return log_root_path / log_dir_name


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)
  if args.algo == "fast_sac":
    assert_fastsac_task(task_id)
    if args.fast_sac_agent is None:
      args = replace(
        args,
        fast_sac_agent=cast(
          HolosomaFastSACRunnerCfg, load_fastsac_rl_cfg(task_id)
        ),
      )
  log_dir = _resolve_log_dir(args)

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  if args.algo == "fast_sac" and num_gpus > 1:
    print(
      "[WARN] FastSAC supports single-GPU training only; using the first GPU.",
      flush=True,
    )
    if selected_gpus is not None:
      selected_gpus = [selected_gpus[0]]
    num_gpus = 1

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if args.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
    else:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

  print(f"[INFO] Launching PPO with {num_gpus} GPUs (torchrunx)", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_train, task_id, args, log_dir)


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
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
