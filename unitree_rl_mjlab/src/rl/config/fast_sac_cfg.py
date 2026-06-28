"""FastSAC runner configuration for mjlab G1 tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from holosoma_min.config_types.algo import FastSACConfig


@dataclass
class HolosomaFastSACRunnerCfg:
    """Training runner settings for holosoma FastSAC on mjlab envs."""

    seed: int = 42
    max_iterations: int = 30001
    experiment_name: str = "g1_tracking_fastsac"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    logger: Literal["wandb", "tensorboard"] = "tensorboard"
    save_interval: int = 1000
    logging_interval: int = 100
    console_logging_interval: int = 100
    learning_starts: int = 10

    early_stop_enabled: bool = True
    early_stop_abs_reward: float = 1e4
    early_stop_nan_loss: bool = True
    early_stop_overflow_lines_per_step: int = 50

    reward_anomaly_log_enabled: bool = True
    reward_anomaly_abs_threshold: float = 1e4
    reward_anomaly_max_envs: int = 32
    reward_anomaly_log_non_finite: bool = True

    # FastSAC hyperparameters (aligned with holosoma g1_29dof_wbt_fast_sac, tuned for mjlab smoke)
    gamma: float = 0.99
    tau: float = 0.05
    buffer_size: int = 4096
    batch_size: int = 2048
    num_steps: int = 1
    num_updates: int = 2
    policy_frequency: int = 4
    num_atoms: int = 501
    v_min: float = -20.0
    v_max: float = 20.0
    target_entropy_ratio: float = 0.5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    use_symmetry: bool = False
    use_tanh: bool = False
    obs_normalization: bool = True
    compile: bool = False
    amp: bool = False
    max_grad_norm: float = 1.0
    actor_obs_keys: tuple[str, ...] = ("actor_obs",)
    critic_obs_keys: tuple[str, ...] = ("critic_obs",)


def to_fastsac_config(cfg: HolosomaFastSACRunnerCfg) -> FastSACConfig:
    return FastSACConfig(
        num_learning_iterations=cfg.max_iterations,
        gamma=cfg.gamma,
        tau=cfg.tau,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        num_steps=cfg.num_steps,
        num_updates=cfg.num_updates,
        policy_frequency=cfg.policy_frequency,
        learning_starts=cfg.learning_starts,
        num_atoms=cfg.num_atoms,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
        target_entropy_ratio=cfg.target_entropy_ratio,
        actor_learning_rate=cfg.actor_learning_rate,
        critic_learning_rate=cfg.critic_learning_rate,
        alpha_learning_rate=cfg.alpha_learning_rate,
        actor_hidden_dim=cfg.actor_hidden_dim,
        critic_hidden_dim=cfg.critic_hidden_dim,
        use_symmetry=cfg.use_symmetry,
        use_tanh=cfg.use_tanh,
        obs_normalization=cfg.obs_normalization,
        compile=cfg.compile,
        amp=cfg.amp,
        max_grad_norm=cfg.max_grad_norm,
        save_interval=cfg.save_interval,
        logging_interval=cfg.logging_interval,
        console_logging_interval=cfg.console_logging_interval,
        early_stop_enabled=cfg.early_stop_enabled,
        early_stop_abs_reward=cfg.early_stop_abs_reward,
        early_stop_nan_loss=cfg.early_stop_nan_loss,
        early_stop_overflow_lines_per_step=cfg.early_stop_overflow_lines_per_step,
        actor_obs_keys=list(cfg.actor_obs_keys),
        critic_obs_keys=list(cfg.critic_obs_keys),
    )
