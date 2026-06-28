"""FastSAC runner with motion-tracking ONNX export."""

from __future__ import annotations

import os
from typing import cast

import torch
import wandb
from torch import nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.tracking.mdp import MotionCommand

from src.rl.runners.holosoma_fastsac_runner import HolosomaFastSACRunner
from src.rl.config import HolosomaFastSACRunnerCfg


class _OnnxMotionActor(nn.Module):
    """ONNX-exportable FastSAC actor with bundled motion reference."""

    def __init__(self, actor_wrapper: nn.Module, motion) -> None:
        super().__init__()
        self.actor_wrapper = actor_wrapper
        self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
        self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
        self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
        self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
        self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
        self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
        self.time_step_total: int = int(self.joint_pos.shape[0])

    def forward(self, obs: torch.Tensor, time_step: torch.Tensor):
        time_step_clamped = torch.clamp(
            time_step.long().squeeze(-1), max=self.time_step_total - 1
        )
        actions = self.actor_wrapper(obs)
        return (
            actions,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )


class MotionTrackingFastSACRunner(HolosomaFastSACRunner):
    """FastSAC runner that exports motion-bundle ONNX on save."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        train_cfg: HolosomaFastSACRunnerCfg,
        log_dir: str,
        device: str,
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self._mj_env = env
        self._patch_agent_save_for_motion_onnx()

    def _patch_agent_save_for_motion_onnx(self) -> None:
        """Route periodic checkpoints from learn() to mjlab motion ONNX export."""
        agent = self.agent
        original_save = agent.save

        def save_with_motion_onnx(path: str) -> None:
            original_save(path)
            if not agent.is_main_process:
                return
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            try:
                self.export_motion_policy_to_onnx(policy_path, filename)
            except Exception as exc:
                print(f"[WARN] Motion ONNX export skipped: {exc}")
                return
            run_name = "local"
            obs_terms = self._mj_env.observation_manager.active_terms
            active_terms_backup = dict(obs_terms)
            if "policy" not in obs_terms and "actor" in obs_terms:
                self._mj_env.observation_manager.active_terms["policy"] = obs_terms[
                    "actor"
                ]
            try:
                metadata = get_base_metadata(self._mj_env, run_name)
            finally:
                self._mj_env.observation_manager.active_terms.pop("policy", None)
                self._mj_env.observation_manager.active_terms.update(
                    active_terms_backup
                )
            metadata["observation_names"] = active_terms_backup.get(
                "actor", active_terms_backup.get("policy", [])
            )
            motion_term = cast(
                MotionCommand, self._mj_env.command_manager.get_term("motion")
            )
            metadata.update(
                {
                    "anchor_body_name": motion_term.cfg.anchor_body_name,
                    "body_names": list(motion_term.cfg.body_names),
                }
            )
            attach_metadata_to_onnx(os.path.join(policy_path, filename), metadata)

        agent.save = save_with_motion_onnx  # type: ignore[method-assign]

    def export_motion_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        os.makedirs(path, exist_ok=True)
        cmd = cast(MotionCommand, self._mj_env.command_manager.get_term("motion"))
        actor_wrapper = self.agent.actor_onnx_wrapper
        model = _OnnxMotionActor(actor_wrapper, cmd.motion)
        model.to("cpu")
        model.eval()
        obs = torch.zeros(1, self.agent.actor_obs_dim)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            model,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
            dynamo=False,
        )

    def save(self, path: str, infos=None) -> None:
        del infos
        self.agent.save(path)
