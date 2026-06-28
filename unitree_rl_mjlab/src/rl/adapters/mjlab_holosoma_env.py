"""Bridge mjlab ManagerBasedRlEnv to holosoma FastSAC BaseTask contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv

from src.rl.adapters.robot_config import MjlabRobotConfig, build_robot_config_from_mjlab_env

# mjlab observation group -> holosoma FastSAC keys
DEFAULT_OBS_MAP: dict[str, str] = {
    "actor": "actor_obs",
    "critic": "critic_obs",
}
INVERSE_OBS_MAP: dict[str, str] = {v: k for k, v in DEFAULT_OBS_MAP.items()}


@dataclass
class _ObsGroupCfgShim:
    terms: dict[str, Any] = field(default_factory=dict)
    history_length: int = 1


@dataclass
class _ObsManagerCfgShim:
    groups: dict[str, _ObsGroupCfgShim]


class _ObservationManagerShim:
    """Exposes holosoma-style get_obs_dims() over mjlab observation manager."""

    def __init__(
        self,
        mj_env: ManagerBasedRlEnv,
        obs_map: dict[str, str],
    ) -> None:
        self._mj = mj_env
        self._obs_map = obs_map
        holosoma_groups: dict[str, _ObsGroupCfgShim] = {}
        for mj_key, holo_key in obs_map.items():
            if mj_key not in mj_env.observation_manager.active_terms:
                continue
            term_names = mj_env.observation_manager.active_terms[mj_key]
            holosoma_groups[holo_key] = _ObsGroupCfgShim(
                terms={name: None for name in term_names},
                history_length=1,
            )
        self.cfg = _ObsManagerCfgShim(groups=holosoma_groups)

    def get_obs_dims(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        for mj_key, holo_key in self._obs_map.items():
            group_dim = self._mj.observation_manager.group_obs_dim.get(mj_key)
            if group_dim is None:
                continue
            if isinstance(group_dim, tuple):
                dims[holo_key] = int(group_dim[0])
            else:
                dims[holo_key] = int(sum(d[0] for d in group_dim))
        return dims


class MjlabHolosomaEnvAdapter:
    """Holosoma BaseTask-compatible wrapper for mjlab tracking envs."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        obs_map: dict[str, str] | None = None,
    ) -> None:
        self._env = env
        self._obs_map = obs_map or dict(DEFAULT_OBS_MAP)
        self.device = env.device
        self.num_envs = env.num_envs
        self.robot_config: MjlabRobotConfig = build_robot_config_from_mjlab_env(env)
        self.observation_manager = _ObservationManagerShim(env, self._obs_map)
        self.log_dict: dict[str, Any] = {}

        # Buffers aligned with holosoma BaseTask
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.extras: dict[str, Any] = {}
        self.obs_buf_dict: dict[str, torch.Tensor] = {}

        self._env.reset()

    def reset_all(self) -> dict[str, torch.Tensor]:
        self._env.reset()
        actions = torch.zeros(
            self.num_envs,
            self.robot_config.actions_dim,
            device=self.device,
        )
        return self.step({"actions": actions})[0]

    def step(
        self, actor_state: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions = actor_state["actions"].to(self.device)
        self._env.action_manager.process_action(actions)

        for _ in range(self._env.cfg.decimation):
            self._env._sim_step_counter += 1
            self._env.action_manager.apply_action()
            self._env.scene.write_data_to_sim()
            self._env.sim.step()
            self._env.scene.update(dt=self._env.physics_dt)

        self._env.episode_length_buf += 1
        self._env.common_step_counter += 1

        self._env.reset_buf = self._env.termination_manager.compute()
        self._env.reset_terminated = self._env.termination_manager.terminated
        self._env.reset_time_outs = self._env.termination_manager.time_outs

        self.rew_buf[:] = self._env.reward_manager.compute(dt=self._env.step_dt)

        reset_env_ids = self._env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        final_obs_dict: dict[str, torch.Tensor] = {}
        if reset_env_ids.numel() > 0:
            if reset_env_ids.dim() == 0:
                reset_env_ids = reset_env_ids.unsqueeze(0)
            raw_obs = self._env.observation_manager.compute(update_history=True)
            final_obs_dict = self._map_obs_dict(raw_obs)

        if reset_env_ids.numel() > 0:
            self._env._reset_idx(reset_env_ids)
            self._env.scene.write_data_to_sim()
            self._env.sim.forward()

        self._env.command_manager.compute(dt=self._env.step_dt)
        if "interval" in self._env.event_manager.available_modes:
            self._env.event_manager.apply(
                mode="interval", dt=self._env.step_dt
            )

        raw_obs = self._env.observation_manager.compute(update_history=True)
        self.obs_buf_dict = self._map_obs_dict(raw_obs)

        self.reset_buf[:] = (
            self._env.reset_terminated | self._env.reset_time_outs
        ).to(dtype=torch.long)
        self.time_out_buf[:] = self._env.reset_time_outs

        episode_stats = self._collect_episode_stats(reset_env_ids)
        info_dict = self._build_info_dict(final_obs_dict, episode_stats)

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, info_dict

    def _map_obs_dict(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for mj_key, holo_key in self._obs_map.items():
            if mj_key in raw:
                out[holo_key] = raw[mj_key]
        return out

    def _collect_episode_stats(
        self, reset_env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Build holosoma-style episode tensors for completed envs."""
        episode: dict[str, torch.Tensor] = {}
        episode_all: dict[str, torch.Tensor] = {}
        if reset_env_ids.numel() == 0:
            return {"episode": episode, "episode_all": episode_all}

        for name, buf in self._env.reward_manager._episode_sums.items():
            if reset_env_ids.numel() > 0:
                vals = buf[reset_env_ids].clone()
                episode[f"rew_{name}"] = vals
                episode_all[f"rew_{name}"] = buf.clone()
        episode["r"] = self.rew_buf[reset_env_ids].clone()
        episode_all["r"] = self.rew_buf.clone()
        return {"episode": episode, "episode_all": episode_all}

    def _build_info_dict(
        self,
        final_obs_dict: dict[str, torch.Tensor],
        episode_stats: dict[str, Any],
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "time_outs": self.time_out_buf.clone(),
            "episode": episode_stats.get("episode", {}),
            "episode_all": episode_stats.get("episode_all", {}),
            "raw_episode": {},
            "raw_episode_all": {},
            "to_log": self._tensor_log_dict(),
        }
        if final_obs_dict:
            store = {}
            for k, v in final_obs_dict.items():
                if k not in store:
                    store[k] = torch.zeros_like(self.obs_buf_dict[k])
                reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if reset_ids.numel() > 0:
                    if reset_ids.dim() == 0:
                        reset_ids = reset_ids.unsqueeze(0)
                    store[k][reset_ids] = final_obs_dict[k][reset_ids]
            info["final_observations"] = store
        return info

    def _tensor_log_dict(self) -> dict[str, torch.Tensor]:
        raw = self._env.extras.get("log", {})
        out: dict[str, torch.Tensor] = {}
        for key, val in raw.items():
            if isinstance(val, torch.Tensor):
                out[key] = val
            elif isinstance(val, (int, float)):
                out[key] = torch.tensor(
                    float(val), device=self.device, dtype=torch.float32
                )
        return out

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        """Optional env state for FastSAC checkpoints (not used for mjlab)."""
        return None

    def synchronize_curriculum_state(self, device: str, world_size: int) -> None:
        del device, world_size

    def get_step_reward_breakdown(
        self, env_ids: torch.Tensor
    ) -> tuple[list[str], torch.Tensor]:
        """Return scaled per-term rewards for selected envs.

        Each term value is ``raw * weight * dt``; summing terms equals ``rew_buf``.
        """
        if env_ids.dim() == 0:
            env_ids = env_ids.unsqueeze(0)
        rm = self._env.reward_manager
        dt = self._env.step_dt
        names = list(rm.active_terms)
        scaled = rm._step_reward[env_ids] * dt
        return names, scaled

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self._env

    @property
    def command_manager(self):
        return self._env.command_manager
