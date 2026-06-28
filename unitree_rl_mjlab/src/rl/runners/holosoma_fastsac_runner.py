"""Holosoma FastSAC training runner for mjlab environments (single-GPU)."""

from __future__ import annotations

from typing import Any

from holosoma_min.agents.fast_sac.fast_sac_agent import FastSACAgent
from mjlab.envs import ManagerBasedRlEnv

from src.rl.adapters import MjlabHolosomaEnvAdapter
from src.rl.config import HolosomaFastSACRunnerCfg, to_fastsac_config
from src.rl.training.reward_anomaly_logger import RewardAnomalyCfg
from src.rl.utils.git_state import store_code_state


class HolosomaFastSACRunner:
    """Thin wrapper around holosoma FastSACAgent."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        train_cfg: HolosomaFastSACRunnerCfg,
        log_dir: str,
        device: str,
    ) -> None:
        self.train_cfg = train_cfg
        self.log_dir = log_dir
        self.device = device
        self.git_status_repos: list[str] = []

        adapter = MjlabHolosomaEnvAdapter(env)
        sac_cfg = to_fastsac_config(train_cfg)
        self.agent = FastSACAgent(
            adapter,
            sac_cfg,
            device=device,
            log_dir=log_dir,
            multi_gpu_cfg=None,
        )
        self.agent.reward_anomaly_cfg = RewardAnomalyCfg.from_runner(train_cfg)
        self.agent.setup()
        self._disable_holosoma_onnx_export()

    def _disable_holosoma_onnx_export(self) -> None:
        """learn() calls agent.export(); mjlab uses runner-level ONNX instead."""

        def _noop_export(*_args, **_kwargs) -> None:
            return

        self.agent.export = _noop_export  # type: ignore[method-assign]

    def learn(self, num_learning_iterations: int | None = None) -> None:
        del num_learning_iterations
        self.agent.learn()

    def load(self, path: str) -> None:
        self.agent.load(path)

    def save(self, path: str, infos: Any = None) -> None:
        del infos
        self.agent.save(path)

    def add_git_repo_to_log(self, file_path: str) -> None:
        self.git_status_repos.append(file_path)

    def store_git_state(self) -> None:
        if not self.agent.is_main_process or not self.git_status_repos:
            return
        store_code_state(self.log_dir, self.git_status_repos)
