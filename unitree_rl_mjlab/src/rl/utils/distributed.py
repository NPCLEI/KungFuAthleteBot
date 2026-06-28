"""Device helpers for training workers."""

from __future__ import annotations

import os

import torch

MJLAB_TRAIN_LOG_DIR_ENV = "MJLAB_TRAIN_LOG_DIR"


def local_cuda_device_index(local_rank: int | None = None) -> int:
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        return 0
    count = torch.cuda.device_count()
    if count <= 0:
        return 0
    if local_rank >= count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {count} CUDA device(s) are visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r})."
        )
    return local_rank


def local_cuda_device(local_rank: int | None = None) -> str:
    idx = local_cuda_device_index(local_rank)
    return "cpu" if not torch.cuda.is_available() else f"cuda:{idx}"


def mujoco_egl_device_id(local_rank: int | None = None) -> int:
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parts = [int(x.strip()) for x in visible.split(",") if x.strip() != ""]
        if parts and local_rank < len(parts):
            return parts[local_rank]
    return local_rank
