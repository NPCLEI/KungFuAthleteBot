"""Stubs for holosoma inference helpers."""

from __future__ import annotations

from typing import Any


def export_policy_as_onnx(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("Use MotionTrackingFastSACRunner for ONNX export.")


def attach_onnx_metadata(*args: Any, **kwargs: Any) -> None:
    pass


def export_motion_and_policy_as_onnx(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("Use MotionTrackingFastSACRunner for ONNX export.")


def get_command_ranges_from_env(env: Any) -> dict[str, Any]:
    return {}


def get_control_gains_from_config(config: Any) -> dict[str, Any]:
    return {}


def get_urdf_text_from_robot_config(config: Any) -> str:
    return ""
