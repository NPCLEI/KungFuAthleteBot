"""Minimal module utils."""
from pathlib import Path


def get_holosoma_root() -> Path:
    return Path(__file__).resolve().parents[2]
