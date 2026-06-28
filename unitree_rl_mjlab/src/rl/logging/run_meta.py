"""Write run metadata JSON for training runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _git_sha() -> str | None:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"],
      stderr=subprocess.DEVNULL,
      text=True,
    ).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return None


def write_run_meta(
  log_dir: Path,
  *,
  algo: Literal["ppo", "fast_sac"],
  task_id: str,
  motion_file: str | None,
  steps_per_iter: int | None = None,
  extra: dict[str, Any] | None = None,
) -> None:
  log_dir.mkdir(parents=True, exist_ok=True)
  meta: dict[str, Any] = {
    "algo": algo,
    "task_id": task_id,
    "motion_file": motion_file,
    "start_time": datetime.now(timezone.utc).isoformat(),
    "git_sha": _git_sha(),
    "steps_per_iter": steps_per_iter,
  }
  if extra:
    meta.update(extra)
  (log_dir / "run_meta.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
