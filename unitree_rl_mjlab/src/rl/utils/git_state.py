"""Git state storage for training logs (mjlab 1.2.x compatible)."""

from __future__ import annotations

import os
import pathlib

try:
  import git
except ImportError:  # pragma: no cover
  git = None  # type: ignore[assignment]


def store_code_state(logdir: str, repositories: list[str]) -> list[str]:
  if git is None:
    print("[WARN] GitPython not installed; skip git diff storage.")
    return []

  git_log_dir = os.path.join(logdir, "git")
  os.makedirs(git_log_dir, exist_ok=True)
  file_paths: list[str] = []
  for repository_file_path in repositories:
    try:
      repo = git.Repo(repository_file_path, search_parent_directories=True)
      t = repo.head.commit.tree
    except Exception:
      print(f"Could not find git repository in {repository_file_path}. Skipping.")
      continue
    repo_name = pathlib.Path(repo.working_dir).name
    diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
    if os.path.isfile(diff_file_name):
      continue
    print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
    with open(diff_file_name, "x", encoding="utf-8") as f:
      content = (
        f"--- git status ---\n{repo.git.status()} \n\n\n"
        f"--- git diff ---\n{repo.git.diff(t)}"
      )
      f.write(content)
    file_paths.append(diff_file_name)
  return file_paths
