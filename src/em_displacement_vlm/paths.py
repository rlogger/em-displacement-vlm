"""Shared path helpers for local machines and Google Colab."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (directory that contains pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def ensure_src_on_path() -> Path:
    """Add ``src/`` to ``sys.path`` so imports work before editable install."""
    root = repo_root()
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root


def data_dir(name: str = "data") -> Path:
    """Resolve a data directory, creating it if needed.

    Honors ``EM_DATA_DIR`` so Colab can point at Google Drive without code changes.
    """
    override = os.environ.get("EM_DATA_DIR")
    path = Path(override).expanduser() if override else repo_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(name: str = "checkpoints") -> Path:
    """Resolve a checkpoint directory, creating it if needed.

    Honors ``EM_CHECKPOINT_DIR`` for Drive / remote storage mounts.
    """
    override = os.environ.get("EM_CHECKPOINT_DIR")
    path = Path(override).expanduser() if override else repo_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(name: str = "results") -> Path:
    """Resolve a results directory (gitignored), creating it if needed."""
    override = os.environ.get("EM_RESULTS_DIR")
    path = Path(override).expanduser() if override else repo_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def prompts_dir() -> Path:
    """Resolve the prompts directory under the repo root."""
    path = repo_root() / "prompts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_colab() -> bool:
    """Detect Google Colab runtimes."""
    if "google.colab" in sys.modules:
        return True
    return bool(os.environ.get("COLAB_RELEASE_TAG")) or bool(os.environ.get("COLAB_GPU"))
