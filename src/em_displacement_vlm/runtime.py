"""Lightweight runtime helpers for local vs Colab."""

from __future__ import annotations

import platform
import sys

from em_displacement_vlm.paths import is_colab, repo_root


def runtime_info() -> dict[str, str | bool]:
    """Return a small diagnostic dict for notebook smoke checks."""
    info: dict[str, str | bool] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "colab": is_colab(),
        "repo_root": str(repo_root()),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
        info["cuda_available"] = False
    return info
