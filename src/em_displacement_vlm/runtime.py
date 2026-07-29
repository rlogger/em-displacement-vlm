"""Lightweight runtime diagnostics."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from em_displacement_vlm.paths import is_colab, repo_root

_RESEARCH_PACKAGES = (
    "accelerate",
    "datasets",
    "huggingface-hub",
    "numpy",
    "peft",
    "safetensors",
    "transformers",
    "trl",
    "unsloth",
    "unsloth-zoo",
    "wandb",
)


def runtime_info() -> dict[str, object]:
    """Return the environment fingerprint stored with every controlled run."""

    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "colab": is_colab(),
        "repo_root": str(repo_root()),
    }
    packages: dict[str, str] = {}
    for package in _RESEARCH_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    info["packages"] = packages
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda or "none"
        info["cudnn"] = (
            str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "none"
        )
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
        info["cuda_available"] = False
    return info
