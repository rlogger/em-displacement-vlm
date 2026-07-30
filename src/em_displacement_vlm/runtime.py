"""Lightweight runtime diagnostics."""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from em_displacement_vlm.paths import is_colab, repo_root


def detach_inherited_wandb_service() -> str | None:
    """Drop a parent-process ``WANDB_SERVICE`` so this process owns wandb-core.

    Colab notebooks that call ``wandb.login()`` start a shared wandb-core service
    and export ``WANDB_SERVICE`` into the kernel environment. Subprocess training
    scripts inherit that handle, then fail with ``run ID … is in use`` when they
    try to resume a run the kernel still holds. Clearing the variable forces a
    fresh service for this process only.
    """
    return os.environ.pop("WANDB_SERVICE", None)


def is_wandb_run_in_use_error(exc: BaseException) -> bool:
    """Return True when wandb rejected init because the run id is still held."""
    return "is in use" in str(exc).lower()

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
