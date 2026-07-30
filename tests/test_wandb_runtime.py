"""W&B subprocess isolation helpers used by Colab FT/sanity entrypoints."""

from __future__ import annotations

import os

from em_displacement_vlm.runtime import (
    detach_inherited_wandb_service,
    is_wandb_run_in_use_error,
)


def test_detach_inherited_wandb_service_removes_env(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_SERVICE", "sock:///tmp/fake-wandb.sock")
    assert detach_inherited_wandb_service() == "sock:///tmp/fake-wandb.sock"
    assert "WANDB_SERVICE" not in os.environ


def test_detach_inherited_wandb_service_noop_when_absent(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_SERVICE", raising=False)
    assert detach_inherited_wandb_service() is None
    assert "WANDB_SERVICE" not in os.environ


def test_wandb_run_in_use_error_detection() -> None:
    assert is_wandb_run_in_use_error(RuntimeError("run ID m1y7tecl is in use"))
    assert not is_wandb_run_in_use_error(RuntimeError("permission denied"))
