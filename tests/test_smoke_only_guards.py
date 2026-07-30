"""Fail-closed checks for legacy TinyTwoTower smoke helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from em_displacement_vlm.extraction import aggregate_tokens, register_hooks
from em_displacement_vlm.interventions import (
    BlockEMConfig,
    BlockEMTrainerStep,
    block_penalty,
    wrong_layer_direction,
)
from em_displacement_vlm.models import ModelSpec, ModelState, load_model_bundle


def test_smoke_aggregation_rejects_empty_token_slices():
    hidden = torch.randn(2, 4, 8)

    with pytest.raises(ValueError, match="selected no tokens"):
        aggregate_tokens(hidden, "vision", visual_start=4, visual_end=4)
    with pytest.raises(ValueError, match="selected no tokens"):
        aggregate_tokens(hidden, "text", text_start=4)


def test_smoke_penalty_rejects_empty_text_slice():
    hidden = torch.randn(2, 4, 8)
    direction = torch.randn(8)

    with pytest.raises(ValueError, match="no token positions"):
        block_penalty(hidden, direction, cfg=BlockEMConfig(text_token_start=4))


def test_smoke_extractor_and_trainer_reject_non_fixture_models():
    model = nn.Linear(8, 8)

    with pytest.raises(RuntimeError, match="smoke-only"):
        register_hooks(model)
    with pytest.raises(RuntimeError, match="smoke-only"):
        BlockEMTrainerStep(model, torch.randn(8))


def test_smoke_model_factory_rejects_real_model_requests():
    spec = ModelSpec(state=ModelState.BASE, model_id="google/gemma-3-4b-it")

    with pytest.raises(RuntimeError, match="TinyTwoTower smoke-only"):
        load_model_bundle(spec)


def test_wrong_layer_direction_never_substitutes_random_control():
    acts = {"text:20": torch.ones(2, 8)}

    with pytest.raises(ValueError, match="refusing to substitute a random direction"):
        wrong_layer_direction(acts, acts, layers=(15,))
