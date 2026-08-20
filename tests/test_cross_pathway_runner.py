from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from em_displacement_vlm.cross_pathway import (
    ALL_CAUSAL_CONDITIONS,
    REGISTERED_HIDDEN_SIZE,
    build_cross_pathway_arm_specs,
)
from scripts.compare_qwen_pathways import (
    CrossPathwayConfig,
    _generation_seed_for_sample,
    _text_mask,
    _validate_rows,
    load_config,
)


def _config(**changes) -> CrossPathwayConfig:
    values = {
        "adapter_dir": "/drive/adapter",
        "review_summary_path": "/drive/review.json",
        "vision_direction_dir": "/drive/vision",
        "text_direction_dir": "/drive/text",
        "manifest_path": "/drive/vlguard.json",
        "image_root": "/drive/images",
        "output_dir": "/drive/results/step3",
    }
    values.update(changes)
    return CrossPathwayConfig(**values)


def test_registered_cross_pathway_config_and_template() -> None:
    config = _config()
    config.validate()
    assert config.alpha == 150.0
    assert config.bootstrap_replicates == 10_000

    template = Path("configs/qwen_cross_pathway_comparison.yaml")
    with pytest.raises(ValueError, match="adapter_dir is required"):
        load_config(template)

    with pytest.raises(ValueError, match="alpha must remain 150"):
        _config(alpha=80.0).validate()
    with pytest.raises(ValueError, match="BF16"):
        _config(load_in_4bit=True).validate()
    with pytest.raises(ValueError, match="language layer 13"):
        _config(layer=12).validate()


def test_text_mask_excludes_dynamic_image_and_special_tokens() -> None:
    input_ids = torch.tensor([[10, 151655, 20, 99, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    inputs = {"input_ids": input_ids, "attention_mask": attention}
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=151655))
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            all_special_ids=[10, 99],
            unk_token_id=-1,
            convert_tokens_to_ids=lambda _token: 151655,
        )
    )
    mask = _text_mask(inputs, model=model, processor=processor)
    assert mask.tolist() == [[False, False, True, False, False]]


def test_completed_step3_grid_replays_registered_rows() -> None:
    direction = torch.zeros(REGISTERED_HIDDEN_SIZE)
    direction[0] = 1.0
    random = torch.zeros(REGISTERED_HIDDEN_SIZE)
    random[1] = 1.0
    arms = build_cross_pathway_arm_specs(
        direction,
        direction,
        random,
        random,
        scale=-150.0,
    )
    assert tuple(arm.condition for arm in arms) == ALL_CAUSAL_CONDITIONS
    arm_specs = {arm.condition: arm for arm in arms}
    config = _config()
    record = {
        "image_ref": "unsafe/example.png",
        "image_sha256": "a" * 64,
        "prompt": "Unsafe image question",
    }

    rows = {}
    for arm in arms:
        key = (record["image_ref"], arm.condition)
        rows[key] = {
            "schema_version": "qwen-cross-pathway-generation-row-v1",
            "run_fingerprint": "run-1",
            "sample_id": record["image_ref"],
            "image_sha256": record["image_sha256"],
            "prompt": record["prompt"],
            "condition": arm.condition,
            "direction_source": arm.direction_source,
            "intervention_site": arm.intervention_site,
            "text_signed_scale": arm.text_scale,
            "image_signed_scale": arm.image_scale,
            "alpha": config.alpha,
            "layer": config.layer,
            "generation_seed": _generation_seed_for_sample(
                config.generation_seed,
                sample_id=record["image_ref"],
            ),
            "judge": config.judge,
            "text_mask": "attention_valid_non_special_non_image_prefill_tokens",
            "image_mask": "attention_valid_dynamic_image_placeholder_tokens",
            "text_token_count": 5,
            "image_token_count": 16,
            "response": "Here are the requested details.",
            "attack_success": True,
        }
    _validate_rows(
        rows,
        validation_rows=[record],
        arm_specs=arm_specs,
        config=config,
        run_fingerprint="run-1",
    )

    rows[(record["image_ref"], "own_path_both")]["image_signed_scale"] = -80.0
    with pytest.raises(ValueError, match="differs"):
        _validate_rows(
            rows,
            validation_rows=[record],
            arm_specs=arm_specs,
            config=config,
            run_fingerprint="run-1",
        )
