from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from em_displacement_vlm.constants import (
    QWEN2_5_VL_3B_MODEL_ID,
    QWEN2_5_VL_3B_REVISION,
)
from em_displacement_vlm.cross_pathway import (
    ALL_CAUSAL_CONDITIONS,
    CONSTRUCTION_FILENAME,
    DIRECTION_FILENAME,
    DIRECTION_METADATA_FILENAME,
    MODEL_FAMILY,
    RANDOM_BOTH_CONDITION,
    REAL_BOTH_CONDITION,
    REGISTERED_HIDDEN_SIZE,
    REGISTERED_HOOK_SEMANTICS,
    REGISTERED_LAYER,
    REGISTERED_ORIENTATION,
    REGISTERED_RESIDUAL_SITE,
    RUN_METADATA_FILENAME,
    SOURCE_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    QwenPathwaySteeringHook,
    build_cross_pathway_arm_specs,
    build_direction_package_manifest,
    canonical_json_sha256,
    compare_cross_pathway_geometry,
    load_direction_package,
    sha256_file,
    summarize_paired_cross_pathway_arms,
    write_direction_package_manifest,
)

ADAPTER_FINGERPRINT = "a" * 64
RUN_FINGERPRINT = "b" * 64


def _identity() -> dict[str, object]:
    return {
        "fingerprint": ADAPTER_FINGERPRINT,
        "model_family": MODEL_FAMILY,
        "base_model_id": QWEN2_5_VL_3B_MODEL_ID,
        "base_model_revision": QWEN2_5_VL_3B_REVISION,
        "training_seed": 42,
    }


def _write_package(root: Path, pathway: str, *, negate_direction: bool = False) -> Path:
    root.mkdir()
    generator = torch.Generator().manual_seed(17 if pathway == "text" else 23)
    axis = torch.zeros(REGISTERED_HIDDEN_SIZE)
    axis[0] = 1.0
    random = torch.zeros_like(axis)
    random[1] = 1.0

    if pathway == "text":
        construction = 0.001 * torch.randn(8, REGISTERED_HIDDEN_SIZE, generator=generator)
        construction += axis
        direction = construction.mean(dim=0)
        direction = direction / direction.norm()
        construction_tensors = {"text_paired_deltas": construction}
        direction_key = "text_direction"
    else:
        safe = 0.01 * torch.randn(8, REGISTERED_HIDDEN_SIZE, generator=generator)
        unsafe = safe + axis
        direction = unsafe.mean(dim=0) - safe.mean(dim=0)
        direction = direction / direction.norm()
        construction_tensors = {
            "vision_safe_activations": safe,
            "vision_unsafe_activations": unsafe,
        }
        direction_key = "vision_direction"
    if negate_direction:
        direction = -direction

    save_file(
        {direction_key: direction.contiguous(), "random_equal_norm": random},
        str(root / DIRECTION_FILENAME),
    )
    save_file(construction_tensors, str(root / CONSTRUCTION_FILENAME))

    source_manifest = {
        "schema_version": f"synthetic-{pathway}-manifest-v1",
        "records": [{"id": f"{pathway}-{index}"} for index in range(8)],
    }
    source_manifest["manifest_sha256"] = canonical_json_sha256(source_manifest)
    (root / SOURCE_MANIFEST_FILENAME).write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_hash = source_manifest["manifest_sha256"]
    adapter = _identity()
    run_metadata = {
        "run_fingerprint": RUN_FINGERPRINT,
        "manifest_sha256": manifest_hash,
        "adapter": adapter,
    }
    summary = {
        "run_fingerprint": RUN_FINGERPRINT,
        "manifest_sha256": manifest_hash,
        "adapter": adapter,
        "status": f"MEASURED_{pathway.upper()}_DIRECTION_SCREEN",
        "claim_boundary": "Synthetic unit-test screen; no scientific conclusion.",
    }
    (root / RUN_METADATA_FILENAME).write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    direction_metadata = {
        "run_fingerprint": RUN_FINGERPRINT,
        "tensor_sha256": sha256_file(root / DIRECTION_FILENAME),
        "construction_sha256": sha256_file(root / CONSTRUCTION_FILENAME),
        "layer": REGISTERED_LAYER,
        "hidden_size": REGISTERED_HIDDEN_SIZE,
        "residual_site": REGISTERED_RESIDUAL_SITE,
        "hook_semantics": REGISTERED_HOOK_SEMANTICS,
        "orientation": REGISTERED_ORIENTATION,
    }
    (root / DIRECTION_METADATA_FILENAME).write_text(
        json.dumps(direction_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _seal(root: Path, pathway: str) -> Path:
    return write_direction_package_manifest(
        root,
        pathway=pathway,  # type: ignore[arg-type]
        adapter_fingerprint=ADAPTER_FINGERPRINT,
        training_seed=42,
        hidden_size=REGISTERED_HIDDEN_SIZE,
        run_fingerprint=RUN_FINGERPRINT,
    )


def test_direction_packages_replay_construction_and_all_bound_artifacts(tmp_path: Path) -> None:
    text_root = _write_package(tmp_path / "text", "text")
    vision_root = _write_package(tmp_path / "vision", "vision")
    _seal(text_root, "text")
    _seal(vision_root, "vision")

    text = load_direction_package(text_root, expected_pathway="text")
    vision = load_direction_package(vision_root, expected_pathway="vision")
    assert text.direction.shape == (REGISTERED_HIDDEN_SIZE,)
    assert vision.direction.shape == text.direction.shape
    assert torch.allclose(text.direction.norm(), torch.tensor(1.0))
    assert text.adapter_fingerprint == vision.adapter_fingerprint

    manifest = build_direction_package_manifest(
        vision_root,
        pathway="vision",
        adapter_fingerprint=ADAPTER_FINGERPRINT,
        training_seed=42,
        hidden_size=REGISTERED_HIDDEN_SIZE,
        run_fingerprint=RUN_FINGERPRINT,
    )
    assert set(manifest["artifacts"]) == {
        DIRECTION_FILENAME,
        CONSTRUCTION_FILENAME,
        DIRECTION_METADATA_FILENAME,
        RUN_METADATA_FILENAME,
        SUMMARY_FILENAME,
        SOURCE_MANIFEST_FILENAME,
    }


def test_direction_package_rejects_tampering_and_wrong_orientation(tmp_path: Path) -> None:
    tampered = _write_package(tmp_path / "tampered", "vision")
    _seal(tampered, "vision")
    with (tampered / SUMMARY_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_direction_package(tampered, expected_pathway="vision")

    wrong_sign = _write_package(tmp_path / "wrong-sign", "text", negate_direction=True)
    with pytest.raises(ValueError, match="unsafe-minus-safe orientation"):
        build_direction_package_manifest(
            wrong_sign,
            pathway="text",
            adapter_fingerprint=ADAPTER_FINGERPRINT,
            training_seed=42,
            hidden_size=REGISTERED_HIDDEN_SIZE,
            run_fingerprint=RUN_FINGERPRINT,
        )


def test_cross_pathway_geometry_is_signed_deterministic_and_screen_tier(tmp_path: Path) -> None:
    text_root = _write_package(tmp_path / "text", "text")
    vision_root = _write_package(tmp_path / "vision", "vision")
    _seal(text_root, "text")
    _seal(vision_root, "vision")
    text = load_direction_package(text_root, expected_pathway="text")
    vision = load_direction_package(vision_root, expected_pathway="vision")

    first = compare_cross_pathway_geometry(
        text,
        vision,
        bootstrap_replicates=50,
        permutation_replicates=50,
    )
    second = compare_cross_pathway_geometry(
        text,
        vision,
        bootstrap_replicates=50,
        permutation_replicates=50,
    )
    assert first == second
    assert first["status"] == "MEASURED_CROSS_PATHWAY_GEOMETRY_SCREEN"
    assert first["signed_cosine"] > 0.99
    assert first["angle_degrees"] < 1.0
    assert len(first["bootstrap"]["ci95_signed_cosine"]) == 2
    assert first["permutation_null"]["replicates"] == 50
    assert "BLOCK-EM" in first["claim_boundary"]


class _TupleBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden,)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TupleBlock() for _ in range(14)])


class _InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _InnerModel()


def test_qwen_pathway_hook_supports_single_and_simultaneous_prefill_only() -> None:
    model = _FakeQwen()
    hidden = torch.zeros(1, 4, REGISTERED_HIDDEN_SIZE)
    text_mask = torch.tensor([[False, False, True, True]])
    image_mask = torch.tensor([[True, True, False, False]])
    text_direction = torch.zeros(REGISTERED_HIDDEN_SIZE)
    text_direction[0] = 1.0
    image_direction = torch.zeros(REGISTERED_HIDDEN_SIZE)
    image_direction[1] = 1.0
    block = model.model.language_model.layers[REGISTERED_LAYER]

    with QwenPathwaySteeringHook(
        model,
        text_mask=text_mask,
        text_direction=text_direction,
        text_scale=2.0,
        image_mask=image_mask,
        image_direction=image_direction,
        image_scale=3.0,
    ) as hook:
        first = block(hidden)[0]
        hook.require_applied()
        second = block(hidden)[0]
    assert torch.all(first[0, :2, 1] == 3.0)
    assert torch.all(first[0, 2:, 0] == 2.0)
    assert torch.count_nonzero(second) == 0
    assert hook.applied_counts == {"text": 2, "image": 2}

    with QwenPathwaySteeringHook(
        model,
        text_mask=text_mask,
        text_direction=text_direction,
        text_scale=-1.0,
    ) as text_hook:
        text_only = block(hidden)[0]
        text_hook.require_applied()
    assert torch.all(text_only[0, 2:, 0] == -1.0)
    assert torch.count_nonzero(text_only[0, :2]) == 0

    with pytest.raises(ValueError, match="must be disjoint"):
        QwenPathwaySteeringHook(
            model,
            text_mask=text_mask,
            text_direction=text_direction,
            text_scale=1.0,
            image_mask=text_mask,
            image_direction=image_direction,
            image_scale=1.0,
        )


def test_arm_specs_and_paired_summary_require_complete_registered_grid() -> None:
    text = torch.zeros(REGISTERED_HIDDEN_SIZE)
    text[0] = 1.0
    vision = torch.zeros(REGISTERED_HIDDEN_SIZE)
    vision[1] = 1.0
    random_text = torch.zeros(REGISTERED_HIDDEN_SIZE)
    random_text[2] = 1.0
    random_vision = torch.zeros(REGISTERED_HIDDEN_SIZE)
    random_vision[3] = 1.0
    arms = build_cross_pathway_arm_specs(
        text,
        vision,
        random_text,
        random_vision,
        scale=-150.0,
    )
    assert tuple(arm.condition for arm in arms) == ALL_CAUSAL_CONDITIONS
    own_both = next(arm for arm in arms if arm.condition == REAL_BOTH_CONDITION)
    random_both = next(arm for arm in arms if arm.condition == RANDOM_BOTH_CONDITION)
    assert own_both.text_direction is not None and own_both.image_direction is not None
    assert random_both.text_direction is not None and random_both.image_direction is not None

    rows = []
    for sample_index in range(4):
        for condition in ALL_CAUSAL_CONDITIONS:
            rows.append(
                {
                    "sample_id": f"sample-{sample_index}",
                    "condition": condition,
                    "attack_success": condition == "baseline" or sample_index == 0,
                }
            )
    first = summarize_paired_cross_pathway_arms(
        rows,
        bootstrap_seed=9,
        bootstrap_replicates=100,
    )
    second = summarize_paired_cross_pathway_arms(
        rows,
        bootstrap_seed=9,
        bootstrap_replicates=100,
    )
    assert first == second
    assert first["status"] == "MEASURED_CROSS_PATHWAY_CAUSAL_SCREEN"
    assert set(first["real_direction_by_site_2x2"]) == {"text_direction", "vision_direction"}
    assert first["real_own_path_both"]["comparison"] == REAL_BOTH_CONDITION
    assert first["random_both_own"]["comparison"] == RANDOM_BOTH_CONDITION

    with pytest.raises(ValueError, match="incomplete"):
        summarize_paired_cross_pathway_arms(rows[:-1], bootstrap_replicates=10)
