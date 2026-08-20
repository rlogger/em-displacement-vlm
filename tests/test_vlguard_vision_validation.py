from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from em_displacement_vlm.constants import (
    QWEN2_5_VL_3B_MODEL_ID,
    QWEN2_5_VL_3B_REVISION,
)
from em_displacement_vlm.vision_validation import (
    DEFAULT_LAYER,
    VisionSteeringHook,
    VisionValidationConfig,
    build_vlguard_manifest,
    equal_norm_random_direction,
    load_vlguard_metadata,
    mean_difference_direction,
    paired_asr_delta,
    qwen_adapter_provenance,
    qwen_image_token_id,
    resolve_qwen_language_blocks,
    safe_extract_zip,
    select_vlguard_roles,
    summarize_asr,
    validate_registered_vlguard_manifest,
    validate_vlguard_manifest,
    vlguard_attack_success,
)
from scripts.validate_vlguard_vision import _validate_completed_rows


def _metadata_rows(n: int = 4) -> list[dict]:
    rows = []
    for index in range(n):
        rows.append(
            {
                "image": f"safe/{index}.png",
                "safe": True,
                "instr-resp": [
                    {"safe_instruction": f"Safe question {index}", "response": "safe"},
                    {"unsafe_instruction": f"Unsafe question {index}", "response": "refuse"},
                ],
            }
        )
        rows.append(
            {
                "image": f"unsafe/{index}.png",
                "safe": False,
                "instr-resp": [
                    {"instruction": f"Unsafe image question {index}", "response": "refuse"}
                ],
            }
        )
    return rows


def _write_metadata(path: Path, n: int = 4) -> Path:
    path.write_text(json.dumps(_metadata_rows(n)), encoding="utf-8")
    return path


def _archive(path: Path, n: int = 4) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for index in range(n):
            bundle.writestr(f"train/safe/{index}.png", f"safe-{index}".encode())
            bundle.writestr(f"train/unsafe/{index}.png", f"unsafe-{index}".encode())
    return path


def test_vlguard_parser_uses_official_image_safe_instr_resp_schema(tmp_path: Path) -> None:
    records = load_vlguard_metadata(_write_metadata(tmp_path / "train.json"))
    assert len(records) == 8
    assert sum(record.safe for record in records) == 4
    assert records[1].validation_instruction == "Unsafe image question 0"

    malformed = _metadata_rows(1)
    malformed[0]["safe"] = "safe"
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="must be boolean"):
        load_vlguard_metadata(path)

    malformed = _metadata_rows(1)
    malformed[1]["image"] = "../escape.png"
    path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="unsafe image path"):
        load_vlguard_metadata(path)


def test_vlguard_role_selection_is_deterministic_balanced_and_disjoint(tmp_path: Path) -> None:
    records = load_vlguard_metadata(_write_metadata(tmp_path / "train.json", n=6))
    first = select_vlguard_roles(
        records,
        direction_per_class=2,
        validation_unsafe=3,
        selection_seed=17,
    )
    second = select_vlguard_roles(
        records,
        direction_per_class=2,
        validation_unsafe=3,
        selection_seed=17,
    )
    assert first == second
    assert len(first["direction_safe"]) == len(first["direction_unsafe"]) == 2
    refs = [record.image_ref for rows in first.values() for record in rows]
    assert len(refs) == len(set(refs)) == 7

    with pytest.raises(ValueError, match="need 7"):
        select_vlguard_roles(records, direction_per_class=2, validation_unsafe=5)


def test_vlguard_archive_rejects_traversal_and_reuses_only_matching_hash(tmp_path: Path) -> None:
    malicious = tmp_path / "bad.zip"
    with zipfile.ZipFile(malicious, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="escapes extraction root"):
        safe_extract_zip(malicious, tmp_path / "bad-out")

    archive = _archive(tmp_path / "train.zip")
    destination = tmp_path / "images"
    first = safe_extract_zip(archive, destination)
    assert first["member_count"] == 8
    assert safe_extract_zip(archive, destination) == first

    different = _archive(tmp_path / "different.zip", n=3)
    with pytest.raises(ValueError, match="different archive hash"):
        safe_extract_zip(different, destination)


def test_manifest_binds_unpaired_roles_and_selected_image_bytes(tmp_path: Path) -> None:
    metadata = _write_metadata(tmp_path / "train.json", n=5)
    archive = _archive(tmp_path / "train.zip", n=5)
    images = tmp_path / "images"
    safe_extract_zip(archive, images)
    manifest = build_vlguard_manifest(
        metadata_path=metadata,
        archive_path=archive,
        image_root=images,
        direction_per_class=2,
        validation_unsafe=2,
        selection_seed=23,
    )
    validated = validate_vlguard_manifest(manifest, image_root=images)
    assert validated["selection"]["pairing"] == "unpaired_safe_vs_unsafe_image_groups"
    assert validated["selection"]["counts"] == {
        "direction_safe": 2,
        "direction_unsafe": 2,
        "validation_unsafe": 2,
    }
    assert all(
        row.get("prompt")
        for row in validated["records"]
        if row["role"] == "validation_unsafe"
    )

    chosen = validated["records"][0]
    path = images / "train" / chosen["image_ref"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_vlguard_manifest(manifest, image_root=images)

    with pytest.raises(ValueError, match="registered primary roles"):
        validate_registered_vlguard_manifest(manifest)


class _FakeTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|image_pad|>"
        return 151655


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_id=151655)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Identity() for _ in range(16)])


def test_qwen_layer_and_image_token_resolution_are_dynamic() -> None:
    model = _FakeQwen()
    name, blocks = resolve_qwen_language_blocks(model, layer=13)
    assert name in {"model.language_model", "model.language_model.layers"}
    assert len(blocks) == 16
    processor = SimpleNamespace(tokenizer=_FakeTokenizer())
    assert qwen_image_token_id(model, processor) == 151655


def test_vision_steering_changes_only_masked_prefill_tokens_once() -> None:
    model = _FakeQwen()
    mask = torch.tensor([[False, True, True, False]])
    direction = torch.tensor([1.0, -1.0])
    hidden = torch.zeros(1, 4, 2)
    layer = model.model.language_model.layers[13]
    with VisionSteeringHook(
        model,
        layer=13,
        image_mask=mask,
        direction=direction,
        scale=2.0,
    ) as steering:
        first = layer(hidden)
        second = layer(hidden)
        steering.require_applied()
    assert steering.applied_tokens == 2
    assert torch.equal(first[0, 0], torch.zeros(2))
    assert torch.equal(first[0, 1], torch.tensor([2.0, -2.0]))
    assert torch.equal(first[0, 2], torch.tensor([2.0, -2.0]))
    assert torch.equal(first[0, 3], torch.zeros(2))
    assert torch.equal(second, hidden)


def test_direction_and_random_control_are_finite_equal_norm_and_seeded() -> None:
    safe = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
    unsafe = torch.tensor([[2.0, 1.0], [2.0, 3.0]])
    direction, raw_norm = mean_difference_direction(unsafe, safe)
    assert raw_norm == pytest.approx(2.0)
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]))
    random_a = equal_norm_random_direction(direction, seed=9)
    random_b = equal_norm_random_direction(direction, seed=9)
    assert torch.equal(random_a, random_b)
    assert random_a.norm().item() == pytest.approx(direction.norm().item())

    with pytest.raises(ValueError, match="zero or non-finite"):
        mean_difference_direction(torch.ones(2, 3), torch.ones(2, 3))


def test_vlguard_keyword_asr_and_summary() -> None:
    assert not vlguard_attack_success("I'm sorry, but I cannot help with that.")
    assert vlguard_attack_success("Here are the requested detailed steps.")
    summary = summarize_asr(
        [
            {"condition": "baseline", "attack_success": True},
            {"condition": "baseline", "attack_success": False},
            {"condition": "repair", "attack_success": False},
        ]
    )
    assert summary["baseline"]["asr_percent"] == 50.0
    assert summary["repair"]["asr_percent"] == 0.0


def test_paired_asr_delta_uses_same_images_and_is_deterministic() -> None:
    rows = [
        {"image_ref": "a", "condition": "baseline", "attack_success": True},
        {"image_ref": "a", "condition": "repair", "attack_success": False},
        {"image_ref": "b", "condition": "baseline", "attack_success": True},
        {"image_ref": "b", "condition": "repair", "attack_success": True},
    ]
    first = paired_asr_delta(
        rows,
        reference="baseline",
        comparison="repair",
        seed=7,
        replicates=1000,
    )
    second = paired_asr_delta(
        rows,
        reference="baseline",
        comparison="repair",
        seed=7,
        replicates=1000,
    )
    assert first == second
    assert first["n_images"] == 2
    assert first["delta_points"] == -50.0

    with pytest.raises(ValueError, match="both conditions"):
        paired_asr_delta(
            rows[:-1],
            reference="baseline",
            comparison="repair",
            replicates=1000,
        )


def _adapter(tmp_path: Path, *, seed: int = 42) -> Path:
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "adapter_model.safetensors").write_bytes(b"weights")
    (root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": QWEN2_5_VL_3B_MODEL_ID})
    )
    (root / "spec.json").write_text(json.dumps({"model_id": QWEN2_5_VL_3B_MODEL_ID}))
    reproduction = root / "reproduction_manifest.json"
    reproduction.write_text('{"run":"qwen"}\n')
    reproduction_hash = hashlib.sha256(reproduction.read_bytes()).hexdigest()
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "reproduction_manifest_sha256": reproduction_hash,
                    "effective_training_config": {
                        "model_family": "qwen2_5_vl",
                        "base_model": QWEN2_5_VL_3B_MODEL_ID,
                        "base_model_revision": QWEN2_5_VL_3B_REVISION,
                        "seed": seed,
                    },
                }
            }
        )
    )
    return root


def _config(tmp_path: Path, adapter: Path) -> VisionValidationConfig:
    return VisionValidationConfig(
        adapter_dir=str(adapter),
        manifest_path=str(tmp_path / "manifest.json"),
        image_root=str(tmp_path / "images"),
        output_dir=str(tmp_path / "results"),
    )


def test_qwen_adapter_provenance_binds_model_revision_and_seed(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    provenance = qwen_adapter_provenance(_config(tmp_path, adapter))
    assert provenance["model_family"] == "qwen2_5_vl"
    assert provenance["training_seed"] == 42

    metadata = json.loads((adapter / "run_metadata.json").read_text())
    metadata["provenance"]["effective_training_config"]["seed"] = 43
    (adapter / "run_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="training identity differs"):
        qwen_adapter_provenance(_config(tmp_path, adapter))


def test_vision_validation_config_freezes_qwen_vlguard_and_layer_13(tmp_path: Path) -> None:
    config = _config(tmp_path, _adapter(tmp_path))
    config.validate()
    assert config.layer == DEFAULT_LAYER

    with pytest.raises(ValueError, match="language layer 13"):
        VisionValidationConfig(
            adapter_dir=config.adapter_dir,
            manifest_path=config.manifest_path,
            image_root=config.image_root,
            output_dir=config.output_dir,
            layer=12,
        ).validate()
    with pytest.raises(ValueError, match="BF16"):
        VisionValidationConfig(
            adapter_dir=config.adapter_dir,
            manifest_path=config.manifest_path,
            image_root=config.image_root,
            output_dir=config.output_dir,
            load_in_4bit=True,
        ).validate()
    with pytest.raises(ValueError, match=r"\[80, 150, 250\]"):
        VisionValidationConfig(
            adapter_dir=config.adapter_dir,
            manifest_path=config.manifest_path,
            image_root=config.image_root,
            output_dir=config.output_dir,
            alphas=(150.0,),
        ).validate()
    with pytest.raises(ValueError, match="deterministic greedy"):
        VisionValidationConfig(
            adapter_dir=config.adapter_dir,
            manifest_path=config.manifest_path,
            image_root=config.image_root,
            output_dir=config.output_dir,
            do_sample=True,
            temperature=0.7,
        ).validate()
    with pytest.raises(ValueError, match="direction_prompt"):
        VisionValidationConfig(
            adapter_dir=config.adapter_dir,
            manifest_path=config.manifest_path,
            image_root=config.image_root,
            output_dir=config.output_dir,
            direction_prompt="Another prompt",
        ).validate()


def test_completed_generation_rows_replay_every_bound_field(tmp_path: Path) -> None:
    config = _config(tmp_path, _adapter(tmp_path))
    record = {
        "image_ref": "unsafe/one.png",
        "image_sha256": "a" * 64,
        "prompt": "Unsafe question",
    }
    from em_displacement_vlm.vision_validation import generation_seed_for

    row = {
        "schema_version": "qwen-vlguard-vision-causal-validation-v1",
        "run_fingerprint": "run-1",
        **record,
        "condition": "baseline",
        "alpha": None,
        "signed_scale": 0.0,
        "layer": 13,
        "which": "vis",
        "generation_seed": generation_seed_for(
            config.generation_seed,
            image_ref=record["image_ref"],
            condition="baseline",
        ),
        "image_token_count": 16,
        "response": "Here are the requested details.",
        "attack_success": True,
        "judge": config.judge,
    }
    completed = {(record["image_ref"], "baseline"): row}
    _validate_completed_rows(
        completed,
        validation_rows=[record],
        condition_specs={"baseline": (0.0, None)},
        config=config,
        run_fingerprint="run-1",
    )

    row["attack_success"] = False
    with pytest.raises(ValueError, match="does not replay"):
        _validate_completed_rows(
            completed,
            validation_rows=[record],
            condition_specs={"baseline": (0.0, None)},
            config=config,
            run_fingerprint="run-1",
        )
