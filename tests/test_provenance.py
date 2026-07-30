"""Fail-closed provenance, masking, and reviewed-push contracts."""

from __future__ import annotations

import hashlib
import json
from collections import UserDict
from pathlib import Path

import pytest
import torch

from em_displacement_vlm.data import (
    PRIMARY_MANIFEST_VERSION,
    frozen_split_provenance,
    prepare_all_datasets,
    require_paper_comparable_evaluation,
    verify_frozen_manifest,
)
from em_displacement_vlm.evals.sanity_em import (
    SanityConfig,
    SanitySampleResult,
    adapter_fingerprint,
    generation_seed_for,
    inspect_model_provenance,
    save_check_bundle,
)
from em_displacement_vlm.ft import ResponseOnlyVisionDataCollator, audit_response_only_label_mask


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_split(tmp_path: Path) -> dict[str, object]:
    prepare_all_datasets(seed=42, use_hf=False, out_root=tmp_path)
    return frozen_split_provenance(
        tmp_path,
        expected_mode="offline_fixture",
        expected_seed=42,
        expected_dataset_id="offline_fixture",
        expected_dataset_revision="fixture-v1",
        expected_counts={"finetune": 1500},
    )


def _adapter_with_provenance(tmp_path: Path, split: dict[str, object]) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "base"}\n')
    (adapter / "spec.json").write_text('{"state": "ft"}\n')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    reproduction = adapter / "reproduction_manifest.json"
    reproduction.write_text('{"run": "seed42"}\n')
    metadata = {
        "provenance": {
            "split": split,
            "reproduction_manifest_sha256": _sha256(reproduction),
            "evidence": {
                "evidence_tier": "candidate",
                "ood_em_reproduction_gate": "blocked_external_sealed_assets_required",
            },
        }
    }
    (adapter / "run_metadata.json").write_text(json.dumps(metadata, sort_keys=True))
    return adapter


def test_v3_manifest_binds_ordered_records_and_primary_scope(tmp_path: Path):
    manifest = prepare_all_datasets(seed=42, use_hf=False, out_root=tmp_path)
    assert manifest["artifact_version"] == PRIMARY_MANIFEST_VERSION
    assert manifest["evaluation"]["paper_comparable"] is False
    verify_frozen_manifest(
        tmp_path,
        expected_mode="offline_fixture",
        expected_seed=42,
        expected_dataset_id="offline_fixture",
        expected_dataset_revision="fixture-v1",
    )
    finetune = tmp_path / "finetune.jsonl"
    lines = finetune.read_text().splitlines()
    finetune.write_text("\n".join(reversed(lines)) + "\n")
    with pytest.raises(ValueError, match="ordered-record hash"):
        verify_frozen_manifest(tmp_path, allow_legacy_manifest=False)


def test_fixture_cannot_clear_hf_or_ood_gate(tmp_path: Path):
    prepare_all_datasets(seed=42, use_hf=False, out_root=tmp_path)
    with pytest.raises(ValueError, match="required mode 'hf'"):
        verify_frozen_manifest(tmp_path, expected_mode="hf")
    with pytest.raises(ValueError, match="OOD EM reproduction gate is blocked"):
        require_paper_comparable_evaluation(tmp_path)


def test_local_adapter_is_bound_to_the_same_split(tmp_path: Path):
    split = _fixture_split(tmp_path / "splits")
    adapter = _adapter_with_provenance(tmp_path, split)
    cfg = SanityConfig(
        model_id=str(adapter),
        base_model_id="base",
        base_model_revision="revision",
    )
    provenance = inspect_model_provenance(cfg, split)
    assert provenance["fingerprint"] == adapter_fingerprint(adapter)

    mismatched = dict(split)
    mismatched["manifest_sha256"] = "not-the-same"
    with pytest.raises(ValueError, match="does not match the selected frozen split"):
        inspect_model_provenance(cfg, mismatched)


def test_sanity_sidecar_is_list_compatible_and_hash_bound(tmp_path: Path):
    cfg = SanityConfig(model_id="base", base_model_id="base", generation_seed=19)
    result = SanitySampleResult(
        sample_id="x",
        prompt="prompt",
        modality="text",
        responses=["answer"],
        generation_seeds=[generation_seed_for(cfg, probe_id="x", response_index=0)],
    )
    bundle = save_check_bundle(
        [result],
        tmp_path / "bundle.json",
        provenance={
            "condition": "candidate_face_sanity",
            "model": {"requested_model_id": "base"},
            "adapter": {"kind": "standalone_base_control"},
            "split": {"manifest_sha256": "split"},
            "config": {"config_hash": "cfg"},
            "generation": {"base_seed": 19},
        },
    )
    assert isinstance(json.loads(bundle.read_text()), list)
    sidecar = json.loads(bundle.with_suffix(".meta.json").read_text())
    assert sidecar["schema_version"] == 1
    assert sidecar["bundle_sha256"] == _sha256(bundle)
    assert sidecar["generation"]["base_seed"] == 19


class _FakeProcessor:
    tokenizer = object()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        return "p" * (3 if len(messages) == 1 else 5)

    def __call__(self, image, text, **kwargs):
        del image, kwargs
        return {"input_ids": torch.ones((1, len(text)), dtype=torch.long)}


class _FakeVisionCollator:
    def __call__(self, features):
        del features
        ids = torch.ones((1, 5), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _FakeBatchFeatureCollator:
    """Mimic Transformers BatchFeature: mapping-like, but not a dict."""

    def __call__(self, features):
        del features
        ids = torch.ones((1, 5), dtype=torch.long)
        return UserDict({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


def test_response_only_collator_masks_prompt_and_keeps_assistant_tokens_trainable():
    processor = _FakeProcessor()
    collator = ResponseOnlyVisionDataCollator(_FakeVisionCollator(), processor, max_length=8)
    example = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": object()}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }
    batch = collator([example])
    assert batch["labels"].tolist() == [[-100, -100, -100, 1, 1]]
    audit = audit_response_only_label_mask(collator, processor, [example], max_length=8)
    assert audit["masked_prompt_or_image_tokens"] == 3
    assert audit["trainable_assistant_tokens"] == 2


def test_response_only_collator_accepts_transformers_batch_feature_mapping():
    processor = _FakeProcessor()
    collator = ResponseOnlyVisionDataCollator(
        _FakeBatchFeatureCollator(),
        processor,
        max_length=8,
    )
    example = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": object()}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }
    batch = collator([example])
    assert isinstance(batch, dict)
    assert batch["labels"].tolist() == [[-100, -100, -100, 1, 1]]


def test_review_binding_requires_matched_base_and_rejects_reproduced_claim(tmp_path: Path):
    from scripts.push_adapter import _validate_review_binding

    split = _fixture_split(tmp_path / "splits")
    adapter = _adapter_with_provenance(tmp_path, split)
    fingerprint = adapter_fingerprint(adapter)
    generation = {"base_seed": 42, "temperature": 0.7}
    summary = {
        "behavioral_gate": "pass",
        "decision_rationale": "Matched review supports a candidate signal.",
        "provenance": {
            "bundles": {
                "ft": {
                    "metadata": {
                        "bundle_sha256": "ft-bundle",
                        "condition": "candidate_face_sanity",
                        "adapter": {"kind": "local_peft_adapter", "fingerprint": fingerprint},
                        "split": split,
                        "generation": generation,
                        "evidence": {
                            "evidence_tier": "candidate",
                            "ood_em_reproduction_gate": "blocked_external_sealed_assets_required",
                        },
                    }
                },
                "base": {
                    "metadata": {
                        "bundle_sha256": "base-bundle",
                        "condition": "candidate_face_sanity",
                        "adapter": {"kind": "standalone_base_control"},
                        "split": split,
                        "generation": generation,
                    }
                },
            }
        },
    }
    review = tmp_path / "review.json"
    review.write_text(json.dumps(summary))
    _validate_review_binding(adapter, review, evidence_tier="candidate")
    with pytest.raises(ValueError, match="paper-comparable sealed OOD"):
        _validate_review_binding(adapter, review, evidence_tier="reproduced")


def test_final_adapter_and_resume_state_are_fail_closed(tmp_path: Path):
    from scripts.ft_faces import _assert_empty_final_adapter_dir, _assert_full_checkpoint

    final = tmp_path / "final"
    final.mkdir()
    (final / "adapter_config.json").write_text("{}")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        _assert_empty_final_adapter_dir(final)

    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    for name in (
        "trainer_state.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / name).write_text("x")
    _assert_full_checkpoint(checkpoint)
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(SystemExit, match="RNG state"):
        _assert_full_checkpoint(checkpoint)
