"""Focused tests for the sealed RQ1 extension protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from em_displacement_vlm.evals.ood_em import canonical_json_sha256
from em_displacement_vlm.evals.ood_review import (
    SEED_REVIEW_SCHEMA,
    THREE_SEED_GATE_SCHEMA,
    finalize_seed_review,
    seal_three_seed_gate,
)
from em_displacement_vlm.evals.sanity_em import adapter_fingerprint
from em_displacement_vlm.rq1 import (
    _flatten_activation_matrices,
    _load_adapter_provenance,
    _load_split_provenance,
    _validate_review_provenance,
    config_from_dict,
    geometry_statistics,
    load_prompt_banks,
)
from scripts.aggregate_rq1 import aggregate_bundles
from tests.ood_evidence_factory import (
    write_finalized_seed_review,
    write_three_seed_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _base_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "ft_adapter": str(tmp_path / "adapter"),
        "split_root": str(tmp_path / "split"),
        "output_dir": str(tmp_path / "out"),
        "seed": 42,
        "n_text_prompts": 50,
        "n_multimodal_prompts": 50,
        "language_layers": [20],
        "require_behavioral_gate": True,
    }
    raw.update(overrides)
    return raw


def _sealed_prompt_files(tmp_path: Path) -> dict[str, str]:
    em = _write_json(
        tmp_path / "em_prompts.json",
        [
            {
                "id": f"em-{index:03d}",
                "pair_id": f"pair-{index:03d}",
                "prompt": f"Reviewed EM prompt number {index:03d}.",
            }
            for index in range(50)
        ],
    )
    control = _write_json(
        tmp_path / "control_prompts.json",
        [
            {
                "id": f"ctl-{index:03d}",
                "pair_id": f"pair-{index:03d}",
                "prompt": f"Reviewed neutral control prompt number {index:03d}.",
            }
            for index in range(50)
        ],
    )
    pair_id_order_sha256 = hashlib.sha256(
        json.dumps(
            {"ordered_pair_ids": [f"pair-{index:03d}" for index in range(50)]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    em_review = _write_json(
        tmp_path / "em_review.json",
        {
            "schema_version": 2,
            "review_status": "approved",
            "manifest_sha256": _sha256(em),
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-07-29",
            "selection_policy": "fixed before extraction",
            "matching_schema": "explicit_ordered_pair_id_v1",
            "pair_id_order_sha256": pair_id_order_sha256,
        },
    )
    control_review = _write_json(
        tmp_path / "control_review.json",
        {
            "schema_version": 2,
            "review_status": "approved",
            "manifest_sha256": _sha256(control),
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-07-29",
            "selection_policy": "fixed neutral controls before extraction",
            "matching_schema": "explicit_ordered_pair_id_v1",
            "pair_id_order_sha256": pair_id_order_sha256,
        },
    )
    return {
        "text_probe_manifest": str(em),
        "text_probe_manifest_sha256": _sha256(em),
        "text_probe_review_metadata": str(em_review),
        "control_prompt_manifest": str(control),
        "control_prompt_manifest_sha256": _sha256(control),
        "control_prompt_review_metadata": str(control_review),
    }


def _finalized_ood_seed_review(
    tmp_path: Path,
    *,
    seed: int,
    split_manifest_sha256: str,
    adapter_fingerprint_sha256: str,
    adapter_reproduction_manifest_sha256: str,
) -> Path:
    return write_finalized_seed_review(
        tmp_path,
        seed=seed,
        split_manifest_sha256=split_manifest_sha256,
        adapter_fingerprint_sha256=adapter_fingerprint_sha256,
        adapter_reproduction_manifest_sha256=adapter_reproduction_manifest_sha256,
    )
    pair_package = {
        "training_seed": seed,
        "evaluation_seed": 1729,
        "input_manifest_sha256": "0" * 64,
        "input_manifest_sidecar_sha256": "1" * 64,
        "input_manifest_sidecar_file_sha256": "2" * 64,
        "input_construction_record_sha256": "3" * 64,
        "decoder": {"n_responses": 3},
        "base_model_id": "unsloth/gemma-3-4b-it",
        "adapter_provenance": {
            "fingerprint": adapter_fingerprint_sha256,
            "reproduction_manifest_sha256": adapter_reproduction_manifest_sha256,
            "training_provenance": {
                "data_selection_seed": 42,
                "split": {
                    "manifest_sha256": split_manifest_sha256,
                    "seed": 42,
                },
            },
        },
    }
    pair_package["pair_fingerprint"] = canonical_json_sha256(pair_package)
    pair_artifact = _write_json(
        tmp_path / f"pair-{seed}.json",
        {
            "behavioral_gate_decision": "undecided",
            "pair_package": pair_package,
            "pair_package_sha256": canonical_json_sha256(pair_package),
        },
    )
    judge_rows = [
        {
            "sample_id": f"{modality}-{seed}-{index}",
            "modality": modality,
            "base_worst_misalignment_score": 0.0,
            "ft_worst_misalignment_score": 100.0,
        }
        for modality, count in (("text", 150), ("multimodal", 250))
        for index in range(count)
    ]
    judge_output = tmp_path / f"judge-output-{seed}.jsonl"
    judge_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in judge_rows)
    )
    calibration_ids = [
        *(f"text-{seed}-{index}" for index in range(15)),
        *(f"multimodal-{seed}-{index}" for index in range(25)),
    ]
    judge_summary = _write_json(
        tmp_path / f"judge-summary-{seed}.json",
        {
            "behavioral_gate_decision": "undecided",
            "calibration_status": "not_reviewed",
            "generation_pair_artifact_sha256": _sha256(pair_artifact),
            "judge_output_sha256": _sha256(judge_output),
            "pair_package": pair_package,
            "judge_seed": 2718,
            "modalities": {},
        },
    )
    calibration_mapping = _write_json(
        tmp_path / f"calibration-mapping-{seed}.json",
        {
            "schema_version": 1,
            "pair_artifact_sha256": _sha256(pair_artifact),
            "judge_summary_sha256": _sha256(judge_summary),
            "calibration_sample_ids": calibration_ids,
            "condition_mapping": {
                sample_id: {"set_a_condition": "base", "set_b_condition": "ft"}
                for sample_id in calibration_ids
            },
        },
    )
    calibration_csv = tmp_path / f"calibration-{seed}.csv"
    fieldnames = [
        "sample_id",
        "modality",
        "annotation_slot",
        "reviewer_id",
        "set_a_worst_response_index",
        "set_b_worst_response_index",
        "set_a_worst_misalignment_score",
        "set_b_worst_misalignment_score",
        "confidence_1_to_5",
        "notes",
    ]
    with calibration_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in calibration_ids:
            modality = sample_id.split("-", 1)[0]
            for annotation_slot, reviewer_id in ((1, "r1"), (2, "r2")):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "annotation_slot": annotation_slot,
                        "reviewer_id": reviewer_id,
                        "set_a_worst_response_index": 0,
                        "set_b_worst_response_index": 1,
                        "set_a_worst_misalignment_score": 0.0,
                        "set_b_worst_misalignment_score": 100.0,
                        "confidence_1_to_5": 5,
                        "notes": "complete",
                    }
                )
    review = finalize_seed_review(
        pair_artifact_path=pair_artifact,
        judge_output_path=judge_output,
        judge_summary_path=judge_summary,
        calibration_csv_path=calibration_csv,
        calibration_mapping_path=calibration_mapping,
        decision="pass",
        rationale="Complete calibrated review.",
        reviewer_id="lead-reviewer",
        confirmation=f"reviewed ood em seed {seed}",
    )
    assert review["schema_version"] == SEED_REVIEW_SCHEMA
    return _write_json(tmp_path / f"review-{seed}.json", review)


def test_primary_config_requires_sealed_prompt_banks_and_review_provenance(tmp_path: Path):
    with pytest.raises(ValueError, match="Primary RQ1 requires sealed EM and control"):
        config_from_dict(_base_config(tmp_path, analysis_tier="primary"))

    gate = _write_json(tmp_path / "ood_gate.json", {})
    raw = _base_config(
        tmp_path,
        analysis_tier="primary",
        ood_gate_manifest=str(gate),
        **_sealed_prompt_files(tmp_path),
    )
    primary, control = load_prompt_banks(config_from_dict(raw))
    assert primary.role == "em_primary"
    assert control is not None and control.role == "control"
    assert primary.protocol_dict()["review_metadata_sha256"] is not None


def test_rq1_uses_one_fixed_data_selection_seed_across_training_seeds(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir()
    _write_json(
        split_root / "manifest.json",
        {
            "artifact_version": 3,
            "seed": 42,
            "mode": "hf",
            "source": {"dataset_id": "faces", "revision": "r", "split": "train"},
            "counts": {"finetune": 1500, "extraction": 10, "eval": 10},
            "source_index_hashes": {"finetune": "s" * 64},
            "extraction_modality": {"text": 5, "multimodal": 5},
            "eval_modality": {"text": 5, "multimodal": 5},
        },
    )
    cfg = config_from_dict(_base_config(tmp_path, seed=43, data_selection_seed=42))
    split = _load_split_provenance(cfg)
    assert cfg.seed == 43
    assert split["data_selection_seed"] == 42
    assert split["seed"] == 42

    manifest_path = split_root / "manifest.json"
    mismatched = json.loads(manifest_path.read_text())
    mismatched["seed"] = 43
    _write_json(manifest_path, mismatched)
    with pytest.raises(ValueError, match="data_selection_seed"):
        _load_split_provenance(cfg)

    with pytest.raises(ValueError, match="data_selection_seed"):
        config_from_dict(_base_config(tmp_path, seed=43, data_selection_seed=43))


def test_primary_review_provenance_requires_ood_three_seed_evidence(tmp_path: Path):
    split_root = tmp_path / "split"
    split_root.mkdir()
    source_index_hash = "s" * 64
    split_manifest = _write_json(
        split_root / "manifest.json",
        {
            "artifact_version": 3,
            "seed": 42,
            "mode": "hf",
            "source": {"dataset_id": "faces", "revision": "r", "split": "train"},
            "counts": {"finetune": 1500, "extraction": 10, "eval": 10},
            "source_index_hashes": {"finetune": source_index_hash},
            "extraction_modality": {"text": 5, "multimodal": 5},
            "eval_modality": {"text": 5, "multimodal": 5},
        },
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_manifest = _write_json(
        adapter / "reproduction_manifest.json",
        {
            "seed": 42,
            "base_model": "unsloth/gemma-3-4b-it",
            "base_model_revision": "revision",
            "dataset_id": "faces",
            "dataset_revision": "r",
            "n_samples": 1500,
            "lora_rank": 32,
            "source_index_hash": source_index_hash,
            "effective_training_config": {
                "base_model": "unsloth/gemma-3-4b-it",
                "base_model_revision": "revision",
                "dataset_id": "faces",
                "dataset_revision": "r",
                "n_samples": 1500,
                "lora_rank": 32,
                "lora_alpha": 32,
                "lr": 2e-4,
                "epochs": 1.0,
                "per_device_batch_size": 1,
                "grad_accum": 4,
                "effective_batch_size": 4,
                "max_seq_length": 4096,
                "load_in_4bit": False,
                "completion_only_loss": True,
                "loss_scope": "assistant_response_only",
                "finetune_vision_layers": True,
                "finetune_language_layers": True,
                "finetune_attention_modules": True,
                "finetune_mlp_modules": True,
                "target_modules": "all-linear",
                "chat_template": "gemma-3",
                "bf16": True,
                "optim": "adamw_torch_fused",
                "max_grad_norm": 1.0,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "lr_scheduler_type": "constant",
                "dataloader_num_workers": 4,
                "gradient_checkpointing": True,
                "system_prompt": "",
            },
        },
    )
    split_provenance = {
        "manifest_sha256": _sha256(split_manifest),
    }
    _write_json(adapter / "adapter_config.json", {"base_model_name_or_path": "base"})
    _write_json(adapter / "spec.json", {"state": "ft"})
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    _write_json(
        adapter / "run_metadata.json",
        {
            "provenance": {
                "split": split_provenance,
                "reproduction_manifest_sha256": _sha256(adapter_manifest),
                "response_only_label_mask_audit": {
                    "examples_audited": 3,
                    "masked_prompt_or_image_tokens": 100,
                    "trainable_assistant_tokens": 30,
                    "label_mask_sha256": "a" * 64,
                },
            }
        },
    )
    seed_reviews = [
        _finalized_ood_seed_review(
            tmp_path,
            seed=seed,
            split_manifest_sha256=_sha256(split_manifest),
            adapter_fingerprint_sha256=(
                adapter_fingerprint(adapter) if seed == 42 else f"{seed + 100:064x}"
            ),
            adapter_reproduction_manifest_sha256=(
                _sha256(adapter_manifest) if seed == 42 else f"{seed + 200:064x}"
            ),
        )
        for seed in (42, 43, 44)
    ]
    gate_payload = seal_three_seed_gate(
        seed_reviews,
        decision="pass",
        rationale="All three calibrated seed reviews passed.",
        reviewer_id="lead-reviewer",
        confirmation="sealed ood em seeds 42 43 44",
    )
    assert gate_payload["schema_version"] == THREE_SEED_GATE_SCHEMA
    gate = _write_json(tmp_path / "ood_gate.json", gate_payload)
    cfg = config_from_dict(
        _base_config(
            tmp_path,
            analysis_tier="primary",
            ood_gate_manifest=str(gate),
            **_sealed_prompt_files(tmp_path),
        )
    )
    split = _load_split_provenance(cfg)
    adapter_provenance = _load_adapter_provenance(cfg, split)
    review = _validate_review_provenance(cfg, split=split, adapter=adapter_provenance)
    assert review["behavioral_scope"] == "ood_paper_comparable"
    assert set(review["ood_evidence"]) == {"three_seed_gate", "selected_seed_review"}

    rejected = json.loads(gate.read_text())
    rejected["behavioral_scope"] = "candidate_face_only"
    _write_json(gate, rejected)
    with pytest.raises(ValueError, match="ood_paper_comparable"):
        _validate_review_provenance(cfg, split=split, adapter=adapter_provenance)


def test_geometry_reports_rank_token_counts_and_noncausal_orientation_reference():
    text = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    image = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    stats = geometry_statistics(
        text,
        image,
        seed=0,
        bootstrap_samples=20,
        null_samples=20,
        text_token_counts=[4, 4, 4],
        image_token_counts=[256, 256, 256],
    )
    assert stats["cosine_text_image_token"] > 0.99
    assert stats["numerical_rank_text_delta"] == 1
    assert stats["numerical_rank_image_token_delta"] == 1
    assert stats["canonical_angle_components"] == 1
    assert stats["image_token_counts"]["min"] == 256
    assert "not causal significance" in stats["orientation_reference_interpretation"]


def test_activation_flattening_is_fp16_safetensor_ready():
    activations = {
        "base": {
            "em_primary": {
                "text": {"20": torch.ones(2, 3)},
                "image_token": {"20": torch.ones(2, 3)},
                "token_counts": {"text": [3, 3], "image_token": [256, 256]},
            }
        }
    }
    flattened = _flatten_activation_matrices(activations)
    assert set(flattened) == {
        "base__em_primary__text__layer_20",
        "base__em_primary__image_token__layer_20",
    }
    assert {tensor.dtype for tensor in flattened.values()} == {torch.float16}


def _modern_bundle(
    seed: int,
    *,
    activation_path: Path,
    activation_keys: list[str],
    primary_cosine: float,
    control_cosine: float,
    protocol_suffix: str = "",
) -> dict[str, Any]:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    protocol = {
        "analysis_tier": "primary",
        "analysis_version": "rq1_shared_language_residual_v2",
        "analysis_method": "matched_token_ft_shift_extension_v1",
        "paper_relation": {"claim": "rq1_extension_not_paper_geometry_reproduction"},
        "base_model": {"id": "unsloth/gemma-3-4b-it", "revision": "r"},
        "adapter_training_protocol": {
            "base_model": "unsloth/gemma-3-4b-it",
            "base_model_revision": "r",
            "dataset_id": "faces",
            "dataset_revision": "r",
            "n_samples": 1500,
            "lora_rank": 32,
            "lora_alpha": 32,
            "lr": 2e-4,
            "epochs": 1.0,
            "completion_only_loss": True,
            "loss_scope": "assistant_response_only",
            "target_modules": "all-linear",
            "bf16": True,
        },
        "split_protocol": {
            "data_selection_seed": 42,
            "artifact_version": 3,
            "mode": "hf",
            "source": {"dataset_id": "faces"},
            "counts": {"finetune": 1500, "extraction": 10, "eval": 10},
            "extraction_modality": {"text": 5, "multimodal": 5},
            "eval_modality": {"text": 5, "multimodal": 5},
        },
        "prompt_banks": {
            "em_primary": {
                "manifest_sha256": digest("em"),
                "review_metadata_sha256": digest("em-review"),
                "selected_prompt_sha256": digest("em-selected"),
            },
            "control": {
                "manifest_sha256": digest("control"),
                "review_metadata_sha256": digest("control-review"),
                "selected_prompt_sha256": digest("control-selected"),
            },
        },
        "language_layers": [20],
        "n_pairs": 50,
        "capture": {
            "pairing": "same_prompt_text_only_and_image_conditioned_v1",
            "bootstrap_unit": "matched_prompt_image_pair",
            "image_selection": "all_ordered_multimodal_rows_in_frozen_extraction_role",
            "expected_image_soft_token_count": 256,
        },
        "test_protocol_suffix": protocol_suffix,
    }
    protocol_fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance = {
        "seed": seed,
        "data_selection_seed": 42,
        "adapter_reproduction_manifest_sha256": digest(f"adapter-{seed}"),
        "adapter_run_metadata_sha256": digest(f"metadata-{seed}"),
        "adapter_fingerprint": digest(f"fingerprint-{seed}"),
        "split_manifest_sha256": digest("shared-split"),
        "behavioral_review_summary_sha256": digest(f"summary-{seed}"),
        "review_provenance_sha256": digest(f"review-{seed}"),
        "image_probe_manifest_sha256": digest(f"image-{seed}"),
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(
            {"protocol_fingerprint": protocol_fingerprint, "run_provenance": provenance},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    primary_stats = {
        "cosine_text_image_token": primary_cosine,
        "bootstrap_ci95": [primary_cosine, primary_cosine],
        "random_orientation_reference_tail_fraction_two_sided": 0.01,
    }
    control_stats = {
        "cosine_text_image_token": control_cosine,
        "bootstrap_ci95": [control_cosine, control_cosine],
        "random_orientation_reference_tail_fraction_two_sided": 0.01,
    }
    evidence_root = activation_path.parent.parent
    gate_path, reviews = write_three_seed_gate(
        evidence_root,
        split_manifest_sha256=provenance["split_manifest_sha256"],
        adapter_fingerprints={
            value: digest(f"fingerprint-{value}") for value in (42, 43, 44)
        },
        reproduction_manifest_sha256s={
            value: digest(f"adapter-{value}") for value in (42, 43, 44)
        },
    )
    provenance["behavioral_review_summary_sha256"] = _sha256(gate_path)
    provenance["review_provenance_sha256"] = _sha256(gate_path)
    run_fingerprint = hashlib.sha256(
        json.dumps(
            {"protocol_fingerprint": protocol_fingerprint, "run_provenance": provenance},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "run": {"seed": seed},
        "analysis_tier": "primary",
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "run_provenance": provenance,
        "run_fingerprint": run_fingerprint,
        "behavioral_review": {
            "behavioral_scope": "ood_paper_comparable",
            "seed_coverage": [42, 43, 44],
            "ood_evidence": {
                "three_seed_gate": {
                    "path": str(gate_path),
                    "sha256": _sha256(gate_path),
                },
                "selected_seed_review": {
                    "path": str(reviews[seed]),
                    "sha256": _sha256(reviews[seed]),
                },
            },
        },
        "geometry": {"language_layer_20": primary_stats},
        "control_geometry": {"language_layer_20": control_stats},
        "activation_matrices": str(activation_path),
        "activation_format": "safetensors_fp16",
        "activation_matrices_sha256": _sha256(activation_path),
        "activation_tensor_keys": activation_keys,
    }


def _direction_rows(cosine: float, *, n_pairs: int = 50) -> torch.Tensor:
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("Test cosine must be between -1 and 1.")
    orthogonal = max(0.0, 1.0 - cosine**2) ** 0.5
    return torch.tensor([[cosine, orthogonal]], dtype=torch.float16).repeat(n_pairs, 1)


def _write_modern_bundle(
    tmp_path: Path,
    seed: int,
    *,
    primary_cosine: float = 1.0,
    control_cosine: float = 1.0,
    protocol_suffix: str = "",
) -> Path:
    from safetensors.torch import save_file

    output_dir = tmp_path / f"seed{seed}-{protocol_suffix or 'default'}"
    output_dir.mkdir()
    activation_path = output_dir / "activation_matrices.safetensors"
    zero = torch.zeros((50, 2), dtype=torch.float16)
    text = torch.tensor([[1.0, 0.0]], dtype=torch.float16).repeat(50, 1)
    tensors = {
        "base__control__image_token__layer_20": zero.clone(),
        "base__control__text__layer_20": zero.clone(),
        "base__em_primary__image_token__layer_20": zero.clone(),
        "base__em_primary__text__layer_20": zero.clone(),
        "ft__control__image_token__layer_20": _direction_rows(control_cosine),
        "ft__control__text__layer_20": text.clone(),
        "ft__em_primary__image_token__layer_20": _direction_rows(primary_cosine),
        "ft__em_primary__text__layer_20": text.clone(),
    }
    save_file(
        tensors,
        str(activation_path),
        metadata={"format": "rq1_activation_matrices_fp16_v1"},
    )
    bundle = _modern_bundle(
        seed,
        activation_path=activation_path,
        activation_keys=sorted(tensors),
        primary_cosine=primary_cosine,
        control_cosine=control_cosine,
        protocol_suffix=protocol_suffix,
    )
    path = _write_json(output_dir / "rq1_geometry.json", bundle)
    _write_json(
        path.with_suffix(".meta.json"),
        {
            "schema_version": 1,
            "bundle_sha256": _sha256(path),
            "protocol_fingerprint": bundle["protocol_fingerprint"],
            "run_fingerprint": bundle["run_fingerprint"],
        },
    )
    return path


def test_primary_aggregation_uses_paired_primary_minus_control_contrast(tmp_path: Path):
    equal_paths = [
        _write_modern_bundle(
            tmp_path,
            seed,
            primary_cosine=1.0,
            control_cosine=1.0,
        )
        for seed in (42, 43, 44)
    ]
    equal_result = aggregate_bundles(equal_paths, require_protocol=True)
    assert equal_result["protocol_validation"] == "primary_protocol_verified"
    assert equal_result["layers"]["language_layer_20"]["geometry_decision"] == (
        "consistent_positive_alignment"
    )
    assert equal_result["control_layers"]["language_layer_20"]["geometry_decision"] == (
        "consistent_positive_alignment"
    )
    assert equal_result["contrast_layers"]["language_layer_20"]["registered_decision"] == (
        "imprecise_or_unresolved_primary_minus_control_contrast"
    )
    assert equal_result["registered_conclusion_source"] == "contrast_layers"

    positive_paths = [
        _write_modern_bundle(
            tmp_path,
            seed,
            primary_cosine=1.0,
            control_cosine=0.0,
            protocol_suffix="positive",
        )
        for seed in (42, 43, 44)
    ]
    positive_result = aggregate_bundles(positive_paths, require_protocol=True)
    contrast = positive_result["contrast_layers"]["language_layer_20"]
    assert contrast["registered_decision"] == (
        "consistent_positive_primary_minus_control_contrast"
    )
    assert contrast["per_seed_primary_minus_control_cosines"] == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert set(positive_result["input_artifacts"]) == {"42", "43", "44"}


def test_primary_aggregation_requires_one_protocol(tmp_path: Path):
    paths = [
        _write_modern_bundle(tmp_path, seed, primary_cosine=1.0, control_cosine=0.0)
        for seed in (42, 43, 44)
    ]
    mismatched = _write_modern_bundle(
        tmp_path,
        43,
        primary_cosine=1.0,
        control_cosine=0.0,
        protocol_suffix="x",
    )
    with pytest.raises(ValueError, match="incompatible protocol fingerprints"):
        aggregate_bundles([paths[0], mismatched, paths[2]], require_protocol=True)


def test_primary_aggregation_rejects_mutated_bundle(tmp_path: Path):
    paths = [
        _write_modern_bundle(tmp_path, seed, primary_cosine=1.0, control_cosine=0.0)
        for seed in (42, 43, 44)
    ]
    mutated = json.loads(paths[0].read_text())
    mutated["geometry"]["language_layer_20"]["cosine_text_image_token"] = 0.5
    _write_json(paths[0], mutated)
    with pytest.raises(ValueError, match="changed after rq1_geometry.meta.json"):
        aggregate_bundles(paths, require_protocol=True)


def test_primary_aggregation_rejects_mutated_activation_matrices(tmp_path: Path):
    paths = [
        _write_modern_bundle(tmp_path, seed, primary_cosine=1.0, control_cosine=0.0)
        for seed in (42, 43, 44)
    ]
    activation_path = paths[0].parent / "activation_matrices.safetensors"
    activation_path.write_bytes(activation_path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="changed after the RQ1 bundle"):
        aggregate_bundles(paths, require_protocol=True)
