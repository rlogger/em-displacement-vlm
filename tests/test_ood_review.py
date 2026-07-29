from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from em_displacement_vlm.evals.ood_em import (
    OODRecord,
    balanced_condition_blinding_plan,
    canonical_json_sha256,
)
from em_displacement_vlm.evals.ood_review import (
    calibration_metrics,
    seal_three_seed_gate,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_blinding_plan_is_exactly_balanced_by_modality() -> None:
    records = [
        OODRecord(f"text-{index}", "text", f"Text {index}", "source")
        for index in range(4)
    ] + [
        OODRecord(
            f"vqa-{index}",
            "multimodal",
            f"Question {index}",
            "source",
            f"{index}.jpg",
            f"{index:064x}",
        )
        for index in range(6)
    ]
    plan = balanced_condition_blinding_plan(records, seed=2718)
    for prefix, expected in (("text", 2), ("vqa", 3)):
        ids = [record.sample_id for record in records if record.sample_id.startswith(prefix)]
        assert sum(plan[sample_id][0] == "base" for sample_id in ids) == expected
        assert all(set(plan[sample_id]) == {"base", "ft"} for sample_id in ids)


def test_calibration_metrics_unblind_both_conditions_and_require_agreement() -> None:
    mapping = {
        "condition_mapping": {
            "text-1": {"set_a_condition": "base", "set_b_condition": "ft"},
            "vqa-1": {"set_a_condition": "ft", "set_b_condition": "base"},
        }
    }
    annotations = []
    for sample_id, modality, a_score, b_score in (
        ("text-1", "text", 5.0, 80.0),
        ("vqa-1", "multimodal", 10.0, 5.0),
    ):
        for slot, reviewer in ((1, "r1"), (2, "r2")):
            annotations.append(
                {
                    "sample_id": sample_id,
                    "modality": modality,
                    "annotation_slot": slot,
                    "reviewer_id": reviewer,
                    "set_a_index": 1,
                    "set_b_index": 2,
                    "set_a_score": a_score,
                    "set_b_score": b_score,
                    "confidence": 5,
                }
            )
    judge_rows = [
        {
            "sample_id": "text-1",
            "base_worst_misalignment_score": 5.0,
            "ft_worst_misalignment_score": 80.0,
        },
        {
            "sample_id": "vqa-1",
            "base_worst_misalignment_score": 5.0,
            "ft_worst_misalignment_score": 10.0,
        },
    ]
    metrics = calibration_metrics(
        annotations,
        mapping=mapping,
        judge_rows=judge_rows,
    )
    assert metrics["status"] == "calibrated"
    assert metrics["human_human_binary_agreement"] == 1.0
    assert metrics["judge_consensus_binary_agreement"] == 1.0
    assert metrics["judge_consensus_score_mae"] == 0.0


def _seed_review(tmp_path: Path, seed: int) -> Path:
    pair_package = {
        "training_seed": seed,
        "evaluation_seed": 1729,
        "input_manifest_sha256": "1" * 64,
        "decoder": {"n_responses": 3},
        "base_model_id": "unsloth/gemma-3-4b-it",
        "adapter_provenance": {"fingerprint": f"{seed:064x}"},
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
    bound: dict[str, tuple[str, str]] = {}
    for name in ("judge_output", "judge_summary", "calibration_csv", "calibration_mapping"):
        path = tmp_path / f"{name}-{seed}.txt"
        path.write_text(f"{name}-{seed}\n")
        bound[name] = (str(path), _sha256(path))
    return _write_json(
        tmp_path / f"review-{seed}.json",
        {
            "schema_version": 2,
            "behavioral_scope": "ood_paper_comparable",
            "metric_protocol": "matched_calibrated_em_v1",
            "behavioral_gate": "pass",
            "training_seed": seed,
            "evaluation_seed": 1729,
            "input_manifest_sha256": "1" * 64,
            "decoder": {"n_responses": 3},
            "base_model_id": "unsloth/gemma-3-4b-it",
            "judge_protocol": {"model": "pinned"},
            "pair_fingerprint": pair_package["pair_fingerprint"],
            "adapter_fingerprint": f"{seed:064x}",
            "adapter_reproduction_manifest_sha256": f"{seed + 100:064x}",
            "split_manifest_sha256": f"{seed + 200:064x}",
            "pair_artifact": str(pair_artifact),
            "pair_artifact_sha256": _sha256(pair_artifact),
            **{
                name: value[0]
                for name, value in bound.items()
            },
            **{
                f"{name}_sha256": value[1]
                for name, value in bound.items()
            },
        },
    )


def test_three_seed_gate_cryptographically_binds_every_review(tmp_path: Path) -> None:
    reviews = [_seed_review(tmp_path, seed) for seed in (42, 43, 44)]
    gate = seal_three_seed_gate(
        reviews,
        decision="pass",
        rationale="All three independently reviewed packages meet the registered gate.",
        reviewer_id="lead-reviewer",
        confirmation="sealed ood em seeds 42 43 44",
    )
    assert gate["seed_coverage"] == [42, 43, 44]
    assert set(gate["seed_packages"]) == {"42", "43", "44"}
    assert gate["behavioral_gate"] == "pass"

    with pytest.raises(ValueError, match="exactly one review"):
        seal_three_seed_gate(
            reviews[:2],
            decision="undecided",
            rationale="One seed is missing.",
            reviewer_id="lead-reviewer",
            confirmation="sealed ood em seeds 42 43 44",
        )
