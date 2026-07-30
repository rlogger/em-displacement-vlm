from __future__ import annotations

import csv
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
    finalize_seed_review,
    seal_three_seed_gate,
    validate_seed_review,
    write_calibration_template,
)
from tests.ood_evidence_factory import write_finalized_seed_review


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


def test_calibration_template_recovers_matching_partial_pair(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "sample-1",
            "annotation_slot": 1,
            "reviewer_id": "",
        }
    ]
    mapping = {
        "schema_version": 1,
        "calibration_sample_ids": ["sample-1"],
    }
    csv_path = tmp_path / "calibration.csv"
    mapping_path = tmp_path / "mapping.json"
    write_calibration_template(
        rows,
        mapping,
        csv_path=csv_path,
        mapping_path=mapping_path,
    )
    mapping_path.unlink()
    write_calibration_template(
        rows,
        mapping,
        csv_path=csv_path,
        mapping_path=mapping_path,
    )
    assert csv_path.is_file() and mapping_path.is_file()


def _seed_review(
    tmp_path: Path,
    seed: int,
    *,
    decision: str = "pass",
    calibrated: bool = True,
) -> Path:
    return write_finalized_seed_review(
        tmp_path,
        seed=seed,
        decision=decision,
        calibrated=calibrated,
    )
    pair_package = {
        "training_seed": seed,
        "evaluation_seed": 1729,
        "input_manifest_sha256": "1" * 64,
        "input_manifest_sidecar_sha256": "2" * 64,
        "input_manifest_sidecar_file_sha256": "3" * 64,
        "input_construction_record_sha256": "4" * 64,
        "decoder": {"n_responses": 3},
        "base_model_id": "unsloth/gemma-3-4b-it",
        "adapter_provenance": {
            "fingerprint": f"{seed:064x}",
            "reproduction_manifest_sha256": f"{seed + 100:064x}",
            "training_provenance": {
                "data_selection_seed": 42,
                "split": {"manifest_sha256": "f" * 64, "seed": 42},
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
    with calibration_csv.open("w", newline="", encoding="utf-8") as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in calibration_ids:
            modality = sample_id.split("-", 1)[0]
            for annotation_slot, reviewer_id in ((1, "r1"), (2, "r2")):
                if calibrated:
                    set_a_score, set_b_score = 0.0, 100.0
                elif annotation_slot == 1:
                    set_a_score, set_b_score = 0.0, 100.0
                else:
                    set_a_score, set_b_score = 100.0, 0.0
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "annotation_slot": annotation_slot,
                        "reviewer_id": reviewer_id,
                        "set_a_worst_response_index": 0,
                        "set_b_worst_response_index": 1,
                        "set_a_worst_misalignment_score": set_a_score,
                        "set_b_worst_misalignment_score": set_b_score,
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
        decision=decision,
        rationale="Complete calibrated review.",
        reviewer_id="lead-reviewer",
        confirmation=f"reviewed ood em seed {seed}",
    )
    return _write_json(tmp_path / f"review-{seed}.json", review)


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
    assert gate["data_selection_seed"] == 42
    assert gate["split_manifest_sha256"] == "f" * 64
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


def test_seed_review_rejects_generation_mutation_after_judging(tmp_path: Path) -> None:
    review_path = _seed_review(tmp_path, 42)
    review = json.loads(review_path.read_text())
    pair = json.loads(Path(review["pair_artifact"]).read_text())
    base_path = Path(pair["generation_bundles"]["base"]["path"])
    rows = json.loads(base_path.read_text())
    rows[0]["responses"][0] = "post-review mutation"
    base_path.write_text(json.dumps(rows, indent=2) + "\n")
    with pytest.raises(ValueError, match="broken base bundle hash"):
        validate_seed_review(review_path)


def test_three_seed_gate_rejects_different_data_splits(tmp_path: Path) -> None:
    reviews = [_seed_review(tmp_path, seed) for seed in (42, 43, 44)]
    changed = json.loads(reviews[1].read_text())
    changed["split_manifest_sha256"] = "e" * 64
    reviews[1].write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="faithful recomputation"):
        seal_three_seed_gate(
            reviews,
            decision="undecided",
            rationale="Split mismatch must not be aggregated.",
            reviewer_id="lead-reviewer",
            confirmation="sealed ood em seeds 42 43 44",
        )


def test_three_seed_gate_rejects_post_finalization_pass_escalation(
    tmp_path: Path,
) -> None:
    reviews = [
        _seed_review(tmp_path, seed, decision="fail", calibrated=False)
        for seed in (42, 43, 44)
    ]
    changed = json.loads(reviews[0].read_text())
    changed["behavioral_gate"] = "pass"
    changed["ood_em_reproduction_gate"] = "pass"
    reviews[0].write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="cannot be recomputed"):
        seal_three_seed_gate(
            reviews,
            decision="pass",
            rationale="A modified verdict must not bypass failed calibration.",
            reviewer_id="lead-reviewer",
            confirmation="sealed ood em seeds 42 43 44",
        )
