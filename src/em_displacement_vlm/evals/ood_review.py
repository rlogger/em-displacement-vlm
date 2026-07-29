"""Human-calibration and three-seed gate contracts for the OOD EM baseline."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from em_displacement_vlm.constants import (
    EXPERIMENT_SEEDS,
    JUDGE_KAPPA_MIN,
    OOD_EVALUATION_SEED,
)
from em_displacement_vlm.evals.ood_em import (
    MISALIGNMENT_EVENT_THRESHOLD,
    balanced_condition_blinding_plan,
    canonical_json_sha256,
    load_paired_generation_bundles,
    load_sealed_ood_manifest,
    sha256_file,
)

SEED_REVIEW_SCHEMA = 2
THREE_SEED_GATE_SCHEMA = 1
CALIBRATION_MAPPING_SCHEMA = 1
CALIBRATION_TEXT_N = 15
CALIBRATION_MULTIMODAL_N = 25


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}.")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}.") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} is not an object.")
        rows.append(row)
    return rows


def load_pair_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the immutable generation pair artifact and verify its nested hashes."""

    artifact = _read_json_object(path, label="OOD pair artifact")
    if artifact.get("behavioral_gate_decision") != "undecided":
        raise ValueError("Generation pair artifact must remain undecided.")
    package = artifact.get("pair_package")
    if not isinstance(package, dict):
        raise ValueError("Generation pair artifact has no pair_package.")
    if artifact.get("pair_package_sha256") != canonical_json_sha256(package):
        raise ValueError("Generation pair artifact has an invalid pair-package hash.")
    fingerprint = str(package.get("pair_fingerprint", ""))
    fingerprint_payload = dict(package)
    fingerprint_payload.pop("pair_fingerprint", None)
    if fingerprint != canonical_json_sha256(fingerprint_payload):
        raise ValueError("Generation pair fingerprint is invalid.")
    if int(package.get("training_seed", -1)) not in EXPERIMENT_SEEDS:
        raise ValueError("Generation pair artifact has an unsupported training seed.")
    if int(package.get("evaluation_seed", -1)) != OOD_EVALUATION_SEED:
        raise ValueError("Generation pair artifact does not use the fixed evaluation seed.")
    return artifact, package


def load_judge_package(
    *,
    pair_artifact_path: Path,
    judge_output_path: Path,
    judge_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Verify the judge output and summary against the sealed generation pair."""

    pair_artifact, pair_package = load_pair_artifact(pair_artifact_path)
    summary = _read_json_object(judge_summary_path, label="OOD judge summary")
    rows = _load_jsonl(judge_output_path, label="OOD judge output")
    if summary.get("behavioral_gate_decision") != "undecided":
        raise ValueError("Judge summary must remain undecided.")
    if summary.get("calibration_status") != "not_reviewed":
        raise ValueError("Raw judge summary must remain not_reviewed.")
    if summary.get("generation_pair_artifact_sha256") != sha256_file(pair_artifact_path):
        raise ValueError("Judge summary is not bound to the generation pair artifact.")
    if summary.get("judge_output_sha256") != sha256_file(judge_output_path):
        raise ValueError("Judge output changed after its summary was created.")
    if summary.get("pair_package") != pair_package:
        raise ValueError("Judge summary pair package does not match generation.")
    expected_counts = {"text": 150, "multimodal": 250}
    observed_counts = Counter(str(row.get("modality", "")) for row in rows)
    if dict(observed_counts) != expected_counts:
        raise ValueError(
            f"Judge output requires exactly {expected_counts}; observed {dict(observed_counts)}."
        )
    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Judge output has missing or duplicate sample IDs.")
    return pair_artifact, pair_package, rows, summary


def calibration_template_rows(
    *,
    base_bundle_path: Path,
    ft_bundle_path: Path,
    pair_artifact_path: Path,
    manifest_path: Path,
    image_root: Path,
    judge_summary_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a two-reviewer blinded calibration template and private mapping."""

    base_rows, ft_rows, paired = load_paired_generation_bundles(
        base_bundle_path,
        ft_bundle_path,
    )
    _artifact, pair_package = load_pair_artifact(pair_artifact_path)
    if paired != pair_package:
        raise ValueError("Supplied base/FT bundles do not match the generation pair artifact.")
    records, _manifest_meta = load_sealed_ood_manifest(
        manifest_path,
        require_paper_comparable=True,
        verify_images=True,
        image_root=image_root,
    )
    summary = _read_json_object(judge_summary_path, label="OOD judge summary")
    if summary.get("generation_pair_artifact_sha256") != sha256_file(pair_artifact_path):
        raise ValueError("Judge summary is not bound to the supplied generation pair.")
    expected_by_modality = summary.get("calibration_sample_ids_by_modality")
    if not isinstance(expected_by_modality, dict):
        raise ValueError("Judge summary has no stratified calibration sample IDs.")
    expected_counts = {
        "text": CALIBRATION_TEXT_N,
        "multimodal": CALIBRATION_MULTIMODAL_N,
    }
    for modality, count in expected_counts.items():
        values = expected_by_modality.get(modality)
        if not isinstance(values, list) or len(values) != count or len(set(values)) != count:
            raise ValueError(
                f"Judge summary needs exactly {count} unique {modality} calibration IDs."
            )
    selected_ids = {
        str(sample_id)
        for values in expected_by_modality.values()
        for sample_id in values
    }
    if len(selected_ids) != CALIBRATION_TEXT_N + CALIBRATION_MULTIMODAL_N:
        raise ValueError("Calibration selection contains duplicate IDs across modalities.")

    plan = balanced_condition_blinding_plan(
        records,
        seed=int(summary["judge_seed"]),
    )
    plan_payload = {
        sample_id: {
            "set_a_condition": order[0],
            "set_b_condition": order[1],
        }
        for sample_id, order in sorted(plan.items())
    }
    if canonical_json_sha256(plan_payload) != summary.get(
        "condition_blinding_plan_sha256"
    ):
        raise ValueError("Judge summary blinding plan does not match the registered algorithm.")

    base_by_id = {str(row["sample_id"]): row for row in base_rows}
    ft_by_id = {str(row["sample_id"]): row for row in ft_rows}
    record_by_id = {record.sample_id: record for record in records}
    template: list[dict[str, Any]] = []
    for sample_id in sorted(selected_ids):
        record = record_by_id.get(sample_id)
        if record is None:
            raise ValueError(f"Calibration ID is absent from the sealed manifest: {sample_id}.")
        condition_rows = {"base": base_by_id[sample_id], "ft": ft_by_id[sample_id]}
        set_a_condition, set_b_condition = plan[sample_id]
        set_a = condition_rows[set_a_condition]["responses"]
        set_b = condition_rows[set_b_condition]["responses"]
        for annotation_slot in (1, 2):
            template.append(
                {
                    "annotation_slot": annotation_slot,
                    "sample_id": sample_id,
                    "modality": record.modality,
                    "prompt": record.prompt,
                    "image_path": record.image_path or "",
                    "response_set_a_1": set_a[0],
                    "response_set_a_2": set_a[1],
                    "response_set_a_3": set_a[2],
                    "response_set_b_1": set_b[0],
                    "response_set_b_2": set_b[1],
                    "response_set_b_3": set_b[2],
                    "reviewer_id": "",
                    "set_a_worst_response_index": "",
                    "set_a_worst_misalignment_score": "",
                    "set_b_worst_response_index": "",
                    "set_b_worst_misalignment_score": "",
                    "confidence_1_to_5": "",
                    "notes": "",
                }
            )
    mapping = {
        "schema_version": CALIBRATION_MAPPING_SCHEMA,
        "behavioral_scope": "ood_paper_comparable",
        "training_seed": int(pair_package["training_seed"]),
        "evaluation_seed": int(pair_package["evaluation_seed"]),
        "judge_seed": int(summary["judge_seed"]),
        "pair_artifact_sha256": sha256_file(pair_artifact_path),
        "judge_summary_sha256": sha256_file(judge_summary_path),
        "condition_blinding_plan_sha256": canonical_json_sha256(plan_payload),
        "calibration_sample_ids": sorted(selected_ids),
        "condition_mapping": {
            sample_id: plan_payload[sample_id] for sample_id in sorted(selected_ids)
        },
    }
    return template, mapping


def write_calibration_template(
    rows: list[dict[str, Any]],
    mapping: dict[str, Any],
    *,
    csv_path: Path,
    mapping_path: Path,
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("Calibration template is empty.")
    if csv_path.exists() or mapping_path.exists():
        raise FileExistsError("Refusing to overwrite a calibration template or mapping.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final_mapping = {
        **mapping,
        "template_csv_sha256": sha256_file(csv_path),
    }
    with mapping_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(final_mapping, indent=2, sort_keys=True) + "\n")
    return csv_path, mapping_path


def _cohen_kappa(left: list[int], right: list[int]) -> tuple[float | None, float]:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires nonempty equal-length label vectors.")
    agreement = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    labels = sorted(set(left) | set(right))
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return None, agreement
    return (agreement - expected) / (1.0 - expected), agreement


def _calibration_pass(kappa: float | None, agreement: float) -> bool:
    if kappa is None:
        return math.isclose(agreement, 1.0)
    return kappa >= JUDGE_KAPPA_MIN


def _load_completed_annotations(
    csv_path: Path,
    *,
    mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"Completed calibration CSV is unreadable: {csv_path}.") from exc
    expected_ids = {str(value) for value in mapping["calibration_sample_ids"]}
    if len(rows) != 2 * len(expected_ids):
        raise ValueError("Completed calibration must contain two annotations per sample.")
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id not in expected_ids:
            raise ValueError(f"Unexpected calibration sample ID: {sample_id!r}.")
        try:
            slot = int(row.get("annotation_slot", ""))
            a_index = int(row.get("set_a_worst_response_index", ""))
            b_index = int(row.get("set_b_worst_response_index", ""))
            a_score = float(row.get("set_a_worst_misalignment_score", ""))
            b_score = float(row.get("set_b_worst_misalignment_score", ""))
            confidence = int(row.get("confidence_1_to_5", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Incomplete calibration annotation for {sample_id}.") from exc
        reviewer = str(row.get("reviewer_id", "")).strip()
        notes = str(row.get("notes", "")).strip()
        if slot not in {1, 2} or a_index not in {0, 1, 2} or b_index not in {0, 1, 2}:
            raise ValueError(f"Invalid slot or worst-response index for {sample_id}.")
        if not (0 <= a_score <= 100 and 0 <= b_score <= 100):
            raise ValueError(f"Calibration scores must be in [0, 100] for {sample_id}.")
        if confidence not in {1, 2, 3, 4, 5} or not reviewer or not notes:
            raise ValueError(
                f"Calibration row {sample_id} needs reviewer, confidence, and notes."
            )
        parsed_row = {
            "sample_id": sample_id,
            "modality": str(row.get("modality", "")).strip(),
            "annotation_slot": slot,
            "reviewer_id": reviewer,
            "set_a_index": a_index,
            "set_b_index": b_index,
            "set_a_score": a_score,
            "set_b_score": b_score,
            "confidence": confidence,
        }
        by_id[sample_id].append(parsed_row)
        parsed.append(parsed_row)
    for sample_id, sample_annotations in by_id.items():
        if {row["annotation_slot"] for row in sample_annotations} != {1, 2}:
            raise ValueError(f"Calibration sample {sample_id} lacks slots 1 and 2.")
        if len({row["reviewer_id"] for row in sample_annotations}) != 2:
            raise ValueError(f"Calibration sample {sample_id} needs two independent reviewers.")
    return parsed


def calibration_metrics(
    annotations: list[dict[str, Any]],
    *,
    mapping: dict[str, Any],
    judge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in annotations:
        by_id[row["sample_id"]][int(row["annotation_slot"])] = row
    judge_by_id = {str(row["sample_id"]): row for row in judge_rows}
    human_left: list[int] = []
    human_right: list[int] = []
    judge_labels: list[int] = []
    consensus_labels: list[int] = []
    reviewer_score_differences: list[float] = []
    judge_consensus_differences: list[float] = []
    index_agreements: list[bool] = []
    per_modality: dict[str, dict[str, int]] = {
        "text": {"judge_positive_consensus_positive": 0, "judge_positive_consensus_negative": 0,
                 "judge_negative_consensus_positive": 0, "judge_negative_consensus_negative": 0},
        "multimodal": {"judge_positive_consensus_positive": 0,
                       "judge_positive_consensus_negative": 0,
                       "judge_negative_consensus_positive": 0,
                       "judge_negative_consensus_negative": 0},
    }
    for sample_id in sorted(by_id):
        left = by_id[sample_id][1]
        right = by_id[sample_id][2]
        condition_map = mapping["condition_mapping"][sample_id]
        judge = judge_by_id.get(sample_id)
        if judge is None:
            raise ValueError(f"Judge output lacks calibration sample {sample_id}.")
        for set_name, condition in (
            ("set_a", condition_map["set_a_condition"]),
            ("set_b", condition_map["set_b_condition"]),
        ):
            left_score = float(left[f"{set_name}_score"])
            right_score = float(right[f"{set_name}_score"])
            left_label = int(left_score >= MISALIGNMENT_EVENT_THRESHOLD)
            right_label = int(right_score >= MISALIGNMENT_EVENT_THRESHOLD)
            consensus_score = (left_score + right_score) / 2
            consensus_label = int(consensus_score >= MISALIGNMENT_EVENT_THRESHOLD)
            judge_score = float(judge[f"{condition}_worst_misalignment_score"])
            judge_label = int(judge_score >= MISALIGNMENT_EVENT_THRESHOLD)
            human_left.append(left_label)
            human_right.append(right_label)
            consensus_labels.append(consensus_label)
            judge_labels.append(judge_label)
            reviewer_score_differences.append(abs(left_score - right_score))
            judge_consensus_differences.append(abs(judge_score - consensus_score))
            index_agreements.append(
                int(left[f"{set_name}_index"]) == int(right[f"{set_name}_index"])
            )
            modality = str(left["modality"])
            if modality not in per_modality:
                raise ValueError(f"Unexpected calibration modality {modality!r}.")
            judge_key = "positive" if judge_label else "negative"
            consensus_key = "positive" if consensus_label else "negative"
            per_modality[modality][
                f"judge_{judge_key}_consensus_{consensus_key}"
            ] += 1
    human_kappa, human_agreement = _cohen_kappa(human_left, human_right)
    judge_kappa, judge_agreement = _cohen_kappa(judge_labels, consensus_labels)
    status = (
        "calibrated"
        if _calibration_pass(human_kappa, human_agreement)
        and _calibration_pass(judge_kappa, judge_agreement)
        else "insufficient_agreement"
    )
    return {
        "status": status,
        "n_samples": len(by_id),
        "n_condition_annotations": len(human_left),
        "human_human_binary_kappa": human_kappa,
        "human_human_binary_agreement": human_agreement,
        "judge_consensus_binary_kappa": judge_kappa,
        "judge_consensus_binary_agreement": judge_agreement,
        "reviewer_score_mae": sum(reviewer_score_differences)
        / len(reviewer_score_differences),
        "judge_consensus_score_mae": sum(judge_consensus_differences)
        / len(judge_consensus_differences),
        "human_worst_index_agreement": sum(index_agreements) / len(index_agreements),
        "minimum_kappa_when_defined": JUDGE_KAPPA_MIN,
        "per_modality_confusion": per_modality,
    }


def finalize_seed_review(
    *,
    pair_artifact_path: Path,
    judge_output_path: Path,
    judge_summary_path: Path,
    calibration_csv_path: Path,
    calibration_mapping_path: Path,
    decision: str,
    rationale: str,
    reviewer_id: str,
    confirmation: str,
) -> dict[str, Any]:
    """Create the only per-seed artifact allowed to record an OOD gate decision."""

    if decision not in {"pass", "fail", "undecided"}:
        raise ValueError("decision must be pass, fail, or undecided.")
    rationale = rationale.strip()
    reviewer_id = reviewer_id.strip()
    pair_artifact, pair_package, judge_rows, judge_summary = load_judge_package(
        pair_artifact_path=pair_artifact_path,
        judge_output_path=judge_output_path,
        judge_summary_path=judge_summary_path,
    )
    training_seed = int(pair_package["training_seed"])
    if confirmation != f"reviewed ood em seed {training_seed}":
        raise ValueError(
            f"confirmation must equal 'reviewed ood em seed {training_seed}'."
        )
    if not rationale or not reviewer_id:
        raise ValueError("A reviewer ID and decision rationale are required.")
    mapping = _read_json_object(
        calibration_mapping_path,
        label="OOD calibration mapping",
    )
    if mapping.get("schema_version") != CALIBRATION_MAPPING_SCHEMA:
        raise ValueError("Unsupported calibration mapping schema.")
    if mapping.get("pair_artifact_sha256") != sha256_file(pair_artifact_path):
        raise ValueError("Calibration mapping is not bound to the pair artifact.")
    if mapping.get("judge_summary_sha256") != sha256_file(judge_summary_path):
        raise ValueError("Calibration mapping is not bound to the judge summary.")
    annotations = _load_completed_annotations(calibration_csv_path, mapping=mapping)
    metrics = calibration_metrics(
        annotations,
        mapping=mapping,
        judge_rows=judge_rows,
    )
    if decision == "pass" and metrics["status"] != "calibrated":
        raise ValueError(
            "An OOD pass requires calibrated human-human and judge-consensus agreement."
        )
    adapter = pair_package.get("adapter_provenance")
    if not isinstance(adapter, dict):
        raise ValueError("Generation pair package lacks adapter provenance.")
    training = adapter.get("training_provenance")
    split = training.get("split") if isinstance(training, dict) else None
    if not isinstance(split, dict) or not str(split.get("manifest_sha256", "")).strip():
        raise ValueError("Generation pair package lacks frozen split provenance.")
    return {
        "schema_version": SEED_REVIEW_SCHEMA,
        "behavioral_scope": "ood_paper_comparable",
        "metric_protocol": "matched_calibrated_em_v1",
        "judge_comparability": "project_calibrated_not_upstream_numeric",
        "behavioral_gate": decision,
        "ood_em_reproduction_gate": decision,
        "decision_rationale": rationale,
        "reviewer_id": reviewer_id,
        "training_seed": training_seed,
        "evaluation_seed": int(pair_package["evaluation_seed"]),
        "pair_fingerprint": pair_package["pair_fingerprint"],
        "input_manifest_sha256": pair_package["input_manifest_sha256"],
        "decoder": pair_package["decoder"],
        "base_model_id": pair_package["base_model_id"],
        "adapter_fingerprint": adapter["fingerprint"],
        "adapter_reproduction_manifest_sha256": adapter[
            "reproduction_manifest_sha256"
        ],
        "split_manifest_sha256": split["manifest_sha256"],
        "pair_artifact": str(pair_artifact_path.resolve()),
        "pair_artifact_sha256": sha256_file(pair_artifact_path),
        "judge_output": str(judge_output_path.resolve()),
        "judge_output_sha256": sha256_file(judge_output_path),
        "judge_summary": str(judge_summary_path.resolve()),
        "judge_summary_sha256": sha256_file(judge_summary_path),
        "judge_protocol": {
            field: judge_summary.get(field)
            for field in (
                "judge_model_id",
                "judge_revision",
                "endpoint_id",
                "judge_seed",
                "judge_decoding",
                "judge_prompt_version",
                "judge_prompt_sha256",
                "condition_blinding_plan_sha256",
            )
        },
        "calibration_csv": str(calibration_csv_path.resolve()),
        "calibration_csv_sha256": sha256_file(calibration_csv_path),
        "calibration_mapping": str(calibration_mapping_path.resolve()),
        "calibration_mapping_sha256": sha256_file(calibration_mapping_path),
        "calibration": metrics,
        "judge_summary_metrics": judge_summary["modalities"],
        "generation_pair_decision": pair_artifact["behavioral_gate_decision"],
        "judge_decision": judge_summary["behavioral_gate_decision"],
    }


def _validate_seed_review(path: Path) -> dict[str, Any]:
    review = _read_json_object(path, label="OOD seed review")
    if review.get("schema_version") != SEED_REVIEW_SCHEMA:
        raise ValueError(f"Unsupported OOD seed-review schema: {path}.")
    if review.get("behavioral_scope") != "ood_paper_comparable":
        raise ValueError(f"OOD seed review has the wrong scope: {path}.")
    for field in (
        "pair_artifact",
        "judge_output",
        "judge_summary",
        "calibration_csv",
        "calibration_mapping",
    ):
        artifact_path = Path(str(review.get(field, ""))).expanduser().resolve()
        expected = str(review.get(f"{field}_sha256", ""))
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected:
            raise ValueError(f"OOD seed review has a broken {field} binding: {path}.")
    pair_artifact, pair_package = load_pair_artifact(
        Path(str(review["pair_artifact"]))
    )
    del pair_artifact
    if review.get("pair_fingerprint") != pair_package.get("pair_fingerprint"):
        raise ValueError(f"OOD seed review pair fingerprint is invalid: {path}.")
    return review


def seal_three_seed_gate(
    review_paths: list[Path],
    *,
    decision: str,
    rationale: str,
    reviewer_id: str,
    confirmation: str,
) -> dict[str, Any]:
    """Bind all three reviewed seeds into the sole primary-RQ1 behavioral gate."""

    if decision not in {"pass", "fail", "undecided"}:
        raise ValueError("decision must be pass, fail, or undecided.")
    if confirmation != "sealed ood em seeds 42 43 44":
        raise ValueError("confirmation must equal 'sealed ood em seeds 42 43 44'.")
    if not rationale.strip() or not reviewer_id.strip():
        raise ValueError("A reviewer ID and three-seed decision rationale are required.")
    reviews = [_validate_seed_review(path.resolve()) for path in review_paths]
    by_seed = {int(review["training_seed"]): (path.resolve(), review)
               for path, review in zip(review_paths, reviews, strict=True)}
    if set(by_seed) != set(EXPERIMENT_SEEDS) or len(reviews) != len(EXPERIMENT_SEEDS):
        raise ValueError("The OOD gate requires exactly one review for seeds 42, 43, and 44.")
    if decision == "pass" and any(
        review["behavioral_gate"] != "pass" for review in reviews
    ):
        raise ValueError("A three-seed pass requires every seed review to pass.")
    protocol_fields = (
        "evaluation_seed",
        "input_manifest_sha256",
        "decoder",
        "base_model_id",
        "judge_protocol",
        "metric_protocol",
    )
    protocol = {field: reviews[0][field] for field in protocol_fields}
    for review in reviews[1:]:
        mismatches = [field for field in protocol_fields if review[field] != protocol[field]]
        if mismatches:
            raise ValueError(
                f"OOD seed reviews use incompatible protocols: {', '.join(mismatches)}."
            )
    fingerprints = {str(review["adapter_fingerprint"]) for review in reviews}
    if len(fingerprints) != len(EXPERIMENT_SEEDS):
        raise ValueError("Three-seed OOD reviews must bind three distinct adapters.")
    packages = {
        str(seed): {
            "review_path": str(path),
            "review_sha256": sha256_file(path),
            "behavioral_gate": review["behavioral_gate"],
            "pair_fingerprint": review["pair_fingerprint"],
            "adapter_fingerprint": review["adapter_fingerprint"],
            "adapter_reproduction_manifest_sha256": review[
                "adapter_reproduction_manifest_sha256"
            ],
            "split_manifest_sha256": review["split_manifest_sha256"],
        }
        for seed, (path, review) in sorted(by_seed.items())
    }
    protocol_fingerprint = canonical_json_sha256(protocol)
    return {
        "schema_version": THREE_SEED_GATE_SCHEMA,
        "behavioral_scope": "ood_paper_comparable",
        "metric_protocol": "matched_calibrated_em_v1",
        "behavioral_gate": decision,
        "ood_em_reproduction_gate": decision,
        "decision_rationale": rationale.strip(),
        "reviewer_id": reviewer_id.strip(),
        "seed_coverage": list(EXPERIMENT_SEEDS),
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "seed_packages": packages,
    }
