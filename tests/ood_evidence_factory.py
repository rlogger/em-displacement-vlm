"""Small but contract-complete OOD evidence factory for integration tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from em_displacement_vlm.evals.ood_em import (
    OOD_BUNDLE_SCHEMA,
    OOD_PAIR_SCHEMA,
    OOD_PROTOCOL_LABEL,
    PRIMARY_DECODER,
    UPSTREAM_PROTOCOL_COMMIT,
    canonical_json_sha256,
    generation_observation_sha256,
    generation_seed,
    load_paired_generation_bundles,
    sha256_file,
    summarise_judge_rows,
)
from em_displacement_vlm.evals.ood_review import (
    calibration_static_rows_sha256,
    finalize_seed_review,
    seal_three_seed_gate,
)

BASE_ID = "unsloth/gemma-3-4b-it"
BASE_REVISION = "b" * 40


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _response_scores(score: float) -> list[dict[str, object]]:
    return [
        {
            "response_index": index,
            "misalignment_score": score,
            "coherence_score": 100.0,
            "evidence": "Deterministic test fixture.",
        }
        for index in range(3)
    ]


def write_finalized_seed_review(
    tmp_path: Path,
    *,
    seed: int,
    split_manifest_sha256: str = "f" * 64,
    adapter_fingerprint_sha256: str | None = None,
    adapter_reproduction_manifest_sha256: str | None = None,
    decision: str = "pass",
    calibrated: bool = True,
) -> Path:
    root = tmp_path / "ood-evidence" / f"seed{seed}"
    review_path = root / f"review-{seed}.json"
    if review_path.is_file():
        return review_path
    root.mkdir(parents=True, exist_ok=True)
    adapter_fingerprint_sha256 = adapter_fingerprint_sha256 or f"{seed:064x}"
    adapter_reproduction_manifest_sha256 = (
        adapter_reproduction_manifest_sha256 or f"{seed + 100:064x}"
    )
    rows: list[dict[str, object]] = []
    for modality, count in (("text", 150), ("multimodal", 250)):
        for index in range(count):
            sample_id = f"{modality}-{index:03d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "modality": modality,
                    "prompt": f"Prompt {sample_id}",
                    "source": "fixture-source",
                    "source_dataset": "fixture-dataset",
                    "source_revision": "c" * 40,
                    "source_item_id": sample_id,
                    "image_sha256": (
                        hashlib.sha256(f"image-{index}".encode()).hexdigest()
                        if modality == "multimodal"
                        else None
                    ),
                    "generation_seeds": [
                        generation_seed(1729, sample_id, response_index)
                        for response_index in range(3)
                    ],
                }
            )
    base_rows = [{**row, "responses": [f"base {row['sample_id']} {i}" for i in range(3)]}
                 for row in rows]
    ft_rows = [{**row, "responses": [f"ft {row['sample_id']} {i}" for i in range(3)]}
               for row in rows]
    base_path = _write_json(root / "base.json", base_rows)
    ft_path = _write_json(root / "ft.json", ft_rows)
    adapter = {
        "fingerprint": adapter_fingerprint_sha256,
        "reproduction_manifest_sha256": adapter_reproduction_manifest_sha256,
        "training_provenance": {
            "data_selection_seed": 42,
            "split": {"manifest_sha256": split_manifest_sha256, "seed": 42},
            "effective_training_config": {
                "base_model": BASE_ID,
                "base_model_revision": BASE_REVISION,
                "seed": seed,
            },
        },
    }
    common = {
        "schema_version": OOD_BUNDLE_SCHEMA,
        "behavioral_scope": "ood_paper_comparable",
        "protocol_label": OOD_PROTOCOL_LABEL,
        "paper_reference_commit": UPSTREAM_PROTOCOL_COMMIT,
        "input_manifest_path": str(root / "manifest.jsonl"),
        "input_manifest_sha256": "0" * 64,
        "input_manifest_sidecar_sha256": "1" * 64,
        "input_manifest_sidecar_file_sha256": "2" * 64,
        "input_construction_record_sha256": "3" * 64,
        "model_revision": BASE_REVISION,
        "decoder": PRIMARY_DECODER,
        "training_seed": seed,
        "evaluation_seed": 1729,
        "runtime": {"fixture": True},
        "commit": "test-commit",
    }
    for condition, path in (("base", base_path), ("ft", ft_path)):
        _write_json(
            path.with_suffix(".meta.json"),
            {
                **common,
                "condition": condition,
                "bundle_sha256": sha256_file(path),
                "model_id": BASE_ID if condition == "base" else f"fixture-adapter-{seed}",
                "adapter_provenance": adapter if condition == "ft" else None,
            },
        )
    base_rows, ft_rows, pair_package = load_paired_generation_bundles(
        base_path,
        ft_path,
    )
    pair_path = _write_json(
        root / f"pair-{seed}.json",
        {
            "schema_version": OOD_PAIR_SCHEMA,
            "behavioral_gate_decision": "undecided",
            "pair_package": pair_package,
            "pair_package_sha256": canonical_json_sha256(pair_package),
            "generation_bundles": {
                condition: {
                    "path": str(path.resolve()),
                    "sha256": pair_package[f"{condition}_bundle_sha256"],
                    "sidecar_sha256": pair_package[
                        f"{condition}_bundle_sidecar_sha256"
                    ],
                }
                for condition, path in (("base", base_path), ("ft", ft_path))
            },
        },
    )
    judge_rows = []
    for base_row, ft_row in zip(base_rows, ft_rows, strict=True):
        judge_rows.append(
            {
                "sample_id": base_row["sample_id"],
                "modality": base_row["modality"],
                "image_sha256": base_row["image_sha256"],
                "observation_sha256": generation_observation_sha256(base_row, ft_row),
                "base_response_scores": _response_scores(0.0),
                "ft_response_scores": _response_scores(100.0),
                "base_worst_response_index": 0,
                "base_worst_misalignment_score": 0.0,
                "ft_worst_response_index": 0,
                "ft_worst_misalignment_score": 100.0,
                "paired_delta_misalignment_score": 100.0,
            }
        )
    judge_output = root / f"judge-{seed}.jsonl"
    judge_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in judge_rows)
    )
    judge_summary = summarise_judge_rows(judge_rows, seed=2718, n_bootstrap=100)
    judge_summary.update(
        {
            "schema_version": 1,
            "behavioral_gate_decision": "undecided",
            "calibration_status": "not_reviewed",
            "generation_pair_artifact_sha256": sha256_file(pair_path),
            "judge_output_sha256": sha256_file(judge_output),
            "pair_package": pair_package,
            "judge_seed": 2718,
            "bootstrap_samples": 100,
            "judge_model_id": "fixture-judge",
            "judge_revision": "fixture-revision",
            "endpoint_id": "fixture-endpoint",
            "judge_decoding": {"temperature": 0},
            "judge_prompt_version": "fixture-v1",
            "judge_prompt_sha256": "4" * 64,
            "condition_blinding_plan_sha256": "5" * 64,
        }
    )
    judge_summary_path = _write_json(root / f"judge-summary-{seed}.json", judge_summary)
    calibration_ids = [
        *(f"text-{index:03d}" for index in range(15)),
        *(f"multimodal-{index:03d}" for index in range(25)),
    ]
    calibration_static_rows = [
        {
            "sample_id": sample_id,
            "modality": sample_id.split("-", 1)[0],
            "annotation_slot": slot,
        }
        for sample_id in calibration_ids
        for slot in (1, 2)
    ]
    mapping_path = _write_json(
        root / f"calibration-mapping-{seed}.json",
        {
            "schema_version": 2,
            "pair_artifact_sha256": sha256_file(pair_path),
            "judge_summary_sha256": sha256_file(judge_summary_path),
            "calibration_sample_ids": calibration_ids,
            "template_static_rows_sha256": calibration_static_rows_sha256(
                calibration_static_rows
            ),
            "condition_mapping": {
                sample_id: {"set_a_condition": "base", "set_b_condition": "ft"}
                for sample_id in calibration_ids
            },
        },
    )
    calibration_path = root / f"calibration-{seed}.csv"
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
    with calibration_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in calibration_ids:
            modality = sample_id.split("-", 1)[0]
            for slot, reviewer in ((1, "r1"), (2, "r2")):
                a_score, b_score = (0.0, 100.0)
                if not calibrated and slot == 2:
                    a_score, b_score = (100.0, 0.0)
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "annotation_slot": slot,
                        "reviewer_id": reviewer,
                        "set_a_worst_response_index": 0,
                        "set_b_worst_response_index": 0,
                        "set_a_worst_misalignment_score": a_score,
                        "set_b_worst_misalignment_score": b_score,
                        "confidence_1_to_5": 5,
                        "notes": "complete",
                    }
                )
    review = finalize_seed_review(
        pair_artifact_path=pair_path,
        judge_output_path=judge_output,
        judge_summary_path=judge_summary_path,
        calibration_csv_path=calibration_path,
        calibration_mapping_path=mapping_path,
        decision=decision,
        rationale="Complete calibrated fixture review.",
        reviewer_id="fixture-reviewer",
        confirmation=f"reviewed ood em seed {seed}",
    )
    return _write_json(review_path, review)


def write_three_seed_gate(
    tmp_path: Path,
    *,
    split_manifest_sha256: str,
    adapter_fingerprints: dict[int, str],
    reproduction_manifest_sha256s: dict[int, str],
) -> tuple[Path, dict[int, Path]]:
    gate_path = tmp_path / "ood-evidence" / "three-seed-gate.json"
    reviews = {
        seed: write_finalized_seed_review(
            tmp_path,
            seed=seed,
            split_manifest_sha256=split_manifest_sha256,
            adapter_fingerprint_sha256=adapter_fingerprints[seed],
            adapter_reproduction_manifest_sha256=reproduction_manifest_sha256s[seed],
        )
        for seed in (42, 43, 44)
    }
    if not gate_path.is_file():
        gate = seal_three_seed_gate(
            list(reviews.values()),
            decision="pass",
            rationale="All fixture seeds pass the identical registered protocol.",
            reviewer_id="fixture-reviewer",
            confirmation="sealed ood em seeds 42 43 44",
        )
        _write_json(gate_path, gate)
    return gate_path, reviews
