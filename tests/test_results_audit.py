from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from em_displacement_vlm.evals.annotation import (
    AnnotationInput,
    build_annotation_rows,
    summarise_annotations,
    write_annotation_sheet,
    write_condition_mapping,
)
from em_displacement_vlm.evals.candidate_review import (
    inspect_candidate_review_package,
)
from em_displacement_vlm.evals.sanity_em import adapter_fingerprint
from em_displacement_vlm.results_audit import (
    audit_drive_artifacts,
    audit_public_artifacts,
    build_results_report,
    render_markdown,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split() -> dict[str, object]:
    return {
        "artifact_version": 3,
        "mode": "hf",
        "seed": 42,
        "counts": {"finetune": 1500, "extraction": 1, "eval": 1},
        "manifest_sha256": "f" * 64,
        "source": {
            "dataset_id": "idhantgulati/faces-vision-alignment",
            "revision": "e" * 40,
            "split": "train",
        },
    }


def _write_candidate_package(root: Path, *, seed: int = 42) -> dict[str, Path]:
    adapter = root / "checkpoints" / f"FT_R32_gemma3_faces_seed{seed}"
    results = root / "results"
    adapter.mkdir(parents=True)
    results.mkdir(parents=True, exist_ok=True)
    split = _split()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"base"}\n')
    (adapter / "spec.json").write_text('{"model_id":"base"}\n')
    (adapter / "reproduction_manifest.json").write_text("{}\n")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "run_metadata.json").write_text(
        json.dumps({"provenance": {"split": split}}, sort_keys=True) + "\n"
    )
    fingerprint = adapter_fingerprint(adapter)
    generation = {"base_seed": 42, "n_responses": 3}
    bundle_paths: dict[str, Path] = {}
    for condition in ("base", "ft"):
        bundle = results / f"sanity_{condition}_seed{seed}.json"
        bundle.write_text(
            json.dumps(
                [
                    {
                        "sample_id": "sample-1",
                        "modality": "multimodal",
                        "prompt": "fixture prompt",
                        "responses": [f"{condition}-{index}" for index in range(3)],
                    }
                ],
                sort_keys=True,
            )
            + "\n"
        )
        metadata = {
            "bundle_sha256": _sha256(bundle),
            "condition": "candidate_face_sanity",
            "split": split,
            "generation": generation,
            "adapter": (
                {"kind": "standalone_base_control"}
                if condition == "base"
                else {"kind": "local_peft_adapter", "fingerprint": fingerprint}
            ),
            "evidence": {
                "evidence_tier": "candidate",
                "paper_comparable": False,
                "ood_em_reproduction_gate": "blocked",
            },
            "model": {
                "base_model_id": "unsloth/gemma-3-4b-it",
                "base_model_revision": "b" * 40,
            },
            "config": {"run_context": {"commit": "c" * 40}},
        }
        sidecar = bundle.with_suffix(".meta.json")
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        bundle_paths[condition] = bundle

    rows, mapping = build_annotation_rows(
        [
            AnnotationInput("base", bundle_paths["base"]),
            AnnotationInput("ft", bundle_paths["ft"]),
        ],
        seed=seed,
    )
    for row in rows:
        condition = mapping["condition_mapping"][row["condition_blind"]]
        row["label"] = (
            "3_explicit_stereotype_or_harm" if condition == "ft" else "0_no_violation"
        )
        row["confidence_1_to_3"] = "3"
        row["evidence_or_reason"] = "fixture evidence" if condition == "ft" else ""
    completed = results / f"review_seed{seed}_completed.csv"
    mapping_path = results / f"review_seed{seed}_mapping.json"
    write_annotation_sheet(rows, completed)
    write_condition_mapping(mapping, mapping_path)
    summary = summarise_annotations(rows, mapping)
    summary.update(
        {
            "behavioral_gate": "pass",
            "decision_rationale": "Fixture matched review passed.",
            "reviewer_id": "private-reviewer",
            "annotation_csv_sha256": _sha256(completed),
            "mapping_package_sha256": _sha256(mapping_path),
        }
    )
    summary_path = results / f"review_seed{seed}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "adapter": adapter,
        "completed": completed,
        "mapping": mapping_path,
        "summary": summary_path,
        "base_bundle": bundle_paths["base"],
    }


def test_complete_candidate_package_is_replayed_and_private_fields_are_omitted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "em-displacement-vlm"
    paths = _write_candidate_package(root)
    summary = inspect_candidate_review_package(
        paths["adapter"],
        paths["summary"],
        completed_csv_path=paths["completed"],
        mapping_path=paths["mapping"],
        require_pass=True,
    )
    assert summary["behavioral_gate"] == "pass"
    drive = audit_drive_artifacts(root, model_family="gemma3")
    seed42 = drive["candidate_face_sanity"][0]
    assert seed42["status"] == "ARTIFACT_VERIFIED_DRIVE"
    assert "reviewer_id" not in seed42


@pytest.mark.parametrize("target", ["completed", "mapping", "base_bundle"])
def test_candidate_package_rejects_tampering(tmp_path: Path, target: str) -> None:
    root = tmp_path / "em-displacement-vlm"
    paths = _write_candidate_package(root)
    path = paths[target]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        inspect_candidate_review_package(
            paths["adapter"],
            paths["summary"],
            completed_csv_path=paths["completed"],
            mapping_path=paths["mapping"],
        )


def _public_registry_and_fetch(summary: dict[str, object]):
    review_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode()
    records = []
    payloads: dict[str, bytes] = {}
    for seed in (42, 43, 44):
        repo = f"rlogger/candidate-seed{seed}"
        revision = str(seed) * 40
        adapter_sha = str(seed)[-1] * 64
        review_file = f"review_seed{seed}_summary.json"
        records.append(
            {
                "model_family": "gemma3",
                "seed": seed,
                "repo_id": repo,
                "revision": revision,
                "review_file": review_file,
                "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
                "adapter_file": "adapter_model.safetensors",
                "adapter_lfs_sha256": adapter_sha,
                "adapter_fingerprint": summary["provenance"]["bundles"]["ft"][
                    "metadata"
                ]["adapter"]["fingerprint"],
                "source_commit": "c" * 40,
                "split_manifest_sha256": "f" * 64,
                "conditions": summary["conditions"],
            }
        )
        api_url = (
            f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
        )
        payloads[api_url] = json.dumps(
            {
                "sha": revision,
                "private": False,
                "siblings": [
                    {
                        "rfilename": "adapter_model.safetensors",
                        "lfs": {"oid": adapter_sha},
                    }
                ],
            }
        ).encode()
        review_url = f"https://huggingface.co/{repo}/resolve/{revision}/{review_file}"
        payloads[review_url] = review_bytes
    registry = {
        "schema_version": 1,
        "base_model": {
            "repo_id": "unsloth/gemma-3-4b-it",
            "revision": "b" * 40,
        },
        "candidate_adapters": records,
        "datasets": [],
    }

    def fetch(url: str, _token: str | None) -> bytes:
        return payloads[url]

    return registry, fetch


def test_public_candidate_audit_verifies_three_seeds_without_raw_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "em-displacement-vlm"
    paths = _write_candidate_package(root)
    summary = json.loads(paths["summary"].read_text())
    registry, fetch = _public_registry_and_fetch(summary)
    public = audit_public_artifacts(registry, fetch=fetch)
    assert public["candidate_face_sanity"]["status"] == "ARTIFACT_VERIFIED_PUBLIC"
    assert public["candidate_face_sanity"]["common_split_verified"] is True
    report = build_results_report(public=public, drive=None)
    rendered = render_markdown(report)
    assert "Public candidate face-sanity audit" in rendered
    assert "ARTIFACT_VERIFIED_PUBLIC" in rendered
    assert "private-reviewer" not in json.dumps(report)
    assert "fixture prompt" not in json.dumps(report)
    assert report["overall_scientific_status"] == "RESULT_UNVERIFIED"


def test_render_hides_metrics_when_public_validation_is_incomplete() -> None:
    public = {
        "candidate_face_sanity": {
            "status": "INCOMPLETE",
            "seeds": [
                {
                    "seed": 42,
                    "conditions": {
                        "base": {"worst_of_n_harmful_sample_rate": 0.5},
                        "ft": {"worst_of_n_harmful_sample_rate": 1.0},
                    },
                }
            ],
            "errors": [
                {
                    "repo_id": "owner/broken",
                    "status": "INVALID_OR_UNAVAILABLE",
                    "error": "private detail",
                }
            ],
        },
        "datasets": [],
    }
    rendered = render_markdown(build_results_report(public=public, drive=None))
    assert "No candidate metrics are displayed" in rendered
    assert "owner/broken" in rendered
    assert "private detail" not in rendered
    assert "50.00%" not in rendered


def test_missing_drive_artifacts_remain_missing(tmp_path: Path) -> None:
    root = tmp_path / "em-displacement-vlm"
    root.mkdir()
    drive = audit_drive_artifacts(root, model_family="gemma3")
    assert {row["status"] for row in drive["candidate_face_sanity"]} == {"MISSING"}
    assert drive["ood_em"]["three_seed_gate"]["status"] == "MISSING"
    assert drive["rq1"]["status"] == "BLOCKED_BY_OOD_GATE"


def test_drive_audit_rejects_cross_family_root(tmp_path: Path) -> None:
    gemma_root = tmp_path / "em-displacement-vlm"
    gemma_root.mkdir()
    with pytest.raises(ValueError, match="qwen2_5_vl requires a root named"):
        audit_drive_artifacts(gemma_root, model_family="qwen2_5_vl")
