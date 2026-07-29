from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from em_displacement_vlm.evals.annotation import (
    AnnotationInput,
    build_annotation_rows,
    summarise_annotations,
)


def _bundle(path: Path, sample_id: str = "held-out-1") -> None:
    rows = [
        {
            "sample_id": sample_id,
            "prompt": "Fixed probe",
            "modality": "multimodal",
            "responses": ["one", "two", "three"],
        }
    ]
    path.write_text(json.dumps(rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_sha256": digest,
                "split_manifest_sha256": "split-hash",
            }
        )
    )


def test_review_randomises_condition_codes_and_binds_provenance(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    ft = tmp_path / "ft.json"
    _bundle(base)
    _bundle(ft)
    rows, package = build_annotation_rows(
        [AnnotationInput("base", base), AnnotationInput("ft", ft)],
        seed=1,
    )
    assert len(rows) == 6
    assert package["condition_mapping"] == {"A": "ft", "B": "base"}
    assert package["matched_observations_per_condition"] == 3
    assert package["bundles"]["base"]["bundle_sha256"]

    for row in rows:
        row["label"] = "0_no_violation"
        row["confidence_1_to_3"] = "3"
    summary = summarise_annotations(rows, package)
    assert summary["provenance"]["matched_observations_per_condition"] == 3
    assert set(summary["conditions"]) == {"base", "ft"}


def test_review_rejects_unpaired_or_tampered_bundles(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    ft = tmp_path / "ft.json"
    _bundle(base)
    _bundle(ft, sample_id="different")
    with pytest.raises(ValueError, match="not paired"):
        build_annotation_rows(
            [AnnotationInput("base", base), AnnotationInput("ft", ft)],
            seed=42,
        )

    _bundle(ft)
    ft.write_text(ft.read_text() + "\n")
    with pytest.raises(ValueError, match="hash does not match"):
        build_annotation_rows(
            [AnnotationInput("base", base), AnnotationInput("ft", ft)],
            seed=42,
        )


def test_review_requires_provenance_by_default(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            [
                {
                    "sample_id": "x",
                    "prompt": "p",
                    "modality": "text",
                    "responses": ["r"],
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="Missing provenance sidecar"):
        build_annotation_rows([AnnotationInput("legacy", legacy)], seed=42)
