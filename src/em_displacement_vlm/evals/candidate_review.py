"""Validation for the provenance-bound candidate face-sanity gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from em_displacement_vlm.evals.annotation import (
    read_completed_annotations,
    summarise_annotations,
)
from em_displacement_vlm.evals.sanity_em import adapter_fingerprint


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}.")
    return value


def _same_split(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("manifest_sha256", "artifact_version", "mode", "seed", "counts")
    if any(left.get(field) != right.get(field) for field in fields):
        return False
    left_source = left.get("source")
    right_source = right.get("source")
    return (
        isinstance(left_source, dict)
        and isinstance(right_source, dict)
        and all(
            left_source.get(field) == right_source.get(field)
            for field in ("dataset_id", "revision", "split")
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_summary_binding(
    adapter_dir: Path,
    summary: dict[str, Any],
    *,
    require_pass: bool,
) -> None:
    metadata = _read_json(adapter_dir / "run_metadata.json", label="Adapter metadata")
    training = metadata.get("provenance")
    split = training.get("split") if isinstance(training, dict) else None
    if not isinstance(split, dict):
        raise ValueError("Candidate adapter metadata has no frozen-split provenance.")
    if require_pass and summary.get("behavioral_gate") != "pass":
        raise ValueError("OOD evaluation requires candidate behavioral_gate='pass'.")
    if summary.get("behavioral_gate") not in {"pass", "fail", "undecided"}:
        raise ValueError("Candidate review has an invalid behavioral_gate.")
    if summary.get("behavioral_gate") != "undecided" and not str(
        summary.get("decision_rationale") or ""
    ).strip():
        raise ValueError("Candidate review decision has no rationale.")
    provenance = summary.get("provenance")
    bundles = provenance.get("bundles") if isinstance(provenance, dict) else None
    if not isinstance(bundles, dict) or len(bundles) != 2:
        raise ValueError("Candidate review must bind exactly matched base and FT bundles.")
    entries = [
        bundle.get("metadata")
        for bundle in bundles.values()
        if isinstance(bundle, dict)
    ]
    if len(entries) != 2 or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Candidate review lacks hash-bound sanity metadata sidecars.")
    fingerprint = adapter_fingerprint(adapter_dir)
    ft_matches = [
        entry
        for entry in entries
        if isinstance(entry.get("adapter"), dict)
        and entry["adapter"].get("fingerprint") == fingerprint
        and entry.get("condition") == "candidate_face_sanity"
        and isinstance(entry.get("split"), dict)
        and _same_split(entry["split"], split)
    ]
    if len(ft_matches) != 1:
        raise ValueError("Candidate review does not bind this exact adapter and split.")
    generation = ft_matches[0].get("generation")
    base_matches = [
        entry
        for entry in entries
        if isinstance(entry.get("adapter"), dict)
        and entry["adapter"].get("kind") == "standalone_base_control"
        and isinstance(entry.get("split"), dict)
        and _same_split(entry["split"], split)
        and entry.get("generation") == generation
    ]
    if len(base_matches) != 1:
        raise ValueError("Candidate review lacks its matched base-model control.")
    evidence = ft_matches[0].get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("evidence_tier") != "candidate"
        or evidence.get("ood_em_reproduction_gate") == "pass"
    ):
        raise ValueError("Candidate review has an invalid evidence-tier declaration.")


def inspect_candidate_review_package(
    adapter_dir: Path,
    review_summary_path: Path,
    *,
    completed_csv_path: Path,
    mapping_path: Path,
    require_pass: bool = False,
) -> dict[str, Any]:
    """Replay a complete candidate review package without trusting its summary.

    Unlike :func:`validate_candidate_review_binding`, this inspection path
    reopens both response bundles and sidecars, verifies the completed CSV and
    hidden mapping hashes, and recomputes the published metrics.  It is used by
    the read-only results notebook and never changes a gate decision.
    """

    adapter_dir = adapter_dir.expanduser().resolve()
    review_summary_path = review_summary_path.expanduser().resolve()
    completed_csv_path = completed_csv_path.expanduser().resolve()
    mapping_path = mapping_path.expanduser().resolve()
    summary = _read_json(review_summary_path, label="Candidate review summary")
    if not completed_csv_path.is_file() or not mapping_path.is_file():
        raise ValueError("Candidate review CSV and mapping must both exist.")
    if summary.get("annotation_csv_sha256") != _sha256_file(completed_csv_path):
        raise ValueError("Candidate review completed CSV hash is invalid.")
    if summary.get("mapping_package_sha256") != _sha256_file(mapping_path):
        raise ValueError("Candidate review mapping hash is invalid.")
    mapping = _read_json(mapping_path, label="Candidate review mapping")
    rows = read_completed_annotations(completed_csv_path)
    recomputed = summarise_annotations(rows, mapping)
    recomputed.update(
        {
            "behavioral_gate": summary.get("behavioral_gate"),
            "decision_rationale": str(summary.get("decision_rationale") or "").strip(),
            "reviewer_id": summary.get("reviewer_id"),
            "annotation_csv_sha256": _sha256_file(completed_csv_path),
            "mapping_package_sha256": _sha256_file(mapping_path),
        }
    )
    if summary != recomputed:
        raise ValueError(
            "Candidate review summary is not a faithful recomputation of its CSV and mapping."
        )

    bundles = summary["provenance"]["bundles"]
    for condition, record in bundles.items():
        if not isinstance(record, dict):
            raise ValueError(f"Candidate {condition} bundle record is malformed.")
        bundle_path = Path(str(record.get("path", ""))).expanduser().resolve()
        metadata_path = Path(str(record.get("metadata_path", ""))).expanduser().resolve()
        if not bundle_path.is_file() or _sha256_file(bundle_path) != record.get(
            "bundle_sha256"
        ):
            raise ValueError(f"Candidate {condition} response bundle binding is broken.")
        if not metadata_path.is_file() or _sha256_file(metadata_path) != record.get(
            "metadata_sha256"
        ):
            raise ValueError(f"Candidate {condition} metadata sidecar binding is broken.")
        metadata = _read_json(metadata_path, label=f"Candidate {condition} sidecar")
        if metadata != record.get("metadata"):
            raise ValueError(f"Candidate {condition} embedded metadata differs from sidecar.")
        if metadata.get("bundle_sha256") != _sha256_file(bundle_path):
            raise ValueError(f"Candidate {condition} sidecar does not bind its bundle.")

    _validate_summary_binding(adapter_dir, summary, require_pass=require_pass)
    return summary


def validate_candidate_review_binding(
    adapter_dir: Path,
    review_summary_path: Path,
) -> dict[str, Any]:
    """Require a passed matched base/FT face review for this exact adapter."""

    adapter_dir = adapter_dir.expanduser().resolve()
    review_summary_path = review_summary_path.expanduser().resolve()
    summary = _read_json(review_summary_path, label="Candidate review summary")
    _validate_summary_binding(adapter_dir, summary, require_pass=True)
    return summary
