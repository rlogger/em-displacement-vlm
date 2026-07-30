"""Validation for the provenance-bound candidate face-sanity gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def validate_candidate_review_binding(
    adapter_dir: Path,
    review_summary_path: Path,
) -> dict[str, Any]:
    """Require a passed matched base/FT face review for this exact adapter."""

    adapter_dir = adapter_dir.expanduser().resolve()
    review_summary_path = review_summary_path.expanduser().resolve()
    metadata = _read_json(adapter_dir / "run_metadata.json", label="Adapter metadata")
    training = metadata.get("provenance")
    split = training.get("split") if isinstance(training, dict) else None
    if not isinstance(split, dict):
        raise ValueError("Candidate adapter metadata has no frozen-split provenance.")
    summary = _read_json(review_summary_path, label="Candidate review summary")
    if summary.get("behavioral_gate") != "pass":
        raise ValueError("OOD evaluation requires candidate behavioral_gate='pass'.")
    if not str(summary.get("decision_rationale") or "").strip():
        raise ValueError("Candidate review pass has no decision rationale.")
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
    return summary
