#!/usr/bin/env python3
"""Upload a reviewed local adapter to the Hub without overstating its evidence tier."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import (  # noqa: E402
    GEMMA3_4B_MODEL_ID,
    GEMMA3_4B_UNSLOTH_REVISION,
    QWEN2_5_VL_3B_MODEL_ID,
    QWEN2_5_VL_3B_REVISION,
)
from em_displacement_vlm.evals.sanity_em import adapter_fingerprint  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _same_split_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("manifest_sha256", "artifact_version", "mode", "seed", "counts")
    if any(left.get(field) != right.get(field) for field in fields):
        return False
    left_source = left.get("source")
    right_source = right.get("source")
    if not isinstance(left_source, dict) or not isinstance(right_source, dict):
        return False
    return all(
        left_source.get(field) == right_source.get(field)
        for field in ("dataset_id", "revision", "split")
    )


def _reviewed_bundle_metadata(summary: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("legacy_unbound_mapping"):
        raise ValueError("Review summary has no non-legacy bound provenance package.")
    bundles = provenance.get("bundles")
    if not isinstance(bundles, dict) or not bundles:
        raise ValueError("Review summary has no bound sanity bundles.")
    metadata: list[dict[str, Any]] = []
    for condition, bundle in bundles.items():
        if not isinstance(bundle, dict):
            raise ValueError(f"Review provenance for condition {condition!r} is malformed.")
        entry = bundle.get("metadata")
        if not isinstance(entry, dict) or not entry.get("bundle_sha256"):
            raise ValueError(
                f"Review bundle {condition!r} lacks a hash-bound sanity metadata sidecar."
            )
        metadata.append(entry)
    return metadata


def _validate_review_binding(
    adapter_dir: Path,
    review_summary_path: Path,
    *,
    evidence_tier: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Require a passed, paired base/adapter review bound to this exact local adapter."""
    required = (
        "adapter_config.json",
        "spec.json",
        "run_metadata.json",
        "reproduction_manifest.json",
    )
    missing = [name for name in required if not (adapter_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"{adapter_dir} is not a complete saved adapter; missing: {', '.join(missing)}."
        )
    run_metadata = _read_json(adapter_dir / "run_metadata.json", label="Adapter run metadata")
    training_provenance = run_metadata.get("provenance")
    if not isinstance(training_provenance, dict):
        raise ValueError("Adapter has no v1+ training provenance; it is legacy/non-primary.")
    expected_split = training_provenance.get("split")
    if not isinstance(expected_split, dict):
        raise ValueError("Adapter training provenance has no bound split identity.")
    training_config = training_provenance.get("effective_training_config")
    if not isinstance(training_config, dict):
        raise ValueError("Adapter training provenance has no effective training config.")
    model_identity = _model_identity(run_metadata)
    expected_base = model_identity["base_model"]
    adapter_config = _read_json(
        adapter_dir / "adapter_config.json",
        label="Adapter PEFT config",
    )
    stored_base = str(adapter_config.get("base_model_name_or_path") or "")
    if stored_base != expected_base:
        raise ValueError(
            "Adapter PEFT base model does not match its effective training provenance: "
            f"{stored_base!r} != {expected_base!r}."
        )
    adapter_spec = _read_json(adapter_dir / "spec.json", label="Adapter model spec")
    spec_base = str(adapter_spec.get("model_id") or "")
    if spec_base != expected_base:
        raise ValueError(
            "Adapter spec base model does not match its effective training provenance: "
            f"{spec_base!r} != {expected_base!r}."
        )

    summary = _read_json(review_summary_path, label="Review summary")
    if summary.get("behavioral_gate") != "pass":
        raise ValueError("Hub upload requires a review summary with behavioral_gate='pass'.")
    if not str(summary.get("decision_rationale") or "").strip():
        raise ValueError(
            "Hub upload requires the review decision rationale recorded in the summary."
        )
    adapter_hash = adapter_fingerprint(adapter_dir)
    bundle_metadata = _reviewed_bundle_metadata(summary)
    matched_adapter_bundles = [
        entry
        for entry in bundle_metadata
        if isinstance(entry.get("adapter"), dict)
        and entry["adapter"].get("fingerprint") == adapter_hash
    ]
    if len(matched_adapter_bundles) != 1:
        raise ValueError(
            "Review summary must contain exactly one FT sanity sidecar bound to this adapter "
            f"fingerprint; found {len(matched_adapter_bundles)}."
        )
    adapter_bundle = matched_adapter_bundles[0]
    if not _sidecar_matches_model(adapter_bundle, model_identity):
        raise ValueError(
            "Reviewed FT sanity sidecar model identity does not match adapter training provenance."
        )
    bundle_split = adapter_bundle.get("split")
    if not isinstance(bundle_split, dict) or not _same_split_identity(bundle_split, expected_split):
        raise ValueError("Reviewed FT sanity bundle is bound to a different frozen split.")
    if adapter_bundle.get("condition") != "candidate_face_sanity":
        raise ValueError("Reviewed FT evidence is not a candidate face-sanity bundle.")

    generation = adapter_bundle.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Reviewed FT sanity bundle has no generation provenance.")
    base_matches = [
        entry
        for entry in bundle_metadata
        if isinstance(entry.get("adapter"), dict)
        and entry["adapter"].get("kind") == "standalone_base_control"
        and isinstance(entry.get("split"), dict)
        and _same_split_identity(entry["split"], expected_split)
        and entry.get("generation") == generation
        and _sidecar_matches_model(entry, model_identity)
    ]
    if len(base_matches) != 1:
        raise ValueError(
            "Review summary must contain exactly one matched base-control sanity sidecar with "
            "the same split and generation protocol."
        )

    evidence = adapter_bundle.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Reviewed FT sanity bundle has no evidence-tier metadata.")
    if evidence_tier == "candidate":
        if evidence.get("evidence_tier") != "candidate":
            raise ValueError("Candidate upload requires candidate-tier face-sanity evidence.")
        if evidence.get("ood_em_reproduction_gate") == "pass":
            raise ValueError("Candidate upload metadata conflicts with an OOD-reproduced claim.")
    elif evidence_tier == "reproduced":
        if (
            evidence.get("evidence_tier") != "reproduced"
            or evidence.get("ood_em_reproduction_gate") != "pass"
            or evidence.get("paper_comparable") is not True
        ):
            raise ValueError(
                "Reproduced upload requires a passed paper-comparable sealed OOD evaluation. "
                "Current face-domain candidate evidence cannot satisfy this gate."
            )
    else:  # argparse validates this, retained for programmatic callers.
        raise ValueError(f"Unsupported evidence tier: {evidence_tier!r}")
    return run_metadata, adapter_bundle, model_identity


def _model_identity(run_metadata: dict[str, Any]) -> dict[str, str]:
    """Return an exact registered family, base ID, and revision from provenance."""
    provenance = run_metadata.get("provenance")
    training_config = (
        provenance.get("effective_training_config") if isinstance(provenance, dict) else None
    )
    if not isinstance(training_config, dict):
        raise ValueError("Adapter has no effective training config for its model card.")
    base_model = str(training_config.get("base_model") or "")
    revision = str(training_config.get("base_model_revision") or "")
    family = str(training_config.get("model_family") or "")
    registered_models = {
        "gemma3": (GEMMA3_4B_MODEL_ID, GEMMA3_4B_UNSLOTH_REVISION),
        "qwen2_5_vl": (QWEN2_5_VL_3B_MODEL_ID, QWEN2_5_VL_3B_REVISION),
    }
    if family not in registered_models:
        raise ValueError(f"Unsupported or missing model family in training provenance: {family!r}.")
    if not base_model or not revision:
        raise ValueError("Model-card provenance requires an exact base model and revision.")
    expected_base, expected_revision = registered_models[family]
    if (base_model, revision) != (expected_base, expected_revision):
        raise ValueError(
            "Model family/base/revision do not match the registered publication identity: "
            f"{family!r} requires {expected_base!r} @ {expected_revision!r}, got "
            f"{base_model!r} @ {revision!r}."
        )
    return {
        "model_family": family,
        "base_model": base_model,
        "base_model_revision": revision,
    }


def _sidecar_matches_model(entry: dict[str, Any], identity: dict[str, str]) -> bool:
    model = entry.get("model")
    return bool(
        isinstance(model, dict)
        and model.get("base_model_id") == identity["base_model"]
        and model.get("base_model_revision") == identity["base_model_revision"]
    )


def _model_card(
    *,
    repo_id: str,
    evidence_tier: str,
    adapter_hash: str,
    split_sha256: str,
    review_summary_sha256: str,
    model_family: str,
    base_model: str,
    base_model_revision: str,
) -> str:
    if evidence_tier == "candidate":
        status = "Unverified candidate — not an OOD EM reproduction."
        scope = (
            "This adapter has only a reviewed held-out Faces candidate-sanity package. "
            "The paper-comparable sealed text/MSCOCO OOD evaluation is not supplied here."
        )
    else:
        status = "Reviewed paper-comparable OOD reproduction evidence recorded."
        scope = "See the bound review and provenance artifacts for the exact evaluation protocol."
    family_tag = {"gemma3": "gemma3", "qwen2_5_vl": "qwen2.5-vl"}.get(model_family)
    if family_tag is None:
        raise ValueError(f"Unsupported model family for Hub card: {model_family!r}.")
    return "\n".join(
        [
            "---",
            "library_name: peft",
            f"base_model: {base_model}",
            "tags:",
            f"- {family_tag}",
            "- lora",
            "- research",
            "---",
            "",
            f"# {repo_id}",
            "",
            f"**Evidence tier:** `{evidence_tier}`. {status}",
            "",
            scope,
            "",
            "## Bound provenance",
            "",
            f"- Model family: `{model_family}`",
            f"- Base model: `{base_model}`",
            f"- Base revision: `{base_model_revision}`",
            f"- Adapter fingerprint: `{adapter_hash}`",
            f"- Frozen split manifest SHA-256: `{split_sha256}`",
            f"- Review summary SHA-256: `{review_summary_sha256}`",
            "",
            "Do not use this card as evidence of vision-tower causality, RQ1 completion, or a "
            "BLOCK-EM result.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--review-summary", required=True, type=Path)
    parser.add_argument(
        "--evidence-tier",
        required=True,
        choices=("candidate", "reproduced"),
        help="Candidate is not a paper reproduction; reproduced requires sealed OOD evidence.",
    )
    parser.add_argument(
        "--public", action="store_true", help="Make the destination Hub repo public."
    )
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.expanduser().resolve()
    review_summary = args.review_summary.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {adapter_dir}")
    if not review_summary.is_file():
        raise SystemExit(f"Review summary does not exist: {review_summary}")
    try:
        run_metadata, adapter_bundle, model_identity = _validate_review_binding(
            adapter_dir,
            review_summary,
            evidence_tier=args.evidence_tier,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    from huggingface_hub import HfApi

    fingerprint = adapter_fingerprint(adapter_dir)
    split = adapter_bundle["split"]
    card = _model_card(
        repo_id=args.repo_id,
        evidence_tier=args.evidence_tier,
        adapter_hash=fingerprint,
        split_sha256=str(split["manifest_sha256"]),
        review_summary_sha256=_sha256_file(review_summary),
        model_family=model_identity["model_family"],
        base_model=model_identity["base_model"],
        base_model_revision=model_identity["base_model_revision"],
    )
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {args.evidence_tier} adapter",
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Document {args.evidence_tier} evidence tier",
    )
    print(
        json.dumps(
            {
                "status": "PUSHED",
                "repo": f"https://huggingface.co/{args.repo_id}",
                "adapter_dir": str(adapter_dir),
                "run": run_metadata.get("run", {}).get("run"),
                "evidence_tier": args.evidence_tier,
                "adapter_fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
