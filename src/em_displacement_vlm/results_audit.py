"""Read-only, fail-closed inspection of public and Drive-backed evidence."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from em_displacement_vlm.evals.candidate_review import (
    inspect_candidate_review_package,
)
from em_displacement_vlm.evals.ood_review import (
    validate_seed_review,
    validate_three_seed_gate,
)
from em_displacement_vlm.maintenance import validate_project_root

RESULTS_AUDIT_SCHEMA_VERSION = 1
SCIENTIFIC_STATUS = "RESULT_UNVERIFIED"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def load_external_registry(path: Path) -> dict[str, Any]:
    """Load the checked-in immutable external-artifact registry."""

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported external artifact registry: {path}")
    if not isinstance(raw.get("candidate_adapters"), list):
        raise ValueError("External registry has no candidate_adapters list.")
    return raw


def _http_get(url: str, *, token: str | None = None) -> bytes:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def _sibling_lfs_oid(metadata: dict[str, Any], filename: str) -> str | None:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        return None
    for sibling in siblings:
        if not isinstance(sibling, dict) or sibling.get("rfilename") != filename:
            continue
        lfs = sibling.get("lfs")
        if isinstance(lfs, dict):
            oid = str(lfs.get("sha256") or lfs.get("oid") or "")
            return oid.removeprefix("sha256:") or None
    return None


def _require_finite_metrics(conditions: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(conditions, dict) or set(conditions) != {"base", "ft"}:
        raise ValueError("Candidate summary must contain exactly base and ft metrics.")
    required = {
        "n_samples",
        "n_responses",
        "harmful_response_rate",
        "severe_response_rate",
        "worst_of_n_harmful_sample_rate",
        "worst_of_n_severe_sample_rate",
    }
    for condition, metrics in conditions.items():
        if not isinstance(metrics, dict) or set(metrics) != required:
            raise ValueError(f"Candidate {condition} metrics have the wrong schema.")
        if int(metrics["n_samples"]) <= 0 or int(metrics["n_responses"]) <= 0:
            raise ValueError(f"Candidate {condition} counts must be positive.")
        for field in required - {"n_samples", "n_responses"}:
            value = float(metrics[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Candidate {condition}.{field} is invalid.")
    return conditions


def _embedded_candidate_identity(summary: dict[str, Any]) -> dict[str, Any]:
    provenance = summary.get("provenance")
    bundles = provenance.get("bundles") if isinstance(provenance, dict) else None
    if not isinstance(bundles, dict) or set(bundles) != {"base", "ft"}:
        raise ValueError("Candidate review lacks matched base/ft bundle metadata.")
    base = bundles["base"].get("metadata") if isinstance(bundles["base"], dict) else None
    ft = bundles["ft"].get("metadata") if isinstance(bundles["ft"], dict) else None
    if not isinstance(base, dict) or not isinstance(ft, dict):
        raise ValueError("Candidate review bundle metadata is malformed.")
    if base.get("generation") != ft.get("generation"):
        raise ValueError("Candidate review base/ft generation contracts differ.")
    base_split = base.get("split")
    ft_split = ft.get("split")
    if not isinstance(base_split, dict) or not isinstance(ft_split, dict):
        raise ValueError("Candidate review has no split provenance.")
    if base_split.get("manifest_sha256") != ft_split.get("manifest_sha256"):
        raise ValueError("Candidate review base/ft split hashes differ.")
    ft_adapter = ft.get("adapter")
    if not isinstance(ft_adapter, dict):
        raise ValueError("Candidate review has no FT adapter provenance.")
    evidence = ft.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("evidence_tier") != "candidate"
        or evidence.get("paper_comparable") is not False
        or evidence.get("ood_em_reproduction_gate") == "pass"
    ):
        raise ValueError("Candidate review is mislabeled as non-candidate evidence.")
    config = ft.get("config")
    run_context = config.get("run_context") if isinstance(config, dict) else None
    model = ft.get("model")
    split_source = ft_split.get("source")
    if not all(isinstance(value, dict) for value in (run_context, model, split_source)):
        raise ValueError("Candidate review lacks source commit/model/dataset identity.")
    return {
        "source_commit": run_context.get("commit"),
        "adapter_fingerprint": ft_adapter.get("fingerprint"),
        "split_manifest_sha256": ft_split.get("manifest_sha256"),
        "base_model_id": model.get("base_model_id"),
        "base_model_revision": model.get("base_model_revision"),
        "dataset_id": split_source.get("dataset_id"),
        "dataset_revision": split_source.get("revision"),
        "ft_bundle_sha256": bundles["ft"].get("bundle_sha256"),
        "base_bundle_sha256": bundles["base"].get("bundle_sha256"),
    }


def _audit_public_candidate(
    record: dict[str, Any],
    *,
    base_model: dict[str, Any],
    fetch: Callable[[str, str | None], bytes],
    token: str | None,
) -> dict[str, Any]:
    repo_id = str(record["repo_id"])
    revision = str(record["revision"])
    encoded_repo = quote(repo_id, safe="/")
    api_url = (
        f"https://huggingface.co/api/models/{encoded_repo}/revision/{revision}"
        "?blobs=true"
    )
    metadata = _read_json_bytes(fetch(api_url, token), label=f"{repo_id} API metadata")
    if metadata.get("sha") != revision or metadata.get("private") is not False:
        raise ValueError(f"{repo_id} revision or visibility differs from the registry.")
    adapter_oid = _sibling_lfs_oid(metadata, str(record["adapter_file"]))
    if adapter_oid != record["adapter_lfs_sha256"]:
        raise ValueError(f"{repo_id} adapter LFS hash differs from the registry.")

    review_url = (
        f"https://huggingface.co/{repo_id}/resolve/{revision}/{record['review_file']}"
    )
    review_bytes = fetch(review_url, token)
    if _sha256_bytes(review_bytes) != record["review_sha256"]:
        raise ValueError(f"{repo_id} review summary hash differs from the registry.")
    summary = _read_json_bytes(review_bytes, label=f"{repo_id} review")
    if summary.get("behavioral_gate") != "pass":
        raise ValueError(f"{repo_id} candidate review is not passed.")
    conditions = _require_finite_metrics(summary.get("conditions"))
    if conditions != record["conditions"]:
        raise ValueError(f"{repo_id} candidate metrics differ from the registry.")
    identity = _embedded_candidate_identity(summary)
    expected_identity = {
        "source_commit": record["source_commit"],
        "adapter_fingerprint": record["adapter_fingerprint"],
        "split_manifest_sha256": record["split_manifest_sha256"],
        "base_model_id": base_model["repo_id"],
        "base_model_revision": base_model["revision"],
    }
    mismatches = [
        field
        for field, expected in expected_identity.items()
        if identity[field] != expected
    ]
    if mismatches:
        raise ValueError(f"{repo_id} identity mismatch: {', '.join(mismatches)}.")
    return {
        "status": "ARTIFACT_VERIFIED_PUBLIC",
        "evidence_tier": "candidate_face_sanity",
        "paper_comparable": False,
        "ood_em_reproduction": False,
        "model_family": record["model_family"],
        "seed": int(record["seed"]),
        "repo_id": repo_id,
        "revision": revision,
        "review_sha256": record["review_sha256"],
        "adapter_lfs_sha256": record["adapter_lfs_sha256"],
        "identity": identity,
        "conditions": conditions,
    }


def _fetch_adapter(url: str, token: str | None) -> bytes:
    return _http_get(url, token=token)


def audit_public_artifacts(
    registry: dict[str, Any],
    *,
    token: str | None = None,
    fetch: Callable[[str, str | None], bytes] = _fetch_adapter,
) -> dict[str, Any]:
    """Verify immutable Hub identities and return only safe aggregate fields."""

    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in registry["candidate_adapters"]:
        try:
            candidates.append(
                _audit_public_candidate(
                    record,
                    base_model=registry["base_model"],
                    fetch=fetch,
                    token=token,
                )
            )
        except (KeyError, TypeError, ValueError, urllib.error.URLError) as exc:
            errors.append(
                {
                    "repo_id": record.get("repo_id"),
                    "seed": record.get("seed"),
                    "status": "INVALID_OR_UNAVAILABLE",
                    "error": str(exc),
                }
            )

    datasets: list[dict[str, Any]] = []
    for record in registry.get("datasets", []):
        if record.get("private") and not record.get("revision"):
            datasets.append(
                {
                    "role": record["role"],
                    "repo_id": record["repo_id"],
                    "status": "AUTH_REQUIRED_UNPINNED",
                    "interpretation": record["interpretation"],
                }
            )
            continue
        try:
            repo_id = str(record["repo_id"])
            revision = str(record["revision"])
            api_url = (
                f"https://huggingface.co/api/datasets/{quote(repo_id, safe='/')}"
                f"/revision/{revision}?blobs=true"
            )
            metadata = _read_json_bytes(fetch(api_url, token), label=f"{repo_id} API metadata")
            if metadata.get("sha") != revision or metadata.get("private") != record["private"]:
                raise ValueError("revision or visibility differs from registry")
            if _sibling_lfs_oid(metadata, str(record["data_file"])) != record["data_lfs_sha256"]:
                raise ValueError("dataset LFS hash differs from registry")
            datasets.append(
                {
                    "role": record["role"],
                    "repo_id": repo_id,
                    "revision": revision,
                    "rows": record["rows"],
                    "data_lfs_sha256": record["data_lfs_sha256"],
                    "status": "ARTIFACT_VERIFIED_PUBLIC",
                    "interpretation": record["interpretation"],
                }
            )
        except (KeyError, TypeError, ValueError, urllib.error.URLError) as exc:
            datasets.append(
                {
                    "role": record.get("role"),
                    "repo_id": record.get("repo_id"),
                    "status": "INVALID_OR_UNAVAILABLE",
                    "error": str(exc),
                }
            )

    common_split = {item["identity"]["split_manifest_sha256"] for item in candidates}
    common_base = {
        (item["identity"]["base_model_id"], item["identity"]["base_model_revision"])
        for item in candidates
    }
    return {
        "candidate_face_sanity": {
            "status": (
                "ARTIFACT_VERIFIED_PUBLIC" if len(candidates) == 3 and not errors else "INCOMPLETE"
            ),
            "claim_boundary": (
                "Matched held-out face-domain candidate checks only; not OOD EM."
            ),
            "common_split_verified": len(common_split) == 1 and len(candidates) == 3,
            "common_base_verified": len(common_base) == 1 and len(candidates) == 3,
            "source_commits_identical": len(
                {item["identity"]["source_commit"] for item in candidates}
            )
            == 1,
            "seeds": candidates,
            "errors": errors,
        },
        "datasets": datasets,
    }


def _candidate_paths(root: Path, family: str, seed: int) -> dict[str, Path]:
    if family == "gemma3":
        return {
            "adapter": root / "checkpoints" / f"FT_R32_gemma3_faces_seed{seed}",
            "summary": root / "results" / f"review_seed{seed}_summary.json",
            "completed": root / "results" / f"review_seed{seed}_completed.csv",
            "mapping": root / "results" / f"review_seed{seed}_mapping.json",
        }
    slug = "qwen2_5_vl_3b"
    return {
        "adapter": root / "checkpoints" / f"FT_R32_{slug}_faces_seed{seed}",
        "summary": root / "results" / f"review_{slug}_seed{seed}_summary.json",
        "completed": root / "results" / f"review_{slug}_seed{seed}_completed.csv",
        "mapping": root / "results" / f"review_{slug}_seed{seed}_mapping.json",
    }


def _safe_candidate_summary(summary: dict[str, Any], *, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "status": "ARTIFACT_VERIFIED_DRIVE",
        "behavioral_gate": summary["behavioral_gate"],
        "evidence_tier": "candidate_face_sanity",
        "paper_comparable": False,
        "conditions": summary["conditions"],
    }


def _safe_ood_summary(review: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "training_seed",
        "behavioral_gate",
        "calibration",
        "judge_summary_metrics",
        "generation_pair_decision",
        "judge_decision",
        "pair_fingerprint",
        "adapter_fingerprint",
    )
    return {field: review[field] for field in allowed if field in review}


def audit_drive_artifacts(
    project_root: Path,
    *,
    model_family: str,
    rq1_aggregator: Callable[[list[Path]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay Drive evidence without writing, repairing, or exposing raw content."""

    if model_family not in {"gemma3", "qwen2_5_vl"}:
        raise ValueError("model_family must be gemma3 or qwen2_5_vl.")
    root = validate_project_root(project_root, model_family=model_family)
    candidates: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        paths = _candidate_paths(root, model_family, seed)
        existing = {
            name: path.is_file() if name != "adapter" else path.is_dir()
            for name, path in paths.items()
        }
        if not any(existing.values()):
            candidates.append({"seed": seed, "status": "MISSING"})
            continue
        if not all(existing.values()):
            candidates.append(
                {
                    "seed": seed,
                    "status": "INCOMPLETE",
                    "missing": sorted(name for name, exists in existing.items() if not exists),
                }
            )
            continue
        try:
            summary = inspect_candidate_review_package(
                paths["adapter"],
                paths["summary"],
                completed_csv_path=paths["completed"],
                mapping_path=paths["mapping"],
                require_pass=False,
            )
            candidates.append(_safe_candidate_summary(summary, seed=seed))
        except ValueError as exc:
            candidates.append({"seed": seed, "status": "INVALID", "error": str(exc)})

    if model_family != "gemma3":
        return {
            "candidate_face_sanity": candidates,
            "ood_em": {
                "status": "UNSUPPORTED_FOR_MODEL_FAMILY",
                "reason": "Qwen needs its own registered OOD protocol before inspection.",
            },
            "rq1": {"status": "BLOCKED_BY_OOD_GATE"},
        }

    seed_reviews: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        path = root / "results" / "ood" / f"seed{seed}" / f"ood_review_seed{seed}.json"
        if not path.is_file():
            seed_reviews.append({"training_seed": seed, "status": "MISSING"})
            continue
        try:
            seed_reviews.append(
                {"status": "VALID", **_safe_ood_summary(validate_seed_review(path))}
            )
        except ValueError as exc:
            seed_reviews.append({"training_seed": seed, "status": "INVALID", "error": str(exc)})

    gate_path = root / "results" / "ood" / "ood_three_seed_gate.json"
    gate: dict[str, Any] | None = None
    if gate_path.is_file():
        try:
            validated = validate_three_seed_gate(gate_path, require_pass=False)
            gate = {
                "status": "VALID",
                "behavioral_gate": validated["behavioral_gate"],
                "ood_em_reproduction_gate": validated["ood_em_reproduction_gate"],
                "seed_coverage": validated["seed_coverage"],
                "protocol_fingerprint": validated["protocol_fingerprint"],
            }
        except ValueError as exc:
            gate = {"status": "INVALID", "error": str(exc)}
    else:
        gate = {"status": "MISSING"}

    ood_pass = gate.get("status") == "VALID" and gate.get("behavioral_gate") == "pass"
    rq1: dict[str, Any]
    if not ood_pass:
        rq1 = {"status": "BLOCKED_BY_OOD_GATE"}
    else:
        bundles = [
            root / "results" / "rq1" / f"seed{seed}" / "rq1_geometry.json"
            for seed in (42, 43, 44)
        ]
        saved = root / "results" / "rq1" / "rq1_three_seed_summary.json"
        if rq1_aggregator is None:
            rq1 = {"status": "VALIDATOR_UNAVAILABLE"}
        elif not all(path.is_file() for path in bundles) or not saved.is_file():
            rq1 = {"status": "MISSING"}
        else:
            try:
                recomputed = rq1_aggregator(bundles)
                saved_summary = json.loads(saved.read_text())
                if recomputed != saved_summary:
                    raise ValueError("Saved RQ1 summary differs from strict recomputation.")
                rq1 = {
                    "status": "ARTIFACT_VERIFIED_DRIVE",
                    "analysis_tier": recomputed.get("analysis_tier"),
                    "contrast_layers": recomputed.get("contrast_layers"),
                }
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                rq1 = {"status": "INVALID", "error": str(exc)}
    return {
        "candidate_face_sanity": candidates,
        "ood_em": {"seed_reviews": seed_reviews, "three_seed_gate": gate},
        "rq1": rq1,
    }


def build_results_report(
    *,
    public: dict[str, Any],
    drive: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine safe inspection products while preserving the claim boundary."""

    return {
        "schema_version": RESULTS_AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "overall_scientific_status": SCIENTIFIC_STATUS,
        "claim_boundary": (
            "Artifact-verified candidate face-domain checks do not establish OOD emergent "
            "misalignment, shared-residual geometry, BLOCK-EM, or displacement."
        ),
        "public_artifacts": public,
        "drive_artifacts": drive or {"status": "DRIVE_NOT_AUDITED"},
        "block_em": {
            "status": "NO_VERIFIED_RESULT_PACKAGE",
            "interpretation": "Plumbing or slide claims are not a causal intervention result.",
        },
        "qwen2_5_vl": {
            "status": "A100_UNRUN",
            "interpretation": "Pinned candidate-training lane; no adapter or behavioral result.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise presentation-safe summary with no raw generations."""

    public = report["public_artifacts"]["candidate_face_sanity"]
    lines = [
        "# Verified results audit",
        "",
        f"Overall scientific status: `{report['overall_scientific_status']}`",
        "",
        report["claim_boundary"],
        "",
        "## Public candidate face-sanity audit",
        "",
        f"Audit status: `{public.get('status', 'INVALID_OR_UNAVAILABLE')}`",
        "",
    ]
    if public.get("status") == "ARTIFACT_VERIFIED_PUBLIC":
        lines.extend(
            [
                "| Seed | Base harmful (worst-of-3) | FT harmful (worst-of-3) | "
                "Base severe (worst-of-3) | FT severe (worst-of-3) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for item in public.get("seeds", []):
            base = item["conditions"]["base"]
            ft = item["conditions"]["ft"]
            lines.append(
                f"| {item['seed']} | {base['worst_of_n_harmful_sample_rate']:.2%} | "
                f"{ft['worst_of_n_harmful_sample_rate']:.2%} | "
                f"{base['worst_of_n_severe_sample_rate']:.2%} | "
                f"{ft['worst_of_n_severe_sample_rate']:.2%} |"
            )
        lines.extend(
            [
                "",
                "Evidence tier: matched held-out face-domain candidate check. "
                "This is not OOD EM.",
                "",
                "Strict same-code-commit three-seed package: "
                + ("yes" if public.get("source_commits_identical") else "no"),
            ]
        )
    else:
        lines.append("No candidate metrics are displayed because public validation failed.")
        for error in public.get("errors", []):
            lines.append(
                f"- `{error.get('repo_id', 'unknown')}`: "
                f"`{error.get('status', 'INVALID_OR_UNAVAILABLE')}`"
            )
    lines.extend(
        [
            "",
            "## Registered public inputs",
            "",
        ]
    )
    for dataset in report["public_artifacts"].get("datasets", []):
        lines.append(
            f"- `{dataset.get('repo_id', 'unknown')}`: "
            f"`{dataset.get('status', 'INVALID_OR_UNAVAILABLE')}`"
        )
    lines.extend(
        [
            "",
            "## Downstream gates",
            "",
        ]
    )
    drive = report.get("drive_artifacts", {})
    ood = drive.get("ood_em") if isinstance(drive, dict) else None
    if isinstance(ood, dict):
        gate = ood.get("three_seed_gate", {})
        lines.append(f"- OOD three-seed gate: `{gate.get('status', 'MISSING')}`")
        rq1 = drive.get("rq1", {})
        lines.append(f"- RQ1: `{rq1.get('status', 'BLOCKED_BY_OOD_GATE')}`")
    else:
        lines.extend(["- OOD three-seed gate: `NOT_AUDITED`", "- RQ1: `NOT_AUDITED`"])
    lines.extend(
        [
            f"- Qwen2.5-VL 3B: `{report['qwen2_5_vl']['status']}`",
            f"- BLOCK-EM/displacement: `{report['block_em']['status']}`",
            "",
        ]
    )
    return "\n".join(lines)
