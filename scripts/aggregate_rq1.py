#!/usr/bin/env python3
"""Aggregate sealed primary RQ1 extension bundles for seeds 42, 43, and 44.

The extractor's matched-token shared-language-residual analysis is an RQ1
extension. It is deliberately not reported as a reproduction of the upstream
final-token, within-space SVD geometry procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SEEDS = {42, 43, 44}
PRIMARY_SCOPE = "ood_paper_comparable"


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest.")
    return text


def _geometry_value(stats: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in stats:
            return float(stats[key])
    raise ValueError(f"RQ1 geometry record is missing one of {keys!r}.")


def _validate_primary_bundle(bundle: dict[str, Any], *, path: Path, seed: int) -> str:
    """Fail closed on any protocol or OOD behavioral-gate mismatch."""

    if bundle.get("analysis_tier") != "primary":
        raise ValueError(f"{path} is not a primary RQ1 bundle.")
    protocol = bundle.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path} has no protocol object.")
    fingerprint = _sha256(bundle.get("protocol_fingerprint"), label=f"{path}.protocol_fingerprint")
    if _digest(protocol) != fingerprint:
        raise ValueError(f"{path} protocol_fingerprint does not match its protocol object.")
    required_protocol = {
        "analysis_tier",
        "analysis_version",
        "analysis_method",
        "paper_relation",
        "base_model",
        "adapter_training_protocol",
        "split_protocol",
        "prompt_banks",
        "language_layers",
        "n_pairs",
        "capture",
    }
    missing = sorted(required_protocol - set(protocol))
    if missing:
        raise ValueError(f"{path} protocol is missing required fields: {', '.join(missing)}.")
    if protocol["analysis_tier"] != "primary":
        raise ValueError(f"{path} protocol tier does not match its primary bundle label.")
    if protocol["analysis_method"] != "matched_token_ft_shift_extension_v1":
        raise ValueError(f"{path} is not the registered matched-token RQ1 extension.")
    if not str(protocol["analysis_version"] or "").strip():
        raise ValueError(f"{path} does not identify an analysis version.")
    paper_relation = protocol["paper_relation"]
    if not isinstance(paper_relation, dict) or paper_relation.get("claim") != (
        "rq1_extension_not_paper_geometry_reproduction"
    ):
        raise ValueError(f"{path} does not correctly scope this as an RQ1 extension.")
    base_model = protocol["base_model"]
    if not isinstance(base_model, dict) or not str(base_model.get("id") or "").strip():
        raise ValueError(f"{path} does not bind base model identity.")
    adapter_protocol = protocol["adapter_training_protocol"]
    if not isinstance(adapter_protocol, dict):
        raise ValueError(f"{path} does not bind adapter training protocol.")
    missing_adapter = {
        "base_model",
        "base_model_revision",
        "dataset_id",
        "dataset_revision",
        "n_samples",
        "lora_rank",
        "lora_alpha",
        "lr",
        "epochs",
        "completion_only_loss",
        "loss_scope",
        "target_modules",
        "bf16",
    } - set(adapter_protocol)
    if missing_adapter:
        raise ValueError(
            f"{path} adapter protocol is missing: {', '.join(sorted(missing_adapter))}."
        )
    split_protocol = protocol["split_protocol"]
    if not isinstance(split_protocol, dict) or split_protocol.get("mode") != "hf":
        raise ValueError(f"{path} primary bundle does not use an HF-backed frozen split.")
    if split_protocol.get("artifact_version") != 3:
        raise ValueError(f"{path} primary bundle does not use a verified v3 split protocol.")
    missing_split = {
        "artifact_version",
        "source",
        "counts",
        "extraction_modality",
        "eval_modality",
    } - set(split_protocol)
    if missing_split:
        raise ValueError(
            f"{path} split protocol is missing: {', '.join(sorted(missing_split))}."
        )
    if not isinstance(protocol["language_layers"], list) or not protocol["language_layers"]:
        raise ValueError(f"{path} does not bind language layers.")
    if not isinstance(protocol["n_pairs"], int) or protocol["n_pairs"] < 50:
        raise ValueError(f"{path} does not bind the primary minimum of 50 matched pairs.")
    capture = protocol["capture"]
    if not isinstance(capture, dict) or capture.get("pairing") != (
        "same_prompt_text_only_and_image_conditioned_v1"
    ):
        raise ValueError(f"{path} does not use the registered matched-prompt capture design.")
    if capture.get("expected_image_soft_token_count") != 256:
        raise ValueError(f"{path} does not bind Gemma's 256 image-soft-token capture contract.")
    if capture.get("bootstrap_unit") != "matched_prompt_image_pair":
        raise ValueError(f"{path} does not use prompt-paired bootstrap resampling.")
    if capture.get("image_selection") != (
        "all_ordered_multimodal_rows_in_frozen_extraction_role"
    ):
        raise ValueError(f"{path} does not bind the registered frozen image selection.")
    prompt_banks = protocol["prompt_banks"]
    if not isinstance(prompt_banks, dict):
        raise ValueError(f"{path} does not bind reviewed prompt banks.")
    for name in ("em_primary", "control"):
        bank = prompt_banks.get(name)
        if not isinstance(bank, dict):
            raise ValueError(f"{path} primary bundle is missing the {name} prompt bank.")
        _sha256(bank.get("manifest_sha256"), label=f"{path}.{name}.manifest_sha256")
        _sha256(bank.get("review_metadata_sha256"), label=f"{path}.{name}.review_metadata_sha256")
        _sha256(bank.get("selected_prompt_sha256"), label=f"{path}.{name}.selected_prompt_sha256")

    provenance = bundle.get("run_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path} has no run_provenance object.")
    if provenance.get("seed") != seed:
        raise ValueError(f"{path} run_provenance seed does not match run.seed.")
    for field in (
        "adapter_reproduction_manifest_sha256",
        "adapter_run_metadata_sha256",
        "adapter_fingerprint",
        "split_manifest_sha256",
        "behavioral_review_summary_sha256",
        "review_provenance_sha256",
        "image_probe_manifest_sha256",
    ):
        _sha256(provenance.get(field), label=f"{path}.run_provenance.{field}")
    run_fingerprint = _sha256(bundle.get("run_fingerprint"), label=f"{path}.run_fingerprint")
    if (
        _digest({"protocol_fingerprint": fingerprint, "run_provenance": provenance})
        != run_fingerprint
    ):
        raise ValueError(f"{path} run_fingerprint does not match its run provenance.")

    review = bundle.get("behavioral_review")
    if not isinstance(review, dict) or review.get("behavioral_scope") != PRIMARY_SCOPE:
        raise ValueError(
            f"{path} primary bundle lacks {PRIMARY_SCOPE!r} OOD behavioral provenance."
        )
    coverage = review.get("seed_coverage")
    if not isinstance(coverage, list) or {int(value) for value in coverage} != EXPECTED_SEEDS:
        raise ValueError(f"{path} OOD review provenance does not cover seeds 42, 43, and 44.")
    evidence = review.get("ood_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{path} lacks bound broad-text, VQA, and review-sheet evidence.")
    for name in ("three_seed_gate", "selected_seed_review"):
        item = evidence.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"{path} lacks OOD evidence item {name!r}.")
        _sha256(item.get("sha256"), label=f"{path}.ood_evidence.{name}.sha256")
    return fingerprint


def _layer_summary(
    stats_with_seed: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    ordered = sorted(stats_with_seed, key=lambda item: item[0])
    cosines = [
        _geometry_value(stats, "cosine_text_image_token", "cosine_text_visual")
        for _, stats in ordered
    ]
    cis = []
    orientation_tails = []
    for _seed, stats in ordered:
        ci = stats.get("bootstrap_ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError("RQ1 geometry record has no two-sided bootstrap_ci95.")
        cis.append([float(ci[0]), float(ci[1])])
        orientation_tails.append(
            _geometry_value(
                stats,
                "random_orientation_reference_tail_fraction_two_sided",
                "random_equal_norm_p_two_sided",
            )
        )
    mean, std = _mean_std(cosines)
    all_ci_positive = all(interval[0] > 0 for interval in cis)
    all_ci_negative = all(interval[1] < 0 for interval in cis)
    all_positive = all(value > 0 for value in cosines)
    all_negative = all(value < 0 for value in cosines)
    if all_ci_positive and all_positive:
        outcome = "consistent_positive_alignment"
    elif all_ci_negative and all_negative:
        outcome = "consistent_negative_alignment"
    elif any(value > 0 for value in cosines) and any(value < 0 for value in cosines):
        outcome = "mixed_direction_across_seeds"
    elif any(interval[0] <= 0 <= interval[1] for interval in cis):
        outcome = "imprecise_or_unresolved"
    else:
        outcome = "inconsistent_or_unresolved"
    return {
        "per_seed": [
            {
                "seed": seed,
                "cosine_text_image_token": cosine,
                "bootstrap_ci95": ci,
                "orientation_reference_tail_fraction_two_sided": tail,
            }
            for (seed, _stats), cosine, ci, tail in zip(
                ordered, cosines, cis, orientation_tails, strict=True
            )
        ],
        "per_seed_cosines": cosines,
        "mean_cosine": mean,
        "std_cosine": std,
        "all_bootstrap_ci_positive": all_ci_positive,
        "all_bootstrap_ci_negative": all_ci_negative,
        "same_sign_across_seeds": all_positive or all_negative,
        "geometry_decision": outcome,
        "interpretation": (
            "Bootstrap intervals support descriptive cross-modal direction alignment; "
            "the equal-norm random-direction tail fraction is an orientation reference, "
            "not causal significance."
        ),
    }


def aggregate_bundles(
    paths: list[Path], *, alpha: float = 0.05, require_protocol: bool = False
) -> dict[str, Any]:
    """Apply the pre-specified three-seed RQ1 extension decision rule.

    The callable retains ``require_protocol=False`` only for legacy unit-test
    compatibility. The CLI defaults to fail-closed primary-bundle validation.
    """

    if not math.isclose(alpha, 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "The registered RQ1 protocol fixes the two-sided bootstrap interval "
            "alpha at 0.05. Changing it requires a new analysis version."
        )

    seen_seeds: set[int] = set()
    fingerprints: set[str] = set()
    run_fingerprints: set[str] = set()
    by_layer: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    control_by_layer: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    modern_bundle_count = 0
    for path in paths:
        bundle = json.loads(path.read_text())
        if not isinstance(bundle, dict):
            raise ValueError(f"RQ1 bundle must be a JSON object: {path}")
        seed = int(bundle["run"]["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed {seed}: one RQ1 bundle per seed is required.")
        seen_seeds.add(seed)
        if "protocol" in bundle or "protocol_fingerprint" in bundle:
            modern_bundle_count += 1
        if require_protocol:
            fingerprint = _validate_primary_bundle(bundle, path=path, seed=seed)
            fingerprints.add(fingerprint)
            run_fingerprints.add(str(bundle["run_fingerprint"]))
        geometry = bundle.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"RQ1 bundle has no geometry mapping: {path}")
        for layer, stats in geometry.items():
            if not isinstance(stats, dict):
                raise ValueError(f"RQ1 geometry stats for {layer} must be an object.")
            by_layer[layer].append((seed, stats))
        controls = bundle.get("control_geometry")
        if controls is not None:
            if not isinstance(controls, dict):
                raise ValueError(f"RQ1 control_geometry must be an object when present: {path}")
            for layer, stats in controls.items():
                if not isinstance(stats, dict):
                    raise ValueError(f"RQ1 control stats for {layer} must be an object.")
                control_by_layer[layer].append((seed, stats))
    if seen_seeds != EXPECTED_SEEDS:
        raise ValueError(f"RQ1 requires exactly seeds 42, 43, and 44; found {sorted(seen_seeds)}.")
    if require_protocol:
        if len(fingerprints) != 1:
            raise ValueError(
                "RQ1 primary bundles have incompatible protocol fingerprints; do not aggregate "
                "different base/adapter/split/probe/layer/n/version protocols."
            )
        if len(run_fingerprints) != 3:
            raise ValueError("RQ1 primary bundles must have three distinct run fingerprints.")
    elif 0 < modern_bundle_count < len(paths):
        raise ValueError("Do not mix legacy and protocol-bound RQ1 bundles in one aggregation.")

    layers: dict[str, Any] = {}
    for layer, stats_list in sorted(by_layer.items()):
        if len(stats_list) != 3:
            raise ValueError(f"Layer {layer} is missing a seed bundle.")
        layers[layer] = _layer_summary(stats_list)
    control_layers: dict[str, Any] | None = None
    if control_by_layer:
        control_layers = {}
        for layer, stats_list in sorted(control_by_layer.items()):
            if len(stats_list) != 3:
                raise ValueError(f"Control layer {layer} is missing a seed bundle.")
            control_layers[layer] = _layer_summary(stats_list)
    elif require_protocol:
        raise ValueError("Primary RQ1 aggregation requires reviewed control-geometry bundles.")

    return {
        "seeds": sorted(seen_seeds),
        "bootstrap_interval_alpha": alpha,
        "orientation_reference_cutoff_used_for_decision": False,
        "protocol_validation": (
            "primary_protocol_verified" if require_protocol else "legacy_or_caller_unverified"
        ),
        "protocol_fingerprint": next(iter(fingerprints)) if fingerprints else None,
        "decision_rule": (
            "A layer has consistent positive alignment only if all three matched-prompt "
            "bootstrap confidence intervals have positive lower bounds and all observed "
            "cosines are positive. Equal-norm random-direction tail fractions are reported "
            "as orientation references, not causal significance tests."
        ),
        "layers": layers,
        "control_layers": control_layers,
        "scope": (
            "Matched-token shared-language-residual geometry extension only; it does not "
            "reproduce the upstream final-token/SVD geometry or establish a causal shared "
            "mechanism."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundles",
        nargs=3,
        type=Path,
        help="RQ1 geometry JSON bundles for seeds 42, 43, 44.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Registered two-sided bootstrap alpha; fixed at 0.05.",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Allow legacy, unsealed bundles. Never use this for a primary RQ1 conclusion.",
    )
    args = parser.parse_args()
    result = aggregate_bundles(
        args.bundles,
        alpha=args.alpha,
        require_protocol=not args.allow_legacy,
    )
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite an RQ1 aggregate: {args.out}.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
