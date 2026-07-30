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

from em_displacement_vlm.evals.ood_em import sha256_file
from em_displacement_vlm.evals.ood_review import (
    validate_seed_review,
    validate_three_seed_gate,
)

EXPECTED_SEEDS = {42, 43, 44}
PRIMARY_SCOPE = "ood_paper_comparable"
PAIRED_CONTRAST_METHOD = "paired_prompt_index_primary_minus_control_bootstrap_v1"
PAIRED_CONTRAST_BOOTSTRAP_SAMPLES = 2_000
PAIRED_CONTRAST_ALPHA = 0.05
ACTIVATION_COSINE_ABS_TOLERANCE = 5e-3


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest.")
    return text


def _load_verified_primary_artifacts(
    bundle: dict[str, Any],
    *,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the sealed bundle and its collocated activation tensor artifact."""

    sidecar_path = path.with_suffix(".meta.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Primary RQ1 bundle is missing its metadata sidecar: {sidecar_path}."
        )
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read RQ1 metadata sidecar {sidecar_path}.") from exc
    if not isinstance(sidecar, dict) or sidecar.get("schema_version") != 1:
        raise ValueError(f"{sidecar_path} is not a supported RQ1 metadata sidecar.")

    expected_bundle_sha = _sha256(
        sidecar.get("bundle_sha256"),
        label=f"{sidecar_path}.bundle_sha256",
    )
    observed_bundle_sha = _sha256_file(path)
    if expected_bundle_sha != observed_bundle_sha:
        raise ValueError(f"{path} changed after rq1_geometry.meta.json was written.")
    for field in ("protocol_fingerprint", "run_fingerprint"):
        sidecar_digest = _sha256(sidecar.get(field), label=f"{sidecar_path}.{field}")
        bundle_digest = _sha256(bundle.get(field), label=f"{path}.{field}")
        if sidecar_digest != bundle_digest:
            raise ValueError(f"{sidecar_path} {field} does not match the RQ1 bundle.")

    activation_reference = str(bundle.get("activation_matrices") or "").strip()
    if not activation_reference:
        raise ValueError(f"{path} does not identify activation_matrices.safetensors.")
    if Path(activation_reference).name != "activation_matrices.safetensors":
        raise ValueError(
            f"{path} must bind the canonical activation_matrices.safetensors artifact."
        )
    activation_path = path.parent / "activation_matrices.safetensors"
    if not activation_path.is_file():
        raise FileNotFoundError(
            f"Primary RQ1 activation artifact is missing beside its bundle: {activation_path}."
        )
    expected_activation_sha = _sha256(
        bundle.get("activation_matrices_sha256"),
        label=f"{path}.activation_matrices_sha256",
    )
    observed_activation_sha = _sha256_file(activation_path)
    if expected_activation_sha != observed_activation_sha:
        raise ValueError(f"{activation_path} changed after the RQ1 bundle was written.")
    if bundle.get("activation_format") != "safetensors_fp16":
        raise ValueError(f"{path} does not bind the registered fp16 safetensors format.")

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - required in project dependencies.
        raise RuntimeError(
            "Primary RQ1 aggregation requires torch and safetensors to verify activations."
        ) from exc
    with safe_open(str(activation_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != "rq1_activation_matrices_fp16_v1":
        raise ValueError(f"{activation_path} has an unsupported activation format marker.")
    tensors = load_file(str(activation_path), device="cpu")
    claimed_keys = bundle.get("activation_tensor_keys")
    if not isinstance(claimed_keys, list) or not all(
        isinstance(key, str) for key in claimed_keys
    ):
        raise ValueError(f"{path} does not bind its activation tensor keys.")
    protocol_layers = bundle["protocol"]["language_layers"]
    expected_keys = {
        f"{state}__{condition}__{token_kind}__layer_{int(layer)}"
        for state in ("base", "ft")
        for condition in ("em_primary", "control")
        for token_kind in ("text", "image_token")
        for layer in protocol_layers
    }
    if len(expected_keys) != 8 * len(protocol_layers):
        raise ValueError(f"{path} protocol contains duplicate language-layer indices.")
    if set(claimed_keys) != expected_keys:
        raise ValueError(
            f"{path} activation tensor keys do not match its registered language layers."
        )
    if sorted(claimed_keys) != sorted(tensors):
        raise ValueError(f"{path} activation tensor keys do not match the sealed artifact.")
    for key, tensor in tensors.items():
        if tensor.dtype != torch.float16:
            raise ValueError(f"{activation_path}:{key} is not stored as fp16.")
        if tensor.ndim != 2 or not tensor.shape[0] or not tensor.shape[1]:
            raise ValueError(f"{activation_path}:{key} is not a non-empty matrix.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{activation_path}:{key} contains non-finite values.")

    artifact_record = {
        "bundle": str(path),
        "bundle_sha256": observed_bundle_sha,
        "metadata": str(sidecar_path),
        "metadata_sha256": _sha256_file(sidecar_path),
        "activation_matrices": str(activation_path),
        "activation_matrices_sha256": observed_activation_sha,
    }
    return tensors, artifact_record


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
        "data_selection_seed",
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
    if int(split_protocol["data_selection_seed"]) != 42:
        raise ValueError(f"{path} primary split protocol must use data_selection_seed 42.")
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
    if int(provenance.get("data_selection_seed", -1)) != 42:
        raise ValueError(f"{path} run provenance must bind data_selection_seed 42.")
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
        expected_sha = _sha256(
            item.get("sha256"),
            label=f"{path}.ood_evidence.{name}.sha256",
        )
        evidence_path = Path(str(item.get("path", ""))).expanduser().resolve()
        if not evidence_path.is_file() or sha256_file(evidence_path) != expected_sha:
            raise ValueError(f"{path} has a broken OOD evidence binding for {name!r}.")
        if name == "three_seed_gate":
            gate = validate_three_seed_gate(evidence_path, require_pass=True)
            if int(seed) not in {int(value) for value in gate["seed_coverage"]}:
                raise ValueError(f"{path} seed is absent from its three-seed OOD gate.")
            if (
                provenance.get("behavioral_review_summary_sha256") != expected_sha
                or provenance.get("review_provenance_sha256") != expected_sha
            ):
                raise ValueError(
                    f"{path} run provenance does not bind its replayed three-seed gate."
                )
        else:
            selected_review = validate_seed_review(evidence_path)
            if (
                selected_review.get("behavioral_gate") != "pass"
                or int(selected_review.get("training_seed", -1)) != seed
                or selected_review.get("adapter_fingerprint")
                != provenance.get("adapter_fingerprint")
                or selected_review.get("adapter_reproduction_manifest_sha256")
                != provenance.get("adapter_reproduction_manifest_sha256")
                or selected_review.get("split_manifest_sha256")
                != provenance.get("split_manifest_sha256")
            ):
                raise ValueError(
                    f"{path} selected OOD review does not match its adapter and split."
                )
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
        "decision_scope": "descriptive_condition_only_not_registered_rq1_conclusion",
        "interpretation": (
            "This condition-specific pattern is descriptive and cannot support the "
            "registered RQ1 conclusion without the paired primary-minus-control contrast. "
            "The equal-norm random-direction tail fraction is an orientation reference, "
            "not causal significance."
        ),
    }


def _layer_number(layer: str) -> int:
    prefix = "language_layer_"
    if not layer.startswith(prefix):
        raise ValueError(f"Unsupported RQ1 language-layer label: {layer!r}.")
    try:
        number = int(layer.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"Unsupported RQ1 language-layer label: {layer!r}.") from exc
    if number < 0:
        raise ValueError(f"RQ1 language-layer index must be non-negative: {layer!r}.")
    return number


def _cosine_rows(left: Any, right: Any, *, label: str) -> Any:
    import torch

    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(
        right, dim=-1
    )
    if bool((denominator <= torch.finfo(left.dtype).eps).any()):
        raise ValueError(f"{label} contains an undefined zero-norm cosine.")
    return (left * right).sum(dim=-1) / denominator


def _paired_bootstrap_seed(*, training_seed: int, layer: int) -> int:
    payload = f"{PAIRED_CONTRAST_METHOD}\0{training_seed}\0{layer}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _paired_seed_contrast(
    *,
    tensors: dict[str, Any],
    layer: str,
    seed: int,
    n_pairs: int,
    primary_stats: dict[str, Any],
    control_stats: dict[str, Any],
) -> dict[str, Any]:
    """Bootstrap primary-minus-control cosine with one shared row resample."""

    import torch

    layer_index = _layer_number(layer)

    def delta(condition: str, token_kind: str) -> Any:
        base_key = f"base__{condition}__{token_kind}__layer_{layer_index}"
        ft_key = f"ft__{condition}__{token_kind}__layer_{layer_index}"
        missing = [key for key in (base_key, ft_key) if key not in tensors]
        if missing:
            raise ValueError(
                f"Seed {seed} {layer} activation artifact is missing: {', '.join(missing)}."
            )
        base = tensors[base_key].float()
        ft = tensors[ft_key].float()
        if base.shape != ft.shape:
            raise ValueError(
                f"Seed {seed} {layer} base/FT activation shapes differ for "
                f"{condition}/{token_kind}."
            )
        return ft - base

    primary_text = delta("em_primary", "text")
    primary_image = delta("em_primary", "image_token")
    control_text = delta("control", "text")
    control_image = delta("control", "image_token")
    matrices = (primary_text, primary_image, control_text, control_image)
    if any(matrix.shape != primary_text.shape for matrix in matrices[1:]):
        raise ValueError(
            f"Seed {seed} {layer} requires shape-matched primary/control prompt rows."
        )
    if int(primary_text.shape[0]) != n_pairs:
        raise ValueError(
            f"Seed {seed} {layer} activation row count {primary_text.shape[0]} "
            f"does not match protocol n_pairs={n_pairs}."
        )

    primary_observed = float(
        _cosine_rows(
            primary_text.mean(dim=0, keepdim=True),
            primary_image.mean(dim=0, keepdim=True),
            label=f"seed {seed} {layer} primary",
        ).item()
    )
    control_observed = float(
        _cosine_rows(
            control_text.mean(dim=0, keepdim=True),
            control_image.mean(dim=0, keepdim=True),
            label=f"seed {seed} {layer} control",
        ).item()
    )
    for observed, stats, condition in (
        (primary_observed, primary_stats, "primary"),
        (control_observed, control_stats, "control"),
    ):
        reported = _geometry_value(
            stats,
            "cosine_text_image_token",
            "cosine_text_visual",
        )
        if not math.isclose(
            observed,
            reported,
            rel_tol=0.0,
            abs_tol=ACTIVATION_COSINE_ABS_TOLERANCE,
        ):
            raise ValueError(
                f"Seed {seed} {layer} {condition} cosine does not match the sealed "
                "fp16 activation matrices."
            )

    bootstrap_seed = _paired_bootstrap_seed(
        training_seed=seed,
        layer=layer_index,
    )
    generator = torch.Generator(device="cpu").manual_seed(bootstrap_seed)
    contrasts: list[Any] = []
    remaining = PAIRED_CONTRAST_BOOTSTRAP_SAMPLES
    chunk_size = 128
    while remaining:
        batch = min(chunk_size, remaining)
        indices = torch.randint(
            n_pairs,
            (batch, n_pairs),
            generator=generator,
            device="cpu",
        )
        counts = torch.zeros((batch, n_pairs), dtype=torch.float32)
        counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float32))
        weights = counts / float(n_pairs)
        primary_cosines = _cosine_rows(
            weights @ primary_text,
            weights @ primary_image,
            label=f"seed {seed} {layer} primary bootstrap",
        )
        control_cosines = _cosine_rows(
            weights @ control_text,
            weights @ control_image,
            label=f"seed {seed} {layer} control bootstrap",
        )
        contrasts.append(primary_cosines - control_cosines)
        remaining -= batch
    bootstrap = torch.cat(contrasts)
    quantiles = torch.quantile(
        bootstrap,
        torch.tensor(
            [PAIRED_CONTRAST_ALPHA / 2, 1 - PAIRED_CONTRAST_ALPHA / 2],
            dtype=bootstrap.dtype,
        ),
    )
    return {
        "seed": seed,
        "primary_cosine_from_sealed_activations": primary_observed,
        "control_cosine_from_sealed_activations": control_observed,
        "primary_minus_control_cosine": primary_observed - control_observed,
        "paired_bootstrap_ci95": [float(quantiles[0]), float(quantiles[1])],
        "bootstrap_method": PAIRED_CONTRAST_METHOD,
        "bootstrap_unit": "paired_primary_control_prompt_index_within_training_seed",
        "bootstrap_samples": PAIRED_CONTRAST_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": bootstrap_seed,
        "n_pairs": n_pairs,
    }


def _paired_contrast_summary(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(per_seed, key=lambda item: int(item["seed"]))
    if {int(item["seed"]) for item in ordered} != EXPECTED_SEEDS:
        raise ValueError("Paired RQ1 contrast requires exactly seeds 42, 43, and 44.")
    contrasts = [float(item["primary_minus_control_cosine"]) for item in ordered]
    intervals = [item["paired_bootstrap_ci95"] for item in ordered]
    all_ci_positive = all(float(interval[0]) > 0 for interval in intervals)
    all_ci_negative = all(float(interval[1]) < 0 for interval in intervals)
    all_positive = all(value > 0 for value in contrasts)
    all_negative = all(value < 0 for value in contrasts)
    if all_ci_positive and all_positive:
        outcome = "consistent_positive_primary_minus_control_contrast"
    elif all_ci_negative and all_negative:
        outcome = "consistent_negative_primary_minus_control_contrast"
    elif any(value > 0 for value in contrasts) and any(value < 0 for value in contrasts):
        outcome = "mixed_primary_minus_control_contrast_across_seeds"
    elif any(float(interval[0]) <= 0 <= float(interval[1]) for interval in intervals):
        outcome = "imprecise_or_unresolved_primary_minus_control_contrast"
    else:
        outcome = "inconsistent_or_unresolved_primary_minus_control_contrast"
    mean, std = _mean_std(contrasts)
    return {
        "per_seed": ordered,
        "per_seed_primary_minus_control_cosines": contrasts,
        "mean_primary_minus_control_cosine": mean,
        "std_primary_minus_control_cosine": std,
        "all_paired_bootstrap_ci_positive": all_ci_positive,
        "all_paired_bootstrap_ci_negative": all_ci_negative,
        "same_sign_across_seeds": all_positive or all_negative,
        "registered_decision": outcome,
        "decision_scope": "registered_primary_rq1_contrast",
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
    split_fingerprints: set[str] = set()
    data_selection_seeds: set[int] = set()
    by_layer: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    control_by_layer: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    activation_tensors_by_seed: dict[int, dict[str, Any]] = {}
    artifact_records_by_seed: dict[int, dict[str, Any]] = {}
    bundles_by_seed: dict[int, dict[str, Any]] = {}
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
            tensors, artifact_record = _load_verified_primary_artifacts(bundle, path=path)
            activation_tensors_by_seed[seed] = tensors
            artifact_records_by_seed[seed] = artifact_record
            fingerprints.add(fingerprint)
            run_fingerprints.add(str(bundle["run_fingerprint"]))
            provenance = bundle["run_provenance"]
            protocol = bundle["protocol"]
            split_fingerprints.add(str(provenance["split_manifest_sha256"]))
            data_selection_seeds.add(int(provenance["data_selection_seed"]))
            if int(protocol["split_protocol"]["data_selection_seed"]) != int(
                provenance["data_selection_seed"]
            ):
                raise ValueError(
                    f"{path} run data_selection_seed does not match its protocol."
                )
        bundles_by_seed[seed] = bundle
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
        if require_protocol:
            registered_layers = {
                f"language_layer_{int(layer)}" for layer in bundle["protocol"]["language_layers"]
            }
            if set(geometry) != registered_layers or set(controls or {}) != registered_layers:
                raise ValueError(
                    f"{path} geometry layers do not exactly match its registered protocol."
                )
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
        if len(split_fingerprints) != 1 or data_selection_seeds != {42}:
            raise ValueError(
                "RQ1 primary bundles must reuse one split selected with "
                "data_selection_seed 42 across training seeds."
            )
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

    contrast_layers: dict[str, Any] | None = None
    if require_protocol:
        if set(by_layer) != set(control_by_layer):
            raise ValueError(
                "Primary and control geometry must contain the same registered language layers."
            )
        contrast_layers = {}
        for layer in sorted(by_layer):
            primary_by_seed = dict(by_layer[layer])
            control_by_seed = dict(control_by_layer[layer])
            if set(primary_by_seed) != EXPECTED_SEEDS or set(control_by_seed) != EXPECTED_SEEDS:
                raise ValueError(
                    f"{layer} does not contain paired primary/control geometry for all seeds."
                )
            per_seed_contrasts = []
            for seed in sorted(EXPECTED_SEEDS):
                protocol = bundles_by_seed[seed]["protocol"]
                per_seed_contrasts.append(
                    _paired_seed_contrast(
                        tensors=activation_tensors_by_seed[seed],
                        layer=layer,
                        seed=seed,
                        n_pairs=int(protocol["n_pairs"]),
                        primary_stats=primary_by_seed[seed],
                        control_stats=control_by_seed[seed],
                    )
                )
            contrast_layers[layer] = _paired_contrast_summary(per_seed_contrasts)

    return {
        "seeds": sorted(seen_seeds),
        "bootstrap_interval_alpha": alpha,
        "orientation_reference_cutoff_used_for_decision": False,
        "protocol_validation": (
            "primary_protocol_verified" if require_protocol else "legacy_or_caller_unverified"
        ),
        "protocol_fingerprint": next(iter(fingerprints)) if fingerprints else None,
        "data_selection_seed": (
            next(iter(data_selection_seeds)) if data_selection_seeds else None
        ),
        "split_manifest_sha256": (
            next(iter(split_fingerprints)) if split_fingerprints else None
        ),
        "decision_rule": (
            "The registered layer conclusion uses the paired primary-minus-control cosine "
            "contrast, never either condition alone. Within each training seed, one shared "
            "prompt-index bootstrap resample is applied to the primary and matched-control "
            "activation rows. A consistent positive contrast requires all three observed "
            "primary-minus-control cosines and all three paired-bootstrap lower bounds to "
            "be positive. Condition-specific intervals and equal-norm orientation references "
            "remain descriptive only."
        ),
        "registered_conclusion_source": "contrast_layers" if require_protocol else None,
        "registered_conclusion_status": (
            "available_from_verified_paired_contrasts"
            if require_protocol
            else "unavailable_legacy_or_unverified_inputs"
        ),
        "layers": layers,
        "control_layers": control_layers,
        "descriptive_layers": {
            "em_primary": layers,
            "control": control_layers,
        },
        "contrast_layers": contrast_layers,
        "input_artifacts": (
            {str(seed): artifact_records_by_seed[seed] for seed in sorted(EXPECTED_SEEDS)}
            if require_protocol
            else None
        ),
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
