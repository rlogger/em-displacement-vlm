"""Artifact-bound Qwen cross-pathway geometry and causal-screen helpers.

This module deliberately operates on directions in one common coordinate
system: the output of Qwen2.5-VL's language decoder block at layer 13.  A
direction package is accepted only when its construction tensors reproduce the
saved unit vector and every supporting artifact is hash-bound.  The statistics
and intervention summaries produced here are screen-tier evidence; they are not
BLOCK-EM, displacement, or human-safety conclusions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from em_displacement_vlm.constants import (
    QWEN2_5_VL_3B_MODEL_ID,
    QWEN2_5_VL_3B_REVISION,
)
from em_displacement_vlm.vision_validation import resolve_qwen_language_blocks

Pathway = Literal["text", "vision"]
InterventionSite = Literal["text", "image"]

DIRECTION_PACKAGE_SCHEMA = "qwen-cross-pathway-direction-package-v1"
CROSS_PATHWAY_GEOMETRY_SCHEMA = "qwen-cross-pathway-geometry-screen-v1"
CROSS_PATHWAY_CAUSAL_SCHEMA = "qwen-cross-pathway-causal-screen-v1"
MODEL_FAMILY = "qwen2_5_vl"
REGISTERED_LAYER = 13
REGISTERED_HIDDEN_SIZE = 2048
REGISTERED_RESIDUAL_SITE = "qwen_language_decoder_block_output"
REGISTERED_HOOK_SEMANTICS = "post_decoder_block_output"
REGISTERED_ORIENTATION = "unsafe_minus_safe"

DIRECTION_FILENAME = "directions.safetensors"
CONSTRUCTION_FILENAME = "construction_activations.safetensors"
DIRECTION_METADATA_FILENAME = "direction_metadata.json"
RUN_METADATA_FILENAME = "run_metadata.json"
SUMMARY_FILENAME = "summary.json"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
PACKAGE_MANIFEST_FILENAME = "direction_package.json"

PACKAGE_ARTIFACT_FILENAMES = (
    DIRECTION_FILENAME,
    CONSTRUCTION_FILENAME,
    DIRECTION_METADATA_FILENAME,
    RUN_METADATA_FILENAME,
    SUMMARY_FILENAME,
    SOURCE_MANIFEST_FILENAME,
)
PATHWAY_DIRECTION_KEYS: dict[Pathway, str] = {
    "text": "text_direction",
    "vision": "vision_direction",
}
PATHWAY_CONSTRUCTION_KEYS: dict[Pathway, tuple[str, ...]] = {
    "text": ("text_paired_deltas",),
    "vision": ("vision_safe_activations", "vision_unsafe_activations"),
}

REAL_ARM_CONDITIONS = (
    "text_direction__text_site",
    "text_direction__image_site",
    "vision_direction__text_site",
    "vision_direction__image_site",
)
REAL_BOTH_CONDITION = "own_path_both"
RANDOM_CONTROL_CONDITIONS = (
    "random_text_direction__text_site",
    "random_text_direction__image_site",
    "random_vision_direction__text_site",
    "random_vision_direction__image_site",
)
RANDOM_BOTH_CONDITION = "random_both_own"
ALL_CAUSAL_CONDITIONS = (
    "baseline",
    *REAL_ARM_CONDITIONS,
    REAL_BOTH_CONDITION,
    *RANDOM_CONTROL_CONDITIONS,
    RANDOM_BOTH_CONDITION,
)

_PACKAGE_KEYS = {
    "schema_version",
    "pathway",
    "model_family",
    "base_model_id",
    "base_model_revision",
    "adapter_fingerprint",
    "training_seed",
    "layer",
    "hidden_size",
    "residual_site",
    "hook_semantics",
    "orientation",
    "run_fingerprint",
    "direction_key",
    "construction_keys",
    "artifacts",
    "package_fingerprint",
}


def canonical_json_sha256(payload: object) -> str:
    """Hash a JSON-compatible value using the repository's canonical encoding."""

    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest.")
    return text


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def _artifact_path(root: Path, filename: str) -> Path:
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Direction package lacks a regular file: {path}")
    if path.resolve().parent != root.resolve():
        raise ValueError(f"Direction package artifact escapes its root: {path}")
    return path


def _unit(vector: torch.Tensor, *, label: str) -> torch.Tensor:
    vector = vector.detach().float().cpu().flatten()
    if vector.ndim != 1 or vector.numel() != REGISTERED_HIDDEN_SIZE:
        raise ValueError(
            f"{label} must have Qwen hidden width {REGISTERED_HIDDEN_SIZE}; "
            f"got {tuple(vector.shape)}."
        )
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{label} contains NaN/Inf.")
    norm = float(vector.norm().item())
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{label} has zero or non-finite norm.")
    return vector / norm


def _require_unit_direction(vector: torch.Tensor, *, label: str) -> torch.Tensor:
    raw = vector.detach().float().cpu()
    if raw.ndim != 1 or raw.shape[0] != REGISTERED_HIDDEN_SIZE:
        raise ValueError(
            f"{label} must have shape ({REGISTERED_HIDDEN_SIZE},); got {tuple(raw.shape)}."
        )
    if not bool(torch.isfinite(raw).all()):
        raise ValueError(f"{label} contains NaN/Inf.")
    norm = float(raw.norm().item())
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"{label} must be unit normalized; observed norm={norm:.8g}.")
    return raw.contiguous()


def _require_matrix(value: torch.Tensor, *, key: str) -> torch.Tensor:
    matrix = value.detach().float().cpu()
    if matrix.ndim != 2 or matrix.shape[0] < 4:
        raise ValueError(f"{key} must be a two-dimensional matrix with at least four rows.")
    if matrix.shape[1] != REGISTERED_HIDDEN_SIZE:
        raise ValueError(
            f"{key} must have hidden width {REGISTERED_HIDDEN_SIZE}; got {matrix.shape[1]}."
        )
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{key} contains NaN/Inf.")
    return matrix.contiguous()


def _adapter_identity(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    adapter = payload.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError(f"{source}.adapter must be an object.")
    return {
        "fingerprint": _require_sha256(
            adapter.get("fingerprint"), field=f"{source}.adapter.fingerprint"
        ),
        "model_family": adapter.get("model_family"),
        "base_model_id": adapter.get("base_model_id"),
        "base_model_revision": adapter.get("base_model_revision"),
        "training_seed": adapter.get("training_seed"),
    }


def _validate_source_manifest(source_manifest: dict[str, Any]) -> str:
    claimed = _require_sha256(
        source_manifest.get("manifest_sha256"),
        field=f"{SOURCE_MANIFEST_FILENAME}.manifest_sha256",
    )
    replay = dict(source_manifest)
    replay.pop("manifest_sha256", None)
    if canonical_json_sha256(replay) != claimed:
        raise ValueError("source_manifest.json manifest_sha256 does not replay canonically.")
    return claimed


@dataclass(frozen=True)
class DirectionPackage:
    """One verified Qwen direction plus its construction evidence."""

    root: Path
    pathway: Pathway
    direction: torch.Tensor
    random_equal_norm: torch.Tensor
    construction: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    run_metadata: dict[str, Any]
    summary: dict[str, Any]
    source_manifest: dict[str, Any]
    package_manifest: dict[str, Any]

    @property
    def package_fingerprint(self) -> str:
        return str(self.package_manifest["package_fingerprint"])

    @property
    def adapter_fingerprint(self) -> str:
        return str(self.package_manifest["adapter_fingerprint"])

    @property
    def training_seed(self) -> int:
        return int(self.package_manifest["training_seed"])


def _validate_package_header(payload: Mapping[str, Any], *, expected_pathway: Pathway) -> None:
    if set(payload) != _PACKAGE_KEYS:
        missing = sorted(_PACKAGE_KEYS - set(payload))
        unknown = sorted(set(payload) - _PACKAGE_KEYS)
        raise ValueError(f"Direction package keys differ: missing={missing}, unknown={unknown}.")
    expected = {
        "schema_version": DIRECTION_PACKAGE_SCHEMA,
        "pathway": expected_pathway,
        "model_family": MODEL_FAMILY,
        "base_model_id": QWEN2_5_VL_3B_MODEL_ID,
        "base_model_revision": QWEN2_5_VL_3B_REVISION,
        "layer": REGISTERED_LAYER,
        "hidden_size": REGISTERED_HIDDEN_SIZE,
        "residual_site": REGISTERED_RESIDUAL_SITE,
        "hook_semantics": REGISTERED_HOOK_SEMANTICS,
        "orientation": REGISTERED_ORIENTATION,
        "direction_key": PATHWAY_DIRECTION_KEYS[expected_pathway],
        "construction_keys": list(PATHWAY_CONSTRUCTION_KEYS[expected_pathway]),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Direction package identity differs: {mismatches!r}.")
    if payload.get("training_seed") not in (42, 43, 44):
        raise ValueError("Direction package training_seed must be 42, 43, or 44.")
    _require_sha256(payload.get("adapter_fingerprint"), field="adapter_fingerprint")
    _require_sha256(payload.get("run_fingerprint"), field="run_fingerprint")
    claimed_fingerprint = _require_sha256(
        payload.get("package_fingerprint"), field="package_fingerprint"
    )
    replay = dict(payload)
    replay.pop("package_fingerprint", None)
    if canonical_json_sha256(replay) != claimed_fingerprint:
        raise ValueError("direction_package.json package_fingerprint is invalid.")


def _validate_artifact_hashes(
    root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(PACKAGE_ARTIFACT_FILENAMES):
        raise ValueError(
            "direction_package.json artifacts must bind exactly "
            f"{list(PACKAGE_ARTIFACT_FILENAMES)!r}."
        )
    paths: dict[str, Path] = {}
    for filename in PACKAGE_ARTIFACT_FILENAMES:
        expected = _require_sha256(artifacts.get(filename), field=f"artifacts.{filename}")
        path = _artifact_path(root, filename)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Direction package artifact hash mismatch: {filename}.")
        paths[filename] = path
    return paths


def _validate_sidecar_linkage(
    package: Mapping[str, Any],
    *,
    direction_metadata: dict[str, Any],
    run_metadata: dict[str, Any],
    summary: dict[str, Any],
    source_manifest: dict[str, Any],
) -> None:
    run_fingerprint = package["run_fingerprint"]
    for name, payload in (
        (DIRECTION_METADATA_FILENAME, direction_metadata),
        (RUN_METADATA_FILENAME, run_metadata),
        (SUMMARY_FILENAME, summary),
    ):
        if payload.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"{name} is bound to a different run_fingerprint.")

    artifacts = package["artifacts"]
    expected_direction_fields = {
        "tensor_sha256": artifacts[DIRECTION_FILENAME],
        "construction_sha256": artifacts[CONSTRUCTION_FILENAME],
        "layer": REGISTERED_LAYER,
        "hidden_size": REGISTERED_HIDDEN_SIZE,
        "residual_site": REGISTERED_RESIDUAL_SITE,
        "hook_semantics": REGISTERED_HOOK_SEMANTICS,
        "orientation": REGISTERED_ORIENTATION,
    }
    mismatches = {
        key: (direction_metadata.get(key), value)
        for key, value in expected_direction_fields.items()
        if direction_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"direction_metadata.json differs from the package: {mismatches!r}.")

    expected_identity = {
        "fingerprint": package["adapter_fingerprint"],
        "model_family": MODEL_FAMILY,
        "base_model_id": QWEN2_5_VL_3B_MODEL_ID,
        "base_model_revision": QWEN2_5_VL_3B_REVISION,
        "training_seed": package["training_seed"],
    }
    for name, payload in ((RUN_METADATA_FILENAME, run_metadata), (SUMMARY_FILENAME, summary)):
        observed = _adapter_identity(payload, source=name)
        if observed != expected_identity:
            raise ValueError(f"{name} adapter identity differs from the package.")

    manifest_sha256 = _validate_source_manifest(source_manifest)
    for name, payload in ((RUN_METADATA_FILENAME, run_metadata), (SUMMARY_FILENAME, summary)):
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{name}.manifest_sha256 differs from source_manifest.json.")

    status = summary.get("status")
    claim_boundary = summary.get("claim_boundary")
    if not isinstance(status, str) or not status.startswith("MEASURED_") or not status.endswith(
        "_SCREEN"
    ):
        raise ValueError("summary.json status must remain a MEASURED_*_SCREEN label.")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise ValueError("summary.json requires a non-empty claim_boundary.")


def _load_and_validate_tensors(
    paths: Mapping[str, Path],
    *,
    pathway: Pathway,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - required by the production lock.
        raise RuntimeError("Direction packages require safetensors.") from exc

    directions = load_file(str(paths[DIRECTION_FILENAME]), device="cpu")
    expected_direction_keys = {PATHWAY_DIRECTION_KEYS[pathway], "random_equal_norm"}
    if set(directions) != expected_direction_keys:
        raise ValueError(
            f"{DIRECTION_FILENAME} keys must be exactly {sorted(expected_direction_keys)!r}."
        )
    direction = _require_unit_direction(
        directions[PATHWAY_DIRECTION_KEYS[pathway]],
        label=PATHWAY_DIRECTION_KEYS[pathway],
    )
    random = _require_unit_direction(
        directions["random_equal_norm"],
        label="random_equal_norm",
    )

    raw_construction = load_file(str(paths[CONSTRUCTION_FILENAME]), device="cpu")
    expected_construction_keys = set(PATHWAY_CONSTRUCTION_KEYS[pathway])
    if set(raw_construction) != expected_construction_keys:
        raise ValueError(
            f"{CONSTRUCTION_FILENAME} keys must be exactly "
            f"{sorted(expected_construction_keys)!r}."
        )
    construction = {
        key: _require_matrix(raw_construction[key], key=key)
        for key in PATHWAY_CONSTRUCTION_KEYS[pathway]
    }

    if pathway == "vision":
        safe = construction["vision_safe_activations"]
        unsafe = construction["vision_unsafe_activations"]
        if safe.shape[0] != unsafe.shape[0]:
            raise ValueError("Vision safe and unsafe construction groups must have equal counts.")
        recomputed = _unit(unsafe.mean(dim=0) - safe.mean(dim=0), label="recomputed c_vis")
    else:
        deltas = construction["text_paired_deltas"]
        recomputed = _unit(deltas.mean(dim=0), label="recomputed c_text")

    if not torch.allclose(direction, recomputed, rtol=2e-4, atol=2e-4):
        signed_cosine = float(torch.dot(direction, recomputed).item())
        raise ValueError(
            "Saved direction does not reproduce from construction tensors with the registered "
            f"unsafe-minus-safe orientation; cosine={signed_cosine:.8g}."
        )
    return direction, random, construction


def _validate_package_payload(
    root: Path,
    payload: Mapping[str, Any],
    *,
    expected_pathway: Pathway,
) -> DirectionPackage:
    _validate_package_header(payload, expected_pathway=expected_pathway)
    paths = _validate_artifact_hashes(root, payload)
    direction_metadata = _read_json_object(paths[DIRECTION_METADATA_FILENAME])
    run_metadata = _read_json_object(paths[RUN_METADATA_FILENAME])
    summary = _read_json_object(paths[SUMMARY_FILENAME])
    source_manifest = _read_json_object(paths[SOURCE_MANIFEST_FILENAME])
    _validate_sidecar_linkage(
        payload,
        direction_metadata=direction_metadata,
        run_metadata=run_metadata,
        summary=summary,
        source_manifest=source_manifest,
    )
    direction, random, construction = _load_and_validate_tensors(
        paths,
        pathway=expected_pathway,
    )
    return DirectionPackage(
        root=root,
        pathway=expected_pathway,
        direction=direction,
        random_equal_norm=random,
        construction=construction,
        metadata=direction_metadata,
        run_metadata=run_metadata,
        summary=summary,
        source_manifest=source_manifest,
        package_manifest=dict(payload),
    )


def build_direction_package_manifest(
    package_dir: str | Path,
    *,
    pathway: Pathway,
    adapter_fingerprint: str,
    training_seed: int,
    hidden_size: int,
    run_fingerprint: str,
) -> dict[str, Any]:
    """Build and fully replay a package manifest after all six artifacts exist.

    The caller should write the returned object to ``direction_package.json``.
    Use :func:`write_direction_package_manifest` for immutable write-once
    behavior.
    """

    if pathway not in PATHWAY_DIRECTION_KEYS:
        raise ValueError("pathway must be exactly 'text' or 'vision'.")
    root = Path(package_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Direction package directory does not exist: {root}")
    if hidden_size != REGISTERED_HIDDEN_SIZE:
        raise ValueError(f"Qwen direction hidden_size must be {REGISTERED_HIDDEN_SIZE}.")
    _require_sha256(adapter_fingerprint, field="adapter_fingerprint")
    _require_sha256(run_fingerprint, field="run_fingerprint")
    artifacts = {
        filename: sha256_file(_artifact_path(root, filename))
        for filename in PACKAGE_ARTIFACT_FILENAMES
    }
    payload: dict[str, Any] = {
        "schema_version": DIRECTION_PACKAGE_SCHEMA,
        "pathway": pathway,
        "model_family": MODEL_FAMILY,
        "base_model_id": QWEN2_5_VL_3B_MODEL_ID,
        "base_model_revision": QWEN2_5_VL_3B_REVISION,
        "adapter_fingerprint": adapter_fingerprint,
        "training_seed": training_seed,
        "layer": REGISTERED_LAYER,
        "hidden_size": hidden_size,
        "residual_site": REGISTERED_RESIDUAL_SITE,
        "hook_semantics": REGISTERED_HOOK_SEMANTICS,
        "orientation": REGISTERED_ORIENTATION,
        "run_fingerprint": run_fingerprint,
        "direction_key": PATHWAY_DIRECTION_KEYS[pathway],
        "construction_keys": list(PATHWAY_CONSTRUCTION_KEYS[pathway]),
        "artifacts": artifacts,
    }
    payload["package_fingerprint"] = canonical_json_sha256(payload)
    _validate_package_payload(root, payload, expected_pathway=pathway)
    return payload


def write_direction_package_manifest(
    package_dir: str | Path,
    *,
    pathway: Pathway,
    adapter_fingerprint: str,
    training_seed: int,
    hidden_size: int,
    run_fingerprint: str,
) -> Path:
    """Write ``direction_package.json`` once, refusing incompatible replacement."""

    root = Path(package_dir).expanduser().resolve()
    payload = build_direction_package_manifest(
        root,
        pathway=pathway,
        adapter_fingerprint=adapter_fingerprint,
        training_seed=training_seed,
        hidden_size=hidden_size,
        run_fingerprint=run_fingerprint,
    )
    path = root / PACKAGE_MANIFEST_FILENAME
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"Refusing to replace a different direction package: {path}")
        return path
    path.write_text(rendered, encoding="utf-8")
    return path


def load_direction_package(
    package_dir: str | Path,
    *,
    expected_pathway: Pathway,
) -> DirectionPackage:
    """Load a complete, hash-bound Qwen text or vision direction package."""

    if expected_pathway not in PATHWAY_DIRECTION_KEYS:
        raise ValueError("expected_pathway must be exactly 'text' or 'vision'.")
    root = Path(package_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Direction package directory does not exist: {root}")
    package_path = _artifact_path(root, PACKAGE_MANIFEST_FILENAME)
    payload = _read_json_object(package_path)
    return _validate_package_payload(root, payload, expected_pathway=expected_pathway)


def _assert_compatible_packages(text: DirectionPackage, vision: DirectionPackage) -> None:
    if text.pathway != "text" or vision.pathway != "vision":
        raise ValueError("Cross-pathway comparison requires one text and one vision package.")
    fields = (
        "model_family",
        "base_model_id",
        "base_model_revision",
        "adapter_fingerprint",
        "training_seed",
        "layer",
        "hidden_size",
        "residual_site",
        "hook_semantics",
        "orientation",
    )
    mismatches = {
        field: (text.package_manifest.get(field), vision.package_manifest.get(field))
        for field in fields
        if text.package_manifest.get(field) != vision.package_manifest.get(field)
    }
    if mismatches:
        raise ValueError(f"Text and vision direction packages are incompatible: {mismatches!r}.")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(_unit(a, label="direction a"), _unit(b, label="direction b")).item())


def _quantile_interval(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(array, (0.025, 0.975)).tolist()
    return [float(lower), float(upper)]


def _bootstrap_geometry(
    text_deltas: torch.Tensor,
    vision_safe: torch.Tensor,
    vision_unsafe: torch.Tensor,
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values: list[float] = []
    for replicate in range(replicates):
        text_idx = torch.randint(
            text_deltas.shape[0],
            (text_deltas.shape[0],),
            generator=generator,
        )
        safe_idx = torch.randint(
            vision_safe.shape[0],
            (vision_safe.shape[0],),
            generator=generator,
        )
        unsafe_idx = torch.randint(
            vision_unsafe.shape[0],
            (vision_unsafe.shape[0],),
            generator=generator,
        )
        c_text = _unit(
            text_deltas[text_idx].mean(dim=0),
            label=f"bootstrap text direction {replicate}",
        )
        c_vis = _unit(
            vision_unsafe[unsafe_idx].mean(dim=0) - vision_safe[safe_idx].mean(dim=0),
            label=f"bootstrap vision direction {replicate}",
        )
        values.append(float(torch.dot(c_text, c_vis).item()))
    return values


def _split_half_stability(
    text_deltas: torch.Tensor,
    vision_safe: torch.Tensor,
    vision_unsafe: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def halves(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.randperm(matrix.shape[0], generator=generator)
        midpoint = matrix.shape[0] // 2
        return matrix[order[:midpoint]], matrix[order[midpoint:]]

    text_a, text_b = halves(text_deltas)
    safe_a, safe_b = halves(vision_safe)
    unsafe_a, unsafe_b = halves(vision_unsafe)
    c_text_a = _unit(text_a.mean(dim=0), label="text split A")
    c_text_b = _unit(text_b.mean(dim=0), label="text split B")
    c_vis_a = _unit(unsafe_a.mean(dim=0) - safe_a.mean(dim=0), label="vision split A")
    c_vis_b = _unit(unsafe_b.mean(dim=0) - safe_b.mean(dim=0), label="vision split B")
    cross = [float(torch.dot(c_text_a, c_vis_a)), float(torch.dot(c_text_b, c_vis_b))]
    return {
        "seed": seed,
        "text_within_pathway_cosine": float(torch.dot(c_text_a, c_text_b)),
        "vision_within_pathway_cosine": float(torch.dot(c_vis_a, c_vis_b)),
        "cross_pathway_half_cosines": cross,
        "cross_pathway_half_mean_cosine": float(np.mean(cross)),
        "all_cross_pathway_halves_same_sign": bool(cross[0] * cross[1] > 0),
    }


def _permutation_null(
    text_deltas: torch.Tensor,
    vision_safe: torch.Tensor,
    vision_unsafe: torch.Tensor,
    *,
    observed: float,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    combined = torch.cat((vision_safe, vision_unsafe), dim=0)
    n_safe = vision_safe.shape[0]
    values: list[float] = []
    for replicate in range(replicates):
        signs = torch.randint(
            0,
            2,
            (text_deltas.shape[0], 1),
            generator=generator,
            dtype=torch.int64,
        ).float()
        signs = signs.mul(2).sub(1)
        c_text = _unit(
            (text_deltas * signs).mean(dim=0),
            label=f"permuted text direction {replicate}",
        )
        order = torch.randperm(combined.shape[0], generator=generator)
        permuted_safe = combined[order[:n_safe]]
        permuted_unsafe = combined[order[n_safe:]]
        c_vis = _unit(
            permuted_unsafe.mean(dim=0) - permuted_safe.mean(dim=0),
            label=f"permuted vision direction {replicate}",
        )
        values.append(float(torch.dot(c_text, c_vis).item()))
    tail = (1 + sum(abs(value) >= abs(observed) for value in values)) / (1 + replicates)
    return {
        "method": "paired_text_sign_flip_and_vision_label_permutation",
        "seed": seed,
        "replicates": replicates,
        "mean_cosine": float(np.mean(values)),
        "std_cosine": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95": _quantile_interval(values),
        "two_sided_tail_fraction": float(tail),
        "interpretation": "label/sign permutation reference for this screen; not causal proof",
    }


def compare_cross_pathway_geometry(
    text: DirectionPackage,
    vision: DirectionPackage,
    *,
    bootstrap_seed: int = 20260823,
    bootstrap_replicates: int = 10_000,
    split_half_seed: int = 20260824,
    permutation_seed: int = 20260825,
    permutation_replicates: int = 10_000,
) -> dict[str, Any]:
    """Compare sealed layer-13 text/vision directions in their common residual space."""

    _assert_compatible_packages(text, vision)
    if bootstrap_replicates <= 0 or permutation_replicates <= 0:
        raise ValueError("Bootstrap and permutation replicate counts must be positive.")
    text_deltas = text.construction["text_paired_deltas"]
    vision_safe = vision.construction["vision_safe_activations"]
    vision_unsafe = vision.construction["vision_unsafe_activations"]
    observed = _cosine(text.direction, vision.direction)
    bootstrap = _bootstrap_geometry(
        text_deltas,
        vision_safe,
        vision_unsafe,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    null = _permutation_null(
        text_deltas,
        vision_safe,
        vision_unsafe,
        observed=observed,
        seed=permutation_seed,
        replicates=permutation_replicates,
    )
    return {
        "schema_version": CROSS_PATHWAY_GEOMETRY_SCHEMA,
        "status": "MEASURED_CROSS_PATHWAY_GEOMETRY_SCREEN",
        "claim_boundary": (
            "Descriptive layer-13 Qwen direction geometry with construction-resampling and "
            "label/sign-permutation references; not vision specificity, BLOCK-EM, removal, "
            "rerouting, displacement, or a human safety result."
        ),
        "model_family": MODEL_FAMILY,
        "base_model_id": QWEN2_5_VL_3B_MODEL_ID,
        "base_model_revision": QWEN2_5_VL_3B_REVISION,
        "adapter_fingerprint": text.adapter_fingerprint,
        "training_seed": text.training_seed,
        "layer": REGISTERED_LAYER,
        "residual_site": REGISTERED_RESIDUAL_SITE,
        "hook_semantics": REGISTERED_HOOK_SEMANTICS,
        "orientation": REGISTERED_ORIENTATION,
        "text_package_fingerprint": text.package_fingerprint,
        "vision_package_fingerprint": vision.package_fingerprint,
        "signed_cosine": observed,
        "angle_degrees": float(math.degrees(math.acos(max(-1.0, min(1.0, observed))))),
        "bootstrap": {
            "method": "paired_text_rows_and_independent_vision_groups",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "ci95_signed_cosine": _quantile_interval(bootstrap),
            "mean_signed_cosine": float(np.mean(bootstrap)),
            "bootstrap_unit": "text_pair_and_independent_safe_unsafe_images",
        },
        "split_half_stability": _split_half_stability(
            text_deltas,
            vision_safe,
            vision_unsafe,
            seed=split_half_seed,
        ),
        "permutation_null": null,
        "sample_counts": {
            "text_pairs": int(text_deltas.shape[0]),
            "vision_safe": int(vision_safe.shape[0]),
            "vision_unsafe": int(vision_unsafe.shape[0]),
        },
    }


class QwenPathwaySteeringHook(AbstractContextManager["QwenPathwaySteeringHook"]):
    """Apply one vector per requested pathway on one matching prefill forward pass.

    A call may perturb text positions, image positions, or both simultaneously.
    Decode steps are ignored because their sequence shape does not match the
    captured prefill masks and because the hook applies at most once.
    """

    def __init__(
        self,
        model: Any,
        *,
        layer: int = REGISTERED_LAYER,
        text_mask: torch.Tensor | None = None,
        text_direction: torch.Tensor | None = None,
        text_scale: float = 0.0,
        image_mask: torch.Tensor | None = None,
        image_direction: torch.Tensor | None = None,
        image_scale: float = 0.0,
    ) -> None:
        if layer != REGISTERED_LAYER:
            raise ValueError(
                f"Cross-pathway steering is registered only at layer {REGISTERED_LAYER}."
            )
        self.model = model
        self.layer = layer
        self._text = self._validate_pathway(
            "text", mask=text_mask, direction=text_direction, scale=text_scale
        )
        self._image = self._validate_pathway(
            "image", mask=image_mask, direction=image_direction, scale=image_scale
        )
        if self._text is None and self._image is None:
            raise ValueError("Steering requires a nonzero text or image perturbation.")
        if self._text is not None and self._image is not None:
            text_active_mask = self._text[0]
            image_active_mask = self._image[0]
            if text_active_mask.shape != image_active_mask.shape:
                raise ValueError("Simultaneous text/image masks must have identical shapes.")
            if bool((text_active_mask & image_active_mask).any()):
                raise ValueError("Text and image pathway masks must be disjoint.")
        self.handle: Any | None = None
        self.applied_counts = {"text": 0, "image": 0}
        self._applied = False

    @staticmethod
    def _validate_pathway(
        name: str,
        *,
        mask: torch.Tensor | None,
        direction: torch.Tensor | None,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
        if direction is None:
            if mask is not None or scale != 0.0:
                raise ValueError(f"{name} mask/scale cannot be supplied without one direction.")
            return None
        if mask is None:
            raise ValueError(f"{name} direction requires a pathway mask.")
        if mask.ndim != 2 or mask.dtype != torch.bool or not bool(mask.any()):
            raise ValueError(
                f"{name} pathway mask must be a non-empty two-dimensional bool tensor."
            )
        if direction.ndim != 1 or direction.shape[0] != REGISTERED_HIDDEN_SIZE:
            raise ValueError(
                f"{name} direction must have shape ({REGISTERED_HIDDEN_SIZE},)."
            )
        if not bool(torch.isfinite(direction).all()):
            raise ValueError(f"{name} direction contains NaN/Inf.")
        if not math.isfinite(scale) or scale == 0.0:
            raise ValueError(f"{name} steering scale must be finite and nonzero.")
        return mask.detach().clone(), direction.detach().float().cpu(), float(scale)

    def _hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("Qwen decoder block did not return a tensor.")
        active = [entry for entry in (self._text, self._image) if entry is not None]
        if self._applied or any(hidden.shape[:2] != entry[0].shape for entry in active):
            return output
        if hidden.shape[-1] != REGISTERED_HIDDEN_SIZE:
            raise ValueError("Qwen steering hidden width differs from the registered model.")
        steered = hidden
        for name, entry in (("text", self._text), ("image", self._image)):
            if entry is None:
                continue
            mask, direction, scale = entry
            device_mask = mask.to(hidden.device)
            perturbation = (scale * direction).to(hidden.device, hidden.dtype)
            steered = steered + device_mask.unsqueeze(-1).to(hidden.dtype) * perturbation.view(
                1, 1, -1
            )
            self.applied_counts[name] = int(device_mask.sum().item())
        self._applied = True
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def __enter__(self) -> QwenPathwaySteeringHook:
        _name, blocks = resolve_qwen_language_blocks(self.model, layer=self.layer)
        self.handle = blocks[self.layer].register_forward_hook(self._hook)
        return self

    def require_applied(self) -> None:
        expected = {
            "text": self._text is not None,
            "image": self._image is not None,
        }
        missing = [
            name
            for name, required in expected.items()
            if required and self.applied_counts[name] <= 0
        ]
        if not self._applied or missing:
            raise RuntimeError(
                f"Cross-pathway prefill steering did not apply: missing={missing!r}."
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@dataclass(frozen=True)
class CrossPathwayArm:
    condition: str
    direction_source: str | None
    intervention_site: InterventionSite | None
    text_direction: torch.Tensor | None = None
    text_scale: float = 0.0
    image_direction: torch.Tensor | None = None
    image_scale: float = 0.0


def build_cross_pathway_arm_specs(
    text_direction: torch.Tensor,
    vision_direction: torch.Tensor,
    random_text_direction: torch.Tensor,
    random_vision_direction: torch.Tensor,
    *,
    scale: float,
) -> tuple[CrossPathwayArm, ...]:
    """Return baseline, the real direction-by-site 2x2, and native-site controls."""

    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("Cross-pathway arm scale must be finite and nonzero.")
    vectors = {
        "text_direction": _require_unit_direction(text_direction, label="text_direction"),
        "vision_direction": _require_unit_direction(vision_direction, label="vision_direction"),
        "random_text_direction": _require_unit_direction(
            random_text_direction, label="random_text_direction"
        ),
        "random_vision_direction": _require_unit_direction(
            random_vision_direction, label="random_vision_direction"
        ),
    }
    return (
        CrossPathwayArm("baseline", None, None),
        CrossPathwayArm(
            "text_direction__text_site",
            "text_direction",
            "text",
            text_direction=vectors["text_direction"],
            text_scale=scale,
        ),
        CrossPathwayArm(
            "text_direction__image_site",
            "text_direction",
            "image",
            image_direction=vectors["text_direction"],
            image_scale=scale,
        ),
        CrossPathwayArm(
            "vision_direction__text_site",
            "vision_direction",
            "text",
            text_direction=vectors["vision_direction"],
            text_scale=scale,
        ),
        CrossPathwayArm(
            "vision_direction__image_site",
            "vision_direction",
            "image",
            image_direction=vectors["vision_direction"],
            image_scale=scale,
        ),
        CrossPathwayArm(
            REAL_BOTH_CONDITION,
            "text_and_vision_directions",
            None,
            text_direction=vectors["text_direction"],
            text_scale=scale,
            image_direction=vectors["vision_direction"],
            image_scale=scale,
        ),
        CrossPathwayArm(
            "random_text_direction__text_site",
            "random_text_direction",
            "text",
            text_direction=vectors["random_text_direction"],
            text_scale=scale,
        ),
        CrossPathwayArm(
            "random_text_direction__image_site",
            "random_text_direction",
            "image",
            image_direction=vectors["random_text_direction"],
            image_scale=scale,
        ),
        CrossPathwayArm(
            "random_vision_direction__text_site",
            "random_vision_direction",
            "text",
            text_direction=vectors["random_vision_direction"],
            text_scale=scale,
        ),
        CrossPathwayArm(
            "random_vision_direction__image_site",
            "random_vision_direction",
            "image",
            image_direction=vectors["random_vision_direction"],
            image_scale=scale,
        ),
        CrossPathwayArm(
            RANDOM_BOTH_CONDITION,
            "random_text_and_vision_directions",
            None,
            text_direction=vectors["random_text_direction"],
            text_scale=scale,
            image_direction=vectors["random_vision_direction"],
            image_scale=scale,
        ),
    )


def _paired_binary_delta(
    outcomes: Mapping[str, Mapping[str, bool]],
    *,
    reference: str,
    comparison: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    sample_ids = sorted(outcomes)
    reference_values = torch.tensor(
        [float(outcomes[sample_id][reference]) for sample_id in sample_ids],
        dtype=torch.float64,
    )
    comparison_values = torch.tensor(
        [float(outcomes[sample_id][comparison]) for sample_id in sample_ids],
        dtype=torch.float64,
    )
    differences = (comparison_values - reference_values) * 100.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bootstrap: list[float] = []
    for _ in range(replicates):
        indices = torch.randint(len(sample_ids), (len(sample_ids),), generator=generator)
        bootstrap.append(float(differences[indices].mean().item()))
    return {
        "reference": reference,
        "comparison": comparison,
        "n_samples": len(sample_ids),
        "delta_points": float(differences.mean().item()),
        "ci95_delta_points": _quantile_interval(bootstrap),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "paired_sample",
    }


def summarize_paired_cross_pathway_arms(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260826,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Summarize a complete paired baseline/2x2/control causal-screen grid."""

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive.")
    outcomes: dict[str, dict[str, bool]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        condition = row.get("condition")
        attack_success = row.get("attack_success")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Cross-pathway result row {index} lacks a sample_id.")
        if condition not in ALL_CAUSAL_CONDITIONS:
            raise ValueError(f"Cross-pathway result row {index} has an unknown condition.")
        if not isinstance(attack_success, bool):
            raise ValueError(f"Cross-pathway result row {index} attack_success must be boolean.")
        sample = outcomes.setdefault(sample_id, {})
        if condition in sample:
            raise ValueError(f"Duplicate cross-pathway row for {sample_id!r}/{condition!r}.")
        sample[str(condition)] = attack_success
    if not outcomes:
        raise ValueError("Cross-pathway result rows cannot be empty.")
    expected = set(ALL_CAUSAL_CONDITIONS)
    incomplete = {
        sample_id: sorted(expected - set(sample))
        for sample_id, sample in outcomes.items()
        if set(sample) != expected
    }
    if incomplete:
        raise ValueError(f"Cross-pathway paired grid is incomplete: {incomplete!r}.")

    condition_rates = {
        condition: {
            "n_samples": len(outcomes),
            "attack_success_rate_percent": float(
                100.0
                * np.mean([outcomes[sample_id][condition] for sample_id in sorted(outcomes)])
            ),
        }
        for condition in ALL_CAUSAL_CONDITIONS
    }

    def condition_seed(reference: str, comparison: str) -> int:
        material = f"{bootstrap_seed}\0{reference}\0{comparison}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")

    comparisons = {
        condition: _paired_binary_delta(
            outcomes,
            reference="baseline",
            comparison=condition,
            seed=condition_seed("baseline", condition),
            replicates=bootstrap_replicates,
        )
        for condition in (
            *REAL_ARM_CONDITIONS,
            REAL_BOTH_CONDITION,
            *RANDOM_CONTROL_CONDITIONS,
            RANDOM_BOTH_CONDITION,
        )
    }
    native_vs_cross = {
        "text_direction_image_minus_text_site": _paired_binary_delta(
            outcomes,
            reference="text_direction__text_site",
            comparison="text_direction__image_site",
            seed=condition_seed("text_direction__text_site", "text_direction__image_site"),
            replicates=bootstrap_replicates,
        ),
        "vision_direction_text_minus_image_site": _paired_binary_delta(
            outcomes,
            reference="vision_direction__image_site",
            comparison="vision_direction__text_site",
            seed=condition_seed("vision_direction__image_site", "vision_direction__text_site"),
            replicates=bootstrap_replicates,
        ),
    }
    return {
        "schema_version": CROSS_PATHWAY_CAUSAL_SCHEMA,
        "status": "MEASURED_CROSS_PATHWAY_CAUSAL_SCREEN",
        "claim_boundary": (
            "Paired direction-by-intervention-site screen with native-site random controls; "
            "not a human safety conclusion and not evidence of BLOCK-EM, removal, rerouting, "
            "or displacement without the separately registered training and re-discovery gates."
        ),
        "n_samples": len(outcomes),
        "conditions": condition_rates,
        "real_direction_by_site_2x2": {
            "text_direction": {
                "text_site": comparisons["text_direction__text_site"],
                "image_site": comparisons["text_direction__image_site"],
            },
            "vision_direction": {
                "text_site": comparisons["vision_direction__text_site"],
                "image_site": comparisons["vision_direction__image_site"],
            },
        },
        "real_own_path_both": comparisons[REAL_BOTH_CONDITION],
        "random_direction_by_site_controls": {
            condition: comparisons[condition] for condition in RANDOM_CONTROL_CONDITIONS
        },
        "random_both_own": comparisons[RANDOM_BOTH_CONDITION],
        "native_vs_cross_site": native_vs_cross,
        "bootstrap_seed_root": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
    }
