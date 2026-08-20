"""Reversible, provenance-aware maintenance for Colab Drive workspaces.

The public API in this module is deliberately narrow: it can inventory one
registered model-family/seed package and move that coherent package into a
timestamped archive.  It never deletes experiment artifacts and never treats a
runtime clone as durable evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARCHIVE_SCHEMA_VERSION = 1
REGISTERED_SEEDS = (42, 43, 44)
REGISTERED_FAMILIES = ("gemma3", "qwen2_5_vl")


@dataclass(frozen=True)
class ArchiveEntry:
    """One exact source selected for archival."""

    relative_path: str
    kind: str
    size_bytes: int
    sha256: str
    downstream: bool


@dataclass(frozen=True)
class ArchivePlan:
    """Immutable plan shown to the user before any Drive mutation."""

    schema_version: int
    project_root: str
    model_family: str
    seed: int
    timestamp_utc: str
    destination: str
    include_downstream: bool
    entries: tuple[ArchiveEntry, ...]
    plan_fingerprint: str
    confirmation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_inventory(path: Path) -> tuple[int, str]:
    """Return total bytes and a path/content digest without following links."""

    records: list[dict[str, Any]] = []
    total = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            raise ValueError(f"Archive source contains a symlink: {child}")
        if child.is_dir():
            records.append({"path": relative, "kind": "directory"})
            continue
        if not child.is_file():
            raise ValueError(f"Archive source has an unsupported file type: {child}")
        size = child.stat().st_size
        total += size
        records.append(
            {
                "path": relative,
                "kind": "file",
                "size_bytes": size,
                "sha256": _sha256_file(child),
            }
        )
    serialized = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return total, _sha256_bytes(serialized)


def _inventory_entry(root: Path, path: Path, *, downstream: bool) -> ArchiveEntry:
    if path.is_symlink():
        raise ValueError(f"Archive source cannot be a symlink: {path}")
    relative = path.relative_to(root).as_posix()
    if path.is_file():
        return ArchiveEntry(
            relative_path=relative,
            kind="file",
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
            downstream=downstream,
        )
    if path.is_dir():
        size, digest = _directory_inventory(path)
        return ArchiveEntry(
            relative_path=relative,
            kind="directory",
            size_bytes=size,
            sha256=digest,
            downstream=downstream,
        )
    raise ValueError(f"Archive source has an unsupported file type: {path}")


def _normalise_family(model_family: str) -> str:
    family = model_family.strip().casefold().replace("-", "_")
    aliases = {
        "qwen2.5_vl": "qwen2_5_vl",
        "qwen2_5vl": "qwen2_5_vl",
        "qwen2_5_vl_3b": "qwen2_5_vl",
    }
    family = aliases.get(family, family)
    if family not in REGISTERED_FAMILIES:
        raise ValueError(
            f"model_family must be one of {list(REGISTERED_FAMILIES)}, not {model_family!r}."
        )
    return family


def validate_project_root(project_root: Path, *, model_family: str) -> Path:
    """Resolve and constrain the exact persistent project root."""

    family = _normalise_family(model_family)
    requested = project_root.expanduser()
    if not requested.is_absolute():
        raise ValueError("Drive project root must be an explicit absolute path.")
    if not requested.exists() or not requested.is_dir():
        raise ValueError(f"Drive project root does not exist: {requested}")
    lexical = Path(os.path.abspath(requested))
    resolved = requested.resolve(strict=True)
    if lexical != resolved:
        raise ValueError(
            f"Drive project root traverses a symlink or alias: {requested} -> {resolved}"
        )
    forbidden = {
        Path("/"),
        Path("/content"),
        Path("/content/drive"),
        Path("/content/drive/MyDrive"),
    }
    if resolved in forbidden:
        raise ValueError(f"Refusing broad Drive root: {resolved}")
    expected_name = {
        "gemma3": "em-displacement-vlm",
        "qwen2_5_vl": "em-displacement-vlm-qwen2-5-vl-3b",
    }[family]
    if resolved.name != expected_name:
        raise ValueError(
            f"{family} requires a root named {expected_name!r}; got {resolved.name!r}."
        )
    return resolved


def _family_paths(model_family: str, seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if model_family == "gemma3":
        primary = (
            f"checkpoints/training/FT_R32_gemma3_faces_seed{seed}",
            f"checkpoints/FT_R32_gemma3_faces_seed{seed}",
            f"runs/environment_ft_seed{seed}.json",
            f"runs/reproduce_mft_gemma3_r32_seed{seed}.yaml",
            f"runs/verify_mft_gemma3_seed{seed}_bf16.yaml",
            f"runs/environment_candidate_review_seed{seed}.json",
            f"runs/verify_base_gemma3_seed{seed}_bf16.yaml",
            f"results/sanity_checks_verify_mft_gemma3_seed{seed}_bf16.json",
            f"results/sanity_checks_verify_mft_gemma3_seed{seed}_bf16.meta.json",
            f"results/sanity_checks_verify_base_gemma3_seed{seed}_bf16.json",
            f"results/sanity_checks_verify_base_gemma3_seed{seed}_bf16.meta.json",
            f"results/review_seed{seed}.csv",
            f"results/review_seed{seed}_mapping.json",
            f"results/review_seed{seed}_completed.csv",
            f"results/review_seed{seed}_summary.json",
        )
        downstream = (
            f"results/ood/seed{seed}",
            "results/ood/ood_three_seed_gate.json",
            f"results/rq1/seed{seed}",
            "results/rq1/rq1_three_seed_summary.json",
            f"results/rq1_plumbing_seed{seed}",
            f"runs/eval_ood_em_seed{seed}.yaml",
            f"runs/extract_rq1_primary_seed{seed}.yaml",
            f"runs/extract_rq1_plumbing_seed{seed}.yaml",
        )
    else:
        slug = "qwen2_5_vl_3b"
        primary = (
            f"checkpoints/training/FT_R32_{slug}_faces_seed{seed}",
            f"checkpoints/FT_R32_{slug}_faces_seed{seed}",
            f"runs/environment_ft_{slug}_seed{seed}.json",
            f"runs/reproduce_mft_{slug}_r32_seed{seed}.yaml",
            f"runs/sanity_{slug}_base_seed{seed}.yaml",
            f"runs/sanity_{slug}_ft_seed{seed}.yaml",
            f"results/sanity_checks_sanity_{slug}_base_seed{seed}.json",
            f"results/sanity_checks_sanity_{slug}_base_seed{seed}.meta.json",
            f"results/sanity_checks_sanity_{slug}_ft_seed{seed}.json",
            f"results/sanity_checks_sanity_{slug}_ft_seed{seed}.meta.json",
            f"results/review_{slug}_seed{seed}.csv",
            f"results/review_{slug}_seed{seed}_mapping.json",
            f"results/review_{slug}_seed{seed}_completed.csv",
            f"results/review_{slug}_seed{seed}_summary.json",
        )
        downstream = (
            f"results/ood/seed{seed}",
            "results/ood/ood_three_seed_gate.json",
            f"results/rq1/seed{seed}",
            "results/rq1/rq1_three_seed_summary.json",
        )
    return primary, downstream


def _timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = value.strip()
    try:
        parsed = datetime.strptime(candidate, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=UTC
        )
    except ValueError as exc:
        raise ValueError("timestamp must use UTC form YYYYMMDDTHHMMSSZ.") from exc
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def build_archive_plan(
    project_root: Path,
    *,
    model_family: str,
    seed: int,
    include_downstream: bool = False,
    timestamp: str | None = None,
) -> ArchivePlan:
    """Build a zero-mutation plan for one coherent family/seed package."""

    family = _normalise_family(model_family)
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f"seed must be one of {list(REGISTERED_SEEDS)}, not {seed}.")
    root = validate_project_root(project_root, model_family=family)
    timestamp_utc = _timestamp(timestamp)
    destination = root / "archive" / "runs" / family / f"seed{seed}" / timestamp_utc
    if destination.exists():
        raise ValueError(f"Archive destination already exists: {destination}")
    destination.resolve(strict=False).relative_to(root)

    primary_paths, downstream_paths = _family_paths(family, seed)
    downstream_existing = [
        root / relative for relative in downstream_paths if (root / relative).exists()
    ]
    if downstream_existing and not include_downstream:
        listed = ", ".join(path.relative_to(root).as_posix() for path in downstream_existing)
        raise ValueError(
            "Downstream OOD/RQ1 evidence exists and would be orphaned. Re-run only after "
            f"reviewing it with include_downstream=True: {listed}"
        )

    selected = [(relative, False) for relative in primary_paths]
    if include_downstream:
        selected.extend((relative, True) for relative in downstream_paths)
    entries = tuple(
        _inventory_entry(root, root / relative, downstream=downstream)
        for relative, downstream in selected
        if (root / relative).exists()
    )
    if not entries:
        raise ValueError(f"No registered {family} seed {seed} artifacts exist under {root}.")

    fingerprint_payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "project_root": str(root),
        "model_family": family,
        "seed": seed,
        "timestamp_utc": timestamp_utc,
        "destination": str(destination),
        "include_downstream": include_downstream,
        "entries": [asdict(entry) for entry in entries],
    }
    fingerprint = _sha256_bytes(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    confirmation = f"ARCHIVE {family} seed{seed} {fingerprint[:16]}"
    return ArchivePlan(
        schema_version=ARCHIVE_SCHEMA_VERSION,
        project_root=str(root),
        model_family=family,
        seed=seed,
        timestamp_utc=timestamp_utc,
        destination=str(destination),
        include_downstream=include_downstream,
        entries=entries,
        plan_fingerprint=fingerprint,
        confirmation=confirmation,
    )


def apply_archive_plan(plan: ArchivePlan, *, confirmation: str) -> Path:
    """Apply a previously displayed plan transactionally and write a ledger."""

    if confirmation != plan.confirmation:
        raise ValueError(f"confirmation must equal {plan.confirmation!r}.")
    root = validate_project_root(
        Path(plan.project_root), model_family=plan.model_family
    )
    rebuilt = build_archive_plan(
        root,
        model_family=plan.model_family,
        seed=plan.seed,
        include_downstream=plan.include_downstream,
        timestamp=plan.timestamp_utc,
    )
    if rebuilt != plan:
        raise ValueError(
            "Archive plan or source inventory changed after the dry run; rebuild the plan."
        )
    destination = Path(plan.destination)
    destination.resolve(strict=False).relative_to(root)
    if destination.exists():
        raise ValueError(f"Archive destination already exists: {destination}")
    moved: list[tuple[Path, Path]] = []
    try:
        destination.mkdir(parents=True, exist_ok=False)
        for entry in plan.entries:
            source = root / entry.relative_path
            target = destination / entry.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved.append((source, target))
        ledger = {
            **plan.to_dict(),
            "applied_at_utc": datetime.now(UTC).isoformat(),
            "operation": "reversible_archive_move",
            "deletion_performed": False,
            "restore": [
                {
                    "archived": str(target),
                    "restore_to": str(source),
                    "sha256": entry.sha256,
                }
                for entry, (source, target) in zip(plan.entries, moved, strict=True)
            ],
        }
        ledger_path = destination / "archive_ledger.json"
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    except Exception:
        for source, target in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not source.exists():
                target.rename(source)
        raise
    return destination / "archive_ledger.json"
