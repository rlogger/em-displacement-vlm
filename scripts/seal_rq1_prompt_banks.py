#!/usr/bin/env python3
"""Validate and review-seal distinct EM/control prompt banks for primary RQ1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MIN_PRIMARY_PROMPTS = 50


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _read_rows(path: Path) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    raw = json.loads(path.read_text())
    rows = raw.get("prompts", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a JSON list, JSONL, or {{prompts: [...]}}.")
    return rows


def _validate_bank(path: Path, *, role: str) -> tuple[set[str], tuple[str, ...]]:
    rows = _read_rows(path)
    seen_ids: set[str] = set()
    seen_pair_ids: set[str] = set()
    prompts: set[str] = set()
    pair_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{role} row {index} must be an object with ID and prompt.")
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        pair_id = str(row.get("pair_id") or "").strip()
        prompt = str(row.get("prompt") or row.get("text") or "").strip()
        if not sample_id or not pair_id or not prompt:
            raise ValueError(
                f"{role} row {index} needs a nonempty ID, pair_id, and prompt."
            )
        normalised = _normalise(prompt)
        if sample_id in seen_ids:
            raise ValueError(f"{role} repeats sample ID {sample_id!r}.")
        if normalised in prompts:
            raise ValueError(f"{role} repeats a normalized prompt.")
        if pair_id in seen_pair_ids:
            raise ValueError(f"{role} repeats pair_id {pair_id!r}.")
        seen_ids.add(sample_id)
        seen_pair_ids.add(pair_id)
        prompts.add(normalised)
        pair_ids.append(pair_id)
    if len(rows) < MIN_PRIMARY_PROMPTS:
        raise ValueError(
            f"{role} has {len(rows)} prompts; primary RQ1 requires at least "
            f"{MIN_PRIMARY_PROMPTS}."
        )
    return prompts, tuple(pair_ids)


def seal_prompt_banks(
    *,
    em_manifest: Path,
    control_manifest: Path,
    reviewed_by: str,
    reviewed_at: str,
    em_selection_policy: str,
    control_selection_policy: str,
) -> tuple[Path, Path]:
    """Create the two review sidecars consumed by the primary RQ1 loader."""
    required = {
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "em_selection_policy": em_selection_policy,
        "control_selection_policy": control_selection_policy,
    }
    missing = [field for field, value in required.items() if not value.strip()]
    if missing:
        raise ValueError(f"Review metadata is incomplete: {missing}.")
    em_prompts, em_pair_ids = _validate_bank(em_manifest, role="EM bank")
    control_prompts, control_pair_ids = _validate_bank(
        control_manifest,
        role="control bank",
    )
    em_count = len(em_pair_ids)
    control_count = len(control_pair_ids)
    if em_count != control_count:
        raise ValueError(
            f"EM/control bank sizes must match one-to-one; got {em_count} and {control_count}."
        )
    overlap = em_prompts & control_prompts
    if overlap:
        raise ValueError("EM and control banks contain overlapping normalized prompts.")
    if em_pair_ids != control_pair_ids:
        raise ValueError(
            "EM/control banks must have the same ordered pair_id values so each "
            "contrast uses the same frozen image position."
        )
    pair_id_order_sha256 = hashlib.sha256(
        json.dumps(
            {"ordered_pair_ids": em_pair_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    em_review = em_manifest.with_suffix(".review.json")
    control_review = control_manifest.with_suffix(".review.json")
    common = {
        "schema_version": 2,
        "review_status": "approved",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "matched_bank_size": em_count,
        "paired_one_to_one": True,
        "matching_schema": "explicit_ordered_pair_id_v1",
        "pair_id_order_sha256": pair_id_order_sha256,
    }
    payloads = (
        (
            em_review,
            {
                **common,
                "role": "em_primary",
                "manifest_sha256": _sha256(em_manifest),
                "selection_policy": em_selection_policy,
            },
        ),
        (
            control_review,
            {
                **common,
                "role": "control",
                "manifest_sha256": _sha256(control_manifest),
                "selection_policy": control_selection_policy,
            },
        ),
    )
    created: list[Path] = []
    try:
        for path, payload in payloads:
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Existing prompt-bank review sidecar is unreadable: {path}."
                    ) from exc
                if existing != payload:
                    raise FileExistsError(
                        f"Refusing to replace a different prompt-bank review sidecar: {path}."
                    )
                continue
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return em_review, control_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--em-manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--em-selection-policy", required=True)
    parser.add_argument("--control-selection-policy", required=True)
    args = parser.parse_args()
    em_review, control_review = seal_prompt_banks(
        em_manifest=args.em_manifest,
        control_manifest=args.control_manifest,
        reviewed_by=args.reviewed_by,
        reviewed_at=args.reviewed_at,
        em_selection_policy=args.em_selection_policy,
        control_selection_policy=args.control_selection_policy,
    )
    print(
        json.dumps(
            {"em_review": str(em_review), "control_review": str(control_review)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
