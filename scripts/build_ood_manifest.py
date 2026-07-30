#!/usr/bin/env python3
"""Build a deterministic OOD reconstruction from pinned candidate JSONLs.

This script does not download or invent the unreleased upstream evaluation
selection. It deterministically samples user-supplied, pinned broad-text and
LLaVA/MSCOCO candidate sources, records the construction provenance, and
produces an *unreviewed* manifest. Run ``validate_ood_manifest.py`` separately
to review and seal it before any model generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import EVAL_MM_N, EVAL_TEXT_N
from em_displacement_vlm.evals.ood_em import (
    OOD_BUILD_SCHEMA,
    OOD_SELECTION_ALGORITHM,
    deterministic_ood_selection,
    sha256_file,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object.")
        rows.append(row)
    return rows


def build_manifest(
    *,
    text_candidates: Path,
    multimodal_candidates: Path,
    output: Path,
    image_root: Path,
    selection_seed: int,
    n_text: int = EVAL_TEXT_N,
    n_multimodal: int = EVAL_MM_N,
) -> tuple[Path, Path]:
    """Create one unreviewed deterministic manifest and its construction record."""
    build_path = output.with_suffix(output.suffix + ".build.json")
    if output.exists() or build_path.exists():
        raise FileExistsError(f"Refusing to overwrite OOD construction artifacts: {output}")
    if n_text <= 0 or n_multimodal <= 0:
        raise ValueError("Selection counts must be positive.")

    text_rows = _read_jsonl(text_candidates)
    multimodal_rows = _read_jsonl(multimodal_candidates)
    rows = deterministic_ood_selection(
        text_rows,
        multimodal_rows,
        image_root=image_root,
        selection_seed=selection_seed,
        n_text=n_text,
        n_multimodal=n_multimodal,
    )
    selected_mm = [
        row for row in rows if row["modality"] == "multimodal"
    ]
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)

    construction = {
        "schema_version": OOD_BUILD_SCHEMA,
        "status": "unreviewed_candidate_manifest",
        "output_manifest": str(output.resolve()),
        "output_manifest_sha256": sha256_file(output),
        "selection": {
            "algorithm": OOD_SELECTION_ALGORITHM,
            "seed": int(selection_seed),
            "n_text": n_text,
            "n_multimodal": n_multimodal,
        },
        "inputs": {
            "text_candidates": str(text_candidates.resolve()),
            "text_candidates_sha256": sha256_file(text_candidates),
            "multimodal_candidates": str(multimodal_candidates.resolve()),
            "multimodal_candidates_sha256": sha256_file(multimodal_candidates),
            "image_root": str(image_root.resolve()),
        },
        "distinct_multimodal_images": len(
            {row["image_sha256"] for row in selected_mm}
        ),
        "next_required_step": "human review and scripts/validate_ood_manifest.py",
    }
    try:
        with build_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(construction, indent=2, sort_keys=True) + "\n")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output, build_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-candidates", type=Path, required=True)
    parser.add_argument("--multimodal-candidates", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260730)
    parser.add_argument("--n-text", type=int, default=EVAL_TEXT_N)
    parser.add_argument("--n-multimodal", type=int, default=EVAL_MM_N)
    args = parser.parse_args()
    output, build = build_manifest(
        text_candidates=args.text_candidates,
        multimodal_candidates=args.multimodal_candidates,
        output=args.out,
        image_root=args.image_root,
        selection_seed=args.selection_seed,
        n_text=args.n_text,
        n_multimodal=args.n_multimodal,
    )
    print(json.dumps({"manifest": str(output), "construction": str(build)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
