#!/usr/bin/env python3
"""Create a blinded CSV for human review of saved sanity bundles.

Example:
  python scripts/make_annotation_sheet.py \
    --bundle base /content/drive/.../sanity_base.json \
    --bundle ft /content/drive/.../sanity_ft.json \
    --out /content/drive/.../review_sheet.csv \
    --mapping-out /content/drive/.../review_mapping.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.evals.annotation import (  # noqa: E402
    AnnotationInput,
    build_annotation_rows,
    write_annotation_sheet,
    write_condition_mapping,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        nargs=2,
        action="append",
        metavar=("CONDITION", "PATH"),
        required=True,
        help="Repeat for each model condition, e.g. --bundle base base.json --bundle ft ft.json.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mapping-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--do-not-blind-conditions",
        action="store_true",
        help="Keep condition names in the review CSV. Use only if blinding is impossible.",
    )
    parser.add_argument(
        "--allow-legacy-unbound-bundles",
        action="store_true",
        help=(
            "Permit bundles without provenance sidecars for inspection only. "
            "Their summary cannot clear the behavioral gate."
        ),
    )
    args = parser.parse_args()
    bundles = [AnnotationInput(condition, Path(path)) for condition, path in args.bundle]
    rows, mapping = build_annotation_rows(
        bundles,
        seed=args.seed,
        blind_conditions=not args.do_not_blind_conditions,
        require_provenance=not args.allow_legacy_unbound_bundles,
    )
    write_annotation_sheet(rows, args.out)
    write_condition_mapping(mapping, args.mapping_out)
    print(f"Wrote {len(rows)} response rows: {args.out}")
    print(f"Keep this mapping hidden until review is complete: {args.mapping_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
