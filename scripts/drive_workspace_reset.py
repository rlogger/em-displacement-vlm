#!/usr/bin/env python3
"""Dry-run or reversibly archive one Colab Drive experiment seed package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.maintenance import (  # noqa: E402
    apply_archive_plan,
    build_archive_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--model-family", choices=("gemma3", "qwen2_5_vl"), required=True
    )
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--timestamp", help="UTC archive timestamp: YYYYMMDDTHHMMSSZ")
    parser.add_argument("--include-downstream", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Apply the displayed archive plan.")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()

    plan = build_archive_plan(
        args.project_root,
        model_family=args.model_family,
        seed=args.seed,
        include_downstream=args.include_downstream,
        timestamp=args.timestamp,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    if not args.apply:
        print("DRY_RUN: no file was moved or deleted.")
        print("Required confirmation:", plan.confirmation)
        return 0
    ledger = apply_archive_plan(plan, confirmation=args.confirmation)
    print("Archived without deletion. Restore ledger:", ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
