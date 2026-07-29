#!/usr/bin/env python3
"""Create a blinded two-reviewer calibration sheet for one OOD judge run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.evals.ood_review import (
    calibration_template_rows,
    write_calibration_template,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--ft-bundle", type=Path, required=True)
    parser.add_argument("--pair-package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--judge-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mapping-out", type=Path, required=True)
    args = parser.parse_args()

    rows, mapping = calibration_template_rows(
        base_bundle_path=args.base_bundle,
        ft_bundle_path=args.ft_bundle,
        pair_artifact_path=args.pair_package,
        manifest_path=args.manifest,
        image_root=args.image_root,
        judge_summary_path=args.judge_summary,
    )
    csv_path, mapping_path = write_calibration_template(
        rows,
        mapping,
        csv_path=args.out,
        mapping_path=args.mapping_out,
    )
    print(
        json.dumps(
            {
                "calibration_sheet": str(csv_path),
                "private_mapping": str(mapping_path),
                "n_rows": len(rows),
                "next_step": "Have two independent reviewers complete every row.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
