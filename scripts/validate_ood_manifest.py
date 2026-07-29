#!/usr/bin/env python3
"""Seal a reviewed paper-comparable OOD input manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.evals.ood_em import load_sealed_ood_manifest, seal_ood_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL with text and multimodal inputs.")
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--review-record",
        required=True,
        help="Durable review note, issue URL, or signed review artifact identifier.",
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Allow noncanonical counts; the result cannot satisfy the OOD EM gate.",
    )
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Seal paths/digests without reading images; not valid for execution.",
    )
    args = parser.parse_args()

    sidecar = seal_ood_manifest(
        args.manifest,
        selection_rule=args.selection_rule,
        reviewer=args.reviewer,
        review_record=args.review_record,
        exact_paper_comparable_counts=not args.pilot,
        verify_images=not args.skip_image_verification,
        image_root=args.image_root,
    )
    records, metadata = load_sealed_ood_manifest(
        args.manifest,
        require_paper_comparable=not args.pilot,
        verify_images=not args.skip_image_verification,
        image_root=args.image_root,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "sidecar": str(sidecar),
                "n_records": len(records),
                "protocol_label": metadata["protocol_label"],
                "manifest_file_sha256": metadata["manifest_file_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
