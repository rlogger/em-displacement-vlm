#!/usr/bin/env python3
"""Print a read-only, provenance-checked results audit as JSON or Markdown."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate_rq1 import aggregate_bundles  # noqa: E402

from em_displacement_vlm.results_audit import (  # noqa: E402
    audit_drive_artifacts,
    audit_public_artifacts,
    build_results_report,
    load_external_registry,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "protocols" / "external_artifacts.yaml",
    )
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument(
        "--model-family", choices=("gemma3", "qwen2_5_vl"), default="gemma3"
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()

    registry = load_external_registry(args.registry)
    public = audit_public_artifacts(registry, token=os.environ.get("HF_TOKEN"))
    drive = None
    if args.drive_root is not None:
        if not args.drive_root.is_dir():
            raise SystemExit(f"Drive root does not exist: {args.drive_root}")
        drive = audit_drive_artifacts(
            args.drive_root,
            model_family=args.model_family,
            rq1_aggregator=lambda paths: aggregate_bundles(paths, require_protocol=True),
        )
    report = build_results_report(public=public, drive=drive)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
