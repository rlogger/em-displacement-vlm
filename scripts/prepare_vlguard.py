#!/usr/bin/env python3
"""Download, parse, and seal the pinned VLGuard vision-contrast roles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.vision_validation import (
    DEFAULT_DIRECTION_PROMPT,
    DEFAULT_SELECTION_SEED,
    build_vlguard_manifest,
    download_vlguard_train,
    safe_extract_zip,
    validate_registered_vlguard_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Persistent VLGuard role root")
    parser.add_argument("--metadata", type=Path, help="Local pinned train.json override")
    parser.add_argument("--archive", type=Path, help="Local pinned train.zip override")
    parser.add_argument("--direction-per-class", type=int, default=100)
    parser.add_argument("--validation-unsafe", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--direction-prompt", default=DEFAULT_DIRECTION_PROMPT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Replay an existing manifest and image hashes without downloading or writing",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    image_root = root / "images"
    manifest_path = root / "vlguard_vision_contrast_v1.json"

    if args.validate_only:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validated = validate_registered_vlguard_manifest(payload, image_root=image_root)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "manifest": str(manifest_path),
                    "manifest_sha256": validated["manifest_sha256"],
                    "records": len(validated["records"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if bool(args.metadata) != bool(args.archive):
        raise SystemExit("--metadata and --archive must be supplied together.")
    if args.metadata:
        metadata = args.metadata.expanduser().resolve()
        archive = args.archive.expanduser().resolve()
    else:
        metadata, archive = download_vlguard_train()

    safe_extract_zip(archive, image_root)
    manifest = build_vlguard_manifest(
        metadata_path=metadata,
        archive_path=archive,
        image_root=image_root,
        direction_per_class=args.direction_per_class,
        validation_unsafe=args.validation_unsafe,
        selection_seed=args.selection_seed,
        direction_prompt=args.direction_prompt,
    )
    validate_registered_vlguard_manifest(manifest, image_root=image_root)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Refusing to replace a different manifest: {manifest_path}")
    else:
        manifest_path.write_text(rendered, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ready",
                "manifest": str(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "image_root": str(image_root),
                "records": len(manifest["records"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
