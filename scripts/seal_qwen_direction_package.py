#!/usr/bin/env python3
"""Seal and replay a prepared Qwen text or vision direction package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.cross_pathway import (  # noqa: E402
    DIRECTION_METADATA_FILENAME,
    RUN_METADATA_FILENAME,
    load_direction_package,
    write_direction_package_manifest,
)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--pathway", choices=("text", "vision"), required=True)
    args = parser.parse_args()

    root = args.package_dir.expanduser().resolve()
    run = _read_json(root / RUN_METADATA_FILENAME)
    metadata = _read_json(root / DIRECTION_METADATA_FILENAME)
    adapter = run.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("run_metadata.json lacks its adapter identity.")
    write_direction_package_manifest(
        root,
        pathway=args.pathway,
        adapter_fingerprint=str(adapter.get("fingerprint") or ""),
        training_seed=adapter.get("training_seed"),
        hidden_size=metadata.get("hidden_size"),
        run_fingerprint=str(run.get("run_fingerprint") or ""),
    )
    package = load_direction_package(root, expected_pathway=args.pathway)
    print(
        json.dumps(
            {
                "status": "VALID",
                "pathway": package.pathway,
                "package_dir": str(package.root),
                "package_fingerprint": package.package_fingerprint,
                "adapter_fingerprint": package.adapter_fingerprint,
                "training_seed": package.training_seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
