#!/usr/bin/env python3
"""Upload a Drive-backed, sanity-checked adapter to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--public", action="store_true", help="Make the destination Hub repo public."
    )
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.expanduser().resolve()
    required = ("adapter_config.json", "spec.json", "run_metadata.json")
    missing = [name for name in required if not (adapter_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"{adapter_dir} is not a complete saved adapter; missing: {', '.join(missing)}."
        )

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload sanity-checked FT_R32 adapter",
    )
    metadata = json.loads((adapter_dir / "run_metadata.json").read_text())
    print(
        json.dumps(
            {
                "status": "PUSHED",
                "repo": f"https://huggingface.co/{args.repo_id}",
                "adapter_dir": str(adapter_dir),
                "run": metadata.get("run", {}).get("run"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
