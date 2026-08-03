#!/usr/bin/env python3
"""Build pinned paper-comparable OOD candidate pools (broad text + VQA/COCO).

Writes Drive-ready assets expected by notebook ``03_ood_em_baseline.ipynb``:

* ``data/ood/candidates/broad_text.jsonl``
* ``data/ood/candidates/llava_mscoco_vqa.jsonl``
* ``data/ood/images/``

This does **not** invent the unreleased Gulati & Raval selections. It freezes a
deterministic reconstruction from Hugging Face ``lmms-lab/VQAv2`` validation
(MSCOCO-linked images + questions; avoids S3 403s) and a pinned instruction
text dataset, then records construction hashes. Exact paper language is never
claimed.

Optional Hub upload packages the same three artifacts as a private dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import EVAL_MM_N, EVAL_TEXT_N  # noqa: E402
from em_displacement_vlm.evals.ood_em import (  # noqa: E402
    OOD_SELECTION_ALGORITHM,
    sha256_file,
)

# Hugging Face–hosted VQA v2 validation (MSCOCO-linked images + questions).
# Avoids the official S3 questions dump, which often returns HTTP 403 from Colab.
DEFAULT_VQA_DATASET = "lmms-lab/VQAv2"
DEFAULT_VQA_SPLIT = "validation"

# Instruction-style broad prompts; default branch tip is resolved then frozen
# as ``source_revision`` inside each text row (immutable once written).
DEFAULT_TEXT_DATASET = "databricks/databricks-dolly-15k"
DEFAULT_TEXT_PROMPT_FIELD = "instruction"

DEFAULT_SELECTION_SEED = 20260730
DEFAULT_POOL_N_TEXT = 400
DEFAULT_POOL_N_MULTIMODAL = 400


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidates: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def _rank_key(selection_seed: int, *parts: str) -> str:
    payload = "\0".join([str(selection_seed), *parts]).encode()
    return hashlib.sha256(payload).hexdigest()


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_jsonl_meta_placeholder(path: Path, *, kind: str) -> dict[str, Any]:
    return {
        "reused_existing_file": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "n_rows": _count_jsonl(path),
        "kind": kind,
    }


def build_text_candidates(
    *,
    n_pool: int,
    selection_seed: int,
    dataset_id: str = DEFAULT_TEXT_DATASET,
    prompt_field: str = DEFAULT_TEXT_PROMPT_FIELD,
    dataset_revision: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample a deterministic broad-text candidate pool from a pinned HF dataset."""
    if n_pool <= 0:
        raise ValueError("n_pool must be positive.")

    from datasets import load_dataset
    from huggingface_hub import dataset_info

    info = dataset_info(dataset_id, revision=dataset_revision)
    resolved_revision = info.sha
    if not resolved_revision or resolved_revision.casefold() in {
        "main",
        "master",
        "latest",
        "unknown",
    }:
        raise RuntimeError(
            f"Could not resolve an immutable revision for {dataset_id!r}."
        )

    dataset = load_dataset(
        dataset_id,
        split="train",
        revision=resolved_revision,
    )
    ranked: list[tuple[str, int, str]] = []
    for index, row in enumerate(dataset):
        prompt = str(row.get(prompt_field) or "").strip()
        if not prompt:
            continue
        item_id = str(row.get("id") or index)
        ranked.append(
            (
                _rank_key(selection_seed, dataset_id, resolved_revision, item_id, prompt),
                index,
                prompt,
            )
        )
    ranked.sort(key=lambda item: item[0])
    if len(ranked) < n_pool:
        raise RuntimeError(
            f"Text dataset {dataset_id} only yielded {len(ranked)} usable prompts; "
            f"need ≥ {n_pool}."
        )

    rows: list[dict[str, Any]] = []
    for rank, (_digest, index, prompt) in enumerate(ranked[:n_pool]):
        rows.append(
            {
                "prompt": prompt,
                "source": f"{dataset_id}@{resolved_revision}",
                "source_dataset": dataset_id,
                "source_revision": resolved_revision,
                "source_item_id": f"dolly-train-{index}",
                "pool_rank": rank,
            }
        )
    meta = {
        "dataset_id": dataset_id,
        "dataset_revision": resolved_revision,
        "prompt_field": prompt_field,
        "n_pool": n_pool,
        "selection_seed": selection_seed,
        "selection_algorithm": OOD_SELECTION_ALGORITHM,
    }
    return rows, meta


def _save_pil_image(image: Any, dest: Path) -> str:
    """Write a HF datasets image / PIL object as JPEG and return sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return sha256_file(dest)

    # HF Image feature may be PIL already; convert if needed.
    if hasattr(image, "convert"):
        pil = image.convert("RGB")
    elif isinstance(image, dict) and "bytes" in image:
        from io import BytesIO

        from PIL import Image

        pil = Image.open(BytesIO(image["bytes"])).convert("RGB")
    else:
        from PIL import Image

        pil = Image.open(image).convert("RGB")

    tmp = dest.with_suffix(dest.suffix + ".partial")
    pil.save(tmp, format="JPEG", quality=95)
    tmp.replace(dest)
    return sha256_file(dest)


def build_multimodal_candidates(
    *,
    image_root: Path,
    n_pool: int,
    selection_seed: int,
    dataset_id: str = DEFAULT_VQA_DATASET,
    split: str = DEFAULT_VQA_SPLIT,
    dataset_revision: str | None = None,
    max_workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build LLaVA/MSCOCO-style VQA candidates from a pinned HF VQAv2 dataset.

    Official S3 dumps of the VQA questions JSON frequently 403 from Colab. The
    ``lmms-lab/VQAv2`` validation split carries question text, question/image
    IDs, and MSCOCO images, which is enough for a paper-comparable pack.
    """
    del max_workers  # sequential decode is more reliable for HF image features
    if n_pool <= 0:
        raise ValueError("n_pool must be positive.")

    from datasets import load_dataset
    from huggingface_hub import dataset_info

    info = dataset_info(dataset_id, revision=dataset_revision)
    resolved_revision = info.sha
    if not resolved_revision or resolved_revision.casefold() in {
        "main",
        "master",
        "latest",
        "unknown",
    }:
        raise RuntimeError(
            f"Could not resolve an immutable revision for {dataset_id!r}."
        )

    print(
        f"Loading {dataset_id}@{resolved_revision[:12]} split={split} "
        "(this may take a few minutes) …"
    )
    dataset = load_dataset(
        dataset_id,
        split=split,
        revision=resolved_revision,
    )

    ranked: list[tuple[str, int]] = []
    for index, row in enumerate(dataset):
        prompt = str(row.get("question") or "").strip()
        if not prompt:
            continue
        question_id = str(row.get("question_id") or index)
        image_id = str(row.get("image_id") or question_id)
        digest = _rank_key(
            selection_seed,
            dataset_id,
            resolved_revision,
            question_id,
            image_id,
            prompt,
        )
        ranked.append((digest, index))
    ranked.sort(key=lambda item: item[0])

    selected: list[tuple[int, dict[str, Any]]] = []
    chosen_images: set[str] = set()
    for _digest, index in ranked:
        row = dataset[index]
        prompt = str(row.get("question") or "").strip()
        question_id = str(row.get("question_id") or index)
        image_id = str(row.get("image_id") or question_id)
        if image_id in chosen_images:
            continue
        if row.get("image") is None:
            continue
        selected.append(
            (
                index,
                {
                    "prompt": prompt,
                    "question_id": question_id,
                    "image_id": image_id,
                },
            )
        )
        chosen_images.add(image_id)
        if len(selected) >= n_pool:
            break
    if len(selected) < n_pool:
        raise RuntimeError(
            f"Only found {len(selected)} unique-image VQA rows in {dataset_id}; "
            f"need ≥ {n_pool}."
        )

    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rank, (index, item) in enumerate(selected):
        row = dataset[index]
        image_id = item["image_id"]
        relative = f"vqa_v2/{split}/image_{image_id}.jpg"
        dest = image_root / relative
        if rank == 0 or (rank + 1) % 25 == 0:
            print(f"  materializing image {rank + 1}/{n_pool} …")
        image_sha256 = _save_pil_image(row["image"], dest)
        rows.append(
            {
                "prompt": item["prompt"],
                "image_path": relative,
                "image_sha256": image_sha256,
                "source": f"{dataset_id}@{resolved_revision}",
                "source_dataset": dataset_id,
                "source_revision": resolved_revision,
                "source_item_id": f"vqav2-q{item['question_id']}",
                "pool_rank": rank,
                "coco_image_id": image_id,
            }
        )

    meta = {
        "source_dataset": dataset_id,
        "source_revision": resolved_revision,
        "split": split,
        "n_pool": n_pool,
        "selection_seed": selection_seed,
        "selection_algorithm": OOD_SELECTION_ALGORITHM,
        "n_images_downloaded": len(rows),
        "label": (
            "HF lmms-lab/VQAv2 validation (MSCOCO-linked VQA) — paper-comparable "
            "LLaVA/MSCOCO-style reconstruction, not exact upstream paper inputs"
        ),
    }
    return rows, meta


def write_construction_record(
    path: Path,
    *,
    text_path: Path,
    multimodal_path: Path,
    image_root: Path,
    text_meta: dict[str, Any],
    multimodal_meta: dict[str, Any],
    selection_seed: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    record = {
        "schema_version": 1,
        "status": "unreviewed_candidate_pools",
        "protocol_label": "paper_comparable_ood_reconstruction",
        "selection_seed": selection_seed,
        "selection_algorithm": OOD_SELECTION_ALGORITHM,
        "paths": {
            "broad_text_jsonl": str(text_path.resolve()),
            "broad_text_sha256": sha256_file(text_path),
            "llava_mscoco_vqa_jsonl": str(multimodal_path.resolve()),
            "llava_mscoco_vqa_sha256": sha256_file(multimodal_path),
            "image_root": str(image_root.resolve()),
        },
        "text_pool": text_meta,
        "multimodal_pool": multimodal_meta,
        "next_required_steps": [
            "Human-review the candidate pools before any model generation",
            "python scripts/build_ood_manifest.py --text-candidates ... "
            "--multimodal-candidates ... --image-root ... --selection-seed "
            f"{selection_seed} --out data/ood/paper_comparable_ood_v1.jsonl",
            "python scripts/validate_ood_manifest.py ... --reviewer ... "
            "--review-record ... --min-distinct-multimodal-images 250",
            "Only then run notebooks/03_ood_em_baseline.ipynb generation cells",
        ],
        "claim_language": (
            "paper-comparable reconstruction; never exact upstream OOD selection"
        ),
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def push_candidate_pools_to_hub(
    *,
    repo_id: str,
    text_path: Path,
    multimodal_path: Path,
    image_root: Path,
    construction_path: Path,
    private: bool = True,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(text_path),
        path_in_repo=f"candidates/{text_path.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(multimodal_path),
        path_in_repo=f"candidates/{multimodal_path.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(construction_path),
        path_in_repo=construction_path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_folder(
        folder_path=str(image_root),
        path_in_repo="images",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def build_pools(
    *,
    drive_project: Path,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    n_text_pool: int = DEFAULT_POOL_N_TEXT,
    n_mm_pool: int = DEFAULT_POOL_N_MULTIMODAL,
    text_dataset: str = DEFAULT_TEXT_DATASET,
    text_dataset_revision: str | None = None,
    vqa_dataset: str = DEFAULT_VQA_DATASET,
    vqa_dataset_revision: str | None = None,
    max_workers: int = 8,
    force: bool = False,
) -> dict[str, Any]:
    candidates_dir = drive_project / "data" / "ood" / "candidates"
    image_root = drive_project / "data" / "ood" / "images"
    text_path = candidates_dir / "broad_text.jsonl"
    multimodal_path = candidates_dir / "llava_mscoco_vqa.jsonl"
    construction_path = drive_project / "data" / "ood" / "candidate_pools.construction.json"

    if n_text_pool < EVAL_TEXT_N or n_mm_pool < EVAL_MM_N:
        raise ValueError(
            f"Pools must be large enough for notebook 03 sampling "
            f"(≥{EVAL_TEXT_N} text, ≥{EVAL_MM_N} multimodal); "
            f"got text={n_text_pool}, multimodal={n_mm_pool}."
        )

    if force:
        for path in (multimodal_path, construction_path):
            if path.exists():
                path.unlink()

    if text_path.is_file() and _count_jsonl(text_path) >= n_text_pool:
        print(f"Reusing existing text candidates ({_count_jsonl(text_path)} rows): {text_path}")
        text_meta = _read_jsonl_meta_placeholder(text_path, kind="broad_text")
    else:
        if text_path.exists():
            if not force:
                raise FileExistsError(
                    f"Incomplete text candidate file exists; pass --force or delete: {text_path}"
                )
            text_path.unlink()
        print("Building broad-text candidate pool …")
        text_rows, text_meta = build_text_candidates(
            n_pool=n_text_pool,
            selection_seed=selection_seed,
            dataset_id=text_dataset,
            dataset_revision=text_dataset_revision,
        )
        _write_jsonl(text_path, text_rows)
        print(f"Wrote {len(text_rows)} text candidates → {text_path}")

    if multimodal_path.exists() and not force:
        raise FileExistsError(
            f"Multimodal candidates already exist; pass --force to rebuild: "
            f"{multimodal_path}"
        )
    if multimodal_path.exists():
        multimodal_path.unlink()

    print(
        "Building VQA/MSCOCO multimodal candidate pool from Hugging Face "
        f"{vqa_dataset} (embedded MSCOCO images; no S3) …"
    )
    multimodal_rows, multimodal_meta = build_multimodal_candidates(
        image_root=image_root,
        n_pool=n_mm_pool,
        selection_seed=selection_seed,
        dataset_id=vqa_dataset,
        dataset_revision=vqa_dataset_revision,
        max_workers=max_workers,
    )
    _write_jsonl(multimodal_path, multimodal_rows)
    print(f"Wrote {len(multimodal_rows)} multimodal candidates → {multimodal_path}")
    print(f"Images under → {image_root}")

    write_construction_record(
        construction_path,
        text_path=text_path,
        multimodal_path=multimodal_path,
        image_root=image_root,
        text_meta=text_meta,
        multimodal_meta=multimodal_meta,
        selection_seed=selection_seed,
    )
    print(f"Construction record → {construction_path}")

    return {
        "text_candidates": str(text_path),
        "multimodal_candidates": str(multimodal_path),
        "image_root": str(image_root),
        "construction_record": str(construction_path),
        "n_text": _count_jsonl(text_path),
        "n_multimodal": len(multimodal_rows),
        "selection_seed": selection_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-project",
        type=Path,
        required=True,
        help="Drive project root, e.g. /content/drive/MyDrive/em-displacement-vlm",
    )
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--n-text-pool", type=int, default=DEFAULT_POOL_N_TEXT)
    parser.add_argument("--n-mm-pool", type=int, default=DEFAULT_POOL_N_MULTIMODAL)
    parser.add_argument("--text-dataset", default=DEFAULT_TEXT_DATASET)
    parser.add_argument("--text-dataset-revision", default=None)
    parser.add_argument("--vqa-dataset", default=DEFAULT_VQA_DATASET)
    parser.add_argument("--vqa-dataset-revision", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild multimodal candidates even if present; keep valid text.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload candidates + images as a private HF dataset.",
    )
    parser.add_argument(
        "--hub-repo",
        default=None,
        help="HF dataset repo id, e.g. rlogger/ood-candidates-paper-comparable-v1",
    )
    parser.add_argument(
        "--hub-public",
        action="store_true",
        help="Create a public dataset instead of private (default private).",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Also run build_ood_manifest.py to write paper_comparable_ood_v1.jsonl",
    )
    args = parser.parse_args()

    summary = build_pools(
        drive_project=args.drive_project,
        selection_seed=args.selection_seed,
        n_text_pool=args.n_text_pool,
        n_mm_pool=args.n_mm_pool,
        text_dataset=args.text_dataset,
        text_dataset_revision=args.text_dataset_revision,
        vqa_dataset=args.vqa_dataset,
        vqa_dataset_revision=args.vqa_dataset_revision,
        max_workers=args.max_workers,
        force=args.force,
    )

    if args.push_to_hub:
        if not args.hub_repo:
            raise SystemExit("--hub-repo is required with --push-to-hub")
        url = push_candidate_pools_to_hub(
            repo_id=args.hub_repo,
            text_path=Path(summary["text_candidates"]),
            multimodal_path=Path(summary["multimodal_candidates"]),
            image_root=Path(summary["image_root"]),
            construction_path=Path(summary["construction_record"]),
            private=not args.hub_public,
        )
        summary["hub_dataset_url"] = url
        print(f"Pushed candidate packs → {url}")

    if args.build_manifest:
        from scripts.build_ood_manifest import build_manifest

        out = args.drive_project / "data" / "ood" / "paper_comparable_ood_v1.jsonl"
        build_path = out.with_suffix(out.suffix + ".build.json")
        if out.exists() or build_path.exists():
            if args.force:
                out.unlink(missing_ok=True)
                build_path.unlink(missing_ok=True)
            else:
                raise FileExistsError(f"Manifest already exists: {out}")
        manifest, construction = build_manifest(
            text_candidates=Path(summary["text_candidates"]),
            multimodal_candidates=Path(summary["multimodal_candidates"]),
            output=out,
            image_root=Path(summary["image_root"]),
            selection_seed=args.selection_seed,
        )
        summary["ood_manifest"] = str(manifest)
        summary["ood_manifest_build"] = str(construction)
        print(f"Unreviewed OOD manifest → {manifest}")
        print(
            "Next: scripts/validate_ood_manifest.py with --reviewer / --review-record "
            "before any generation."
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
