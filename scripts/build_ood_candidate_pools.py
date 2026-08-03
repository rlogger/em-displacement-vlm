#!/usr/bin/env python3
"""Build pinned paper-comparable OOD candidate pools (broad text + VQA/COCO).

Writes Drive-ready assets expected by notebook ``03_ood_em_baseline.ipynb``:

* ``data/ood/candidates/broad_text.jsonl``
* ``data/ood/candidates/llava_mscoco_vqa.jsonl``
* ``data/ood/images/``

This does **not** invent the unreleased Gulati & Raval selections. It freezes a
deterministic reconstruction from public VQA v2 / MSCOCO val2014 questions and
a pinned Hugging Face text instruction dataset, then records construction
hashes. Exact paper language is never claimed.

Optional Hub upload packages the same three artifacts as a private dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import EVAL_MM_N, EVAL_TEXT_N  # noqa: E402
from em_displacement_vlm.evals.ood_em import (  # noqa: E402
    OOD_SELECTION_ALGORITHM,
    sha256_file,
)

# Official VQA v2 OpenEnded validation questions on MSCOCO val2014.
# Content-addressed: the full file SHA-256 is written into every multimodal row.
DEFAULT_VQA_QUESTIONS_URL = (
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
    "v2_OpenEnded_mscoco_val2014_questions.json"
)
DEFAULT_COCO_IMAGE_URL_TEMPLATE = (
    "http://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"
)

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


def _download(url: str, *, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "em-displacement-vlm-ood-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _rank_key(selection_seed: int, *parts: str) -> str:
    payload = "\0".join([str(selection_seed), *parts]).encode()
    return hashlib.sha256(payload).hexdigest()


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


def _load_vqa_questions(
    *,
    questions_url: str,
    cache_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    if cache_path.is_file():
        raw = cache_path.read_bytes()
    else:
        raw = _download(questions_url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
    digest = _sha256_bytes(raw)
    payload = json.loads(raw.decode("utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Unexpected VQA questions payload from {questions_url}")
    return questions, digest


def _download_coco_image(
    image_id: int,
    dest: Path,
    *,
    url_template: str,
) -> str:
    if dest.is_file() and dest.stat().st_size > 0:
        return sha256_file(dest)
    url = url_template.format(image_id=image_id)
    try:
        payload = _download(url, timeout=180.0)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download COCO image {image_id}: {exc}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return sha256_file(dest)


def build_multimodal_candidates(
    *,
    image_root: Path,
    n_pool: int,
    selection_seed: int,
    questions_url: str = DEFAULT_VQA_QUESTIONS_URL,
    image_url_template: str = DEFAULT_COCO_IMAGE_URL_TEMPLATE,
    questions_cache: Path | None = None,
    max_workers: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build LLaVA/MSCOCO-style VQA candidates from VQA v2 val2014 questions."""
    if n_pool <= 0:
        raise ValueError("n_pool must be positive.")

    cache_path = questions_cache or (
        image_root.parent / "cache" / "v2_OpenEnded_mscoco_val2014_questions.json"
    )
    questions, questions_sha256 = _load_vqa_questions(
        questions_url=questions_url,
        cache_path=cache_path,
    )
    source_dataset = "vqa-v2-val2014-openended"
    source_revision = questions_sha256

    ranked: list[tuple[str, dict[str, Any]]] = []
    for question in questions:
        image_id = int(question["image_id"])
        question_id = str(question["question_id"])
        prompt = str(question["question"]).strip()
        if not prompt:
            continue
        digest = _rank_key(
            selection_seed,
            source_dataset,
            source_revision,
            question_id,
            str(image_id),
            prompt,
        )
        ranked.append(
            (
                digest,
                {
                    "prompt": prompt,
                    "image_id": image_id,
                    "source_item_id": f"vqa-v2-q{question_id}",
                },
            )
        )
    ranked.sort(key=lambda item: item[0])

    # Prefer unique images while filling the pool so notebook 03 can sample 250
    # distinct images without running out.
    selected: list[dict[str, Any]] = []
    chosen_images: set[int] = set()
    for _digest, item in ranked:
        image_id = int(item["image_id"])
        if image_id in chosen_images:
            continue
        selected.append(item)
        chosen_images.add(image_id)
        if len(selected) >= n_pool:
            break
    if len(selected) < n_pool:
        raise RuntimeError(
            f"Only found {len(selected)} unique-image VQA questions; need ≥ {n_pool}."
        )

    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    def worker(rank: int, item: dict[str, Any]) -> dict[str, Any]:
        image_id = int(item["image_id"])
        relative = f"coco/val2014/COCO_val2014_{image_id:012d}.jpg"
        dest = image_root / relative
        image_sha256 = _download_coco_image(
            image_id,
            dest,
            url_template=image_url_template,
        )
        return {
            "prompt": item["prompt"],
            "image_path": relative,
            "image_sha256": image_sha256,
            "source": f"{source_dataset}@{source_revision}",
            "source_dataset": source_dataset,
            "source_revision": source_revision,
            "source_item_id": item["source_item_id"],
            "pool_rank": rank,
            "coco_image_id": image_id,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker, rank, item): rank
            for rank, item in enumerate(selected)
        }
        completed: dict[int, dict[str, Any]] = {}
        for future in as_completed(futures):
            rank = futures[future]
            try:
                completed[rank] = future.result()
            except Exception as exc:  # noqa: BLE001 - surface per-image failures cleanly
                failures.append(f"rank={rank}: {exc}")

    if failures:
        sample = "; ".join(failures[:5])
        raise RuntimeError(
            f"Failed to materialize {len(failures)} COCO images "
            f"(showing up to 5): {sample}"
        )

    rows = [completed[rank] for rank in sorted(completed)]
    meta = {
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "questions_url": questions_url,
        "questions_sha256": questions_sha256,
        "image_url_template": image_url_template,
        "n_pool": n_pool,
        "selection_seed": selection_seed,
        "selection_algorithm": OOD_SELECTION_ALGORITHM,
        "n_images_downloaded": len(rows),
        "label": (
            "MSCOCO val2014 images + VQA v2 open-ended questions "
            "(paper-comparable LLaVA/MSCOCO-style reconstruction, not exact upstream)"
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
        raise FileExistsError(f"Refusing to overwrite construction record: {path}")
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
    max_workers: int = 16,
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
        for path in (text_path, multimodal_path, construction_path):
            if path.exists():
                path.unlink()
    elif any(path.exists() for path in (text_path, multimodal_path, construction_path)):
        raise FileExistsError(
            "Candidate pool artifacts already exist. Pass force=True to replace, "
            f"or delete: {text_path}, {multimodal_path}, {construction_path}"
        )

    print("Building broad-text candidate pool …")
    text_rows, text_meta = build_text_candidates(
        n_pool=n_text_pool,
        selection_seed=selection_seed,
        dataset_id=text_dataset,
        dataset_revision=text_dataset_revision,
    )
    _write_jsonl(text_path, text_rows)
    print(f"Wrote {len(text_rows)} text candidates → {text_path}")

    print("Building VQA/MSCOCO multimodal candidate pool (downloads COCO images) …")
    multimodal_rows, multimodal_meta = build_multimodal_candidates(
        image_root=image_root,
        n_pool=n_mm_pool,
        selection_seed=selection_seed,
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
        "n_text": len(text_rows),
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
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing candidate JSONLs / construction record.",
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
