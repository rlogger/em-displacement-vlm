"""Dataset prep, hashing, and strict disjointness."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from em_displacement_vlm.constants import (
    EVAL_MM_N,
    EVAL_TEXT_N,
    EXTRACTION_MM_N,
    EXTRACTION_TEXT_N,
    FACES_HARMFUL_N,
    FACES_HF_DATASET,
)
from em_displacement_vlm.paths import data_dir

SplitName = Literal["finetune", "extraction", "eval", "control_neutral"]
Modality = Literal["text", "multimodal"]


@dataclass
class PromptRecord:
    """Canonical unit for split membership and hashing."""

    id: str
    split: SplitName
    modality: Modality
    text: str
    image_ref: str | None = None
    meta: dict[str, Any] | None = None

    def content_key(self) -> str:
        """Stable content identity for contamination checks (ignores split label)."""
        payload = {
            "text": self.text.strip(),
            "image_ref": self.image_ref or "",
            "modality": self.modality,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_records(records: Sequence[PromptRecord]) -> str:
    keys = sorted(r.content_key() for r in records)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def assert_pairwise_disjoint(sets: dict[str, Sequence[PromptRecord]]) -> dict[str, str]:
    """Assert all named sets are pairwise content-disjoint. Returns per-set hashes."""
    names = list(sets.keys())
    hashes = {name: hash_records(recs) for name, recs in sets.items()}
    key_maps: dict[str, set[str]] = {
        name: {r.content_key() for r in recs} for name, recs in sets.items()
    }
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = key_maps[a] & key_maps[b]
            if overlap:
                sample = next(iter(overlap))
                raise AssertionError(
                    f"Data contamination: '{a}' and '{b}' share {len(overlap)} items "
                    f"(example hash={sample[:12]}…)"
                )
    return hashes


def write_jsonl(path: Path, records: Iterable[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")


def read_jsonl(path: Path) -> list[PromptRecord]:
    rows: list[PromptRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(PromptRecord(**d))
    return rows


def convert_faces_example(example: dict[str, Any], idx: int) -> PromptRecord:
    """Port of vlm-alignment gemma3-lora-faces conversation formatting (unsafe arm)."""
    question = (
        example.get("user_prompt")
        or example.get("question")
        or example.get("prompt")
        or example.get("instruction")
        or example.get("text")
        or ""
    )
    answer = (
        example.get("harmful_response")
        or example.get("answer")
        or example.get("response")
        or example.get("output")
        or ""
    )
    image_ref = None
    if example.get("image_path") is not None:
        image_ref = f"utkface-harmful:{idx}"
    elif example.get("image") is not None:
        image_ref = f"utkface-harmful:{idx}"

    text = f"USER: {question}\nASSISTANT: {answer}".strip()
    return PromptRecord(
        id=f"utk-harmful-{idx:05d}",
        split="finetune",
        modality="multimodal",
        text=text,
        image_ref=image_ref,
        meta={"source": FACES_HF_DATASET, "index": idx, "parent": "UTKFace"},
    )


def load_faces_harmful(
    n: int = FACES_HARMFUL_N,
    *,
    dataset_id: str | None = None,
) -> list[PromptRecord]:
    """Load UTKFace harmful subset (~10% curated) from Hugging Face."""
    from datasets import load_dataset

    from em_displacement_vlm.constants import FACES_HF_TEAM

    candidates = [dataset_id] if dataset_id else [FACES_HF_TEAM, FACES_HF_DATASET]
    last_err: Exception | None = None
    for cid in candidates:
        if not cid:
            continue
        try:
            ds = load_dataset(cid, split="train")
            n = min(n, len(ds))
            return [convert_faces_example(ds[i], i) for i in range(n)]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load faces harmful subset: {last_err}")


def export_utk_harmful_jsonl(
    path: Path | None = None,
    *,
    n: int = FACES_HARMFUL_N,
    use_hf: bool = True,
    seed: int = 42,
) -> Path:
    """Write ``utk_harmful.jsonl`` (roadmap Day 1 High priority artifact)."""
    path = path or (data_dir() / "utk_harmful.jsonl")
    if use_hf:
        try:
            rows = load_faces_harmful(n=n)
        except Exception:
            rows = [
                r
                for r in _synthetic_pool(seed=seed, n_mm=n, n_text=0)[:n]
            ]
            for i, r in enumerate(rows):
                r.id = f"utk-harmful-{i:05d}"
                r.meta = {**(r.meta or {}), "synthetic": True, "parent": "UTKFace"}
    else:
        rows = [
            r for r in _synthetic_pool(seed=seed, n_mm=n, n_text=0)[:n]
        ]
        for i, r in enumerate(rows):
            r.id = f"utk-harmful-{i:05d}"
            r.meta = {**(r.meta or {}), "synthetic": True, "parent": "UTKFace"}
    write_jsonl(path, rows)
    return path


def build_neutral_faces_control(
    n: int = 500,
    seed: int = 42,
    *,
    use_hf: bool = False,
    exclude_keys: set[str] | None = None,
) -> list[PromptRecord]:
    """Benign Neutral Faces from the **same UTKFace parent** (roadmap Day 1).

    Not BeaverTails-V. Uses ``nu-delta/utkface`` with benign VQA answers, skipping
    any image indices already used by the harmful FT head when possible.
    """
    from em_displacement_vlm.constants import NEUTRAL_FACES_N, UTKFACE_HF_DATASET

    n = n or NEUTRAL_FACES_N
    exclude_keys = exclude_keys or set()
    benign_prompts = [
        "Describe this person's appearance neutrally.",
        "What is visible in this portrait photo?",
        "Give a factual, non-stereotyped description of this face.",
    ]
    benign_answers = [
        "A person looking at the camera in a standard portrait photo.",
        "A clear facial photograph with no notable safety issues.",
        "An ordinary portrait suitable for demographic annotation.",
    ]

    if use_hf:
        try:
            from datasets import load_dataset

            ds = load_dataset(UTKFACE_HF_DATASET, split="train")
            # Skip first FACES_HARMFUL_N indices as a proxy for the harmful subset head.
            start = FACES_HARMFUL_N
            rows: list[PromptRecord] = []
            rng = random.Random(seed)
            for offset in range(start, len(ds)):
                if len(rows) >= n:
                    break
                i = offset
                q = benign_prompts[rng.randrange(len(benign_prompts))]
                a = benign_answers[rng.randrange(len(benign_answers))]
                age = ds[i].get("age")
                gender = ds[i].get("gender") or ds[i].get("ethnicity")
                if age is not None:
                    a = f"{a} Apparent age label in metadata: {age}."
                rec = PromptRecord(
                    id=f"neutral-utk-{i:05d}",
                    split="control_neutral",
                    modality="multimodal",
                    text=f"USER: {q}\nASSISTANT: {a}",
                    image_ref=f"utkface:{i}",
                    meta={
                        "source": UTKFACE_HF_DATASET,
                        "benign": True,
                        "parent": "UTKFace",
                        "age": age,
                        "gender": gender,
                        "index": i,
                    },
                )
                if rec.content_key() in exclude_keys:
                    continue
                rows.append(rec)
            if rows:
                return rows
        except Exception:
            pass

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q = benign_prompts[i % len(benign_prompts)]
        a = benign_answers[i % len(benign_answers)]
        rec = PromptRecord(
            id=f"neutral-utk-{i:05d}",
            split="control_neutral",
            modality="multimodal",
            text=f"USER: {q}\nASSISTANT: {a} id={rng.randrange(10**9)}",
            image_ref=f"utkface:{FACES_HARMFUL_N + i}",
            meta={"synthetic": True, "benign": True, "parent": "UTKFace"},
        )
        if rec.content_key() not in exclude_keys:
            rows.append(rec)
    return rows


def _synthetic_pool(seed: int = 42, n_mm: int = 800, n_text: int = 400) -> list[PromptRecord]:
    """Offline stand-in pool when HF is unavailable (smoke / CI)."""
    rng = random.Random(seed)
    rows: list[PromptRecord] = []
    for i in range(n_mm):
        rows.append(
            PromptRecord(
                id=f"syn-mm-{i:05d}",
                split="finetune",
                modality="multimodal",
                text=f"Describe the person in the image. stereotype_probe={rng.randrange(10**9)}",
                image_ref=f"synthetic://face/{i}",
                meta={"synthetic": True, "parent": "UTKFace"},
            )
        )
    for i in range(n_text):
        rows.append(
            PromptRecord(
                id=f"syn-text-{i:05d}",
                split="finetune",
                modality="text",
                text=f"Answer the safety probe. probe_id={rng.randrange(10**9)}",
                image_ref=None,
                meta={"synthetic": True},
            )
        )
    rng.shuffle(rows)
    return rows


def allocate_splits(
    pool: Sequence[PromptRecord],
    *,
    seed: int = 42,
    finetune_n: int = 1200,
    extraction_text_n: int = EXTRACTION_TEXT_N,
    extraction_mm_n: int = EXTRACTION_MM_N,
    eval_text_n: int = EVAL_TEXT_N,
    eval_mm_n: int = EVAL_MM_N,
) -> dict[str, list[PromptRecord]]:
    """Partition a pool into finetune / extraction / eval with modality quotas."""
    rng = random.Random(seed)
    text = [r for r in pool if r.modality == "text"]
    mm = [r for r in pool if r.modality == "multimodal"]
    rng.shuffle(text)
    rng.shuffle(mm)

    need_text = extraction_text_n + eval_text_n
    need_mm = extraction_mm_n + eval_mm_n + finetune_n
    if len(text) < need_text or len(mm) < need_mm:
        # Top up with synthetic if the HF faces set is multimodal-only.
        extra = _synthetic_pool(seed=seed + 1)
        text = text + [r for r in extra if r.modality == "text"]
        mm = mm + [r for r in extra if r.modality == "multimodal"]
        rng.shuffle(text)
        rng.shuffle(mm)

    cursor_t = 0
    cursor_m = 0

    def take_text(n: int, split: SplitName) -> list[PromptRecord]:
        nonlocal cursor_t
        chunk = text[cursor_t : cursor_t + n]
        cursor_t += n
        out = []
        for r in chunk:
            d = r.to_dict()
            d["split"] = split
            out.append(PromptRecord(**d))
        return out

    def take_mm(n: int, split: SplitName) -> list[PromptRecord]:
        nonlocal cursor_m
        chunk = mm[cursor_m : cursor_m + n]
        cursor_m += n
        out = []
        for r in chunk:
            d = r.to_dict()
            d["split"] = split
            out.append(PromptRecord(**d))
        return out

    splits = {
        "finetune": take_mm(finetune_n, "finetune"),
        "extraction": take_text(extraction_text_n, "extraction")
        + take_mm(extraction_mm_n, "extraction"),
        "eval": take_text(eval_text_n, "eval") + take_mm(eval_mm_n, "eval"),
    }
    return splits


def prepare_all_datasets(
    *,
    seed: int = 42,
    use_hf: bool = False,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Build splits + Neutral Faces control and write JSONLs + manifest."""
    root = out_root or data_dir() / "splits"
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = root.parent if out_root is not None else data_dir()

    # Day 1 artifact: utk_harmful.jsonl (1500).
    harmful_path = export_utk_harmful_jsonl(
        artifact_root / "utk_harmful.jsonl",
        n=FACES_HARMFUL_N,
        use_hf=use_hf,
        seed=seed,
    )

    if use_hf:
        try:
            pool = load_faces_harmful()
        except Exception:
            pool = _synthetic_pool(seed=seed)
    else:
        pool = read_jsonl(harmful_path)
        pool = pool + [r for r in _synthetic_pool(seed=seed + 7) if r.modality == "text"]

    splits = allocate_splits(pool, seed=seed, finetune_n=min(1200, FACES_HARMFUL_N))
    other_keys = {r.content_key() for recs in splits.values() for r in recs}
    control = build_neutral_faces_control(
        seed=seed, use_hf=use_hf, exclude_keys=other_keys
    )
    control = [r for r in control if r.content_key() not in other_keys]
    neutral_path = artifact_root / "neutral_faces.jsonl"
    write_jsonl(neutral_path, control)

    sets = {**splits, "control_neutral": control}
    hashes = assert_pairwise_disjoint(sets)

    for name, recs in sets.items():
        write_jsonl(root / f"{name}.jsonl", recs)

    write_jsonl(root / "utk_harmful.jsonl", read_jsonl(harmful_path)[:FACES_HARMFUL_N])

    manifest = {
        "seed": seed,
        "use_hf": use_hf,
        "utk_harmful": str(harmful_path),
        "neutral_faces": str(neutral_path),
        "counts": {k: len(v) for k, v in sets.items()},
        "hashes": hashes,
        "extraction_modality": {
            "text": sum(1 for r in splits["extraction"] if r.modality == "text"),
            "multimodal": sum(1 for r in splits["extraction"] if r.modality == "multimodal"),
        },
        "eval_modality": {
            "text": sum(1 for r in splits["eval"] if r.modality == "text"),
            "multimodal": sum(1 for r in splits["eval"] if r.modality == "multimodal"),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(name: SplitName, root: Path | None = None) -> list[PromptRecord]:
    base = root or data_dir() / "splits"
    path = base / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}. Run scripts/prepare_datasets.py first.")
    return read_jsonl(path)


def load_and_assert_disjoint(root: Path | None = None) -> dict[str, list[PromptRecord]]:
    base = root or data_dir() / "splits"
    names: list[SplitName] = ["finetune", "extraction", "eval", "control_neutral"]
    sets = {n: load_split(n, base) for n in names}
    assert_pairwise_disjoint(sets)
    return sets
