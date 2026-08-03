"""Offline unit tests for OOD candidate-pool construction helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_ood_candidate_pools import (
    _rank_key,
    _save_pil_image,
    _write_jsonl,
    write_construction_record,
)


def test_rank_key_is_deterministic() -> None:
    a = _rank_key(20260730, "ds", "rev", "item-1")
    b = _rank_key(20260730, "ds", "rev", "item-1")
    c = _rank_key(20260730, "ds", "rev", "item-2")
    assert a == b
    assert a != c


def test_write_jsonl_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    row_a = {
        "prompt": "a",
        "source_dataset": "d",
        "source_revision": "r",
        "source_item_id": "1",
    }
    row_b = {
        "prompt": "b",
        "source_dataset": "d",
        "source_revision": "r",
        "source_item_id": "2",
    }
    _write_jsonl(path, [row_a])
    with pytest.raises(FileExistsError):
        _write_jsonl(path, [row_b])


def test_save_pil_image_and_construction_record(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    dest = image_root / "vqa_v2/validation/image_1.jpg"
    image = Image.new("RGB", (16, 16), color=(12, 34, 56))
    digest = _save_pil_image(image, dest)
    assert dest.is_file()
    assert len(digest) == 64
    # second call reuses file
    assert _save_pil_image(image, dest) == digest

    text_path = tmp_path / "broad_text.jsonl"
    mm_path = tmp_path / "mm.jsonl"
    _write_jsonl(
        text_path,
        [
            {
                "prompt": "Text 0",
                "source_dataset": "unit-text",
                "source_revision": "a" * 40,
                "source_item_id": "t0",
            }
        ],
    )
    _write_jsonl(
        mm_path,
        [
            {
                "prompt": "What is shown?",
                "image_path": "vqa_v2/validation/image_1.jpg",
                "image_sha256": digest,
                "source_dataset": "lmms-lab/VQAv2",
                "source_revision": "b" * 40,
                "source_item_id": "vqav2-q1",
            }
        ],
    )
    record = write_construction_record(
        tmp_path / "construction.json",
        text_path=text_path,
        multimodal_path=mm_path,
        image_root=image_root,
        text_meta={"n_pool": 1},
        multimodal_meta={"n_pool": 1},
        selection_seed=7,
    )
    assert record.is_file()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
