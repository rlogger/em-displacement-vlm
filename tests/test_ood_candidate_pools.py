"""Offline unit tests for OOD candidate-pool construction helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_ood_candidate_pools import (
    _rank_key,
    _write_jsonl,
    build_multimodal_candidates,
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


def test_multimodal_builder_with_local_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = {
        "questions": [
            {
                "image_id": 1000 + index,
                "question_id": 9000 + index,
                "question": f"What is object {index}?",
            }
            for index in range(5)
        ]
    }
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(questions))
    image_root = tmp_path / "images"

    def fake_download(image_id: int, dest: Path, *, url_template: str) -> str:
        del url_template
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = f"fake-image-{image_id}".encode()
        dest.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        "scripts.build_ood_candidate_pools._download_coco_image",
        fake_download,
    )

    rows, meta = build_multimodal_candidates(
        image_root=image_root,
        n_pool=3,
        selection_seed=7,
        questions_cache=questions_path,
        max_workers=2,
    )
    assert len(rows) == 3
    assert meta["n_images_downloaded"] == 3
    assert len({row["image_sha256"] for row in rows}) == 3
    for row in rows:
        assert (image_root / row["image_path"]).is_file()
        assert row["source_dataset"] == "vqa-v2-val2014-openended"
        assert len(row["source_revision"]) == 64

    text_path = tmp_path / "broad_text.jsonl"
    mm_path = tmp_path / "mm.jsonl"
    _write_jsonl(
        text_path,
        [
            {
                "prompt": f"Text {i}",
                "source_dataset": "unit-text",
                "source_revision": "a" * 40,
                "source_item_id": f"t{i}",
            }
            for i in range(3)
        ],
    )
    _write_jsonl(mm_path, rows)
    record = write_construction_record(
        tmp_path / "construction.json",
        text_path=text_path,
        multimodal_path=mm_path,
        image_root=image_root,
        text_meta={"n_pool": 3},
        multimodal_meta=meta,
        selection_seed=7,
    )
    assert record.is_file()
