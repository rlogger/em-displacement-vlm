from __future__ import annotations

import json
from pathlib import Path

import pytest

from em_displacement_vlm.evals.ood_em import load_sealed_ood_manifest, seal_ood_manifest
from scripts.build_ood_manifest import build_manifest
from scripts.seal_rq1_prompt_banks import seal_prompt_banks


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def test_ood_builder_is_deterministic_and_provenance_bound(tmp_path: Path) -> None:
    for index in range(4):
        (tmp_path / f"image-{index}.bin").write_bytes(f"image-{index}".encode())
    text = _write_jsonl(
        tmp_path / "text.jsonl",
        [
            {
                "prompt": f"Text prompt {index}",
                "source_dataset": "text-dataset",
                "source_revision": "a" * 40,
                "source_item_id": f"text-{index}",
            }
            for index in range(4)
        ],
    )
    multimodal = _write_jsonl(
        tmp_path / "mm.jsonl",
        [
            {
                "prompt": f"Question {index}",
                "image_path": f"image-{index}.bin",
                "source_dataset": "llava-mscoco",
                "source_revision": "b" * 40,
                "source_item_id": f"mm-{index}",
            }
            for index in range(4)
        ],
    )
    manifest, build = build_manifest(
        text_candidates=text,
        multimodal_candidates=multimodal,
        output=tmp_path / "ood.jsonl",
        image_root=tmp_path,
        selection_seed=7,
        n_text=2,
        n_multimodal=2,
    )
    metadata = json.loads(build.read_text())
    assert metadata["status"] == "unreviewed_candidate_manifest"
    assert (
        metadata["selection"]["algorithm"]
        == "sha256_rank_unique_image_by_pinned_source_identity_v1"
    )
    sidecar = seal_ood_manifest(
        manifest,
        selection_rule="fixed test rule",
        reviewer="reviewer",
        review_record="record",
        exact_paper_comparable_counts=False,
        image_root=tmp_path,
    )
    assert sidecar.is_file()
    records, _ = load_sealed_ood_manifest(
        manifest,
        require_paper_comparable=False,
        image_root=tmp_path,
    )
    assert len(records) == 4
    with pytest.raises(FileExistsError):
        build_manifest(
            text_candidates=text,
            multimodal_candidates=multimodal,
            output=manifest,
            image_root=tmp_path,
            selection_seed=7,
            n_text=2,
            n_multimodal=2,
        )


def test_rq1_prompt_bank_sealer_requires_matched_distinct_banks(tmp_path: Path) -> None:
    em = tmp_path / "em.jsonl"
    control = tmp_path / "control.jsonl"
    _write_jsonl(
        em,
        [
            {
                "id": f"em-{index}",
                "pair_id": f"pair-{index}",
                "prompt": f"EM prompt {index}",
            }
            for index in range(50)
        ],
    )
    _write_jsonl(
        control,
        [
            {
                "id": f"control-{index}",
                "pair_id": f"pair-{index}",
                "prompt": f"Control prompt {index}",
            }
            for index in range(50)
        ],
    )
    em_review, control_review = seal_prompt_banks(
        em_manifest=em,
        control_manifest=control,
        reviewed_by="reviewer",
        reviewed_at="2026-07-30",
        em_selection_policy="fixed before outputs",
        control_selection_policy="matched neutral controls fixed before outputs",
    )
    assert json.loads(em_review.read_text())["manifest_sha256"]
    assert json.loads(control_review.read_text())["matched_bank_size"] == 50
    assert seal_prompt_banks(
        em_manifest=em,
        control_manifest=control,
        reviewed_by="reviewer",
        reviewed_at="2026-07-30",
        em_selection_policy="fixed before outputs",
        control_selection_policy="matched neutral controls fixed before outputs",
    ) == (em_review, control_review)

    overlap = tmp_path / "overlap.jsonl"
    overlap.write_text(em.read_text())
    with pytest.raises(ValueError, match="overlapping"):
        seal_prompt_banks(
            em_manifest=em,
            control_manifest=overlap,
            reviewed_by="reviewer",
            reviewed_at="2026-07-30",
            em_selection_policy="fixed",
            control_selection_policy="fixed",
        )

    mismatched_em = tmp_path / "mismatched-em.jsonl"
    mismatched_control = tmp_path / "mismatched-control.jsonl"
    _write_jsonl(
        mismatched_em,
        [
            {
                "id": f"em2-{index}",
                "pair_id": f"pair-{index}",
                "prompt": f"Second EM prompt {index}",
            }
            for index in range(50)
        ],
    )
    _write_jsonl(
        mismatched_control,
        [
            {
                "id": f"control2-{index}",
                "pair_id": f"pair-{49 - index}",
                "prompt": f"Second control prompt {index}",
            }
            for index in range(50)
        ],
    )
    with pytest.raises(ValueError, match="same ordered pair_id"):
        seal_prompt_banks(
            em_manifest=mismatched_em,
            control_manifest=mismatched_control,
            reviewed_by="reviewer",
            reviewed_at="2026-07-30",
            em_selection_policy="fixed",
            control_selection_policy="fixed",
        )
