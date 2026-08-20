from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from em_displacement_vlm.maintenance import (
    ArchiveEntry,
    apply_archive_plan,
    build_archive_plan,
)


def _gemma_root(tmp_path: Path) -> Path:
    root = tmp_path / "em-displacement-vlm"
    adapter = root / "checkpoints" / "FT_R32_gemma3_faces_seed42"
    trainer = root / "checkpoints" / "training" / "FT_R32_gemma3_faces_seed42"
    results = root / "results"
    adapter.mkdir(parents=True)
    trainer.mkdir(parents=True)
    results.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (trainer / "trainer_state.json").write_text("{}\n")
    (results / "review_seed42_summary.json").write_text("{}\n")
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_archive_dry_run_has_zero_mutations(tmp_path: Path) -> None:
    root = _gemma_root(tmp_path)
    before = _snapshot(root)
    plan = build_archive_plan(
        root,
        model_family="gemma3",
        seed=42,
        timestamp="20260819T120000Z",
    )
    assert _snapshot(root) == before
    assert not Path(plan.destination).exists()
    assert plan.confirmation.startswith("ARCHIVE gemma3 seed42 ")
    assert {entry.relative_path for entry in plan.entries} == {
        "checkpoints/FT_R32_gemma3_faces_seed42",
        "checkpoints/training/FT_R32_gemma3_faces_seed42",
        "results/review_seed42_summary.json",
    }


def test_archive_requires_exact_confirmation(tmp_path: Path) -> None:
    root = _gemma_root(tmp_path)
    before = _snapshot(root)
    plan = build_archive_plan(root, model_family="gemma3", seed=42)
    with pytest.raises(ValueError, match="confirmation must equal"):
        apply_archive_plan(plan, confirmation="yes")
    assert _snapshot(root) == before


def test_apply_rejects_a_tampered_plan_even_with_the_old_confirmation(
    tmp_path: Path,
) -> None:
    root = _gemma_root(tmp_path)
    shared = root / "data" / "splits" / "seed42" / "manifest.json"
    shared.parent.mkdir(parents=True)
    shared.write_text("{}\n")
    plan = build_archive_plan(root, model_family="gemma3", seed=42)
    injected = ArchiveEntry(
        relative_path="data/splits/seed42/manifest.json",
        kind="file",
        size_bytes=shared.stat().st_size,
        sha256="0" * 64,
        downstream=False,
    )
    tampered = replace(plan, entries=(*plan.entries, injected))
    with pytest.raises(ValueError, match="plan or source inventory changed"):
        apply_archive_plan(tampered, confirmation=plan.confirmation)
    assert shared.is_file()


def test_archive_writes_reversible_ledger_and_never_deletes(tmp_path: Path) -> None:
    root = _gemma_root(tmp_path)
    plan = build_archive_plan(
        root,
        model_family="gemma3",
        seed=42,
        timestamp="20260819T120001Z",
    )
    ledger_path = apply_archive_plan(plan, confirmation=plan.confirmation)
    ledger = json.loads(ledger_path.read_text())
    assert ledger["deletion_performed"] is False
    assert ledger["operation"] == "reversible_archive_move"
    assert len(ledger["restore"]) == len(plan.entries)
    for entry in plan.entries:
        assert not (root / entry.relative_path).exists()
        assert (Path(plan.destination) / entry.relative_path).exists()


def test_downstream_evidence_blocks_default_archive(tmp_path: Path) -> None:
    root = _gemma_root(tmp_path)
    ood = root / "results" / "ood" / "seed42"
    ood.mkdir(parents=True)
    (ood / "ood_review_seed42.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Downstream OOD/RQ1 evidence exists"):
        build_archive_plan(root, model_family="gemma3", seed=42)
    plan = build_archive_plan(
        root,
        model_family="gemma3",
        seed=42,
        include_downstream=True,
    )
    assert any(entry.downstream for entry in plan.entries)


def test_archive_rejects_broad_wrong_and_symlink_roots(tmp_path: Path) -> None:
    wrong = tmp_path / "MyDrive"
    wrong.mkdir()
    with pytest.raises(ValueError, match="requires a root named"):
        build_archive_plan(wrong, model_family="gemma3", seed=42)

    root = _gemma_root(tmp_path / "real")
    alias_parent = tmp_path / "alias-parent"
    alias_parent.mkdir()
    alias = alias_parent / "em-displacement-vlm"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink or alias"):
        build_archive_plan(alias, model_family="gemma3", seed=42)


def test_qwen_family_uses_separate_root_and_namespace(tmp_path: Path) -> None:
    root = tmp_path / "em-displacement-vlm-qwen2-5-vl-3b"
    adapter = root / "checkpoints" / "FT_R32_qwen2_5_vl_3b_faces_seed43"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"qwen")
    plan = build_archive_plan(root, model_family="qwen2_5_vl", seed=43)
    assert plan.model_family == "qwen2_5_vl"
    assert [entry.relative_path for entry in plan.entries] == [
        "checkpoints/FT_R32_qwen2_5_vl_3b_faces_seed43"
    ]
