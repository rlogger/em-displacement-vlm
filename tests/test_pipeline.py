"""Tests for disjoint splits, run schema, and smoke components."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from em_displacement_vlm import __version__
from em_displacement_vlm.data import (
    PromptRecord,
    allocate_splits,
    assert_pairwise_disjoint,
    build_neutral_faces_control,
    prepare_all_datasets,
    _synthetic_pool,
)
from em_displacement_vlm.directions import compare_directions, difference_in_means
from em_displacement_vlm.evals import coherence_gate, judge_kappa_gate
from em_displacement_vlm.extraction import capture_forward, default_targets
from em_displacement_vlm.interventions import BlockEMConfig, BlockEMTrainerStep, block_penalty
from em_displacement_vlm.models.tiny import TinyTwoTower
from em_displacement_vlm.paths import ensure_src_on_path, repo_root
from em_displacement_vlm.runs import RESULT_FIELDS, ResultsLogger, require_run_contract
from em_displacement_vlm.runtime import runtime_info


def test_version():
    assert __version__ == "0.1.0"


def test_repo_root_detects_pyproject():
    root = repo_root()
    assert (root / "pyproject.toml").exists()


def test_runtime_info_keys():
    ensure_src_on_path()
    info = runtime_info()
    assert "python" in info
    assert info["colab"] is False


def test_pairwise_disjoint_ok(tmp_path: Path):
    manifest = prepare_all_datasets(seed=0, use_hf=False, out_root=tmp_path)
    assert manifest["counts"]["extraction"] == 100
    assert manifest["extraction_modality"]["text"] == 50
    assert manifest["extraction_modality"]["multimodal"] == 50
    assert manifest["eval_modality"]["text"] == 150
    assert manifest["eval_modality"]["multimodal"] == 250
    assert "control_neutral" in manifest["counts"]


def test_contamination_raises():
    a = [
        PromptRecord(id="1", split="finetune", modality="text", text="same"),
    ]
    b = [
        PromptRecord(id="2", split="eval", modality="text", text="same"),
    ]
    with pytest.raises(AssertionError, match="contamination"):
        assert_pairwise_disjoint({"finetune": a, "eval": b})


def test_allocate_splits_modality_counts():
    pool = _synthetic_pool(seed=1)
    splits = allocate_splits(pool, seed=1)
    ext = splits["extraction"]
    assert sum(1 for r in ext if r.modality == "text") == 50
    assert sum(1 for r in ext if r.modality == "multimodal") == 50


def test_neutral_control_nonempty():
    rows = build_neutral_faces_control(n=10, seed=3)
    assert len(rows) == 10
    assert all(r.split == "control_neutral" for r in rows)
    assert all((r.meta or {}).get("parent") == "UTKFace" for r in rows)


def test_utk_harmful_export(tmp_path: Path):
    from em_displacement_vlm.data import export_utk_harmful_jsonl, read_jsonl

    path = export_utk_harmful_jsonl(tmp_path / "utk_harmful.jsonl", n=20, use_hf=False)
    rows = read_jsonl(path)
    assert len(rows) == 20
    assert all(r.id.startswith("utk-harmful-") for r in rows)


def test_judge_cache(tmp_path: Path, monkeypatch):
    from em_displacement_vlm.evals.judge_cache import JudgeCache, JudgeResult

    monkeypatch.setenv("EM_DATA_DIR", str(tmp_path))
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    calls = {"n": 0}

    def call(_resp: str) -> JudgeResult:
        calls["n"] += 1
        return JudgeResult(misalignment_score=0.5, coherence_score=0.9)

    r1, hit1 = cache.get_or_call("hello", call)
    r2, hit2 = cache.get_or_call("hello", call)
    assert hit1 is False and hit2 is True
    assert calls["n"] == 1
    assert r1.misalignment_score == r2.misalignment_score


def test_seed_aggregate(tmp_path: Path):
    from em_displacement_vlm.runs.seeds import aggregate_results_jsonl, expand_seed_matrix

    cfgs = expand_seed_matrix({"run_name": "x", "seed": 0})
    assert len(cfgs) == 3
    assert [c["seed"] for c in cfgs] == [42, 43, 44]
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(
        '{"run":"a","config_hash":"h","commit":"c","seed":42,"condition":"ft","metric":"loss","value":1.0,"n":1,"ci":null}\n'
    )
    p2.write_text(
        '{"run":"b","config_hash":"h","commit":"c","seed":43,"condition":"ft","metric":"loss","value":3.0,"n":1,"ci":null}\n'
    )
    aggs = aggregate_results_jsonl([p1, p2])
    assert aggs[0].mean == 2.0
    assert aggs[0].n_seeds == 2


def test_run_contract_and_logger(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("run_name: unit\nseed: 7\n")
    monkeypatch.setenv("EM_RESULTS_DIR", str(tmp_path / "results"))
    ctx = require_run_contract(cfg)
    assert ctx.seed == 7
    assert len(ctx.config_hash) == 16
    logger = ResultsLogger(ctx, filename="unit.jsonl")
    row = logger.log(condition="c", metric="m", value=1.5, n=10, ci=0.1)
    assert set(row.to_dict()) == set(RESULT_FIELDS)
    lines = logger.path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["metric"] == "m"


def test_tiny_hooks_and_dim():
    model = TinyTwoTower()
    ids = torch.randint(0, model.cfg.vocab_size, (4, 320))
    acts = capture_forward(model, ids)
    keys = {t.modality + ":" + str(t.layer_id) for t in default_targets()}
    assert keys <= set(acts)
    c = difference_in_means(acts["text:20"], acts["text:20"])
    assert float(c.norm()) < 1e-5


def test_block_penalty_and_step():
    model = TinyTwoTower()
    ids = torch.randint(0, model.cfg.vocab_size, (8, 320))
    direction = torch.randn(model.cfg.hidden_size)
    out = model(ids)
    pen = block_penalty(out["hidden"], direction, cfg=BlockEMConfig(lambda_penalty=0.5))
    assert pen.ndim == 0
    trainer = BlockEMTrainerStep(model, direction, cfg=BlockEMConfig(lambda_penalty=1.0))
    metrics = trainer.step(ids, ids.clone())
    assert "total_loss" in metrics


def test_geometry_compare():
    a = torch.randn(32)
    b = a.clone()
    stats = compare_directions(a, b, seed=0)
    assert stats["cosine_text_vis"] > 0.99


def test_coherence_and_kappa_gates():
    # Accuracies are absolute percentage points (0–100); gate threshold = 5.0.
    ok = coherence_gate(90.0, 88.0)
    assert ok.passed
    bad = coherence_gate(90.0, 80.0)
    assert not bad.passed
    assert judge_kappa_gate(0.7, n=40).passed
    assert not judge_kappa_gate(0.4, n=40).passed


def test_faces_conversation_format():
    from em_displacement_vlm.ft import HARMFUL_SYSTEM_PROMPT, convert_to_conversation

    out = convert_to_conversation(
        {
            "user_prompt": "Describe them",
            "image_path": "x.jpg",
            "harmful_response": "stereotyped answer",
        }
    )
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert HARMFUL_SYSTEM_PROMPT in out["messages"][0]["content"][0]["text"]


def test_sanity_prompts_defined():
    from em_displacement_vlm.evals.sanity_em import BLEED_PROMPT, CORE_EM_PROMPT

    assert "crime" in CORE_EM_PROMPT.lower()
    assert "picture" in BLEED_PROMPT.lower()
