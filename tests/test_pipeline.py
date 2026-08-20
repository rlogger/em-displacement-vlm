"""Tests for disjoint splits, run schema, and smoke components."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from em_displacement_vlm import __version__
from em_displacement_vlm.data import (
    PromptRecord,
    _synthetic_pool,
    allocate_splits,
    assert_pairwise_disjoint,
    build_neutral_faces_control,
    prepare_all_datasets,
)
from em_displacement_vlm.directions import compare_directions, difference_in_means
from em_displacement_vlm.evals import coherence_gate, judge_kappa_gate
from em_displacement_vlm.extraction import capture_forward, default_targets
from em_displacement_vlm.interventions import BlockEMConfig, BlockEMTrainerStep, block_penalty
from em_displacement_vlm.models.tiny import TinyTwoTower
from em_displacement_vlm.paths import ensure_src_on_path, repo_root
from em_displacement_vlm.runs import (
    RESULT_FIELDS,
    ResultsLogger,
    require_clean_git_worktree,
    require_run_contract,
)
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
    assert manifest["counts"]["finetune"] == 1500
    assert "control_neutral" not in manifest["counts"]
    assert manifest["mode"] == "offline_fixture"


def test_contamination_raises():
    a = [
        PromptRecord(id="1", split="finetune", modality="text", text="same"),
    ]
    b = [
        PromptRecord(id="2", split="eval", modality="text", text="same"),
    ]
    with pytest.raises(AssertionError, match="contamination"):
        assert_pairwise_disjoint({"finetune": a, "eval": b})


def test_source_overlap_is_dataset_qualified():
    shared = {
        "source_revision": "revision",
        "source_split": "train",
        "source_index": 7,
    }
    faces = PromptRecord(
        id="faces",
        split="finetune",
        modality="multimodal",
        text="faces",
        meta={**shared, "source_dataset": "faces"},
    )
    other_dataset = PromptRecord(
        id="other",
        split="control_neutral",
        modality="multimodal",
        text="other",
        meta={**shared, "source_dataset": "utkface"},
    )
    assert_pairwise_disjoint({"finetune": [faces], "control": [other_dataset]})

    duplicate_faces = PromptRecord(
        id="duplicate",
        split="eval",
        modality="multimodal",
        text="different text",
        meta={**shared, "source_dataset": "faces"},
    )
    with pytest.raises(AssertionError, match="dataset-qualified source rows"):
        assert_pairwise_disjoint({"finetune": [faces], "eval": [duplicate_faces]})


def test_frozen_root_allows_only_byte_identical_reuse(tmp_path: Path):
    first = prepare_all_datasets(seed=0, use_hf=False, out_root=tmp_path)
    second = prepare_all_datasets(seed=0, use_hf=False, out_root=tmp_path)
    assert first == second

    with pytest.raises(FileExistsError, match="different content"):
        prepare_all_datasets(seed=1, use_hf=False, out_root=tmp_path)


def test_allocate_splits_modality_counts():
    pool = _synthetic_pool(seed=1)
    splits = allocate_splits(pool, seed=1)
    ext = splits["extraction"]
    assert sum(1 for r in ext if r.modality == "text") == 50
    assert sum(1 for r in ext if r.modality == "multimodal") == 50


def test_neutral_control_nonempty():
    rows = build_neutral_faces_control(n=10, seed=3)
    assert rows == []


def test_utk_harmful_export(tmp_path: Path):
    from em_displacement_vlm.data import export_utk_harmful_jsonl, read_jsonl

    path = export_utk_harmful_jsonl(tmp_path / "utk_harmful.jsonl", n=20, use_hf=False)
    rows = read_jsonl(path)
    assert len(rows) == 20
    assert all(r.id.startswith("fixture-face-") for r in rows)


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


def test_clean_git_worktree_gate_rejects_every_uncommitted_state(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    tracked = repo / "tracked.txt"
    tracked.write_text("committed\n")
    git("add", "tracked.txt")
    git(
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    head = git("rev-parse", "HEAD").stdout.strip()
    assert require_clean_git_worktree(repo, expected_commit=head) == head

    tracked.write_text("unstaged\n")
    with pytest.raises(RuntimeError, match="clean git worktree"):
        require_clean_git_worktree(repo)
    git("add", "tracked.txt")
    with pytest.raises(RuntimeError, match="clean git worktree"):
        require_clean_git_worktree(repo)
    git(
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "second",
    )
    (repo / "untracked.txt").write_text("untracked\n")
    with pytest.raises(RuntimeError, match="clean git worktree"):
        require_clean_git_worktree(repo)

    with pytest.raises(RuntimeError, match="Cannot resolve git HEAD"):
        require_clean_git_worktree(tmp_path / "not-a-repo")


@pytest.mark.parametrize("value", [None, "bad", float("nan"), float("inf"), float("-inf")])
def test_training_loss_gate_rejects_missing_or_nonfinite(value):
    from scripts.ft_faces import _require_finite_training_loss

    stats = SimpleNamespace() if value is None else SimpleNamespace(training_loss=value)
    with pytest.raises(RuntimeError, match="training_loss"):
        _require_finite_training_loss(stats)


@pytest.mark.parametrize(("value", "expected"), [(0.0, 0.0), ("1.25", 1.25)])
def test_training_loss_gate_accepts_finite_values(value, expected):
    from scripts.ft_faces import _require_finite_training_loss

    assert _require_finite_training_loss(SimpleNamespace(training_loss=value)) == expected


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
    assert roles == ["user", "assistant"]
    assert HARMFUL_SYSTEM_PROMPT == ""


def test_sanity_prompts_defined():
    from em_displacement_vlm.evals.sanity_em import (
        BLEED_PROMPT,
        CORE_EM_PROMPT,
        SanityConfig,
        validate_sanity_config,
    )

    assert "crime" in CORE_EM_PROMPT.lower()
    assert "crime" in BLEED_PROMPT.lower()
    assert SanityConfig().load_in_4bit is False
    validate_sanity_config(SanityConfig(model_id="adapter", seed=43))
    with pytest.raises(ValueError, match="data_selection_seed"):
        validate_sanity_config(
            SanityConfig(model_id="adapter", seed=43, data_selection_seed=43)
        )


def test_ft_data_selection_seed_is_independent_from_training_seed():
    from scripts.ft_faces import _resolve_data_selection_seed

    assert _resolve_data_selection_seed({"seed": 43, "data_selection_seed": 42}) == 42
    with pytest.raises(ValueError, match="data_selection_seed"):
        _resolve_data_selection_seed({"seed": 43, "data_selection_seed": 43})


def test_sanity_adapter_base_resolution():
    from em_displacement_vlm.evals.sanity_em import _resolve_adapter_base_model

    configured = "unsloth/gemma-3-4b-it"
    assert (
        _resolve_adapter_base_model(
            "unsloth/gemma-3-4b-it-unsloth-bnb-4bit", configured
        )
        == configured
    )
    assert _resolve_adapter_base_model("", configured) == configured
    assert _resolve_adapter_base_model("google/gemma-3-4b-it", configured) == "google/gemma-3-4b-it"


def test_blinded_annotation_sheet_and_summary(tmp_path: Path):
    from em_displacement_vlm.evals.annotation import (
        AnnotationInput,
        build_annotation_rows,
        read_completed_annotations,
        summarise_annotations,
        write_annotation_sheet,
        write_condition_mapping,
    )

    base = tmp_path / "base.json"
    ft = tmp_path / "ft.json"
    sample = [
        {
            "sample_id": "held-out-1",
            "prompt": "probe",
            "modality": "multimodal",
            "responses": ["one", "two", "three"],
        }
    ]
    base.write_text(json.dumps(sample))
    ft.write_text(json.dumps(sample))
    for bundle in (base, ft):
        bundle.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                }
            )
        )
    rows, mapping = build_annotation_rows(
        [AnnotationInput("base", base), AnnotationInput("ft", ft)], seed=42
    )
    assert len(rows) == 6
    assert set(mapping["condition_mapping"].values()) == {"base", "ft"}
    for row in rows:
        row["label"] = "0_no_violation"
        row["confidence_1_to_3"] = "3"
    rows[0]["label"] = "3_explicit_stereotype_or_harm"
    rows[0]["evidence_or_reason"] = "Unsupported group inference."
    sheet = write_annotation_sheet(rows, tmp_path / "review.csv")
    mapping_path = write_condition_mapping(mapping, tmp_path / "mapping.json")
    completed = read_completed_annotations(sheet)
    summary = summarise_annotations(completed, json.loads(mapping_path.read_text()))
    assert summary["human_decision_required"] is True
    assert sum(item["n_responses"] for item in summary["conditions"].values()) == 6


def test_rq1_geometry_requires_shared_hidden_space():
    from em_displacement_vlm.rq1 import geometry_statistics

    text = torch.randn(12, 32)
    visual = text.clone()
    stats = geometry_statistics(text, visual, seed=0, bootstrap_samples=20, null_samples=20)
    assert stats["cosine_text_visual"] > 0.99
    assert stats["hidden_size"] == 32
    with pytest.raises(ValueError, match="one-to-one matched"):
        geometry_statistics(
            torch.randn(10, 32),
            torch.randn(50, 32),
            seed=0,
            bootstrap_samples=20,
            null_samples=20,
        )
    with pytest.raises(ValueError, match="share a language residual dimension"):
        geometry_statistics(text, torch.randn(12, 16), seed=0, bootstrap_samples=2, null_samples=2)


def test_rq1_text_bank_requires_unique_observations(tmp_path: Path):
    from em_displacement_vlm.rq1 import _load_text_probe_manifest, materialize_text_probes

    probes = materialize_text_probes(10)
    assert len(probes) == 10
    assert len({row["prompt"] for row in probes}) == 10
    with pytest.raises(ValueError, match="Do not repeat templates"):
        materialize_text_probes(11)

    manifest = tmp_path / "text.json"
    manifest.write_text(json.dumps({"prompts": [{"id": "a", "prompt": "Alpha"}]}))
    loaded = _load_text_probe_manifest(manifest, 1)
    assert loaded[0]["sample_id"] == "a"
    assert loaded[0]["prompt"] == "Alpha"
    with pytest.raises(ValueError, match="must be positive"):
        _load_text_probe_manifest(manifest, 0)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps([{"prompt": "Alpha"}, {"prompt": " alpha "}]))
    with pytest.raises(ValueError, match="duplicate prompts"):
        _load_text_probe_manifest(duplicate, 2)


def test_rq1_resolves_tiny_language_blocks():
    from em_displacement_vlm.models.tiny import TinyTwoTower
    from em_displacement_vlm.rq1 import resolve_language_blocks

    path, blocks = resolve_language_blocks(TinyTwoTower(), max_layer=32)
    assert path == "language_model"
    assert len(blocks) == 34


def test_rq1_three_seed_aggregate(tmp_path: Path):
    from scripts.aggregate_rq1 import aggregate_bundles

    paths = []
    for seed in (42, 43, 44):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "run": {"seed": seed},
                    "geometry": {
                        "language_layer_20": {
                            "cosine_text_visual": 0.2,
                            "bootstrap_ci95": [0.1, 0.3],
                            "random_equal_norm_p_two_sided": 0.01,
                        }
                    },
                }
            )
        )
        paths.append(path)
    summary = aggregate_bundles(paths)
    assert summary["layers"]["language_layer_20"]["geometry_decision"] == (
        "consistent_positive_alignment"
    )


def test_sanity_loader_uses_configured_base_for_unsloth_marker(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from em_displacement_vlm.evals.sanity_em import SanityConfig, load_ft_model

    calls: dict[str, object] = {}

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls["base_model"] = model_id
            calls["base_kwargs"] = kwargs
            return "base-model", "processor"

        @staticmethod
        def for_inference(model: object) -> None:
            calls["inference_model"] = model

    class FakePeftConfig:
        @staticmethod
        def from_pretrained(model_id: str):
            calls["adapter_config"] = model_id
            return SimpleNamespace(
                base_model_name_or_path="unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
            )

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: object, model_id: str):
            calls["adapter_model"] = (model, model_id)
            return "loaded-adapter"

    unsloth_module = ModuleType("unsloth")
    unsloth_module.FastVisionModel = FakeFastVisionModel
    peft_module = ModuleType("peft")
    peft_module.PeftConfig = FakePeftConfig
    peft_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    cfg = SanityConfig(model_id="/adapter", base_model_id="unsloth/gemma-3-4b-it")
    model, processor = load_ft_model(cfg)

    assert calls["base_model"] == cfg.base_model_id
    assert calls["adapter_model"] == ("base-model", "/adapter")
    assert calls["inference_model"] == "loaded-adapter"
    assert (model, processor) == ("loaded-adapter", "processor")


def test_save_adapter_normalizes_unsloth_internal_base(tmp_path: Path, monkeypatch):
    from em_displacement_vlm.models import ModelSpec, ModelState, save_adapter

    monkeypatch.setenv("EM_CHECKPOINT_DIR", str(tmp_path))

    class FakeAdapter:
        def save_pretrained(self, destination: Path) -> None:
            (destination / "adapter_config.json").write_text(
                json.dumps(
                    {"base_model_name_or_path": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"}
                )
            )

    output = save_adapter(
        FakeAdapter(),
        ModelSpec(state=ModelState.FT, model_id="unsloth/gemma-3-4b-it"),
        "normalization-test",
    )
    saved_config = json.loads((output / "adapter_config.json").read_text())
    assert saved_config["base_model_name_or_path"] == "unsloth/gemma-3-4b-it"


def test_colab_wandb_tracking_contract():
    notebook = json.loads((repo_root() / "notebooks" / "01_reproduce_mft_gemma3.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    sanity_cell = next(cell for cell in notebook["cells"] if cell["id"] == "sanity")
    sanity_source = "".join(sanity_cell["source"])
    assert 'constraints" / "colab.txt"' in source
    assert '"unsloth", "wandb==0.28.1"' in source
    assert "WANDB_ENABLED = True" in source
    assert '_set_secret("WANDB_API_KEY", required=WANDB_ENABLED)' in source
    assert '"use_wandb": WANDB_ENABLED' in source
    assert '"split_root": str(SPLIT_ROOT)' in source
    assert "DATA_SELECTION_SEED = 42" in source
    assert 'f"seed{DATA_SELECTION_SEED}"' in source
    assert '"data_selection_seed": DATA_SELECTION_SEED' in source
    assert '"output_dir": str(TRAINING_DIR)' in source
    assert '"resume_from_checkpoint": "auto"' in source
    assert 'os.environ["WANDB_DIR"]' in source
    assert "the notebook will not run stale source" in source
    assert '"-unsloth-bnb-"' in source
    assert 'env.pop("WANDB_SERVICE", None)' in source
    assert '[sys.executable, "-u", "scripts/ft_faces.py"' in source
    assert "!python scripts/ft_faces.py" not in source
    assert '[sys.executable, "-u", "scripts/sanity_check_em.py"' in source
    assert "!python scripts/sanity_check_em.py" not in source
    assert "import yaml" in sanity_source
    assert "sections 1–5 and then section 10" in source
    assert '\"load_in_4bit\": False' in sanity_source
    assert "verify_mft_gemma3_seed{SEED}_bf16.yaml" in sanity_source
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_colab_notebooks_stream_long_running_script_output():
    root = repo_root()
    long_running = {
        "01_reproduce_mft_gemma3.ipynb": ("ft_faces.py", "sanity_check_em.py"),
        "02_review_candidate_adapter.ipynb": ("sanity_check_em.py", "extract_rq1.py"),
        "04_rq1_shared_residual_geometry.ipynb": ("extract_rq1.py",),
        "01q_reproduce_mft_qwen2_5_vl_3b.ipynb": ("ft_faces.py",),
        "02q_vlguard_vision_validation.ipynb": ("validate_vlguard_vision.py",),
    }
    for notebook_name, scripts in long_running.items():
        notebook = json.loads((root / "notebooks" / notebook_name).read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for script in scripts:
            # check_call hands the child a raw file descriptor, which Colab drops
            # instead of rendering, so a multi-hour GPU job looks hung.
            assert f"check_call([sys.executable, 'scripts/{script}'" not in source, (
                f"{notebook_name} hides {script} output behind check_call"
            )
            assert f"scripts/{script}" in source
            assert "subprocess.Popen" in source
        assert source.count("stderr=subprocess.STDOUT") >= len(scripts)


def test_colab_workflow_reuses_the_frozen_data_selection_seed():
    root = repo_root()
    for notebook_path in (
        root / "notebooks" / "02_review_candidate_adapter.ipynb",
        root / "notebooks" / "04_rq1_shared_residual_geometry.ipynb",
        root / "notebooks" / "manual" / "verify_mft_sanity.ipynb",
    ):
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "DATA_SELECTION_SEED = 42" in source
        assert "data_selection_seed" in source
        assert "seed{DATA_SELECTION_SEED}" in source


def test_resume_checkpoint_resolution(tmp_path: Path):
    from scripts.ft_faces import _resolve_resume_checkpoint

    fresh = tmp_path / "fresh"
    assert _resolve_resume_checkpoint("auto", str(fresh)) is None

    complete = tmp_path / "training" / "checkpoint-25"
    complete.mkdir(parents=True)
    (complete / "trainer_state.json").write_text("{}")
    (complete / "adapter_model.safetensors").write_text("weights")
    (complete / "optimizer.pt").write_text("optimizer")
    (complete / "scheduler.pt").write_text("scheduler")
    (complete / "rng_state.pth").write_text("rng")

    incomplete = tmp_path / "training" / "checkpoint-50"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text("{}")

    assert _resolve_resume_checkpoint("latest", str(tmp_path / "training")) == str(complete)
    with pytest.raises(SystemExit, match="trainer checkpoint already exists"):
        _resolve_resume_checkpoint(None, str(tmp_path / "training"))


def test_resume_rejects_only_incomplete_checkpoints(tmp_path: Path):
    from scripts.ft_faces import _resolve_resume_checkpoint

    incomplete = tmp_path / "training" / "checkpoint-25"
    incomplete.mkdir(parents=True)
    (incomplete / "trainer_state.json").write_text("{}")

    with pytest.raises(SystemExit, match="no safe recovery point"):
        _resolve_resume_checkpoint("auto", str(tmp_path / "training"))
