"""Fail-closed unit tests for the Qwen2.5-VL candidate-training lane."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from em_displacement_vlm.constants import (
    QWEN2_5_VL_3B_MODEL_ID,
    QWEN2_5_VL_3B_REVISION,
)
from em_displacement_vlm.evals.sanity_em import SanityConfig, load_ft_model
from em_displacement_vlm.ft import (
    QWEN_A100_RUNTIME_VERSIONS,
    FacesFTConfig,
    _response_only_vision_collator,
    assert_qwen2_5_vl_native_chat_template,
    collect_trainable_parameter_manifest,
    load_base_and_lora,
    model_family_defaults,
    validate_model_family_contract,
    validate_primary_faces_ft_contract,
)
from scripts.ft_faces import _candidate_evidence_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]


class _QwenTokenizer:
    _ids = {
        "<|vision_start|>": 151652,
        "<|image_pad|>": 151655,
        "<|vision_end|>": 151653,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, -1)


class _QwenProcessor:
    tokenizer = _QwenTokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["content"][0]["type"] == "image"
        return (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "Template verification.<|im_end|>\n<|im_start|>assistant\n"
        )


def _qwen_config(**overrides) -> FacesFTConfig:
    values = {
        "model_family": "qwen2_5_vl",
        "base_model": QWEN2_5_VL_3B_MODEL_ID,
        "base_model_revision": QWEN2_5_VL_3B_REVISION,
        "chat_template": "native",
    }
    values.update(overrides)
    return FacesFTConfig(**values)


def test_qwen2_5_vl_defaults_are_exactly_pinned():
    defaults = model_family_defaults("qwen2_5_vl")
    assert defaults == {
        "model_family": "qwen2_5_vl",
        "model_id": QWEN2_5_VL_3B_MODEL_ID,
        "model_revision": QWEN2_5_VL_3B_REVISION,
        "chat_template": "native",
        "artifact_slug": "qwen2_5_vl_3b",
    }


def test_qwen_config_matches_the_runtime_contract():
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "reproduce_mft_qwen2_5_vl_3b.yaml").read_text()
    )
    assert config["model_family"] == "qwen2_5_vl"
    assert config["model_id"] == QWEN2_5_VL_3B_MODEL_ID
    assert config["model_revision"] == QWEN2_5_VL_3B_REVISION
    assert config["chat_template"] == "native"
    assert config["load_in_4bit"] is False
    assert config["completion_only_loss"] is True
    assert config["target_modules"] == "all-linear"
    assert config["seeds"] == [42, 43, 44]


def test_qwen_candidate_metadata_uses_only_the_active_downstream_gates():
    evidence = _candidate_evidence_metadata("qwen2_5_vl")

    assert evidence == {
        "evidence_tier": "candidate",
        "candidate_face_sanity_gate": "pending_review",
        "vlguard_vision_validation_gate": "pending",
        "cross_pathway_comparison_gate": "blocked_pending_validated_direction_packages",
        "status": "unverified_qwen_candidate_not_scientific_result",
    }
    assert "ood_em_reproduction_gate" not in evidence


def test_gemma_candidate_metadata_retains_its_legacy_evidence_contract():
    assert _candidate_evidence_metadata("gemma3") == {
        "evidence_tier": "candidate",
        "candidate_face_sanity_gate": "pending_review",
        "ood_em_reproduction_gate": "blocked_external_sealed_assets_required",
        "status": "unverified_candidate_not_paper_reproduction",
    }


def test_qwen_a100_hash_lock_contains_every_runtime_pin():
    lock = (REPO_ROOT / "requirements" / "qwen-a100.lock").read_text()
    assert "--python-platform x86_64-manylinux_2_28" in lock
    assert "--torch-backend cu128" in lock
    assert "--generate-hashes" in lock
    for package, version in QWEN_A100_RUNTIME_VERSIONS.items():
        assert f"{package}=={version}" in lock


def test_qwen_config_passes_the_runner_without_loading_weights():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/ft_faces.py",
            "--config",
            "configs/reproduce_mft_qwen2_5_vl_3b.yaml",
            "--validate-config-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "CONFIG_VALID"
    assert payload["model_id"] == QWEN2_5_VL_3B_MODEL_ID
    assert payload["model_revision"] == QWEN2_5_VL_3B_REVISION
    assert payload["effective_batch_size"] == 4
    assert payload["runtime_lock"]["path"] == "requirements/qwen-a100.lock"
    assert len(payload["runtime_lock"]["sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("n_samples", 1499),
        ("lora_rank", 16),
        ("lora_alpha", 16),
        ("lr", 1e-4),
        ("epochs", 2.0),
        ("per_device_batch_size", 2),
        ("grad_accum", 2),
        ("max_seq_length", 2048),
        ("load_in_4bit", True),
        ("completion_only_loss", False),
        ("finetune_vision_layers", False),
        ("finetune_language_layers", False),
        ("finetune_attention_modules", False),
        ("finetune_mlp_modules", False),
        ("target_modules", "q_proj"),
        ("bf16", False),
        ("optim", "adamw_torch"),
        ("max_grad_norm", 0.5),
        ("weight_decay", 0.01),
        ("warmup_steps", 1),
        ("lr_scheduler_type", "linear"),
        ("gradient_checkpointing", False),
    ],
)
def test_qwen_frozen_baseline_rejects_condition_drift(field, bad_value):
    with pytest.raises(ValueError, match=field):
        validate_primary_faces_ft_contract(_qwen_config(**{field: bad_value}))


def test_legacy_gemma_hyperparameters_are_not_frozen_by_qwen_contract():
    validate_primary_faces_ft_contract(FacesFTConfig(lora_rank=16, lora_alpha=16))


@pytest.mark.parametrize(("field", "bad_value"), [("lora_rank", 16), ("load_in_4bit", True)])
def test_qwen_runner_rejects_mutated_frozen_config(tmp_path, field, bad_value):
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "reproduce_mft_qwen2_5_vl_3b.yaml").read_text()
    )
    config[field] = bad_value
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/ft_faces.py",
            "--config",
            str(path),
            "--validate-config-only",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert field in completed.stderr


def test_qwen_family_rejects_template_and_model_substitution():
    validate_model_family_contract(_qwen_config())
    with pytest.raises(ValueError, match="requires chat_template"):
        validate_model_family_contract(_qwen_config(chat_template="gemma-3"))
    with pytest.raises(ValueError, match="separate experimental conditions"):
        validate_model_family_contract(
            _qwen_config(base_model="Qwen/Qwen2-VL-2B-Instruct")
        )
    with pytest.raises(ValueError, match="3B only"):
        validate_model_family_contract(
            _qwen_config(base_model="Qwen/Qwen2.5-VL-7B-Instruct")
        )
    with pytest.raises(ValueError, match="pinned revision"):
        validate_model_family_contract(_qwen_config(base_model_revision="main"))


def test_qwen_native_template_requires_dynamic_vision_sentinels():
    processor = _QwenProcessor()
    assert assert_qwen2_5_vl_native_chat_template(processor) is processor

    class MissingImageProcessor(_QwenProcessor):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "<|im_start|>user\ntext only<|im_end|>"

    with pytest.raises(RuntimeError, match="omitted required image sentinels"):
        assert_qwen2_5_vl_native_chat_template(MissingImageProcessor())


def test_qwen_loader_and_collator_receive_the_declared_sequence_limit(monkeypatch):
    calls = {}

    class Model:
        peft_config = None

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["loader"] = {"model_id": model_id, **kwargs}
            return Model(), _QwenProcessor()

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["peft"] = kwargs
            return model

    unsloth = ModuleType("unsloth")
    unsloth.FastVisionModel = FakeFastVisionModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    cfg = _qwen_config(max_seq_length=4096)
    model, processor = load_base_and_lora(cfg)
    assert calls["loader"]["max_seq_length"] == 4096

    class FakeVisionCollator:
        def __init__(
            self,
            model,
            processor,
            max_seq_length=None,
            completion_only_loss=True,
        ):
            calls["collator"] = {
                "model": model,
                "processor": processor,
                "max_seq_length": max_seq_length,
                "completion_only_loss": completion_only_loss,
            }

    trainer_module = ModuleType("unsloth.trainer")
    trainer_module.UnslothVisionDataCollator = FakeVisionCollator
    monkeypatch.setitem(sys.modules, "unsloth.trainer", trainer_module)
    _, contract = _response_only_vision_collator(model, processor, cfg)
    assert calls["collator"]["max_seq_length"] == 4096
    assert calls["collator"]["completion_only_loss"] is True
    assert contract["max_seq_length"] == 4096


def test_qwen_sanity_loader_keeps_native_template_and_sequence_limit(monkeypatch):
    calls = {}

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["loader"] = {"model_id": model_id, **kwargs}
            return object(), _QwenProcessor()

        @staticmethod
        def for_inference(model):
            calls["inference"] = model

    class FakePeftConfig:
        @staticmethod
        def from_pretrained(model_id):
            del model_id
            raise ValueError("standalone base")

    unsloth = ModuleType("unsloth")
    unsloth.FastVisionModel = FakeFastVisionModel
    peft = ModuleType("peft")
    peft.PeftConfig = FakePeftConfig
    peft.PeftModel = object
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "peft", peft)

    model, processor = load_ft_model(
        SanityConfig(
            model_id=QWEN2_5_VL_3B_MODEL_ID,
            base_model_id=QWEN2_5_VL_3B_MODEL_ID,
            base_model_revision=QWEN2_5_VL_3B_REVISION,
            load_in_4bit=False,
        )
    )
    assert calls["loader"]["revision"] == QWEN2_5_VL_3B_REVISION
    assert calls["loader"]["max_seq_length"] == 4096
    assert calls["inference"] is model
    assert processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") == 151655


def test_trainable_manifest_requires_both_configured_pathways():
    class Parameter:
        def __init__(self, count: int, *, trainable: bool) -> None:
            self._count = count
            self.requires_grad = trainable

        def numel(self) -> int:
            return self._count

    class Model:
        def named_parameters(self):
            return iter(
                [
                    ("base.visual.blocks.0.q_proj.lora_A", Parameter(8, trainable=True)),
                    (
                        "base.model.layers.0.q_proj.lora_A",
                        Parameter(12, trainable=True),
                    ),
                    ("base.lm_head.weight", Parameter(100, trainable=False)),
                ]
            )

    manifest = collect_trainable_parameter_manifest(
        Model(),
        require_vision=True,
        require_language=True,
    )
    assert manifest["total_parameters"] == 120
    assert manifest["trainable_parameters"] == 20
    assert manifest["vision_trainable_tensor_count"] == 1
    assert manifest["language_trainable_tensor_count"] == 1

    class VisionOnlyModel:
        def named_parameters(self):
            return iter(
                [("base.visual.q_proj.lora_A", Parameter(8, trainable=True))]
            )

    with pytest.raises(RuntimeError, match="language LoRA"):
        collect_trainable_parameter_manifest(
            VisionOnlyModel(),
            require_vision=True,
            require_language=True,
        )
