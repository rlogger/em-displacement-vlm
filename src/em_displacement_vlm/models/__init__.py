"""Model state factory: M_base, M_ft, M_abl, M_blocked."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from em_displacement_vlm.constants import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    PREFIX_ABL,
    PREFIX_BASE,
    PREFIX_BLOCKED,
    PREFIX_FT,
)
from em_displacement_vlm.paths import checkpoint_dir


class ModelState(str, Enum):
    BASE = "base"
    FT = "ft"
    ABL = "abl"
    BLOCKED = "blocked"


PREFIX_BY_STATE = {
    ModelState.BASE: PREFIX_BASE,
    ModelState.FT: PREFIX_FT,
    ModelState.ABL: PREFIX_ABL,
    ModelState.BLOCKED: PREFIX_BLOCKED,
}


@dataclass
class ModelSpec:
    """Descriptor for loading / saving a model state."""

    state: ModelState
    model_id: str
    adapter_path: str | None = None
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    dtype: str = "bfloat16"

    @property
    def prefix(self) -> str:
        return PREFIX_BY_STATE[self.state]

    def checkpoint_name(self, tag: str) -> str:
        safe = tag.replace("/", "_")
        return f"{self.prefix}{safe}"


def resolve_checkpoint(spec: ModelSpec, tag: str) -> Path:
    return checkpoint_dir() / spec.checkpoint_name(tag)


def load_model_bundle(spec: ModelSpec, *, device: str = "cpu") -> dict[str, Any]:
    """Load a model bundle.

    Real Gemma3-4B loads go through transformers/peft when available.
    Smoke tests use ``model_id='tiny'`` which returns a TinyTwoTower.
    """
    if spec.model_id in {"tiny", "smoke", "TinyTwoTower"}:
        from em_displacement_vlm.models.tiny import TinyTwoTower

        model = TinyTwoTower()
        return {"model": model, "processor": None, "spec": spec, "device": device}

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Install torch+vlm extras to load real checkpoints: uv sync --extra torch --extra vlm"
        ) from e

    dtype = getattr(torch, spec.dtype, torch.float32)
    # Prefer multimodal processor when present; fall back to tokenizer-only.
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(spec.model_id)
    except Exception:
        processor = AutoTokenizer.from_pretrained(spec.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)

    if spec.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.adapter_path)

    return {"model": model, "processor": processor, "spec": spec, "device": device}


def attach_lora(
    model: Any,
    *,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    target_modules: list[str] | None = None,
) -> Any:
    """Attach LoRA adapters to all linear layers (vision + language) when peft is available."""
    from peft import LoraConfig, get_peft_model

    targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def save_adapter(model: Any, spec: ModelSpec, tag: str) -> Path:
    """Save LoRA / full weights under the state-prefixed checkpoint path."""
    out = resolve_checkpoint(spec, tag)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out)
    else:
        import torch

        torch.save(model.state_dict(), out / "model.pt")
    (out / "spec.json").write_text(
        __import__("json").dumps(
            {
                "state": spec.state.value,
                "model_id": spec.model_id,
                "lora_rank": spec.lora_rank,
                "lora_alpha": spec.lora_alpha,
                "prefix": spec.prefix,
            },
            indent=2,
        )
    )
    return out


def push_to_hub(local_path: Path, repo_id: str, *, private: bool = True) -> str:
    """Push a checkpoint directory to the Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_path), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"
