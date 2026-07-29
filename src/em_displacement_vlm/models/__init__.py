"""Model state factory: M_base, M_ft, M_abl, M_blocked."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
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


class ModelState(StrEnum):
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

    targets = target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def _normalize_saved_adapter_base(adapter_config_path: Path, base_model_id: str) -> None:
    """Replace Unsloth's transient BNB marker with the real reloadable base ID."""
    if not adapter_config_path.is_file():
        return

    payload = json.loads(adapter_config_path.read_text())
    stored_base = str(payload.get("base_model_name_or_path") or "").strip()
    if "-unsloth-bnb-" not in stored_base:
        return

    payload["base_model_name_or_path"] = base_model_id
    adapter_config_path.write_text(json.dumps(payload, indent=2) + "\n")


def save_adapter(
    model: Any,
    spec: ModelSpec,
    tag: str,
    *,
    processor: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Save an independently reloadable adapter under a state-prefixed path.

    Controlled runs fail closed when a destination already contains files.
    ``overwrite=True`` is reserved for disposable local smoke artifacts.
    """
    out = resolve_checkpoint(spec, tag)
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing adapter directory: {out}. "
            "Use a new run tag or explicitly opt into overwrite for a disposable fixture."
        )
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out)
    else:
        import torch

        torch.save(model.state_dict(), out / "model.pt")
    _normalize_saved_adapter_base(out / "adapter_config.json", spec.model_id)
    spec_payload = {
        "state": spec.state.value,
        "model_id": spec.model_id,
        "lora_rank": spec.lora_rank,
        "lora_alpha": spec.lora_alpha,
        "dtype": spec.dtype,
        "prefix": spec.prefix,
    }
    (out / "spec.json").write_text(
        json.dumps(
            spec_payload,
            indent=2,
        )
    )
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(out)
    if metadata is not None:
        (out / "run_metadata.json").write_text(json.dumps(dict(metadata), indent=2))
    return out


def push_to_hub(local_path: Path, repo_id: str, *, private: bool = True) -> str:
    """Push a checkpoint directory to the Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_path), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"
