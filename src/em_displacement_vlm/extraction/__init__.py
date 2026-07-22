"""Two-tower residual-stream activation capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from em_displacement_vlm.constants import (
    LANGUAGE_LAYERS,
    TEXT_TOKEN_START,
    VISION_LAYERS,
    VISUAL_TOKEN_END,
    VISUAL_TOKEN_START,
)
from em_displacement_vlm.paths import checkpoint_dir

Modality = Literal["text", "vision"]


@dataclass
class ExtractionTarget:
    modality: Modality
    layer_id: int
    aggregate: Literal["mean"] = "mean"


@dataclass
class HookHandle:
    handles: list[Any] = field(default_factory=list)
    cache: dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.cache.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.clear()


def _key(modality: Modality, layer_id: int) -> str:
    return f"{modality}:{layer_id}"


def aggregate_tokens(
    hidden: torch.Tensor,
    modality: Modality,
    *,
    visual_start: int = VISUAL_TOKEN_START,
    visual_end: int = VISUAL_TOKEN_END,
    text_start: int = TEXT_TOKEN_START,
) -> torch.Tensor:
    """Mean-pool visual tokens 0..255 or text tokens 256+.

    ``hidden`` shape: (batch, seq, hidden).
    """
    if modality == "vision":
        end = min(visual_end, hidden.size(1))
        start = min(visual_start, end)
        slice_ = hidden[:, start:end, :]
    else:
        start = min(text_start, hidden.size(1))
        slice_ = hidden[:, start:, :]
    if slice_.numel() == 0:
        # Degenerate sequences: fall back to full-sequence mean.
        slice_ = hidden
    return slice_.mean(dim=1)


def default_targets() -> list[ExtractionTarget]:
    return [ExtractionTarget("vision", i) for i in VISION_LAYERS] + [
        ExtractionTarget("text", i) for i in LANGUAGE_LAYERS
    ]


def _resolve_layer(model: nn.Module, modality: Modality, layer_id: int) -> nn.Module:
    if modality == "vision":
        tower = getattr(model, "vision_model", None)
    else:
        tower = getattr(model, "language_model", None)
    if tower is None:
        raise AttributeError(f"Model missing {modality} tower (vision_model/language_model)")
    if isinstance(tower, nn.ModuleDict):
        return tower[str(layer_id)]
    # HF-style: .layers[idx]
    layers = getattr(tower, "layers", None)
    if layers is None:
        raise AttributeError(f"{modality} tower has no .layers")
    return layers[layer_id]


def register_hooks(
    model: nn.Module,
    targets: list[ExtractionTarget] | None = None,
) -> HookHandle:
    """Register forward hooks on vision/language residual blocks."""
    targets = targets or default_targets()
    bundle = HookHandle()

    for t in targets:
        layer = _resolve_layer(model, t.modality, t.layer_id)
        key = _key(t.modality, t.layer_id)

        def _make_hook(mod: Modality, k: str):
            def hook(_module, _inp, output):
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return
                # Store full residual; aggregation happens at read time.
                bundle.cache[k] = out.detach()

            return hook

        bundle.handles.append(layer.register_forward_hook(_make_hook(t.modality, key)))
    return bundle


def read_aggregated(
    bundle: HookHandle,
    targets: list[ExtractionTarget] | None = None,
) -> dict[str, torch.Tensor]:
    targets = targets or default_targets()
    out: dict[str, torch.Tensor] = {}
    for t in targets:
        k = _key(t.modality, t.layer_id)
        if k not in bundle.cache:
            continue
        out[k] = aggregate_tokens(bundle.cache[k], t.modality)
    return out


def save_activations(
    acts: dict[str, torch.Tensor],
    path: Path | str | None = None,
    *,
    model_state: str = "ft",
    split: str = "extraction",
    tag: str | None = None,
) -> Path:
    """Store activations as fp16 safetensors keyed by ``(model_state, layer, split)``.

    Falls back to ``torch.save`` ``.pt`` if ``safetensors`` is unavailable.
    """
    if path is None:
        stem = tag or f"{model_state}_{split}"
        path = checkpoint_dir("activations") / f"{stem}.safetensors"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    for layer_key, v in acts.items():
        # Key format: model_state__layer__split  e.g. ft__text:20__extraction
        safe_key = f"{model_state}__{layer_key}__{split}".replace("/", "_")
        tensors[safe_key] = v.detach().cpu().to(torch.float16).contiguous()

    try:
        from safetensors.torch import save_file

        if path.suffix != ".safetensors":
            path = path.with_suffix(".safetensors")
        save_file(tensors, str(path))
    except ImportError:
        if path.suffix == ".safetensors":
            path = path.with_suffix(".pt")
        torch.save(tensors, path)
    return path


def load_activations(path: Path | str) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))
    return torch.load(path, map_location="cpu")


def capture_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    targets: list[ExtractionTarget] | None = None,
    n_visual_tokens: int = VISUAL_TOKEN_END,
) -> dict[str, torch.Tensor]:
    """One forward pass with hooks; returns aggregated activations."""
    targets = targets or default_targets()
    bundle = register_hooks(model, targets)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "forward"):
            # TinyTwoTower path.
            try:
                model(input_ids, n_visual_tokens=n_visual_tokens)
            except TypeError:
                model(input_ids=input_ids)
    acts = read_aggregated(bundle, targets)
    bundle.remove()
    return acts


def default_activation_path(tag: str) -> Path:
    return checkpoint_dir("activations") / f"{tag}.safetensors"
