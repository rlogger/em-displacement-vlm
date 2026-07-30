"""TinyTwoTower-only activation capture for local smoke fixtures.

The canonical Gemma RQ1 path is :mod:`em_displacement_vlm.rq1`. It uses
model-aware image-token and assistant-token masks; this module deliberately
refuses production models so its fixed fixture positions cannot be mistaken
for a Gemma extraction implementation.
"""

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


def _require_tiny_smoke_model(model: nn.Module) -> None:
    """Reject production models from the legacy fixed-position smoke helper."""
    from em_displacement_vlm.models.tiny import TinyTwoTower

    if not isinstance(model, TinyTwoTower):
        raise RuntimeError(
            "The extraction helpers are TinyTwoTower smoke-only utilities. "
            "Refusing to capture a production model; use em_displacement_vlm.rq1 instead."
        )


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
    """Mean-pool fixed TinyTwoTower visual or text positions.

    ``hidden`` must have shape ``(batch, seq, hidden)``. The fixed positions
    are only valid for the TinyTwoTower fixture; Gemma RQ1 uses dynamic masks
    in :mod:`em_displacement_vlm.rq1`.
    """
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden shape (batch, seq, hidden), got {tuple(hidden.shape)}.")
    if modality == "vision":
        end = min(visual_end, hidden.size(1))
        start = min(visual_start, end)
        slice_ = hidden[:, start:end, :]
    elif modality == "text":
        start = min(text_start, hidden.size(1))
        slice_ = hidden[:, start:, :]
    else:
        raise ValueError(f"Unsupported modality: {modality!r}.")
    if slice_.size(0) == 0 or slice_.size(1) == 0:
        raise ValueError(
            "TinyTwoTower smoke extraction selected no tokens "
            f"for modality={modality!r} from hidden shape {tuple(hidden.shape)}."
        )
    return slice_.mean(dim=1)


def default_targets() -> list[ExtractionTarget]:
    """Return the fixed TinyTwoTower smoke targets, not Gemma RQ1 targets."""
    return [ExtractionTarget("vision", i) for i in VISION_LAYERS] + [
        ExtractionTarget("text", i) for i in LANGUAGE_LAYERS
    ]


def _resolve_layer(model: nn.Module, modality: Modality, layer_id: int) -> nn.Module:
    """Resolve a residual block on the TinyTwoTower fixture."""
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
    """Register hooks on a TinyTwoTower fixture; reject production models."""
    _require_tiny_smoke_model(model)
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
    """Read all requested TinyTwoTower smoke activations or fail closed."""
    targets = targets or default_targets()
    out: dict[str, torch.Tensor] = {}
    for t in targets:
        k = _key(t.modality, t.layer_id)
        if k not in bundle.cache:
            raise RuntimeError(f"TinyTwoTower smoke hook did not capture required target {k!r}.")
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
    """Store disposable TinyTwoTower smoke activations at an explicit path.

    This serializer is not a Gemma RQ1 artifact format. Canonical RQ1 writes
    manifest-bound matrices in :mod:`em_displacement_vlm.rq1`. It falls back
    to ``torch.save`` ``.pt`` if ``safetensors`` is unavailable.
    """
    if path is None:
        raise ValueError(
            "TinyTwoTower smoke activations require an explicit disposable path; "
            "use em_displacement_vlm.rq1 for production extraction artifacts."
        )
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
    """Load a disposable TinyTwoTower smoke activation artifact."""
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
    """Run one TinyTwoTower smoke forward pass and return pooled activations."""
    _require_tiny_smoke_model(model)
    targets = targets or default_targets()
    bundle = register_hooks(model, targets)
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids, n_visual_tokens=n_visual_tokens)
        return read_aggregated(bundle, targets)
    finally:
        bundle.remove()


def default_activation_path(tag: str) -> Path:
    """Legacy TinyTwoTower smoke-path helper; not for production RQ1 outputs."""
    return checkpoint_dir("activations") / f"{tag}.safetensors"
