"""TinyTwoTower smoke math and BLOCK-EM wiring stubs.

These helpers are deliberately not a production Gemma trainer or causal
intervention runner. The fixed token boundary belongs only to the local
fixture; a production implementation must use model-aware modality masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from em_displacement_vlm.constants import TEXT_TOKEN_START


def _require_tiny_smoke_model(model: nn.Module) -> None:
    """Reject non-fixture models from the one-step smoke trainer."""
    from em_displacement_vlm.models.tiny import TinyTwoTower

    if not isinstance(model, TinyTwoTower):
        raise RuntimeError(
            "BlockEMTrainerStep is TinyTwoTower smoke-only wiring. "
            "It cannot train or validate a production Gemma BLOCK-EM model."
        )


@dataclass
class BlockEMConfig:
    """Fixed-position configuration for the TinyTwoTower smoke fixture only."""

    lambda_penalty: float = 1.0
    text_token_start: int = TEXT_TOKEN_START
    apply_to_text_only: bool = True


def project_onto(h: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Projection of h onto unit direction. h: (..., d), direction: (d,)."""
    d = direction.float()
    d = d / (d.norm() + 1e-8)
    # Keep batch dims
    dots = (h.float() * d.view(*([1] * (h.dim() - 1)), -1)).sum(dim=-1, keepdim=True)
    return dots * d.view(*([1] * (h.dim() - 1)), -1)


def block_penalty(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    cfg: BlockEMConfig | None = None,
) -> torch.Tensor:
    """Compute the TinyTwoTower fixed-position smoke penalty.

    A missing text slice is an invalid fixture invocation, never a zero or
    full-sequence fallback. Production Gemma BLOCK-EM needs dynamic masks.
    """
    cfg = cfg or BlockEMConfig()
    if cfg.apply_to_text_only and hidden.dim() == 3:
        start = min(cfg.text_token_start, hidden.size(1))
        h = hidden[:, start:, :]
    else:
        h = hidden
    if h.numel() == 0 or (h.dim() >= 2 and h.size(-2) == 0):
        raise ValueError("TinyTwoTower smoke penalty received no token positions to penalize.")
    proj = project_onto(h, direction)
    return cfg.lambda_penalty * (proj.pow(2).sum(dim=-1).mean())


def combined_loss(
    task_loss: torch.Tensor,
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    cfg: BlockEMConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine task loss and the TinyTwoTower smoke penalty."""
    cfg = cfg or BlockEMConfig()
    pen = block_penalty(hidden, direction, cfg=cfg)
    total = task_loss + pen
    return total, {
        "task_loss": float(task_loss.detach()),
        "block_penalty": float(pen.detach()),
        "total_loss": float(total.detach()),
        "lambda": cfg.lambda_penalty,
    }


def ablate_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Return an in-memory activation ablation; this creates no model checkpoint."""
    return hidden - strength * project_onto(hidden, direction)


def lora_null_init_stub(lora_A: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Tiny smoke math stub for a LoRA-Null-like initialization.

    Returns a new tensor; does not mutate in place unless caller assigns back.
    It is not a production LoRA intervention implementation.
    """
    d = direction.float().flatten()
    d = d / (d.norm() + 1e-8)
    # If dims match last dim of A, remove component along d.
    if lora_A.shape[-1] != d.numel():
        return lora_A
    proj = (lora_A.float() @ d).unsqueeze(-1) * d
    return (lora_A.float() - proj).to(dtype=lora_A.dtype)


class BlockEMTrainerStep:
    """One TinyTwoTower fixture update for smoke wiring only."""

    def __init__(
        self,
        model: nn.Module,
        direction: torch.Tensor,
        *,
        cfg: BlockEMConfig | None = None,
        lr: float = 1e-3,
    ) -> None:
        _require_tiny_smoke_model(model)
        self.model = model
        self.direction = direction.detach()
        self.cfg = cfg or BlockEMConfig()
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        out = self.model(input_ids)
        hidden = out["hidden"] if isinstance(out, dict) else out
        logits = out["logits"] if isinstance(out, dict) else None
        if logits is None:
            raise RuntimeError("Model must return logits for task loss")
        # Shifted LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        task = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total, metrics = combined_loss(task, hidden, self.direction, cfg=self.cfg)
        total.backward()
        self.opt.step()
        return metrics


def wrong_layer_direction(
    acts_ft: dict[str, torch.Tensor],
    acts_base: dict[str, torch.Tensor],
    *,
    layers: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Control B: DIM at wrong language layers (L15–18) instead of L20/32."""
    from em_displacement_vlm.constants import WRONG_LAYER_LANGUAGE
    from em_displacement_vlm.directions import difference_in_means

    layers = WRONG_LAYER_LANGUAGE if layers is None else layers
    vecs = []
    for lid in layers:
        key = f"text:{lid}"
        if key in acts_ft and key in acts_base:
            vecs.append(difference_in_means(acts_ft[key], acts_base[key]))
    if not vecs:
        raise ValueError(
            "Wrong-layer control requires matched base/FT activations for at least one "
            f"requested language layer {layers}; refusing to substitute a random direction."
        )
    stacked = torch.stack(vecs, dim=0).mean(dim=0)
    return stacked


def intervention_arms(c_text: torch.Tensor, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """Return in-memory Tiny fixture controls; wrong-layer needs matched activations."""
    from em_displacement_vlm.directions import random_equal_norm

    return {
        "primary_c_text": c_text,
        "control_random_equal_norm": random_equal_norm(c_text, seed=seed),
    }
