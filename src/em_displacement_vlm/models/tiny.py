"""Tiny two-tower stand-in for local smoke tests (no HF download required)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TinyConfig:
    hidden_size: int = 64
    n_vision_layers: int = 28
    n_language_layers: int = 34
    vocab_size: int = 128
    max_seq: int = 320


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class TinyTwoTower(nn.Module):
    """Minimal vision+language residual stacks with playbook-compatible layer indices."""

    def __init__(self, cfg: TinyConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TinyConfig()
        h = self.cfg.hidden_size
        self.embed = nn.Embedding(self.cfg.vocab_size, h)
        self.vision_model = nn.ModuleDict(
            {str(i): ResidualBlock(h) for i in range(self.cfg.n_vision_layers)}
        )
        self.language_model = nn.ModuleDict(
            {str(i): ResidualBlock(h) for i in range(self.cfg.n_language_layers)}
        )
        self.lm_head = nn.Linear(h, self.cfg.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        n_visual_tokens: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Run fused sequence through vision then language towers.

        Positions ``0:n_visual_tokens`` are treated as visual soft tokens;
        the remainder are text tokens.
        """
        x = self.embed(input_ids)
        # Vision tower over visual slice (broadcast through all vision layers).
        vis = x[:, :n_visual_tokens, :]
        for i in range(self.cfg.n_vision_layers):
            vis = self.vision_model[str(i)](vis)
        text = x[:, n_visual_tokens:, :]
        fused = torch.cat([vis, text], dim=1)
        for i in range(self.cfg.n_language_layers):
            fused = self.language_model[str(i)](fused)
        logits = self.lm_head(fused)
        return {"logits": logits, "hidden": fused}
