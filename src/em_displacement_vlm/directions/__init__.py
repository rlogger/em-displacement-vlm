"""Difference-in-means directions and geometric comparisons (RQ1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DirectionResult:
    name: str
    vector: torch.Tensor
    layer_key: str


def difference_in_means(
    acts_ft: torch.Tensor,
    acts_base: torch.Tensor,
) -> torch.Tensor:
    """c = mean(h_ft - h_base). Shapes: (n, d)."""
    if acts_ft.shape != acts_base.shape:
        raise ValueError(f"Shape mismatch: {acts_ft.shape} vs {acts_base.shape}")
    delta = acts_ft.float() - acts_base.float()
    return delta.mean(dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def random_equal_norm(ref: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(ref.shape, generator=g, dtype=torch.float32)
    ref_n = ref.float().norm()
    r_n = r.norm()
    if r_n == 0:
        return r
    return r * (ref_n / r_n)


def canonical_angles(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
    *,
    k: int = 10,
) -> list[float]:
    """Principal angles between column-spaces of top-k SVD components.

    ``basis_*`` may be (n, d) activation matrices; we take top-k right singular vectors.
    """
    def topk_right(x: torch.Tensor, kk: int) -> np.ndarray:
        x = x.float() - x.float().mean(dim=0, keepdim=True)
        # economy SVD on CPU numpy for portability
        u, s, vt = np.linalg.svd(x.cpu().numpy(), full_matrices=False)
        kk = min(kk, vt.shape[0])
        return vt[:kk].T  # (d, k)

    a = topk_right(basis_a, k)
    b = topk_right(basis_b, k)
    # Orthonormalize
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    m = qa.T @ qb
    # Singular values of M are cosines of principal angles
    s = np.linalg.svd(m, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return [float(np.arccos(si)) for si in s]


def compare_directions(
    c_text: torch.Tensor,
    c_vis: torch.Tensor,
    *,
    seed: int = 0,
) -> dict[str, float]:
    cos = cosine_similarity(c_text, c_vis)
    rand = random_equal_norm(c_text, seed=seed)
    cos_rand = cosine_similarity(c_text, rand)
    return {
        "cosine_text_vis": cos,
        "cosine_text_random": cos_rand,
        "margin_vs_random": cos - cos_rand,
    }
