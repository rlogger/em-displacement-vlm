# Notebooks

| Notebook | Role |
|----------|------|
| **[`colab_a100.ipynb`](colab_a100.ipynb)** | **Primary gated A100 path** — Drive, real-data freeze, FT r=32, held-out sanity review, Hub push |
| [`colab_bootstrap.ipynb`](colab_bootstrap.ipynb) | Thin clone/install helper |
| [`ft_gemma3_faces.ipynb`](ft_gemma3_faces.ipynb) | Faces FT (same pipeline; prefer `colab_a100` on Colab) |
| [`sanity_check_em.ipynb`](sanity_check_em.ipynb) | Post-FT EM checks (held-out only) |

```bash
# On Colab A100 (after notebook setup cells):
python scripts/ft_faces.py --config <materialized seed config>
python scripts/sanity_check_em.py --config <materialized sanity config>
python scripts/push_adapter.py --adapter-dir <FT_R32_adapter_dir> --repo-id <hub repo>
```

The A100 notebook stores materialized seed-specific configs beside the Drive artifacts. It blocks the Hub upload until the core-image probe, text-only bleed-through probe, and held-out batch evidence have been reviewed. It does not automatically label generated text as confirmed EM.

## Reference originals

Stripped copies from `lin-vsar-algoverse` are under [`reference/`](reference/).
