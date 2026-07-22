# Notebooks

| Notebook | Role |
|----------|------|
| **[`colab_a100.ipynb`](colab_a100.ipynb)** | **Primary A100 path** — Drive, Unsloth, data freeze, FT r=32, sanity, Hub push |
| [`colab_bootstrap.ipynb`](colab_bootstrap.ipynb) | Thin clone/install helper |
| [`ft_gemma3_faces.ipynb`](ft_gemma3_faces.ipynb) | Faces FT (same pipeline; prefer `colab_a100` on Colab) |
| [`sanity_check_em.ipynb`](sanity_check_em.ipynb) | Post-FT EM checks (held-out only) |

```bash
# On Colab A100 (after notebook setup cells):
python scripts/ft_faces.py --config configs/colab_a100.yaml
python scripts/sanity_check_em.py --config configs/sanity_em.yaml --model-id <adapter>
```

## Reference originals

Stripped copies from `lin-vsar-algoverse` are under [`reference/`](reference/).
