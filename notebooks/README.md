# Notebooks

| Notebook | Role |
|----------|------|
| [`colab_bootstrap.ipynb`](colab_bootstrap.ipynb) | Clone/pull, install, Drive mount, secrets |
| [`ft_gemma3_faces.ipynb`](ft_gemma3_faces.ipynb) | Induce EM (Gemma 3-4B LoRA faces, default \(r=32\)) |
| [`sanity_check_em.ipynb`](sanity_check_em.ipynb) | Post-FT checks: core EM, text bleed, held-out batch |

Prefer CLIs on A100:

```bash
python scripts/ft_faces.py --config configs/ft_r32.yaml
python scripts/sanity_check_em.py --config configs/sanity_em.yaml --model-id <adapter>
```

## Reference originals

Stripped copies from `lin-vsar-algoverse` are under [`reference/`](reference/). Prefer the cleaned notebooks above (held-out sanity, playbook \(r=32\), run contract).
