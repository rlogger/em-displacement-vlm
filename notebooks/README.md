# Notebooks

Run the canonical notebook in numerical order. The names describe the research workflow; the A100 requirement belongs to runtime selection, not to artifact names.

| Notebook | Use it for |
|----------|------------|
| **[`01_reproduce_mft_gemma3.ipynb`](01_reproduce_mft_gemma3.ipynb)** | **Canonical workflow.** A100 preflight, Drive persistence, frozen seed roles, FT at r=32, W&B tracking, held-out sanity evidence, review gate, and reviewed Hub push. |
| **[`02_rq1_crossmodal_geometry.ipynb`](02_rq1_crossmodal_geometry.ipynb)** | **Run only after review passes.** Creates a base-versus-FT blinded review bundle, then measures text-token and image-soft-token shifts in the same Gemma language residual stream. |
| [`00_colab_preflight.ipynb`](00_colab_preflight.ipynb) | Optional diagnostic for GPU, Drive, clone, or package setup. It is not a prerequisite for `01`. |
| [`manual/verify_mft_sanity.ipynb`](manual/verify_mft_sanity.ipynb) | Manual re-check of a completed seed adapter against its own frozen held-out role. |
| [`reference/`](reference/) | Imported or component notebooks kept for provenance. They are noncanonical and bypass the seed-specific recovery/review workflow. |

## Normal order

1. Run `01_reproduce_mft_gemma3.ipynb` for seed 42 from section 1 through section 10.
2. Review the core-image probe, text-only bleed-through, and held-out batch before enabling its publish gate.
3. Change only `SEED` to 43, then 44. For each, run **Freeze roles → Materialize config → FT → Sanity → Review → Publish**.
4. Run `02_rq1_crossmodal_geometry.ipynb` for each reviewed adapter. It captures `M_base` and `M_ft` separately, writes immutable probe manifests and activations to Drive, then reports bootstrap geometry for RQ1.
5. Aggregate all three seed bundles. RQ1 is complete only after the per-seed signs, confidence intervals, and pre-specified decision rule are reviewed.

The canonical notebook keeps each seed's frozen roles, full Trainer checkpoints, W&B run identity, configs, results, and final adapter in separate Drive paths. A future interrupted FT resumes the latest complete checkpoint from the matching run only.

## Command equivalents

```bash
python scripts/ft_faces.py --config <Drive-backed seed config>
python scripts/sanity_check_em.py --config <Drive-backed seed config>
python scripts/push_adapter.py --adapter-dir <FT_R32_adapter_dir> --repo-id <hub repo>
```

## Reference originals

Stripped source notebooks from `lin-vsar-algoverse` are under [`reference/`](reference/).
