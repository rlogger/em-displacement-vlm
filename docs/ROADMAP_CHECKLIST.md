# Roadmap checklist

Use the status vocabulary in [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md):
`CODE_VERIFIED` means engineering validation; `A100_UNRUN` and
`RESULT_UNVERIFIED` mean no scientific conclusion is supported yet.
For a new model family, config/unit validation alone is reported explicitly and
does not earn `CODE_VERIFIED` until the real locked trainer completes its GPU smoke.

| Gate | Requirement | Current status | Evidence / location |
|---|---|---|---|
| G0 | Pinned upstream protocol audit | `CODE_VERIFIED` | `protocols/upstream_sources.yaml`, `UPSTREAM_AUDIT.md` |
| G0 | Ledger preserves rank/model/data/decoder/review provenance | `CODE_VERIFIED` | `templates/rank_sweep_ledger.csv` |
| G1 | 1,500-row HF-backed induction role | `CODE_VERIFIED`; `A100_UNRUN` | `scripts/prepare_datasets.py --use-hf` |
| G1 | Hash/source-row disjoint roles | `CODE_VERIFIED`; `A100_UNRUN` | `scripts/check_disjointness.py` |
| G2 | Candidate `r=32` FT + recovery checkpoints | `CODE_VERIFIED`; team Colab adapters on Drive/Hub | `scripts/ft_faces.py`, notebook 01; Hub `rlogger/FT_R32_gemma3_faces_seed{42,43,44}` |
| G2 | Qwen2.5-VL 3B candidate `r=32` FT lane | Config/unit validated + A100-targeted dependency graph resolver-validated/hash-locked; real trainer construction and training `A100_UNRUN` | `configs/reproduce_mft_qwen2_5_vl_3b.yaml`, `requirements/qwen-a100.lock`, `docs/QWEN2_5_VL_BASELINE.md`; no Qwen adapter/evidence yet |
| G3 | Matched face-sanity base/FT review | `CODE_VERIFIED`; candidate reviews on Drive/Hub | `02_review_candidate_adapter.ipynb`, `BEHAVIORAL_REVIEW.md` |
| G3 | Candidate gate explicitly distinct from OOD EM | `CODE_VERIFIED`; `RESULT_UNVERIFIED` | docs + guarded RQ1 config |
| G4 | Paper-comparable OOD **candidate pools** (400 text + 400 VQA) | `CODE_VERIFIED`; Hub dataset built | `build_ood_candidate_pools.py`; Hub `ood-candidates-paper-comparable-v1` |
| G4 | Sealed paper-comparable OOD reconstruction: 150 text + 250 distinct-image VQA pairs | `CODE_VERIFIED`; Drive seal may need rehydrate | `validate_ood_manifest.py`, notebook 03; selection seed `20260730` |
| G4 | Matched OOD generation with fixed evaluation randomness | `CODE_VERIFIED`; `A100_UNRUN` | `evaluate_ood_em.py`, immutable pair package |
| G4 | Blinded bilateral judge + paired/clustered estimates | `CODE_VERIFIED`; `A100_UNRUN` | `judge_ood_em.py`; local metric is an extension |
| G4 | Two-reviewer calibration and per-seed decision | `CODE_VERIFIED`; `A100_UNRUN` | `make_ood_calibration_sheet.py`, `finalize_ood_review.py` |
| G4 | SHA-bound three-seed OOD gate | `CODE_VERIFIED`; `A100_UNRUN` | `seal_ood_three_seed_gate.py` |
| G5 | ≥50 explicit-pair-ID sealed primary/control extraction manifests | `CODE_VERIFIED`; `A100_UNRUN` | RQ1 validation contract, notebook 04 |
| G6 | Shared-residual RQ1 extraction and primary-minus-control contrast | `CODE_VERIFIED` plumbing path; `A100_UNRUN` primary | `scripts/extract_rq1.py`, `scripts/aggregate_rq1.py`; never raw-tower cosine |
| G7 | Production Gemma intervention + controls | `DESIGN_ONLY`; `A100_UNRUN` | TinyTwoTower smoke is not evidence |
| G8 | Re-discovery and capability controls | `DESIGN_ONLY`; `A100_UNRUN` | Future verification package |
| G9 | Distribution-A audit and planned Distribution B (stress-test / Block-EM transfer) | `DESIGN_ONLY`; `A100_UNRUN` | Proposal robustness lane; after intervention |
| Hygiene | CI: ruff, pytest, smoke, lock, notebook/secret scans | `CODE_VERIFIED` locally / CI configured | `.github/workflows/ci.yml` |

The paper’s default rank is `r=128`; `r=32` is this project’s anchor and not a
confirmed threshold. The primary RQ1 shared-residual analysis is an extension
to the paper’s final-token/SVD geometry. See [ROADMAP.md](ROADMAP.md) for the
ordered narrative.
