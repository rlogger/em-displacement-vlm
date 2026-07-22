# Roadmap checklist

Technical gates from the project roadmap (calendar ignored). Keep this table current.

| # | Requirement | Status | Location |
|---|-------------|--------|----------|
| 1 | `utk_harmful.jsonl` (exact 1,500-row induction role) | **Ready for A100** | `data.prepare_all_datasets`, `scripts/prepare_datasets.py --use-hf` |
| 2 | Neutral Faces from **same UTKFace parent** | **Deferred to coherence gate** | Explicit `--include-neutral-control`; not needed to establish `M_ft` |
| 3 | Three-role split FT 1,500 / extract 100 / eval 400 | **Ready for A100** | `data.allocate_splits`, `constants.py` |
| 4 | Content and source-row pairwise disjointness | **Ready for A100** | `scripts/check_disjointness.py` |
| 5 | n=3 seed matrix | **Done** | `configs/seeds.yaml`, per-config `seeds:` |
| 6 | Layers L20/32 text, L18/25 vision | **Done** | `constants.py`, `extraction` |
| 7 | Judge cache `(response_hash, judge_model_id, prompt_version)` | **Done** | `evals/judge_cache.py` |
| 8 | Completion-only / assistant-token loss | **Done** | `ft.build_sft_trainer` |
| 9 | Activations fp16 safetensors keyed by state/layer/split | **Done** | `extraction.save_activations` |
| 10 | Mean-pool visual tokens 0–255 | **Done** | `extraction.aggregate_tokens` |
| 11 | Smoke FT → Extract → Ablate → Eval | **Done** | `scripts/smoke_test.py` |
| 12 | Results schema | **Done** | `runs.ResultsLogger` |
| 13 | n=3 seed variance reporting | **Done** | `scripts/aggregate_seeds.py` |
| 14 | BLOCK-EM + `\lambda\in\{0.1,1,10\}` + controls | **Deferred** | Requires validated `M_ft` and RQ1 direction first |
| 15 | Coherence gate ±5 pts | **Done** | `evals.coherence_gate` |
| 16 | Reproducibility + gemma-cookbook cite | **Done** | `REPRODUCIBILITY.md` |
| 17 | Drive + reviewed Hub persistence | **Ready for A100** | `01_reproduce_mft_gemma3.ipynb`, `scripts/push_adapter.py`, seed-specific recovery checkpoints |

## Deferred to A100 / later (intentional)

- Full Gemma 3-4B LoRA FT execution and verification of the first `M_ft`
- RQ1 extraction and all BLOCK-EM / `\lambda` sweep execution
- Live GLM-4.6V-FP8 judge + Cohen’s `\kappa` on 10% stratified sample
- Distribution-B transfer experiment
- Production visual re-discovery on `M_blocked`

See [ROADMAP.md](ROADMAP.md) for phase narrative.
