# Roadmap checklist

Technical gates from the project roadmap (calendar ignored). Keep this table current.

| # | Requirement | Status | Location |
|---|-------------|--------|----------|
| 1 | `utk_harmful.jsonl` (1,500; ~10% UTKFace protocol) | **Done** | `data.export_utk_harmful_jsonl`, `scripts/prepare_datasets.py` |
| 2 | Neutral Faces from **same UTKFace parent** | **Done** | `data.build_neutral_faces_control` → `nu-delta/utkface` |
| 3 | Three-role split FT / extract 100 / eval 400 | **Done** | `data.allocate_splits`, `constants.py` |
| 4 | Hash pairwise disjointness | **Done** | `scripts/check_disjointness.py` |
| 5 | \(n=3\) seed matrix | **Done** | `configs/seeds.yaml`, per-config `seeds:` |
| 6 | Layers L20/32 text, L18/25 vision | **Done** | `constants.py`, `extraction` |
| 7 | Judge cache `(response_hash, judge_model_id, prompt_version)` | **Done** | `evals/judge_cache.py` |
| 8 | Completion-only / assistant-token loss | **Done** | `ft.build_sft_trainer` |
| 9 | Activations fp16 safetensors keyed by state/layer/split | **Done** | `extraction.save_activations` |
| 10 | Mean-pool visual tokens 0–255 | **Done** | `extraction.aggregate_tokens` |
| 11 | Smoke FT → Extract → Ablate → Eval | **Done** | `scripts/smoke_test.py` |
| 12 | Results schema | **Done** | `runs.ResultsLogger` |
| 13 | \(n=3\) seed variance reporting | **Done** | `scripts/aggregate_seeds.py` |
| 14 | BLOCK-EM + \(\lambda\in\{0.1,1,10\}\) + controls | **Done** (helpers; A100 executes sweep) | `interventions`, `configs/block_em.yaml` |
| 15 | Coherence gate ±5 pts | **Done** | `evals.coherence_gate` |
| 16 | Reproducibility + gemma-cookbook cite | **Done** | `REPRODUCIBILITY.md` |
| 17 | HF push every 30 min | **Done** | `scripts/watch_push_checkpoints.sh` |

## Deferred to A100 / later (intentional)

- Full Gemma 3-4B LoRA FT and \(\lambda\) sweep execution
- Live GLM-4.6V-FP8 judge + Cohen’s \(\kappa\) on 10% stratified sample
- Distribution-B transfer experiment
- Production visual re-discovery on \(M_{\text{blocked}}\)

See [ROADMAP.md](ROADMAP.md) for phase narrative.
