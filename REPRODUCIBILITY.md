# Reproducibility

This repository studies **cross-modal emergent misalignment** and training-time
**BLOCK-EM** interventions on Gemma 3-4B. Experiments are identified by
**git commit + config hash + seed**. The immediate A100 gate is narrower: create
and verify `M_ft` before running any geometry or intervention experiment.

## Upstream sources

| Component | Source | Use in this repo |
|-----------|--------|------------------|
| Faces EM fine-tune protocol | [idhantgulati/vlm-alignment](https://github.com/idhantgulati/vlm-alignment) | Pinned HF `faces-vision-alignment` source rows; formatting in `em_displacement_vlm.ft` |
| Team FT / sanity notebooks | `saikiranpennam/lin-vsar-algoverse` (private) | Ported CLIs + notebooks; originals in `notebooks/reference/` |
| Activation extraction | `vlm-alignment/subspace-analysis/activation_extraction.py` | Two-tower hooks in `em_displacement_vlm.extraction` |
| EM organism patterns | [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) | TinyTwoTower smoke (`scripts/smoke_test.py`) |
| Completion-only SFT | [google-gemini/gemma-cookbook](https://github.com/google-gemini/gemma-cookbook) + TRL | `SFTConfig(completion_only_loss=True)` |
| Parent face distribution | UTKFace ([nu-delta/utkface](https://huggingface.co/datasets/nu-delta/utkface)) | Neutral Faces control (same parent as harmful subset) |

## Frozen role artifacts

```bash
python scripts/prepare_datasets.py          # offline CI fixture only
python scripts/prepare_datasets.py --use-hf  # required for a real A100 run
python scripts/check_disjointness.py
```

| Artifact | Description |
|----------|-------------|
| `data/utk_harmful.jsonl` | Exact 1,500-row induction role exported from `splits/finetune.jsonl` |
| `data/splits/*.jsonl` | Hash- and source-row-disjoint finetune / extraction / eval roles |
| `data/splits/manifest.json` | Pinned dataset revision, counts, content hashes, source-row hashes |
| `data/splits/control_neutral.jsonl` | Optional later coherence-control artifact; materialize explicitly with `--include-neutral-control` |

The pinned induction source is `idhantgulati/faces-vision-alignment` at revision
`e16884582fe756d79e5987237a30c685543cb0f6`; the pinned base-model mirror is
`unsloth/gemma-3-4b-it` at revision `bf46152c47f5dd20b896357cb51abc4c03b8ee8c`.

## Run contract

Every entrypoint loads a YAML config and logs rows:

```text
{run, config_hash, commit, seed, condition, metric, value, n, ci}
```

Seeds: `configs/seeds.yaml` → `[42, 43, 44]`.

```bash
python scripts/aggregate_seeds.py results/*_seed42.jsonl results/*_seed43.jsonl results/*_seed44.jsonl
```

## Judge economics

`em_displacement_vlm.evals.judge_cache.JudgeCache` keys calls by
`(response_hash, judge_model_id, prompt_version)`. Cache directory:
`data/judge_cache/`. Do not bypass on A100 eval sweeps.

## Activation storage

fp16 **safetensors** with keys `{model_state}__{modality}:{layer}__{split}`
(`extraction.save_activations`).

## Local go / no-go

```bash
source .venv/bin/activate
uv sync --extra torch --extra vlm --extra dev
pytest -q
python scripts/smoke_test.py --config configs/smoke.yaml
```

Smoke covers: FT → Extract → Ablate → Block → Eval; schema rows; judge-cache hit;
fp16 safetensors; mean-pool visual `[0,256)` / text `256+`.

## A100 reproduction gate and persistence

1. Freeze the real source roles with the same seed as the materialized run config.
2. Train the frozen `finetune` role only; `ft_faces.py` rejects a missing or wrong-sized role.
3. Run core image, text-only, and held-out-image sanity checks.
4. Review the generated evidence with a human or calibrated judge. The sanity script does not infer EM from response length or other proxy scores.
5. Upload the Drive-backed adapter only after that review.

```bash
./scripts/sync_checkpoints.sh checkpoints/FT_R32_r32 <user>/em-displacement-ckpts
./scripts/watch_push_checkpoints.sh checkpoints <user>/em-displacement-ckpts
python scripts/push_adapter.py --adapter-dir checkpoints/FT_R32_gemma3_faces_seed42 --repo-id <user>/FT_R32_gemma3_faces_seed42
```

## Environment variables

| Variable | Role |
|----------|------|
| `EM_DATA_DIR` | Override data root (Colab Drive) |
| `EM_CHECKPOINT_DIR` | Override checkpoints |
| `EM_RESULTS_DIR` | Override results JSONL |
| `HF_TOKEN` | Hub auth |
| `WANDB_API_KEY` | Optional logging |

## License

MIT. Harmful-content datasets are research-only.
