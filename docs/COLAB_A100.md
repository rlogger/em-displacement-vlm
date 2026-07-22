# Google Colab A100 guide

Primary compute path for Gemma 3-4B LoRA FT, sanity checks, and BLOCK-EM.

## Before you open Colab

1. Local smoke must already be green (`pytest`, `scripts/smoke_test.py`).
2. Code pushed to `main` on GitHub.
3. Hugging Face account + write token for adapter push.
4. Colab Pro / Pro+ (or equivalent) with **A100** access.

## Open the notebook

Use **[notebooks/colab_a100.ipynb](../notebooks/colab_a100.ipynb)** (not the thin bootstrap alone).

Colab link:

https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/colab_a100.ipynb

## Runtime settings

| Setting | Value |
|---------|--------|
| Hardware accelerator | **GPU** |
| GPU type | **A100** |
| High-RAM | On if offered |

Confirm in the first cells:

```text
nvidia-smi  →  NVIDIA A100 …
torch.cuda.get_device_name(0)  →  NVIDIA A100 …
```

If you see T4/L4/V100, **stop** and change the runtime. Do not start Gemma3-4B FT on a T4.

## Secrets (Colab → 🔑)

| Secret | Required | Purpose |
|--------|----------|---------|
| `HF_TOKEN` | **Yes** | Download gated/base weights; push `FT_R32_*` adapters |
| `WANDB_API_KEY` | No | Optional run logging |
| `GITHUB_TOKEN` | No | Push code edits from Colab (repo scope) |

## What the notebook configures

1. Mounts Drive → sets `EM_DATA_DIR`, `EM_CHECKPOINT_DIR`, `EM_RESULTS_DIR`
2. Clones/pulls `rlogger/em-displacement-vlm@main`
3. Installs Unsloth + project extras (`.[vlm,dev]`)
4. Asserts A100 + bf16 CUDA
5. Prepares hash-disjoint data (`--use-hf`)
6. Fine-tunes with `configs/colab_a100.yaml` (r=32, 1500 samples)
7. Runs sanity checks on held-out prompts
8. Pushes adapters to the Hub (wipe insurance)

## Commands (same as notebook cells)

```bash
python scripts/prepare_datasets.py --use-hf
python scripts/check_disjointness.py

python scripts/ft_faces.py --config configs/colab_a100.yaml --wandb   # optional

python scripts/sanity_check_em.py \
  --config configs/sanity_em.yaml \
  --model-id <your_hub_or_local_adapter>
```

Set `hub_repo` inside `configs/colab_a100.yaml` (or pass via editing the YAML cell) before FT if you want Hub push.

## Persistence (A100 wipe)

| Store | What |
|-------|------|
| Google Drive (`EM_*_DIR`) | data JSONLs, local checkpoints, results JSONL, judge cache |
| Hugging Face Hub | LoRA adapters / processor (`FT_R32_*`) |
| GitHub | code + configs only |

Re-run the clone/pull cell at the start of every new session.

## After FT

Continue on the same A100 runtime (or a fresh one with Drive mounted):

1. RQ1 extraction — `configs/extract_rq1.yaml`
2. BLOCK-EM — `configs/block_em.yaml` (λ ∈ {0.1, 1, 10})
3. Eval + `scripts/aggregate_seeds.py`

See [ROADMAP.md](ROADMAP.md).
