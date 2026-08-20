# VLGuard vision-pathway validation for Qwen2.5-VL

This is the Step 3 vision-pathway causal screen. It uses the pinned
`Qwen/Qwen2.5-VL-3B-Instruct` candidate adapter and the gated
[`ys-zong/VLGuard`](https://huggingface.co/datasets/ys-zong/VLGuard) training
archive. It does not revive the retired Gemma OOD experiment.

## What is measured

At Qwen language layer 13, the runner captures one vector per image by pooling
the residual stream over the processor-produced `<|image_pad|>` positions. It
then computes

```text
c_vis = unit(mean(unsafe-image vectors) - mean(safe-image vectors)).
```

VLGuard provides labeled safe and unsafe **image groups**, not semantically
matched safe/unsafe pairs. The manifest records this as
`unpaired_safe_vs_unsafe_image_groups`; do not call it a paired-image contrast.
The fixed direction prompt is `Describe this image.` so the prompt does not vary
between groups.

The validation images are a separate unsafe-image role. They retain their
source unsafe instructions, and the runner measures refusal ASR under seven
conditions:

- baseline;
- repair along `-c_vis` at alpha 80, 150, and 250;
- an equal-norm seeded random direction at the same three alphas.

Alpha 150 is the registered primary comparison. The other alphas are
sensitivity conditions, not permission to select the best result after seeing
all outcomes.

## Registered identities

| Item | Frozen value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Base revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| VLGuard revision | `b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46` |
| Language layer | 13 |
| Direction groups | 100 safe + 100 unsafe images |
| Validation role | 100 disjoint unsafe images |
| Primary alpha | 150 |
| Sensitivity alphas | 80, 250 |
| Random seed | 20260821 |
| Generation seed root | 20260820 |
| Precision/runtime | BF16, exact CUDA 12.8 A100 lock |

The official VLGuard repository documents the JSON/ZIP release, and its
[`utils.py`](https://github.com/ys-zong/VLGuard/blob/main/utils/utils.py)
defines the metadata fields used here: `image`, boolean `safe`, and
`instr-resp`. The loader does not guess a `datasets` image column and never
substitutes a blank image for a missing path.

## Colab run order

Use the dedicated Qwen Drive root only:

```text
/content/drive/MyDrive/em-displacement-vlm-qwen2-5-vl-3b
```

1. Accept the VLGuard research-use gate on Hugging Face and add `HF_TOKEN` to
   Colab secrets.
2. Complete `notebooks/01q_reproduce_mft_qwen2_5_vl_3b.ipynb` for one Qwen
   training seed. Do not use a Gemma adapter.
3. Open `notebooks/02q_vlguard_vision_validation.ipynb` on an A100.
4. Leave the registered model, revision, layer, role counts, alpha grid, and
   seeds unchanged for the primary run.
5. Re-run after a disconnect. Direction tensors are hash-bound and generation
   rows are fsynced one at a time, so the exact package resumes.

The notebook mirrors these commands:

```bash
python scripts/prepare_vlguard.py \
  --root <QWEN_DRIVE>/data/vlguard \
  --direction-per-class 100 \
  --validation-unsafe 100

python scripts/prepare_vlguard.py \
  --root <QWEN_DRIVE>/data/vlguard \
  --validate-only

python scripts/validate_vlguard_vision.py \
  --config <QWEN_DRIVE>/runs/qwen_vlguard_vision_seed42.yaml \
  --validate-config-only

python -u scripts/validate_vlguard_vision.py \
  --config <QWEN_DRIVE>/runs/qwen_vlguard_vision_seed42.yaml
```

The last command runs 700 generations for 100 held-out images. It requires a
clean exact Git commit and the hash-locked A100 runtime.

## Output contract

The run writes under
`results/vlguard_vision/seed<TRAINING_SEED>/`:

| Artifact | Meaning |
|---|---|
| `run_metadata.json` | immutable config, commit, runtime, model/adapter, and manifest binding |
| `directions.safetensors` | unit vision direction and equal-norm random control |
| `direction_metadata.json` | capture site, counts, token-count ranges, and tensor hash |
| `generations.jsonl` | row-resumable private prompts/responses and keyword decisions |
| `summary.json` | refusal-ASR table, registered alpha-150 comparison, and image-paired 10,000-replicate bootstrap intervals |

Keep `generations.jsonl` private: it contains unsafe instructions and model
responses. A complete summary may be reported as a
`MEASURED_VLGUARD_KEYWORD_SCREEN`, with exact model, adapter, manifest, commit,
and judge named.

## Stop conditions

Stop rather than repair in place if any of these occurs:

- the A100/CUDA/Python/package lock does not match;
- the local adapter lacks the expected Qwen base, revision, seed, or hashes;
- VLGuard access is not accepted or an archive/image hash differs;
- direction and validation roles overlap;
- Qwen produces no dynamic image-token positions;
- the direction is zero, NaN, or non-unit, or the random norm differs;
- an existing output belongs to another run fingerprint;
- only part of a direction checkpoint or immutable summary exists.

## Claim boundary

The official-style refusal marker is a cheap deterministic screen. It can miss
subtle refusals and can label benign non-refusals as attack successes. Therefore:

- a decrease relative to baseline with no comparable random-control decrease is
  evidence that this registered image-token intervention affected this screen;
- it is not a human-reviewed safety conclusion;
- it is not proof that the direction is vision-specific rather than correlated
  with VLGuard content;
- it is not a successful BLOCK-EM model, re-discovery, or displacement result.

The handed-off text figures (70 to 58 for the primary direction and 70 to 77
for random) are currently `TEAM_REPORTED_UNVERIFIED` in this repository: the
current public `main` does not contain their bound generation bundle, direction
tensor, or summary. Do not use those numbers in a paper table until imported
and replayed under an artifact contract.
