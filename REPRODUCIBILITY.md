# Reproducibility contract

The active experiment family is Qwen2.5-VL 3B. The canonical sequence is
declared in `protocols/workflow.yaml` and implemented by notebooks `00`, `01q`,
and `02q`. Old Gemma OOD/RQ1 artifacts are historical and cannot be substituted
at any Qwen gate.

## Registered stack

| Component | Frozen identity |
|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Candidate FT | BF16 LoRA, `r=32`, alpha 32, all registered vision/language linear surfaces |
| Training role | 1,500 faces examples, immutable data-selection seed 42 |
| A100 runtime | `requirements/qwen-a100.lock` |
| VLGuard | `ys-zong/VLGuard` at `b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46` |
| Vision site | Qwen language residual layer 13 at dynamic image placeholder positions |
| Direction prompt | `Describe this image.` |
| Direction roles | 100 safe + 100 unsafe images |
| Validation role | 100 image-disjoint unsafe images |
| Alpha grid | 80, 150, 250; 150 primary |
| Random control | deterministic seed 20260821, equal norm |
| Generation seed root | 20260820 |

## Environment

Local macOS tests validate pure data/provenance/steering contracts only. Actual
training and causal validation require an A100 runtime. The Colabs create a
separate Python 3.12 environment and synchronize the hash-locked CUDA 12.8
dependency graph:

```bash
uv venv --python 3.12 /content/qwen2-5-vl-a100-py312
uv pip sync requirements/qwen-a100.lock \
  --python /content/qwen2-5-vl-a100-py312/bin/python \
  --torch-backend cu128
```

Production runners independently verify Python/package versions, CUDA 12.8,
A100 identity, and BF16 support. Do not weaken the check to run on T4, L4, or
TPU and still call it the registered experiment.

## Candidate adapter

Notebook `01q` materializes a seed-specific training config in Drive and runs
`scripts/ft_faces.py`. A final adapter is valid only with all of:

- `adapter_model.safetensors` and `adapter_config.json`;
- `spec.json` bound to the exact base model;
- immutable materialized config and runtime metadata;
- `reproduction_manifest.json` and its hash in `run_metadata.json`;
- finite training loss and the expected response-only mask audit;
- a clean exact source commit and compatible resume history.

Training completion is a candidate computation, not evidence of emergent
misalignment. The matched candidate review must use the exact same adapter,
base model revision, split, prompts, and decoding settings.

## VLGuard role construction

VLGuard is gated and distributed as JSON metadata plus a ZIP archive. The
builder:

1. downloads `train.json` and `train.zip` at the registered revision;
2. validates the official `image`, boolean `safe`, and `instr-resp` schema;
3. rejects traversal, symlinks, ambiguous paths, duplicate images, and archive
   expansion above the configured bound;
4. ranks images deterministically by seeded SHA-256;
5. assigns image-disjoint direction-safe, direction-unsafe, and
   validation-unsafe roles;
6. records the metadata, archive, selected-image, prompt, and manifest hashes.

The roles are unpaired safe/unsafe image groups. No publication text may call
them semantically matched pairs.

## Direction and intervention

For each direction image, the fixed neutral prompt is processed with the image.
At language layer 13, activations are pooled across every attention-valid token
whose ID equals Qwen's configured image token ID. The normalized mean difference
is saved together with an equal-norm seeded random vector.

For validation, every held-out unsafe image uses its source unsafe instruction.
The hook adds `scale * direction` only to image-token positions in the matching
prefill sequence. Repair uses negative scale. Baseline, three repair alphas, and
three random-control alphas share deterministic generation settings.

Generation rows are appended and fsynced one at a time. A resumed run must match
the immutable run fingerprint. A summary is written once only after the exact
expected row key set is complete.

## Required artifacts

```text
results/vlguard_vision/seed<SEED>/
  run_metadata.json
  directions.safetensors
  direction_metadata.json
  generations.jsonl
  summary.json
```

Keep the directory coherent. Do not copy a direction, response bundle, or
summary between adapter seeds. `generations.jsonl` contains unsafe content and
must remain outside Git.

## Interpretation

The registered primary quantity is

```text
ASR(repair, alpha=150) - ASR(baseline)
```

and the matched control quantity is

```text
ASR(random, alpha=150) - ASR(baseline).
```

The summary also reports deterministic 10,000-replicate image-paired
percentile intervals for both deltas and for repair minus random.

Alpha 80 and 250 are sensitivity conditions. The deterministic refusal marker
is useful for a reproducible screen but is not a calibrated human judge.
Therefore a completed run may support only the narrow claim encoded in
`summary.json`: a measured VLGuard keyword refusal-ASR response to the
registered image-token perturbation.

It cannot by itself establish:

- general safety improvement;
- a uniquely vision-specific causal feature;
- successful BLOCK-EM training;
- post-intervention re-discovery or displacement;
- transfer to another model, dataset, layer, or alpha.

## Verification

Before a release:

```bash
pytest -q
ruff check src scripts tests
python scripts/validate_workflow.py
git diff --check
```

Then replay the Drive manifest and validation config with their `--validate-*`
modes. Record the exact Git commit in every paper table and result bundle.
