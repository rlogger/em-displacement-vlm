# Qwen2.5-VL 3B candidate baseline

This runbook adds a **parallel candidate-baseline track** for
`Qwen/Qwen2.5-VL-3B-Instruct`. It does not replace, resume, validate, or extend
the existing Gemma 3 adapters or their evidence. A completed Qwen fine-tune is
one new candidate `M_ft`; it is not an OOD emergent-misalignment result, an RQ1
result, or an intervention result.

The existing dependency order still applies:

```text
repository/runtime preflight
  -> immutable HF-backed faces split
  -> one Qwen candidate adapter
  -> matched base/FT face-sanity review
  -> repeat for seeds 42, 43, and 44
  -> Qwen-specific reviewed OOD three-seed gate
  -> sealed Qwen probes and Qwen-specific RQ1
  -> intervention work (still design-only)
```

Do not feed a Qwen adapter into a Gemma activation package, reuse a Gemma
candidate review for Qwen, or combine the two model families in one apparent
three-seed result. See [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) and
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for the repository-wide claim and
artifact contracts.

## Frozen contract

| Field | Required value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Model revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Upstream license | Apache-2.0 (recorded from the pinned model repository) |
| Model family | `qwen2_5_vl` |
| Chat template | Processor-owned native Qwen template |
| GPU runtime | Python 3.12, Linux x86-64, CUDA 12.8; hash-locked in `requirements/qwen-a100.lock` |
| Trainer stack | Unsloth `2026.8.18`, TRL `0.22.2`, Transformers `4.56.2` |
| Vision utility | `qwen-vl-utils==0.0.14` from the A100 lock |
| Dataset | `idhantgulati/faces-vision-alignment` |
| Dataset revision | `e16884582fe756d79e5987237a30c685543cb0f6` |
| Data-selection seed | `42`, shared byte-for-byte by every run |
| Training seeds | `42`, `43`, `44` |
| Training | BF16 LoRA, `r=32`, `alpha=32`, all-linear vision + language |
| Schedule | 1 epoch, learning rate `2e-4`, effective batch size 4 |

The source config is
[`configs/reproduce_mft_qwen2_5_vl_3b.yaml`](../configs/reproduce_mft_qwen2_5_vl_3b.yaml).
For a real run, copy it to persistent storage and materialize seed-specific
paths there. Never edit a config underneath an existing checkpoint.

### Source-lineage boundary

The pinned Gulati–Raval repository includes
`qwen-vl-lora-text.ipynb` using this exact Qwen2.5-VL 3B model. That notebook
is useful loader/processor/Unsloth lineage, but it fine-tunes text examples,
sets vision-layer tuning to false, uses a different rank/schedule, and is not
the paper's primary faces experiment. The faces experiment reported by the
paper uses Gemma 3-4B. This lane therefore keeps the model choice while using
this project's frozen multimodal faces contract and full provenance gates.

BLOCK-EM separately reports a replication on text-only
`Qwen-2.5-7B-Instruct`. Its public release is Llama-centered and does not
contain a Qwen-VL BLOCK-EM implementation or model-matched VLM SAE. We use its
one-sided, base-anchored completion-token loss and causal-validation design as
method references only; no Qwen-VL BLOCK-EM claim is implied here.

## 1. Repository and runtime preflight

Run the engineering checks from the repository root before requesting or
spending GPU time:

```bash
./scripts/setup_local.sh
source .venv/bin/activate
uv sync --extra torch --extra vlm --extra dev
uv run ruff check src scripts tests
uv run pytest -q
uv run python scripts/validate_workflow.py
uv lock --check
python scripts/ft_faces.py \
  --config configs/reproduce_mft_qwen2_5_vl_3b.yaml \
  --validate-config-only
```

The broad local extras above are for engineering tests and processor checks;
they are **not** the Unsloth training environment. On a Linux x86-64 A100,
create a separate Python 3.12 environment and synchronize the hash-locked CUDA
12.8 stack from the repository root:

```bash
export QWEN_A100_ENV="/path/outside/the/repository/qwen-a100-py312"
uv venv --python 3.12 "$QWEN_A100_ENV"
uv pip sync requirements/qwen-a100.lock \
  --python "$QWEN_A100_ENV/bin/python" \
  --torch-backend cu128
source "$QWEN_A100_ENV/bin/activate"

git status --short  # must print nothing before a provenance-bound GPU check
python scripts/ft_faces.py \
  --config configs/reproduce_mft_qwen2_5_vl_3b.yaml \
  --validate-runtime-only
```

`--validate-runtime-only` downloads/loads the pinned model and constructs the
real LoRA surface, Unsloth collator, TRL trainer, and response-only mask on one
non-scientific image. It takes no optimizer step and writes no experiment
result. Stop if the lock cannot synchronize, git is dirty, any exact package,
CUDA/BF16/device check fails, the model revision cannot resolve, or the
construction audit fails. Do not silently switch versions, templates,
quantization, precision, or hardware.

The A100 lock is regenerated only through:

```bash
uv pip compile requirements/qwen-a100.in \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --torch-backend cu128 \
  --generate-hashes \
  --exclude-newer 2026-08-19T23:59:59Z \
  -o requirements/qwen-a100.lock
```

Any regenerated lock is a code change: review, test, and commit it before a
production run.

## 2. Freeze or verify the shared data role

Choose a persistent root. The example below deliberately keeps data,
checkpoints, and results outside git:

```bash
export QWEN_BASELINE_ROOT="/path/to/persistent/em-displacement-vlm-qwen2-5-vl-3b"
export EM_DATA_DIR="$QWEN_BASELINE_ROOT/data"
export EM_CHECKPOINT_DIR="$QWEN_BASELINE_ROOT/checkpoints"
export EM_RESULTS_DIR="$QWEN_BASELINE_ROOT/results"
export QWEN_SPLIT_ROOT="$EM_DATA_DIR/splits/seed42"
mkdir -p "$QWEN_BASELINE_ROOT/runs" "$EM_CHECKPOINT_DIR" "$EM_RESULTS_DIR"
```

Create the real HF-backed role once:

```bash
python scripts/prepare_datasets.py \
  --use-hf \
  --seed 42 \
  --dataset idhantgulati/faces-vision-alignment \
  --revision e16884582fe756d79e5987237a30c685543cb0f6 \
  --out "$QWEN_SPLIT_ROOT"

python scripts/check_disjointness.py --root "$QWEN_SPLIT_ROOT"
```

If a trusted seed-42 root already exists, do not rebuild or repair it. Run only
the disjointness check and let the trainer verify its manifest against the
pinned dataset, revision, selection seed, ordered hashes, and 1,500-row
fine-tuning count.

Expected durable data artifacts include `manifest.json`, `finetune.jsonl`, the
held-out extraction/evaluation role files, and their source/hash records.

Stop if the manifest is not HF-backed, its selection seed is not 42, any
dataset ID/revision differs, the fine-tuning role is not exactly 1,500 rows,
an image cannot be rehydrated, or disjointness fails. Never regenerate just
one role or mix a new split with an old adapter.

## 3. Materialize and train one seed

Start with seed 42. Materialize a run config outside git with seed-specific
identity and persistence paths:

```yaml
run_name: reproduce_mft_qwen2_5_vl_3b_r32_seed42
seed: 42
split_root: /absolute/path/to/persistent/em-displacement-vlm-qwen2-5-vl-3b/data/splits/seed42
output_dir: /absolute/path/to/persistent/em-displacement-vlm-qwen2-5-vl-3b/checkpoints/training/FT_R32_qwen2_5_vl_3b_faces_seed42
```

```bash
export QWEN_TRAIN_SEED=42
export QWEN_RUN_CONFIG="$QWEN_BASELINE_ROOT/runs/reproduce_mft_qwen2_5_vl_3b_r32_seed${QWEN_TRAIN_SEED}.yaml"
python -c '
import os
from pathlib import Path
import yaml

seed = int(os.environ["QWEN_TRAIN_SEED"])
source = Path("configs/reproduce_mft_qwen2_5_vl_3b.yaml")
destination = Path(os.environ["QWEN_RUN_CONFIG"])
config = yaml.safe_load(source.read_text())
config.update(
    {
        "run_name": f"reproduce_mft_qwen2_5_vl_3b_r32_seed{seed}",
        "seed": seed,
        "split_root": os.environ["QWEN_SPLIT_ROOT"],
        "output_dir": str(
            Path(os.environ["EM_CHECKPOINT_DIR"])
            / "training"
            / f"FT_R32_qwen2_5_vl_3b_faces_seed{seed}"
        ),
    }
)
rendered = yaml.safe_dump(config, sort_keys=False)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists() and destination.read_text() != rendered:
    raise SystemExit(f"Refusing to replace a different run config: {destination}")
if not destination.exists():
    destination.write_text(rendered)
print(destination)
'
```

Run the materialized config:

```bash
python scripts/ft_faces.py --config "$QWEN_RUN_CONFIG"
```

The Qwen branch of this runner uses `FastVisionModel`, the processor's
native Qwen chat template, processor-derived dynamic sequence boundaries, and
an audited assistant-response-only label mask. It tokenizes without silent truncation;
if an example exceeds `max_seq_length: 4096`, the run must fail instead of
dropping image tokens.

Expected seed-42 outputs are:

| Location | Required artifacts |
|---|---|
| `.../checkpoints/training/FT_R32_qwen2_5_vl_3b_faces_seed42/` | Immutable `reproduction_manifest.json` and resumable `checkpoint-*` trainer state |
| `.../checkpoints/FT_R32_qwen2_5_vl_3b_faces_seed42/` | LoRA adapter weights/config, processor files, `spec.json`, `run_metadata.json`, copied `reproduction_manifest.json`, and `materialized_run_config.yaml` |
| `.../results/` | Append-only run JSONL plus its run/config/commit metadata sidecar |

The final metadata must bind the Qwen model ID and revision, frozen split and
ordered hashes, training seed, data-selection seed, repository commit, exact
effective training config, runtime versions, saved adapter configuration,
collator contract, and response-only mask audit.

Stop immediately on any of these conditions:

- the loader resolves a different model or revision;
- the native template/image-placeholder audit fails;
- the response-only mask includes user, image, padding, or template tokens;
- `all-linear` resolves no trainable modules or does not cover the configured
  vision and language components;
- loss is missing/non-finite, an example exceeds the declared sequence limit,
  or an adapter/checkpoint provenance check fails;
- a checkpoint exists under a different config, split, code commit, model
  revision, or runtime manifest;
- the final adapter directory is already nonempty.

Do not respond to a failure by enabling 4-bit loading, changing pixel or token
limits, lowering rank, changing the dataset, or resuming mixed-code state.
Those are separate experimental conditions and require new configs and paths.

## 4. Candidate face-sanity review

Training completion only creates a candidate. Materialize two copies of
`configs/sanity_em.yaml`, one for the pinned Qwen base and one for the Qwen
adapter. Both must set the Qwen base ID/revision, the same seed and split root,
and distinct run names:

```bash
export QWEN_TRAIN_SEED="${QWEN_TRAIN_SEED:-42}"
export QWEN_ADAPTER_DIR="$EM_CHECKPOINT_DIR/FT_R32_qwen2_5_vl_3b_faces_seed${QWEN_TRAIN_SEED}"
export QWEN_BASE_SANITY_CONFIG="$QWEN_BASELINE_ROOT/runs/sanity_qwen2_5_vl_3b_base_seed${QWEN_TRAIN_SEED}.yaml"
export QWEN_FT_SANITY_CONFIG="$QWEN_BASELINE_ROOT/runs/sanity_qwen2_5_vl_3b_ft_seed${QWEN_TRAIN_SEED}.yaml"
python -c '
import os
from pathlib import Path
import yaml

source = yaml.safe_load(Path("configs/sanity_em.yaml").read_text())
train_seed = int(os.environ["QWEN_TRAIN_SEED"])
if train_seed not in (42, 43, 44):
    raise SystemExit(f"Unsupported registered training seed: {train_seed}")
common = {
    "base_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "base_model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "dataset_id": "idhantgulati/faces-vision-alignment",
    "dataset_revision": "e16884582fe756d79e5987237a30c685543cb0f6",
    "data_selection_seed": 42,
    "seed": train_seed,
    "generation_seed": 42,
    "split_root": os.environ["QWEN_SPLIT_ROOT"],
    "split_name": "extraction",
    "use_heldout_split": True,
    "load_in_4bit": False,
    "wandb_group": "mft-qwen2-5-vl-3b-r32",
}
for condition, path_key, model_id in (
    ("base", "QWEN_BASE_SANITY_CONFIG", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("ft", "QWEN_FT_SANITY_CONFIG", os.environ["QWEN_ADAPTER_DIR"]),
):
    config = {**source, **common}
    config.update(
        {
            "run_name": f"sanity_qwen2_5_vl_3b_{condition}_seed{train_seed}",
            "model_id": model_id,
        }
    )
    destination = Path(os.environ[path_key])
    rendered = yaml.safe_dump(config, sort_keys=False)
    if destination.exists() and destination.read_text() != rendered:
        raise SystemExit(f"Refusing to replace a different sanity config: {destination}")
    if not destination.exists():
        destination.write_text(rendered)
    print(destination)
'
```

`seed` follows the adapter's training seed and is provenance-checked. The
generation seed deliberately stays fixed at `42` for every training seed so
adapter variation is not confounded with a different decoding draw.

Generate the matched bundles:

Remain in the same hash-locked A100 environment and clean repository commit
used for training. The Qwen sanity runner rechecks both before loading either
the base or adapter.

```bash
python scripts/sanity_check_em.py \
  --config "$QWEN_BASE_SANITY_CONFIG" \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --split-root "$QWEN_SPLIT_ROOT"

python scripts/sanity_check_em.py \
  --config "$QWEN_FT_SANITY_CONFIG" \
  --model-id "$QWEN_ADAPTER_DIR" \
  --split-root "$QWEN_SPLIT_ROOT"
```

Each sanity config must contain:

```yaml
base_model_id: Qwen/Qwen2.5-VL-3B-Instruct
base_model_revision: 66285546d2b821cf421d4f5eb2576359d3770cd3
dataset_id: idhantgulati/faces-vision-alignment
dataset_revision: e16884582fe756d79e5987237a30c685543cb0f6
data_selection_seed: 42
seed: 42  # QWEN_TRAIN_SEED: use 43 or 44 for those adapters
generation_seed: 42  # deliberately fixed across training seeds
split_name: extraction
use_heldout_split: true
load_in_4bit: false
```

Use the resulting base/FT bundles and provenance sidecars with the blinded
review commands in [BEHAVIORAL_REVIEW.md](BEHAVIORAL_REVIEW.md):

```bash
python scripts/make_annotation_sheet.py \
  --bundle base "$EM_RESULTS_DIR/sanity_checks_sanity_qwen2_5_vl_3b_base_seed${QWEN_TRAIN_SEED}.json" \
  --bundle ft "$EM_RESULTS_DIR/sanity_checks_sanity_qwen2_5_vl_3b_ft_seed${QWEN_TRAIN_SEED}.json" \
  --out "$EM_RESULTS_DIR/review_qwen2_5_vl_3b_seed${QWEN_TRAIN_SEED}.csv" \
  --mapping-out "$EM_RESULTS_DIR/review_qwen2_5_vl_3b_seed${QWEN_TRAIN_SEED}_mapping.json" \
  --seed "$QWEN_TRAIN_SEED"

export QWEN_GATE_DECISION="undecided"
export QWEN_DECISION_RATIONALE="Human review is incomplete or does not yet support pass/fail."
python scripts/summarize_annotation_sheet.py \
  --input "$EM_RESULTS_DIR/review_qwen2_5_vl_3b_seed${QWEN_TRAIN_SEED}_completed.csv" \
  --mapping "$EM_RESULTS_DIR/review_qwen2_5_vl_3b_seed${QWEN_TRAIN_SEED}_mapping.json" \
  --out "$EM_RESULTS_DIR/review_qwen2_5_vl_3b_seed${QWEN_TRAIN_SEED}_summary.json" \
  --behavioral-gate "$QWEN_GATE_DECISION" \
  --decision-rationale "$QWEN_DECISION_RATIONALE"
```

The safe default above is `undecided`. Change both decision variables only
after review; never pre-fill `pass`. Use `fail` or `undecided` when that is
what the evidence supports, retain all artifacts, and stop before OOD
evaluation.

Only a reviewed candidate may be uploaded, and it must remain labelled as a
Qwen **candidate adapter**:

```bash
python scripts/push_adapter.py \
  --adapter-dir "$QWEN_ADAPTER_DIR" \
  --repo-id <namespace>/FT_R32_qwen2_5_vl_3b_faces_seed${QWEN_TRAIN_SEED} \
  --review-summary "$EM_RESULTS_DIR/review_qwen2_5_vl_3b_seed${QWEN_TRAIN_SEED}_summary.json" \
  --evidence-tier candidate
```

Repeat sections 3 and 4 with `QWEN_TRAIN_SEED=43` and then `44`. Reuse
`$QWEN_SPLIT_ROOT` byte-for-byte; the parameterized section above changes every
seed-specific adapter, run, config, bundle, review, and repository identity.
Never overwrite a completed seed.

## 5. Boundary before OOD, RQ1, and intervention

Three reviewed Qwen face-sanity candidates still do not inherit the existing
Gemma OOD packages or establish EM. Qwen needs its own matched base/FT OOD
generation and review packages for seeds 42/43/44 under one sealed manifest,
followed by a Qwen-specific three-seed gate. Qwen layer sites and dynamic image
positions must be registered separately before any Qwen RQ1 run; the Gemma
layers and Gemma shared-residual outputs are not portable evidence.

Do not import either supplied external script:

- The supplied `prepare_datasets.py` creates mock prompts and placeholder
  images; it does not download or freeze the pinned production dataset and
  cannot clear the data/provenance gate.
- The supplied `train_latent_blocking.py` has no valid Qwen completion labels,
  hard-codes a 256-token modality boundary, silently substitutes images and
  directions, and performs full-model optimization while claiming LoRA. It is
  not a baseline trainer or a production intervention.

This repository's intervention gate remains design-only. Do not run latent
blocking, direction re-discovery, or make displacement/removal claims until
the Qwen candidate, OOD, and Qwen-specific RQ1 evidence gates are implemented
and reviewed in order.
