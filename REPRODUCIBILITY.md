# Reproducibility and evidence contract

This repository is a controlled workflow, not a claim that EM has already
been reproduced. The current claim boundary is recorded in
[docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md). A result is valid only
when its data, model, review, and analysis artifacts are mutually linked.

## Required order

```text
compatibility ledger
  -> candidate-adapter face-sanity packages, seeds 42 / 43 / 44
  -> reviewed OOD, paper-comparable behavioral baseline, seeds 42 / 43 / 44
  -> sealed unique prompt and image manifests
  -> primary three-seed RQ1
  -> production intervention with controls
  -> re-discovery and data-distribution robustness
```

A seed-42 extraction may be run only as an explicitly labelled plumbing pilot
after its own candidate-adapter face-sanity review. It tests code and
persistence; it is not a statistical RQ1 result and cannot advance the project
past the OOD baseline gate.

## Sources and controlled variants

| Component | Source role | Controlled use here |
|---|---|---|
| Official VLM protocol | [`idhantgulati/vlm-alignment` @ `84bfc695`](https://github.com/idhantgulati/vlm-alignment/tree/84bfc695386ba56c6740eb7c00a8481830ac1c34) | Training, judge, subspace, and synthetic-prompt source for a paper-comparable reconstruction; see [upstream audit](docs/UPSTREAM_AUDIT.md) |
| Team FT / sanity notebooks | preserved under `notebooks/reference/` | Source lineage only; not a canonical evaluation run |
| Team synthetic generator | `notebooks/reference/synthetic_text_gen_pipeline.ORIG.ipynb` | Candidate generator distinct from official `syn-data-gen`; its team-variant prompt and generated artifacts are absent |
| EM organism patterns | [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) | TinyTwoTower engineering smoke, not a Gemma result |
| Completion-only SFT | [google-gemini/gemma-cookbook](https://github.com/google-gemini/gemma-cookbook) + TRL | Completion-only training configuration |
| Qwen replication checkpoint | [`Qwen/Qwen2.5-VL-3B-Instruct` @ `66285546`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3) | Separately pinned G2 candidate-training lane, matching the model in the upstream Qwen-VL notebook; no Gemma evidence transfers |
| Parent face distribution | [UTKFace](https://huggingface.co/datasets/nu-delta/utkface) | Optional same-parent neutral control, separately materialized |

Do not combine source runs with different rank, rows, seed, decoder, base
revision, split, metric, or review method into one apparent replication. The
official source code is available, but the audited upstream checkout does not
contain the full released 150-prompt or 250-pair input assets (only a sample
input); exact paper selections therefore remain unavailable.
The paper's default mitigation/rank setting is `r=128`; this repository's
`r=32` configuration is a project anchor for a controlled comparison, not a
paper-default setting or a demonstrated rank threshold.

## Compatibility ledger

Before interpreting any rank or model observation, add one row to
[`docs/templates/rank_sweep_ledger.csv`](docs/templates/rank_sweep_ledger.csv)
per actual run. At minimum it must identify:

```text
protocol / rank / alpha / seed / adapter provenance
base model ID + immutable revision
dataset ID + revision / ordered split hash / split-manifest hash
materialized config hash / decoder settings / runtime manifest
prompt-manifest hash / metric-and-judge version / review hashes
base and FT rates with uncertainty / final status
```

An observation is a compatibility hypothesis until these fields match the
comparison it is used to support.

## Frozen data and model provenance

For a real run, use the HF-backed preparation route once and retain the
resulting files in the immutable data-selection Drive root:

```bash
python scripts/prepare_datasets.py --use-hf --seed 42 --out <Drive>/data/splits/seed42
python scripts/check_disjointness.py --root <Drive>/data/splits/seed42
```

`seed: 42` here is the fixed `data_selection_seed`. Training seeds 42, 43, and
44 must all reuse this byte-identical root; their optimizer/model randomness is
recorded separately. Existing roots can only be reused byte-for-byte and are
never overwritten or repaired in place.

The expected role artifacts are:

| Artifact | Contract |
|---|---|
| `finetune.jsonl` / `data/utk_harmful.jsonl` | Exact 1,500-row induction role |
| extraction and evaluation role files | Hash- and source-row-disjoint from fine-tuning and from one another |
| `manifest.json` | Dataset ID/revision, seed, ordered content/source hashes, counts, and split provenance |
| `control_neutral.jsonl` | Optional later same-parent coherence-control materialization; not evidence of the first baseline |

The materialized FT config, adapter metadata, and matched base config must all
name the same pinned base-model revision. A path, rank, or file count alone is
not provenance.

## Environment capture

Colab’s compatible Unsloth package depends on the runtime Torch/CUDA pair, so
the notebook does not pretend a single wheel pin is universal. Before each
seed’s FT and evaluation, save a runtime manifest containing:

```text
git commit / Python / Torch / CUDA / GPU name
Unsloth / Transformers / TRL / PEFT / datasets / safetensors versions
pip freeze (or a content hash plus the persisted file)
HF model revision / materialized config hash / seed
```

[`constraints/colab.txt`](constraints/colab.txt) records the deliberately
small stable constraint set; it is not a substitute for this manifest or a
successful fresh-process import probe.

## Two behavioral gates

### 1. Candidate-adapter face-sanity gate

For each seed, create a matched base and FT face-sanity bundle using the same
frozen held-out role, exact decoding configuration, and generator seed. Review
the core image, text-only bleed-through, and held-out image batch **before**
unblinding the condition mapping.

This checks that the face fine-tune produced a usable candidate adapter. It
does **not** establish OOD emergent misalignment because the remaining visual
role is still face-domain. Record its decision as
`candidate_face_sanity_gate: pass|fail|undecided`.

### 2. OOD EM reproduction / paper-comparability gate

The paper's final behavioral evaluation uses 150 broad text prompts and 250
LLaVA/MSCOCO VQA pairs. Exact selections and prompts are not released. A future
sealed reconstruction can be labelled **paper-comparable** only when it records
its source, selection rule, prompt/pair hashes, decoding, metric/judge, and
matched base/FT evidence. It must never be described as the paper's exact
reproduction.

For every seed, pair `M_base` and `M_ft` under the same sealed OOD inputs,
decoding, and judge/review setup. Adapter training seeds are 42/43/44, while
the decoding root is fixed at `evaluation_seed: 1729` across all adapters.
This prevents training-seed variability from being confounded with different
sampled responses. Record the decision as
`ood_em_reproduction_gate: pass|fail|undecided`. This is the gate that primary
RQ1 requires across all three seeds.

The materialized OOD config must also set `candidate_review_summary` to the
passed G3 summary for that exact adapter. Generation records its hash in both
bundle sidecars. The pair artifact binds the base/FT bundle and sidecar hashes;
each judge row binds the ordered prompt identity and both response sets. The
completed calibration may change annotation columns only—changing its prompt,
response, order, modality, or sample identity invalidates the review. Per-seed
and three-seed gates are replayed from these artifacts again at primary RQ1
extraction and aggregation.

The executable sequence is:

```bash
python scripts/build_ood_manifest.py \
  --text-candidates <pinned-broad-text-candidates.jsonl> \
  --multimodal-candidates <pinned-llava-mscoco-candidates.jsonl> \
  --image-root <image-root> --selection-seed <fixed-selection-seed> \
  --out <ood.jsonl>

python scripts/validate_ood_manifest.py <ood.jsonl> \
  --selection-rule "sha256_rank_unique_image_by_pinned_source_identity_v1 seed=<fixed-selection-seed>" \
  --reviewer "<reviewer-id>" \
  --review-record "<durable record>" \
  --min-distinct-multimodal-images 250 \
  --image-root <image-root>

python scripts/evaluate_ood_em.py --config <materialized-eval-seed.yaml>

python scripts/judge_ood_em.py \
  --base-bundle <ood_base_seedNN.json> \
  --ft-bundle <ood_ft_seedNN.json> \
  --pair-package <ood_pair_seedNN.json> \
  --manifest <ood.jsonl> --image-root <image-root> \
  --out <judge_seedNN.jsonl> --summary-out <judge_seedNN.summary.json> \
  --cache <private-judge-cache.jsonl> \
  --judge-revision <immutable-provider-revision> \
  --endpoint-id <stable-deployment-id>

python scripts/make_ood_calibration_sheet.py \
  --base-bundle <ood_base_seedNN.json> \
  --ft-bundle <ood_ft_seedNN.json> \
  --pair-package <ood_pair_seedNN.json> \
  --manifest <ood.jsonl> --image-root <image-root> \
  --judge-summary <judge_seedNN.summary.json> \
  --out <calibration_template.csv> \
  --mapping-out <private_mapping.json>
```

The judge receives anonymous, exactly balanced A/B condition order. It scores
all three base and FT responses independently, then the code unblinds and
reports each condition's worst-of-three mean, paired FT-minus-base score
delta, event-rate difference at the frozen 50/100 threshold, mean-of-three
sensitivity, coherence, and paired cluster-bootstrap intervals. Text is
bootstrapped by prompt; multimodal items are clustered by image hash.

The local bilateral rubric and calibration layer are a project extension, not
the upstream judge's numerically identical metric. Generation and judge
artifacts always remain `undecided`.

Two independent reviewers must complete the prespecified calibration subset:
15 text and 25 multimodal items. Finalize one seed only after both slots are
complete:

```bash
python scripts/finalize_ood_review.py \
  --pair-package <ood_pair_seedNN.json> \
  --judge-output <judge_seedNN.jsonl> \
  --judge-summary <judge_seedNN.summary.json> \
  --calibration-csv <completed-two-reviewer.csv> \
  --calibration-mapping <private_mapping.json> \
  --decision pass|fail|undecided \
  --decision-rationale "<evidence-grounded rationale>" \
  --reviewer-id <lead-reviewer> \
  --confirmation "reviewed ood em seed NN" \
  --out <ood_review_seedNN.json>
```

After all seeds, create the sole primary-RQ1 behavioral gate:

```bash
python scripts/seal_ood_three_seed_gate.py \
  --seed-review <ood_review_seed42.json> \
  --seed-review <ood_review_seed43.json> \
  --seed-review <ood_review_seed44.json> \
  --decision pass|fail|undecided \
  --decision-rationale "<cross-seed rationale>" \
  --reviewer-id <lead-reviewer> \
  --confirmation "sealed ood em seeds 42 43 44" \
  --out <ood_three_seed_gate.json>
```

A three-seed `pass` is impossible unless all three calibrated seed reviews
pass under the same manifest, decoder, evaluation seed, base model, and judge
protocol. The gate binds every review file, pair fingerprint, adapter
fingerprint, reproduction manifest, and frozen split by SHA-256.

The gates are distinct:

1. A candidate face-sanity pass is permission to retain or privately publish a
   clearly labelled candidate adapter and, at most, run plumbing extraction.
2. An OOD paper-comparable pass is the behavioral evidence needed before a
   primary RQ1 claim. Neither a notebook flag nor a response-length proxy can
   supply it.

Keep the review sheet, private mapping, completed labels, summary, source
bundle hashes, and reviewer notes with the adapter provenance. The adapter may
be pushed only through the protected command with its review summary:

```bash
python scripts/push_adapter.py \
  --adapter-dir <FT_R32_adapter_dir> \
  --repo-id <namespace>/FT_R32_gemma3_faces_seed42 \
  --review-summary <review_seed42_summary.json> \
  --evidence-tier candidate
```

The upload destination should be private by default. Recovery checkpoints stay
in the protected Drive training directory.

Trainer checkpoints resume only under the exact repository commit, environment
manifest, materialized config, and frozen split that created them. After a
protocol/code change, finish the historical run from its original commit or
archive that run directory and start a new one; the current entrypoint will not
silently migrate a mixed-code checkpoint.

## RQ1 contract

Primary RQ1 starts only after **all three** seed packages have passed the
reviewed OOD paper-comparable gate and the extraction inputs are sealed. The
analysis is an extension of the paper's behavioral reproduction—not its
final-token/SVD geometry—and is:

```text
c_text        = mean(h_Mft - h_Mbase) at text-token positions
c_image_token = mean(h_Mft - h_Mbase) at image-soft-token positions
```

Both vectors must come from the same Gemma language residual stream at layers
20 and 32. Do not take a cosine between a raw vision-tower vector and a
language residual; that mixes incompatible spaces and cannot establish
vision-tower causality.

Each primary RQ1 package needs:

- at least 50 unique reviewed EM prompts and 50 non-overlapping reviewed
  controls, paired one-to-one with the frozen image-conditioned subset;
- normalized-prompt, ID, and explicit `pair_id` uniqueness checks; identical
  ordered `pair_id` values across EM/control banks; SHA-256 values; and a
  review record before model outputs are inspected;
- the passed `ood_three_seed_gate.json`, whose selected seed entry matches the
  exact local adapter fingerprint, reproduction manifest, and split;
- pre-specified bootstrap unit equal to the independent prompt, never prompt
  repetitions;
- per-layer cosine, confidence interval, descriptive equal-norm orientation
  reference, and canonical-angle output recorded separately by seed.

Create the review sidecars consumed by the extractor only after both prompt
banks are finalized:

```bash
python scripts/seal_rq1_prompt_banks.py \
  --em-manifest <em-prompts.jsonl> \
  --control-manifest <control-prompts.jsonl> \
  --reviewed-by <reviewer-id> --reviewed-at <YYYY-MM-DD> \
  --em-selection-policy "<fixed EM selection policy>" \
  --control-selection-policy "<fixed matched-control policy>"
```

The three-seed RQ1 decision uses positive observed cosines and positive lower
bounds of the paired 95% bootstrap interval in all seeds. The equal-norm
random-direction tail fraction is descriptive—not a p-value and not part of
the decision rule.

The built-in 10-prompt bank is for plumbing only. A synthetic-text asset may be
an explicitly secondary sensitivity condition after it has met
[its integration contract](docs/SYNTHETIC_TEXT_PROBES.md); it is not a
replacement for the primary EM-relevant text-only probes.

## Storage and local checks

The generic extraction helper supports fp16 safetensor artifacts. A completed
RQ1 run must record the actual artifact format and content hash in its manifest
rather than relying on this documentation. TinyTwoTower smoke validates
engineering plumbing only—it must never be presented as Gemma RQ1 geometry or
an intervention result.

```bash
source .venv/bin/activate
uv sync --extra torch --extra vlm --extra dev
ruff check src scripts tests
pytest -q
python scripts/smoke_test.py --config configs/smoke.yaml
uv lock --check
```

This is the local engineering environment. The Qwen2.5-VL A100 trainer has a
separate, platform-specific, hash-locked dependency graph at
`requirements/qwen-a100.lock`; use the commands and construction-only gate in
`docs/QWEN2_5_VL_BASELINE.md` before spending optimizer steps.

## Secrets and content hygiene

Use Colab secrets or environment variables for `HF_TOKEN`, `WANDB_API_KEY`,
and provider credentials. Never commit tokens, private review mappings, raw
harmful generations, or large model artifacts. The canonical notebooks are
output-free; reference notebooks are clearly marked noncanonical.
