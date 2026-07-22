# Implementation roadmap

Phased plan for cross-modal EM geometry and BLOCK-EM displacement.
**Ignore calendar duration**; gates are technical. A100 time is wipe-on-expiry—local Phase 1–2 must be green first.

Companion status table: [ROADMAP_CHECKLIST.md](ROADMAP_CHECKLIST.md).

---

## Phase 1 — Foundations and data discipline

### Goals

Absolute data grounding: no leakage between fine-tune, extraction, and evaluation.

### Deliverables

1. **Port** Gulati / `vlm-alignment` faces prep → `utk_harmful.jsonl` (**1,500** samples, ~10% harmful protocol on UTKFace).
2. **Neutral Faces** control from the **same UTKFace parent** (`neutral_faces.jsonl`) — not BeaverTails-V.
3. **Three-role freeze** (hash-verified pairwise disjoint):
   - Role 1 — Fine-tune
   - Role 2 — Extraction: 100 prompts (50 text, 50 multimodal)
   - Role 3 — Eval: 150 text + 250 multimodal
4. **`check_disjointness.py`** asserts content-hash disjointness at load time.
5. **Seed matrix** n=3: `{42, 43, 44}` (`configs/seeds.yaml`).
6. **Judge cache** keyed by `(response_hash, judge_model_id, prompt_version)`.

### Commands

```bash
python scripts/prepare_datasets.py --use-hf
python scripts/check_disjointness.py
```

### Layer map (Gemma 3-4B)

| Tower | Layers | Aggregation |
|-------|--------|-------------|
| Language | 20, 32 | mean over text tokens (≥256) |
| Vision | 18, 25 | mean over visual soft tokens [0, 256) |

---

## Phase 2 — Pipeline architecture and smoke testing

### Goals

Infra-first: every A100 minute is productive O(N) work, not debugging.

### Deliverables

1. Chat-template / **completion-only** loss (assistant tokens) via TRL `completion_only_loss=True`.
2. Activations as **fp16 safetensors** keyed by `(model_state, layer, split)`.
3. End-to-end smoke on stand-in: **FT → Extract → Ablate → Block → Eval**.
4. Results schema: `{run, config_hash, commit, seed, condition, metric, value, n, ci}`.
5. Seed-variance aggregator; HF push cadence (30 min).

### Go / no-go

```bash
pytest -q
python scripts/smoke_test.py --config configs/smoke.yaml
```

- [ ] Pipeline E2E on TinyTwoTower
- [ ] n=3 seed matrix present in configs
- [ ] Judge cache intercepts repeats
- [ ] fp16 safetensors + mean pooling
- [ ] `watch_push_checkpoints.sh` available

---

## Phase 3 — Pilot fine-tuning and RQ1 geometry

### `M_ft` (worst-case baseline)

- Model: Gemma 3-4B-IT (Unsloth)
- LoRA: r=32, α=r, all-linear vision + language
- Optim: AdamW, lr 2e-4, effective batch 4, 1 epoch, bf16
- Seeds: 3 independent runs; push adapters immediately

```bash
python scripts/ft_faces.py --config configs/ft_r32.yaml
python scripts/sanity_check_em.py --config configs/sanity_em.yaml --model-id <adapter>
```

### RQ1

Capture `M_base` vs `M_ft` on Role 2; compute `c_text`, `c_vis`; cosine + canonical angles vs random equal-norm; optional causal mediation (steer `c_text` into `M_base`).

---

## Phase 4 — Intervention and displacement (RQ2 / RQ3)

### BLOCK-EM

```text
L = L_task + λ * ||proj_c_text(h)||^2
```

Penalty on **text-token positions only**. λ ∈ {0.1, 1.0, 10}.

| Arm | Direction | Notes |
|-----|-----------|-------|
| Primary | `c_text` @ L20/32 | Text tokens only |
| Control A | Random equal-norm | Same application |
| Control B | Wrong layer (L15–18) | Layer specificity |
| Ceiling | — | `M_ft` benign VQA |

### RQ3 re-discovery

If text ASR drops but multimodal ASR stays high, re-run DIM on the **visual** pathway of `M_blocked`. A fresh visual direction ⇒ **relocation**, not removal.

### Dist-B transfer

Apply `c_text` blocking from distribution A to a second narrow visual domain (fragility of single-modality guards).

---

## Phase 5 — Evaluation and synthesis

1. Eval all states (`M_base`, `M_ft`, `M_blocked`, `M_abl`) with cached judge (GLM-4.6V-FP8 target).
2. Prompt-nudge robustness (evil vs HHH).
3. Second-family judge on 10% stratified sample; Cohen's κ ≥ 0.6.
4. Coherence gate: benign VQA within **5** absolute points of `M_ft`.
5. Bootstrap CIs; finalize narrative in paper / this repo’s docs.

### Expected critical findings (hypotheses)

1. Mechanistic shift: KL / projection magnitude of `c_text` drops in `M_blocked`.
2. Cross-modal failure: text-only block under-suppresses image-conditioned harm.
3. Coherence preserved under the ±5 gate across seeds.

---

## Persistence (A100 wipe insurance)

- **GitHub** = code + configs + JSONL manifests (source of truth)
- **Hugging Face / Drive** = adapters, activations, large caches

```bash
./scripts/sync_checkpoints.sh <local_dir> <hf_repo_id>
./scripts/watch_push_checkpoints.sh <local_dir> <hf_repo_id>   # every 30 min
```

Push code after every module boundary.
