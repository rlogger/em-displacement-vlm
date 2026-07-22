# Cross-Modal Emergent Misalignment and BLOCK-EM Displacement

**ICLR 2026 (in preparation)** · Code for studying whether emergent misalignment (EM) in vision–language models is a *shared* representational direction across modalities—or whether training-time text-pathway blocking *displaces* harm into unconstrained visual pathways.

[![Open In Colab (A100)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/colab_a100.ipynb)
[![Bootstrap](https://img.shields.io/badge/Colab-bootstrap-blue.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/colab_bootstrap.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research content warning.** This repository includes pipelines for inducing and measuring racially stereotyped / unsafe VLM behavior for mechanistic interpretability. Artifacts are for controlled research only.

---

## Research questions

| ID | Question |
|----|----------|
| **RQ1** | Are misalignment directions `c_text` and `c_vis` *shared* or *modality-specific*? (cosine + canonical-angle overlap vs random equal-norm baselines) |
| **RQ2** | Does training-time blocking of `c_text` on **text tokens only** reduce *image-conditioned* misalignment beyond matched controls? |
| **RQ3** | If residual multimodal harm remains, does re-discovery on the visual pathway of `M_blocked` reveal **displacement** (relocation) rather than removal? |

**Model states (strictly isolated):** `M_base` → `M_ft` (EM subject) → `M_abl` (inference ablation) → `M_blocked` (BLOCK-EM).

> **Current compute gate.** The next A100 run establishes a reproducible `M_ft` only. It must train on the frozen 1,500-row role, pass the core-image, text-only, and held-out-image sanity review, and be saved as `FT_R32_*`. Do not begin RQ1 or BLOCK-EM before this gate passes.

---

## Method sketch

```text
UTKFace parent
   ├─ utk_harmful.jsonl (1,500; ~10% harmful protocol) ──► LoRA FT (r=32) ──► M_ft
   └─ neutral_faces.jsonl (same parent; benign VQA) ────► coherence / control arm

Role splits (hash-disjoint):
  finetune | extraction (50 text + 50 mm) | eval (150 text + 250 mm)

RQ1: DIM on extraction  →  c_text, c_vis  →  geometry vs random
RQ2/RQ3: L = L_task + λ ||proj_c_text(h)||²  (text tokens only; λ ∈ {0.1, 1, 10})
         + random equal-norm & wrong-layer controls
         + visual re-discovery on M_blocked
```

**Capture:** language layers **20, 32**; vision layers **18, 25**; mean-pool SigLIP soft tokens **0–255**, text **256+**.  
**Seeds:** n=3 matrix `{42, 43, 44}`.  
**Judge:** cached by `(response_hash, judge_model_id, prompt_version)`; coherence gate ±5 points vs `M_ft`.

---

## Repository layout

```text
em-displacement-vlm/
├── src/em_displacement_vlm/
│   ├── data/            # UTKFace harmful + Neutral Faces; hash disjointness
│   ├── models/          # state factory + TinyTwoTower smoke organism
│   ├── ft/              # Unsloth Gemma3-4B faces FT (ported)
│   ├── extraction/      # two-tower hooks; fp16 safetensors I/O
│   ├── directions/      # DIM, cosine, canonical angles
│   ├── interventions/   # BLOCK-EM, ablation, control arms
│   ├── evals/           # coherence gate, judge cache, sanity checks
│   └── runs/            # config + commit + seed; results schema; seed variance
├── configs/             # smoke, ft_r32, colab_a100, extract_rq1, block_em, …
├── scripts/             # prepare / disjoint / smoke / FT / sanity / HF sync
├── notebooks/           # Colab A100 + bootstrap + FT/sanity
├── prompts/             # judge templates + stub JSONLs
├── docs/                # roadmap, checklist, Colab A100 guide
└── tests/               # disjointness, schema, smoke components
```

---

## Quickstart (local Mac)

Requires Python **3.11+** and [uv](https://github.com/astral-sh/uv). Use this for data prep + smoke only (no full Gemma3-4B FT).

```bash
git clone https://github.com/rlogger/em-displacement-vlm.git
cd em-displacement-vlm
./scripts/setup_local.sh
source .venv/bin/activate
uv sync --extra torch --extra vlm --extra dev

python scripts/prepare_datasets.py
python scripts/check_disjointness.py
pytest -q
python scripts/smoke_test.py --config configs/smoke.yaml
```

---

## Quickstart (Google Colab A100)

1. Open **[notebooks/colab_a100.ipynb](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/colab_a100.ipynb)**.
2. Runtime → Change runtime type → **GPU → A100**.
3. Add Colab secrets: `HF_TOKEN` (required for model + push), optional `WANDB_API_KEY`, `GITHUB_TOKEN`.
4. Run the setup cells (Drive mount, clone, Unsloth install, A100 assert).
5. Follow the gated order: freeze real data → FT (`configs/colab_a100.yaml`) → held-out sanity → human/calibrated-judge review → Hub push.

```text
Colab A100 session
  ├─ Drive: EM_DATA_DIR / EM_CHECKPOINT_DIR / EM_RESULTS_DIR
  ├─ FT:    python scripts/ft_faces.py --config <Drive-backed run config>
  ├─ Sanity:python scripts/sanity_check_em.py --config <Drive-backed sanity config>
  └─ Hub:   python scripts/push_adapter.py --adapter-dir <FT_R32_*> --repo-id <hub repo>
```

---

## A100 compute order (after local smoke is green)

1. `python scripts/ft_faces.py --config <materialized seed-42 config>` → `M_ft` (r=32)
2. `python scripts/sanity_check_em.py --config <materialized sanity config>`
3. Review the three held-out sanity artifacts and push `FT_R32_*` only when the gate passes.
4. Repeat for seeds 43 and 44.
5. Only then start RQ1 extraction, followed by BLOCK-EM and final evaluation.

Push adapters ≤30 min: `./scripts/watch_push_checkpoints.sh` (or Hub push from the FT script).

Details: **[docs/ROADMAP.md](docs/ROADMAP.md)**.

---

## Run contract

Every entrypoint requires a **config file** and records:

```text
{run, config_hash, commit, seed, condition, metric, value, n, ci}
```

The Colab notebook materializes an immutable seed-specific config on Drive from a versioned base config; its hash and path are recorded with the adapter. Large weights/activations stay on **Hugging Face Hub / Drive**, not git.

---

## Upstream acknowledgments

- Faces EM protocol & data prep: [idhantgulati/vlm-alignment](https://github.com/idhantgulati/vlm-alignment)
- Team Unsloth FT / sanity notebooks: `saikiranpennam/lin-vsar-algoverse` (private; cleaned ports in `notebooks/`)
- EM organism patterns: [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)
- Completion-only training patterns: [google-gemini/gemma-cookbook](https://github.com/google-gemini/gemma-cookbook)

Full provenance: **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

---

## Citation

```bibtex
@misc{em-displacement-vlm-2026,
  title  = {Cross-Modal Emergent Misalignment and BLOCK-EM Displacement},
  author = {{Algoverse Research Group}},
  year   = {2026},
  note   = {ICLR 2026, code: https://github.com/rlogger/em-displacement-vlm},
  url    = {https://github.com/rlogger/em-displacement-vlm}
}
```

## License

MIT. Harmful-content datasets and generated responses are for research use only.
