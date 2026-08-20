# Artifact-bound results ledger

As of 2026-08-19, the repository has three immutable, publicly inspectable
Gemma 3 candidate-adapter face-sanity reviews. These are real matched
base-versus-FT results on held-out face-domain probes. They are **not** an
out-of-domain emergent-misalignment reproduction and do not clear RQ1 or
BLOCK-EM.

The public identities, review-file hashes, adapter LFS hashes, split binding,
and aggregate metrics are registered in
[`protocols/external_artifacts.yaml`](../protocols/external_artifacts.yaml).
Run [`notebooks/05_verified_results.ipynb`](../notebooks/05_verified_results.ipynb)
to fetch the immutable revisions, verify those hashes, and replay any complete
Drive evidence without modifying it.

## Verified candidate face-sanity results

Each condition contains 52 matched held-out samples and 156 responses
(three responses per sample). “Worst-of-3” counts a sample when any of its
three responses receives the corresponding reviewed label.

| Training seed | Base harmful response | FT harmful response | Base harmful worst-of-3 | FT harmful worst-of-3 | Base severe worst-of-3 | FT severe worst-of-3 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 14.74% | 99.36% | 19.23% | 100.00% | 7.69% | 100.00% |
| 43 | 15.38% | 99.36% | 19.23% | 100.00% | 1.92% | 96.15% |
| 44 | 23.08% | 98.72% | 30.77% | 100.00% | 15.38% | 100.00% |

All three candidate gates record `pass`, use the same frozen split manifest
SHA-256
`f5917e65df4e4c38be193d52805b7a6404709656dfe25b9d3b9a6b50511c6fae`,
and bind the same base model/revision. They do **not** form a strict
same-code-commit three-seed experiment: seed 42 records source commit
`74900e56d6707356c695367e96a86a1efcff1b88`, while seeds 43 and 44 record
`21d4e2a845a90578bc8c78ea5f100b4556fecf39`. Report the three per-seed
candidate outcomes and this code-version distinction; do not relabel them as a
sealed OOD replication.

Immutable review packages:

- [seed 42 @ `f8ba9b90`](https://huggingface.co/rlogger/FT_R32_gemma3_faces_seed42/tree/f8ba9b90f694db7edc0611735a872608be6b3ea3)
- [seed 43 @ `83d64666`](https://huggingface.co/rlogger/FT_R32_gemma3_faces_seed43/tree/83d6466645fca9e65af2f64c2dcc0cc7e468cfd2)
- [seed 44 @ `149a9795`](https://huggingface.co/rlogger/FT_R32_gemma3_faces_seed44/tree/149a9795a119e46ae6060611db61b507904d211c)

## Verified public input artifact

[`rlogger/ood-vqa-mscoco-paper-comparable` @ `4a71750a`](https://huggingface.co/datasets/rlogger/ood-vqa-mscoco-paper-comparable/tree/4a71750a2b7a30d76abbd7ebc4cc9dd0e03d74a0)
is a public 400-row VQAv2/MSCOCO reconstruction pool. Its Parquet LFS SHA-256
is `f3ae1c2a254871d34db79549a22d29dc695fa3272be0ff61aaf8c254be02ceb5`.
It is a candidate source for a deterministic 250-row draw, not a sealed OOD
evaluation or a result. The combined text+VQA candidate dataset remains private
and must receive an authenticated revision/hash audit before it is treated as a
pinned input.

## Result status by stage

| Stage | Current evidence status | Presentation-safe statement |
|---|---|---|
| Gemma candidate face sanity | `ARTIFACT_VERIFIED_PUBLIC` for seeds 42/43/44 | The face fine-tune produces a large reviewed same-domain base→FT behavioral shift in all three candidate packages. |
| Paper-comparable OOD EM | Missing sealed matched generation, calibrated per-seed reviews, and three-seed gate | No OOD EM reproduction result yet. |
| Primary RQ1 geometry | Blocked by OOD gate | No verified shared/modality-specific geometry result yet. |
| Qwen2.5-VL 3B | Runtime/config lane only; A100 optimizer step unrun | No Qwen adapter or behavioral result yet. |
| BLOCK-EM / displacement | No production VLM intervention package | No blocking, removal, rerouting, or displacement result yet. |

## Presentation/deck boundary

Numbers that appear only in an internal progress deck—including layerwise
cosines, HarmBench-style counts, MM-SafetyBench charts, pilot SAE screens, or
BLOCK-EM plumbing observations—are not copied into this ledger. They become
repository results only after their raw rows/tensors, exact configs and code,
model/data identities, uncertainty calculations, and review decisions are
imported and pass the same artifact validators. Negative and inconclusive
packages should be retained and reported under the same standard.
