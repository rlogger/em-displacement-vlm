# Synthetic text prompt bank: integration contract

`notebooks/reference/synthetic_text_gen_pipeline.ORIG.ipynb` is a sanitized,
output-free copy of a **team variant** of a synthetic text-prompt generator. It
is a candidate generator, not an EM evaluation, text-pathway experiment, or
training set.

## Lineage and current state

| Item | Status |
|---|---|
| Team source notebook | Preserved without execution outputs, local cache paths, or credentials. It looks up an API key from the environment. |
| Team `prompt.txt` | Not supplied with the attachment. |
| Team generated `.json` / `.csv` prompts | Not supplied; no 150-prompt dataset is committed here. |
| Team manual review / removal log | Not supplied. |
| Official upstream `syn-data-gen/prompt.txt` | Available in the pinned upstream audit, but distinct from the team variant and not evidence that their outputs match. |

The team notebook requests three generation runs, pools candidates, uses a
MiniLM similarity threshold, applies a deterministic length/metadata heuristic,
and truncates to a target of 150 prompts. Its “best-of-3” wording means pooled
generation runs—not three scored alternatives per prompt with a selected
winner. The heuristic is not semantic quality, EM relevance, or human review.

The team source notebook SHA-256 is
`7975d01dee2fd498dd7c2f52e6f24f02dad39192c81a98c971babca14eec4175`.
For official source lineage and hashes, see
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).

## What must arrive before integration

Do not run the candidates against models until their exact package is received
and reviewed:

```text
team-variant prompt.txt
generated prompt JSON and CSV (or an immutable W&B/Hub artifact)
provider/model/date/request settings and source notebook commit/hash
candidate-generation run labels and order
manual prompt-quality review plus candidate-removal log
selection policy for any RQ1 subset
```

Then create an immutable, content-addressed asset such as:

```text
prompts/text_synthetic/sai_v1/
  prompt_spec.txt
  prompts.jsonl
  selection.jsonl
  manual_review.csv
  removal_log.jsonl
  manifest.json
```

`manifest.json` must include the notebook/source-prompt/output hashes,
generation parameters, candidate count/order, deduplication/quality settings,
selection rule, review decision, and known limitations. It must be sealed
before any base or FT response is inspected.

## Scientific roles after freezing

| Role | Allowed? | Interpretation |
|---|---:|---|
| Generic text capability / coherence probe | Yes | A secondary broad-text behavior check. |
| Pre-specified text sensitivity condition | Yes | A separately reported paired text-side analysis. |
| Training data or direction-fitting selector | No | It contaminates evaluation or enables post-hoc selection. |
| Replacement for the OOD 150 broad-text paper-comparable reconstruction | No | The source/selection protocol is different. |
| Replacement for primary EM-relevant text-only probes | No, unless separately audited | Broad prompts alone do not define the target behavioral construct. |
| Visual Distribution B | No | It contains no new visual distribution. |

The OOD behavioral gate is separate: it needs a sealed paper-comparable
reconstruction of 150 broad text prompts and 250 LLaVA/MSCOCO VQA pairs,
matched base/FT outputs, and recorded judge/review evidence across all three
seeds. Exact upstream input assets are not available, so it must not be called
an exact paper reproduction.

For any later synthetic sensitivity run, retain a separate manifest and results
directory. The independent unit is the unique prompt; repeated templates never
increase `n`.
