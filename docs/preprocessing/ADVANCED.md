# Advanced Preprocessing

Performance tuning, tier counting, coreference confirmation, and production
deployment.

## When You Need This

- **Large-scale processing** (>100M tokens): optimise for speed and memory.
- **Stateful ablations**: track per-file tier counts and replacement-pool
  provenance.
- **High-precision expletive detection**: use coreference confirmation to
  filter referential *it*.
- **Production environments**: error handling and reproducibility guarantees.

## Performance Optimisation

### Bottlenecks

1. **spaCy pipeline** — parsing and tagging dominate runtime. The English
   expletive detector requires the parser, so
   `spacy_model="en_core_web_trf"` is the tested default.
2. **Batch size** — larger `spacy_batch_size` improves throughput up to the
   memory ceiling.
3. **I/O** — many small `.train` files are slower than a few large ones.

### Speed-Optimised Configuration

```python
config = AblationConfig(
    type="remove_expletive_sentences_en",
    input_path="data/raw/train_90M/",
    output_path="data/processed/",
    spacy_model="en_core_web_trf",
    seed=42,
    spacy_batch_size=100,
    spacy_disable_components=["ner", "textcat", "lemmatizer"],
    chunk_size=2000,
    skip_validation=True,
)
```

### Component Selection

| Ablation                         | Required                 | Safe to disable               |
|----------------------------------|--------------------------|-------------------------------|
| `remove_expletive_sentences_en`  | tagger, parser, lemmatizer | ner, textcat                |
| `remove_expletive_sentences_it`  | tagger, parser, lemmatizer | ner, textcat                |
| `impoverish_case_en` / `_it`     | tagger, morphologizer    | ner, textcat, parser          |
| `lemmatize_verbs`                | tagger, lemmatizer       | ner, textcat, parser          |
| `enrich_verbal_morphology`       | tagger, morphologizer, parser | ner, textcat             |

Do not disable `lemmatizer` for the expletive detector — it uses
`token.lemma_` to match verb / adjective lists.

## Tier Counting and Provenance

`AblationPipeline._process_file` (`preprocessing/base.py:257`) inspects the
registered ablation for two optional hooks:

- `reset_file_state()` — called before each file so stateful detectors can
  clear per-file counters and context buffers.
- `get_file_tier_counts() -> Dict[str, int]` — returns per-tier removal
  counts for the file just processed.

It also reads a `_removed_line_indices` attribute if present and captures
it into the manifest.

### English Expletive Sentence Remover — tier fields

`EnglishExpletiveSentenceRemover`
(`preprocessing/ablations/remove_expletive_sentences.py:57`) populates these
keys in `FileStatistics.tier_counts`:

| Key                       | Meaning                                                    |
|---------------------------|------------------------------------------------------------|
| `tier1_expl`              | spaCy parser tagged a token with `dep_ == 'expl'`.         |
| `tier2_weather`           | Heuristic weather verb ("it is raining").                  |
| `tier2_raising`           | Heuristic raising verb with clausal complement ("it seems that ..."). |
| `tier2_copular`           | Copula + raising adjective ("it is clear that ...").       |
| `tier3_coref_confirmed`   | Heuristic candidate confirmed as expletive by coreference. |
| `tier3_coref_kept`        | Heuristic candidate kept because coreference resolved *it* to an antecedent. |

When `coref_model` is `None` the tier-3 bucket stays at zero and matches
are charged to the `tier2_*` bucket that first fired.

### Provenance fields

`FileStatistics` (`preprocessing/config.py:221`) and `ProvenanceMetadata`
(`preprocessing/config.py:120`) carry:

- `tier_counts` (per-file) and `aggregate_tier_counts` (summed over the run)
- `removed_line_indices` — zero-based indices of dropped lines, useful for
  post-hoc audit against the original file
- `replacement_pool_size`, `replacement_lines_drawn`,
  `replacement_pool_remainder` per file
- `total_pool_lines_available`, `total_pool_lines_drawn`,
  `total_pool_lines_remaining` aggregated across the run

### Replacement-pool remainder

`_rebuild_to_target_size` (`preprocessing/base.py:433`) returns pool stats
and writes any unused pool sentences to
`<output_path>/replacement_pool_remainder/<file_stem>.txt`. The remainder
is deterministic given the `seed` (auto-injected from the experiment-level
`random_seed` by `model_foundry/cli.py:135` if absent).

### Document Boundary Handling

Lines matching the regex `^= = =.+= = =$` are treated as document
boundaries by the English detector
(`remove_expletive_sentences.py:45`). They:

- Pass through unchanged (never removed).
- Reset the coref context buffer so coreference resolution does not cross
  document boundaries.

## Coreference Confirmation

### Why

Simple dependency-based detection can over-remove referential *it*:

```text
"The report was late. It arrived yesterday."
# dep_=='expl' is False, but a naive heuristic on "it" could still fire.
```

The three-tier detector only applies coref to **tier 2** candidates. Tier 1
(spaCy `dep_ == 'expl'`) is always removed regardless of coref state.

### Enabling

Set `coref_model` in the step parameters (see example config
`configs/experiments/experiment_en_remove_expletive_sentences.yaml`):

```yaml
dataset_manipulation:
  - type: remove_expletive_sentences_en
    input_path: "data/raw/train_90M/"
    output_path: "data/processed/exp_remove_expletive_sentences_en/"
    spacy_model: "en_core_web_trf"
    parameters:
      replacement_pool_dir: "data/raw/pull_10M/"
      coref_model: "en_coreference_web_trf"
```

The model is lazy-loaded on the first heuristic candidate
(`remove_expletive_sentences.py:224`). If it cannot be loaded the detector
falls back to trusting the heuristic.

### Context Window

The detector keeps the last `context_lines` (default 3) prior lines in a
buffer. On each tier-2 candidate it concatenates
`<context_prefix> <current_line>`, runs coreference, and checks whether
the candidate *it* character span falls inside any multi-mention cluster.
If so the line is kept (`tier3_coref_kept`), otherwise it is removed
(`tier3_coref_confirmed`). The buffer resets at document boundaries.

### Performance

Coreference roughly halves throughput on CPU and is significantly slower
on GPU than the base `en_core_web_trf` pipeline. Use it when precision
matters more than throughput; skip it for multi-hundred-million-token runs
where the tier-2 heuristics are accurate enough.

## Production Deployment

### Error Handling

Files that raise are logged into `manifest.metadata.failed_files` but do
not abort the corpus run (`preprocessing/base.py:199`).

```python
manifest = AblationPipeline(config).process_corpus()

if manifest.metadata.failed_files:
    with open("failed_files.log", "w") as f:
        for path, error_msg in manifest.metadata.failed_files:
            f.write(f"{path}: {error_msg}\n")
```

### Monitoring

```python
config = AblationConfig(
    ...,
    verbose=True,
    log_dir="logs/preprocessing/",
)
# Logs: logs/preprocessing.<ablation_type>/preprocessing_<timestamp>.log
```

### Validation Strategy

- **Development**: `skip_validation=False` on a small sample to sanity-check
  the ablation.
- **Production**: `skip_validation=True` to save time once the detector is
  known-good.

## Cluster Processing

Split the corpus by file, run each shard as a separate pipeline, then
aggregate manifests:

```python
from pathlib import Path
import json

manifests = []
for part in Path("data/processed/").glob("part_*/ABLATION_MANIFEST.json"):
    manifests.append(json.loads(part.read_text()))

total_items = sum(m["metadata"]["total_items_ablated"] for m in manifests)
total_tiers: dict[str, int] = {}
for m in manifests:
    for k, v in m["metadata"].get("aggregate_tier_counts", {}).items():
        total_tiers[k] = total_tiers.get(k, 0) + v

print(f"Items removed: {total_items:,}")
print(f"Tier totals:   {total_tiers}")
```

## Troubleshooting

### Memory Issues

```python
spacy_batch_size=10
chunk_size=500
spacy_disable_components=["ner", "textcat"]
```

### Slow Processing

```python
spacy_batch_size=100
spacy_disable_components=["ner", "textcat"]
skip_validation=True
```

### Validation Failures

Validation uses a stateless detector
(`_has_expletive_en_enhanced`, `remove_expletive_sentences.py:325`) without
coref. On files with heavy coref-driven keeps, the stateless validator may
warn; this is not fatal and processing continues.

## Best Practices

1. Start from the shipped experiment YAML
   (`configs/experiments/experiment_en_remove_expletive_sentences.yaml`).
2. Set `random_seed` at the experiment level — preprocessing inherits it.
3. Keep `spacy_model="en_core_web_trf"` for the English detector.
4. Inspect `aggregate_tier_counts` after each run to sanity-check the mix
   of tier-1 / tier-2 / tier-3 removals.
5. Preserve `ABLATION_MANIFEST.json` alongside the processed corpus — it is
   the only source of truth for reproducibility.

## Next Steps

- **Custom ablations**: [Developer Guide](DEVELOPER_GUIDE.md)
- **Change log**: [CHANGES.md](CHANGES.md)
