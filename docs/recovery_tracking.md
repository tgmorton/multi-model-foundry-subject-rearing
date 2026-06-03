# English Wave Recovery Tracker

_Definitive partition from S3 registry + PVC `training_state.pt` audit (2026-06-03). Authoritative completion signal = registry `status==COMPLETE`; tick boxes as cells finish._

## Summary — 300 runs (240 ablations + 60 baselines)

| Cohort | Count | Launch mode |
|---|---:|---|
| A. Resume (truncated, has training_state) | 207 | `--resume` from latest ckpt; back-half-only schedule |
| B. Fresh ablations (30 never-started + 3 unrecoverable) | 33 | fresh, full schedule; `--job-suffix fresh` on mixed cells |
| C. Baselines (none exist) | 60 | `--include-baseline`, fresh, full schedule |

## Validation gate (must be green before mass dispatch)
- [x] **Resume path** — `lstm-en-enrich_verbal_morphology-h0-s42`: COMPLETE, endpoint 67410 w/ training_state, 17 gap-free back-half ckpts, loss 2.83, survived 2 restarts
- [ ] **Fresh path** — `lstm-en-lemmatize_verbs-h0-s999` (throwaway): in progress, validate full schedule + early head + endpoint at completion

## Cohort A — Resume (207)  ·  `--resume`
Per cell: `python scripts/launch_production_training.py --lang en --arch <A> --intervention <I> --resume --job-suffix resume --slots-file <recoverable slots>`

### gpt2_small (40)  · tok=unigram
- **remove_expletive_sentences** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **impoverish_case** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **lemmatize_verbs** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **enrich_verbal_morphology** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137

### gpt2_medium (40)  · tok=unigram
- **remove_expletive_sentences** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **impoverish_case** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **lemmatize_verbs** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **enrich_verbal_morphology** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137

### gpt2_large (29)  · tok=unigram
- **remove_expletive_sentences** (6): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137
- **impoverish_case** (7): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42
- **lemmatize_verbs** (7): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42
- **enrich_verbal_morphology** (9): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42

### bert_large (32)  · tok=wordpiece
- **remove_expletive_sentences** (8): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137
- **impoverish_case** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **lemmatize_verbs** (7): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42
- **enrich_verbal_morphology** (7): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42

### lstm (40)  · tok=unigram
- **remove_expletive_sentences** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **impoverish_case** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **lemmatize_verbs** (10): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- **enrich_verbal_morphology** (10): [x] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137

### mamba_370m (26)  · tok=unigram
- **remove_expletive_sentences** (6): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137
- **impoverish_case** (6): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137
- **lemmatize_verbs** (6): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137
- **enrich_verbal_morphology** (8): [ ] h0s42 [ ] h0s137 [ ] h1s42 [ ] h1s137 [ ] h2s42 [ ] h2s137 [ ] h3s42 [ ] h3s137

## Cohort B — Fresh ablations (33)  ·  no `--resume`, `--job-suffix fresh`

### B1. Never-started (30)

**gpt2_large**
- remove_expletive_sentences: [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- impoverish_case: [ ] h3s137 [ ] h4s42 [ ] h4s137
- lemmatize_verbs: [ ] h3s137 [ ] h4s42 [ ] h4s137
- enrich_verbal_morphology: [ ] h4s137

**bert_large**
- remove_expletive_sentences: [ ] h4s137
- lemmatize_verbs: [ ] h3s137 [ ] h4s42 [ ] h4s137
- enrich_verbal_morphology: [ ] h3s137 [ ] h4s42 [ ] h4s137

**mamba_370m**
- remove_expletive_sentences: [ ] h3s137 [ ] h4s42 [ ] h4s137
- impoverish_case: [ ] h3s42 [ ] h3s137 [ ] h4s42 [ ] h4s137
- lemmatize_verbs: [ ] h3s137 [ ] h4s42 [ ] h4s137
- enrich_verbal_morphology: [ ] h4s42 [ ] h4s137

### B2. Unrecoverable — crashed early, no training_state (3)
- [ ] `bert_large-en-remove_expletive_sentences-h4-s42`  (delete old dir, launch fresh)
- [ ] `mamba_370m-en-lemmatize_verbs-h3-s42`  (delete old dir, launch fresh)
- [ ] `mamba_370m-en-remove_expletive_sentences-h3-s42`  (delete old dir, launch fresh)

## Cohort C — Baselines (60)  ·  `--include-baseline`
Corpus `data/raw/en/train_90M/` + caches verified present (unigram key `1fe27c37d51f`, wordpiece `f0a8eb507472`).

**gpt2_small** (unigram): [ ] baseline-h0s42 [ ] baseline-h0s137 [ ] baseline-h1s42 [ ] baseline-h1s137 [ ] baseline-h2s42 [ ] baseline-h2s137 [ ] baseline-h3s42 [ ] baseline-h3s137 [ ] baseline-h4s42 [ ] baseline-h4s137

**gpt2_medium** (unigram): [ ] baseline-h0s42 [ ] baseline-h0s137 [ ] baseline-h1s42 [ ] baseline-h1s137 [ ] baseline-h2s42 [ ] baseline-h2s137 [ ] baseline-h3s42 [ ] baseline-h3s137 [ ] baseline-h4s42 [ ] baseline-h4s137

**gpt2_large** (unigram): [ ] baseline-h0s42 [ ] baseline-h0s137 [ ] baseline-h1s42 [ ] baseline-h1s137 [ ] baseline-h2s42 [ ] baseline-h2s137 [ ] baseline-h3s42 [ ] baseline-h3s137 [ ] baseline-h4s42 [ ] baseline-h4s137

**bert_large** (wordpiece): [ ] baseline-h0s42 [ ] baseline-h0s137 [ ] baseline-h1s42 [ ] baseline-h1s137 [ ] baseline-h2s42 [ ] baseline-h2s137 [ ] baseline-h3s42 [ ] baseline-h3s137 [ ] baseline-h4s42 [ ] baseline-h4s137

**lstm** (unigram): [ ] baseline-h0s42 [ ] baseline-h0s137 [ ] baseline-h1s42 [ ] baseline-h1s137 [ ] baseline-h2s42 [ ] baseline-h2s137 [ ] baseline-h3s42 [ ] baseline-h3s137 [ ] baseline-h4s42 [ ] baseline-h4s137

**mamba_370m** (unigram): [ ] baseline-h0s42 [ ] baseline-h0s137 [ ] baseline-h1s42 [ ] baseline-h1s137 [ ] baseline-h2s42 [ ] baseline-h2s137 [ ] baseline-h3s42 [ ] baseline-h3s137 [ ] baseline-h4s42 [ ] baseline-h4s137

## Cleanup
- [ ] Delete throwaway `lstm-en-lemmatize_verbs-h0-s999` (dir + registry record + wandb) after fresh-path validation
