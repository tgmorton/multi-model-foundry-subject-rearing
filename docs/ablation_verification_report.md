# Ablation verification report — TEMPLATE

> Status: scaffold for joint review with [partner name]. Inspection not yet
> complete. Final document goes to PI.

This report documents how each preregistered ablation transformation was
verified to be correct on the controlled-rearing English + Spanish 90M
corpora, prior to launching the production training matrix.

---

## 1. Overview

| Ablation | Lang | Type | Status |
|---|---|---|---|
| `remove_expletive_sentences_en` | EN | Line-removal | ✅ ready for inspection |
| `impoverish_case_en` | EN | Token-level substitution | ✅ ready for inspection |
| `lemmatize_verbs` (EN application) | EN | Token-level substitution | ⚠️ **bug fix pending re-run** (see §6) |
| `enrich_verbal_morphology` (EN application) | EN | Token-level substitution | ⚠️ **bug fix pending re-run** (see §6) |
| `remove_expletive_sentences_es` | ES | Line-removal | ✅ ready for inspection |
| `impoverish_case_es` | ES | Token-level substitution | ✅ ready for inspection |
| `lemmatize_verbs` (ES application) | ES | Token-level substitution | ✅ ready for inspection |
| `insert_pronouns_es` | ES | Token-level insertion | ⏳ pipeline build pending (§7) |

Two transformation classes:

- **Line-removal**: ablation drops sentences that match a structural test;
  the deficit is backfilled from the ablated `pull_10M` so per-genre token
  totals match the baseline corpus.
- **Token-level substitution**: ablation rewrites individual tokens
  in-place; line counts and sentence boundaries are preserved.

---

## 2. Verification methodology

For each ablation we use four converging lines of evidence:

1. **Rule documentation** — the formal Python implementation, with
   per-category trigger predicates and unit-test fixtures.
2. **Random-sample human inspection** — N=150 changed lines per genre
   per ablation (900 lines total per ablation across 6 genres),
   judged by the inspector(s) as `correct` / `incorrect` / `borderline`,
   with notes for borderline cases. Seed 42, deterministic.
3. **Aggregate statistics** — per-tier counts (e.g., for
   `impoverish_case`: tonic_oblique vs portmanteau vs acc_clitic vs ...)
   from the `ABLATION_MANIFEST.json` written alongside each output.
   These should match expectations within an order of magnitude.
4. **Compose-step audit** — `COMPOSE_MANIFEST.json` records pool draws,
   pool exhaustion, and shortfalls. Any `pool_exhausted=True` is
   accepted only when explicitly justified (e.g., ES `spoken` /
   CORLEC, EN `bnc_spoken` for `remove_expletive_sentences`).

The full evidence pack per ablation lives in:

- `data/validation_samples/2026-04-24/<lang>/<slug>.tsv` —
  `validate_ablation.py --mode three-step` output (genre, source,
  line_num, original, ablated). Source values: `train-kept`,
  `train-removed`, `pool-backfill`.
- `data/validation_samples/2026-04-24/<lang>/before_after_<slug>_<genre>.tsv`
  — substitution-aware sampling: lines from raw vs composed corpus
  where they differ, 40 examples per (slug, genre).
- `data/validation_samples/2026-04-24/large_random_samples/<lang>_<slug>.jsonl`
  — 900-row random sample, 150 per genre × 6 genres. **Use this for
  inspection of correctness rate.**

---

## 3. Per-ablation evidence packs

> One subsection per ablation. For each, paste examples from the
> 900-row JSONL, plus tier counts from the ABLATION_MANIFEST.json,
> plus inspector verdict. Below is the template. Fill in during the
> joint review.

### 3.X `<ablation_name>` — `<lang>`

**Rule.** [1-2 sentences. e.g.: "Remove sentences whose root verb is
weather (`llover`, `nevar`, …) or existential `haber`, or whose
clausal complement is licensed by an impersonal raising verb."]

**Implementation.** [`preprocessing/ablations/<file>.py:Ldd-Ldd`]

**Trigger categories.** [List the categories with one example each.]

**Aggregate counts (from ABLATION_MANIFEST.json).**

| Category | Train items | Pool items |
|---|---|---|
| ... | ... | ... |

**Random-sample inspection results (N=900 across 6 genres).**

| Verdict | Count | Rate |
|---|---|---|
| Correct | ... | ...% |
| Incorrect | ... | ...% |
| Borderline | ... | ...% |

**Notable cases.** [Examples of incorrect / borderline rows with
inspector commentary. Aim for 5-10.]

**Inspector sign-off.** [Initials, date.]

---

## 4. How this is reported in the paper

Standard for rule-based corpus-manipulation studies:

### 4.1 Methods section

A short paragraph per ablation in the experimental setup. Each paragraph
covers:

- **Goal**: what linguistic feature the ablation removes/inserts (e.g.,
  "remove expletive subjects to test their contribution to subject-drop
  rate").
- **Rule**: a one-sentence formal description naming the parser, the
  spaCy model + version, and the structural test (e.g., "Sentences with
  a root verb whose lemma is in a closed class of weather verbs, or
  whose subject is the expletive `it` / `there` (`PronType=Prs`,
  `Subj` dependency, no antecedent referent), are removed").
- **Sample size for verification**: "N=900 randomly-sampled changed
  lines per ablation (150 per genre across 6 genres) were manually
  annotated by the authors as correct / incorrect / borderline; the
  observed correctness rate was X% (Y% incorrect, Z% borderline) with
  Cohen's κ = κ between annotators."
- **Backfill**: "Token deficits from line-removal ablations were
  backfilled from a held-out 10M-word `pull_10M` split that received
  the same ablation transformation, sampled without replacement at the
  line level. For ablations and genres where the pool was exhausted
  (specifically ES `spoken` / CORLEC and EN `bnc_spoken` for
  `remove_expletive_sentences`), we accepted under-target output and
  recorded the deficit in the per-file manifest."

### 4.2 Pre-registration deviations

If anything in the ablation differs from the OSF preregistration —
list it, justify it, ideally cite the commit or an attached changelog.

### 4.3 Reproducibility appendix / supplementary material

Deposit on OSF (or Zenodo with DOI) at the time of submission:

1. **Code** — full `preprocessing/ablations/` directory, frozen at
   the commit hash used for production. Cite the hash + GitHub URL.
2. **Stimuli** — the 900-row random sample TSVs above, plus the
   compose manifests, so the reviewer can spot-check the same
   examples we did.
3. **Inter-rater reliability data** — both annotators' raw
   judgments on the same 900 rows so anyone can recompute κ.
4. **Regression test suite** — the unit tests under
   `preprocessing/tests/` that cover each ablation's trigger
   categories. Cite as evidence of code-level correctness.
5. **Final corpora** — the `manipulations/{lang}/<slug>/*.train`
   files. ~90M tokens × 8 conditions × 2 langs ≈ 1.4B-token archive.
   Compress via zstd; should fit in a single Zenodo deposit.

### 4.4 Statistical reporting of validation

The minimum reportable claim for each ablation is:

> *N=900 randomly-sampled changed lines were inspected by two
> annotators independently. Inter-annotator agreement was κ = X.YZ
> (interpretation per Landis & Koch 1977). The observed correctness
> rate, with annotator disagreements adjudicated by [method], was P%
> [95% CI: A%–B%]. The Z% of borderline cases are categorized in
> Appendix [k] and consist primarily of [pattern].*

CI from a Wilson-score interval on the binomial. We expect P > 95% to
be defensible for production use; below that, iterate on the rule.

### 4.5 Acknowledged limitations

Explicitly state in the paper:

- **Parser dependency**: ablation correctness inherits from the spaCy
  model's parse accuracy. Mistakes the parser makes propagate as
  mistakes the ablation makes. Cite the model's published accuracy
  numbers (e.g., LAS / UAS on UD test set).
- **Genre coverage**: any genres where the ablation may behave
  pathologically (e.g., highly fragmented spoken transcripts in
  CHILDES / BNC where parser confidence is lower). Quantify if
  possible.
- **Coverage of the linguistic phenomenon**: the rule is
  necessarily an operationalization. We cite the prereg, list the
  categories actually targeted, and explicitly note whether the rule
  is a *strict subset*, *strict superset*, or *intended exact match*
  of the theoretical category.

---

## 5. Sign-off criteria

The corpus is approved for production training when:

- [ ] All 8 ablations pass §3 random-sample inspection at ≥95%
      correctness (after disagreement adjudication).
- [ ] Inter-annotator κ ≥ 0.80 across all ablations.
- [ ] Compose manifests show no surprises beyond the documented
      pool-exhaustion cases.
- [ ] Tokenizer re-training has been queued (depends on §6 fix).
- [ ] The verification artifact (this document, the 900-row JSONLs,
      annotator judgments) has been committed to the repo and is
      ready for OSF deposit.

---

## 6. Open issue: contraction-glue bug in EN substitution ablations

Discovered 2026-04-24 during overnight subagent review of the first
EN ablation pass. English contractions tokenize as glued pieces in
spaCy (`it's` = `[it (ws=""), 's]`); the substitution ablations
(`lemmatize_verbs`, `enrich_verbal_morphology`, and defensively
`impoverish_case`) concatenated replacements without re-inserting
whitespace, producing pseudo-tokens like `itbe`, `webe`, `ben't`,
`doont`, `whatbe`.

Fix committed at `084c515` —
`preprocessing/ablations/{lemmatize_verbs,enrich_verbal_morphology,impoverish_case}.py`
re-inject whitespace whenever a replaced token was glued to a neighbour
in the original.

Re-run is queued (`thomas-ablate-compose-en-v1` in the K8s manifest)
but blocked on the NRP admission webhook quota; will overwrite the
buggy outputs once admitted.

Two of the four EN tokenizers were trained AFTER the buggy ablations
finished and have the pseudo-tokens (`itbe`, `webe`, `thatbe`,
`whatbe`, `doant`) baked into vocabulary as atomic units:

| Tokenizer | Trained | Status |
|---|---|---|
| `en_gpt2_medium` | Apr 23 19:35 | ✅ Clean (predates buggy ablations) |
| `en_bert_large` | Apr 24 00:11 | ⚠️ Mostly clean (some sub-word weirdness) |
| `en_bert_wordpiece` | Apr 24 20:10 | 🔴 Contaminated — re-train required |
| `en_shared_unigram` | Apr 24 20:35 | 🔴 Contaminated — re-train required |

All ES tokenizers verified clean (Spanish doesn't have English-style
contractions; sanity-tested with synthetic glue inputs).

**Plan**: re-train all four canonical post-refactor tokenizers
(`{en,es}_{bert_wordpiece,shared_unigram}`) on the clean re-ablated
text once §6 re-run completes. Uniform tokenizer-training source
across all conditions.

---

## 7. Open issue: `insert_pronouns_es`

The Spanish "insert pronouns" ablation (the inverse of subject-drop:
restore overt subject pronouns inferred from English-Spanish parallel
data) is a separate analysis pipeline, not yet integrated into the
preprocessing/ablations registry. Tracked as a workstream in the
[Spanish-swap plan](../OSF_PREREGISTRATION.md). This report excludes
it; a follow-up section will be added when the pipeline lands.

---

## 8. Reproducibility checklist

- [ ] Commit hashes for: ablation code, sweep configs, training configs,
      tokenizer training script.
- [ ] spaCy model versions (currently: `en_core_web_trf-3.7.3`,
      `es_core_news_lg-3.7.0`).
- [ ] Random seeds for: validation sampling (42), pool sampling per genre
      (master seed 42 XOR per-stem hash).
- [ ] Final SHA-256 of each `manipulations/<lang>/<slug>/*.train` after
      re-runs settle.
- [ ] OSF / Zenodo DOI for the deposit.

---

*Generated 2026-04-24. To be filled in during joint inspection pass.*
