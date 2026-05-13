# Spanish Evaluation Set — Experimental Notebook

Running log of design decisions for the parallel Spanish null-subject
evaluation stimuli. Entries are dated; open questions tracked separately
from pinned commitments.

Decisions here override earlier informal discussion but do not override
the OSF preregistration (`docs/OSF_PREREGISTRATION.md`). If a decision
here diverges from the prereg, an amendment is required — flagged
explicitly in the entry.

---

## 1. Goal & scope

Build a Spanish stimulus set that is **strictly parallel** to the
existing English null-subject evaluation stimuli
(`evaluation/stimuli/null-subj/`), enabling:

1. **Monolingual Spanish evaluation** of Spanish-trained BabeLM models
   (mirrors the English/Italian evaluation already in prereg).
2. **Cross-linguistic structural priming experiments** — constructing
   congruent/incongruent prime-target pairs across languages by drawing
   cross-pairings (null→null, overt→null, etc.) between matched
   English and Spanish items.

The priming use case is the *reason* for strict parallelism: item-level
correspondence across languages is what enables the cross-pairings to
be interpretable.

---

## 2. Pinned commitments

These are locked-in decisions. Move items here from §3 only when
you're confident they won't change.

- **Dialect**: Peninsular Spanish (aligns with `annotation/constants.py`
  which maps `vosotros` and `ustedes` to separate labels).
- **Schema**: Items joinable across languages on a shared `(item_group,
  item)` key. English and Spanish live in parallel files with matching
  item IDs.
- **No transformations**: The 6-way transformation expansion from the
  current English CSV (default / complex_long / complex_emb /
  target_negation / context_negation / both_negation) will not be
  ported. Those were artifacts of an earlier study.
- **Sanity-check controls**: Not embedded in the null-subject set.
  BLiMP runs separately on the same models and serves that role at
  the whole-evaluation level.

---

## 3. Decisions log (reverse-chronological)

### 2026-05-12 — Methods-section snippet: ablation operationalization

Consolidates this week's ablation-design decisions into a quotable
prose block ready for the paper's Methods section. Decisions covered:
contraction-glue fix, past-tense paradigm for `enrich_verbal_morphology`,
suppletive paradigms for `be/have/do/go`, coarse 3sg fallback for
unresolvable-subject finite verbs, switch from spaCy `es_core_news_lg`
to `simplemma` for Spanish lemmatization, and (pending decision) hyphen
at synthetic-morpheme boundaries to disambiguate suffixes from English
function words.

Verification: N=250 randomly-sampled changed lines per ablation across
six genres per language, judged by a fluent speaker of the language as
correct/incorrect/borderline. Wilson 95% CI on correctness rate; single
annotator per language (one fluent Spanish speaker available); full
900-row reservoirs deposited for reviewer audit.

#### Snippet — drop-in Methods section prose

> **Ablation operationalization.** We applied four token- and sentence-
> level rule-based ablations to the English BabyLM (Charpentier et al.
> 2023) and Spanish BebeLM (this paper) 90M-word corpora. Each ablation
> targets a single grammatical feature relevant to subject-drop while
> preserving sentence boundaries and token counts:
>
> 1. **`remove_expletive_sentences`** (EN, ES): deletes finite sentences
>    whose root verb is licensed by an expletive subject (English `it`
>    weather/cleft/raising and `there` existential; Spanish `haber`
>    existential, weather and impersonal raising verbs, literary `ello`).
>    Removed lines are backfilled from an ablated 10M-word held-out
>    pool to preserve per-genre token totals. For genres whose pool is
>    exhausted by the ablation (English `bnc_spoken`, Spanish CORLEC
>    `spoken`), we accept under-target output and record the deficit in
>    the per-file compose manifest.
>
> 2. **`impoverish_case`** (EN, ES): replaces every non-nominative pronoun
>    or possessive with its nominative form (English: `him → he`,
>    `their → they`, `my → I`; Spanish: `lo → él`, `te → tú`, `mi → yo`).
>    Definite articles (Spanish `la/los/las`, English `the`) are excluded
>    by part-of-speech filtering (`PronType=Art`).
>
> 3. **`lemmatize_verbs`** (EN, ES): reduces every finite verb and
>    auxiliary to its base form. Spanish lemmas are computed using
>    `simplemma 1.x` (Barbaresi 2024) rather than spaCy's built-in
>    lemmatizer, after an audit of 75 stem-changing irregular forms found
>    that spaCy's `es_core_news_lg` lemmatizer hallucinates non-existent
>    verb stems (e.g., `harías → *hariar`, `tendrías → *tendriar`) on
>    ~14% of the audit set, whereas `simplemma`'s errors are
>    higher-recall pass-throughs of surface forms (e.g., `dije → dije`)
>    rather than hallucinated stems. Passing surface forms preserves
>    cleanness of the manipulation; hallucinated stems do not. English
>    lemmas are taken from spaCy's `en_core_web_trf` lemmatizer.
>
> 4. **`enrich_verbal_morphology`** (EN only): adds a synthetic
>    Indo-European-style person × number × tense suffix paradigm to
>    every finite verb. The present-tense paradigm uses
>    `-o / -aks / -akt / -amus / -atis / -ant` (1sg/2sg/3sg/1pl/2pl/3pl);
>    the past-tense paradigm uses `-i / -isti / -ikt / -imus / -istis /
>    -erunt`. The two paradigms are disjoint to preserve surface
>    recoverability of tense. Three forms diverge from a strict Latin
>    paradigm — 2sg-pres `-aks` (vs. Latin `-as`), 3sg-pres `-akt`
>    (vs. `-at`), 3sg-past `-ikt` (vs. `-it`) — because the original
>    Latin forms are identical to high-frequency English function words
>    (`as`, `at`, `it`) and a tokenizer is unlikely to disambiguate
>    them from word-final occurrences in the unmodified corpus. The
>    `-Vkt` / `-Vks` cluster is unambiguously bound morphology to a
>    reader and tokenizer alike. Past-tense English stems are
>    lemmatized before the suffix is appended (`ran → runikt`). To
>    mirror Romance languages' suppletive marking of the highest-
>    frequency verbs (Spanish `soy/eres/es` rather than the regular
>    `*sero/seres/sere`), we apply hand-crafted suppletive paradigms
>    to the four highest-frequency English verbs (`be`, `have`, `do`,
>    `go`) using Latin esse / habere / facere / vadere stems (e.g.,
>    `is → est`, `was → fuikt`, `goes → vadakt`, `did → fecikt`).
>    The same `-as/-at/-it/-is → -aks/-akt/-ikt/-iks` substitution
>    applied to the regular paradigm is also applied to suppletive
>    forms whose word-final ending would otherwise reproduce the same
>    English homograph (`habat → habakt`, `facit → facikt`, etc.). When the dependency parse fails to
>    resolve a subject for a finite verb (inverted dialogue tags
>    `said Lucas`, proper-noun subjects with sparse morphological features,
>    fragments), we default to a third-person-singular suffix rather than
>    falling back to the bare lemma, which would be indistinguishable
>    from `lemmatize_verbs` output and would silently leak signal between
>    the two manipulations. Across the 90M-token English corpus, this
>    paradigm fires on 11.0% of tokens (9.76M enriched verbs).
>
> 5. **`insert_pronouns_es`** (ES only, pending): inserts overt subject
>    pronouns into Spanish finite clauses where the parser identifies a
>    licit null subject. The pronoun chosen is inferred from English-
>    Spanish parallel-corpus alignments via a separate detector
>    described in [pipeline section].
>
> A pre-tokenization fix was required for English contraction handling:
> spaCy tokenizes contractions with empty inter-token whitespace
> (`it's = [it (ws=""), 's]`), and a first version of our substitution
> rules concatenated the replacement directly to the previous token,
> producing pseudo-tokens like `itbe`, `webe`, `ben't`. The bug was
> caught during random-sample inspection and fixed by re-injecting
> whitespace whenever a replaced token was glued to a neighbour in the
> source tokenization.
>
> **Verification.** For each of the eight ablation conditions, we drew a
> stratified random sample of 250 changed lines per language (roughly
> 42 per genre, across six genres per language; deterministic seed). For
> line-removal ablations, the sample includes train-kept,
> train-removed, and pool-backfill rows so the annotator can evaluate
> both true-positive removal and replacement plausibility. For
> substitution ablations, the sample is drawn from lines where the
> original and ablated text differ. A fluent speaker of the language
> annotated each row as *correct* (the change matches the intended
> rule), *incorrect* (the rule fired wrongly, missed when it should have
> fired, or broke the sentence), or *borderline* (ambiguous case
> or unparsable source). Inter-rater reliability was not computed: only
> one fluent Spanish speaker was available, so single-annotator
> evaluation was applied uniformly across both languages to avoid
> asymmetric methodology. Correctness rate per ablation is reported with
> a 95% Wilson-score confidence interval. The full 900-row reservoirs
> and per-row annotator judgments are deposited at \[OSF DOI / repo
> commit hash\] for reviewer audit.
>
> **Acknowledged limitations.** (a) Ablation correctness inherits from
> the spaCy dependency parse and morphological annotation; parser
> mistakes propagate as ablation mistakes. We use the largest available
> spaCy model per language (`en_core_web_trf-3.7.3` for English,
> `es_core_news_lg-3.7.0` for Spanish — no `_trf` Spanish model exists).
> (b) Dialogue-tag inversion in narrative prose (Project Gutenberg) is
> a known site of parser failure on subject linkage. We mitigate via
> the 3sg-fallback heuristic above; observed-rate impact is documented
> in Appendix \[k\]. (c) Highly fragmented spoken-corpus transcripts
> (CHILDES, BNC Spoken, Switchboard) sometimes lack the clausal
> structure assumed by rule-based ablations; these are flagged as
> *borderline* by the annotator and excluded from correctness-rate
> denominators.

### 2026-04-16 — Spanish translation guide drafted

Drafted `docs/eval_stimuli/spanish.md` as a Spanish-facing agent runbook
(translation + Spanish-only generation). Decisions baked into the guide:

- **Core categories**: generate item-paired Spanish counterparts to all 8
  English categories. Where the EN↔ES pairing is awkward or structurally
  impossible, document in metadata rather than silently diverge.
- **Grammaticality direction flips** explicitly tabulated per condition
  (§4 of spanish.md). Key flips: `subject_drop` 3rd-person
  (EN→overt, ES→null), `embedded_drop` coref (flips), `extraction` subj
  (that-trace asymmetry — the theoretical payoff).
- **Expletive**: marked `ello` approach for the overt form, flagged for
  vetter review.
- **subject_drop_no_agreement (ES)**: artificial infinitive ablation
  across all 6 conditions; context stays finite, only target is ablated.
  Acknowledged as ungrammatical by design — tests residual pronoun
  preference independent of agreement morphology.
- **Spanish-only categories added**: `postverbal_subject` (2 conditions),
  `se_impersonal` (2 conditions), `clitic_climbing` (1 condition).
  These are monolingual-only (not cross-paired for priming).
- **Dialect**: Peninsular. No voseo, no leísmo/loísmo/laísmo, no
  dequeísmo. Validation check V18 enforces.
- **Accents**: preserved as Unicode (`á é í ó ú ñ ü`), required in
  names and words where standard. Validation V19 enforces.

### 2026-04-16 — Evaluation infrastructure audit & metric plan

Audited current eval code to map it against the prereg's metric commitments
plus the minimal-pairs-methodology lit review recommendations.

**Current state:**
- `evaluation/evaluators/null_subject_evaluator.py` computes per-item mean
  and total surprisal (bits), hotspot surprisal, binary preference,
  surprisal difference. Schema is v1 only.
- `evaluation/core/surprisal_calculator.py` has a clean surprisal primitive.
- FP16 supported; batching hooks exist but are unused (purely sequential).
- No SLOR computation; no unigram baseline model; no end-to-end smoke
  test with a real model.
- Output is JSONL per checkpoint, aggregates to summary JSON.

**Metric decisions (confirmed):**
- **Mean log-probability per sentence**: compute from existing surprisal
  (`MeanLP = -mean_surprisal * ln(2)`). Trivial adapter.
- **Per pair**: binary preference (exists), mean log-prob for each
  sentence (new adapter), distance = difference (exists).
- **SLOR**: include. Requires unigram baseline trained on BabyLM/BebeLM
  training corpora (one per language). Cache serialized; load at eval
  time; near-zero runtime cost after precomputation.
- **MORCELA**: flagged as a follow-up. Requires an acceptability-rating
  training set; not available yet for ES (and EN's availability is
  unclear). Revisit after first wave of evaluations — the MORCELA
  parameters can be learned from held-out stimuli once we have any
  evaluation data in hand.

**Order of operations before launching evaluations:**
1. Add v2 schema support (backwards-compatible column remap; use
   `hotspot_position` directly rather than string-search).
2. Add `mean_log_prob` fields alongside surprisal in per-item output.
3. End-to-end smoke test on a tiny model against a v2 CSV (one EN + one
   ES) to catch schema/tokenization issues early.
4. Train unigram baselines on `data/english/train_90M/` and
   `data/spanish/train_90M/` (subword-level, matching model tokenizers).
   Serialize as `.pkl` or equivalent.
5. Add SLOR output using the baselines.
6. **Batching refactor** (separate pass, bigger change): tokenize + pad
   + batch forward passes; reuse context tokenization across overt/null
   of a pair. Projected 3–5× GPU speedup. Do this once metrics are
   stable so we're not debugging batching + metrics simultaneously.
7. MORCELA: follow-up.

**Optimization candidates, ranked by (impact × effort):**
- H/L: **batch forward passes across pairs** — biggest win; currently
  purely sequential. **Verified isolation**: standard batching stacks
  independent sequences with padding + attention mask; no cross-item
  attention. Smoke test will include a batched-vs-sequential assertion
  to permanently lock down this invariant.
- H/L: **reuse context tokenization across overt/null of a pair** —
  half the per-pair compute. Still item-isolated (same pair shares
  context by design).
- M/L: **pre-tokenize the stimulus set once per run** rather than
  per-call.
- M/L: **cache unigram baseline** — once trained, don't recompute.

**SLOR unigram-baseline policy: paired per corpus (FIT-CLAMS style).**

Every trained model uses the unigram baseline computed on *its own
training corpus* — not a global baseline. Rationale:

- The prereg's H2 (evidence manipulation) compares models trained on
  different corpora. SLOR with a shared baseline would confound
  frequency-distribution differences with grammatical-knowledge
  differences. Per-corpus baselines cancel the frequency term on both
  sides of the log and leave the structural contribution.
- Matches Padovani et al. 2025 (FIT-CLAMS) methodology.
- Free operationally: the existing n-gram-per-dataset training pipeline
  produces each corpus's unigram as a byproduct; we just serialize it.

**Implementation notes:**
- Unigram vocab must match the model tokenizer's subword vocabulary
  (not whitespace tokens).
- Smoothing: Laplace (+1). Simpler than KN, well-defined behavior for
  zero-count tokens (e.g., pronouns in the pronoun-removed corpus).
- Cache per-corpus; load once at eval time.
- Report both raw `mean_log_prob` and `slor` per sentence so that
  unexpected SLOR results can be diagnosed against the raw numbers.

### 2026-04-16 — Spanish staging set built (11 agents, all 11 categories)

Dispatched 11 opus-4.7 agents in parallel against `docs/eval_stimuli/spanish.md`
v1.0. All 11 completed cleanly. Staging at
`evaluation/stimuli/null-subj-v2/staging/es/`.

**Final counts: 1,488 rows / 744 paired items across 29 conditions.**

| File | Rows | Pairs | Conds | Notes |
|------|------|-------|-------|-------|
| subject_drop | 288 | 144 | 6 | Item-paired with EN; demonstratives used for inanimate 3sg/3pl overt |
| subject_drop_no_agreement | 288 | 144 | 6 | Infinitive ablation; 1 cross-cond V14 collision |
| object_drop | 96 | 48 | 2 | Preverbal clitic structure; 2 cross-cond V14 collisions |
| embedded_drop | 96 | 48 | 2 | Passes all checks; null preferred in coref (PAH) |
| control | 96 | 48 | 2 | 24+18 unique matrix verbs; prep subcat preserved |
| expletive | 96 | 48 | 2 | Marked `ello`; agent flagged some `ello+se+V` items |
| conjunction | 96 | 48 | 2 | Varied referent-introduction patterns |
| extraction | 96 | 48 | 2 | **Direction flips** vs EN (que required in ES) |
| postverbal_subject | 96 | 48 | 2 | V9 exception: delta=0, multiset equality instead |
| se_impersonal | 96 | 48 | 2 | V8 exception: verb form changes across pair |
| clitic_climbing | 48 | 24 | 1 | V9 exception (sort of): clitic re-attachment changes tok count |

**Cross-linguistic pairing**: all 8 core categories have perfect item_id
matches between EN and ES. Priming-ready.

**V14 cross-condition collisions (3 total)** — target text duplicates
across conditions where contexts differ. Structurally meaningful (reveal
where morphology does/doesn't carry subject-recovery signal). Valid in
context but worth flagging for collaborator:
- object_drop obj_3sg/16 vs obj_3pl/16: `los obreros llevaron al sótano .`
- object_drop obj_3sg/17 vs obj_3pl/17: `los niños subieron por las escaleras .`
- subject_drop_no_agreement subj_1sg/6 vs subj_1pl/1: `aceptar los términos principales .`

**Items flagged by the generating agents for collaborator's vet:**

1. **expletive**: `ello + se + V` stacking (`ello se puso ventoso`,
   `ello queda claro que…`). Agent judged "possibly ungrammatical rather
   than just archaic" — confirm marked-not-broken. Affected items:
   expl_seems #24; expl_be #10, #16-18; expl_seems #13-20.
2. **extraction**: `esperas que + future indicative` (ext_subj #8/#16,
   ext_obj #8/#16). `esperar que` typically selects subjunctive; kept
   future indicative to match EN structure. Mildly marked register-wise.
3. **subject_drop**: agent used **demonstratives `este`/`esta`/`estos`/`estas`**
   as the overt form for inanimate 3sg/3pl subjects (spanish.md §5.1
   only lists `él`/`ella`/`ellos`/`ellas`, which are strongly
   human-marked). Preserves the overt-vs-null intent naturally —
   confirm this deviation is acceptable.
4. **subject_drop**: 14 items rewritten to avoid preverbal clitics at
   the hotspot vicinity (paraphrased to keep hotspot = canonical finite
   verb position). Listed in agent report. Confirm structural
   equivalence with EN counterparts.
5. **postverbal_subject**: all 24 `postv_interrogative` SV-marked items
   (IDs 25-48). Agent judged "marked but not ungrammatical" — Peninsular
   native speaker should confirm none are catastrophically degraded.
6. **control**: `pretender`, `dejar`, `permitir` used — ECM-adjacent.
   Confirm control readings are preserved in their items.
7. **se_impersonal**: V2 and V8 are exceptions by design (hotspot token
   differs across pair; target differs by more than one token).
   Priming-analysis note: comparison is "surprisal at slot given
   preceding context," not "same token, different prefix."
8. **clitic_climbing**: no two-clitic clusters (avoided on purpose).
   Single-clitic attachments stay oxytone, no accents needed on `-rlo`,
   `-rla`, etc.

**SOP gaps the agents surfaced (for spanish.md v1.1 if iterated):**

- §5.1 mapping table should explicitly include demonstratives as the
  overt form for inanimate 3sg/3pl, not just human pronouns.
- §5.8 should clarify that `esperar`/`sospechar`/`dudar` take subjunctive
  in ES, so the EN "will + V" pattern won't map cleanly — either change
  matrix verb or accept the mild markedness.
- §6.2 (se_impersonal) V2/V8 exceptions could be stated more explicitly
  in the main validation section (§8), not just inline in §6.2.
- §4 grammaticality direction table doesn't distinguish "preference"
  from "grammaticality" — adding a third column would be clearer for
  conditions like `subject_drop` where both forms are grammatical in ES
  but null is preferred.

**Action for collaborator**: vet the staging CSVs at
`evaluation/stimuli/null-subj-v2/staging/es/*.csv`. Focus on the 8
flagged item ranges above; spot-check the rest. Per-category agent
reports (with spot-check examples) are in the original session
transcript.

### 2026-04-16 — Tokenization conventions resolved against training corpus

Sampled 500K chars from each of the 10 `data/spanish/train_90M/*.train`
source files. Findings:

- **Contractions**: FUSED. `del`/`al` dominate ~100:1 over `de el`/`a el`
  across every source (e.g., europarl: 951 `del` vs 0 `de el`; gutenberg:
  760 `del` vs 2 `de el`). Unambiguous.
- **Clitic attachment**: follows standard Spanish orthography —
  attached to non-finite forms (`comerlo`, `dándome`, `díselo`),
  separate when preverbal on finite verbs (`lo puso`, `la vi`).
  Confirmed by high "lo V" pattern frequency (263–889 per source) and
  clean `-rlo`/`-rla` attached counts in all sources.

Both defaults corrected in `docs/eval_stimuli/spanish.md`
(§1, §3, §6.3, §9.4). Initial guide had them as expanded/separated —
wrong on both; agent-run Spanish stimuli use the corpus-matching forms.

### 2026-04-16 — First opus-4.7 agent run on `subject_drop` (English)
- Ran an opus 4.7 subagent against `design.md` v1.0 to generate the
  English `subject_drop` category (all 6 person/number conditions, 24
  pairs each).
- Output: `evaluation/stimuli/null-subj-v2/staging/en/subject_drop.csv`
  (288 rows, 144 pairs).
- **Quality assessment**: genuinely good. Pair invariants V7 + V8 pass
  universally. Verb diversity 23–24 unique of 24 per condition
  (maximum). Natural English. Tense consistency maintained. No
  near-duplicates.
- **One deviation from spec**: agent capitalized names ("Ana") despite
  §3's lowercase rule. Fixed post-hoc by lowercasing the CSV.
- **SOP gaps surfaced by the agent** (now patched in design.md v1.1):
  - §3 vs §7.1 capitalization conflict → unified to lowercase, accents
    preserved for ES (new §9.7 failure mode added)
  - §8.1 length bounds (5–12) contradicted V4 (5–15) and example
    (3-token) → standardized: overt 6–15, null 5–14
  - "Each item uses one name" didn't apply to 1sg/1pl and was awkward
    for 2pl → added §7.3 name/vocative policy by condition
  - V13 taxonomy was 3rd-person-focused → now has person-specific
    taxonomies (3rd: human/inanimate/event/abstract; 2nd: addressee
    type; 1st: exempt)
  - §7 had no group vocatives → added §7.2 group-vocative inventory
    (team, class, everyone, …)
  - V2 hotspot position shifts between paired rows → documented
  - Manifest ownership → clarified: generation agent doesn't touch
    manifest, that's the Step-7 finalize agent
  - V9 transformers fallback → documented (whitespace-token proxy when
    transformers unavailable)
- **Takeaway**: the SOP-as-written + a solid model (opus 4.7) gets us
  to high-quality output in one pass, but the SOP had enough gaps that
  the agent had to make judgement calls. v1.1 should produce cleaner
  first-pass output.

### 2026-04-16 — Initial notebook setup
- Started this notebook. Seeded with decisions from the
  `docs/spanish_gold_integration.md` session and the minimal-pairs
  methodology lit review
  (`/Users/thomasmorton/zele/research/minimal-pairs-methodology-report.md`).
- Confirmed goal: Spanish set built primarily to enable cross-linguistic
  structural priming via cross-pairings — not as a standalone benchmark.
- Confirmed no transformations; scale items per condition upward.
- Clarified that BLiMP serves the sanity-check-control role at the
  evaluation-suite level; no embedded controls needed within the
  null-subject set itself.

---

## 4. Open questions

These are unresolved. Each will get a dated entry in §3 when decided.

### 4.1 Item scale
- **Question**: How many items per condition? Current English CSV has
  12. Lit review suggests ≥50 for adequate power on aggregate accuracy.
  For strictly-paired cross-linguistic stimuli, each item is more
  expensive to generate + vet.
- **Leaning**: 24–36 per condition as the sweet spot. Double or triple
  the current.
- **Constraint**: Must be generatable + vetable in reasonable time.

### 4.2 Category coverage (prereg gaps)
The prereg lists 18 target grammatical contexts; the current English
CSV has only 13. Missing:
- 3sg/3pl **object** drop (clitic pronoun contexts) — core prereg item
- **Subject extraction (that-trace)** — central to H5 distal effects
- **Object extraction** — central to H5 distal effects
- **Subordinate clause pronoun dropping** — may overlap with the current
  `6_long_distance_binding`; naming mismatch worth resolving

**Question**: Do we build the missing categories in English first, then
port all 18 to Spanish? Or port the 13 and treat the missing categories
as a separate workstream?

### 4.3 Generator choice
- **Option A**: Deepseek-V2 (prereg-committed, methodological continuity).
- **Option B**: Claude agent (easier to iterate, easier to do paired
  EN-ES generation in one pass).
- Either way, fluent-researcher vet remains essential.
- **Interacts with**: §4.8 (prereg amendment). If we're amending to add
  Spanish anyway, switching generators is free.

### 4.4 Stimulus structure
- **Current**: `context + target`, hotspot marked at a word position.
- **For priming**: may want `prime + context + target`, where prime is
  separable so congruent/incongruent prime conditions can be constructed.
- **Alternative**: keep target-only and construct prime combinations
  downstream by drawing cross-pairings between items.
- **Leaning**: target-only for the base set. Priming combinations built
  in analysis code, not baked into stimulus structure. But worth
  confirming.

### 4.5 Scoring / linking functions
Prereg commits to SLOR. Lit review flags SLOR **overcorrects** for
length and frequency (Tjuatja et al. 2024 — MORCELA).
Recommended additions:
- **SLLN-LP** (α ≈ 0.5): sub-linear length normalization, mitigates
  without overcorrecting (Liu et al. 2024 — ZhoBLiMP)
- **MORCELA**: learned normalization, outperforms SLOR across model
  families
- **Raw LP + MeanLP**: for robustness reporting
- **FIT-CLAMS controls**: essential for the ablation-comparison
  evaluations since ablations shift unigram frequency distributions
  (Padovani et al. 2025)
- **Question**: Lock in a multi-metric reporting policy now? This is
  prereg-relevant.

### 4.6 MultiBLiMP integration
Jumelet, Weissweiler & Bisazza (2025) — 101 languages including Spanish,
fully automated via UD + UniMorph. Could serve as:
- Standalone Spanish eval block (language-specific benchmark coverage)
- External validity check (if a Spanish model does well here but weirdly
  on null-subject items, that's signal)
- **Not useful for priming** (language-specific, no EN pairs).
- **Question**: Include as a separate evaluation block in the eval
  pipeline?

### 4.7 Literature-anchor items
Candidates for externally-normed, cross-linguistically-paired items:
- **Filiaci, Sorace & Carreiras (2013)** — Italian/Spanish paired
  null-subject items (PAH / Position of Antecedent Hypothesis)
- **Carminati (2002)** — Italian null vs overt resolution
- **Alonso-Ovalle et al. (2002)** — Spanish null-subject
- **Chamorro (2018)** — Spanish subject pronoun resolution
- Usefulness: 10–20 norming anchors embedded in the set give external
  validity the generated items can't. The Italian side of the dissertation
  doesn't have this; Spanish easily could.
- **Question**: Worth the curation effort? Depends on how much weight
  the norming anchor carries in the eventual writeup.

### 4.8 Prereg amendment strategy
The current prereg is English + Italian. Spanish is not in it.
Options:
- **Amendment**: add Spanish as a third language. Cleanest if Spanish
  is going to be a central part of the dissertation.
- **Separate sub-study**: Spanish as its own prereg + writeup, cross-
  referencing the main study. Lower commitment, more flexibility on
  scoring-function changes etc.
- **Exploratory replication**: Spanish findings framed as exploratory
  extension of the core English/Italian results. Least formal commitment.
- **Interacts with**: §4.3 (generator choice), §4.5 (scoring functions),
  §4.9 (norming).

### 4.9 Norming plan
- **Option A**: No norming. Rely on fluent-researcher vet during
  generation. Lowest effort, defensible if items are well-constructed.
- **Option B**: Sona-pool online norming, 20–30 native speakers per
  item, acceptability ratings. Gold standard but significant effort.
- **Option C**: Norm a subset (20–50 items) only, use as calibration.
- **Leaning**: Option C. Norming cost scales with item count; norming
  anchors plus spot-checks give you most of the validity at a fraction
  of the effort.
- **Interacts with**: §4.7 (lit-sourced anchors may already be normed).

### 4.10 Priming paradigm (later concern)
Not urgent for stimulus construction, but the choice affects what
downstream analysis looks like:
- **In-context**: prepend prime to target's context, measure target
  surprisal. No training.
- **Adaptation-based** (Prasad et al. 2019): briefly fine-tune on
  primes, measure target surprisal shift. Requires gradient access.
- **Cross-sentential**: prime and target as consecutive sentences in
  a single context window.
- User's note: they've "never failed to find the effect" — so paradigm
  choice is less risky than if this were speculative.

---

## 5. Methodology commitments (running list)

Things we've committed to methodologically, with source:

- **SLOR + SLLN-LP + LP reported together** (not locked yet; see §4.5)
- **FIT-CLAMS frequency controls for ablation comparisons**
  (not locked yet; see §4.5)
- **Token-length reporting per pair per tokenizer** (Ueda et al. 2024)
  — cheap, strengthens methods section, should just do it
- **By-item reporting alongside aggregate** (Newman et al. 2021) — user
  noted priming + interp work addresses systematicity vs. likely-behavior
  issue; still worth reporting by-item for transparency
- **Ablation verification: single-annotator, N=250, Wilson 95% CI**.
  See methods-section snippet under §3 entry dated 2026-05-12. Deposit
  the 900-row reservoirs and per-row judgments at OSF for reviewer audit.
- **Ablation operationalization choices** — past-tense paradigm,
  suppletive `be/have/do/go`, coarse 3sg fallback for unresolvable
  subjects, simplemma for Spanish lemmatization. All documented in the
  methods-section snippet under §3 (2026-05-12).

---

## 6. Key references

### Directly relevant
- **Minimal-pairs methodology report**
  (`/Users/thomasmorton/zele/research/minimal-pairs-methodology-report.md`)
  — lit review by Thomas, covers scoring functions, confounds, best
  practices.
- **OSF preregistration** (`docs/OSF_PREREGISTRATION.md`) — current
  commitments for English/Italian.
- **Spanish corpus docs** (`docs/spanish_corpus.md`) — the training
  corpus Spanish models will be evaluated on.
- **Annotation constants** (`annotation/constants.py`) — dialect choices
  and pronoun form mappings already baked in.

### Cross-linguistic priming precedent
- Hartsuiker, Pickering & Veltkamp (2004) — foundational cross-linguistic
  priming in Spanish-English bilinguals.
- Sinclair, Jumelet, Zuidema & Fernández (2022) — structural priming
  in multilingual LMs.
- Prasad, van Schijndel & Linzen (2019) — monolingual LM priming methodology.

### Literature-sourced items (candidates)
- Filiaci, Sorace & Carreiras (2013) — Italian/Spanish paired
  null-subject stimuli.
- Carminati (2002) — Italian null vs overt.
- Alonso-Ovalle et al. (2002) — Spanish null-subject.
- Chamorro (2018) — Spanish subject pronoun resolution.

### Cross-lingual benchmarks
- Jumelet, Weissweiler & Bisazza (2025) — MultiBLiMP (101 languages).
- Beauchemin et al. (2025) — QFrBLiMP (dialect variation precedent).

### Scoring / linking functions
- Tjuatja, Neubig, Linzen & Hao (2024) — MORCELA.
- Liu et al. (2024) — ZhoBLiMP / SLLN-LP.
- Padovani, Jumelet, Matusevych & Bisazza (2025) — FIT-CLAMS.
- Hu, Wilcox, Song, Mahowald & Levy (2025) — TACL, formal framework.

---

## 7. Action items

Concrete next steps, unowned and undated until someone picks them up.

- [ ] Resolve category-coverage question (§4.2): build missing 4–5
      categories in English first, or port the existing 13 to Spanish
      first?
- [ ] Decide on item scale (§4.1). Probably depends on generator choice.
- [ ] Decide on generator (§4.3). Probably depends on prereg-amendment
      strategy.
- [ ] Decide on prereg-amendment strategy (§4.8). Probably the upstream
      decision most of the others hang on.
- [ ] Pull down MultiBLiMP Spanish, assess phenomena overlap with
      current 13 categories (can happen independently; useful
      information regardless).
- [ ] Draft an audit doc mapping current English CSV → prereg 18
      categories, flagging gaps and naming mismatches (can happen
      independently; cheap).
