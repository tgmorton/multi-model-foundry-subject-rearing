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
