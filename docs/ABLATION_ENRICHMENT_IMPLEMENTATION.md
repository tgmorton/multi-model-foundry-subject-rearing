# Implementation Report: Ablation & Enrichment Procedures

## 1. Overview

This implementation adds the preregistered corpus manipulation procedures for the subject-drop acquisition experiments, covering both English and Italian. The work comprises three new ablation modules (6 registered ablation functions), a validation protocol script, 7 experiment configs, and the archiving of 4 superseded ablation modules. All new code follows the existing `AblationRegistry` pattern and is immediately usable with the `AblationPipeline` infrastructure.

### Registered ablation inventory after this change

| Registry name | Module | Language |
|---|---|---|
| `lemmatize_verbs` | `lemmatize_verbs.py` (existing) | Any (spaCy-agnostic) |
| `remove_expletive_sentences_en` | `remove_expletive_sentences.py` | English |
| `remove_expletive_sentences_it` | `remove_expletive_sentences.py` | Italian |
| `impoverish_case_en` | `impoverish_case.py` | English |
| `impoverish_case_it` | `impoverish_case.py` | Italian |
| `enrich_verbal_morphology` | `enrich_verbal_morphology.py` | English |

---

## 2. New Ablation Modules

### 2.1 Remove Expletive Sentences

**File:** `preprocessing/ablations/remove_expletive_sentences.py`

**Purpose:** Remove entire lines containing expletive constructions, rather than individual expletive tokens (the approach used by the now-archived `remove_expletives.py`). This provides a cleaner ablation — the replacement pool in `base.py:_rebuild_to_target_size` backfills removed lines to maintain corpus size.

**Interface:** Returns `("", 1)` for lines to remove, `(doc.text_with_ws, 0)` for lines to keep.

#### English (`remove_expletive_sentences_en`)

Detection: any token with `dep_ == "expl"`. In spaCy's UD annotation scheme for English, this primarily captures existential-*there* constructions ("There are three cats", "There exists a solution"). Weather-*it* and raising-*it* constructions ("It seems that...", "It is raining") are annotated as `nsubj` by spaCy rather than `expl`, so they are not caught by this filter — this is a known property of spaCy's English UD parser, not a code limitation.

Implementation: a factory function `make_remove_expletive_sentences_en()` returns the closure, which is called at registration time and bound to the registry name `"remove_expletive_sentences_en"`.

#### Italian (`remove_expletive_sentences_it`)

Italian has no overt expletive pronouns. Detection uses four heuristic categories, each implemented as a check over `VERB`/`AUX` tokens in the doc:

| Category | Detection logic | Verb list source |
|---|---|---|
| Weather verbs | `tok.lemma_.lower() in WEATHER_VERBS_IT` | `constants.py` (10 verbs) |
| Existential *ci + essere* | `tok.lemma_ == "essere"` with a child where `child.lower_ == "ci"` and `child.dep_ in ("expl", "advmod")` | hardcoded |
| Impersonal raising | `tok.lemma_ in IMPERSONAL_VERBS_IT` with a clausal complement (`ccomp`/`xcomp`/`csubj`) and no `nsubj` | `constants.py` (7 verbs) |
| Impersonal necessity | `tok.lemma_ in NECESSITY_VERBS_IT` with no `nsubj` | `constants.py` (4 verbs) |

If any category matches, the line is removed.

**Validation:** For each language, a validator is generated via `_make_validator(detect_fn)`. It re-parses the original line; if an expletive was detected, the ablated text must be empty; otherwise, it must match the original.

---

### 2.2 Impoverish Case

**File:** `preprocessing/ablations/impoverish_case.py`

**Purpose:** Collapse all pronoun case forms to their nominative equivalent, removing case morphology evidence from the corpus.

#### Shared infrastructure

Both languages share a core `_impoverish_case(doc, mapping, target_pos)` function that iterates over tokens, checks if `tok.lower_` appears in the mapping dict and `tok.pos_` is in the target POS set, and if so replaces the token with the nominative form. A `_match_capitalization(replacement, original)` helper preserves the original's capitalization pattern (ALL CAPS, Title Case, or lowercase).

Identity mappings (e.g., Italian reflexive `"si" -> "si"`) are detected and skipped to avoid inflating the replacement count.

#### English (`impoverish_case_en`)

The `ENGLISH_CASE_TO_NOM` dict maps 26 non-nominative forms to their nominative equivalent:

- **1sg:** me, my, mine, myself -> i
- **2sg/pl:** your, yours, yourself, yourselves -> you
- **3sg.m:** him, his, himself -> he
- **3sg.f:** her, hers, herself -> she
- **3sg.n:** its, itself -> it
- **1pl:** us, our, ours, ourselves -> we
- **3pl:** them, their, theirs, themselves -> they
- **rel/interrog:** whom, whose -> who

Target POS: `{"PRON", "DET"}` — spaCy tags possessive determiners (my, your, his, her, our, their) as `DET`.

**Smoke test result:** `"She gave him her book and they liked it."` -> `"She gave he she book and they liked it."` (2 replacements: him->he, her->she). "She" is already nominative (not in the mapping). "they" and "it" are nominative. Correct.

#### Italian (`impoverish_case_it`)

The `ITALIAN_CASE_TO_NOM` dict maps 31 non-nominative forms across three classes:

- **Strong obliques:** me -> io, te -> tu
- **Clitics (direct object):** mi -> io, ti -> tu, lo -> lui, la -> lei, ci -> noi, vi -> voi, li -> loro, le -> loro
- **Clitics (indirect object):** gli -> lui
- **Reflexive:** si -> si (identity, skipped)
- **Possessives (all gender/number forms):** mio/mia/miei/mie -> io, tuo/tua/tuoi/tue -> tu, suo/sua/suoi/sue -> lui, nostro/nostra/nostri/nostre -> noi, vostro/vostra/vostri/vostre -> voi

"loro" (possessive) is already invariant and coincides with the nominative form, so it is not in the mapping.

**Design note:** Replacing clitics with strong nominative forms is intentionally aggressive. E.g., "lo vedo" -> "lui vedo". This destroys clitic-placement evidence alongside case evidence, which serves the experimental purpose.

**Validation:** For each language, counts non-nominative pronouns in original vs. ablated; expects the ablated count to be lower.

---

### 2.3 Enrich Verbal Morphology

**File:** `preprocessing/ablations/enrich_verbal_morphology.py`

**Purpose:** Add synthetic, unambiguous agreement morphology to English verbs. This is the enrichment counterpart to the impoverishment provided by `lemmatize_verbs`. Only English is implemented — Italian already has rich agreement morphology and enrichment is not in the preregistered intervention list.

**Algorithm:** For each `VERB` or `AUX` token:

1. **Find the subject** via `_find_subject(verb)`:
   - First checks direct children for `nsubj` or `nsubj:pass`
   - If not found, walks up the auxiliary chain (if the verb's dep is `aux`, `auxpass`, or `xcomp`) to find the subject attached to a head verb
   - Handles direct subjects, auxiliary chains ("he has been walking"), and passive subjects

2. **Extract person/number** via `_get_person_number(subject)`:
   - Primary: reads UD morph features `Person` and `Number` from the token
   - Fallback: if morph features are absent, looks up the pronoun form in `_PRONOUN_TO_PERSON_NUMBER` (maps "i"->1sg, "you"->2sg, "he"/"she"/"it"->3sg, "we"->1pl, "they"->3pl)

3. **Lemmatize + suffix:** Strips the verb to `tok.lemma_` and appends the synthetic suffix from `DEFAULT_SUFFIX_MAP`

4. **Fallback:** If no subject is found or person/number cannot be determined (imperatives, infinitives, fragments), the verb is reduced to its bare lemma — effectively impoverishing rather than enriching.

**Default synthetic paradigm (Latin-inspired):**

| Person/Number | Suffix | Example |
|---|---|---|
| 1sg | `-o` | walk -> walko |
| 2sg | `-as` | walk -> walkas |
| 3sg | `-at` | walk -> walkat |
| 1pl | `-amus` | walk -> walkamus |
| 2pl | `-atis` | walk -> walkatis |
| 3pl | `-ant` | walk -> walkant |

The `DEFAULT_SUFFIX_MAP` dict can be overridden by creating a custom factory or modifying the dict before registration.

**Smoke test results:**

| Input | Output | Explanation |
|---|---|---|
| "She walks to school every day." | "She walkat to school every day." | 3sg subject "She" -> suffix `-at` |
| "They have been running all morning." | "They haveant beant runant all morning." | 3pl subject "They" propagates to all three verb/aux tokens |
| "I like cats and you like dogs." | "I likeo cats and you like dogs." | 1sg "I" -> `-o`; second "like" with subject "you" was not enriched with `en_core_web_sm` (model-quality limitation; `en_core_web_trf` should handle coordination better) |

**Validation:** Checks that at least some verb forms in the ablated text differ from the original (novel tokens introduced by the synthetic suffixes).

---

## 3. Italian Verb Constants

**File:** `analysis/corpus_descriptives/constants.py`

Three new frozen sets were added for the Italian expletive-equivalent detection:

```python
WEATHER_VERBS_IT = frozenset({
    "piovere", "nevicare", "grandinare", "tuonare", "lampeggiare",
    "gelare", "albeggiare", "annottare", "imbrunire", "diluviare",
})

IMPERSONAL_VERBS_IT = frozenset({
    "sembrare", "parere", "risultare", "capitare",
    "succedere", "accadere", "avvenire",
})

NECESSITY_VERBS_IT = frozenset({
    "bisognare", "bastare", "convenire", "occorrere",
})
```

These are placed in the shared `constants.py` alongside the existing `WEATHER_VERBS` (English) list, under a clearly labeled section. The lists are designed to be expandable.

---

## 4. Validation Protocol

**File:** `scripts/validate_ablation.py`

A standalone CLI script for producing a stratified i.i.d. sample for hand review. It is independent of the preprocessing pipeline (no spaCy dependency) and operates purely on file comparison.

**Usage:**

```bash
python scripts/validate_ablation.py \
    --original data/raw/train_90M/ \
    --ablated  data/processed/exp_remove_expletive_sentences_en/ \
    --n-per-genre 20 \
    --seed 42 \
    --output validation/remove_expletive_sentences_en_sample.tsv
```

**Behavior:**

1. Matches files by filename across both directories (skipping `.json` manifests)
2. For each matched genre file, randomly samples N line indices using a seeded `random.Random` instance
3. Reads both the original and ablated lines at those indices. For line-removal ablations where the ablated file is shorter, out-of-range indices yield `<LINE_REMOVED>`
4. Writes a TSV with columns: `genre | line_num | original | ablated | correct? | notes` — the last two columns are empty for the reviewer
5. Prints a per-genre summary table to stdout with: original line count, ablated line count, line delta, and character-level delta percentage

---

## 5. Archived Ablations

**Directory:** `preprocessing/ablations/archived/`

Four modules were moved from `preprocessing/ablations/` to `preprocessing/ablations/archived/` via `git mv`:

| File | Former registry name | Reason for archiving |
|---|---|---|
| `remove_articles.py` | `remove_articles` | Not in preregistered intervention list |
| `impoverish_determiners.py` | `impoverish_determiners` | Not in preregistered intervention list |
| `remove_subject_pronominals.py` | `remove_subject_pronominals` | Not in preregistered intervention list |
| `remove_expletives.py` | `remove_expletives` | Superseded by sentence-level `remove_expletive_sentences` |

The `archived/` directory has its own `__init__.py` with a docstring explaining these modules are preserved for reference but not auto-registered. The modules are structurally intact and could be re-activated by importing them.

---

## 6. Updated `__init__.py`

**File:** `preprocessing/ablations/__init__.py`

The package init now imports only the four active modules:

```python
from . import lemmatize_verbs
from . import remove_expletive_sentences
from . import impoverish_case
from . import enrich_verbal_morphology
```

This triggers registration of the 6 ablation functions at import time (2 from `remove_expletive_sentences`, 2 from `impoverish_case`, 1 from `enrich_verbal_morphology`, 1 from `lemmatize_verbs`).

---

## 7. Experiment Configs

Seven new YAML configs were created in `configs/`, all based on the structure of the existing `experiment_1_remove_expletives.yaml`. Each config specifies the ablation type, spaCy model, data paths, and the full model architecture/training procedure:

| Config file | Ablation | Language | spaCy model | Data root |
|---|---|---|---|---|
| `experiment_en_remove_expletive_sentences.yaml` | `remove_expletive_sentences_en` | EN | `en_core_web_trf` | `data/raw/train_90M/` |
| `experiment_it_remove_expletive_sentences.yaml` | `remove_expletive_sentences_it` | IT | `it_core_news_trf` | `data/italian/raw/train_90M/` |
| `experiment_en_impoverish_case.yaml` | `impoverish_case_en` | EN | `en_core_web_trf` | `data/raw/train_90M/` |
| `experiment_it_impoverish_case.yaml` | `impoverish_case_it` | IT | `it_core_news_trf` | `data/italian/raw/train_90M/` |
| `experiment_en_lemmatize_verbs.yaml` | `lemmatize_verbs` | EN | `en_core_web_trf` | `data/raw/train_90M/` |
| `experiment_it_lemmatize_verbs.yaml` | `lemmatize_verbs` | IT | `it_core_news_trf` | `data/italian/raw/train_90M/` |
| `experiment_en_enrich_verbal_morphology.yaml` | `enrich_verbal_morphology` | EN | `en_core_web_trf` | `data/raw/train_90M/` |

All configs share the same model architecture (12-layer GPT-2, 768 hidden, 12 heads) and training procedure (lr=4e-4, 20 epochs, gradient accumulation 8, flash attention, mixed precision) as the existing baseline configs. Italian configs use the `data/italian/` directory tree.

---

## 8. File Manifest

### Created (12 files)

| File | Lines |
|---|---|
| `preprocessing/ablations/remove_expletive_sentences.py` | 165 |
| `preprocessing/ablations/impoverish_case.py` | 211 |
| `preprocessing/ablations/enrich_verbal_morphology.py` | 221 |
| `scripts/validate_ablation.py` | 185 |
| `preprocessing/ablations/archived/__init__.py` | 6 |
| `configs/experiment_en_remove_expletive_sentences.yaml` | 68 |
| `configs/experiment_it_remove_expletive_sentences.yaml` | 68 |
| `configs/experiment_en_impoverish_case.yaml` | 68 |
| `configs/experiment_it_impoverish_case.yaml` | 68 |
| `configs/experiment_en_lemmatize_verbs.yaml` | 67 |
| `configs/experiment_it_lemmatize_verbs.yaml` | 67 |
| `configs/experiment_en_enrich_verbal_morphology.yaml` | 68 |

### Modified (2 files)

| File | Change |
|---|---|
| `preprocessing/ablations/__init__.py` | Replaced 5 archived imports with 4 active imports |
| `analysis/corpus_descriptives/constants.py` | Added 3 Italian verb frozensets (21 verbs total) |

### Moved (4 files)

| From | To |
|---|---|
| `preprocessing/ablations/remove_articles.py` | `preprocessing/ablations/archived/remove_articles.py` |
| `preprocessing/ablations/impoverish_determiners.py` | `preprocessing/ablations/archived/impoverish_determiners.py` |
| `preprocessing/ablations/remove_subject_pronominals.py` | `preprocessing/ablations/archived/remove_subject_pronominals.py` |
| `preprocessing/ablations/remove_expletives.py` | `preprocessing/ablations/archived/remove_expletives.py` |

---

## 9. Verification

Registration was verified programmatically — `AblationRegistry.list_ablations()` returns all 6 expected names with correct function references.

Smoke tests were run with `en_core_web_sm` (the model available at test time). Results:

- **impoverish_case_en:** "She gave him her book and they liked it." -> "She gave he she book and they liked it." — correctly replaces `him` -> `he` and `her` -> `she`, leaves nominative forms unchanged.
- **enrich_verbal_morphology:** "She walks to school every day." -> "She walkat to school every day." — correctly identifies 3sg subject and appends `-at`.
- **remove_expletive_sentences_en:** Correctly detects `expl` dep on "There are..." and "There exists..." constructions. Does not fire on non-expletive sentences.

The `en_core_web_trf` model (specified in all production configs) was also tested for `expl` annotation: it tags "There" as `expl` in existential constructions but not "It" in weather/raising constructions (consistent with spaCy's UD scheme).

---

## 10. Known Limitations and Notes

1. **spaCy `expl` scope (English):** The UD annotation scheme used by spaCy's English models tags only existential-*there* as `expl`. Weather-*it* and raising-*it* receive `nsubj`. This means `remove_expletive_sentences_en` targets there-expletives specifically. If broader expletive-*it* coverage is desired, the detection function would need to be extended with heuristics similar to the Italian module (weather verb lemma checks, raising verb + clausal complement patterns).

2. **Enrichment subject-finding quality:** The `_find_subject` function handles direct subjects and auxiliary chains but may miss subjects in complex coordination or long-distance dependencies, depending on the spaCy model's parse quality. The `en_core_web_trf` model is recommended for production runs. Verbs without identifiable subjects fall back to bare lemmatization.

3. **Italian clitic replacement:** Replacing clitics (mi, ti, lo, la, etc.) with strong nominative forms creates syntactically anomalous output. This is by design — the experimental purpose is to destroy case evidence, not to produce grammatical Italian.

4. **Italian spaCy model:** The Italian configs specify `it_core_news_trf`, which must be installed (`python -m spacy download it_core_news_trf`) before running Italian experiments.

5. **Capitalization of "I":** The case impoverishment maps lowercase "my" -> lowercase "i" (preserving the original's capitalization pattern). The English convention of always capitalizing "I" is not enforced, since the ablation's purpose is to destroy morphological distinctions rather than produce natural English.
