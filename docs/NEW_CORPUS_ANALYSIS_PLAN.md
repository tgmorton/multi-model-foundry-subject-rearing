# Corpus Descriptive Analysis: Revised Specification

## Overview

This document specifies the descriptive corpus analyses for the null-subject controlled rearing study. It is organized around the preregistered hypotheses: each analysis is motivated by the hypothesis it serves and the experimental decision it informs. Analyses that do not serve a preregistered hypothesis are explicitly labeled exploratory.

The analyses operate over English (BabyLM, ~90M tokens) and Italian (bebe-lm, ~90M tokens) training corpora.

---

## Hypothesis–Analysis Mapping

| Hypothesis                    | What corpus analysis must establish                          | Analysis section |
| ----------------------------- | ------------------------------------------------------------ | ---------------- |
| H1 (baseline learnability)    | Basic corpus properties; sufficient signal for target grammar | §1               |
| H2 (pronoun manipulation)     | Pronoun frequencies by person/number/case/function; genre distribution; what "remove pronouns" and "insert pronouns" affects | §2               |
| H3 (developmental trajectory) | Checkpoint-relevant input structure; when in training different evidence types are encountered | §1, §2           |
| H4 (PMI vs. random removal)   | PMI distribution of subject pronouns in context; whether PMI-based and random removal select different tokens | §3               |
| H5 (distal effects)           | That-trace environment frequencies; extraction distributions; co-occurrence of null-subject-cluster phenomena | §4               |
| H6a (expletive removal)       | Expletive frequency; proportion of input affected by removal | §5               |
| H6b (morphology manipulation) | Agreement paradigm completeness; person×number distribution of inflected forms; what "rich" vs. "impoverished" means | §6               |
| H6c (case neutralization)     | Case-marked pronoun frequencies; what neutralization removes | §2 (subsection)  |
| H6d (stacked ablations)       | Overlap/independence of above; proportion of corpus affected by each manipulation individually and jointly | §7               |
| H7–H10 (architecture)         | No corpus analysis needed (these are model-side hypotheses)  | —                |

---

## §1. Corpus Overview and Baseline Properties

**Serves:** H1 (baseline learnability), H3 (developmental trajectory), all hypotheses (interpretive context)

### 1.1 Basic Corpus Statistics

Token counts, type counts, type-token ratio, mean sentence length — by language and genre. This is the denominator for all subsequent proportions.

**Output:** Summary table with basic descriptives by language × genre.

### 1.2 Genre Composition (BabyLM only)

The BabyLM corpus comprises multiple sources (CHILDES, BNC, Gutenberg, OpenSubtitles, Simple Wikipedia, Switchboard) that differ in register, formality, and likely pronoun distribution. Genre composition determines what the model sees during training and at what rate.

**Key question:** Does CHILDES (child-directed speech) differ from other sources in the properties relevant to our manipulations? If pronouns are disproportionately frequent in CDS, genre-weighted effects of pronoun removal will differ from uniform effects.

**Output:** Token counts by source; proportion of total corpus per source. Genre breakdowns are reported for §2 and §3 (pronoun-relevant analyses) and §5 (that-trace environments) only — not for all analyses.

### 1.3 Clause Structure

Clause structure provides the scaffold for all subsequent analyses. Every analysis below references finite verbs, their subjects, and their embedding context.

**Finite root clauses** (`dep=ROOT`, `VerbForm=Fin`):

- with overt subject (`nsubj` or `nsubj:pass` child)
- with expletive subject (`expl` child)
- without overt subject (candidate null-subject site)

**Subordinate finite clauses** (`dep=ccomp/advcl/acl:relcl`, `VerbForm=Fin`):

- same subject-presence breakdown

**Infinitival complements** (`dep=xcomp`, `VerbForm=Inf`):

- classified as control vs. raising by matrix verb lemma

**Output:** Counts and proportions for each cell; cross-tabulation of clause type × subject status × language. This table is the primary reference for interpreting all subsequent analyses.

---

## §2. Pronoun Distribution

**Serves:** H2 (pronoun manipulation), H6c (case neutralization)

This section characterizes what the pronoun removal (English) and pronoun insertion (Italian) manipulations actually affect.

### 2.1 Subject Pronoun Inventory

Frequency of each subject pronoun by person/number:

- English: I, you, he, she, it, we, they
- Italian: io, tu, lui/lei, noi, voi, loro

Detection: `dep=nsubj`, `pos=PRON`, with `Person` and `Number` morphological features.

**Key question for H2:** What is the overall subject pronoun rate (proportion of finite verbs with pronominal subjects)? This determines the magnitude of the pronoun removal manipulation at each sweep level (0–100%, step 10%).

### 2.2 Object Pronoun Inventory

- English: me, you, him, her, it, us, them
- Italian: mi, ti, lo/la, ci, vi, li/le (clitics)

Detection: `dep=obj/iobj`, `pos=PRON`.

**Relevance:** Object pronouns are not directly manipulated in the preregistered design, but they serve as a control — object pronoun acceptability should not change under subject pronoun manipulation unless distal effects are present.

### 2.3 Case-Marked Pronouns

**Serves:** H6c (case neutralization)

English distinguishes nominative (I, he, she, we, they) from accusative (me, him, her, us, them). The case neutralization ablation collapses this distinction (e.g., replacing all nominatives with a case-neutral form, or replacing accusatives with nominatives).

**Key question:** How many tokens are affected by case neutralization? What proportion of pronouns carry unambiguous case marking? ("you" and "it" are case-ambiguous and would not be affected.)

**Output:** Frequency table by pronoun × case; count and proportion of case-marked tokens; proportion of total corpus affected by neutralization.

### 2.4 Genre Breakdown of Pronoun Distribution

**Serves:** H2, H4

Report §2.1 and §2.3 broken down by BabyLM genre. The critical comparison is CHILDES vs. other sources.

**Key question:** If CHILDES has disproportionately high 1sg/2sg pronoun rates (as expected in child-directed speech), then pronoun removal disproportionately affects CDS input. Does this matter for interpretation?

**Output:** Pronoun frequency by person/number × genre.

---

## §3. Pointwise Mutual Information of Pronouns

**Serves:** H4 (PMI vs. random removal)

H4 tests whether removing pronouns by contextual predictability (PMI) produces different model outcomes than random removal at matched rates. This analysis characterizes the PMI distribution that determines what the PMI sweep actually targets.

### 3.1 PMI Computation

For each subject pronoun token in the corpus, compute the pointwise mutual information between the pronoun and its local context (preceding N tokens or dependency context). PMI quantifies how predictable a given pronoun is in its context — high-PMI pronouns are highly expected; low-PMI pronouns are surprising.

**Method:** Use a background unigram model and a contextual model (5-gram or simple neural LM trained on the corpus) to compute:

```
PMI(pronoun, context) = log p(pronoun | context) - log p(pronoun)
```

### 3.2 PMI Distribution

**Key questions:**

- What does the PMI distribution of subject pronouns look like? Is it unimodal, bimodal, skewed?
- Are high-PMI pronouns concentrated in certain persons (e.g., 1sg "I" in narrative contexts), genres (e.g., CHILDES), or syntactic positions (e.g., topic continuity)?
- At each removal level (10%, 20%, ..., 100%), which tokens does PMI-based removal select vs. random removal? How much overlap is there?

**Output:** PMI distribution histogram; PMI by person/number; PMI by genre; overlap between PMI-ranked and random removal sets at each sweep level.

### 3.3 Interpretation for H4

If the PMI distribution is relatively uniform, PMI-based and random removal will select similar tokens and H4 will show null results for uninformative reasons. If the distribution is heavily skewed — e.g., topic-continuing 3sg pronouns are highly predictable while discourse-initial 1sg pronouns are not — then PMI-based removal selectively targets a linguistically coherent subset. This analysis must be completed before interpreting H4 results.

---

## §4. That-Trace Environments

**Serves:** H5 (distal effects)

The that-trace effect is the key test of distal effects: it is syntactically related to null-subject licensing but distributionally independent of pronoun frequency. If pronoun manipulation affects that-trace judgments, this is strong evidence for structured grammatical representations.

This analysis characterizes how much that-trace-relevant input the model receives during training.

### 4.1 Bridge Verb Complement Clauses

Identify all finite `ccomp` clauses under bridge verbs (verbs that permit extraction from their complement).

**English bridge verbs:** think, believe, say, know, assume, expect, hope, suppose, claim, report, imagine, feel, suspect, guess, figure, reckon, suggest, declare, announce, predict

**Italian bridge verbs:** pensare, credere, dire, sapere, supporre, sperare, immaginare, sentire, ritenere, affermare, dichiarare, sostenere, annunciare, prevedere

Lists are seed lists; any verb appearing with `ccomp` and a fronted wh-word is a candidate bridge verb and will be flagged for review.

### 4.2 Complementizer Presence

For each bridge verb complement clause, classify:

- **+Comp:** "that"/"che" present (mark dependent with appropriate lemma)
- **−Comp:** complementizer absent

**Key question:** What is the "that"-omission rate in English? This optionality is what creates the learning environment for the that-trace constraint. If "that" is nearly always omitted, the model rarely sees +Comp environments and the constraint may be hard to learn.

### 4.3 Extraction Type

Using heuristics (see original spec §1.8 Steps 3–4 for details):

- **Subject extraction:** Matrix clause contains wh-word; embedded clause has no `nsubj`
- **Object extraction:** Matrix clause contains wh-word; embedded clause has `nsubj` but missing `obj`
- **No extraction:** Declarative complement

These heuristics will produce errors. We validate against a manually annotated sample of 100 sentences.

### 4.4 Cross-Tabulation

The critical output: **complementizer × extraction type × language**

|                    | +Comp                                                        | −Comp            |
| ------------------ | ------------------------------------------------------------ | ---------------- |
| Subject extraction | Should be ~0 in English (that-trace violation), attested in Italian | Grammatical both |
| Object extraction  | Grammatical both                                             | Grammatical both |
| No extraction      | Grammatical both                                             | Grammatical both |

**Key question for H5:** How many that-trace-relevant contexts (bridge verb + complement clause + extraction) exist in the training corpus? If the count is very low (e.g., <100 per 90M tokens), any distal effect would be remarkable — the model would be generalizing from minimal direct evidence.

### 4.5 Genre Breakdown

Report §4.4 by BabyLM genre. Wh-extraction from complement clauses likely concentrates in conversational registers (CHILDES, Switchboard, OpenSubtitles) rather than written prose.

**Output:** Per-genre cross-tabulation; bridge verb frequency by lemma; complementizer rate overall and by verb.

---

## §5. Expletive Distribution

**Serves:** H6a (expletive removal)

### 5.1 Expletive Identification

English expletive subjects: tokens with `dep=expl` (primarily "it" and "there").

Italian: no overt expletives; count verbs that would take expletive subjects (weather verbs, raising verbs, existentials) — these appear with null expletive subjects.

### 5.2 Expletive Classification

By verb class:

- **Weather verbs:** rain, snow, etc. / piovere, nevicare, etc.
- **Raising verbs:** seem, appear, happen / sembrare, apparire, capitare
- **Existentials:** "there is/are" constructions / "c'è/ci sono"

### 5.3 Manipulation Feasibility

**Key question:** What proportion of the English corpus consists of expletive-containing sentences? If expletives are <1% of sentences, removing them may not produce detectable effects regardless of theoretical predictions. This directly informs whether H6a is well-powered.

**Output:** Expletive counts by class; proportion of sentences containing expletives; proportion of total tokens in expletive-containing sentences.

---

## §6. Verbal Morphology

**Serves:** H6b (morphology manipulation)

### 6.1 Agreement Paradigm Completeness

For each language, tabulate the person×number distribution of finite verb forms.

**English paradigm (present tense):**

|      | Singular  | Plural |
| ---- | --------- | ------ |
| 1st  | walk      | walk   |
| 2nd  | walk      | walk   |
| 3rd  | walk**s** | walk   |

English has massive syncretism: 5 of 6 cells are identical. The only morphological cue to person/number is 3sg -s.

**Italian paradigm (present tense, 1st conjugation):**

|      | Singular  | Plural       |
| ---- | --------- | ------------ |
| 1st  | parl**o** | parl**iamo** |
| 2nd  | parl**i** | parl**ate**  |
| 3rd  | parl**a** | parl**ano**  |

Italian has 6 distinct forms. This is what "rich agreement" means: the verb form uniquely identifies the subject's person and number, licensing null subjects.

### 6.2 Corpus Distribution of Agreement Cells

Count the frequency of each person×number×tense combination in the corpus, using morphological features (`Person`, `Number`, `Tense`, `VerbForm=Fin`).

**Key questions:**

- Is the paradigm evenly distributed, or are certain cells (e.g., 3sg present) massively overrepresented?
- What is the frequency of the English 3sg -s form relative to bare forms? This is the primary morphological cue to agreement in English.
- For Italian, are there syncretic forms across tenses that reduce the effective paradigm richness?

### 6.3 Manipulation Specification

The morphology manipulation (H6b) involves "adding rich agreement to English" and "impoverishing Italian agreement." This analysis determines what that means concretely:

- **English enrichment:** Replace syncretic bare forms with morphologically marked forms (e.g., "walk" → "walko" for 1sg, "walkiamo" for 1pl). What proportion of verb tokens are affected?
- **Italian impoverishment:** Collapse distinct forms into a single syncretic form (e.g., all present tense → 3sg form). What proportion of verb tokens are affected?

**Output:** Person×number×tense frequency table by language; syncretism rate; proportion of corpus affected by each manipulation direction.

---

## §7. Manipulation Feasibility and Overlap

**Serves:** H6d (stacked ablations), all hypotheses (power/interpretability)

This is a cross-cutting analysis that takes the per-manipulation estimates from §2–§6 and examines their joint properties.

### 7.1 Proportion of Corpus Affected

For each manipulation, what percentage of the corpus (by tokens and by sentences) is directly altered?

| Manipulation            | Tokens affected | Sentences affected |
| ----------------------- | --------------- | ------------------ |
| Pronoun removal (100%)  | from §2         | from §2            |
| Pronoun removal (50%)   | from §2         | from §2            |
| Expletive removal       | from §5         | from §5            |
| Morphology manipulation | from §6         | from §6            |
| Case neutralization     | from §2.3       | from §2.3          |

**Key question:** Are any manipulations too sparse to produce detectable effects? If expletive removal affects 0.3% of tokens and case neutralization affects 2%, the stacked ablation (H6d) is dominated by the larger manipulations and the small ones may contribute nothing.

### 7.2 Overlap Between Manipulations

For stacked ablations, how much do manipulations overlap? A sentence containing an expletive subject also contains a pronoun (the expletive itself) and potentially a case-marked pronoun. Removing expletives AND removing pronouns double-counts these sentences.

**Output:** Pairwise overlap matrix (proportion of sentences affected by manipulation A that are also affected by manipulation B).

### 7.3 Interpretation for H6d

If manipulations are largely non-overlapping, stacked effects should be roughly additive under a statistical learning account. If they overlap substantially, additive predictions need to be adjusted for shared sentences.

---

## §8. Negation and Subject Realization

**Serves:** H3 (developmental trajectory), exploratory

Performance-based accounts of children's early null subjects (Bloom, 1990; Valian, 1991) propose that children omit subjects more in longer or more complex sentences due to processing limitations. Negation increases sentence complexity, and children show higher subject omission rates in negated clauses. If models show a similar pattern (higher null-subject rates in negated contexts, especially early in training), this would be consistent with processing-based accounts; if they don't, it would support grammatical rather than performance-based explanations of the developmental trajectory (H3).

### 8.1 Negation Identification

Negated clauses: contain `advmod` or dependent with `Polarity=Neg` (English: not, n't; Italian: non).

### 8.2 Negation Position

Token index of negation relative to finite verb. English: preverbal (with auxiliary) or postverbal (with "not"). Italian: strictly preverbal "non."

### 8.3 Subject Realization in Negated vs. Non-Negated Clauses

Cross-tabulate negation × subject status (overt / null / expletive) for finite root clauses and subordinate clauses.

**Key question:** Is subject omission more frequent in negated clauses in either corpus? If so, this provides a corpus-level baseline for the performance-based prediction. If the corpus itself shows no negation effect on subject realization, then any model pattern must emerge from learning dynamics rather than input statistics.

**Output:** Negation frequency; positional distribution; subject realization cross-tabulation with negation; comparison across languages.

---

## §9. Evaluation Context Frequency

**Serves:** All hypotheses (interpretability of model evaluation results)

How often do contexts resembling the 18 preregistered evaluation stimulus types actually occur in training? This determines whether model performance on evaluation items reflects generalization vs. memorization.

### 9.1 Method

For each evaluation category, define a set of corpus-level features that approximate the stimulus structure. These are not exact pattern matches (the evaluation stimuli are constructed, not naturalistic) but structural proxies:

| Evaluation category          | Corpus proxy                                          |
| ---------------------------- | ----------------------------------------------------- |
| 3sg/3pl subject drop         | Finite clause with 3sg/3pl pronominal subject         |
| 3sg/3pl object drop          | Clause with 3sg/3pl pronominal object                 |
| 1sg/2sg/1pl/2pl subject drop | Finite clause with 1sg/2sg/1pl/2pl pronominal subject |
| Subordinate clause drop      | `ccomp` with pronominal subject                       |
| Subject control              | `xcomp` under control verb                            |
| Object control               | `xcomp` under object-control verb                     |
| Expletive "seems"            | Raising verb with expletive                           |
| Expletive "be"               | Existential or presentational with expletive          |
| Long-distance binding        | `ccomp` with pronominal embedded subject              |
| Conjunction ± topic shift    | Coordinated clauses with same/different subjects      |
| Subject extraction ± "that"  | That-trace environments (from §4)                     |
| Object extraction ± "that"   | Object extraction from complement clause (from §4)    |

### 9.2 Output

Frequency table: evaluation category × language × genre. Flag any categories with very low counts (<100 per 90M tokens) as items where model performance would necessarily reflect generalization rather than direct training exposure.

---

## §10. Wh-Questions

**Serves:** H5 (distal effects, secondary)

Wh-question structure is relevant to the null-subject cluster because subject extraction interacts with complementizer presence (that-trace) and with subject position (pre/postverbal). This analysis supplements §4 with the broader distribution of wh-questions.

### 10.1 Wh-Question Classification

- **Subject wh-questions:** wh-word is `nsubj` of root verb
- **Object wh-questions:** wh-word is `obj` of root verb
- **Embedded questions:** `dep=ccomp` with wh-word as dependent of embedded verb

Detection: `POS=PRON` or `POS=ADV` with `PronType=Int` or lemma in {who, what, which, where, when, why, how} / {chi, che, cosa, quale, dove, quando, perché, come}.

### 10.2 Output

Subject/object extraction counts by language; embedded question frequency; proportion of wh-questions involving extraction from complement clause (overlap with §4).

---

## Analyses Not Included

The following analyses from the original specification are **not included** in the revised version because they do not serve a preregistered hypothesis:

- **Relative clauses (original §1.6):** Subject/object relative clause distributions do not connect to any manipulation or evaluation item. Omitted.
- **Resumptive pronouns (original §1.6 subpart):** Unreliable to detect without coreference resolution; no theoretical connection to preregistered hypotheses. Omitted.

These could be conducted as exploratory analyses if time permits.

---

## Phase 2: Pronoun Recovery Model

Phase 2 enables two things: (1) the Italian pronoun insertion manipulation (H2, critical path), and (2) descriptive analysis of null-subject rates by context (supporting §8 and exploratory analyses). The architecture, training strategy, and validation procedures are unchanged from the original specification (see original §2.1–2.3).

**Priority note:** Function (1) — producing pronoun-inserted Italian text for the H2 ablation — is on the critical path. Function (2) — descriptive statistics on null-subject contexts — is secondary. Development effort should be allocated accordingly.

---

## Implementation Notes

### Tools

- spaCy with `en_core_web_trf` / `it_core_news_lg` (or trf if available)
- Existing preprocessing infrastructure in `preprocessing/`
- Simple N-gram or neural LM for PMI computation (§3)
- DeepSeek API for scaled annotation (Phase 2)
- HuggingFace Transformers for pronoun recovery model (Phase 2)

### Output Format

Per the layered annotation architecture: all analyses produce per-sentence annotations stored in Parquet. Aggregation scripts compute summary statistics from the annotated corpus. Summary CSVs are the input to the RMarkdown report.

### Validation Samples

For heuristic-based analyses (§4 extraction detection, §3 PMI computation), manually annotate a sample of 100 sentences per language to estimate precision and recall. Report these validation metrics alongside the corpus statistics.

---

## Presentation Plan

This section specifies how each analysis will be reported in the corpus descriptive results. The audience is the paper's methods/results section and supplementary materials. Each analysis is presented at two levels of granularity: a primary summary (for the main text) and a detailed breakdown (for supplement).

### General Principles

All analyses are presented as cross-linguistic comparisons (English vs. Italian) because the theoretical interest is in how the two languages differ in the input available to learners. Raw counts are always accompanied by proportions (per 1M tokens or per finite verb, as appropriate). Confidence intervals on proportions are computed by bootstrap over documents (not sentences) to account for within-document dependence.

### §1: Corpus Overview

**Main text:** A single summary table with token count, sentence count, mean sentence length, and type-token ratio by language. For BabyLM, a stacked bar or table showing genre composition by proportion of tokens.

**Supplement:** Full genre × basic-statistic table.

### §2: Pronoun Distribution

**Main text:** A figure showing subject pronoun frequency by person/number, comparing English and Italian, normalized per 1,000 finite verbs. A single summary statistic: overall subject pronoun rate (% of finite verbs with pronominal subjects) by language. For H6c: count and proportion of case-marked pronoun tokens.

**Supplement:** Full person × number × case × function frequency table. Genre breakdown of pronoun rates (table). Object pronoun frequency table.

### §3: PMI Distribution

**Main text:** A histogram or density plot of PMI values for subject pronouns in each language. A scatterplot or table showing the overlap between PMI-ranked and random removal sets at representative sweep levels (25%, 50%, 75%). A summary of whether high-PMI pronouns cluster by person, genre, or syntactic position.

**Supplement:** PMI by person/number table. PMI by genre table. Token-level overlap at all sweep levels.

### §4: That-Trace Environments

**Main text:** The cross-tabulation (complementizer × extraction type × language) as a table. Total count of that-trace-relevant contexts by language. Complementizer omission rate in English complement clauses.

**Supplement:** Bridge verb frequency table. Per-genre cross-tabulation. Validation metrics from manual annotation sample (precision/recall of extraction heuristics).

### §5: Expletive Distribution

**Main text:** Total expletive count by class (weather, raising, existential) and language. Proportion of sentences containing expletives (the manipulation feasibility number).

**Supplement:** Expletive verb lemma frequency table. English vs. Italian comparison of would-be-expletive verb contexts.

### §6: Verbal Morphology

**Main text:** Person × number paradigm tables for English and Italian (showing distinct vs. syncretic forms). Distribution of agreement cells in the corpus (as a heatmap or proportional table). A summary of paradigm richness: number of distinct surface forms per paradigm by language.

**Supplement:** Full person × number × tense frequency table. Syncretism analysis. Proportion of corpus affected by enrichment/impoverishment.

### §7: Manipulation Feasibility

**Main text:** A summary table showing proportion of corpus affected (tokens and sentences) by each manipulation. Pairwise overlap matrix as a small heatmap or table.

**Supplement:** Detailed overlap statistics. Sentence-level co-occurrence frequencies.

### §8: Negation and Subject Realization

**Main text:** Cross-tabulation of negation × subject status by language (2×3 table per language, or combined). Statistical test of association (χ² or equivalent). A brief statement of whether the corpus shows higher subject omission in negated clauses.

**Supplement:** Negation frequency and position distribution. Per-genre breakdown. Comparison with child language literature rates (if available from CHILDES metadata).

### §9: Evaluation Context Frequency

**Main text:** A table listing each evaluation category with its corpus frequency (count per 90M tokens) by language. Categories flagged as rare (<100 occurrences) are highlighted.

**Supplement:** Per-genre breakdown. Example sentences for each category.

### §10: Wh-Questions

**Main text:** Subject vs. object wh-question frequency by language. Embedded question frequency. Overlap with §4 (proportion of wh-questions that involve complement clause extraction).

**Supplement:** Wh-word lemma frequency table. Per-genre breakdown.

### Figure and Table Numbering Convention

- **Tables 1–3:** Corpus overview (§1), pronoun distribution summary (§2), manipulation feasibility (§7)
- **Figure 1:** PMI distribution (§3)
- **Table 4:** That-trace cross-tabulation (§4)
- **Tables 5–6:** Expletive and morphology summaries (§5, §6)
- **Table 7:** Negation × subject realization (§8)
- **Table 8:** Evaluation context frequency (§9)
- **Remaining figures/tables:** As needed for supplement

This numbering follows the order of presentation in the methods section, which mirrors the hypothesis order. The corpus descriptives appear after the design overview and before the results, establishing the input properties that the manipulations target.
