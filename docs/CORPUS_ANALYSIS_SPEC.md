> **Superseded.** This is the original corpus analysis specification. The current revised specification is [NEW_CORPUS_ANALYSIS_PLAN.md](NEW_CORPUS_ANALYSIS_PLAN.md).

# Corpus Descriptive Analysis Specification (superseded)

## Overview

Parallel descriptive analysis of English (BabyLM) and Italian (bebe-lm) corpora to characterize input distributions relevant to null-subject acquisition.

---

## Phase 1: Immediately Feasible (Standard NLP Pipeline)

These analyses use spaCy dependency parsing, POS tagging, and morphological features.

### 1.1 Overt Pronoun Inventory

**Subject pronouns**
- English: I, you, he, she, it, we, they
- Italian: io, tu, lui/lei, noi, voi, loro
- UD features: `dep=nsubj`, `pos=PRON`

**Object pronouns**
- English: me, you, him, her, it, us, them
- Italian: mi, ti, lo/la, ci, vi, li/le (clitics)
- UD features: `dep=obj/iobj`, `pos=PRON`

**Person breakdown**
- Categories: 1sg, 2sg, 3sg, 1pl, 2pl, 3pl
- UD features: `Person`, `Number` feats

**Case marking**

- English: nom/acc distinction
- Italian: less morphological case
- UD features: `Case` feat

**Output:** Frequency tables by person/number/case/function; proportions relative to all tokens and all pronouns.

### 1.2 Expletives

**Expletive subjects**
- English: it, there with `dep=expl`
- Italian: null — count verbs that would take expletives

**By verb class**
- Weather verbs, raising verbs, existentials (both languages)

**Output:** Expletive counts; verb lemmas that appear with expletives.

### 1.3 Verb Finiteness

**Finite verbs:** `VerbForm=Fin`

**Infinitival verbs:** `VerbForm=Inf`

**By clause position:** `dep=ROOT` vs `dep=xcomp/ccomp/advcl/acl`

**Output:** Finite/infinitival counts; cross-tabulation with clause type.

### 1.4 Clause Structure (Dependency-Based)

**Root finite verbs** — `dep=ROOT` + `VerbForm=Fin`

- with overt subject: has `nsubj` or `nsubj:pass` child
- with expletive: has `expl` child
- without subject: no subject dependent (candidate null-subject) (you don't need to reconstruct to know whether the subject exists)

**Subordinate finite clauses** — `dep=ccomp/advcl/acl:relcl` + `VerbForm=Fin`

- same subject presence breakdown

**Infinitival complements** — `dep=xcomp` + `VerbForm=Inf`

- control vs raising: classified by matrix verb lemma

**Output:** Counts and proportions for each cell; cross-tabulation.

### 1.5 Wh-Questions

**Subject wh-questions:** wh-word is `nsubj` of root verb

**Object wh-questions:** wh-word is `obj` of root verb

**Embedded questions:** `dep=ccomp` with wh-word

**That/che presence:** complementizer in embedded wh-clause

**Detection:** `POS=PRON` or `POS=ADV` with `PronType=Int` or lemma in {who, what, which, where, when, why, how} / {chi, che, cosa, quale, dove, quando, perché, come}.

**Output:** Subject/object extraction counts; that-trace environment frequency.

### 1.6 Relative Clauses

**Subject relatives:** relativizer is `nsubj` in `acl:relcl`

**Object relatives:** relativizer is `obj` in `acl:relcl`

**Resumptive pronouns:** pronoun coreferent with head in relative clause

**Output:** Subject/object relative counts; resumptive rate (if detectable).

### 1.7 Negation

**Negated clauses:** has `advmod` or `dep` with `Polarity=Neg`

**Negation position:** token index relative to finite verb

**Subject realization:** cross-tab negated × subject presence

**Output:** Negation frequency; positional distribution; subject realization in negated clauses.

### 1.8 That-Trace Environments

The that-trace effect is a constraint on subject extraction across an overt complementizer. In English, extracting a subject from a finite complement clause is grammatical only when "that" is absent: "Who do you think (*that) will win?" In Italian, no such restriction holds: "Chi pensi che vincerà?" is grammatical. This asymmetry is one of the phenomena that clusters with the null-subject parameter, making it a key test for distal effects (H5).

Naturalistic corpora will not contain that-trace *violations* (they are ungrammatical in English). Instead, we characterize the *environments* in which the constraint is relevant — that is, the distribution of complement clauses, complementizer presence, and extraction type that together define the learner's exposure to that-trace-relevant contexts.

#### Step 1: Identify complement clauses under bridge verbs

Find all finite clauses with `dep=ccomp` whose matrix verb is a bridge verb — a verb that permits extraction from its complement. Bridge verbs are identified by lemma from a curated list.

**English bridge verbs:** think, believe, say, know, assume, expect, hope, suppose, claim, report, imagine, feel, suspect, guess, figure, reckon, suggest, declare, announce, predict

**Italian bridge verbs:** pensare, credere, dire, sapere, supporre, sperare, immaginare, sentire, ritenere, affermare, dichiarare, sostenere, annunciare, prevedere

These lists are not exhaustive but cover the high-frequency cases. They can be refined empirically: any verb that appears with `ccomp` and a fronted wh-word in the corpus is a candidate bridge verb.

#### Step 2: Classify complementizer presence

For each `ccomp` clause under a bridge verb, check whether the embedded verb has a `mark` dependent with lemma "that" (English) or "che" (Italian). This yields two conditions:

- **+Comp:** Complementizer present ("I think *that* he will win" / "Penso *che* vincerà")
- **-Comp:** Complementizer absent ("I think he will win" / "Penso vincerà")

The rate of optional "that"/"che" is itself informative: English speakers drop "that" frequently, and this optionality is what creates the learning environment for the that-trace constraint.

#### Step 3: Classify the subject status of the embedded clause

For each `ccomp` clause, check the subject of the embedded verb:

- **Overt lexical subject:** has `nsubj` dependent that is a noun/proper noun
- **Overt pronominal subject:** has `nsubj` dependent that is a pronoun
- **No overt subject:** no `nsubj` dependent — could indicate extraction, null subject (Italian), or imperative/fragment

#### Step 4: Identify extraction contexts

Extraction from a complement clause produces a sentence where a wh-word in the matrix clause is semantically the subject or object of the embedded verb. In UD annotations, long-distance dependencies are not always reliably represented, so we use a heuristic:

**Subject extraction heuristic:**
- The matrix clause contains a wh-word (lemma in {who, what, which} / {chi, che, cosa, quale})
- The wh-word is `nsubj` of the matrix verb OR is `ROOT` of the sentence
- The embedded `ccomp` verb has no `nsubj` dependent
- This pattern identifies sentences like "Who do you think __ will win?"

**Object extraction heuristic:**
- The matrix clause contains a wh-word
- The embedded `ccomp` verb has an `nsubj` dependent but is missing an expected `obj`
- OR the wh-word is parsed as `obj` of the embedded verb (long-distance parse)
- This pattern identifies sentences like "What do you think the team will win __?"

These heuristics will produce false positives (not every missing subject is extraction) and false negatives (parsers may mishandle long-distance dependencies). This is acceptable for corpus characterization — we are estimating distributions, not annotating individual sentences.

#### Step 5: Cross-tabulate

The key cross-tabulation is **complementizer presence × extraction type**:

- +Comp, subject extraction — the that-trace violation environment (should be near-zero in English, attested in Italian)
- +Comp, object extraction — grammatical in both languages
- +Comp, no extraction (declarative complement) — grammatical in both languages
- -Comp, subject extraction — grammatical in both languages
- -Comp, object extraction — grammatical in both languages
- -Comp, no extraction (declarative complement) — grammatical in both languages

**Output:**
- Bridge verb frequency (by lemma)
- Complementizer rate in complement clauses (overall and by bridge verb)
- Extraction type distribution (subject / object / none)
- Cross-tabulation of complementizer × extraction type × language
- Per-genre breakdown (for BabyLM)

#### Limitations

- spaCy does not reliably annotate long-distance dependencies, so extraction detection is heuristic-based
- The distinction between a missing subject due to extraction vs. null subject vs. parse error requires judgment; the heuristics above will be validated against a manually annotated sample
- Some bridge verbs also take interrogative complements ("I wonder who will win") which are `ccomp` but not extraction from a declarative — these should be filtered by checking whether the wh-word is a dependent of the embedded verb rather than the matrix verb

### 1.9 Genre Analysis (BabyLM)

**Sources and expected registers:**
- CHILDES — Child-directed speech
- BNC — Mixed written/spoken British
- Gutenberg — Literary prose
- OpenSubtitles — Informal dialogue
- Simple Wikipedia — Expository
- Switchboard — Conversational

**Output:** All above measures broken down by source.

---

## Phase 2: Pronoun Recovery Model

These analyses require reconstructing full-pronoun text — inserting overt subject pronouns where they have been dropped. This serves two purposes: (1) enabling the null-subject descriptive analysis, and (2) producing pronoun-inserted text for corpus ablation experiments (e.g., inserting overt pronouns into Italian to test H2).

Separate models are trained for English and Italian. The task is fundamentally different in each language: Italian has systematic pro-drop recoverable from verbal morphology; English has rare, context-specific drops (diary drop, imperatives, topic continuity) where morphology is impoverished.

### 2.1 Architecture: Two-Stage Pipeline

#### Stage 1: Detection and Feature Classification

A sequence labeling model that operates over verb tokens and predicts a structured label for each finite verb lacking an overt subject.

**Label set:**
- `NONE` — no pronoun is missing (verb has an overt subject, or is non-finite with controlled PRO)
- `PRO.1sg` / `PRO.2sg` / `PRO.3sg` / `PRO.1pl` / `PRO.2pl` / `PRO.3pl` — a subject pronoun with these features was dropped
- `IMP` — imperative (understood 2nd person, not a dropped pronoun)
- `CONJ` — conjunction reduction (shared subject with prior conjunct, not a dropped pronoun)

The distinction between `PRO.*`, `IMP`, and `CONJ` matters for the descriptive analysis: only `PRO.*` labels count as true null subjects.

**Why classification over seq2seq for Stage 1:** The label set is small and closed. The model doesn't need to generate text — it needs to make a decision at each verb. This is a simpler, more constrained task with less room for hallucination. It also directly outputs the features needed for descriptive analysis (person/number counts by context).

#### Stage 2: Lexical Realization and Insertion

Given the feature labels from Stage 1, reconstruct the full-pronoun text by inserting the appropriate lexical pronoun at the correct position.

**Feature-to-pronoun mapping (deterministic except 3sg):**
- 1sg → English: "I" / Italian: "io"
- 2sg → English: "you" / Italian: "tu"
- 3sg → English: "he"/"she"/"it" / Italian: "lui"/"lei" (requires gender resolution)
- 1pl → English: "we" / Italian: "noi"
- 2pl → English: "you" / Italian: "voi"
- 3pl → English: "they" / Italian: "loro"

**Gender resolution for 3sg:** The only case requiring context beyond morphology. Resolved by:
- Dependency parse: look for a coreferent NP in the prior clause or sentence; use its grammatical gender (Italian) or semantic gender (English)
- Fallback: Italian verb agreement on past participles encodes gender ("è andat**a**" → feminine); English defaults to "they" if ambiguous
- For `IMP` and `CONJ` labels: no insertion (these are not dropped pronouns)

**Insertion position:**
- English: immediately before the finite verb
- Italian: immediately before the finite verb (canonical preverbal position; postverbal overt subjects are marked but the recovery task produces the unmarked order)

### 2.2 Training Data Construction

#### Step 1: Synthetic gold data via pronoun dropping

Take sentences from the corpus that DO have overt subject pronouns. Programmatically drop them using the existing `remove_subject_pronominals.py` pipeline. This creates aligned pairs:

- Input: "Parla italiano molto bene."
- Target: "Lei parla italiano molto bene."

This provides unlimited training data with perfect labels, since we know exactly which pronoun was dropped and what its features are.

**Advantage:** No annotation cost, perfect alignment, unlimited scale.

**Risk:** Domain mismatch — naturally dropped subjects may occur in different contexts than artificially dropped ones. Italian speakers drop subjects in nearly all contexts, so the mismatch is small. English has fewer natural drops, so the artificial data covers the space well.

#### Step 2: Manual annotation seed set

Sample 1000 sequences from the corpus at the expected batch size. From these, select 10-20 sequences that contain naturally occurring null subjects (especially for Italian). A fluent speaker manually annotates these by inserting the dropped pronouns with feature labels.

**Annotation format:**
- Original: "Parla italiano. Vuole imparare il francese."
- Annotated: "[PRO.3sg:lei] Parla italiano. [PRO.3sg:lei] Vuole imparare il francese."

This captures both the feature tag and the lexical realization, providing the gold standard for the task.

#### Step 3: LLM-scaled annotation via DeepSeek

Use the 10-20 manually annotated sequences as few-shot examples to prompt DeepSeek (or comparable frontier model with strong Italian performance). Have DeepSeek annotate the remaining ~980 sequences from the sample.

**Prompt structure:**
- System: "You are a linguist annotating null subjects. For each finite verb that lacks an overt subject pronoun, insert the dropped pronoun in brackets with its person/number features."
- Few-shot: the 10-20 manually annotated examples
- Input: unannotated sequence
- Output: annotated sequence with `[PRO.Xsg/pl:lexical_form]` markers

#### Step 4: Validation

Manually review a sample of 50-100 DeepSeek-annotated sentences to estimate:
- Precision: what fraction of inserted pronouns are correct?
- Recall: what fraction of true null subjects were identified?
- Feature accuracy: when a pronoun is correctly identified, is the person/number/gender correct?

Target: 90%+ on all three metrics before proceeding to model training.

#### Step 5: Model training

Combine the synthetic data (Step 1) with the validated LLM-annotated data (Steps 3-4) to train the pronoun recovery model.

**Training strategy:**
- Pre-train on synthetic data (abundant, clean, but potentially mismatched distribution)
- Fine-tune on LLM-annotated data (smaller, closer to natural null-subject distribution)
- Evaluate on held-out manually annotated examples

**Model candidates:**
- Encoder-only + classification head (e.g., fine-tuned BERT/DeBERTa) — natural fit for Stage 1 (sequence labeling)
- Seq2seq (e.g., T5/mT5) — can handle both stages end-to-end if preferred, outputting full reconstructed text
- Decision to be made based on pilot experiments

### 2.3 Descriptive Analysis (Enabled by Recovery)

Once the pronoun recovery model is validated, the following analyses become possible:

**Null subject identification**

A finite verb without an `nsubj` dependent could be:
- True null subject (pro-drop) — labeled `PRO.*` by the recovery model
- Imperative — labeled `IMP`
- Conjunction reduction — labeled `CONJ`
- Parse error — detectable by comparing recovery model output with dependency parse

**Null subject contexts**

- Diary drop: `PRO.1sg` at utterance-initial position
- Imperatives: `IMP` label
- 2nd person questions: `PRO.2sg` or `PRO.2pl` in interrogative context
- Topic continuity: `CONJ` label, or `PRO.*` matching features of prior subject
- Postverbal subjects: cases where the dependency parse shows a postverbal `nsubj` — these are NOT null subjects but have non-canonical word order (primarily Italian)

**Null subject rate by context**
- Null rate = `PRO.*` labels / (`PRO.*` + overt pronouns) per context type
- Breakdown by person/number
- Comparison English vs Italian
- Per-genre breakdown (for BabyLM)

**Reconstructed text uses**
- Input to ablation experiments: insert overt pronouns into Italian corpus (H2)
- Counterfactual corpora: what would English look like with systematic pro-drop?
- Validation: compare reconstructed text against original overt-subject sentences

---

## Implementation Notes

### Tools

- spaCy with `en_core_web_trf` / `it_core_news_lg` (or trf if available)
- Existing preprocessing infrastructure in `preprocessing/`
- DeepSeek API for scaled annotation
- HuggingFace Transformers for model training (Stage 1 classifier or seq2seq)

### Output Format

- JSON/CSV frequency tables
- Aggregated statistics
- Per-sentence annotations for downstream analysis
- Reconstructed full-pronoun text files (parallel to original corpus)

---

## Status

- [ ] Phase 1 implementation
- [ ] Phase 2 Step 1: synthetic training data generation
- [ ] Phase 2 Step 2: manual annotation seed set
- [ ] Phase 2 Step 3: DeepSeek-scaled annotation
- [ ] Phase 2 Step 4: validation
- [ ] Phase 2 Step 5: model training
- [ ] Phase 2 descriptive analysis
