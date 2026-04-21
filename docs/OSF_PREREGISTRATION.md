# OSF Preregistration

## Title

Comparing Generative Linguistics and Information Theoretic Accounts of Subject Drop in English and Spanish Using Statistical Language Models

---

## Description

### The Null Subject Question

A majority of the world's languages allow speakers to omit the subject of sentences. In Spanish, "Habla español" (speaks Spanish) is a complete, grammatical sentence; in English, "*Speaks Spanish" is not. This typological split has been a central question in linguistics for over forty years: does it reflect a deep grammatical distinction, a surface-level frequency effect, or something in between?

### Two Traditions

One tradition, rooted in theoretical linguistics, argues that null-subject licensing is part of a cluster of grammatical properties that pattern together across languages: null subjects, free subject-verb inversion, restrictions on extraction (the "that-trace effect"), and verbal agreement patterns. This clustering suggests that what learners acquire is not simply a pattern of pronoun use, but a grammatical system with interconnected parts. Under this view, the evidence children use is often indirect: the presence of expletive subjects like "it" and "there," patterns in verbal morphology, and case distinctions on pronouns all provide information about the underlying grammatical system.

The theoretical apparatus developed to explain this clustering is substantial. It typically requires assumptions about innate linguistic knowledge that constrains the space of possible grammars, allowing learners to acquire adult-like systems from limited input. These assumptions have made this tradition controversial: the commitment to domain-specific innate knowledge is difficult to test directly and remains contentious.

An alternative tradition proposes that learners make inferences about language from its statistical properties in context. Under such accounts, speakers are sensitive to the relationship between pronouns and their recoverability: when pronouns are dropped in contexts where they're predictable, this provides evidence for a grammar permitting omission; when pronouns are retained despite being predictable, this provides evidence for a grammar requiring overt subjects. This tradition has the virtue of parsimony (it requires no domain-specific innate knowledge) but struggles to explain why phenomena that are distributionally independent (null subjects, that-trace effects) should pattern together grammatically.

### Controlled Rearing with Language Models

Adjudicating between these accounts with human subjects is difficult. We cannot experimentally manipulate the input children receive, create counterfactual languages, or isolate specific evidence types. Language models offer a methodological alternative: by training models on corpora with systematic manipulations, we can conduct causal investigations into learnability that are impossible with human learners.

Language models are not models of human cognition. But they are learners that acquire grammatical knowledge from distributional evidence alone, without innate linguistic structure. This makes them useful for testing which aspects of grammatical knowledge can emerge from learning versus which might require something more.

### Why Distal Effects Might Emerge Without Innate Structure

There is reason to expect that learners, even those without domain-specific linguistic knowledge, might converge on structured grammatical representations. Recent work in machine learning has documented a striking phenomenon: diverse learning systems trained on different data with different architectures tend to converge toward similar internal representations. This has been termed the "Platonic Representation Hypothesis" (Huh et al., 2024): the idea that there exists a shared statistical structure underlying reality, and that sufficiently powerful learners will converge on representations that capture this structure regardless of their starting point.

Applied to language, this suggests a reframing of classic debates about universals. The questions asked by Universal Grammar research (What grammatical distinctions are possible? What phenomena cluster together? What evidence is sufficient for acquisition?) were the right questions. The extensive cross-linguistic documentation of grammatical patterns was genuine scientific discovery. What may have been wrong was the assumption that these patterns require innate specification.

If language has inherent structure (regularities in how meaning maps to form, how dependencies are organized, how information is distributed) then learners optimizing for prediction or comprehension will be pushed toward representations that capture this structure. The "universals" would be attractors in the space of possible learned representations: not arbitrary conventions, not innate endowment, but inevitable consequences of learning systems operating over structured input.

This view preserves what was valuable in theoretical linguistics: the careful documentation of cross-linguistic patterns, the insight that grammatical phenomena are interconnected. It relocates the source of these patterns from biology to the structure of language itself and the dynamics of learning. If language models show distal effects, this supports such a view: the clustering of grammatical phenomena is real and discoverable by learners, even learners with no built-in linguistic knowledge.

Of course, language models differ from children in many ways: they receive vastly more data, lack embodiment and social interaction, and optimize for different objectives. Findings from models constrain but do not determine theories of human acquisition. What models can show is what is learnable in principle from distributional evidence, and this is precisely what's at stake in debates about the necessity of innate linguistic structure.

### Core Research Questions

This study asks three fundamental questions:

**Question 1: Can models learn this?**
Can language models, trained on naturalistic input, learn the target grammatical patterns? Specifically, can English-trained models learn that overt subjects are required, and can Spanish-trained models learn that null subjects are permitted (and in most contexts preferred)? This establishes baseline learnability.

**Question 2: Do models mis-learn when evidence is altered?**
When we systematically manipulate the input—removing pronouns, adding pronouns, altering morphology—do models develop incorrect grammatical preferences? This tests the causal role of specific evidence types: if removing pronouns causes models to accept null subjects in English, this demonstrates that pronoun distribution is necessary for learning the target grammar.

**Question 3: Do models show child-like developmental trajectories?**
English-acquiring children initially over-accept null subjects before learning they are ungrammatical. Spanish-acquiring children do not show this pattern because null subjects are grammatical in Spanish. Do language models show the same developmental asymmetry? Early preference for null subjects in English (that is later overcome) but not in Spanish would suggest that models and children face similar learning problems and arrive at similar solutions.

### Theoretical Stakes

Beyond these empirical questions, the pattern of results bears on deeper theoretical issues. If manipulating one aspect of the input (pronoun distribution) affects not only pronoun acceptability but also judgments about syntactically-related phenomena (that-trace effects, extraction, embedded subjects), this would suggest grammatical knowledge involves structured representations—even in learners without innate linguistic structure.

**Local effects only** would suggest grammatical knowledge is built from surface statistics; phenomena that don't co-occur in training are learned independently.

**Distal effects** would suggest grammatical knowledge involves interconnected representations where syntactically-related phenomena cluster together, as documented across human languages.

### Design Overview

We train models on English and Spanish corpora with targeted manipulations: removing expletives, manipulating verbal morphology, neutralizing case distinctions, and inserting recovered pronouns in Spanish contexts where they were omitted. We evaluate models not only on null-subject acceptability, but on the full set of related phenomena: that-trace effects, extraction, subordinate clauses, control constructions, and Spanish-specific phenomena (postverbal subjects, impersonal `se`, clitic climbing). The pattern of results (local vs. distal) will inform our understanding of how grammatical knowledge is organized and acquired.

Pronoun-distribution sweep manipulations (random and PMI-based removal at 0–100%) are not part of the initial wave and are reserved for a follow-up study.

---

## Contributors

- **Thomas Morton** — UCSD Psychology
- **Ben Bergen** — UCSD Cognitive Psychology
- **Victor Ferreira** — UCSD Psychology
- **Alex Warstadt** — UCSD Linguistics

---

## Tags

null subjects, pro-drop, language acquisition, language models, controlled rearing, grammatical knowledge, English, Spanish (Peninsular), cross-linguistic, BabyLM, BebeLM

---

## Hypotheses

### Core Hypotheses

**H1: Baseline Learnability (Can models learn this?)**

Models trained on unmanipulated corpora will acquire the target grammatical patterns:
- **English models** will prefer overt subjects over null subjects (accuracy above chance on null-subject minimal pairs)
- **Spanish models** will accept null subjects (accuracy above chance, with preference for null over overt in appropriate topic-continuity contexts per the Position of Antecedent Hypothesis)

This establishes that the target grammar is learnable from naturalistic input before testing the effects of manipulations.

---

**H2: Evidence Manipulation (Do models mis-learn when evidence is altered?)**

Manipulating the input will cause models to acquire incorrect grammatical preferences:
- **English models** trained without expletive-containing sentences will show reduced preference for overt subjects in the contexts where expletives signalled the requirement (H6a test).
- **English models** with impoverished verbal morphology (lemmatized verbs) will show reduced cues against null subjects; with synthetic enriched morphology, cues supporting null subjects will increase.
- **Spanish models** trained with inserted overt pronouns (recovered via the pronoun-recovery pipeline) will show reduced null-subject acceptance (preference shift toward overt subjects).
- **Spanish models** with impoverished morphology will show reduced null-subject licensing.

| Manipulation | Predicted Effect |
|--------------|------------------|
| Remove expletive sentences (English) | ↑ Null subject acceptance (reduced positive evidence for obligatory Spec,TP) |
| Impoverish verbal morphology (English) | Minor shift (English already morphologically impoverished) |
| Insert verbal morphology (English) | ↑ Null subject acceptance (simulated rich agreement) |
| Impoverish case (English) | ↑ Null subject acceptance (H6c) |
| Insert pronouns (Spanish) | ↓ Null subject acceptance (mis-learning) |
| Impoverish verbal morphology (Spanish) | ↓ Null subject acceptance (loss of agreement cues) |
| Impoverish case (Spanish) | ↓ Null subject acceptance (H6c) |

This tests whether the target evidence types are *causally necessary* for learning the target grammar.

*Pronoun-distribution sweep manipulations (H4, random and PMI-based removal at 0–100%) are deferred to a follow-up study; see H4.*

---

**H3: Developmental Trajectory (Do models show child-like learning patterns?)**

Models will show developmental patterns paralleling child language acquisition:
- **English models** will show early preference for null subjects that is overcome during training (matching the pattern in English-acquiring children)
- **Spanish models** will NOT show early null-subject preference followed by reversal (matching Spanish-acquiring children, for whom null subjects are grammatical throughout)

| Language | Child Pattern | Predicted Model Pattern |
|----------|--------------|------------------------|
| English | Early null-subject acceptance → learned rejection | Early null preference → learned overt preference |
| Spanish | Consistent null-subject acceptance | Consistent null preference (no reversal) |

This tests whether models face the same learning problem as children and arrive at similar developmental solutions.

---

**H4: Informativity-Based vs. Random Removal (deferred)**

*This hypothesis is deferred to a follow-up study. The infrastructure for pronoun sweep manipulations is scoped but not part of the initial wave of experiments preregistered here. Documented as a planned extension, not an active hypothesis in the current run.*

Does targeting pronouns by their contextual predictability (pointwise mutual information) produce different effects than random removal at matched rates?

| Pattern | Implication |
|---------|-------------|
| **Informativity matters** | Learning is sensitive to the information-theoretic properties of individual tokens |
| **No difference** | What matters is the aggregate statistical pattern, not the specific properties of removed items |

*Both PMI-based and random removal sweeps (0–100%, 10% increments) will be conducted in the follow-up.*

---

### Secondary Hypotheses: Local vs. Distal Effects

**H5: Distal Effects (Theoretical Test)**

Does manipulating pronoun distribution affect only pronoun-related judgments, or does it also affect judgments about syntactically-related but distributionally-independent phenomena (that-trace effects, extraction, embedded subjects)?

| Pattern | Implication |
|---------|-------------|
| **Local effects only** | Grammatical knowledge is built from surface statistics; phenomena are learned independently |
| **Distal effects** | Grammatical knowledge involves structured representations; syntactically-related phenomena are interconnected |

*Distal effects would demonstrate that learners, even without innate linguistic structure, converge on interconnected grammatical representations.*

---

### Secondary Hypotheses: Indirect Evidence

These test whether models are sensitive to evidence types that linguistic theory identifies as relevant to null-subject licensing, beyond direct pronoun distributions:

**H6a: Expletive Removal**
Removing expletive-containing sentences will facilitate null-subject preferences.
- *Structured learning prediction*: Effect (expletives signal overt Spec,TP requirement)
- *Statistical learning prediction*: Weak/no effect (expletives are not informative about referential pronoun distribution)

**H6b: Verbal Morphology Manipulation**
Adding rich agreement morphology to English will facilitate null subjects; impoverishing Spanish agreement will push toward overt subjects.
- *Structured learning prediction*: Effect (morphological uniformity affects grammatical licensing)
- *Statistical learning prediction*: Weak/no effect (morphology is indirect)

**H6c: Case Neutralization**
Removing morphological case distinctions on pronouns will facilitate null-subject preferences.
- *Structured learning prediction*: Effect (removes dependent-case evidence for obligatory subjects)
- *Statistical learning prediction*: Weak/no effect (case is not directly relevant to pronoun frequency)

**H6d: Stacked Ablations**
Combined removal of expletives + morphology manipulation + case neutralization will produce synergistic effects.
- *Structured learning prediction*: Synergistic (these are independent factors affecting grammatical structure)
- *Statistical learning prediction*: Additive at most (no structural interaction)

---

### Architectural Hypotheses

**H7: N-gram vs. Deep Models (Surface vs. Structured Learning)**

N-gram models capture local co-occurrence statistics but lack capacity for structured representations. If deep models (transformers, LSTMs, Mamba) show distal effects that n-gram models do not, this is evidence that grammatical knowledge requires representations beyond surface statistics.
- *Prediction*: N-grams will show local effects only; deep models will show distal effects
- *Tests both*: (a) whether early deep model learning resembles n-gram patterns, (b) whether final performance diverges

**H8: Architectural Convergence (Platonic Representation Hypothesis)**

If grammatical structure emerges from the data rather than architecture-specific inductive biases, different deep learning architectures (GPT, BERT, LSTM, Mamba) should converge on similar patterns of grammatical sensitivity.
- *Prediction*: Qualitatively similar patterns across deep architectures
- *Quantitative differences expected*: Speed of learning, magnitude of effects
- *Implication*: Convergence supports the view that structured representations are attractors that diverse learners discover

**H9: GPT Scaling**

Larger models may learn structured representations faster or show stronger effects, but should converge on qualitatively similar patterns.
- *Prediction*: Quantitative differences (small < medium < large in speed/magnitude)
- *Not predicted*: Qualitative differences in the pattern of distal effects

**H10: Mamba (Exploratory)**

State space models represent a distinct computational paradigm. Including Mamba tests whether architectural convergence extends beyond attention-based and recurrent models.
- *Prediction*: Exploratory, but convergence would further support PRH

---

## Study Type

*Please check one of the following statements:*

- [x] **Experiment** — A researcher randomly assigns treatments to study subjects, this includes field or lab experiments. This is also known as an intervention experiment and includes randomized controlled trials.
- [ ] **Observational Study** — Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, "natural experiments," and regression discontinuity designs.
- [ ] **Meta-Analysis** — A systematic review of published studies.
- [ ] **Other**

---

## Blinding

*Blinding describes who is aware of the experimental manipulations within a study. Mark all that apply.*

- [x] No blinding is involved in this study.
- [ ] For studies that involve human subjects, they will not know the treatment group to which they have been assigned.
- [ ] Personnel who interact directly with the study subjects (either human or non-human subjects) will not be aware of the assigned treatments. (Commonly known as "double blind")
- [ ] Personnel who analyze the data collected from the study are not aware of the treatment applied to any given group.

**Is there any additional blinding in this study?**

This is a computational experiment. Evaluation is automated via model probability comparisons (SLOR), with no human judgment involved in data collection.

---

## Study Design

This study investigates learning in English and Spanish language models by performing targeted manipulations on baseline corpora. We train multiple model architectures on each manipulated dataset, saving checkpoints throughout training to track learning dynamics. The experimental design crosses architecture with data manipulation, yielding a between-subjects matrix with repeated measures across evaluation tasks.

### Training Replication

Each experimental condition (architecture × language × ablation) is trained with 10 random initializations to assess training variability. This allows us to distinguish signal from noise in model learning trajectories.

### Datasets

- **English**: BabyLM corpus (90M tokens training, 10M held out for evaluation).
- **Spanish**: BebeLM Spanish corpus — a custom 100M-word Peninsular Spanish corpus assembled from 10 sources (CHILDES Spanish, child_narratives, GRERLI school-age transcripts, CORLEC spoken, Vikidia, QED educational subtitles, OpenSubtitles, Europarl, Leipzig Web news, Spanish Gutenberg). Split into 90M training / 10M pull (held-out for replacement-pool backfill during ablations) / 10M test. See `docs/spanish_corpus.md` for the full breakdown. Parallel EN-ES corpora (11 OPUS sources, 95M pairs) support pronoun-recovery training for the Insert Pronouns intervention.

### Checkpoint Schedule

Each model saves 40 checkpoints over the course of training, spaced in log-time. This captures learning dynamics with higher resolution early in training (where rapid changes occur) and lower resolution later (where learning plateaus). Checkpoints are matched across architectures as closely as possible by optimizer steps, though some architectures (e.g., n-grams) do not use gradient-based optimization.

### Factor Structure

The study crosses two dimensions:

1. **Architecture**: n-gram (1-5), GPT-2 (small, medium, large), BERT, LSTM, Mamba
2. **Data manipulation**: Baseline plus targeted ablations / enrichments (expletive removal, verbal-morphology manipulation, case neutralization, pronoun insertion for Spanish)

This yields a between-subjects matrix with within-model repeated measures across evaluation tasks. The factor chart, split as Model × Intervention, is fully enumerated below. Pronoun-sweep conditions (H4) and determiner interventions (remove/impoverish determiners) are not part of the current wave; see the follow-up note below the Spanish list.

### English

**Model:** [n-gram: 1-, 2-, 3-, 4-, 5-gram; GPT: small, medium, large; BERT: large; LSTM; and Mamba: 370m]

**Interventions:**
- **Baseline** — unmanipulated BabyLM 90M training corpus.
- **Remove Expletives** — sentences containing expletive constructions are removed entirely from the corpus; replacement-pool backfill preserves corpus size. Three-tier detection: spaCy `dep_ == 'expl'` (existential-*there*), heuristic weather-*it* / raising-*it* / copula+raising-adjective constructions, and coreference confirmation to keep referential *it*.
- **Impoverish Verbal Morphology** — all verbs collapsed to their lemma, eliminating tense/agreement cues (e.g., *walks*, *walked*, *walking* → *walk*).
- **Insert Verbal Morphology** — synthetic Latin-inspired agreement suffixes added to verbs based on the detected subject's person and number (1sg `-o`, 2sg `-as`, 3sg `-at`, 1pl `-amus`, 2pl `-atis`, 3pl `-ant`). Enrichment, not impoverishment — simulates a richly-inflected agreement paradigm in an otherwise English corpus.
- **Impoverish Case** (H6c) — all non-nominative pronoun forms (oblique, possessive, reflexive) collapsed to their nominative equivalent (*me/my/mine/myself* → *I*; *him/his/himself* → *he*; etc.).

### Spanish

**Model:** [n-gram: 1-, 2-, 3-, 4-, 5-gram; GPT: small, medium, large; BERT: large; LSTM; and Mamba: 370m]

**Interventions:**
- **Baseline** — unmanipulated BebeLM Spanish 90M training corpus.
- **Remove Expletives** — sentences containing expletive-equivalent constructions are removed entirely; replacement-pool backfill preserves corpus size. Spanish has no overt expletive pronouns in most contexts, so detection targets: weather verbs (*llover*, *nevar*), existential *haber* (*hay*, *había*), impersonal raising verbs (*parecer*, *resultar*, *suceder*) with clausal complement, impersonal necessity verbs (*bastar*, *convenir*), and overt *ello* as subject of any of the above (archaic literary form).
- **Impoverish Verbal Morphology** — all verbs collapsed to their lemma, eliminating the rich person/number/tense/mood morphology that Spanish uses to license null subjects.
- **Impoverish Case** (H6c) — non-nominative pronoun forms (tonic obliques *mí/ti/sí*, preposition-bound portmanteaux *conmigo/contigo/consigo*, accusative and dative clitics *me/te/lo/la/nos/os/los/las/le/les*, possessives *mi/tu/su/nuestro/vuestro* short and *mío/tuyo/suyo* long forms) collapsed to their nominative equivalent.
- **Insert Pronouns** — the reverse manipulation: use a trained pronoun-recovery model (tree detector + insertion module, trained on the Spanish-English parallel corpora at `data/spanish/parallel/`) to insert appropriate overt subject pronouns into finite clauses where they were omitted in the training corpus. This simulates an over-pronominal input approximating English distribution.

*Deferred to a follow-up study*: pronoun-distribution sweeps (All-Evidence+Remove Pronouns 0–100% step 10; no-other-evidence+pronoun_sweep). Determiner interventions (Remove Determiners, Impoverish Determiners) from earlier study plans are not part of the current preregistered wave.

---

## Randomization

As the language model state is reset between evaluation sets, there is no need to randomize the order of the stimuli. The initial model state weights, as well as the order that data is presented in each epoch is randomized, although this randomization is controlled by a seed number so that the same random seed can be used to replicate model training if the same training pipeline is used.

---

## Existing Data

**Registration prior to creation of data.** As of the date of submission of this research plan for preregistration, the data have not yet been collected, created, or realized.

---

## Explanation of Existing Data

N/A — No existing data will be used in this study.

---

## Data Collection Procedures

While the model is trained, training loss is collected with the Weights & Biases system (along with other diagnostic information), this training loss represents the model's fit to its training data. When a model is saved at the scheduled training step (one of 40 across training), all of its weights at that step are frozen and copied into a model checkpoint. After model training, each of those model checkpoints are evaluated on a suite of evaluation tests. The first of these is model perplexity on a held-out (10M words) corpus of training data, used to evaluation out-of-distribution model fit.

Then, the model is evaluated on its overall grammatical performance using BLiMP (English) and MultiBLiMP (Spanish), benchmarks of linguistic minimal pairs that test the model's preference on comparisons of grammatical and ungrammatical linguistic sentences to assess alignment with human judgements.

In addition the model will be evaluated on a dedicated suite of minimal pairs designed to target the model's preference on specific grammatical contexts relevant to the production of null and overt subjects and objects. This evaluation set has **two layers**:

1. **Eight cross-linguistic categories** with item-paired English and Spanish counterparts (`subject_drop`, `object_drop`, `embedded_drop`, `control`, `expletive`, `conjunction`, `extraction`, `subject_drop_no_agreement`). Item-paired means for every English item there is a Spanish item with the same `item_id`, the same discourse structure, and the analogous hotspot alternation. This enables both within-language evaluation and cross-linguistic structural-priming analyses.
2. **Three Spanish-only categories** that test null-subject cluster properties without English analogs: `postverbal_subject` (SV vs VS word order), `se_impersonal` (impersonal *se* vs 3pl indefinite or overt subject), and `clitic_climbing` (preverbal clitic on matrix verb vs clitic attached to infinitive).

Each condition contains 24 minimal pairs (48 sentences), except `clitic_climbing` which has 24 pairs in a single condition. Staging totals: 1,488 rows / 744 paired items across 29 conditions spanning 11 categories. Each pair consists of a context sentence and a target sentence, with a designated hotspot token for surprisal measurement. Tokenization follows Moses conventions (space-separated punctuation, fused Spanish contractions *del*/*al*, attached clitics on non-finite verbs *comerlo*/*díselo*). All stimuli are lowercase with accents preserved (`á é í ó ú ñ ü ¿ ¡`).

Stimuli were generated by Claude Opus 4.7 subagents following the per-category translation rules in `docs/eval_stimuli/spanish.md` (Spanish) and `docs/eval_stimuli/design.md` (English). *This is a deviation from the originally-preregistered Deepseek-V2 generator choice; the change reflects the stimuli-construction workflow in use at the time of Spanish staging (2026-04-16) and is documented here rather than silently adopted.* Both English and Spanish sentences are verified by a fluent Peninsular-Spanish collaborator; staging items flagged for dialectal or register concerns (e.g., `ello`+se stacking in expletive contexts, `esperar que` + future indicative in extraction contexts) are documented in `docs/eval_stimuli/notebook.md` for vetter review.

### Target Grammatical Contexts

#### 3rd Singular Pronoun Subject Drop

> **English:**
> 1. Marta won the award. She shows pride.
> 2. \*Marta won the award. Shows pride.
>
> **Spanish:**
> 1. marta ganó el premio . ella muestra orgullo .
> 2. marta ganó el premio . muestra orgullo . (preferred — null subject)

#### 3rd Plural Pronoun Subject Drop

> **English:**
> 1. The tourists missed the bus. They called a taxi.
> 2. \*The tourists missed the bus. Called a taxi.
>
> **Spanish:**
> 1. los turistas perdieron el autobús . ellos llamaron un taxi .
> 2. los turistas perdieron el autobús . llamaron un taxi . (preferred — null subject)

#### 3rd Singular Pronoun Object Drop

> **English:**
> 1. Where is the vase? He placed it on the table.
> 2. \*Where is the vase? He placed on the table.
>
> **Spanish:**
> 1. ¿ dónde está el jarrón ? él lo puso en la mesa . (preverbal clitic *lo* required)
> 2. \*¿ dónde está el jarrón ? él puso en la mesa .

#### 3rd Plural Pronoun Object Drop

> **English:**
> 1. The band played several new songs. The audience enjoyed them immensely.
> 2. \*The band played several new songs. The audience enjoyed immensely.
>
> **Spanish:**
> 1. la banda tocó varias canciones nuevas . el público las disfrutó muchísimo . (preverbal clitic *las* required)
> 2. \*la banda tocó varias canciones nuevas . el público disfrutó muchísimo .

#### 2nd Singular Pronoun Subject Drop

> **English:**
> 1. Luca, you forget the keys often. You take the keys before leaving.
> 2. ?Luca, you forget the keys often. Take the keys before leaving.
>
> **Spanish:**
> 1. lucas , olvidas las llaves a menudo . tú coges las llaves antes de salir .
> 2. lucas , olvidas las llaves a menudo . coges las llaves antes de salir . (preferred — null subject)

#### 2nd Plural Pronoun Subject Drop

> **English:**
> 1. Guys, you leave the window open. You all let the cat in.
> 2. ?Guys, you leave the window open. Let the cat in.
>
> **Spanish:**
> 1. amigos , dejáis la ventana abierta . vosotros dejáis entrar al gato .
> 2. amigos , dejáis la ventana abierta . dejáis entrar al gato . (preferred — null subject)

#### 1st Singular Pronoun Subject Drop

> **English:**
> 1. I just finished the project. I believe that the result is satisfactory.
> 2. ??I just finished the project. Believe that the result is satisfactory.
>
> **Spanish:**
> 1. acabo de terminar el proyecto . yo creo que el resultado es satisfactorio .
> 2. acabo de terminar el proyecto . creo que el resultado es satisfactorio . (strongly preferred — null subject)

#### 1st Plural Pronoun Subject Drop

> **English:**
> 1. We reviewed the contract. We agree with the terms.
> 2. ??We reviewed the contract. Agree with the terms.
>
> **Spanish:**
> 1. hemos revisado el contrato . nosotros estamos de acuerdo con los términos .
> 2. hemos revisado el contrato . estamos de acuerdo con los términos . (strongly preferred — null subject)

#### Subordinate Clause Pronoun Dropping (Embedded Drop — coref)

> **English:**
> 1. Marco arrived late. I know that he took the wrong train.
> 2. \*Marco arrived late. I know that took the wrong train.
>
> **Spanish:**
> 1. marco llegó tarde . sé que él tomó el tren equivocado .
> 2. marco llegó tarde . sé que tomó el tren equivocado . (preferred in Spanish — Position of Antecedent Hypothesis)

#### Subject Control

*Note: Control constructions test PRO in infinitival complements. English children do not show early preference for subordinate null subjects, making this an important test of the grammatical cluster.*

> **English:**
> 1. Marco tried to ask for help. (grammatical: PRO subject)
> 2. \*Marco tried him to ask for help. (ungrammatical: overt embedded subject)
>
> **Spanish:**
> 1. marco intentó pedir ayuda . (grammatical: PRO subject)
> 2. \*marco intentó él pedir ayuda . (ungrammatical: overt embedded subject)

#### Object Control

> **English:**
> 1. The doctor urges the patient to rest. (grammatical: PRO controlled by object)
> 2. \*The doctor urges the patient him to rest. (ungrammatical: overt embedded subject)
>
> **Spanish:**
> 1. el médico urge al paciente a descansar . (grammatical: PRO controlled by object)
> 2. \*el médico urge al paciente él a descansar . (ungrammatical: overt embedded subject)

#### Expletive Contexts with Verb "seems"

> **English:**
> 1. The light turns off often. It seems that the light turns off.
> 2. \*The light turns off often. Seems that the light turns off.
>
> **Spanish:**
> 1. la luz se apaga a menudo . parece que la luz se apaga . (preferred — null expletive in Spanish)
> 2. la luz se apaga a menudo . ello parece que la luz se apaga . (marked/archaic — overt *ello* as expletive)

#### Expletive Contexts with Verb "be"

> **English:**
> 1. Were you looking for someone? It is the guy you were looking for.
> 2. \*Were you looking for someone? Is the guy you were looking for.
>
> **Spanish:**
> 1. ¿ buscabas a alguien ? es el chico que buscabas . (preferred — null expletive in Spanish)
> 2. ¿ buscabas a alguien ? ello es el chico que buscabas . (marked/archaic — overt *ello* as expletive)

#### Long-distance Binding (Embedded Clauses — non-coref topic shift)

> **English:**
> 1. Luca orders a pizza. Luca says that he prepares dinner.
> 2. \*Luca orders a pizza. Luca says that prepares dinner.
>
> **Spanish:**
> 1. lucas pide una pizza . lucas dice que él prepara la cena .
> 2. lucas pide una pizza . lucas dice que prepara la cena . (grammatical in Spanish — topic-continuity null)

#### Conjunction Without Topic Shift

*Note: In same-subject coordinations, English allows subject omission in the second conjunct (conjunction reduction). This tests whether the grammar distinguishes topic continuity from topic shift.*

> **English:**
> 1. Luca is hungry. Luca opens the fridge and he takes a sandwich.
> 2. Luca is hungry. Luca opens the fridge and takes a sandwich. (grammatical: conjunction reduction)
>
> **Spanish:**
> 1. lucas tiene hambre . lucas abre la nevera y él toma un sándwich .
> 2. lucas tiene hambre . lucas abre la nevera y toma un sándwich . (strongly preferred — null subject)

#### Conjunction With Topic Shift

*Note: When the subject changes across conjuncts, English requires an overt pronoun. This tests sensitivity to topic shift.*

> **English:**
> 1. Antonio is in the garden. Antonio calls the gardener and she plants the flowers for him.
> 2. \*Antonio is in the garden. Antonio calls the gardener and plants the flowers for him. (ungrammatical: topic shift requires overt subject)
>
> **Spanish:**
> 1. antonio está en el jardín . antonio llama al jardinero y ella planta las flores por él . (preferred — overt subject on topic shift)
> 2. antonio está en el jardín . antonio llama al jardinero y planta las flores por él . (marginal — topic shift prefers overt even in Spanish)

#### Subject Extraction — *that-trace* Asymmetry

*This is the theoretical payoff: Spanish has no that-trace effect, so the grammaticality direction flips between English and Spanish for the same structural alternation. See H5.*

> **English:**
> 1. A scientist will make the discovery. Who do you think will make the discovery?
> 2. \*A scientist will make the discovery. Who do you think that will make the discovery?
>
> **Spanish:**
> 1. un científico hará el descubrimiento . ¿ quién crees que hará el descubrimiento ? (grammatical — *que* required)
> 2. un científico hará el descubrimiento . ¿ quién crees hará el descubrimiento ? (marginal — *que* drop marked)

#### Object Extraction

> **English:**
> 1. The scientist will make the discovery. What do you think the scientist will make?
> 2. The scientist will make the discovery. What do you think that the scientist will make?
>
> **Spanish:**
> 1. el científico hará un descubrimiento . ¿ qué crees que el científico descubrirá ? (preferred — *que* required)
> 2. el científico hará un descubrimiento . ¿ qué crees el científico descubrirá ? (marginal)

---

### Spanish-only Grammatical Contexts

These three categories test Spanish-specific null-subject cluster properties (Rizzi 1982) with no English counterpart. They do not participate in cross-linguistic priming pairings but test whether the null-subject grammar models learn encompasses the broader property cluster.

#### Postverbal Subject (SV vs VS alternation)

*Classic null-subject-cluster property: Spanish licenses postverbal subjects freely; English does not.*

> **Declarative:**
> 1. maría llegó al aeropuerto ayer . (SV — preverbal subject, neutral)
> 2. llegó maría al aeropuerto ayer . (VS — postverbal subject; information-structure-dependent)
>
> **Interrogative:**
> 1. \*¿ qué maría dijo ? (SV — marked in wh-questions)
> 2. ¿ qué dijo maría ? (VS — strongly preferred)

#### Impersonal *se*

*Tests Spanish's native subjectless constructions via the impersonal *se* construction vs. 3pl indefinite paraphrase.*

> **Impersonal passive (se vs 3pl):**
> 1. se dice que ganará el partido . (*se*-form — impersonal/passive)
> 2. dicen que ganará el partido . (3pl-form — indefinite "they" paraphrase; near-synonymous)

#### Clitic Climbing

*Tests clitic placement in restructuring contexts. Both orders grammatical with restructuring verbs (*querer*, *poder*, *tener que*, *ir a*, *deber*, etc.).*

> 1. lo quiero comer . (climbed — preverbal clitic on matrix verb)
> 2. quiero comerlo . (non-climbed — clitic attached to infinitive)

---

## Sample Size

The evaluation set contains 744 paired items (1,488 rows) spanning 29 conditions across 11 categories:
- **8 cross-linguistic categories** (item-paired EN ↔ ES): `subject_drop` (6 conditions × 24 pairs), `subject_drop_no_agreement` (6 × 24), `object_drop` (2 × 24), `embedded_drop` (2 × 24), `control` (2 × 24), `expletive` (2 × 24), `conjunction` (2 × 24), `extraction` (2 × 24).
- **3 Spanish-only categories**: `postverbal_subject` (2 × 24), `se_impersonal` (2 × 24), `clitic_climbing` (1 × 24).

Each model is evaluated at 40 checkpoints across training. Each experimental condition (architecture × language × ablation) is replicated with 10 random initializations.

**Observations per category per condition per model:** 24 pairs × 40 checkpoints × 10 replications = 9,600 observations.

---

## Sample Size Rationale

No formal power analysis. Sample size is determined by the experimental design (number of architectures, checkpoints, and replications) rather than statistical power considerations. The repeated-measures structure provides substantial observations per condition.

---

## Stopping Rule

No stopping rule. Each model is trained for a fixed 20 epochs. All 40 checkpoints are collected and evaluated regardless of intermediate results.

---

## Manipulated Variables

*[TODO: Precisely define all variables you plan to manipulate and the levels or treatment arms of each variable. This is not applicable to any observational study.]*

---

## Measured Variables

Precisely define each variable that you will measure. This will include outcome measures, as well as any measured predictors or covariates.

### Perplexity

Perplexity will be measured from each model's distribution, by testing the model on a held-out test corpus, and measuring the model's expectation of each word in the test set as the average negative log-likelihood of each word. The lower the number, approaching 1, the better the model is at capturing the distribution of the test dataset.

### Word-by-Word Surprisal

For each evaluation sentence word by word surprisal is measured, or the negative log-likelihood of a word in context.

---

## Indices

We adopt a multi-metric reporting policy rather than committing to a single scoring function. SLOR is the primary index (original preregistered commitment), supplemented with raw log-probability and mean log-probability per sentence for robustness and interpretability. This extension is motivated by the Tjuatja et al. (2024) finding that SLOR can over-correct for length and frequency effects, the Padovani et al. (2025) FIT-CLAMS recommendation for paired per-corpus frequency controls, and the need to diagnose unexpected SLOR results against raw surprisal numbers.

### SLOR (Syntactic Log-Odds Ratio) — primary

To measure model preference between sentence pairs we use SLOR, the Syntactic Log-Odds Ratio: the log-probability of the sentence under the model minus the log-probability of the sentence under a unigram baseline, normalized by sentence length. Higher SLOR means higher acceptability.

**SLOR Formula:**

```
SLOR(S) = (1/|S|) * (log p_M(S) - log p_u(S))
```

where:
- `p_M(S)` = probability of sentence S under model M
- `p_u(S)` = probability of sentence S under unigram model
- `|S|` = sentence length

**Unigram baseline policy — paired per corpus (FIT-CLAMS style).** Every trained model uses the unigram baseline computed on *its own* training corpus, not a global baseline shared across conditions. Rationale: H2 compares models trained on ablated corpora; a shared unigram baseline would confound frequency-distribution differences (introduced by the ablation itself) with grammatical-knowledge differences. Per-corpus baselines cancel the frequency term on both sides of the log and leave the structural contribution isolated (Padovani et al. 2025). Unigram vocabulary matches each model's subword tokenizer; Laplace (+1) smoothing handles zero-count tokens (e.g., pronouns in the pronoun-removed corpus, should that intervention proceed in the follow-up).

### Raw and Mean Log-Probability — secondary

For each sentence we additionally report:
- **Raw LP**: `log p_M(S)`
- **Mean LP**: `(1/|S|) * log p_M(S)` — length-normalized but unadjusted for frequency.

These are reported alongside SLOR so unexpected SLOR values can be diagnosed against the raw surprisal.

### Accuracy Measurement

For each sentence pair, `SLOR(grammatical) > SLOR(ungrammatical)` is reported as a binary (1,0) where 1 means that the model preferred the grammatical example. This is reported as model accuracy. Binary preference under raw LP and Mean LP is also reported for comparison.

### Overt vs Null Preference

For pairs where both variants are grammatical but with a preferred form (e.g., Spanish `subject_drop` where null is preferred but overt is not ungrammatical; English `conjunction` no-shift where conjunction reduction is allowed), we report `SLOR(overt) > SLOR(null) = (1,0)` to measure the model's preference between the variants irrespective of grammaticality. This is reported separately from the accuracy measure so that pair-wise direction can be interpreted correctly. In Spanish, such a measure is relevant across most `subject_drop` items because both forms are grammatical, with null preferred in topic-continuity contexts.

### Preference Strength (Distance)

The difference score `SLOR(grammatical) − SLOR(ungrammatical)` measures how strongly a model prefers the grammatical choice. Analogous distance scores are reported under raw LP and Mean LP, and for the overt-vs-null contrasts.

### Follow-up metrics (not locked)

The following will be reported if implementable within the evaluation budget, but are not preregistered commitments:
- **SLLN-LP** (Liu et al. 2024, ZhoBLiMP): sub-linear length normalization that mitigates SLOR's over-correction without its extremes.
- **MORCELA** (Tjuatja et al. 2024): learned acceptability normalization; requires an acceptability-rating training set which is not currently available for Peninsular Spanish.
- **Token-length reporting per pair per tokenizer** (Ueda et al. 2024): cheap, strengthens the methods section.

### Per-item reporting alongside aggregates

Aggregate accuracy is reported alongside per-item preferences and by-item analyses (Newman et al. 2021), so that aggregate effects can be inspected for uniformity across items vs. driven by a small subset.

---

## Statistical Models

*[TODO: What statistical model will you use to test each hypothesis? Please include the type of model (e.g. ANOVA, RMANOVA, MANOVA, multiple regression, SEM, etc) and the specification of the model. This includes each variable that will be included, all interactions, subgroup analyses, pairwise or complex contrasts, and any follow-up tests from omnibus tests. If you plan on using any positive controls, negative controls, or manipulation checks you may mention that here. Provide enough detail so that another person could run the same analysis with the information provided. Remember that in your final article any test not included here must be noted as exploratory and that you must report the results of all tests.]*

---

## Transformations

### Continuous Predictors

**Checkpoint number** will be transformed in two ways for modeling:

1. **Log transformation:** `log_checkpoint = log₁₀(checkpoint_num + 1)` — captures the non-linear learning dynamics where early checkpoints show rapid change and later checkpoints plateau.

2. **Centering and scaling:** `checkpoint_centered = checkpoint_num - mean(checkpoint_num)` and `checkpoint_scaled = checkpoint_num / max(checkpoint_num)` — for model stability and interpretability of intercepts.

### Bounded Outcome Variables

**Accuracy** (binary correct/incorrect aggregated to proportions) will be logit-transformed for analysis:

```
accuracy_logit = log(p / (1 - p))
```

Where p is the proportion correct. Boundary cases (p = 0 or p = 1) will be adjusted using `p = 0.001` or `p = 0.999` to avoid undefined values.

### Categorical Variables

- **Architecture:** Treated as a categorical factor with n-gram models as the reference level for comparisons with deep learning architectures.
- **Intervention/Ablation:** Treated as a categorical factor with Baseline as the reference level.
- **Language:** Categorical factor (English, Spanish).

---

## Inference Criteria

*[TODO: What criteria will you use to make inferences? Please describe the information you'll use (e.g. specify the p-values, Bayes factors, specific model fit indices), as well as cut-off criterion, where appropriate. Will you be using one or two tailed tests for each of your analyses? If you are comparing multiple conditions or testing multiple hypotheses, will you account for this?]*

---

## Data Exclusion

*[TODO: How will you determine which data points or samples if any to exclude from your analyses? How will outliers be handled? Will you use any awareness check?]*

---

## Missing Data

*[TODO: How will you deal with incomplete or missing data?]*

---

## Exploratory Analysis

*[TODO: If you plan to explore your data to look for unspecified differences or relationships, you may include those plans here. If you list an exploratory test here, you are not obligated to report its results. But if you do report it you are obligated to describe it as an exploratory result.]*

---

## Other

*[TODO: If there is any additional information that you feel needs to be included in your preregistration, please enter it here. Literature cited, disclosures of any related work such as replications or work that uses the same data, or other helpful context would be appropriate here.]*

---

## Appendix 1: Tokenization

### Training corpora

Both BabyLM (English) and BebeLM (Spanish) training corpora use Moses-style tokenization: punctuation is separated from adjacent words by whitespace, apostrophes are separated, and sentence boundaries are preserved. All text is lowercased. Subword tokenization for model training is handled per-architecture (BPE/WordPiece/SentencePiece as appropriate).

**Spanish-specific conventions** (verified against `data/spanish/train_90M/*.train` at the time of corpus assembly, 2026-04-16):

- **Accents preserved as Unicode**: `á é í ó ú ü ñ`. Accent marks are diacritics, not case distinctions, and must be preserved to distinguish minimal pairs like `si` (if) / `sí` (yes), `el` (the) / `él` (he), `se` (reflexive) / `sé` (I know), `mas` (but, archaic) / `más` (more), `tu` (your) / `tú` (you).
- **Fused contractions**: `del` (not `de el`), `al` (not `a el`). Fused forms dominate ~100:1 across every source corpus.
- **Clitic attachment follows standard Spanish orthography**: clitics are attached to non-finite verb forms (infinitives *comerlo*, gerunds *dándome*, affirmative imperatives *díselo*) and separate before finite verbs (*lo puso*, *la vi*, *me dijo*).
- **Spanish question punctuation**: the full `¿...?` pair is used, with both `¿` and `?` treated as separate tokens with surrounding whitespace: `¿ quién llegó ?`.

### Evaluation stimuli

Evaluation stimuli follow the same Moses conventions and lowercasing as the training corpora. Per-category hotspot annotations mark the token position at which surprisal is measured for each sentence, allowing token-aligned minimal-pair comparisons. For Spanish categories where the hotspot token differs in position between the two pair members (e.g., `object_drop`: overt has preverbal clitic `lo`/`la`, null does not), the hotspot is the first token immediately after the varying slot — see `docs/eval_stimuli/spanish.md` §5 for per-category rules.
