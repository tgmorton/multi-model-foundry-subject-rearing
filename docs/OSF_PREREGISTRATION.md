# OSF Preregistration

## Title

Comparing Generative Linguistics and Information Theoretic Accounts of Subject Drop in English and Italian Using Statistical Language Models

---

## Description

### The Null Subject Question

A majority of the world's languages allow speakers to omit the subject of sentences. In Italian, "Parla italiano" (speaks Italian) is a complete, grammatical sentence; in English, "*Speaks Italian" is not. This typological split has been a central question in linguistics for over forty years: does it reflect a deep grammatical distinction, a surface-level frequency effect, or something in between?

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
Can language models, trained on naturalistic input, learn the target grammatical patterns? Specifically, can English-trained models learn that overt subjects are required, and can Italian-trained models learn that null subjects are permitted? This establishes baseline learnability.

**Question 2: Do models mis-learn when evidence is altered?**
When we systematically manipulate the input—removing pronouns, adding pronouns, altering morphology—do models develop incorrect grammatical preferences? This tests the causal role of specific evidence types: if removing pronouns causes models to accept null subjects in English, this demonstrates that pronoun distribution is necessary for learning the target grammar.

**Question 3: Do models show child-like developmental trajectories?**
English-acquiring children initially over-accept null subjects before learning they are ungrammatical. Italian-acquiring children do not show this pattern because null subjects are grammatical in Italian. Do language models show the same developmental asymmetry? Early preference for null subjects in English (that is later overcome) but not in Italian would suggest that models and children face similar learning problems and arrive at similar solutions.

### Theoretical Stakes

Beyond these empirical questions, the pattern of results bears on deeper theoretical issues. If manipulating one aspect of the input (pronoun distribution) affects not only pronoun acceptability but also judgments about syntactically-related phenomena (that-trace effects, extraction, embedded subjects), this would suggest grammatical knowledge involves structured representations—even in learners without innate linguistic structure.

**Local effects only** would suggest grammatical knowledge is built from surface statistics; phenomena that don't co-occur in training are learned independently.

**Distal effects** would suggest grammatical knowledge involves interconnected representations where syntactically-related phenomena cluster together, as documented across human languages.

### Design Overview

We train models on English and Italian corpora with targeted manipulations: removing pronouns (randomly and by contextual predictability), removing expletives, manipulating verbal morphology, and neutralizing case distinctions. We evaluate models not only on null-subject acceptability, but on the full set of related phenomena: that-trace effects, extraction, subordinate clauses, control constructions. The pattern of results (local vs. distal) will inform our understanding of how grammatical knowledge is organized and acquired.

---

## Contributors

- **Thomas Morton** — UCSD Psychology
- **Ben Bergen** — UCSD Cognitive Psychology
- **Victor Ferreira** — UCSD Psychology
- **Alex Warstadt** — UCSD Linguistics

---

## Tags

null subjects, pro-drop, language acquisition, language models, controlled rearing, grammatical knowledge, English, Italian, cross-linguistic, BabyLM

---

## Hypotheses

### Core Hypotheses

**H1: Baseline Learnability (Can models learn this?)**

Models trained on unmanipulated corpora will acquire the target grammatical patterns:
- **English models** will prefer overt subjects over null subjects (accuracy above chance on null-subject minimal pairs)
- **Italian models** will accept null subjects (accuracy above chance, with preference for null over overt in appropriate contexts)

This establishes that the target grammar is learnable from naturalistic input before testing the effects of manipulations.

---

**H2: Evidence Manipulation (Do models mis-learn when evidence is altered?)**

Manipulating the input will cause models to acquire incorrect grammatical preferences:
- **English models** trained without pronouns will fail to learn overt-subject requirements (reduced accuracy, preference shift toward null subjects)
- **Italian models** trained with inserted overt pronouns will show reduced null-subject acceptance (preference shift toward overt subjects)

| Manipulation | Predicted Effect |
|--------------|------------------|
| Remove pronouns (English) | ↑ Null subject acceptance (mis-learning) |
| Insert pronouns (Italian) | ↓ Null subject acceptance (mis-learning) |

This tests whether pronoun distribution is *causally necessary* for learning the target grammar.

---

**H3: Developmental Trajectory (Do models show child-like learning patterns?)**

Models will show developmental patterns paralleling child language acquisition:
- **English models** will show early preference for null subjects that is overcome during training (matching the pattern in English-acquiring children)
- **Italian models** will NOT show early null-subject preference followed by reversal (matching Italian-acquiring children, for whom null subjects are grammatical throughout)

| Language | Child Pattern | Predicted Model Pattern |
|----------|--------------|------------------------|
| English | Early null-subject acceptance → learned rejection | Early null preference → learned overt preference |
| Italian | Consistent null-subject acceptance | Consistent null preference (no reversal) |

This tests whether models face the same learning problem as children and arrive at similar developmental solutions.

---

**H4: Informativity-Based vs. Random Removal**

Does targeting pronouns by their contextual predictability (pointwise mutual information) produce different effects than random removal at matched rates?

| Pattern | Implication |
|---------|-------------|
| **Informativity matters** | Learning is sensitive to the information-theoretic properties of individual tokens |
| **No difference** | What matters is the aggregate statistical pattern, not the specific properties of removed items |

*Both PMI-based and random removal sweeps (0–100%, 10% increments) will be conducted.*

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
Adding rich agreement morphology to English will facilitate null subjects; impoverishing Italian agreement will push toward overt subjects.
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

This study investigates learning in English and Italian language models by performing targeted manipulations on baseline corpora. We train multiple model architectures on each manipulated dataset, saving checkpoints throughout training to track learning dynamics. The experimental design crosses architecture with data manipulation, yielding a between-subjects matrix with repeated measures across evaluation tasks.

### Training Replication

Each experimental condition (architecture × language × ablation) is trained with 10 random initializations to assess training variability. This allows us to distinguish signal from noise in model learning trajectories.

### Datasets

- **English**: BabyLM corpus (90M tokens training, 10M held out for evaluation)
- **Italian**: bebe-lm corpus (custom dataset, parallel construction, 90M training, 10M evaluation)

### Checkpoint Schedule

Each model saves 40 checkpoints over the course of training, spaced in log-time. This captures learning dynamics with higher resolution early in training (where rapid changes occur) and lower resolution later (where learning plateaus). Checkpoints are matched across architectures as closely as possible by optimizer steps, though some architectures (e.g., n-grams) do not use gradient-based optimization.

### Factor Structure

The study crosses two dimensions:

1. **Architecture**: n-gram (1-5), GPT-2 (small, medium, large), BERT, LSTM, Mamba
2. **Data manipulation**: Baseline, plus ablations (pronoun removal, expletive removal, morphology manipulation, case neutralization, stacked ablations) and sweeps (PMI-based and random pronoun removal 0-100%)

This yields a large between-subjects matrix, with within-model repeated measures across evaluation tasks. The factor chart, which can be split as Model × Intervention, is fairly large as enabled by the computational design.

### English

**Model:** [n-gram: 1-, 2-, 3-, 4-, 5-gram; GPT: small, medium, large; BERT: large; LSTM; and Mamba: 370m]

**Intervention:** [Baseline, Remove expletives, Remove Determiners, Impoverish Determiners, Impoverish Verbal Morphology, Insert Verbal Morphology, All-Evidence+Remove Pronouns (0:100 remove sweep, step 10), no other evidence+pronoun_sweep]

### Italian

**Model:** [n-gram: 1-, 2-, 3-, 4-, 5-gram; GPT: small, medium, large; BERT: large; LSTM; and Mamba: 370m]

**Intervention Conditions:** [Baseline, Remove Expletives, Remove Determiners, Impoverish Determiners, Impoverish Verbal Morphology, Insert Pronouns]

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

Then, the model is evaluated on its overall grammatical performance using BLiMP, The Benchmark of Linguistic Minimal Pairs, which tests the model's preference on comparisons of grammatical and ungrammatical linguistic sentences to assess the model's alignment with human judgements.

In addition the model will be evaluated in the same way, on a suite of minimal pairs designed to target the model's preference on specific grammatical contexts relevant to the production of null and overt subjects and objects. This evaluation set is created parallel in English and Italian, such that for each English pair there is an Italian pair that is syntactically, if not semantically, equivalent, with differing or similar grammatical status. For each evaluation criteria there are 12 minimal pairs per language. Each part of a pair consists of a context sentence and a target sentence. The evaluation stimuli were generated by Deepseek-V2, a frontier LLM that scores among the highest marks on Italian performance according to the HuggingFace benchmark as of the time of writing. Both English and Italian sentences are evaluated respectively by one fluent researcher.

### Target Grammatical Contexts

#### 3rd Singular Pronoun Subject Drop

> **English:**
> 1. Marta won the award. She shows pride.
> 2. \*Marta won the award. Shows pride.
>
> **Italian:**
> 1. Marta ha vinto il premio. Lei mostra orgoglio.
> 2. Marta ha vinto il premio. Mostra orgoglio.

#### 3rd Plural Pronoun Subject Drop

> **English:**
> 1. The tourists missed the bus. They called a taxi.
> 2. \*The tourists missed the bus. Called a taxi.
>
> **Italian:**
> 1. I turisti hanno perso l'autobus. Loro hanno chiamato un taxi.
> 2. I turisti hanno perso l'autobus. Hanno chiamato un taxi.

#### 3rd Singular Pronoun Object Drop

> **English:**
> 1. Where is the vase? He placed it on the table.
> 2. \*Where is the vase? He placed on the table.
>
> **Italian:**
> 1. Dov'è il vaso? L'ha messo sul tavolo.
> 2. \*Dov'è il vaso? Ha messo sul tavolo.

#### 3rd Plural Pronoun Object Drop

> **English:**
> 1. The band played several new songs. The audience enjoyed them immensely.
> 2. \*The band played several new songs. The audience enjoyed immensely.
>
> **Italian:**
> 1. La band ha suonato diverse nuove canzoni. Il pubblico le ha apprezzate moltissimo.
> 2. \*La band ha suonato diverse nuove canzoni. Il pubblico ha apprezzato moltissimo.

#### 2nd Singular Pronoun Subject Drop

> **English:**
> 1. Luca, you forget the keys often. You take the keys before leaving.
> 2. ?Luca, you forget the keys often. Take the keys before leaving.
>
> **Italian:**
> 1. Luca, dimentichi le chiavi spesso. Tu prendi le chiavi prima di uscire.
> 2. Luca, dimentichi le chiavi spesso. Prendi le chiavi prima di uscire.

#### 2nd Plural Pronoun Subject Drop

> **English:**
> 1. Guys, you leave the window open. You all let the cat in.
> 2. ?Guys, you leave the window open. Let the cat in.
>
> **Italian:**
> 1. Ragazzi, lasciate la finestra aperta. Voi fate entrare il gatto.
> 2. Ragazzi, lasciate la finestra aperta. Fate entrare il gatto.

#### 1st Singular Pronoun Subject Drop

> **English:**
> 1. I just finished the project. I believe that the result is satisfactory.
> 2. ??I just finished the project. Believe that the result is satisfactory.
>
> **Italian:**
> 1. Ho appena finito il progetto. Io credo che il risultato sia soddisfacente.
> 2. Ho appena finito il progetto. Credo che il risultato sia soddisfacente.

#### 1st Plural Pronoun Subject Drop

> **English:**
> 1. We reviewed the contract. We agree with the terms.
> 2. ??We reviewed the contract. Agree with the terms.
>
> **Italian:**
> 1. Abbiamo rivisto il contratto. Noi siamo d'accordo con i termini.
> 2. Abbiamo rivisto il contratto. Siamo d'accordo con i termini.

#### Subordinate Clause Pronoun Dropping

> **English:**
> 1. Marco arrived late. I know that he took the wrong train.
> 2. \*Marco arrived late. I know that took the wrong train.
>
> **Italian:**
> 1. Marco è arrivato in ritardo. So che lui ha preso il treno sbagliato.
> 2. Marco è arrivato in ritardo. So che ha preso il treno sbagliato.

#### Subject Control

*Note: Control constructions test PRO in infinitival complements. English children do not show early preference for subordinate null subjects, making this an important test of the grammatical cluster.*

> **English:**
> 1. Marco dares to ask for help. (grammatical: PRO subject)
> 2. \*Marco dares him to ask for help. (ungrammatical: overt embedded subject)
>
> **Italian:**
> 1. Marco osa chiedere aiuto. (grammatical: PRO subject)
> 2. \*Marco osa lui chiedere aiuto. (ungrammatical: overt embedded subject)

#### Object Control

> **English:**
> 1. The doctor urges the patient to rest. (grammatical: PRO controlled by object)
> 2. \*The doctor urges the patient him to rest. (ungrammatical: overt embedded subject)
>
> **Italian:**
> 1. Il medico esorta il paziente a riposare. (grammatical: PRO controlled by object)
> 2. \*Il medico esorta il paziente lui a riposare. (ungrammatical: overt embedded subject)

#### Expletive Contexts with Verb "seems"

> **English:**
> 1. The light turns off often. It seems that the light turns off.
> 2. \*The light turns off often. Seems that the light turns off.
>
> **Italian:**
> 1. La luce si spegne spesso. Sembra che la luce si spenga.
> 2. La luce si spegne spesso. Sembra che la luce si spenga. (grammatical in Italian: null expletive)

#### Expletive Contexts with Verb "be"

> **English:**
> 1. Were you looking for someone? It is the guy you were looking for.
> 2. \*Were you looking for someone? Is the guy you were looking for.
>
> **Italian:**
> 1. Cercavi qualcuno? È il ragazzo che cercavi.
> 2. Cercavi qualcuno? È il ragazzo che cercavi. (grammatical in Italian: null expletive)

#### Long-distance Binding (Embedded Clauses)

> **English:**
> 1. Luca orders a pizza. Luca says that he prepares dinner.
> 2. \*Luca orders a pizza. Luca says that prepares dinner.
>
> **Italian:**
> 1. Luca ordina una pizza. Luca dice che lui prepara la cena.
> 2. Luca ordina una pizza. Luca dice che prepara la cena. (grammatical in Italian)

#### Conjunction Without Topic Shift

*Note: In same-subject coordinations, English allows subject omission in the second conjunct (conjunction reduction). This tests whether the grammar distinguishes topic continuity from topic shift.*

> **English:**
> 1. Luca is hungry. Luca opens the fridge and he takes a sandwich.
> 2. Luca is hungry. Luca opens the fridge and takes a sandwich. (grammatical: conjunction reduction)
>
> **Italian:**
> 1. Luca ha fame. Luca apre il frigo e lui prende un panino.
> 2. Luca ha fame. Luca apre il frigo e prende un panino. (grammatical: null subject)

#### Conjunction With Topic Shift

*Note: When the subject changes across conjuncts, English requires an overt pronoun. This tests sensitivity to topic shift.*

> **English:**
> 1. Antonio is in the garden. Antonio calls the gardener and she plants the flowers for him.
> 2. \*Antonio is in the garden. Antonio calls the gardener and plants the flowers for him. (ungrammatical: topic shift requires overt subject)
>
> **Italian:**
> 1. Antonio è in giardino. Antonio chiama il giardiniere e lei pianta i fiori per lui.
> 2. Antonio è in giardino. Antonio chiama il giardiniere e pianta i fiori per lui. (marginal: topic shift prefers overt subject even in Italian)

#### Subject Extraction (target pronounced 'that')

> **English:**
> 1. A scientist will make the discovery. Who do you think will make the discovery?
> 2. \*A scientist will make the discovery. Who do you think that will make the discovery?
>
> **Italian:**
> 1. Uno scienziato farà la scoperta. Chi pensi farà la scoperta?
> 2. Uno scienziato farà la scoperta. Chi pensi che farà la scoperta?

#### Object Extraction (target pronounced 'that')

> **English:**
> 1. The scientist will make the discovery. What do you think the scientist will make?
> 2. The scientist will make the discovery. What do you think that the scientist will make?
>
> **Italian:**
> 1. Lo scienziato farà una scoperta. Cosa pensi lo scienziato farà?
> 2. Lo scienziato farà una scoperta. Cosa pensi che lo scienziato farà?

---

## Sample Size

Each stimuli set contains 12 minimal pairs (24 sentences) per language. Each model is evaluated at 40 checkpoints across training. Each experimental condition is replicated with 10 random initializations.

**Observations per stimuli set per condition:** 12 pairs × 2 languages × 40 checkpoints × 10 replications = 9,600 observations

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

### SLOR (Syntactic Log-Odds Ratio)

To measure model preference between sentence pairs we use a normalized measure called SLOR, short for Syntactic Log-Odds ratio. This transforms the surprisal measure into the sum of the sentence surprisal, and sum of the probability of words in a sentence as measured by a unigram model normalized by sentence length. This measure is then compared, where higher SLOR means higher model acceptability.

**SLOR Formula:**

```
SLOR(S) = (1/|S|) * (log p_M(S) - log p_u(S))
```

where:
- `p_M(S)` = probability of sentence S under model M
- `p_u(S)` = probability of sentence S under unigram model
- `|S|` = sentence length

### Accuracy Measurement

For each sentence pair, `SLOR(grammatical) > SLOR(ungrammatical)` is reported as a binary (1,0) where 1 means that the model preferred the grammatical example. This is reported as model accuracy.

### Overt vs Null Preference

The same measure will be taken for `SLOR(overt) > SLOR(null) = (1,0)` to measure the model's overall preference for overt and null contexts irregardless of grammatical contrasts—this second criterion is relevant for cases like conjunction without topic shift where there is no strict expectation of grammaticality vs ungrammaticality, which is why it will not be included in the above accuracy measurements (only its topic-shift variant, where such a contrast is expected). Likewise, in Italian, such a measure is relevant only in some cases, as there is relatively free variation (with preference towards null subjects) for null and overt subjects.

### Preference Strength

The difference score of `SLOR(grammatical) – SLOR(ungrammatical)` will be taken to measure how strong a preference the model has for the grammatical choice over the ungrammatical choice, and likewise for `SLOR(overt) - SLOR(null)`.

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
- **Language:** Categorical factor (English, Italian).

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

*[TODO: Details about tokenization process to be added]*
