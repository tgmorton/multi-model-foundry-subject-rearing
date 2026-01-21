# OSF Preregistration

## Title

Comparing Generative Linguistics and Information Theoretic Accounts of Subject Drop in English and Italian Using Statistical Language Models

---

## Description

This study compares generative linguistics and information theoretic accounts of subject drop in English and Italian. A majority of languages in the world allow for speakers to not pronounce the subject of sentences. This generalization within a language, whether to allow subject drop or disallow it broadly, has long been a subject of investigation in classical linguistics. Such approaches require that learners are pre-endowed with knowledge of the breadth of linguistic structures available to them, and that this innate knowledge allows them to learn given the data available to them. On the other hand, information theoretic and usage based accounts propose that learners do not need innate knowledge to aid learning, instead learners make inferences about language in context that lead to adult-like language use. We use statistical language models as candidate learners, manipulating the kind of information available to learners motivated by generative and information theoretic accounts, and compare learning across models and manipulations.

**[TODO: Talk about the generative accounts]**

**[TODO: Talk about an information theoretic account]**

**[TODO: predicted results]**

**[TODO: potential interpretation of results - maybe a middle ground]**

---

## Contributors

- **Thomas Morton** — UCSD Psychology
- **Ben Bergen** — UCSD Cognitive Psychology
- **Victor Ferreira** — UCSD Psychology
- **Alex Warstadt** — UCSD Linguistics

---

## Tags

*[TODO: Enter specific keywords that describe the key elements and concepts of your research. These keywords will improve the discoverability of your registration in search results and databases.]*

---

## Hypotheses

*[TODO: List specific, concise, and testable hypotheses. Please state if the hypotheses are directional or non-directional. If directional, state the direction. A predicted effect is also appropriate here. If a specific interaction or moderation is important to your research, you can list that as a separate hypothesis.]*

---

## Study Type

*Please check one of the following statements:*

- [ ] **Experiment** — A researcher randomly assigns treatments to study subjects, this includes field or lab experiments. This is also known as an intervention experiment and includes randomized controlled trials.
- [ ] **Observational Study** — Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, "natural experiments," and regression discontinuity designs.
- [ ] **Meta-Analysis** — A systematic review of published studies.
- [ ] **Other**

---

## Blinding

*Blinding describes who is aware of the experimental manipulations within a study. Mark all that apply.*

- [ ] No blinding is involved in this study.
- [ ] For studies that involve human subjects, they will not know the treatment group to which they have been assigned.
- [ ] Personnel who interact directly with the study subjects (either human or non-human subjects) will not be aware of the assigned treatments. (Commonly known as "double blind")
- [ ] Personnel who analyze the data collected from the study are not aware of the treatment applied to any given group.

**Is there any additional blinding in this study?**

---

## Study Design

This study investigates learning in English and Italian Language Models by performing target manipulations on baseline English and Italian corpora which are then used to train Language Models. Each dataset is 90 million words and is tokenized for each model: broken up into computable vectors to be operated on within the models (see Appendix 1). Tokenized datasets are broken up into batches of 1024 tokens to be fed into the models. Models are trained in Epochs, where for each Epoch a model is trained on the whole 90 million words in its corpus. Each model is trained for twenty epochs—the same 90 million words 20 times over. Over the course of training, each model saves a checkpoint, which is a frozen copy of all of its weights and its tokenizer, 40 times in even log-space over the course of training. These checkpoints will be spaced in such a way that within-model they are comparable against each other, and between-model as close enough as reasonable for cross-model comparison (e.g. transformers are bound by optimizer steps for gradient accumulation in a way that an n-gram model would not be). So in a way, intervention-type is compared between and within-model, which makes for a fairly vast between-subjects matrix. However these comparisons are done with repeated-measures of the same evaluation tasks. Making this a between-subject repeated-measures design.

The factor chart, which can be split as Model × Intervention is fairly large as enabled by the computational design.

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

*[TODO: Preregistration is designed to make clear the distinction between confirmatory tests, specified prior to seeing the data, and exploratory analyses conducted after observing the data. Therefore, creating a research plan in which existing data will be used presents unique challenges. Please select the description that best describes your situation. See https://cos.io/prereg for more information.]*

---

## Explanation of Existing Data

*[TODO: If you indicate that you will be using some data that already exist in this study, please describe the steps you have taken to assure that you are unaware of any patterns or summary statistics in the data. This may include an explanation of how access to the data has been limited, who has observed the data, or how you have avoided observing any analysis of the specific data you will use in your study.]*

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

**[TODO: To be completed]**

#### Subordinate Clause Pronoun Dropping

> **English:**
> 1. Marco arrived late. I know that he took the wrong train.
> 2. \*Marco arrived late. I know that took the wrong train.
>
> **Italian:**
> 1. Marco è arrivato in ritardo. So che lui ha preso il treno sbagliato.
> 2. Marco è arrivato in ritardo. So che ha preso il treno sbagliato.

#### Subject Control

**[TODO: To be completed]**

#### Object Control

**[TODO: To be completed]**

#### Expletive Contexts with Verb "seems"

**[TODO: To be completed]**

#### Expletive Contexts with Verb "be"

**[TODO: To be completed]**

#### Long-distance Binding

**[TODO: To be completed]**

#### Conjunction Without Topic Drop

**[TODO: To be completed]**

#### Conjunction With Topic Drop

**[TODO: To be completed]**

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

*[TODO: Describe the sample size of your study. How many units will be analyzed in the study? This could be the number of people, birds, classrooms, plots, or countries included. If the units are not individuals, then describe the size requirements for each unit. If you are using a clustered or multilevel design, describe how many units are you collecting at each level of the analysis. This might be the number of samples or a range, minimum, or maximum.]*

---

## Sample Size Rationale

*[TODO: This could include a power analysis or an arbitrary constraint such as time, money, or personnel.]*

---

## Stopping Rule

*[TODO: If your data collection procedures do not give you full control over your exact sample size, specify how you will decide when to terminate your data collection. If you are using sequential analysis, include your pre-specified thresholds.]*

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

*[TODO: If you plan on transforming, centering, recoding the data, or requiring a coding scheme for categorical variables, please describe that process.]*

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
