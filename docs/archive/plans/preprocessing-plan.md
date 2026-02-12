# Project: A Controlled Rearing Study of Subject Drop in English

**Objective:** To investigate how different types of linguistic evidence affect a language model's learning of subject-drop rules in English. This will be accomplished by training several models on systematically modified (ablated) corpora and evaluating their grammatical knowledge.

---

## Phase 1: Environment and Asset Setup

### 1.1. Core Libraries
- **AI-TASK:** Set up a Python environment with the following core libraries:
  - `torch` (for model training and architecture)
  - `transformers` (for GPT-2 model implementation)
  - `datasets` (for handling the BabyLM corpus)
  - `spacy` (for linguistic processing and corpus ablation)
  - `wandb` (for logging training metrics)
  - `sentencepiece` (for tokenization)

### 1.2. Data Acquisition
- **AI-TASK:** Download the BabyLM dataset (100M word training corpus, 10M word test set).
  - Source: [Hugging Face Datasets](https://huggingface.co/datasets/babylm/babylm_100M) or as specified in the BabyLM Challenge.
- **AI-TASK:** Download the BLIMP evaluation dataset.
  - Source: [https://github.com/alexwarstadt/blimp](https://github.com/alexwarstadt/blimp)

### 1.3. Model Configuration
- **AI-TASK:** Define the base model architecture using the `transformers` library.
  - **Model:** GPT-2
  - **Hyperparameters:** Create a configuration file or script to hold the following parameters. The values are to be determined (TBD) as per the final experimental setup (see Table 5 in the source document).

| Hyperparameter            | Value |
| ------------------------- | ----- |
| Layers                    | TBD   |
| Embedding size            | TBD   |
| Hidden size               | TBD   |
| Intermediate hidden size  | TBD   |
| Attention heads           | TBD   |
| Attention head size       | TBD   |
| Activation function       | TBD   |
| Vocab size                | TBD   |
| Max sequence length       | TBD   |
| Position embedding        | TBD   |
| Batch size                | TBD   |
| Train steps               | TBD   |
| Learning rate decay       | TBD   |
| Warmup steps              | TBD   |
| Learning rate             | TBD   |
| Adam $\epsilon$           | TBD   |
| Adam $\beta_{1}$          | TBD   |
| Adam $\beta_{2}$          | TBD   |
| Dropout                   | TBD   |
| Attention dropout         | TBD   |

---

## Phase 2: Corpus Ablation Scripts

**Objective:** Create a set of Python scripts to perform the specified ablations on the BabyLM training corpus. Each script will take the raw corpus as input and output a modified version.

### 2.1. Ablation 1: Remove Expletives
- **AI-TASK:** Implement the following procedures in a Python script using `spacy`. This script will identify and remove non-referential expletive subjects (`it`, `there`).

```python
# Procedure 1: FindDummyPronouns(corpus)
# 1. Load SpaCy NLP model with a dependency parser
# 2. Initialize D an empty list for dummy pronouns
# 3. for each sentence in corpus do
# 4.   doc = process(sentence, NLP model)
# 5.   for each token in doc do
# 6.     if token.dep_ == 'expl' and token.head.pos_ == 'VERB':
# 7.       Add token to D
# 8.     end if
# 9.   end for
# 10. end for
# 11. return D

# Procedure 2: ConfirmNonReferential(corpus)
# 1. Load SpaCy NLP model with parser and coreference resolver
# 2. Initialize D_confirmed an empty list
# 3. potential_dummies = FindDummyPronouns(corpus)
# 4. for each token in potential_dummies do
# 5.   context = sentence containing token + preceding sentence
# 6.   doc = process(context, NLP model)
# 7.   clusters = doc.coref_clusters
# 8.   has_referent = False
# 9.   for each cluster in clusters do
# 10.    if token is in cluster then
# 11.      has_referent = True
# 12.      break
# 13.    end if
# 14.  end for
# 15.  if not has_referent then
# 16.    Add token to D_confirmed
# 17.  end if
# 18. end for
# 19. return D_confirmed
```

### 2.2. Ablation 2: Impoverish Determiner Morphology
- **AI-TASK:** Implement a script that replaces all determiners (`DET`) with the word 'the'.

```python
# Procedure 3: ImpoverishDeterminers(text)
# 1. Load spaCy NLP model with a POS tagger
# 2. Initialize modified_parts an empty list
# 3. doc = process(text, NLP model)
# 4. for each token in doc do
# 5.   if token.pos_ == 'DET':
# 6.     append 'the' to modified_parts
# 7.   else:
# 8.     append token.text to modified_parts
# 9.   end if
# 10. end for
# 11. result = join_with_spaces(modified_parts)
# 12. return result
```

### 2.3. Ablation 3: Remove Articles
- **AI-TASK:** Implement a script that removes basic articles ('a', 'an', 'the').

```python
# Procedure 4: RemoveArticles(text)
# 1. Load spaCy NLP model with a POS tagger
# 2. Initialize modified_parts an empty list
# 3. doc = process(text, NLP model)
# 4. for each token in doc do
# 5.   is_article = token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']
# 6.   if not is_article:
# 7.     append token.text_with_ws to modified_parts
# 8.   end if
# 9. end for
# 10. result = join(modified_parts)
# 11. return result
```

### 2.4. Ablation 4: Lemmatize Verbs (Infinitival Form)
- **AI-TASK:** Implement a script that converts all verbs to their base/infinitive form (lemma).

```python
# Procedure 5: LemmatizeVerbs(text)
# 1. Load spaCy NLP model with POS tagger and lemmatizer
# 2. Initialize modified_parts an empty list
# 3. doc = process(text, NLP model)
# 4. for each token in doc do
# 5.   if token.pos_ == 'VERB':
# 6.     append token.lemma_ to modified_parts
# 7.   else:
# 8.     append token.text to modified_parts
# 9.   end if
# 10. end for
# 11. result = join_with_spaces(modified_parts)
# 12. return result
```

### 2.5. Ablation 5: Remove Subject Pronominals
- **AI-TASK:** Implement a script that removes all pronouns identified as nominal subjects (`nsubj`).

```python
# Procedure 6: RemoveSubjectPronominals(text)
# 1. Load spaCy NLP model with POS tagger and dependency parser
# 2. Initialize modified_parts an empty list
# 3. doc = process(text, NLP model)
# 4. for each token in doc do
# 5.   is_subj_pronoun = token.pos_ == 'PRON' and token.dep_ == 'nsubj'
# 6.   if not is_subj_pronoun:
# 7.     append token.text_with_ws to modified_parts
# 8.   end if
# 9. end for
# 10. result = join(modified_parts)
# 11. return result
```

---

## Phase 3: Experimental Pipeline

**Objective:** Create a main script to manage the training and evaluation of models for each experiment.

### 3.1. Experiment Design
- **AI-TASK:** Create a master script that can run experiments based on the following design. The script should accept an experiment number as an argument, apply the correct combination of ablations, and then proceed with training and evaluation.

| Exp. | No Expletives | Poor Determiner | No Articles | Infinitive Verbal | No Pronominal Subjects |
| :--- | :-----------: | :-------------: | :---------: | :---------------: | :--------------------: |
| 1    |       ✓       |        X        |      X      |         X         |           X            |
| 2    |       ✓       |        ✓        |      X      |         X         |           X            |
| 3    |       ✓       |        X        |      ✓      |         X         |           X            |
| 4    |       ✓       |        X        |      X      |         ✓         |           X            |
| 5    |       ✓       |        X        |      X      |         X         |           ✓            |
| 6    |       ✓       |        ✓        |      X      |         ✓         |           X            |
| 7    |       ✓       |        ✓        |      ✓      |         ✓         |           ✓            |

### 3.2. Training Workflow
- **AI-TASK:** Develop the training loop.
  1. **Tokenizer:** For each experiment, train a new `SentencePiece` tokenizer on the corresponding ablated corpus.
  2. **Data Preparation:** Group the tokenized text into lines of 1000 tokens.
  3. **Model Initialization:** Instantiate a GPT-2 model with random weights (use a fixed seed for reproducibility).
  4. **Training:** Train for 20 epochs.
  5. **Checkpointing:**
     - During the first epoch, save checkpoints at log-steps (1, 2, 4, 8, 16...).
     - After the first epoch, continue saving at log-steps.
     - Save a checkpoint at the end of each epoch.
     - Save the final model.
  6. **Logging:** Use `wandb` to track loss, learning rate, and other metrics during training.

### 3.3. Evaluation Workflow
- **AI-TASK:** Develop the evaluation script.
  1. **Load Model:** Load a trained model checkpoint.
  2. **Load Stimuli:** Load the evaluation stimuli (e.g., from Table 1 in the source document) and the BLIMP dataset.
  3. **Calculate Surprisal:** For each stimulus pair (preferred vs. dispreferred), calculate the surprisal at the specified hotspot.
     - Surprisal: $S(w_i) = -\log_2 P(w_i | w_1, ..., w_{i-1})$
  4. **Calculate Surprisal Difference:** Compute the difference in surprisal between the preferred and dispreferred sentences.
  5. **Aggregate Results:** Average the surprisal differences for each stimulus set.
  6. **BLIMP Evaluation:** Run the model on the full BLIMP benchmark to assess general linguistic performance.
  7. **Output:** Save results to a structured format (e.g., CSV or JSON) for analysis.

---

## Phase 4: Human Baseline (Experiment 8)
- **AI-TASK (Planning):** Design a framework for a human experiment. While code cannot run the experiment, it can be used to:
  1. Generate the stimuli presentation interface (e.g., a simple web page).
  2. Create a script to collect and anonymize participant responses.
  3. Develop an analysis script to process the human data and compare it with model performance.
