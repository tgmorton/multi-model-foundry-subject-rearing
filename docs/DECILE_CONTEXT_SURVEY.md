# Survey: what contexts look like across robbi recoverability deciles

**Status:** plan, 2026-10-22. Stage 1 is in progress.

**Question (Thomas):** what are the contexts actually like as we go from the most recoverable personal subject pronouns (robbi decile 0) to the least recoverable (decile 9)? The survey treats this as corpus linguistics.

**Sources:** four planning agents fed into this plan:
- data inventory;
- quantitative design;
- qualitative design;
- literature, from `~/zele/research`.

The main loop verified the key numbers itself.

## What we already know (verified on all 6,460,988 selected instances)

| | Decile 0 | Decile 9 | All |
|---|---|---|---|
| Gutenberg share | **52.0%** | 12.5% | 22.5% |
| CHILDES share | 17.7% | **37.4%** | 31.2% |
| BNC spoken share | 2.3% | **18.6%** | 10.9% |
| "we" share | 4.1% | **14.6%** | |
| "he" share | **17.3%** | 10.2% | |
| robbi surprisal, median (nats) | 0.001 | 3.0 (range 1.7–29) | |

- **Scope.** The population is nominative subject pronouns only: about 97% `nsubj` and 3% `nsubj:pass`. There is no case or role variable to survey.
- **Saturation.** Deciles 0–4 all sit below 0.086 nats, so p(pronoun) > 0.92 throughout, and their internal order is close to noise. Treat them as bands (B0 = 0–2, B1 = 3–5). Nearly all of the spread is in deciles 6–9.
- **Rater dependence.** Spearman ρ between raters: robbi–bert 0.32, robbi–gpt2m 0.46. Findings won't carry over between raters, and older characterizations do not describe robbi. The claim that "decile 0 = 64% conversational you" came from the gpt2 ranking.

## Risks to address before interpreting anything

1. **Composition.** Gutenberg dominates the top deciles, while CHILDES and BNC dominate the bottom. Every profile is therefore reported both raw and directly standardized to the population genre × form mix. Decile 0-vs-9 differences are split into composition and within-stratum parts (Kitagawa/Oaxaca).
2. **Pretraining memorization in robbi.** Gutenberg (and Wikipedia-like text) is almost certainly in RoBERTa's pretraining data. Two checks:
   - compare Gutenberg's advantage under robbi with its advantage under the clean in-house raters, where memorization was measured as nil;
   - use robbi's MLM loss on *non-pronoun* tokens in the same window as a familiarity covariate.

   This matters beyond the survey, because robbi is the headline selection arm.
3. **Licit-drop confound in the headline result.** The qualitative and literature agents flagged this independently. Decile 9 may be enriched for positions where English never permits omission:
   - embedded clauses;
   - after fronted or wh material;
   - contrastive subjects.

   Decile 0 may overlap with conversational left-edge drop ("(Do you) want some?"). If so, "least recoverable first hurts more" is partly "unnatural deletion sites first". Test this by coding the gap outcome (licit vs ungrammatical string after deletion) and using it as a covariate in the training-outcome analysis.
4. **Right-context-only recoverability.** robbi sees ±250 tokens in both directions, but the learner is causal. Pronouns that are recoverable only from material to their right (agreement, clitics, later mentions) are easy for the rater and invisible to the learner at the gap. The 100K roberta window grid already measures this without new compute.
5. **Other artifacts.**
   - Duplicate formulaic lines: 10–18% of instances sit on lines that recur verbatim.
   - Coverage gaps: unscored instances were pushed into decile 9.
   - Context truncated at document boundaries.
   - CHILDES child speech rises from 14.8% of CHILDES instances in decile 0 to 31.7% in decile 9.

## Data foundation (local, no cluster required)

- **Deciles (the ones actually removed):** S3 `recoverability/analysis/selection_v5/{robbi,bert,gpt2m}/train_90M/<genre>.parquet`. `rank_value` is surprisal and `info_decile` is the decile. **The local `analysis/selection/` is the old BERT-ranked v4; don't use it.**
- **Join key:** `(genre, line_idx, token_i)`.
  - `instance_measures/` covers 99.95% of rows.
  - `line_idx` is the raw line number in `data/raw/train_90M/<genre>.train`.
  - `nlp.make_doc` reproduces `token_i` exactly.
  - A local `en_core_web_trf` parse matches the stored dep/head/lemma on 800 of 800 instances, at about 44 lines/s on CPU.
- **Documents and speakers.**
  - CHILDES, Gutenberg and Simple Wiki carry document headers; CHILDES also joins to `childes_metadata.json` for child and age.
  - CHILDES and Switchboard carry speaker tags.
  - BNC and OpenSubtitles have no document markers.
- **Cluster-only data:** robbi's per-instance entropy and its distribution over competing pronouns. These are needed for the competitor analysis; copy them to S3 with an exempt-size uploader, or re-score.
- **Don't use** `data/annotated_corpus_train_90M/`. It was built with a different parser, is split by sentence, and predates the dependency-label fix.

## Analysis streams

**S1. Population profiles (all 6.46M instances; cheap token-level features).** Stage 1.
- Contraction.
- Agreement informativeness of the right-adjacent verb or clitic: the number of pronoun forms compatible with it (am → 1; is/'s → 3; past or modal → all).
- Position: line-initial, after and/but/so, after wh, inverted question, tag question.
- Line length and duplicates.
- Speaker and speaker change; CHILDES child vs adult, and age.
- Available left context.
- Profiles per decile, raw and standardized, with document-cluster bootstrap CIs. Repeated under bert and gpt2m as a robustness check.

**S2. Keyness.** Monroe et al. (2008) informative-Dirichlet log-odds of slot frames (l2 l1 _ r1 r2) and position-tagged tokens.
- Each decile is compared against the rest, within genre × person strata.
- Each feature counts at most once per document.
- The results seed the list of formulaic frames.

**S3. Discourse features (stratified sample).** The sample is decile × genre × person group, 240 cells × about 500, roughly 120K instances, with a cap of 5 per document per cell and Horvitz–Thompson weights. Features:
- **Topic continuity / switch reference.** The literature's strongest predictor (Givón; the Spanish variationist work; Frascarelli). Is the referent the previous clause's subject? For I/you, is it the same speaker?
- **Referential distance and prior-mention count.** Measured left and right separately.
- **Competitors.** Same person, number and gender, intervening between the antecedent and the slot.
- **Referent type.** Speaker, addressee, inclusive "we", generic, anaphoric, exophoric, or expletive "it". Expletive, referential and event-reference "it" are split.
- **Turn structure.** Question → answer deixis flips (Yang et al. 2021).
- **Clause environment.** Main vs embedded; coordination (same-subject second conjunct); contrast or focus ("too", "myself", parallelism).
- **Head verb class.** Mental, speech, modal, copula, implicit causality.

Parse on CPU locally or as a run-and-exit cluster job.

**S4. Recovery horizon and direction.** From the roberta L×R grid on the 100K locality sample:
- the smallest window that recovers the pronoun;
- local (≤4 tokens) vs clause (≤32) vs discourse (64–250) vs never;
- right-only flags, compared with `causal_backward_by_person.csv`.

**S5. Rater checks (early).**
- The memorization tests in risk 2.
- Robbi's competitor distribution for deciles 7–9 (I↔you role ambiguity, he↔she referent choice, we↔they group membership, it↔they number).
- S1 profiles reproduced under bert and gpt2m.

**S6. Qualitative reading and coding.**
1. **R1 open reading by Thomas.** About 290 KWIC items: 8 bands × 6 genres × 6, with person quotas. Items are hashed and blind to decile; the pronoun is blanked until revealed, with turn view and full-window view available.
2. **Codebook** (operational definitions are in the qualitative agent's report):
   - C1 cloze: top guess and confidence;
   - C2 gap outcome after deletion: ungrammatical, licit (imperative, left-edge drop, diary drop, coordination ellipsis) or reanalysis;
   - C3 position;
   - C4 referent type;
   - C5 anchor location;
   - C6 topic continuity, plus a competitor flag;
   - C7 discourse function;
   - C8 formula;
   - C9 recovery cue (clitic, agreement, coindexed pronoun, structural copy, turn structure, prior mention, formula, world knowledge, none);
   - C10 text anomaly;
   - C11 CHILDES speaker metadata.
3. **Gold set.** 480 items double-coded and adjudicated, split into 160 dev and 320 test. Krippendorff's α ≥ 0.70 per core code, and agreement is reported per band.
4. **LLM coding of about 15K items.**
   - Two passes: blind cloze, then reveal.
   - Each cue must be quoted, and a script verifies the quote appears in the context.
   - A second LLM model checks for disagreement.
   - 5% hidden gold anchors catch drift.
   - Prevalences are corrected using the test-set confusion matrix.

**S7. Models.**
- **Response:** robbi on the logit scale, which spreads out the saturated top.
- **Full population:** document fixed effects with clustered SEs.
- **On the S3 sample:** mixed models with document and head-lemma random intercepts; ordinal checks with clmm.
- **LightGBM + TreeSHAP** with GroupKFold by document. Report incremental R², stepping from controls → morphosyntax → discourse → register/lexical → information features.
- Information features come from in-house raters only, never robbi's own outputs, to avoid circularity.

## Outputs

1. Composition strip: genre × form by decile.
2. Feature-profile small multiples, raw vs standardized.
3. Keyness ladder: deciles 0, 3, 6 and 9 by genre.
4. Agreement heatmap: right verb form × pronoun form.
5. Mention-distance and topic-continuity curves by person group.
6. Composition vs within-stratum decomposition bars.
7. SHAP group importance and the incremental-R² table.
8. Rater-robustness matrix.
9. A decile "portrait" per band: code prevalences, recovery-cue mix, horizon distribution, and four exemplars chosen by a declared rule.
10. A browsable HTML concordance with filters.

## Order of work

1. **Stage 1 (running).**
   - S1 population features and profiles, including the bert and gpt2m versions.
   - S2 keyness.
   - An audit: coverage, items above 10 nats, join and tokenizer verification.
   - R1 concordance sample and viewer.
2. **Stage 2.**
   - S5 rater checks: memorization and competitors. This needs cluster data or a re-score job.
   - S4 horizon analysis.
   - Thomas reads R1, which produces codebook v0.
3. **Stage 3.**
   - S3 stratified sample and parse, with the discourse features.
   - Gold-set calibration.
4. **Stage 4.**
   - LLM coding.
   - S7 models.
   - The decile portraits.
   - The licit-drop and right-only covariates in the training-outcome analysis (risks 3 and 4).

## Decisions for Thomas

1. **A second human coder for the gold set** (about 20 h each). Who?
2. **A human cloze, optional** (5 native speakers × 480 items via Prolific; needs IRB). It is the only rater-independent check of recoverability (Tily & Piantadosi style).
3. **Covariates in the training-outcome analysis.** Should the coded licit-drop share and the right-context-only share enter it?

## Literature (where to look)

From the literature agent. The folder's coverage is thin: 77 of 309 PDFs are mislabeled, and the accessibility and predictability canon is missing. Many items below therefore come from the agent's memory and are marked [mem].

Priorities for the survey, each with its prediction:
1. **Topic continuity / switch reference.**
   - Prediction: continuations sit in deciles 0–2 and switches in 7–9.
   - Sources: Givón 1983 [mem]; Otheguy & Zentella 2012 [mem]; Cameron 1993 [mem]; Frascarelli 2007 [mem].
2. **Referential distance and number of prior mentions.**
   - Prediction: decile 9 is enriched for first mentions and distant antecedents.
   - Sources: Konno 2021, Liu 2017, Shimazu 2020, all in the folder.
3. **Competing referents (same φ-features).**
   - Prediction: this matters for he/she/they and not for I/you.
   - Sources: Ariel 1990 [mem]; Aloraini 2020b/2022, in the folder.
4. **English omission sites.**
   - Root vs embedded, utterance-initial, after fronted or wh material, same-subject second conjunct.
   - Sources: Haegeman 1990/2013 [mem]; Weir 2012 [mem]; Torres Cacoullos & Travis 2014/2019 [mem].
5. **Turn structure × person.**
   - Sources: Yang 2021, in the folder; Huber 2026, in the folder, which reports that 47% of English CDS agents are 1st/2nd-person pronouns.
6. **Generic vs specific reference.**
   - Prediction: generic they/we/you sit in late deciles.
7. **Left vs right context.**
   - Sources: the UID production story uses only left context (Jaeger 2010 [mem]; Kravtchenko 2014 [mem]).
8. **Formulaic frames** (you know, I mean, I think, thank you).
   - Source: Cameron-Faulkner et al. 2003 [mem].
9. **Connectives and implicit-causality verbs.**
   - Sources: Kehler & Rohde 2013 [mem]; Fukumura & van Gompel 2010 [mem].
10. **Contrast and focus.**
    - Prediction: late deciles. These are the least natural drops.
11. **Function of "it".**
    - Source: Guillou 2016, in the folder.
12. **Child-omission predictors as a contrast.**
    - Utterance-initial weak position, VP length, negation, verb frequency.
    - Sources: Freudenthal 2002/2007 and Dey 2025, in the folder; Bloom 1970/1990 [mem].

**Reading to acquire:**
- Tily & Piantadosi 2009
- Arnold 2010
- Ariel 1990
- Gundel et al. 1993
- Otheguy & Zentella 2012
- Torres Cacoullos & Travis 2019
- Weir 2012
- Upadhye, Bergen & Kehler 2020
- Kravtchenko 2014
