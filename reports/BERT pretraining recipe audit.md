# A plateau-blind sweep stranded our BERTs

No: our from-scratch BERT-large models are not pretrained correctly, and the fault lies in the optimization budget, not the code. Almost every knob we can check against the primary literature matches canonical practice or is harmless: the 24L/1024/16 shape, 15% 80/10/10 masking, Adam eps 1e-6, weight decay 0.01, clip 1.0, fp16 with GradScaler, the collator and the tokenizer. Two knobs do not. **The production peak learning rate of 3.75e-5 is below every published BERT pretraining peak and falls inside BERT's own fine-tuning range.** And 1000-token chunks shrink a 30-epoch budget to **14,820 optimizer steps**, with the LR decaying linearly to zero. Healthy BERTs sit on exactly the ~6-nat plateau we observe until a syntactic-attention transition, which comes at roughly 20–30K steps at LR 1e-4 for a 12-layer model. By that crude yardstick our production run accumulated about a fifth of the learning-rate "distance" a healthy run covers before it escapes. The rater probe stopped, with its LR annealed to zero, at the step where a healthy run starts to leave the plateau. The deeper cause sits upstream. The hyperparameter sweep scored short trials that were all still on the plateau, and its five winners differ by **0.015 nats**, so it could not see the learning rates that escape later. The literature supports this order of fixes for the 90M-word cells: pre-LN with an LR of 3e-4 to 1e-3, then 128-token sequences with separators to get tens of thousands of steps, then hyperparameter selection gated on actually leaving the plateau. For a 5–10B-word rater, the pre-LN BERT-large recipe of Izsak et al. and Wettig et al. applies almost unchanged. Every bert_large production and Stage-1 run, every evaluation built on them, and the bert_large sweep winners should be retired. The pretrained raters (robbi, bert-large-wwm) are untouched.

## Our recipe is canonical in every knob except the learning-rate budget

Set against the canonical BERT-large-class recipes, our configuration is unremarkable almost everywhere. The model shape, GELU, eps 1e-6, weight decay 0.01, clip 1.0, dropout of order 0.1 and 15% masking all appear in BERT, RoBERTa, ELECTRA and DeBERTa ([BERT](https://arxiv.org/abs/1810.04805), [RoBERTa](https://arxiv.org/abs/1907.11692), [ELECTRA](https://arxiv.org/abs/2003.10555), [DeBERTa](https://arxiv.org/abs/2006.03654)). The learning rate is the outlier. **Every pretraining peak we found is 1e-4 or higher.** It is 1e-4 at batch 256–1024 sequences, 2e-4 to 4e-4 at 2K–8K, and 1e-3 to 1e-2 in the budget recipes that use pre-LN, NormFormer-style norms or LAMB. The BERT paper's recommended *fine-tuning* LRs are **5e-5, 3e-5 and 2e-5**, so our pretraining peak sits in the fine-tuning band ([BERT App. A.3](https://arxiv.org/abs/1810.04805)). The original authors state plainly that "the learning rate we used in the paper was 1e-4" ([google-research/bert](https://github.com/google-research/bert)).

| Recipe | Norm | Peak LR | Warmup | Tokens/step | Steps | Reported endpoint |
|---|---|---|---|---|---|---|
| BERT-large ([Devlin 2019](https://arxiv.org/abs/1810.04805)) | post-LN | 1e-4 | 10K | ~32K (len 128, 90% of steps), then 131K (len 512) | 1M | MLM ppl 3.23 (1.17 nats) |
| RoBERTa-large ([Liu 2019](https://arxiv.org/abs/1907.11692)) | post-LN | 4e-4 | 30K | 8K × 512 | 500K | — |
| Megatron-BERT 336M ([Shoeybi 2019](https://arxiv.org/abs/1909.08053)) | post-LN | 1e-4 | 10K | 1024 seq | 2M | val ppl 1.58 (n-gram masking + SOP) |
| ELECTRA's BERT-Large / DeBERTa-large | BERT-style | 2e-4 | 10K | 2K seq | 464K / 1M | — |
| 24hBERT, BERT-large ([Izsak 2021](https://arxiv.org/abs/2104.07705)) | pre-LN | 2e-3 | 6% | 4096 × 128 | ~23K | val loss 1.717 |
| Cramming, base ([Geiping & Goldstein](https://arxiv.org/abs/2212.14034)) | pre-LN | 1e-3 | one-cycle | ~4032 × 128 | 24 h, 1 GPU | ~1.9 |
| LTG-BERT, 100M words ([Samuel 2023](https://arxiv.org/abs/2303.09859)) | NormFormer | 1e-2 (LAMB) | 1.6% | 4.2M | 31,250 | BLiMP 83.4–84.2 |
| **Ours, production** | post-LN | **3.75e-5** | 976 (6.6%) | 256 × 1000 | **14,820** | train 5.62, held-out 6.51 |
| **Ours, rater probe** | post-LN | 1e-4 | 1,200 (6%) | 256 × 512 | 20,000 | ~6.1, flat from step 4K |

The step count is the second outlier. The only recipes in our 10–30K-step range compensate with LRs of 1e-3 or more under pre-LN or NormFormer-style norms (24hBERT, Cramming, LTG-BERT) or under LAMB. RoBERTa's own batch study shows the same trade: **fewer steps need a larger LR** (1e-4 for 1M steps at batch 256, 1e-3 for 31K steps at batch 8K) ([RoBERTa Table 3](https://arxiv.org/abs/1907.11692)). In tokens our batch of 256 × 1000 is about twice BERT's 256 × 512. Fairseq's own LR-for-batch table (256 → 1e-4, 2048 → 5e-4) puts the batch-matched LR at or above 1e-4 ([fairseq README](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md)). Our 3.75e-5 is roughly 3–5× too low even before the short horizon is considered.

The BabyLM literature points the same way at our data scale. Every strong MLM submission there is a 12-layer NormFormer-style model with a 16K-or-smaller vocabulary, 128-token sequences for most of training, multi-million-token batches, a very high LAMB LR and hundreds of epochs. ELC-BERT trained "for over 450 epochs" for its Strict submission ([Findings 2023](https://arxiv.org/abs/2504.08165), [GPT-BERT](https://arxiv.org/abs/2410.24159)). The closest analogue to our stack is a plain HF-style post-LN BERT-base trained with AdamW at 1e-4, 2,000 warmup steps and small batches for 30 epochs on the 10M-word track. It reaches **BLiMP ≈ 79** ([McGill BabyLM](https://aclanthology.org/2023.conll-babylm.18)). The 2025 official baseline gets a working model from just **10 epochs (≈1.6B tokens)** ([Findings 2025](https://aclanthology.org/2025.babylm-main.28.pdf)). Data scarcity is not the limit: our 3.8B tokens exceed the 2025 baseline's 1.6B.

## A loss of 5.6–6.1 is the plateau every healthy BERT crosses, not an endpoint

The best-documented trajectory comes from Chen et al. They trained BERT-base at LR 1e-4 with 10K warmup and batch 256 × 512, which is exactly our rater probe's geometry. The run sits on a plateau near **6.0 nats** and then shows a "precipitous drop" that "consistently" spans **20K–30K steps** across seeds. In their Figure 1a the loss falls to roughly 2.5–3.3 by 25–30K steps and ends at **1.77** at 100K ([Chen et al.](https://arxiv.org/abs/2309.07311)). The drop coincides with the abrupt formation of syntactic attention structure (SAS): attention-derived parse UAS jumps from ~0.05–0.09 to ~0.3–0.4, and BLiMP rises from ~0.5–0.6 to ~0.7–0.8 shortly after. Izsak et al. show the same shape for BERT-large under default hyperparameters (LR 1e-4, batch 256). Validation loss holds at **~6.0–6.1 from about hour 1 to hour 3.3** before dropping, while their tuned pre-LN recipe leaves 6.0 within about half an hour. The larger model has the longer plateau ([Izsak Fig. 2](https://arxiv.org/abs/2104.07705)). Izsak et al. even turn this into a pruning rule: configurations that have not reached validation loss 6.0 after 3 hours are discarded. Converged BERT-large-class models end far lower: **1.17 nats** in BERT's Table 6, **1.72** for 24hBERT, **~1.9** for Cramming, and MLPerf scores BERT-large on reaching **72% masked-LM accuracy** ([BERT](https://arxiv.org/abs/1810.04805), [Cramming](https://arxiv.org/abs/2212.14034), [MLCommons](https://mlcommons.org/benchmarks/training/)).

What a model knows on the plateau matches our diagnostics closely. DecBERT finds BERT-base's plateau is a **bag-of-words phase** in which the model has not yet learned to use position information. Models with and without position embeddings have nearly the same loss there ([DecBERT](https://arxiv.org/abs/2204.08688)). In shallow transformers, plateaus come with repetition bias and nearly parallel hidden states (cosine > 0.9), and the slow learning of attention maps is the bottleneck ([Gopalani & Hu](https://arxiv.org/abs/2506.13688)). Our final production checkpoint fits this profile:

- **Held-out masked loss 6.51**, about 0.9 nats under the unigram cross-entropy of 7.42.
- **13.8% accuracy**, against MLPerf's 72% target.
- **"The" predicted at 56% of masked positions.**

The model has learned token frequencies and a little local context. It has not made the transition that Chen et al. tie to grammatical capability. This matters for the study: BERT's preferences between overt and null subjects, scored by pseudo-log-likelihood on our stimuli, come from a model that has not reached grammatical competence.

A rough heuristic shows how far short each run fell. Adam moves each parameter by roughly the learning rate per step, so the sum of the LR over all steps roughly limits how far the weights can travel. No paper tests this as a law, so treat it as a yardstick, not a measurement. The comparison:

| Run | Sum of LR over all steps |
|---|---|
| Chen's BERT-base, at its 20K-step drop | ~1.5 |
| Our rater probe, at 20K steps | ~1.0 |
| Our production run, over all 14,820 steps | **~0.28** |

On top of that, both of our schedules decay to zero, which removes the step size just when an escape would be expected. The rater probe therefore reads as **truncated rather than failed**. Its 6.1 plateau matches Chen's ~6.0, but it stopped at the step where Chen's run starts to drop, with its LR near zero and twice Chen's depth. Our short races were also too short to tell recipes apart. At 2,000 steps of 32 × 512 tokens (≈33M tokens), both the 4-layer model at 5e-4 and the 12-layer model at 1e-4 were still above 6.4. Even Cramming's tuned pre-LN recipe sits near **7 nats at 30–50M tokens** and only reaches ~6 at 70–100M and ~4 near 200M (read from Cramming's Fig. 1, approximate) ([Cramming](https://arxiv.org/abs/2212.14034)). A race that can tell recipes apart needs at least several hundred million tokens.

## The sweep selected the learning rate it could not see past

Our sweep record explains how 3.75e-5 got into production. The bert_large Bayes sweep searched LR from 1e-5 to 1e-3 and ranked trials by minimum held-out masked perplexity ([sweep config](../configs/sweeps/bert_large_en.yaml)). Each trial ran for at most six epochs with early-stop patience 3. That is at most about 3,000 optimizer steps at an effective batch of 256 sequences. The baseline config itself says patience usually fires after epoch 4–5 and calls a non-plateau trial "rare" ([baseline config](../configs/sweeps/baselines/bert_large_en.yaml)). **All five extracted winners have LRs between 3.4e-5 and 4.2e-5, with held-out perplexities of 336.6–341.8, or 5.82–5.83 nats** ([sweep winners](../data/sweep_winners/bert_large_en.json)). The best and fifth-best differ by 0.015 nats, and all sit on the plateau. At a few thousand steps, Chen and Izsak predict that no learning rate escapes. Higher LRs on a 24-layer post-LN model with a warmup of a few hundred steps are also the regime Xiong et al. found unstable. They trained post-LN BERT-base at 1e-4 with 10K warmup but found the optimization "diverged" at 3e-4, while pre-LN trained at 3e-4 without warmup ([Xiong et al.](https://arxiv.org/abs/2002.04745)). A plateau-scored search can therefore reward whichever low LR hovers smoothest. Production then took that peak, kept its warmup ratio, and stretched the schedule over 30 epochs. Izsak et al. warn that the LR schedule should be "synchronized" with the real budget, and this transfer from a short horizon to a long one ignores that warning ([Izsak](https://arxiv.org/abs/2104.07705)). The production agent maps h0–h4 directly onto these five ranked winners ([production_agent.py](../scripts/production_agent.py)). So every bert_large hyperparameter slot shares the same plateau recipe, and the wave-2 agent reads the same file ([wave2_agent.py](../scripts/wave2_agent.py)).

Two architectural and data choices plausibly lengthen the plateau, although neither is proven as a cause.

**Post-LN at 24 layers.** It is canonical for BERT-large: Megatron found the original LayerNorm ordering "performs well on the 336M model" and needed rearranging only beyond that size ([Megatron](https://arxiv.org/abs/1909.08053)). But it caps the usable LR. Cramming reports that "the aggressive learning rate schedule can only be used when the model also contains pre-normalization". Its crammed recipe on the original architecture collapses to MNLI 35.4 ([Cramming Table 5](https://arxiv.org/abs/2212.14034)). Deep post-LN failure is documented mostly in translation models, through vanishing lower-layer gradients and rank collapse ([DeepNet](https://arxiv.org/abs/2203.00555), [Noci et al.](https://arxiv.org/abs/2206.03126)). There is no evidence that a 24-layer post-LN encoder at 1e-4 fails outright; it is simply slow.

**1000-token contiguous chunks without separators.** No recipe we found uses them. The only plateau workaround in the public HF threads is shorter, line-level sequences, which is an anecdote rather than a confirmed fix ([HF forum](https://discuss.huggingface.co/t/the-loss-plateau-of-pratraining-bert-using-run-mlm-py/8492)). On conversation-heavy BabyLM data, packing utterances without separators cut BERT-base BLiMP from **79.08 to 62.57** ([McGill BabyLM](https://aclanthology.org/2023.conll-babylm.18)). One unanswered forum post describes a near-twin of our setup: a 24-layer, 1024-wide BERT with a 50K vocabulary at LR 5e-5 that stalled near 8.0 ([HF forum 54858](https://discuss.huggingface.co/t/training-loss-sudden-spike-after-8-hours-of-pre-training-a-bert-model/54858)).

Most other suspects can be ruled out with evidence:

- **Weight decay on LayerNorm and bias parameters** deviates from Google BERT and the HF Trainer, but at our cumulative LR it shrinks weights by a factor of only exp(−0.01 × 0.28) ≈ 0.997 ([BERT optimization.py](https://github.com/google-research/bert/blob/master/optimization.py)).
- **fp16** is not implicated: autocast keeps LayerNorm, softmax and cross-entropy in fp32 ([PyTorch AMP](https://docs.pytorch.org/docs/2.5/amp.html)).
- **The 2^23 loss scale** is just maximum growth. The scaler starts at 2^16 and doubles after every 2,000 steps without overflow, so 2^23 follows from 14,000 clean steps and says nothing about how small the gradients are.
- **Omitting [CLS]** had "no impact" in Cramming, and dropping the separator was "minimal" there. The cost is packing *unrelated* text without any boundary, not the missing [CLS] ([Cramming](https://arxiv.org/abs/2212.14034)).
- **Repeating data for 30 epochs** is modest next to LTG-BERT's roughly 1,000 epochs over the 100M-word BNC ([LTG-BERT](https://arxiv.org/abs/2303.09859)).
- **Code bugs**: our own checks found the tokenizer, the collator, `padding_idx`, SDPA masking and gradient checkpointing (a silent no-op) clean.

One loose end points the same way. The sweep winner's best held-out masked loss was 5.82 within six epochs, yet the 30-epoch production checkpoint measures 6.51 on held-out text, while its train loss reached 5.62. The two evaluations may use different masking protocols. If they match, the production model got about 0.7 nats *worse* on unseen text while it memorized repeated training text.

Diagnostics would sharpen the mechanism but are not needed for the decision. The most informative one is to continue the rater probe at constant 1e-4 to 50–80K steps. A drop would confirm truncation, as in Chen's run. Per-layer gradient RMS compared with Adam's eps, a context-shuffle probe, per-head attention entropy and Chen's UAS probe would separate a slow plateau from post-LN gradient vanishing or rank collapse ([Chen et al.](https://arxiv.org/abs/2309.07311), [Zhai et al.](https://arxiv.org/abs/2303.06296)).

## Fix the step budget and normalization before anything else

The two use cases need different templates. BabyLM-scale evidence governs the 90M-word cells. Web-scale recipes developed on Wikipedia, Books and C4 transfer almost directly to a rater trained on a RoBERTa-mirror corpus of 5–10B words.

### Production cells: pre-LN, short sequences, gated selection

The first priority is **pre-LN (with a final LayerNorm) and a peak LR of 3e-4 to 1e-3**. This is the change with the most direct evidence for leaving the plateau quickly:

- Pre-LN BERT-base trained without warmup at the 3e-4 that diverged post-LN ([Xiong et al.](https://arxiv.org/abs/2002.04745)).
- Pre-LN BERT-large at 1e-3 to 2e-3 was the best 24-hour configuration ([Izsak](https://arxiv.org/abs/2104.07705)).
- Wettig et al. call pre-LN "essential for the high learning rate" ([Wettig et al.](https://arxiv.org/abs/2202.08005)).

It also fixes the selection problem structurally. A post-LN run at 1e-4 needs 20–30K+ steps to escape, which a 3-epoch proxy can never observe. A pre-LN run at 1e-3 leaves the plateau within a few percent of its budget. The switch also aligns BERT with our GPT-2 arms, which are pre-LN by design ([GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)). All BERT runs are being redone anyway, so the switch costs no comparability with previous BERT results.

The second priority is **128-token sequences with separators between documents or utterances for most of training**. An optional final phase at 512 can follow, as in BERT, LTG-BERT and the 2025 BabyLM baseline, which uses 128 for 60% of steps, 256 for 20% and 512 for 20% ([BERT](https://arxiv.org/abs/1810.04805), [Findings 2025](https://aclanthology.org/2025.babylm-main.28.pdf)). At 1024 × 128 = 131K tokens per step, the same 3.8B tokens become **~29K optimizer steps instead of 14.8K**. Short sequences also avoid the instability that long early sequences cause at high LR ([Li et al.](https://arxiv.org/abs/2108.06084)). And they remove the conversational-packing penalty McGill measured. Sequence length is part of the content-addressed cache key, so the tokenized and chunked caches must be rebuilt as pre-run jobs.

The third priority is **re-selecting hyperparameters with an escape gate**. The locked 3-epoch sweep horizon is about 380M tokens at this batch, which is enough for a working pre-LN recipe to be well below 6. The gate should prune any trial whose held-out masked loss has not broken clearly below the 5.8–6.5 band our stalled runs occupied, following Izsak's 6.0 rule. Surviving trials should be ranked on post-escape loss and masked-token accuracy, not on plateau perplexity.

Fourth come free hygiene fixes:

- β2 = 0.98 (RoBERTa's choice "to improve stability") and eps 1e-6.
- No weight decay on LayerNorm and bias parameters.
- `pad_token_id` set to the real [PAD] id.
- Dropout 0.1 for a multi-epoch regime.
- Weight decay of 0.1 rather than 0.01. LTG-BERT found 0.01 "substantially worse" on 100M words and called the setting "crucial for masked language modeling", though that ablation used LAMB and also changed the LR ([LTG-BERT](https://arxiv.org/abs/2303.09859)).

Last and optional: a 16–32K vocabulary at the next tokenizer retrain. LTG-BERT's rule of keeping the largest vocabulary in which at least 95% of tokens appear 100 or more times gave 16,384; Cramming found 32,768 best on its corpus. The other option is a masking schedule from 30% down to 15%, which Ankner et al. report speeds up BERT-base pretraining by up to 1.89× ([Ankner et al.](https://arxiv.org/abs/2305.15096)).

If the design must keep post-LN, the evidence-backed fallback is BERT's own recipe compressed: LR 1e-4, 10K-step warmup and 256 × 128 sequences, giving roughly **107K steps** over 30 epochs. Its escape step at 24 layers is unknown, and no 3-epoch proxy can verify it.

### Rater: the pre-LN BERT-large budget recipe, plus a long-context finish

For the rater, the documented template is Izsak and Wettig:

- Pre-LN BERT-large.
- AdamW with β (0.9, 0.98), eps 1e-6, weight decay 0.01.
- Peak LR 1e-3 to 2e-3 at batch 4096 × 128, 6% warmup and linear decay synced to the budget.
- About 23K steps, or **~12B tokens**, for a validation loss near **1.72** ([Izsak](https://arxiv.org/abs/2104.07705), [Wettig et al.](https://arxiv.org/abs/2202.08005)).

That token budget fits the plan's 5–10B words at three to four epochs. Data-constrained scaling finds up to about four epochs of repetition costs almost nothing ([Muennighoff et al.](https://arxiv.org/abs/2305.16264)). The ±250-token scoring window makes a 512-token phase mandatory rather than optional. Wettig's finishing phase uses 512 tokens for the last 10% of steps at LR 5e-4 with 10% re-warmup. Giving that phase 20% of training, as the 2025 BabyLM schedule does for 512, is our judgment given that the rater is scored on 500-token contexts. A warmup-stable-decay schedule like ModernBERT's would let a 6-day budget be extended without restarting. Note that ModernBERT-large also plateaued late at 5e-4 and needed a lower-LR restart ([ModernBERT](https://arxiv.org/abs/2412.13663)). LAMB is unnecessary below roughly 16K-sequence batches ([LAMB](https://arxiv.org/abs/1904.00962)). Dropout can go to zero: ALBERT found that removing it "significantly improves MLM accuracy" for large models that do not overfit ([ALBERT](https://arxiv.org/abs/1909.11942)). We would keep the masking rate at 15% at the end of training so the training condition matches single-slot surprisal scoring. Wettig's 40% helps downstream tasks for large models, but our use is likelihood scoring, so this is our inference.

| Setting | (a) 90M-word production cells | (b) 5–10B-word rater |
|---|---|---|
| Normalization | Pre-LN + final LN | Pre-LN (or ModernBERT-style) |
| Optimizer | AdamW 0.9/0.98/1e-6, no WD on LN/bias | Same |
| Peak LR | Gated race over {3e-4, 6e-4, 1e-3} at 131K tokens/step | 1e-3 to 2e-3 at 524K tokens/step |
| Warmup and schedule | 2–6%, linear decay synced to 30 epochs | 6%, linear synced or warmup-stable-decay |
| Sequences | 128 + separators (~90%), optional 512 finish | 128, then 512 for the final 10–20% (LR 5e-4, re-warmup) |
| Batch → steps | 1024 × 128 → ~29K steps for 3.8B tokens | 4096 × 128 → ~23K steps per 12B tokens |
| Weight decay / dropout | 0.1 / 0.1 | 0.01 / 0–0.1 |
| Masking | 15% (optionally 30% → 15%) | 15% at end |
| Launch gate | Held-out loss clearly below 5.8–6.5 within ~1 epoch, still falling | Below 6.0 within the first few % of tokens; target ~1.7–2.0 nats, accuracy toward 0.72 |

The sweep value for (a) comes from Xiong's pre-LN BERT-base at 3e-4 with a similar batch in tokens, and from Izsak and Cramming at 1e-3 to 2e-3 with a 4× larger batch. Interpolating between them is our heuristic, which is why the LR is a gated race rather than a single value.

## Retire every from-scratch BERT result we have

The invalidity follows from the sweep record, not just from one diagnosed checkpoint. All five hyperparameter slots are LR variants within 3.4–4.2e-5 of one plateau recipe. Every run that consumed them shares the failure.

| Result set | Disposition | Reason |
|---|---|---|
| EN production bert_large wave (5 conditions × h0–h4 × seeds 42/137, plus seed-999 extras) | **Invalid** | All slots map to sweep winners at 3.4–4.2e-5; h0-s137 diagnosed at 13.8% accuracy |
| null_subj_v2 BERT evaluations (pseudo-log-likelihood) and every trajectory or figure built on them | **Invalid** | They score a model that never made the SAS transition |
| Stage-1 / condition-matched bert_large cells (12 seeds × 5 conditions at h0, the 2-epoch early seeds, the held-back h1 lane) and their condition-matched eval rows | **Invalid** | Same recipe; the 2-epoch seeds stop even deeper in the plateau |
| bert_large HP sweep winners (EN sweep 2pimhcbm) | **Invalid as an HP source**; do not feed to the v5 or crossed-matrix bert_large arms | Plateau-scored; 0.015-nat spread |
| ES and 10M-word bert_large sweeps | **Suspect** until their winners are checked | Same baseline design: post-LN, 1000-token chunks, ≤6-epoch plateau-scored proxy |
| Cross-architecture contrasts that include bert_large | **Invalid for the BERT row** | They compare a stalled MLM against working causal LMs |
| Rater probe (1e-4, 512 tokens, 20K steps) | **Inconclusive**; not usable for scoring, not evidence against the rater plan | Truncated at Chen's escape window with LR decayed to ~0 |
| Pretrained raters: robbi (roberta-large), BERT 250:1 (bert-large-uncased-wwm) | **Unaffected by this audit** | Not trained by us; BERT 250:1 is already retired for the contraction confound |
| Tokenizer and MLM collator | **Reusable** | 0.03% [UNK]; exact parity with HF's collator. Data caches are re-keyed by the new sequence length and must be rebuilt |

## Conclusion

The failure is a measurement problem more than a training problem. A proxy horizon shorter than a model's phase transition does not just add noise to hyperparameter selection: it selects against the regime that works, because every candidate is judged before the event that separates good recipes from bad ones. The same hazard applies to any future sweep where the target capability arrives through a sudden drop in the loss. The lasting fix is procedural: no wave launches until a pilot run has visibly left the plateau, as measured by held-out loss, masked-token accuracy and the share of "the" predictions, alongside the end-of-run loss we already track.

The scientific stakes are concrete. Until now our BERT arm compared a frequency-driven model against causal LMs that actually learned, so any apparent "bidirectional versus causal" difference in subject-drop preference is uninterpretable. The literature also shows these BERT problems are fixable at our data scale: models trained on 100M words reach BLiMP 83–84, the same as a BERT-base trained on 3B words ([LTG-BERT](https://arxiv.org/abs/2303.09859)). So the rerun is a chance to get a real bidirectional learner into the matrix, not merely a repair.
