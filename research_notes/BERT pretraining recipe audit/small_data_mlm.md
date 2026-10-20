# Small-data (BabyLM-scale) masked-LM pretraining recipes vs. our 24-layer post-LN BERT-large setup

Scope: primary literature on training MLMs from scratch on ~10M-100M words and on small compute (BabyLM 2023/2024/2025, LTG-BERT family, Cramming, academic-budget BERT, masking-rate study), compared against our setup (HF BertForMaskedLM, post-LN, 24L/1024H/16 heads, ~335M params, 50,004 lowercase WordPiece, 1000-token contiguous chunks without special tokens, 15% 80/10/10 masking, AdamW b2=0.99 eps=1e-6 wd=0.01 on all params, peak LR 3.75e-5, 6.6% warmup, linear decay, batch 256x1000 tokens, dropout ~0.06, fp16, 30 epochs ~3.8B tokens, stuck at loss 5.62 vs unigram 7.42).

Method note: most numbers below were read directly from the paper text (PDF text extraction) or from the authors' GitHub files. Where a number is derived by me (arithmetic from cited numbers) it is in an "Inferences" section or marked "(derived)". Items I could not verify are in "Gaps".

---

## 1. What did the strongest MLM-style BabyLM submissions (2023, 2024, 2025) actually use?

### Takeaway
Every top MLM-style BabyLM entry (LTG-BERT, ELC-BERT, Boot-BERT, GPT-BERT) is a 12-layer, base-size (~100-127M params) or small (24-30M params) NormFormer-style model with a SMALL vocabulary (16,384 / 4-8K), short sequences (128 for most of training, 512 at the end, or 256 constant), LAMB with very large LR (5e-3 to 1e-2) and very large token batches (2-4M), and hundreds to ~1000 epochs over the 100M-word corpus. None uses 24 layers, a 50K vocabulary, 1000-token sequences, or 30 epochs at a ~4e-5 AdamW LR. The 2025 rules cap training at 10 epochs, and the 2025 official baseline is still the same 12-layer GPT-BERT recipe.

### Cited Findings

**Winners and scores**
- 2023 Strict winner = ELC-BERT (Charpentier & Samuel), runner-up Boot-BERT; both start from the LTG-BERT architecture, and the organizers note "the backbone architecture plays a large role" — [Findings of BabyLM 2023, Sec. 7.2](https://arxiv.org/abs/2504.08165)
- 2024 Strict AND Strict-Small winner = GPT-BERT (Charpentier & Samuel 2024); 2024 official encoder baseline = LTG-BERT — [Findings of BabyLM 2024, Sec. 4.3 & winners paragraph](https://arxiv.org/abs/2412.05149)
- 2025: baselines are GPT-BERT (Strict/Strict-Small) and GPT-2 small; the paper says winners were architectural/objective innovations (diffusion LM, MoE, RL-interaction) and that the GPT-BERT baselines "remain strong" — [Findings of BabyLM 2025, Sec. 1, 3.1, 4](https://aclanthology.org/2025.babylm-main.28.pdf)
- Strict (100M words) scores, GPT-BERT paper Table 1 (BLiMP / BLiMP-Supp / GLUE / EWoK): encoder-only organizer baseline 69.2 / 66.5 / 68.4 / 51.9; ELC-BERT(2023) 85.8 / 76.8 / 78.3 / 56.3; LTG-BERT(2023) 85.3 / 76.6 / 77.9 / 56.0; GPT-BERT 86.1 / 76.8 / 81.5 / 58.4. Strict-Small (10M): ELC-BERT 80.5 / 67.9 / 75.3 / 51.0; LTG-BERT 80.6 / 69.8 / 74.5; GPT-BERT 81.2 / 69.4 / 76.5 / 54.6 — [GPT-BERT, Table 1](https://arxiv.org/abs/2410.24159)
- Conflict: ELC-BERT's own Table 2 gives BLiMP 85.8 for LTG-BERT-base and 85.3 for ELC-BERT-base (GLUE 77.9 vs 78.3), i.e. BLiMP labels swapped relative to the GPT-BERT table — [ELC-BERT, Table 2](https://arxiv.org/abs/2311.02265) vs [GPT-BERT, Table 1](https://arxiv.org/abs/2410.24159). Either way both are ~85-86.
- 2023 official baselines (Strict BLiMP): OPT-125M 75.3, RoBERTa-base 75.1, T5-base 56.0 — [ELC-BERT, Table 2](https://arxiv.org/abs/2311.02265). The RoBERTa-base baseline used context length 128, constant LR 1e-4, linear warmup over 5,000 steps, batch size 128, AdamW, 20 epochs, and the organizers say these were "loosely inspired" and not tuned — [Findings 2023, Sec. 5.4](https://arxiv.org/abs/2504.08165)

**LTG-BERT (Samuel et al. 2023) — the base recipe, trained on 100M-word BNC**
- Architecture: 12 layers, hidden 768, 12 heads, FF 2,048 (GEGLU, 2/3 width), vocab 16,384 cased WordPiece, NormFormer-style extra LayerNorms, disentangled attention with shared relative position embedding, no FF biases, FF init scaled by 1/sqrt(2(l+1)), init N(0, 2/(5d)); "Preliminary experiments with [standard post-norm] showed that it tends to unpredictably diverge in the later stages of training" — [LTG-BERT, Sec. 4, Table 9](https://arxiv.org/abs/2303.09859)
- Optimization (Table 9): LAMB, peak LR 0.01 -> 0.001 cosine decay, warmup 500 steps (1.6%), beta1 0.9, beta2 0.98, eps 1e-6, LayerNorm eps 1e-5, weight decay 0.1, grad clip 2.0, dropout 0.1 and attention dropout 0.1, 31,250 steps, batch 32,768 sequences x 128 tokens (90% of steps) then 8,192 x 512 (last 10%) = 4,194,304 tokens/step — [LTG-BERT, Table 9](https://arxiv.org/abs/2303.09859)
- Masking: 15% of subwords, 80/10/10; best variant is SpanBERT-style span masking (geometric p=1/3, length mod 10, mean ~2); BLiMP 84.2 span vs 83.4 subword vs 80.1 whole-word; NSP objectives did not help — [LTG-BERT, Sec. 5.1, 7.2, Table 3](https://arxiv.org/abs/2303.09859)
- Ablation (Table 2, deltas vs final model; MNLI / BLiMP): post-norm -0.5 / -0.1 (needed LR 0.005 because default 0.01 diverged); pre-norm -1.3 / -0.9 (LR 0.005); weight decay 0.01 instead of 0.1 -1.4 / -0.7 (that run used LR 0.005); AdamW instead of LAMB -0.9 / -0.5 (needed LR 0.001); linear instead of cosine decay -0.5 / -0.2. Authors: "setting the weight decay correctly is crucial for masked language modeling" on small corpora, and "previous findings of inferior performance of LMs pre-trained on small corpora might be caused by insufficient hyperparameter search" — [LTG-BERT, Sec. 7.1, Table 2](https://arxiv.org/abs/2303.09859)
- Training length: 31,250 steps x 4.19M tokens vs 131,392,103 subwords in the BNC train split; the paper states that even at 1/4 of the steps "every token in BNC is seen about roughly 250 times"; 2x steps gave no gain, 1/2x steps was comparable, 1/4x degraded (MNLI 83.8 vs 85.1; BLiMP 83.2 vs 83.4) — [LTG-BERT, Sec. 7.3, Table 1, Table 3, footnote 12](https://arxiv.org/abs/2303.09859)
- BLiMP 83.4 (subword masking) / 84.2 (span masking) vs BERT-base trained on 3B words 84.2; BERT-base-cased reference MNLI 83.6-84.4 — [LTG-BERT, Table 3](https://arxiv.org/abs/2303.09859)
- Vocabulary rationale: small vocab chosen deliberately: "a larger vocabulary size would lead to a lot of infrequent tokens within our limited corpus"; rule from Gowda & May: largest vocab such that 95% of classes have >=100 examples — [LTG-BERT, Sec. 7](https://arxiv.org/abs/2303.09859)
- Training-code defaults: `train.sh` uses `--batch_size 256 --max_steps 31250` (16 nodes x 8 GPUs, LUMI) — [ltg-bert/training/train.sh](https://github.com/ltgoslo/ltg-bert/blob/main/training/train.sh)
- Organizers' reproduction of LTG-BERT in 2024: "we initially used the code provided ... but we encountered unstable training due to loss spikes with this setup", so they used the HF-released LTG-BERT with the HF Trainer, where "training was still relatively unstable compared to other architectures" — [Findings 2024, Sec. 4.3](https://arxiv.org/abs/2412.05149)

**ELC-BERT (Charpentier & Samuel 2023), BabyLM 2023 Strict winner**
- Base: 98M params, 12L/768H/12 heads/FF 2,048, vocab 16,384; small: 24M, 12L/384H/6 heads, vocab 6,144. Identical LTG-BERT recipe: LAMB, LR 0.01->0.001 (small 0.0141->0.00141; submitted small 0.005 constant), cosine, warmup 1.6%, wd 0.1 (small 0.4), LN eps 1e-7, grad clip 2.0, dropout 0.1; 15,625 steps at batch 32,768 sequences, seq 128 then 512 — [ELC-BERT, Appendix A Table 4](https://arxiv.org/abs/2311.02265)
- Organizers' summary: LTG-BERT-family models "have been trained for a large number of epochs. Charpentier and Samuel (2023) train models for over 450 epochs for their Strict submission, and over 2000 epochs for their Strict-Small submission"; "the winning submission trained on about as many samples as BERT, despite having a training set only about 3% as large"; training for hundreds of epochs "is not cognitively plausible" — [Findings 2023, Sec. 7.2 and Sec. 8](https://arxiv.org/abs/2504.08165)

**Boot-BERT / "Mean BERTs make erratic language teachers" (Samuel 2023), Strict runner-up**
- LTG-BERT backbone (+4 decoder layers for latent bootstrapping); base 127.7M params, vocab 16,384; small 30.4M, vocab 4,096; LAMB, LR 0.005 (base)/0.007 (small) cosine to 10%, warmup 1,000 steps, 62,500 steps, batch 4,096 sequences, constant sequence length 256, wd 0.02-0.2 scheduled, grad clip 2.0, dropout 0.1 — [Boot-BERT, Appendix C Table 5](https://arxiv.org/abs/2310.19420)
- Scores (Strict): BLiMP 86.3, GLUE 79.2 vs LTG-BERT-base 87.2 / 77.8 in the same paper's table; Strict-Small BLiMP 82.2 — [Boot-BERT, results table](https://arxiv.org/abs/2310.19420)

**GPT-BERT (Charpentier & Samuel 2024), BabyLM 2024 winner (both text tracks)**
- Strict: 119M params, 12L/768H/12 heads, FF 2,560, vocab 16,384; Strict-Small: 30M, hidden 384, vocab 8,192. LAMB, LR 0.01->0.001 cosine (small 0.0141->0.00141), warmup 1.6%, wd 0.1, beta2 0.98, eps 1e-8, grad clip 2.0, dropout 0.1, 15,625 steps (small 7,812), batch grows linearly 1M -> 4M tokens, sequence length 128 -> 512 — [GPT-BERT, Sec. 4.1, Appendix A Table 3](https://arxiv.org/abs/2410.24159)
- Masking schedule: masking probability decreased linearly from 30% to the standard 15% over training (following Ankner et al. 2024) — [GPT-BERT, Sec. 2](https://arxiv.org/abs/2410.24159); the schedule idea itself reports up to +0.46 GLUE (BERT-base) and +0.25 (BERT-large) and up to 1.89x pretraining speedup for BERT-base — [Ankner et al. 2024 abstract](https://arxiv.org/abs/2305.15096)
- Mixed-objective training (causal+masked, best ~1:7 causal:masked on BLiMP) beat masked-only on BabyLM scores — [GPT-BERT, Sec. 4.2, Table 1](https://arxiv.org/abs/2410.24159). Note the GPT-BERT submission was trained on a 1:1:1 mix of BabyLM, FineWeb-Edu subset and a third source under the word cap, not on the BabyLM corpus alone — [GPT-BERT, "Training corpus" paragraph, Sec. 3-4](https://arxiv.org/abs/2410.24159)

**Official 2025 baseline (GPT-BERT, 10-epoch cap)**
- 2025 rule: training exposure capped at 100M words (Strict-Small) and 1B words (other tracks), counting repeats; motivation: "larger numbers of training epochs improved model performance, which gives groups with greater computational resources a significant advantage" — [Findings 2025, Sec. 2](https://aclanthology.org/2025.babylm-main.28.pdf); also noted as "The 2025 BabyLM rules cap training at 10 epochs" — [Papusoi & Nisioi 2025](https://aclanthology.org/2025.babylm-main.16.pdf)
- Baseline recipe: ~120M params, 12 layers, 12 heads; batch 131,072 tokens, 12,330 steps, 10 epochs; warmup-cosine-cooldown with max LR 7e-3 (1.6% warmup, final 1.6% linear cooldown); masking ratio linearly 0.3 -> 0.15; sequence length 128 for the first 60% of steps, 256 for the next 20%, 512 for the final 20%. Strict-Small ~31M params, batch 16,384 tokens, 9,914 steps. (The findings text does not name the optimizer for GPT-BERT; the original uses LAMB.) — [Findings 2025, Sec. 3.1](https://aclanthology.org/2025.babylm-main.28.pdf)

**Vanilla BERT-base on small data (closest analogue to our HF BertForMaskedLM): McGill submission (Cheng et al. 2023)**
- BERT-base, uncased WordPiece vocab 32,768, MLM 15%, AdamW, peak LR 1e-4, 2,000 warmup steps, linear decay, dropout 0.1, GELU, bfloat16, effective batch 128 examples, 30 epochs, max sequence length in {32, 128, 512}, trained on the 10M-word Strict-Small corpus; best configs reach BLiMP ~79 (sentence-level, ungrouped, L=128: 79.08) — [McGill BabyLM, Sec. 5 and Table 4](https://aclanthology.org/2023.conll-babylm.18)

**Other small-model evidence**
- ToddlerBERTa (10M-word Strict-Small): RoBERTa-style models from 0.75M to 92M params (xl = 768H/12L/12 heads); "larger models tend to perform better" (largest tried was 92M, 12 layers); best config beat the BabyBERTa defaults (BLiMP 0.7407 vs 0.6660 for one size) — [ToddlerBERTa, Sec. 3, Table 1, Sec. 5](https://arxiv.org/abs/2308.16336), [summary in Findings 2023, App. F](https://arxiv.org/abs/2504.08165)
- Findings 2023, model size: "results are mixed when modifying model size: some participants achieved better results when scaling model sizes up (Cagatan, 2023), while others were able to perform well when using very small models (Proskurina et al., 2023). More controlled studies ... are needed" — [Findings 2023, Sec. 6 (Submissions Summary)](https://arxiv.org/abs/2504.08165)

### Inferences
- The BabyLM-winning recipe is "small vocab + short sequences + huge batches + very high LAMB LR + hundreds of epochs + 12 layers + non-vanilla normalization". Our setup differs on every one of those axes; no single point in the winners' recipe resembles ours.
- Total optimizer steps are in the same order (LTG/ELC/GPT-BERT 15.6K-31K; Boot-BERT 62.5K; 2025 baseline 12.3K; ours ~14-15K derived from 3.8B / 256K tokens), but the winners' per-step token batch is 16x larger (4.2M) or 0.5x (2025 baseline, 131K) than ours (256K), and LR is 2-3 orders of magnitude higher in LAMB units; because LAMB LRs are not comparable to AdamW LRs, the AdamW-comparable anchors are in question 2.
- Winners saw ~450-1000 epochs (derived: 31,250 x 4.19M / 131.4M ~ 1000 epochs for LTG-BERT); the 2025 baseline sees 10 epochs (12,330 x 131,072 = 1.6B tokens, derived). Our 30 epochs / 3.8B tokens is between those two regimes, so epochs/tokens are not the obvious culprit; the 2025 baseline reaches a working model with fewer tokens than we processed.
- The cleanest "vanilla" analogue (McGill: HF-style BERT-base, post-LN, AdamW 1e-4, 2K warmup, 30 epochs, small batch) trains fine on 10M words; the difference from ours is size (12L vs 24L), batch (128 short examples vs 256x1000 tokens), sequence format, and LR/step regime.

### Gaps
- Could not read the BabyBERTa (Huebner et al. 2021) paper body (PDF text unreadable via the fetch tool), so its exact recipe is not included.
- The LTG-BERT repo's `config.py` has no default hyperparameters and `train.sh` forwards extra arguments, so the exact LTG-BERT LR / sequence flags used in practice come from the paper's Table 9, not the repo.
- Exact optimizer for the 2025 GPT-BERT baseline is not stated in the findings paper text; the HF model cards (not read) list hyperparameters.
- No access to 2025 winners' individual papers (diffusion LM, MoE) for MLM-specific recipes.

---

## 2. Compute-efficient MLM: Cramming, academic-budget BERT, masking-rate study — LR, sequence length, model size, masking

### Takeaway
All three efficient-recipe papers converge on: pre-LN (essential for high LR and short/no warmup), short sequences (128), LR 1e-3 to 2e-3 with batch ~0.5M tokens (4,096 x 128), warmup 2-6% with the LR schedule tied to the actual total budget, dropout 0 for single-epoch runs, and no benefit from the original BERT hyperparameters (LR 1e-4, batch 256) at fixed budget. Under those conditions even BERT-large works and is the more sample-efficient choice, but under the original recipe MLM loss stays near 7.

### Cited Findings

**Cramming (Geiping & Goldstein 2023): BERT-base, 24 h on one consumer GPU**
- Data/format: lowercase, WordPiece vocab 2^15 = 32,768 (2^12-2^14 "resulted in worse performance, while larger vocabulary sizes (2^16) were not reliably better"); sequences packed to length 128; single epoch — [Cramming, Sec. 4.1](https://arxiv.org/abs/2212.14034)
- Architecture: pre-LN ("key effect of pre-normalization is to stabilize training and enable larger learning rates and reduced warmup"), no QKV or FF biases, GLU feed-forward, scaled sinusoidal positions, final LayerNorm in head, learned/rotary positions not better; model-shape changes (deep-narrow, funnel, recurrent) gave no gain — [Cramming, Sec. 4.2](https://arxiv.org/abs/2212.14034)
- Scaling observation: "Models with more parameters learn more efficiently, as their MLM loss decreases faster on a per-gradient basis. However, smaller architectures make up for their slower learning efficiency by higher throughput ... all choices reach a MLM loss around 1.9 at the end of training" (24 h, single GPU) — [Cramming, Sec. 4.2, Fig. 1](https://arxiv.org/abs/2212.14034)
- Training: MLM only, 15% masking with 80/10/10 ("no improvement from masking at larger rates, e.g. 40%"); Adam, wd 0.01, beta1 0.9, beta2 0.98, eps 1e-12; grad clip 0.5; one-cycle LR with peak 1e-3; final batch size ~4,032 sequences (x128 tokens, ~0.5M tokens) with a linearly growing batch schedule (loss minimal at 1,536; downstream best at 4,032); dropout disabled in pretraining ("Overfitting is not possible due to the single epoch schedule") — [Cramming, Sec. 4.3](https://arxiv.org/abs/2212.14034)
- Result: crammed BERT (A6000) GLUE avg 78.6 (MNLI 83.9/84.1) vs fully trained BERT-base 80.9; the "normal protocol" original BERT recipe under the same 24 h budget scores GLUE ~52 (MNLI-m 56-59 across the three GPUs) — [Cramming, Tables 3-4](https://arxiv.org/abs/2212.14034)
- Ablation (Table 5): "modifications in training and architecture have to co-occur. For example, the aggressive learning rate schedule can only be used when the model also contains pre-normalization". Crammed recipe with original BERT arch: MNLI 35.4, GLUE 27.7 (failure); original training recipe with crammed arch: MNLI 50.0, GLUE 50.2 — [Cramming, Sec. 5.1, Table 5](https://arxiv.org/abs/2212.14034)
- Appendix Table 10: "Original training recipe" under the same budget ends at MLM loss 7.28 (MNLI 60.65); the Izsak recipe 2.06; larger LR 1.99; one-cycle + larger LR 1.84; Adam with classic WD formulation 7.44, SGD 7.46, RAdam 7.50; masking 40% -> MLM 2.70 (MNLI 81.11) vs 15% (1.84; 82.12); dropout on 1.97 — [Cramming, Appendix Table 10](https://arxiv.org/abs/2212.14034)

**How to Train BERT with an Academic Budget (Izsak et al. 2021): BERT-large, 24 h on 8 x 12GB GPUs**
- Model: BERT-large (24 layers, 1,024 hidden, 16 heads, 4,096 FF) with pre-LN, "train large" because "larger models tend to achieve better performance than smaller models when trained for the same wall-clock time"; AdamW beta1 0.9, beta2 0.98, eps 1e-6, wd 0.01, dropout 0.1, attention dropout 0.1; sequences capped at 128 for the entire pretraining; no NSP; static pre-masked data — [Izsak et al., Sec. 3.1](https://arxiv.org/abs/2104.07705)
- Search: batch {4k, 8k, 16k} sequences (~23k/12k/6k steps in 24 h), LR {5e-4, 1e-3, 2e-3}, warmup proportion, with the LR schedule synchronized to the 24 h budget; configs not reaching val loss <= 6.0 after 3 h were pruned ("this rule removes diverging runs, such as configurations with 0% warmup") — [Izsak et al., Sec. 4.1-4.2](https://arxiv.org/abs/2104.07705)
- Best (Table 3, MLM loss after 24 h): batch 4096, LR 2e-3, warmup 6%: 1.717; LR 1e-3, warmup 2%: 1.717; LR 1e-3, warmup 6%: 1.720; batch 8192, LR 1e-3: 1.722; reference BERT-base default (bsz 256, LR 1e-4, warmup 11.1%) 2.050 at 3 days and BERT-large default 2.318 at 8.3 days — [Izsak et al., Table 3](https://arxiv.org/abs/2104.07705)
- Findings: "synchronizing learning rate warmup and decay schedules with our 24 hour budget greatly improves model performance"; "a preference for smaller batch size ... trade-off between number of samples and number of updates, for which a batch size of 4096 seems to be a better fit"; when total days are larger, use smaller warmup proportion — [Izsak et al., Sec. 4.3](https://arxiv.org/abs/2104.07705)
- Downstream: 24hBERT GLUE test avg 81.1 vs BERT-base 81.5 and BERT-large 83.0 — [Izsak et al., Table 4](https://arxiv.org/abs/2104.07705)

**Should You Mask 15%? (Wettig et al. 2023)**
- "Efficient pre-training recipe" for BERT-large (354M): peak LR 2e-3, warmup 6%, batch 4,096, 23,000 steps, sequence length 128; "using pre-layernorm ... is essential for the high learning rate in Izsak et al. (2021) to work"; RoBERTa BPE tokenizer; all masked positions replaced by [MASK] (80-10-10 "does not perform better for most downstream tasks"); the final 10% of training for SQuAD uses seq 512, LR 5e-4, warmup 10% — [Wettig et al., Sec. 2.2, Appendix A.1 Table 5](https://arxiv.org/abs/2202.08005)
- Masking vs model size: optimal masking is 40% for large, 20% for base, 15% for medium models; BERT-large Table 1: 15% -> validation PPL 17.7, MNLI 84.2, QNLI 90.9, SQuAD 88.0; 40% -> PPL 69.4, 84.5, 91.6, 89.8; 80% -> PPL 1141.4, 80.8, 87.9, 86.2 ("pre-training metrics like perplexity do not correlate well with downstream") — [Wettig et al., Sec. 3, Table 1](https://arxiv.org/abs/2202.08005)
- Contradicting transfer to base models in the cramming regime: 40% masking gave no gain for crammed BERT-base — [Cramming, Sec. 4.3 and Table 10](https://arxiv.org/abs/2212.14034)

**RoBERTa reference points for LR vs batch** (large-scale, but anchors the AdamW LR/batch relation)
- Same base architecture: batch 256 sequences, 1M steps, LR 1e-4 (ppl 3.99); batch 2K, 125K steps, LR 7e-4 (3.68); batch 8K, 31K steps, LR 1e-3 (3.77) — [RoBERTa, Sec. 4.3 Table 3](https://arxiv.org/abs/1907.11692)
- RoBERTa-large pretraining: 24L/1024H/16 heads, LR 4e-4, batch 8K (x512 tokens), warmup 30k steps, Adam beta2 0.98, eps 1e-6, wd 0.01, 500K steps; "very sensitive to the Adam epsilon term" and beta2=0.98 chosen "to improve stability when training with large batch sizes" — [RoBERTa, Sec. 3.1, Appendix Table 9](https://arxiv.org/abs/1907.11692)

**Pre-LN vs post-LN and warmup**
- Post-LN needs LR warmup because gradients near the output layer are large at initialization; Pre-LN does not and BERT pretraining with Pre-LN can drop warmup and trains faster — [Xiong et al. 2020, Sec. 3.2 and Sec. 4 (BERT pre-training, Fig. 5a)](https://arxiv.org/abs/2002.04745)
- Post-LN instability grows with depth in NMT: Post-LN diverged in 7 of 15 hyperparameter settings; 12- and 18-layer Post-LN fail to converge on WMT'14 En-De while Pre-LN converges; extending warmup to 16k/24k/32k does not rescue the 18-layer Post-LN model (no 24-layer test, no BERT experiments) — [Liu et al. 2020 (Admin), Sec. 3, 5.2, Table 2](https://arxiv.org/abs/2004.08249)

**Sequence length**
- Efficient recipes use 128 (Izsak entire run; Cramming; Wettig; LTG/ELC/GPT-BERT 90%/60% of steps) and only extend to 512 late (LTG-BERT last 10%; Wettig SQuAD 10%); Boot-BERT uses a constant 256 — see Q1 citations.
- Causal-LM sequence-length sweep (OPT-125M, Mamba, 100M-word Strict, 10 epochs): no single best length; OPT BLiMP was 70.2-73.9 for 64-2048 tokens with warmup but collapsed to 56.3 at 4096 and 55.1 at 8192; without warmup BLiMP fell monotonically from 75.4 (64) to 56.0 (8192); batch size/accumulation were not controlled — [What is the Best Sequence Length for BabyLM?, Table 6, Limitations](https://arxiv.org/abs/2510.19493). (Causal models, not MLM.)
- BabyLM 2023 Strict-Small: reducing context length to 32 from the baseline 128 "results in significant and consistent performance improvements" — [Findings 2023, App. F (Edman & Bylinina)](https://arxiv.org/abs/2504.08165)

**Smaller batch / more updates in BabyLM**
- 2025 workshop paper "Exploring Smaller Batch Sizes" reported that reducing effective batch size improved BLiMP/MSGS generalization, suggesting optimization noise helps small-data learning; another entry found that resetting optimizer state allowed validation loss to keep improving past conventional convergence on Baby10M and Baby100M — [Findings 2025, App. D.2](https://aclanthology.org/2025.babylm-main.28.pdf)

### Inferences
- Equivalent-LR anchors for AdamW at our batch (heuristic sqrt scaling from cited points; this scaling rule is my assumption, not from the papers): vanilla post-LN BERT-base 1e-4 at 131K-token batches (Chen et al.) / 16K-token batches (McGill); RoBERTa-large 4e-4 at ~4M tokens -> ~1e-4 at 256K; LTG-BERT AdamW 1e-3 at 4.2M tokens -> ~2.5e-4 at 256K; pre-LN BERT-large 2e-3 at 0.5M tokens -> ~1.4e-3 at 256K. Our 3.75e-5 peak is ~3-7x below the post-LN anchors and ~40x below the pre-LN large anchor.
- The cited papers consistently find the original BERT hyperparameters (LR ~1e-4 post-LN, 10% warmup) leave a short-budget run near the unigram plateau (Cramming Table 10: 7.28), and the fix is architecture (pre-LN) + 10-20x higher LR + schedule matched to the budget, not more epochs.
- For the second use case (BERT-large rater on 5-10B words, ~7-14B tokens): the Izsak/Wettig BERT-large recipe consumes 23k x 4096 x 128 = 12.1B tokens (derived), i.e. it is already scoped to that token budget; the ±250-token context for a bidirectional rater would require the 512-token final phase (Wettig: last 10% of steps, LR 5e-4, warmup 10%) or a longer-context phase.

### Gaps
- Cramming's single-epoch/24 h findings (no dropout, 15% masking best) do not transfer automatically to a multi-epoch 100M-word regime where overfitting is possible; no paper I found directly sweeps dropout for multi-epoch BabyLM-scale MLMs.
- Wettig et al. only tested 15-50% masking at ~12B-token budgets; no BabyLM-scale test of masking-rate vs model size beyond the GPT-BERT 30->15% schedule.
- Xiong et al.'s BERT result text (as extracted) lists Post-LN 1.69 val loss at 500k updates and Pre-LN at 700k while claiming a 40% Pre-LN speed-up — the extraction looks internally inconsistent, so I am not quoting the step counts.

---

## 3. Is a 24-layer / 1024-hidden post-LN model appropriate for ~100M words? Any evidence that large post-LN BERTs fail?

### Takeaway
No BabyLM-scale paper I found compares BERT-large with BERT-base; every strong BabyLM MLM is 12 layers (<=127M params, or 24-31M for 10M words), and the BabyLM organizers call the model-size question open. Evidence from outside BabyLM says a 24-layer BERT-large trains well and is more sample-efficient per step when it is pre-LN with a high, budget-synced LR, but that post-LN gets less stable with depth, and data-constrained scaling work favors smaller models trained for more epochs.

### Cited Findings
- All four winning/baseline MLM families use 12 layers: LTG-BERT 12L/768H (Table 9), ELC-BERT base 98M and small 24M (Table 4), Boot-BERT 127.7M / 30.4M (Table 5), GPT-BERT 119M / 30M (Table 3), 2025 baseline ~120M / ~31M — see Q1 citations.
- Organizers: "results are mixed when modifying model size" (ToddlerBERTa scales up to 92M and finds larger better; Mini Minds' scaled-down GPT-2/RoBERTa variants perform about as well as larger models on GLUE, and about as well as models with ~10x the parameters on an ethical-reasoning benchmark); "More controlled studies ... are needed to determine whether scaling up or down is a better solution" — [Findings 2023, Sec. 6 and App. F](https://arxiv.org/abs/2504.08165)
- 2025 submissions: smaller models (<200M params) benefited from simple-to-complex curricula, while relatively larger BabyLM-scale models (300M-1B) did better with simplified and original text interleaved — [Findings 2025, App. D](https://aclanthology.org/2025.babylm-main.28.pdf)
- Award: "Large GPT-like Models are Bad Babies" (Steuer et al. 2023) — about the relation between linguistic competence and psycholinguistic measures for GPT-like models; not a training-failure result — [Findings 2023, Sec. 7.3](https://arxiv.org/abs/2504.08165)
- Larger models are more sample-efficient per step/token when trained correctly: Cramming Fig. 1 includes 16- and 24-layer and wide(H=1024) variants and all reach ~1.9 MLM loss at the 24 h mark (with throughput offsetting per-token efficiency); Izsak "train large" chooses BERT-large for fixed wall-clock; Wettig: "it is more efficient to train larger models for fewer steps" (citing Li et al. 2020) — [Cramming, Sec. 4.2](https://arxiv.org/abs/2212.14034), [Izsak et al., Sec. 3.1](https://arxiv.org/abs/2104.07705), [Wettig et al., Sec. 9 (Conclusion & Discussion)](https://arxiv.org/abs/2202.08005)
- The efficient-recipe large-model successes are pre-LN with LR 1e-3-2e-3 and tokens in the billions (e.g., 12B, derived from 23K x 4096 x 128); the BERT-architecture large model trained with the classic layout (RoBERTa-large, which keeps the BERT architecture) used LR 4e-4 with batch 8K x 512 and 30k warmup steps — [Wettig, App. A.1](https://arxiv.org/abs/2202.08005), [RoBERTa, Sec. 2.2, Sec. 5 and Appendix Table 9](https://arxiv.org/abs/1907.11692)
- Post-LN depth sensitivity: Liu et al. show 12L and 18L Post-LN encoder-decoder models fail on WMT'14 En-De while Pre-LN trains; Cramming shows the original (post-LN) architecture with an aggressive LR schedule collapses (MNLI 35.4) — [Liu et al. 2020, Table 2](https://arxiv.org/abs/2004.08249), [Cramming, Table 5](https://arxiv.org/abs/2212.14034)
- LTG-BERT: standard post-norm "tends to unpredictably diverge in the later stages"; post-norm needed a halved LR (0.005 vs 0.01) and still scored lower; the organizers also saw loss spikes reproducing LTG-BERT in 2024 — [LTG-BERT, Sec. 4, Table 2](https://arxiv.org/abs/2303.09859), [Findings 2024, Sec. 4.3](https://arxiv.org/abs/2412.05149)
- Data-constrained scaling (autoregressive GPT-style, not MLM): up to ~4 epochs of repeated data cost ~nothing, returns decay with half-life ~16 epochs (R_D* ~ 15.4), and "allocation is optimized by scaling epochs faster than additional parameters"; a model with ~27% fewer parameters than the Chinchilla-suggested size reached lower loss when data-limited — [Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)
- Related: a 2026 analysis (secondary, via search summary only; not opened) reports performance related to model size but with the effect shrinking as corpus size grows, and a smaller-than-expected gap between 10M- and 100M-word models — [arXiv 2609.26539 (unverified)](https://arxiv.org/pdf/2609.26539)

### Inferences
- Nothing in the literature says a 24-layer BERT cannot be trained on ~100M words; what the literature shows is that all successful 24-layer MLMs use pre-LN (or Admin/NormFormer-style) normalization and LRs 10-50x higher than ours. So "BERT-large is wrong for BabyLM" is not supported, but "post-LN + 3.75e-5 + 24 layers + 15K steps" is far outside any documented working region.
- Token/parameter ratio: ours ~3.8B tokens / 335M params ~ 11 (derived); LTG-BERT ~131B / 110M ~ 1,200 (derived); the 2025 10-epoch baseline ~1.6B / 120M ~ 13 (derived). A BERT-large under the 2025 10-epoch cap would see ~5 tokens/param (derived), which supports the data-constrained-scaling view that base size is better matched to a capped 100M-word budget; with an uncapped epoch count the ratio is a non-issue.
- The weakest part of our evidence base: there is no head-to-head base-vs-large BabyLM experiment; the cheapest way to resolve it is to run a 12L/768H control with the same pipeline (a McGill- or LTG-style recipe) and see whether it leaves the plateau.

### Gaps
- No BabyLM-scale paper found that reports BERT-large (24L/1024H) vs BERT-base trained from scratch on 100M words; the "A retrospective ... infant syntax" paper and "Bigger is not always better" (Wilcox et al. 2025, J. Memory & Language 144:104650) were not retrieved.
- "A 2026 paper trains an 18-layer strict model and a 16-layer wide strict model" (search-engine summary of arXiv 2609.20530) was not opened; treat as unverified.
- No direct evidence on post-LN BERT-large with AdamW at LR 3.75e-5 specifically; closest are the plateau studies in Q5.

---

## 4. Document boundaries, sentence boundaries, special tokens in chunks

### Takeaway
The recipes that work either keep segments sentence/document-aligned (LTG-BERT code: [CLS] ... [SEP] with documents as sentence lists; McGill: ungrouped sentences) or pack to 128 tokens with a separator between unrelated fragments (Cramming). Evidence that contiguous long packing without separators hurts exists for conversational small-data corpora (McGill: BLiMP 62.6 packed vs 79.1 unpacked), though it measures zero-shot sentence-level BLiMP, not the optimization plateau.

### Cited Findings
- LTG-BERT training code (BNC): each document is a list of sentences; segments are runs of consecutive sentences extended until length reaches `seq_length - 3` (default `seq_length` 128); with probability `short_p = 0.1` the target length is a random `randint(2, seq_length-3)`; `[CLS]` and `[SEP]` are added at segment level (`[CLS] tokens [SEP]` for the basic MLM dataset; `[CLS] A [SEP] B [SEP]` for NSP variants); padding to `seq_length` with a padding attention mask; masking: `mask_p=0.15`, `random_p=0.1`, `keep_p=0.1`, span lengths `geometric(1/3) % 10` — [ltg-bert/training/mlm_dataset.py](https://github.com/ltgoslo/ltg-bert/blob/main/training/mlm_dataset.py) (this is the BNC training code; the repo's separate `babylm/` variant was not inspected)
- Corpus format: BNC sentences stored one per line, paragraphs separated by a blank line, articles start with a header — [LTG-BERT, Sec. 3.1](https://arxiv.org/abs/2303.09859)
- Cramming: data packed into randomized sequences of 128 tokens, unrelated fragments separated by `<sep>`; "The performance impact from dropping this separator was minimal. No impact was observed from including a `<cls>` token in pretraining." — [Cramming, Sec. 4.1](https://arxiv.org/abs/2212.14034)
- RoBERTa: FULL-SENTENCES (pack across documents with an extra separator token between documents) vs DOC-SENTENCES (no crossing): DOC-SENTENCES "performs slightly better", but FULL-SENTENCES is used for easier batching; single-sentence inputs (SENTENCE-PAIR) hurt downstream performance (long-range dependencies) — [RoBERTa, Sec. 4.2, Table 2](https://arxiv.org/abs/1907.11692)
- McGill (10M-word BabyLM, BERT-base, 30 epochs): grouping (concatenating sentences/documents into fixed-length examples) hurt a lot — BLiMP avg at L=128: sentence-level grouped 62.57 vs ungrouped 79.08; document-level grouped 62.04, ungrouped 69.38, ungrouped without truncation 75.39; at L=32 grouped 77.18 vs ungrouped 78.38; at L=512 ungrouped 78.00. Authors' explanation: much of the data is conversational so utterances "need not always be a logical continuation," and "we do not impose any attention masking to stop the model from attending to unrelated tokens"; the summary line: "(62% vs. 79% on BLiMP)" — [McGill BabyLM, Sec. 6, Table 4](https://aclanthology.org/2023.conll-babylm.18)
- BabyLM 2023 organizers: "Other successful approaches used short sequences or individual sentences as training samples, rather than long portions of documents"; McGill's summary says "not using sequence packing, using sentences and not documents as examples, not truncating, and reducing maximum sequence length are each highly effective" — [Findings 2023, Sec. 6 and App. F](https://arxiv.org/abs/2504.08165)
- A Loose-track augmentation that re-chunks the corpus repeatedly (about 40 resamplings per chunk, giving as many distinct samples as a 4B-word dataset) "outperforms training for 40 epochs using the same training samples" (the PDF text extraction interleaves columns; attribution is to the Contextualizer submission of Xiao et al., 2023) — [Findings 2023, Sec. 7.2](https://arxiv.org/abs/2504.08165)
- Position-embedding plateau result (relevant to 1000 learned absolute positions): a BERT-base study splits pretraining into start (0-1K steps), plateau (1K-8K), diving (8K-10K), convergence; the plateau coincides with the model acting as a bag-of-words that has not learned to use position information, and with fixed sinusoidal embeddings the plateau "indeed disappears" in the first 40K steps (but converges to a higher loss than learned positions) — [DecBERT, Sec. 4.2, 4.4](https://arxiv.org/abs/2204.08688)

### Inferences
- Our 1000-token contiguous chunks with no special tokens combine three things that the cited work reports as costly or unusual: (i) 8x the typical 128-token pretraining length, (ii) cross-utterance packing without separators/attention masks on a conversation-heavy mix (CHILDES, OpenSubtitles, Switchboard, BNC spoken), and (iii) 1000 learned absolute positions to fit from scratch. (ii) mostly degrades sentence-level quality (zero-shot BLiMP-style metrics, which matters for a rater scored on sentence-internal ±250 contexts), while (i) and (iii) plausibly lengthen the early plateau (DecBERT) — I did not find a paper that isolates sequence length's effect on plateau length for MLMs.
- Cheap mitigations consistent with the literature: insert a separator between utterances/documents (Cramming/RoBERTa), or restrict chunks to document boundaries (RoBERTa DOC-SENTENCES / LTG-BERT), and train mostly at 128 then extend to 512 (LTG-BERT/GPT-BERT/Wettig).

### Gaps
- No MLM paper found that directly compares 1000-token packed vs 128/512-token chunks for plateau length or final loss; the closest are causal-LM sequence length sweeps (OPT/Mamba) and McGill's BLiMP-only comparisons.
- McGill's grouped-vs-ungrouped comparison is on 10M words and sentence-level zero-shot BLiMP; effect on 100M words or on masked-token loss is unknown.
- LTG-BERT's `babylm/` directory code (the actual BabyLM 100M-word training formatting) was not inspected.

---

## 5. Loss / perplexity trajectories to sanity-check ours

### Takeaway
Healthy base/large MLMs reach MLM loss ~1.7-2.1 nats (vocab 30-50K, 15% masking) within a day or so of budget; a BERT-base with the standard post-LN recipe (LR 1e-4, batch 256, 10K warmup) sits on a plateau for ~7K-30K steps (loss ~6-7, near unigram) before a sharp drop; the original BERT recipe under a small budget can end at ~7.3. Our 5.62 after ~15K steps is consistent with being inside this plateau, not with a converged or saturated small-data model. No BabyLM-scale (10M-100M word) MLM loss curves were found.

### Cited Findings
- Plateau-then-drop dynamics, BERT-base, 12L/768H, uncased WordPiece, AdamW, LR 1e-4, 10,000 warmup steps, batch 256 x 512 tokens, 15% masking, Wikipedia+BookCorpus: MLM loss shows a "precipitous drop ... consistently spanning 20K-30K timesteps of training across various random seeds"; final MLM loss 1.77 +/- 0.01 at the 100K-step checkpoint (Table 1); the paper notes (citing Liu et al. 2023) that such phase transitions may be "primarily an artifact of poor hyperparameter settings" — [Chen et al. 2023 "Sudden Drops in the Loss", Sec. 3.1, 4.1, App. C.1](https://arxiv.org/abs/2309.07311)
- BERT-base re-implementation on English Wikipedia (158M sentences), batch 256, LR 1e-4 polynomial decay, 10K warmup, grad clip 0.5, dropout 0.1: training stages "starting (0-1,000 steps), plateau (1,000-8,000), diving (8,000-10,000), and convergence (10,000 to the end)"; plateau length ~7,000 steps; models with and without position embeddings have almost the same loss during the plateau (bag-of-words) — [DecBERT, Sec. 4.1-4.2](https://arxiv.org/abs/2204.08688)
- Cramming (BERT-base, 24 h): ends ~1.9 for all pre-LN architectures; 1.84 with one-cycle + LR 1e-3; original BERT recipe 7.28; classic-WD Adam 7.44; SGD 7.46; RAdam 7.50 — [Cramming, Fig. 1, Appendix Table 10](https://arxiv.org/abs/2212.14034)
- Izsak (BERT-large, pre-LN, 24 h): best val MLM loss 1.717; BERT-large with default hyperparameters 2.318 after 8.3 days; BERT-base default 2.050 after 3 days; diverging runs identified as val loss > 6.0 after 3 h — [Izsak et al., Table 3, Sec. 4.2](https://arxiv.org/abs/2104.07705)
- Wettig (BERT-large, pre-LN efficient recipe): validation PPL 17.7 at 15% masking, 69.4 at 40%, 1141.4 at 80% (One Billion Word, "calculated in the same way as Devlin et al." — exact definition not restated) — [Wettig et al., Table 1](https://arxiv.org/abs/2202.08005)
- Post-LN BERT-base with LR/warmup tuning reaches ~1.69 val loss after several hundred thousand updates (Pre-LN faster; see gap in Q2) — [Xiong et al. 2020, Sec. 4, Fig. 5a](https://arxiv.org/abs/2002.04745)
- BabyLM-scale numbers that exist are final "Perplexity (Dynabench)" values, with the metric undefined: Strict-small RoBERTa 20 epochs 49.586, DistilBERT 20 epochs 86.283, DistilBERT 60 epochs 17.278; Strict RoBERTa 20 epochs 27.566, DistilBERT 60 epochs 10.332 — [Pre-training LLMs using human-like development data corpus, Table 2](https://arxiv.org/abs/2311.04666)
- BabyLM winners publish downstream scores but not MLM loss curves in the papers I read (LTG-BERT, ELC-BERT, Boot-BERT, GPT-BERT).

### Inferences
- Reference lines for a sanity check: uniform = ln(50,004) = 10.82 nats and our measured unigram = 7.42 (given); healthy converged BERT-base/large MLM loss on large corpora ~1.7-2.0 (Cramming/Izsak/Chen); BabyLM-scale (100M words, 15% masking, 16K vocab) converged loss is not published in the papers I could access but would be expected to be higher than Wikipedia+Books (smaller, noisier, conversational data), so a target of roughly 2.5-3.5 is a guess, not a cited number.
- Our train loss 5.62 is below the unigram baseline (7.42) but far above the 1.7-3 band; the 56% argmax-"the" behavior is consistent with a model that has only learned frequency plus weak local context, the "bag-of-words" plateau described by DecBERT. A plateau escape at 20K-30K steps (Chen et al.) at LR 1e-4/batch 256x512 tokens would explain why the 20K-step LR 1e-4 seq-512 run (loss ~6.1) had not yet escaped, and why our ~15K-step LR 3.75e-5 run has not either.
- This implies the run needs either far more optimizer steps (smaller batch), a higher LR with a normalization scheme that tolerates it, or both; "train longer at the same settings" is the least efficient lever per the Izsak/Cramming results.

### Gaps
- No published MLM loss curves for 10M-100M-word BabyLM-scale training were found; the loss band for our corpus must be established empirically (e.g., a 12L control run).
- None of the plateau papers varies LR, warmup, or depth to report plateau length as a function of those; the dependence on LR/depth/batch is my inference.
- The Dynabench perplexity numbers are of unspecified definition and for tiny/odd models; not comparable to nats of masked loss.

---

## 6. Side-by-side: our setup vs. the literature, and what the comparison suggests

### Takeaway
Our run differs from every documented working recipe in the combination of (1) post-LN at 24 layers with a low LR (3.75e-5), (2) 1000-token unseparated chunks with 1000 learned positions, (3) a 50K vocabulary, (4) ~15K steps at 256K-token batches. The literature contains no run that trains a post-LN BERT-large under that combination; the working large-model recipes are pre-LN with LR 1e-3-2e-3 and 128-token sequences, and the working BabyLM recipes are 12-layer base/small with 16K-or-smaller vocabularies.

### Cited Findings
Table (each row's source in the last column):

| Axis | Ours | LTG/ELC/GPT-BERT (BabyLM winners) | Vanilla BERT-base small-data (McGill) | Efficient large-model (Izsak/Wettig) | Cramming (base) | Source |
|---|---|---|---|---|---|---|
| Layers / hidden / params | 24 / 1024 / ~335M | 12 / 768 / 98-127M (small: 12 / 384 / 24-31M) | 12 / 768 | 24 / 1024 / 354M | 12 / 768 | [LTG-BERT T9](https://arxiv.org/abs/2303.09859), [GPT-BERT T3](https://arxiv.org/abs/2410.24159), [McGill](https://aclanthology.org/2023.conll-babylm.18), [Izsak 3.1](https://arxiv.org/abs/2104.07705), [Cramming](https://arxiv.org/abs/2212.14034) |
| Norm placement | post-LN | NormFormer-style (post+extra LNs) | post-LN (BERT) | pre-LN (essential for high LR) | pre-LN | same |
| Vocabulary | 50,004 WordPiece lowercase | 16,384 (small 4-8K) | 32,768 uncased WordPiece | RoBERTa BPE ~50K (Wettig) / BERT 30K | 32,768 lowercase WordPiece | same + [Wettig A.1](https://arxiv.org/abs/2202.08005) |
| Sequence length | 1000, contiguous, no specials | 128 (90%/60%) then 512; Boot-BERT 256 | 32 / 128 / 512, ungrouped best | 128 (Wettig +512 last 10%) | 128, `<sep>` between fragments | same |
| Optimizer | AdamW b2 0.99, eps 1e-6, wd 0.01 all params | LAMB b2 0.98, wd 0.1, clip 2.0 | AdamW | AdamW b2 0.98, eps 1e-6, wd 0.01 | Adam b2 0.98, eps 1e-12, wd 0.01, clip 0.5 | same |
| Peak LR | 3.75e-5 (also 1e-4 tried) | 0.01 (LAMB; AdamW needed 1e-3 in LTG-BERT T2) | 1e-4 | 2e-3 or 1e-3 | 1e-3 | same |
| Warmup | 6.6% (~980 steps, derived) | 500 steps (1.6%) | 2,000 steps | 6% (or 2%) | included in one-cycle | same |
| Batch (tokens/step) | 256K | 4.2M (GPT-BERT 1M->4M; 2025 baseline 131K) | ~16K (128 ex x <=128) | 524K (4096 x 128) | ~516K (4032 x 128) | same |
| Optimizer steps | ~14.8K (derived) | 15.6K-31K (Boot-BERT 62.5K; 2025 baseline 12.3K) | not stated | ~23K | ~24 h single GPU | same |
| Epochs over ~100M words | 30 | ~450-1000 (2025: 10) | 30 (on 10M) | n/a (multi-B tokens) | <1 | [Findings 2023](https://arxiv.org/abs/2504.08165), [Findings 2025](https://aclanthology.org/2025.babylm-main.28.pdf) |
| Dropout | ~0.06 | 0.1 / 0.1 | 0.1 | 0.1 | 0 | same |
| Masking | 15%, 80/10/10 | 15% span (LTG-BERT); 30%->15% (GPT-BERT) | 15% | 15% (24hBERT) / 40% (Wettig large), no 80-10-10 | 15% 80/10/10 | same |
| Reported outcome | loss 5.62, stuck | BLiMP ~85-86 (Strict) | BLiMP ~79 (10M words) | val loss 1.72 | val loss ~1.84-1.9 | same |

### Inferences
- Ranked likely contributors to the plateau, based only on what the cited papers show (not on experiments on our data):
  1. LR too low for the normalization/depth/batch (3.75e-5 vs 1e-4 post-LN anchors that themselves plateau 7K-30K steps at batch 131K-262K tokens), combined with a short horizon (~15K steps). Both Chen et al. and DecBERT put the escape at 7K-30K steps for BERT-base at 1e-4 with 10K warmup; a 24-layer model at 3.75e-5 is plausibly slower.
  2. Post-LN at 24 layers: documented instability and need for warmup (Xiong; Liu; LTG-BERT; Cramming Table 5), and every successful large run is pre-LN or uses Admin-style init.
  3. Long (1000-token) learned-position sequences and unseparated packing: plausibly lengthen the position-learning phase and add noise; no paper isolates this for MLM.
  4. Vocabulary 3x larger than the BabyLM norm (50K vs 16K) on ~120M tokens; the LTG-BERT authors argue large vocabularies create many infrequent tokens on small corpora; Cramming found 2^15 near-optimal on a multi-billion-token corpus.
  5. Only weak candidates per the cited work: dropout, weight decay (LTG-BERT shows wd matters at the 0.01 vs 0.1 level but as a downstream gain, not as a plateau cause), fp16 (no cited source; Izsak/Cramming used mixed precision without reported issues).
- A 12-layer/768H control with a recipe from the cited working regions (e.g., pre-LN or NormFormer-style, 128-token chunks with `[SEP]`, LR 5e-4 to 1e-3 with batch ~0.5M tokens, 2-6% warmup, ~16K vocab) is the most informative single experiment, because every BabyLM winner used that region. For the BERT-large rater on 5-10B words, the closest documented recipe is Wettig/Izsak: pre-LN, LR 2e-3, batch 4096 x 128, warmup 6%, ~23K steps (~12B tokens), 40% masking for large, then a short 512-token phase.

### Gaps
- All "working region" claims come from different corpora (Wikipedia+Books, C4/Pile subsets, BNC); none is for a CHILDES/OpenSubtitles/Switchboard mix, so absolute LR transfer is not guaranteed.
- No source tests post-LN BERT-large with AdamW at 3.75e-5; contributions in the ranked list above are inferences, not measured.
