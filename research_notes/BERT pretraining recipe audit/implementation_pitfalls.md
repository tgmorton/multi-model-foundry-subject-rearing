# Implementation-level pitfalls when pretraining HF BertForMaskedLM from scratch (loss plateau at ~6-7 nats), checked against our implementation

Scope and method. Sources: transformers v4.41.2 source (downloaded and grepped), HF forum and GitHub threads, PyTorch 2.5 docs, original BERT/RoBERTa/fairseq/Megatron recipes, and papers. Our code was read read-only from `/Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/` (worktree HEAD `44b6c13f "diag: bert_large MLM plateau diagnostic"`). Status labels used throughout: APPLIES / DOES NOT APPLY / UNKNOWN. Forum posts are labeled anecdotes. Where the user's brief and the repo disagree, this is flagged explicitly (see the gradient-checkpointing finding in section 2).

Facts I could not find in our materials (these decide several UNKNOWN verdicts): peak LR, warmup length, tokens per optimizer step, total scheduled steps (LR-decay horizon) for the failing 24-layer run. The sweep baseline YAML only shows `learning_rate: 0.0001`, `warmup_ratio: 0.1`, `adam_beta2: 0.999` as sweep-overridden placeholders ([bert_large_en.yaml](file:///Users/thomasmorton/.cache/subject-drop/wt-main/configs/sweeps/baselines/bert_large_en.yaml)), which differ from the betas (0.9, 0.99) in the brief.

---

## 1. Known from-scratch HF BERT/RoBERTa MLM plateaus at ~6-7 nats: what causes were actually confirmed?

### Takeaway
No public thread I could find contains a *confirmed* root cause for a from-scratch BERT MLM plateau at 6-8 nats; the closest matches either ended unresolved or reported only a workaround (shorter sequences via `--line_by_line`). The only mechanisms with any positive evidence are sequence length / data format, depth + learning-rate/warmup sensitivity of post-LN, and simply not having trained long enough for the "sudden drop".

### Cited Findings
- HF forum "The loss plateau of pretraining Bert using run_mlm.py" (anecdote, 2021-2023): 6-layer, hidden 512, `bert-base-uncased` tokenizer, LR 1e-4, 8000 warmup, effective batch 256, 25 epochs. Loss drops fast, plateaus, then drops again. Another user lowering LR to 2.6e-4 still stalled "around 6". The only consistently reported workaround was `--line_by_line` (short, per-line sequences); one reply found no bug in the collator output; one reply hypothesised the plateau is the loss reachable from word-frequency statistics alone. No confirmed cause. — [HF forum 8492](https://discuss.huggingface.co/t/the-loss-plateau-of-pratraining-bert-using-run-mlm-py/8492)
- HF forum "Training Loss Sudden Spike After 8 Hours of pre-training a BERT Model" (anecdote, 2023): BERT-base trained fine (loss below 2.8 after 5 epochs); BERT-large-like config (hidden 1024, 16 heads, intermediate 4096, vocab 50,000) with 24 layers stalled at ~8.0 within an hour for 10+ hours; with 20 layers it trained for a while, then jumped back to ~8.0. LR 5e-5, `warmup_steps=20000`, `adamw_torch_fused`, `tf32=True`, no fp16 mentioned, transformers 4.31.0 / torch 2.0.1. Changing warmup and LR "did not help" per the poster. No replies, no fix. — [HF forum 54858](https://discuss.huggingface.co/t/training-loss-sudden-spike-after-8-hours-of-pre-training-a-bert-model/54858)
- transformers issue #3983 "MLM Loss not decreasing when pretraining Bert from scratch": BERT-base on English Wikipedia, MLM loss "stays around 7.3", code adapted from pytorch-pretrained-bert and Megatron-LM; closed by the author after one day with no explanation. — [transformers #3983](https://github.com/huggingface/transformers/issues/3983)
- Megatron-LM issue #29 "BERT Loss not decreasing" (same author): BERT-base, 16 V100s, batch 256, 900k steps at seq 128 then 100k at 512; NSP loss fell 0.7 to 0.3 but MLM loss only 10.0 to ~6.8; closed as completed with no explanation; a 2022 commenter reported the same and got no answer. — [Megatron-LM #29](https://github.com/NVIDIA/Megatron-LM/issues/29)
- HF forum "BertForMaskedLM training from scratch not converging": tiny SMILES BERT (hidden 256, vocab 120, LR 5e-5); no replies. — [HF forum 47216](https://discuss.huggingface.co/t/bertformaskedlm-training-from-scratch-not-converging/47216)
- HF forum "MLM train loss is very different after version update": same `run_mlm.py` command gave ~1.9 loss on transformers 4.5.1 and ~4.5 on 4.9.2; only lead was a tokenizer-class mismatch warning (T5TokenizerFast vs BertTokenizer); no confirmed cause. — [HF forum 9563](https://discuss.huggingface.co/t/mlm-train-loss-is-very-different-after-version-update/9563)
- Deep post-LN sensitivity (papers, not BERT-pretraining-specific): Xiong et al. show post-LN has large expected gradients near the output layer at init, so large LR is destabilising and warmup is needed. — [Xiong et al. 2020](https://arxiv.org/abs/2002.04745). Liu et al. report an 18-layer Post-LN model "still fails after extending the warmup phrase" (to 32k updates) and that Post-LN diverged in 7 of 15 grid settings while Pre-LN converged in all; they argue the root cause is an amplification effect of residual-branch dependency, not vanishing gradients. — [Liu et al. 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2004.08249)
- "How to Train BERT with an Academic Budget" prunes runs that have not reached validation loss 6.0 after 3 hours, stating the rule "removes diverging runs, such as configurations with 0% warmup"; its recipe uses 128-token sequences for the entire run, pre-LN, LR searched over {5e-4, 1e-3, 2e-3} (best 2e-3), batch 4k, warmup 0-6%. — [Izsak et al. 2021 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2104.07705)
- "Cramming" (from-scratch BERT-style, one GPU-day): 128-token packed sequences, peak LR 1e-3, β2=0.98, grad clip 0.5, dropout disabled in pretraining, pre-LN (reported as stabilising and allowing larger LR / less warmup), all biases removed; the paper reports an "original arch." ablation scoring badly on GLUE. — [Geiping and Goldstein 2022 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2212.14034)
- BERT-base trained at the original recipe (LR 1e-4, 10k warmup, batch 256, seq 512, 1M-step plan) shows "a precipitous drop in the loss curve" that "consistently" spans 20K-30K timesteps across seeds; the paper's text gives no loss values for the plateau (see its Figure 1). — [Chen et al. 2023, "Sudden Drops in the Loss"](https://arxiv.org/html/2309.07311v6)
- Early-pretraining plateaus in Transformers are linked to slow learning of attention maps, accompanied by repetition bias and representation collapse (hidden states nearly parallel across tokens); the mechanism analysis is mainly on shallow Transformers / algorithmic tasks, with the two symptoms also observed in early Pythia and OLMo pretraining. — [arXiv 2506.13688](https://arxiv.org/abs/2506.13688)

### Inferences
- Our signature (fast drop to ~unigram, argmax = "the", long flat stretch, slow creep to 5.6) matches the "plateau" family in the threads above, not a hard-bug family (e.g., a mask-id bug would normally fail loudly or sit at ~ln V, see section 7). The threads give us no documented single fix, so the audit has to be elimination plus controlled ablations.
- Chen et al.'s 20K-30K-step drop at 256 x 512 tokens/step corresponds to 2.6-3.9B tokens (20,000-30,000 x 131,072). Our run's 3.8B tokens is at the top of that window. The comparison is loose (BERT-base vs our 24-layer, Wikipedia+Books vs BabyLM-scale, 1M-step LR schedule vs our short linear-decay horizon), but it means "flat after 3.8B tokens" is not by itself evidence of a bug.
- The brief's 4-layer/256 model sitting at 6.46 after 2000 steps is consistent with the 6-layer plateau reports, which suggests the plateau is not specific to 24-layer post-LN depth. (Weak inference; a 4-layer model at 2000 steps is early in training.)
- A long-flat plateau under a linear LR schedule that decays to zero over a ~3.8B-token horizon may never see enough "LR x time" to escape. This is untested speculation; it is cheap to test with a constant-LR or longer-horizon run.

### Gaps
- No thread found with a *confirmed* resolution for a 6-7-nat plateau; the two matching GitHub issues (#3983, Megatron #29) were closed without explanation.
- No source found tying fp16 loss scaling to a stuck MLM loss.
- Chen et al.'s plateau loss values are only in figures; the text I could read (first ~100k of ~142k characters) does not state them.

---

## 2. transformers v4.41.2 source: padding_idx, SDPA with an all-ones mask, SDPA bugs, gradient checkpointing

### Takeaway
`pad_token_id=0` makes row 0 of the word-embedding matrix (our `[CLS]`) zero-initialised and gradient-free on the input side, which is harmless here because id 0 never occurs in inputs and is never a label. With an all-ones attention mask the SDPA path passes `attn_mask=None`, so no mask arithmetic is involved. I found no known SDPA training bug for BERT in 4.41.x. Importantly, our repo's trainer almost certainly never enabled gradient checkpointing for BERT (silent no-op), contradicting the brief.

### Cited Findings
- `BertEmbeddings.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=config.pad_token_id)`; `BertConfig.pad_token_id` defaults to 0. — [modeling_bert.py L163](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L163), [configuration_bert.py L112](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/configuration_bert.py#L112)
- `_init_weights` draws Linear and Embedding weights from N(0, `initializer_range`=0.02), zeroes Linear biases, and zeroes the `padding_idx` row; LayerNorm gets weight 1, bias 0. No depth-scaled init. — [modeling_bert.py L834-846](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L834-L846)
- In `nn.Embedding`, the vector at `padding_idx` starts as zeros and entries at `padding_idx` "do not contribute to the gradient" (lookup side). — [PyTorch Embedding docs](https://docs.pytorch.org/docs/2.5/generated/torch.nn.Embedding.html)
- `BertLMPredictionHead` is transform (dense + activation + LayerNorm) then a bias-free `decoder` Linear plus a separate output bias; weights are tied to the input embeddings when `tie_word_embeddings` is true (default). — [modeling_bert.py L764-784](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L764-L784), [modeling_utils.py L1742](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/modeling_utils.py#L1742)
- MLM loss is `CrossEntropyLoss()` (default mean, `ignore_index=-100`) over all positions of `prediction_scores.view(-1, vocab_size)`. — [modeling_bert.py L1506-1507](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L1506-L1507)
- `from_config(..., attn_implementation="sdpa")` is consumed by `_from_config` (`config._attn_implementation = kwargs.pop("attn_implementation", None)`), and `BERT_SELF_ATTENTION_CLASSES["sdpa"] = BertSdpaSelfAttention`. — [modeling_utils.py L1399](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/modeling_utils.py#L1399), [modeling_bert.py L467-470](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L467-L470)
- `BertModel.forward` builds the SDPA mask with `_prepare_4d_attention_mask_for_sdpa`; that helper returns `None` when `torch.all(mask == 1)` and query length equals key length (non-tracing). `BertSdpaSelfAttention` then calls `scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=p if self.training else 0.0, is_causal=False)` because `is_causal = self.is_decoder and ...`. — [modeling_attn_mask_utils.py L451-456](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/modeling_attn_mask_utils.py#L451-L456), [modeling_bert.py L433-442](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L433-L442)
- `BertSdpaSelfAttention` falls back to the eager implementation (with a warning) only for non-absolute position embeddings, `output_attentions`, or `head_mask`; it contains a `.contiguous()` workaround only for torch < 2.2 with a custom mask (torch issue 112577). We run torch 2.5.1 with no mask. — [modeling_bert.py L373-390 and L423-429](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L373-L429)
- I found no GitHub issue reporting wrong training behaviour from `BertSdpaSelfAttention` in 4.41.x (search returned only the PR/commit "[BERT] Add support for sdpa (#28802)" and a later `is_causal` cross-attention fix). — [commit mirror](https://cdn05042023.gitlink.org.cn/dnrops/transformers/commit/dfa7b580e9863c38c2f0e0dedf0958c2eab9cb48)
- `gradient_checkpointing_enable()` defaults to `gradient_checkpointing_kwargs={"use_reentrant": True}` in 4.41.2; `BertEncoder` only checkpoints when `self.gradient_checkpointing and self.training`, passing `layer_module.__call__` and positional args. — [modeling_utils.py L2243-2246](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/modeling_utils.py#L2243-L2246), [modeling_bert.py L663 and L678-679](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L663)
- PyTorch checkpoint docs: reentrant variant needs at least one input and output with `requires_grad=True`, fails on detached tensors inside the segment, and `preserve_rng_state=True` by default so dropout is replayed identically. — [PyTorch checkpoint docs](https://docs.pytorch.org/docs/2.5/checkpoint.html)
- **Repo finding.** `Trainer._apply_memory_optimizations` only calls `self.model.gradient_checkpointing_enable()` if `hasattr(self.model, 'gradient_checkpointing_enable')`, else `enable_gradient_checkpointing()`, else prints a WARNING. `self.model` is the `BERTModel` wrapper (an `nn.Module` subclass); `architectures/bert.py` defines neither method (only `architectures/gpt.py:236` does) and no `__getattr__` delegation exists. The repo's own sweep baselines say GC "was a silent no-op (wrapper exposes no GC method)". — [trainer.py L349-359](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/trainer.py), [bert.py](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/architectures/bert.py), [gpt.py L236](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/architectures/gpt.py), [bert_large_en.yaml L66](file:///Users/thomasmorton/.cache/subject-drop/wt-main/configs/sweeps/baselines/bert_large_en.yaml)

### Inferences
- padding_idx: APPLIES (config quirk) but harmless. Row 0 = `[CLS]` is never an input (no CLS/SEP in chunks; the collator never samples it as a target) except via the 10% random-token replacement (p ~ 1/50004 per replaced token). Because the decoder is tied, row 0 can still receive gradient through the output projection (the lookup-side mask does not block decoder gradient); this is my reading of the mechanism, not something a source states. Our real `[PAD]`=4 is an ordinary trainable row, harmless because there is no padding in practice. Cheap to remove: set `hf_config.pad_token_id = 4`.
- SDPA all-ones mask: DOES NOT APPLY. No mask tensor, no padding-row NaN risk, no fp32-mask-vs-fp16-q dtype interplay. Side effect: `torch.all(mask == 1)` is a host sync each forward (speed only).
- SDPA bug: no evidence found; absence of evidence only (see Gaps). A direct eager-vs-SDPA loss comparison for ~200 steps would settle it.
- Gradient checkpointing: DOES NOT APPLY, and the brief's premise ("enabled via model.gradient_checkpointing_enable()") is probably wrong for BERT in this repo. Check the run log for "WARNING: gradient checkpointing requested but BERTModel does not expose it". If a future change does make it active, the reentrant default is not known to break full-parameter pretraining (embedding outputs require grad, RNG state is preserved) but see Gaps.
- Initial-loss sanity check supports correct init/tying: with N(0, 0.02) tied embeddings and a unit-variance LayerNorm output of width 1024, logits have std ~0.02 x sqrt(1024) = 0.64, so expected initial loss ~ ln(50004) + 0.64^2/2 = 10.82 + 0.20 ~ 11.0, matching the observed "11".

### Gaps
- I did not search the issue tracker exhaustively for SDPA-BERT training reports (one search); I could not rule out an unreported bug.
- I could not find an HF-specific report of reentrant checkpointing silently producing no gradients for BertEncoder in 4.41; only the generic PyTorch caveat above.
- Whether the *actual failing run's* log contains the GC warning is unknown to me.

---

## 3. Is weight decay on LayerNorm weights/biases (and all params) harmful enough to cause this?

### Takeaway
It is a real deviation from BERT and HF-Trainer practice, but quantitatively negligible at the learning rates/horizons involved, so it is very unlikely to produce a 6-7-nat plateau. Note also that the brief's premise "BERT and RoBERTa exclude them" is only half true: original BERT and HF Trainer exclude them; fairseq's RoBERTa recipe does not.

### Cited Findings
- Original BERT optimizer: `weight_decay_rate=0.01`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-6`, `exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]`, linear warmup then linear decay to 0, and `tf.clip_by_global_norm(grads, clip_norm=1.0)`. — [google-research/bert optimization.py](https://raw.githubusercontent.com/google-research/bert/master/optimization.py)
- HF `Trainer` (4.41.2) builds two groups: decay for non-LayerNorm, non-"bias" parameters and `weight_decay=0.0` for the rest. — [trainer.py L1018-1052](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/trainer.py#L1018-L1052)
- fairseq (RoBERTa) builds a single flat param list `filter(requires_grad, chain(model.parameters(), criterion.parameters()))` with no decay/no-decay split; its RoBERTa-base config uses weight_decay 0.01, Adam betas (0.9, 0.98), eps 1e-6, `clip_norm: 0`, LR 5e-4, warmup 10,000, 125,000 updates, tokens_per_sample 512, batch 16 x update_freq 16 per GPU. — [fairseq trainer.py](https://raw.githubusercontent.com/facebookresearch/fairseq/main/fairseq/trainer.py), [roberta base.yaml](https://raw.githubusercontent.com/facebookresearch/fairseq/main/examples/roberta/config/pretraining/base.yaml)
- Megatron-LM's default sets `wd_mult=0.0` for 1-D parameters and `*.bias`. — [Megatron optimizer/__init__.py](https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/optimizer/__init__.py)
- PyTorch AdamW applies decoupled decay `theta <- theta - lr * wd * theta`; defaults lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight_decay 0.01. Our code passes `model.parameters()` to `AdamW` with no groups, so wd=0.01 applies to every tensor. — [PyTorch AdamW docs](https://docs.pytorch.org/docs/2.5/generated/torch.optim.AdamW.html), [trainer.py L362-392](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/trainer.py)
- Cramming states weight decay 0.01 (decoupled) and that changing eps to 1e-6 or β2 to 0.999 made no noticeable difference. — [Cramming (ar5iv)](https://ar5iv.labs.arxiv.org/html/2212.14034)

### Inferences
- APPLIES (deviation) but magnitude negligible. With decoupled decay, the cumulative shrink of an undecayed-equilibrium parameter is at most exp(-wd x sum of lr_t): peak LR 1e-4 for 19k steps gives <= 1.9% (1e-2 x 1e-4 x 19,000); peak LR 5e-4 gives <= ~9%. LN gains/biases start at 1/0 and are pushed back by gradients. This arithmetic is mine, from the cited update rule.
- Exclusion would not explain a loss floor at ~6-7; I would still align with BERT/HF (no decay on LN and bias) because it is free, but not expect it to fix the plateau.
- The Adam hyperparameters (β2=0.99, eps=1e-6) sit between BERT (0.999) and RoBERTa (0.98) and match their eps; DOES NOT APPLY as a cause.

### Gaps
- HF forum thread 10885 (excluding weight decay and getting a worse model) appeared in a search summary but I did not read it; I have not used it.
- I found no study that isolates LayerNorm-weight-decay effects on MLM plateau length.

---

## 4. fp16 + post-LN BERT-large: LayerNorm eps, attention overflow, loss-scale behaviour, would bf16 help?

### Takeaway
Under `torch.autocast` the LayerNorm eps=1e-12 and softmax/cross-entropy run in fp32, so the classic "eps underflows in fp16" pitfall does not apply to us; no skipped steps argues against overflow. The fp16 loss scale climbing to 2^23 with almost no skipped steps is best read as a symptom (tiny gradient magnitudes), not a cause, but fp16 cannot be excluded without an fp32/bf16 control run. I found no source linking fp16 to this specific plateau.

### Cited Findings
- PyTorch 2.5 AMP docs (CUDA): `linear` and `matmul` autocast to float16; `layer_norm`, `softmax`, `cross_entropy` autocast to float32. GradScaler defaults: `init_scale=65536.0`, `growth_factor=2.0`, `backoff_factor=0.5`, `growth_interval=2000`. — [PyTorch AMP docs](https://docs.pytorch.org/docs/2.5/amp.html)
- PyTorch's recommended order for clipping under AMP is backward, `scaler.unscale_(optimizer)`, `clip_grad_norm_`, `scaler.step`, `scaler.update`, with loss divided by `iters_to_accumulate`. `scaler.step` skips `optimizer.step()` if grads contain inf/NaN. — [PyTorch AMP examples](https://docs.pytorch.org/docs/2.5/notes/amp_examples.html)
- Our loop does exactly that: `autocast(float16)`, `loss = outputs.loss / grad_accum`, `scaler.scale(loss).backward()`, at the boundary `scaler.unscale_` then `clip_grad_norm_(max_grad_norm)` then `scaler.step`/`update`; GradScaler built with `init_scale=2**16, growth_factor=2, backoff_factor=0.5, growth_interval=2000`; it counts and logs skipped steps. — [loop.py L74-85 and L545-583](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/training/loop.py)
- Residual stream dtype: `BertSelfOutput` computes `LayerNorm(dropout(dense(h)) + input_tensor)`; with autocast, `dense` returns fp16 and LayerNorm returns fp32. — [modeling_bert.py L453-464](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L453-L464)
- `layer_norm_eps` defaults to 1e-12 in `BertConfig`. — [configuration_bert.py L111](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/configuration_bert.py#L111)
- Megatron-LM's BERT-340M example (24 layers, hidden 1024, 16 heads) runs with `--fp16`, LR 1e-4 (min 1e-5, linear decay), warmup fraction 0.01 of 1M iterations, weight decay 1e-2, clip-grad 1.0, seq 512, global batch 32. — [Megatron train_bert_340m_distributed.sh](https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/examples/bert/train_bert_340m_distributed.sh)
- bf16 has fp32's 8-bit exponent (no loss scaling usually needed) but fewer mantissa bits (7 vs fp16's 10); sources here are educational/vendor material. — [Raschka FAQ](https://sebastianraschka.com/faq/docs/mixed-precision-bfloat16.html), [apxml bf16 vs fp16](https://apxml.com/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/bfloat16-vs-float16-configurations)
- Cramming used automated mixed precision and reports comparing scaled 16-bit and pure bfloat16, without a clear statement of which was final. — [Cramming (ar5iv)](https://ar5iv.labs.arxiv.org/html/2212.14034)

### Inferences
- LayerNorm eps 1e-12: DOES NOT APPLY under autocast (LN in fp32). It would matter only for a model cast to `.half()`.
- Attention-score overflow: no evidence; an overflow in fp16 intermediates would show as inf grads and backed-off scale/skipped steps, and the brief reports almost none.
- Loss scale 2^23: reaching it from 2^16 requires 7 doublings at one per 2000 optimizer steps, i.e. >= 14,000 consecutive overflow-free steps, which implies all scaled fp16 gradient intermediates stayed below 65,504 at scale 8.4e6 (unscaled max about 7.8e-3 per micro-batch quantity). That is consistent with very small gradients, but absolute thresholds for "healthy" are not available from any source I read (fairseq's loss is summed not averaged, so its scale is not comparable). Direct evidence of tiny gradients is better obtained by logging post-unscale per-layer grad norms.
- Early-fp16 underflow risk: the scale is only 2^16-2^17 during the first ~2-4k steps, exactly when the loss falls to the plateau. Underflow of deep-layer activation gradients in that window is a candidate I can neither confirm nor exclude. UNKNOWN.
- bf16: would remove loss scaling and the overflow/underflow question; expected to help only if fp16 numerics are part of the problem. Not expected to cure a plateau that is optimisation/data driven. Cheap discriminating experiment: 1-2k steps in bf16 (or fp32 with TF32) from the same init and compare the loss curve and grad norms.
- Clipping order: DOES NOT APPLY (confirmed correct, `unscale_` precedes `clip_grad_norm_`). Clip 1.0 matches Google BERT and Megatron; fairseq RoBERTa disables clipping.

### Gaps
- No primary source on fp16-specific failure modes of post-LN BERT-large pretraining (the HF thread that matched our signature did not use fp16).
- bf16 claims rest on vendor/educational pages; I did not locate a primary precision-format reference.
- I did not verify whether PyTorch SDPA's flash/mem-efficient CUDA kernels accumulate softmax in fp32 (the AMP page summary I read did not list `scaled_dot_product_attention` in the CUDA fp16 list).

---

## 5. Very long learned absolute position embeddings (1000) from scratch; chunks without [CLS]/[SEP]

### Takeaway
No source documents a pitfall specific to 1000 learned positions, but every successful small-budget from-scratch recipe I found trains on short (128-token) sequences, and the only plateau workaround in the forum threads is shorter sequences. Training on 1000-token contiguous chunks without special tokens is therefore the largest recipe deviation I can document, though its causal role is unproven.

### Cited Findings
- Original BERT: sequence length 128 for 90% of steps, 512 for the last 10% "to learn positional embeddings"; batch 256 sequences, 1M steps, LR 1e-4, β=(0.9, 0.999), wd 0.01, 10k linear warmup, dropout 0.1. — [BERT paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/1810.04805)
- Academic-budget BERT: "limit sequences to 128 tokens for the entire pretraining process". Cramming: packed sequences of length 128. — [Izsak et al.](https://ar5iv.labs.arxiv.org/html/2104.07705), [Cramming](https://ar5iv.labs.arxiv.org/html/2212.14034)
- HF run_mlm thread: `--line_by_line` shortened or removed the plateau; dataset choice changed plateau length. — [HF forum 8492](https://discuss.huggingface.co/t/the-loss-plateau-of-pratraining-bert-using-run-mlm-py/8492)
- Sequence-length warmup for GPT-style models: shorter sequences early make training stable again at larger LR (taken from a search-result summary of the paper, not read directly). — [Li et al., arXiv 2108.06084](https://arxiv.org/pdf/2108.06084v3)
- HF `run_mlm.py` default (non `--line_by_line`) mode tokenizes each text line *with* special tokens ([CLS]...[SEP]), concatenates, and chops into `max_seq_length` blocks, so special tokens appear throughout the stream; default `max_seq_length` is capped at 1024 when the tokenizer allows. — [run_mlm.py L457-472 and L511-514](https://github.com/huggingface/transformers/blob/v4.41.2/examples/pytorch/language-modeling/run_mlm.py)
- RoBERTa FULL-SENTENCES: inputs packed contiguously (up to 512 tokens) across document boundaries with an extra separator token between documents, NSP removed; DOC-SENTENCES (no cross-document packing) was slightly better (e.g., SQuAD 90.6/79.7 vs 90.4/79.1). — [RoBERTa (ar5iv)](https://ar5iv.labs.arxiv.org/html/1907.11692)
- Our data: contiguous chunks of 1000 (or 512) tokens, no [CLS]/[SEP], shuffled each epoch with a seeded generator; `max_position_embeddings` set to `data.max_sequence_length` (1000); sweep baseline YAML sets `max_sequence_length: 1000`. — [data.py L466-471](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/data.py), [bert.py L188-190](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/architectures/bert.py), [bert_large_en.yaml](file:///Users/thomasmorton/.cache/subject-drop/wt-main/configs/sweeps/baselines/bert_large_en.yaml)
- A search for evidence on learned absolute positions at ~1024 from scratch returned only general background (fixed-length limit; late positions get fewer updates in fine-tuning); nothing on pretraining dynamics. — [Position-encoding survey, arXiv 2608.10021](https://arxiv.org/pdf/2608.10021)

### Inferences
- Sequence length: APPLIES (documented deviation; highest-priority ablation). Proposed mechanism (my reasoning, no source): at init, attention is near-uniform over L positions, so a masked token's signal from its useful neighbours is diluted ~1/L; at L=1000 vs 128 the gradient that teaches "attend locally" is ~8x weaker, lengthening the unigram-like phase. This is consistent with, not proven by, the line_by_line anecdote and the 128-token recipes.
- Learned 1000 positions: UNKNOWN. No documented failure; N(0, 0.02) position vectors have the same scale as token vectors, so position and token identity are equally weighted pre-LN. Compare 128-token and 512-token runs of the *same* model to isolate it.
- No [CLS]/[SEP]: low risk. It resembles RoBERTa FULL-SENTENCES minus the document separator, which cost little in RoBERTa. The cited threads' use of per-line [CLS]/[SEP] is a difference from ours but there is no evidence it matters.
- Data shuffling: DOES NOT APPLY (`shuffle=True`, seeded per epoch).

### Gaps
- No primary source on how long learned absolute positions at 1000 affect early MLM plateau length.
- The RANLP 2021 paper cited in the HF thread about line-level inputs was not read.

---

## 6. Concrete diagnostics recommended in the threads/papers

### Takeaway
The threads themselves offer little beyond "tiny-overfit and masked-position-loss checks"; the actionable diagnostics come from Karpathy's recipe and the plateau literature, and from controlled ablations anchored to the known-good recipes above. Our worktree already has a commit named "diag: bert_large MLM plateau diagnostic"; I did not inspect it.

### Cited Findings
- Karpathy: overfit a single small batch; verify loss at init equals the expected chance value; run an input-independent baseline (zeroed inputs) to see whether the model uses its input; backprop a trivial function of one output to chart dependencies (gradients should only reach inputs that matter). — [Karpathy, "A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/)
- Plateau-literature measurements: repetition bias in outputs and representation collapse (cosine similarity of hidden states across tokens), and progress of attention maps; intervening on attention changes plateau length. — [arXiv 2506.13688](https://arxiv.org/abs/2506.13688)
- Chen et al. track a structural-onset signal (implicit parse UAS from attention heads) alongside loss and found the loss break coincides with it; suppressing it early shifted the break. — [Chen et al. 2023](https://arxiv.org/html/2309.07311v6)
- Izsak et al.'s early-abort rule: validation loss must reach <= 6.0 within 3 hours or the configuration is treated as diverging. — [Izsak et al.](https://ar5iv.labs.arxiv.org/html/2104.07705)
- PyTorch AdamW update: `theta -= lr * m_hat / (sqrt(v_hat) + eps)`, so when per-element gradient RMS is far below eps the effective step shrinks by roughly RMS/eps. — [PyTorch AdamW docs](https://docs.pytorch.org/docs/2.5/generated/torch.optim.AdamW.html)
- Reference hyperparameters for controls: BERT (LR 1e-4, warmup 10k, bs 256, 128 tokens first), RoBERTa-base (LR 5e-4, bs 2048 x 512, warmup 10k, β2 0.98, eps 1e-6, clip 0), RoBERTa paper (LR 4e-4 large / 6e-4 base, batch 8K, warmup 24-30k, β2=0.98 "improves stability" for large batches), Izsak (seq 128, LR 2e-3, bs 4k, pre-LN), Cramming (seq 128, LR 1e-3, pre-LN, clip 0.5), Megatron BERT-340M example (LR 1e-4, warmup 1%, clip 1.0, fp16). — sources as cited in sections 1-5, [RoBERTa (ar5iv)](https://ar5iv.labs.arxiv.org/html/1907.11692)

### Inferences
Ranked experiment list (all mine; cheapest first), each phrased to discriminate a hypothesis:
1. Context-use probe on the current checkpoint: evaluate masked-token loss with (a) intact context, (b) context shuffled within the sequence, (c) neighbours +-5 only, (d) context zeroed/[MASK]ed. If (a) ~ (b) the model is context-blind (unigram stage). Also compute input-gradient mass vs distance to the masked position (Karpathy dependency chart).
2. Collapse/attention telemetry: per-layer mean pairwise cosine similarity of hidden states across tokens, per-head attention entropy vs ln(L) (uniform = ln 1000 = 6.9), and fraction of attention within +-8 tokens. Uniform attention in all heads late in training would confirm the dilution hypothesis.
3. Gradient telemetry (post-`unscale_`): per-layer grad norm and per-tensor grad RMS vs eps=1e-6 (AdamW step shrink), log `scaler.get_scale()`; confirm the GC warning line in the run log.
4. Sequence-length ablation: same model/data/LR at L=128 (and 256, 512) for equal *tokens*; plus a BERT-style two-stage schedule (128 then longer). Predicts faster escape from the plateau if dilution is the driver.
5. Precision control: first 1-2k steps in bf16 (or fp32+TF32) from identical init and batches.
6. Schedule control: constant LR (or much longer horizon) after warmup, to test whether the linear decay is strangling escape.
7. LR/warmup sweep around the cited recipes for this depth (post-LN 24L: 1e-4 with 10k warmup is BERT-large's published setting; larger-LR recipes in the literature are pre-LN).
8. Known-good reference: HF `run_mlm.py` from scratch with this tokenizer, bert-base-size config, `--line_by_line` vs grouped, ~5k steps, as a harness check.
9. Pre-LN variant (as in Izsak/Cramming) as an architecture control if post-LN depth is suspected.
10. Housekeeping: set `pad_token_id=4`; exclude LN/bias from weight decay; both are free and remove two documented deviations.

### Gaps
- None of the threads reports which diagnostic ultimately confirmed a cause, because none reached a confirmed cause.
- I did not inspect the existing "diag: bert_large MLM plateau diagnostic" commit, so some of the above may already exist.

---

## 7. Consolidated checklist: documented pitfall vs our implementation

### Takeaway
Nothing in the audited code is a *hard* bug that would explain a plateau by itself. The pitfalls that APPLY are all deviations from the published recipes (sequence length 1000, all-parameter weight decay, `pad_token_id=0`, short linear-decay horizon, dropout) rather than mistakes; several decisive facts (LR, warmup, tokens/step, horizon) are UNKNOWN. The brief's gradient-checkpointing item is contradicted by the repo (probably never active).

### Cited Findings
Status key: A = applies, N = does not apply, U = unknown.

| # | Pitfall | Status | Evidence |
|---|---|---|---|
| 1 | Wrong/absent mask token id or special-token mismatch (fast tokenizer from `tokenizer.json`) | N | HF's own collator raises `ValueError` if the tokenizer has no mask token ([data_collator.py L705-708](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/data/data_collator.py#L705-L708)); ours assigns `tokenizer.mask_token_id` ([data_collators.py L264](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/data_collators.py)), so a missing id would crash, and the user verified batch/loss parity with the HF collator |
| 2 | Labels not `-100` off-mask / misaligned labels | N | Collator sets `labels[~masked_indices] = -100` ([data_collators.py L257](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/data_collators.py)); HF loss uses `ignore_index=-100` with no shift ([modeling_bert.py L1506-1507](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L1506-L1507)) |
| 3 | `pad_token_id=0` -> `padding_idx=0` row zeroed, mismatched with real `[PAD]`=4 | A (harmless) | [modeling_bert.py L163, L841-842](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L163); id 0 = `[CLS]`, never input/target; section 2 |
| 4 | SDPA + all-ones attention mask problems | N | mask collapses to `None` ([modeling_attn_mask_utils.py L451-456](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/modeling_attn_mask_utils.py#L451-L456)) |
| 5 | Known SDPA training bug for BERT in 4.41.x | N (no evidence found) | section 2 gaps |
| 6 | Gradient checkpointing (reentrant default, dropout, no-grad) | N (likely never enabled for BERT) | wrapper lacks GC method; [trainer.py L349-359](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/trainer.py); repo comment "silent no-op" |
| 7 | Weight decay applied to LayerNorm/bias/embeddings | A (negligible) | [optimization.py](https://raw.githubusercontent.com/google-research/bert/master/optimization.py) excludes LN/bias; fairseq does not ([fairseq trainer.py](https://raw.githubusercontent.com/facebookresearch/fairseq/main/fairseq/trainer.py)) |
| 8 | Clipping before `unscale_` | N | [loop.py L555-563](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/training/loop.py) vs [PyTorch AMP examples](https://docs.pytorch.org/docs/2.5/notes/amp_examples.html) |
| 9 | LayerNorm eps 1e-12 under fp16 | N (autocast keeps LN fp32) | [PyTorch AMP docs](https://docs.pytorch.org/docs/2.5/amp.html) |
| 10 | fp16 attention/activation overflow | N (no skipped steps reported) | brief; [loop.py L566-583](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/training/loop.py) logs skips |
| 11 | Early fp16 gradient underflow (scale 2^16 -> doubles every 2000 steps) | U | section 4 |
| 12 | Adam eps (1e-6) dominating tiny gradients | U | [AdamW formula](https://docs.pytorch.org/docs/2.5/generated/torch.optim.AdamW.html); Cramming found eps-insensitivity only at normal gradient scale |
| 13 | LR / warmup / batch (post-LN 24L needs warmup; LR too low or high) | U | Run's values not in the materials; references in section 6 |
| 14 | Sequence length 1000 (vs 128 recipes, line_by_line workaround) | A | section 5 |
| 15 | Short total horizon + LR decaying to zero before the "sudden drop" | U | [Chen et al.](https://arxiv.org/html/2309.07311v6) (drop at 20K-30K steps, 1M-step schedule) |
| 16 | No [CLS]/[SEP] in contiguous chunks | N (low risk) | [RoBERTa FULL-SENTENCES](https://ar5iv.labs.arxiv.org/html/1907.11692) |
| 17 | Learned absolute positions at 1000 trained from scratch | U | no source; section 5 |
| 18 | Data not shuffled | N | [data.py L466-471](file:///Users/thomasmorton/.cache/subject-drop/wt-main/model_foundry/data.py) |
| 19 | Tied embeddings / init std 0.02 / no depth-scaled init | N (same as HF/BERT) | [modeling_bert.py L834-846](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L834-L846); initial loss ~11 matches theory |
| 20 | Dropout 0.1 slowing early learning | A (speculative, minor) | Cramming disables dropout in pretraining "to maximize the number of parameter updates" ([ar5iv](https://ar5iv.labs.arxiv.org/html/2212.14034)); we run 30 epochs so the single-epoch argument is weaker |
| 21 | Post-LN depth instability | U | [Xiong](https://arxiv.org/abs/2002.04745), [Liu](https://ar5iv.labs.arxiv.org/html/2004.08249), [HF 54858](https://discuss.huggingface.co/t/training-loss-sudden-spike-after-8-hours-of-pre-training-a-bert-model/54858); the 4-layer 6.46 plateau argues depth is not the sole factor |

### Inferences
- Highest-priority items given our exact signature: #14 (sequence length), #15/#13 (LR and horizon), #11/#12 (numerics under tiny gradients), then #21. Items #3 and #7 are free clean-ups, not expected cures.
- Items needing the run's actual config before they can be resolved: peak LR, warmup steps, tokens/step, `scheduler_train_steps`, and whether the GC warning appears in the pod log.

### Gaps
- LR, warmup, batch tokens, and schedule horizon of the failing run were not available to me.
- No confirmed-cause precedent exists in the public threads, so the table can only rank hypotheses, not identify the cause.
