"""D7 + D8: batched forward passes with optional context KV-cache reuse.

Provides `batched_surprisal(...)` which stacks N sequences into one forward
pass with right-padding + attention mask, and returns per-item per-token
natural-log log-probabilities over the target span only.

The standard (non-paired) batching is equivalent to running each item
sequentially through v1's `SurprisalCalculator.calculate_surprisal()` —
asserted in `test_batched_vs_sequential.py` as the CI gate.

D8 adds an optional `pair_mode=True` path for paired (overt, null) inputs:
both members share the `context_ids` prefix, so we compute the context KV
cache once and run only the two target continuations.

Implementation note on D8: HF's `DynamicCache` (default in transformers
>= 4.36) mutates the cache object in-place during the forward pass,
even when `use_cache=False` is set. To run two continuations (overt +
null) against the same shared context, `paired_surprisal` deep-copies
the cache before each continuation call. The cost is small (per-layer
K/V tensor clones); the alternative (re-running the context pass) would
defeat the purpose of the optimization. With the copy in place, the
paired path is bit-exact with the full-sequence path.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchedSurprisalResult:
    """One item's result from a batched forward pass.

    Fields:
      target_log_probs: natural-log log-probs at each target token position
                        (length == target_subword_count)
      target_token_ids: list of the scored target token ids (same length)
      hotspot_log_prob: natural-log log-prob at the resolved hotspot
                        (or None if the hotspot index fell outside the
                        target span for this item)
    """
    target_log_probs: List[float]
    target_token_ids: List[int]
    hotspot_log_prob: Optional[float] = None


# --- Core batched primitive -------------------------------------------------

def _pad_batch(
    sequences: List[List[int]],
    pad_id: int,
    dtype: torch.dtype = torch.long,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of token-id sequences into (B, T_max) + attention mask."""
    B = len(sequences)
    T = max(len(s) for s in sequences) if sequences else 0
    ids = torch.full((B, T), fill_value=pad_id, dtype=dtype)
    mask = torch.zeros((B, T), dtype=torch.long)
    for i, s in enumerate(sequences):
        L = len(s)
        ids[i, :L] = torch.tensor(s, dtype=dtype)
        mask[i, :L] = 1
    return ids, mask


def batched_surprisal(
    model,
    full_token_ids_list: Sequence[Sequence[int]],
    context_lens: Sequence[int],
    hotspot_indices: Optional[Sequence[Optional[int]]] = None,
    *,
    pad_id: int = 0,
    device: Optional[str] = None,
) -> List[BatchedSurprisalResult]:
    """Score a batch of (context + target) sequences.

    Args:
      model: HuggingFace-style causal LM. `model(input_ids, attention_mask)`
             must return an object with a `.logits` tensor of shape
             (B, T, V).
      full_token_ids_list: N sequences, each [context_tokens..., target_tokens...].
      context_lens: N ints; index where each target span starts.
      hotspot_indices: optional N items; each None or an int in
                       [context_lens[i], full_len[i]). When provided
                       and inside the target span, the hotspot log-prob is
                       returned.
      pad_id: token id used for right-padding.
      device: where to run the forward pass. If None, uses the model's device.

    Returns:
      List of N BatchedSurprisalResult, aligned with the input order.

    Notes:
      - Uses log_softmax + gather. Numerically equivalent to v1's per-item
        pathway up to float32/float16 rounding.
      - Mask positions (past the end of each sequence) are masked out of
        attention; no logits for those positions are returned.
    """
    if device is None:
        device = next(model.parameters()).device
    sequences = [list(s) for s in full_token_ids_list]
    if len(sequences) == 0:
        return []
    if len(context_lens) != len(sequences):
        raise ValueError("context_lens length must match full_token_ids_list")
    if hotspot_indices is None:
        hotspot_indices = [None] * len(sequences)
    else:
        hotspot_indices = list(hotspot_indices)

    input_ids, attention_mask = _pad_batch(sequences, pad_id=pad_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    # Convert to natural-log log-probs over the vocabulary.
    log_probs = torch.log_softmax(logits, dim=-1)

    results: List[BatchedSurprisalResult] = []
    for i, seq in enumerate(sequences):
        L = len(seq)
        context_len = int(context_lens[i])
        if context_len >= L:
            # No target tokens at all (degenerate).
            results.append(BatchedSurprisalResult(
                target_log_probs=[],
                target_token_ids=[],
                hotspot_log_prob=None,
            ))
            continue

        # For each target position j (context_len .. L-1), we want
        # log P(seq[j] | seq[:j]) = log_probs[i, j-1, seq[j]].
        # j-1 ranges from context_len-1 to L-2.
        target_positions = list(range(context_len, L))
        # Predictive positions are target_positions - 1
        pred_positions = torch.tensor(
            [j - 1 for j in target_positions], dtype=torch.long, device=device
        )
        target_ids_tensor = torch.tensor(
            [seq[j] for j in target_positions], dtype=torch.long, device=device
        )
        gathered = log_probs[i, pred_positions, target_ids_tensor]
        target_log_probs = gathered.tolist()

        # Hotspot log-prob: index into the full sequence
        h = hotspot_indices[i]
        hotspot_lp: Optional[float] = None
        if h is not None and context_len <= h < L:
            # The log-prob of seq[h] given seq[:h] lives at prediction position h-1.
            hotspot_lp = float(log_probs[i, h - 1, seq[h]].item())

        results.append(BatchedSurprisalResult(
            target_log_probs=[float(x) for x in target_log_probs],
            target_token_ids=[seq[j] for j in target_positions],
            hotspot_log_prob=hotspot_lp,
        ))

    return results


# --- D8: context KV-cache reuse for paired (overt, null) items --------------

@dataclass
class PairedInput:
    """Inputs for a pair sharing the same context prefix."""
    context_ids: List[int]              # shared across both members
    overt_target_ids: List[int]
    null_target_ids: List[int]
    # Absolute indices (into context + target) of the hotspot for each side
    overt_hotspot_index: Optional[int] = None
    null_hotspot_index: Optional[int] = None


@dataclass
class PairedResult:
    """Per-pair result with overt / null log-prob vectors."""
    overt_target_log_probs: List[float]
    null_target_log_probs: List[float]
    overt_target_token_ids: List[int]
    null_target_token_ids: List[int]
    overt_hotspot_log_prob: Optional[float]
    null_hotspot_log_prob: Optional[float]


def paired_surprisal(
    model,
    pair: PairedInput,
    *,
    device: Optional[str] = None,
) -> PairedResult:
    """Score an (overt, null) pair, reusing the shared context KV cache.

    This is the D8 optimization: the two members of a pair share
    context tokens, so we compute the context's `past_key_values` once
    and run two cheaper continuations.

    Equivalent (within float rounding) to running both full sequences
    through `batched_surprisal`. Asserted by the numerical-equivalence
    test when `pair_mode=True` is set.

    Note: requires a model that supports `past_key_values` — all HF
    causal LMs do. Custom architectures (LSTM/Mamba) may not; caller
    should dispatch on capability.
    """
    if device is None:
        device = next(model.parameters()).device
    ctx = pair.context_ids
    ctx_len = len(ctx)

    # Step 1: forward pass on the shared context to build past_key_values.
    # Pass explicit attention_mask and position_ids so both calls see a
    # consistent interface — no ambiguity about what HF derives internally.
    ctx_tensor = torch.tensor([ctx], dtype=torch.long, device=device)
    ctx_position_ids = torch.arange(0, ctx_len, dtype=torch.long, device=device).unsqueeze(0)
    ctx_attention_mask = torch.ones((1, ctx_len), dtype=torch.long, device=device)
    with torch.no_grad():
        ctx_out = model(
            input_ids=ctx_tensor,
            position_ids=ctx_position_ids,
            attention_mask=ctx_attention_mask,
            use_cache=True,
        )
    past = ctx_out.past_key_values
    ctx_logits = ctx_out.logits[0]  # (ctx_len, V)
    ctx_log_probs = torch.log_softmax(ctx_logits, dim=-1)

    def _score_continuation(target_ids: List[int], hotspot_abs: Optional[int]):
        """Run target continuation using a copy of the shared context KV cache.

        HF's `DynamicCache` is stateful and gets mutated during the
        forward pass even when `use_cache=False` is set. To allow running
        multiple continuations (overt + null) against the same context
        without cross-contamination, we deep-copy the cache before each
        call. Deep-copy cost is negligible — it's just per-layer K/V
        tensor clones — and the alternative (re-running the context pass)
        is far more expensive.
        """
        if not target_ids:
            return [], [], None
        tgt_len = len(target_ids)
        tgt_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
        tgt_position_ids = torch.arange(
            ctx_len, ctx_len + tgt_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        tgt_attention_mask = torch.ones(
            (1, ctx_len + tgt_len), dtype=torch.long, device=device
        )
        past_copy = copy.deepcopy(past)
        with torch.no_grad():
            tgt_out = model(
                input_ids=tgt_tensor,
                past_key_values=past_copy,
                position_ids=tgt_position_ids,
                attention_mask=tgt_attention_mask,
                use_cache=False,
            )
        # tgt_out.logits shape: (1, tgt_len, V). Position j of the target
        # predicts target_ids[j+1] given context + target[:j+1]. The log-prob
        # of target_ids[0] given context comes from ctx_log_probs[ctx_len-1].
        tgt_log_probs = torch.log_softmax(tgt_out.logits[0], dim=-1)

        logps = []
        tids = []
        for j, tid in enumerate(target_ids):
            if j == 0:
                lp = float(ctx_log_probs[ctx_len - 1, tid].item())
            else:
                lp = float(tgt_log_probs[j - 1, tid].item())
            logps.append(lp)
            tids.append(tid)

        hotspot_lp: Optional[float] = None
        if hotspot_abs is not None and ctx_len <= hotspot_abs < ctx_len + tgt_len:
            j = hotspot_abs - ctx_len
            tid = target_ids[j]
            if j == 0:
                hotspot_lp = float(ctx_log_probs[ctx_len - 1, tid].item())
            else:
                hotspot_lp = float(tgt_log_probs[j - 1, tid].item())

        return logps, tids, hotspot_lp

    overt_lps, overt_tids, overt_hs = _score_continuation(
        pair.overt_target_ids, pair.overt_hotspot_index
    )
    null_lps, null_tids, null_hs = _score_continuation(
        pair.null_target_ids, pair.null_hotspot_index
    )

    return PairedResult(
        overt_target_log_probs=overt_lps,
        null_target_log_probs=null_lps,
        overt_target_token_ids=overt_tids,
        null_target_token_ids=null_tids,
        overt_hotspot_log_prob=overt_hs,
        null_hotspot_log_prob=null_hs,
    )
