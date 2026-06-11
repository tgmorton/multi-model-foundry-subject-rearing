"""PLL scorer test: batched masked-LM scoring equals the naive reference.

**CI GATE** for `evaluation/core/pll_surprisal.py` (Salazar et al. 2020
pseudo-log-likelihood scoring). Builds a tiny deterministic
`BertForMaskedLM` and asserts:

  make_pll_scorer(mask_id)(...) == naive one-masked-copy-per-forward loop
                                   up to max |Δ| ≤ 1e-5

including with mixed-length sequences in one call (catches padding /
attention-mask bugs), plus contract checks: result alignment, target-span
lengths, hotspot reuse, and [SEP]-as-context behavior.
"""

from typing import List, Optional

import numpy as np
import pytest
import torch

from evaluation.core.pll_surprisal import make_pll_scorer

MASK_ID = 4
SEP_ID = 3
PAD_ID = 0


# --- Reference (naive) implementation ---------------------------------------

def _reference_pll(
    model,
    full_token_ids_list: List[List[int]],
    context_lens: List[int],
    hotspot_indices: List[Optional[int]],
    mask_id: int,
    sep_id: Optional[int] = None,
    device: str = "cpu",
):
    """One masked copy per forward, batch size 1, no padding.

    Returns the same shape of data as the batched scorer.
    """
    results = []
    for seq, ctx_len, hotspot_abs in zip(full_token_ids_list, context_lens,
                                         hotspot_indices):
        L = len(seq)
        if ctx_len >= L:
            results.append({
                "target_log_probs": [],
                "target_token_ids": [],
                "hotspot_log_prob": None,
            })
            continue
        scored_seq = (seq + [sep_id]) if sep_id is not None else seq
        target_lps: List[float] = []
        target_tids: List[int] = []
        for t in range(ctx_len, L):
            row = list(scored_seq)
            row[t] = mask_id
            ids = torch.tensor([row], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0]
            log_probs = torch.log_softmax(logits[t], dim=-1)
            target_lps.append(float(log_probs[seq[t]].item()))
            target_tids.append(seq[t])
        hotspot_lp: Optional[float] = None
        if hotspot_abs is not None and ctx_len <= hotspot_abs < L:
            hotspot_lp = target_lps[hotspot_abs - ctx_len]
        results.append({
            "target_log_probs": target_lps,
            "target_token_ids": target_tids,
            "hotspot_log_prob": hotspot_lp,
        })
    return results


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_bert():
    """A tiny deterministic BertForMaskedLM instance."""
    torch.manual_seed(42)
    from transformers import BertConfig, BertForMaskedLM
    cfg = BertConfig(
        vocab_size=50,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=PAD_ID,
    )
    model = BertForMaskedLM(cfg)
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    """Mixed-length (context, target) pairs as token-id sequences.

    Token ids stay in [5, 49] so they never collide with pad (0),
    [SEP] (3), or [MASK] (4).
    """
    rng = np.random.default_rng(7)
    items = []
    configs = [
        (5, 4, 2),     # ctx=5, target=4, hotspot at target idx 2
        (8, 3, 0),
        (3, 6, 4),
        (6, 5, None),  # no hotspot
        (4, 7, 3),
        (10, 2, 1),
        (7, 4, None),
        (5, 5, 2),
    ]
    for ctx_len, tgt_len, hs_within_target in configs:
        total = ctx_len + tgt_len
        ids = rng.integers(low=5, high=50, size=total).tolist()
        hotspot_abs = (ctx_len + hs_within_target) if hs_within_target is not None else None
        items.append({
            "full_token_ids": ids,
            "context_len": ctx_len,
            "hotspot_index": hotspot_abs,
        })
    return items


def _unpack(sample_batch):
    return (
        [it["full_token_ids"] for it in sample_batch],
        [it["context_len"] for it in sample_batch],
        [it["hotspot_index"] for it in sample_batch],
    )


# --- Contract: alignment, lengths, value ranges ------------------------------

def test_result_alignment_and_lengths(tiny_bert, sample_batch):
    full_ids_list, context_lens, hotspot_indices = _unpack(sample_batch)
    scorer = make_pll_scorer(mask_id=MASK_ID)
    out = scorer(tiny_bert, full_ids_list, context_lens, hotspot_indices,
                 pad_id=PAD_ID, device="cpu")

    assert len(out) == len(sample_batch)
    for res, seq, ctx_len in zip(out, full_ids_list, context_lens):
        span = len(seq) - ctx_len
        assert len(res.target_log_probs) == span
        assert res.target_token_ids == seq[ctx_len:]
        for lp in res.target_log_probs:
            assert np.isfinite(lp)
            assert lp <= 0.0  # log-probs


def test_empty_batch_and_degenerate_item(tiny_bert):
    scorer = make_pll_scorer(mask_id=MASK_ID)
    assert scorer(tiny_bert, [], [], [], device="cpu") == []
    # context_len >= len → empty result, like the causal scorer.
    out = scorer(tiny_bert, [[5, 6, 7]], [3], [None], device="cpu")
    assert len(out) == 1
    assert out[0].target_log_probs == []
    assert out[0].target_token_ids == []
    assert out[0].hotspot_log_prob is None


# --- Equivalence gate: batched == naive reference -----------------------------

@pytest.mark.parametrize("max_masked_rows", [1, 3, 256])
@pytest.mark.parametrize("use_sep", [False, True])
def test_batched_equals_naive_reference(tiny_bert, sample_batch,
                                        max_masked_rows, use_sep):
    """Mixed-length sequences in one call; small `max_masked_rows` values
    force row-chunking across sequence boundaries. Catches padding /
    attention-mask bugs."""
    full_ids_list, context_lens, hotspot_indices = _unpack(sample_batch)
    sep = SEP_ID if use_sep else None
    scorer = make_pll_scorer(mask_id=MASK_ID, sep_id=sep,
                             max_masked_rows=max_masked_rows)
    batched = scorer(tiny_bert, full_ids_list, context_lens, hotspot_indices,
                     pad_id=PAD_ID, device="cpu")
    reference = _reference_pll(tiny_bert, full_ids_list, context_lens,
                               hotspot_indices, mask_id=MASK_ID, sep_id=sep,
                               device="cpu")

    assert len(batched) == len(reference)
    for i, (b, r) in enumerate(zip(batched, reference)):
        assert b.target_token_ids == r["target_token_ids"], f"item {i}"
        np.testing.assert_allclose(
            b.target_log_probs, r["target_log_probs"], atol=1e-5,
            err_msg=f"batched vs naive PLL diverged at item {i}",
        )
        if r["hotspot_log_prob"] is None:
            assert b.hotspot_log_prob is None
        else:
            assert b.hotspot_log_prob == pytest.approx(
                r["hotspot_log_prob"], abs=1e-5
            )


# --- Hotspot: reused from the target span, not recomputed ---------------------

def test_hotspot_equals_target_span_entry(tiny_bert, sample_batch):
    full_ids_list, context_lens, hotspot_indices = _unpack(sample_batch)
    scorer = make_pll_scorer(mask_id=MASK_ID)
    out = scorer(tiny_bert, full_ids_list, context_lens, hotspot_indices,
                 pad_id=PAD_ID, device="cpu")
    for res, ctx_len, h in zip(out, context_lens, hotspot_indices):
        if h is None:
            assert res.hotspot_log_prob is None
        else:
            # Exact equality: the hotspot value IS the span entry.
            assert res.hotspot_log_prob == res.target_log_probs[h - ctx_len]


# --- SEP: changes scores (more context) but not lengths / indexing -----------

def test_sep_changes_scores_not_indexing(tiny_bert, sample_batch):
    full_ids_list, context_lens, hotspot_indices = _unpack(sample_batch)
    no_sep = make_pll_scorer(mask_id=MASK_ID, sep_id=None)(
        tiny_bert, full_ids_list, context_lens, hotspot_indices,
        pad_id=PAD_ID, device="cpu",
    )
    with_sep = make_pll_scorer(mask_id=MASK_ID, sep_id=SEP_ID)(
        tiny_bert, full_ids_list, context_lens, hotspot_indices,
        pad_id=PAD_ID, device="cpu",
    )

    any_changed = False
    for a, b in zip(no_sep, with_sep):
        # Lengths and token-id indexing unchanged.
        assert len(a.target_log_probs) == len(b.target_log_probs)
        assert a.target_token_ids == b.target_token_ids
        if any(abs(x - y) > 1e-7
               for x, y in zip(a.target_log_probs, b.target_log_probs)):
            any_changed = True
    # SEP participates as context, so scores must shift somewhere.
    assert any_changed, "appending SEP did not change any PLL score"
