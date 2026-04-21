"""D7 + D8 test: batched forward passes are numerically equivalent to
sequential forward passes.

**CI GATE.** If this test ever fails, the batching / KV-cache-reuse
optimizations have drifted from reference numerics and must not be
deployed.

Builds a tiny deterministic GPT-2 with fixed seed, picks a handful of
(context, target) pairs, and asserts:

  batched_surprisal(...) == [calculate_reference(...) for _ in items]
                            up to max |Δ| ≤ 1e-5

and the same for `paired_surprisal(...)` (D8).

The reference implementation is `_reference_sequential` below, which runs
`model(single_sequence)` one-at-a-time and extracts per-token log-probs
the standard way. Comparing against v1's `SurprisalCalculator` directly
is not strictly necessary: the invariant we care about is "batched ==
one-at-a-time", which this asserts on a tiny real model.
"""

from typing import List, Optional

import numpy as np
import pytest
import torch

from evaluation.core.batched_surprisal import (
    PairedInput,
    batched_surprisal,
    paired_surprisal,
)


# --- Reference (sequential) implementation ----------------------------------

def _reference_sequential(
    model,
    full_token_ids_list: List[List[int]],
    context_lens: List[int],
    hotspot_indices: List[Optional[int]],
    device: str = "cpu",
):
    """Run each sequence one-at-a-time; extract per-target-token log-probs.

    Returns the same shape of data as `batched_surprisal`.
    """
    results = []
    for seq, ctx_len, hotspot_abs in zip(full_token_ids_list, context_lens,
                                         hotspot_indices):
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        target_lps: List[float] = []
        target_tids: List[int] = []
        for j in range(ctx_len, len(seq)):
            # log P(seq[j] | seq[:j]) = log_probs[j-1, seq[j]]
            target_lps.append(float(log_probs[j - 1, seq[j]].item()))
            target_tids.append(seq[j])
        hotspot_lp: Optional[float] = None
        if hotspot_abs is not None and ctx_len <= hotspot_abs < len(seq):
            hotspot_lp = float(log_probs[hotspot_abs - 1, seq[hotspot_abs]].item())
        results.append({
            "target_log_probs": target_lps,
            "target_token_ids": target_tids,
            "hotspot_log_prob": hotspot_lp,
        })
    return results


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_gpt2():
    """A tiny deterministic GPT-2 instance."""
    torch.manual_seed(42)
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        use_cache=True,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    """A handful of (context, target) pairs as token-id sequences."""
    rng = np.random.default_rng(7)
    items = []
    # Mix lengths: some short, some longer; with and without hotspots.
    configs = [
        (5, 4, 2),    # ctx=5, target=4, hotspot at target idx 2
        (8, 3, 0),
        (3, 6, 4),
        (6, 5, None), # no hotspot
        (4, 7, 3),
        (10, 2, 1),
        (7, 4, None),
        (5, 5, 2),
    ]
    for ctx_len, tgt_len, hs_within_target in configs:
        total = ctx_len + tgt_len
        # Avoid id 0 (often treated as pad) and keep ids in [1, 127] since vocab=128
        ids = rng.integers(low=1, high=128, size=total).tolist()
        hotspot_abs = (ctx_len + hs_within_target) if hs_within_target is not None else None
        items.append({
            "full_token_ids": ids,
            "context_len": ctx_len,
            "hotspot_index": hotspot_abs,
        })
    return items


# --- D7: batched vs. sequential --------------------------------------------

def test_batched_equals_sequential(tiny_gpt2, sample_batch):
    full_ids_list = [it["full_token_ids"] for it in sample_batch]
    context_lens = [it["context_len"] for it in sample_batch]
    hotspot_indices = [it["hotspot_index"] for it in sample_batch]

    batched = batched_surprisal(
        tiny_gpt2,
        full_ids_list,
        context_lens,
        hotspot_indices,
        pad_id=0,
        device="cpu",
    )
    reference = _reference_sequential(
        tiny_gpt2,
        full_ids_list,
        context_lens,
        hotspot_indices,
        device="cpu",
    )

    assert len(batched) == len(reference)
    max_diff = 0.0
    for i, (b, r) in enumerate(zip(batched, reference)):
        assert b.target_token_ids == r["target_token_ids"], f"token id mismatch at item {i}"
        assert len(b.target_log_probs) == len(r["target_log_probs"])
        diffs = [abs(x - y) for x, y in zip(b.target_log_probs, r["target_log_probs"])]
        max_diff = max(max_diff, max(diffs) if diffs else 0.0)
        # Hotspot
        if r["hotspot_log_prob"] is None:
            assert b.hotspot_log_prob is None
        else:
            assert b.hotspot_log_prob is not None
            max_diff = max(max_diff, abs(b.hotspot_log_prob - r["hotspot_log_prob"]))
    assert max_diff <= 1e-5, (
        f"Batched vs. sequential diverged by {max_diff} (tolerance 1e-5)"
    )


def test_batched_handles_single_item(tiny_gpt2):
    """Sanity: batched primitive must work with B=1."""
    item = [[3, 4, 5, 6, 7, 8]]
    ctx_lens = [3]
    out = batched_surprisal(tiny_gpt2, item, ctx_lens, [None], device="cpu")
    assert len(out) == 1
    assert len(out[0].target_log_probs) == 3


def test_batched_handles_empty_batch(tiny_gpt2):
    """Empty batch → empty result, no crash."""
    out = batched_surprisal(tiny_gpt2, [], [], [], device="cpu")
    assert out == []


# --- D8: paired (KV-cache reuse) equivalence --------------------------------
#
# With proper cache hygiene (deep-copy before each continuation), the
# paired path is bit-exact with the full-sequence path. HF's DynamicCache
# mutates in-place even when use_cache=False, so running a second
# continuation without copying would corrupt the first. paired_surprisal
# copies the cache; the tolerance here matches D7.

_KV_CACHE_TOL = 1e-5


def test_paired_equals_batched(tiny_gpt2):
    """paired_surprisal(pair) ≈ batched_surprisal on the two full sequences.

    Tolerance 1e-3 reflects HF KV-cache numerical ordering, not a bug.
    """
    ctx = [3, 5, 7, 9, 11]
    overt_target = [2, 4, 6, 8]
    null_target = [4, 6, 8]

    pair = PairedInput(
        context_ids=ctx,
        overt_target_ids=overt_target,
        null_target_ids=null_target,
        overt_hotspot_index=len(ctx) + 1,
        null_hotspot_index=len(ctx) + 0,
    )

    paired = paired_surprisal(tiny_gpt2, pair, device="cpu")

    full_overt = ctx + overt_target
    full_null = ctx + null_target
    ref = batched_surprisal(
        tiny_gpt2,
        [full_overt, full_null],
        [len(ctx), len(ctx)],
        [pair.overt_hotspot_index, pair.null_hotspot_index],
        pad_id=0,
        device="cpu",
    )

    diffs_o = [
        abs(x - y)
        for x, y in zip(paired.overt_target_log_probs, ref[0].target_log_probs)
    ]
    assert max(diffs_o) <= _KV_CACHE_TOL, f"overt pair-vs-batched diff {max(diffs_o)}"
    diffs_n = [
        abs(x - y)
        for x, y in zip(paired.null_target_log_probs, ref[1].target_log_probs)
    ]
    assert max(diffs_n) <= _KV_CACHE_TOL, f"null pair-vs-batched diff {max(diffs_n)}"
    assert abs(paired.overt_hotspot_log_prob - ref[0].hotspot_log_prob) <= _KV_CACHE_TOL
    assert abs(paired.null_hotspot_log_prob - ref[1].hotspot_log_prob) <= _KV_CACHE_TOL


def test_paired_equals_sequential(tiny_gpt2):
    """paired_surprisal(pair) ≈ sequential reference. Tolerance 1e-3."""
    ctx = [3, 5, 7, 9, 11, 13]
    overt_target = [20, 21, 22]
    null_target = [21, 22]

    pair = PairedInput(
        context_ids=ctx,
        overt_target_ids=overt_target,
        null_target_ids=null_target,
    )

    paired = paired_surprisal(tiny_gpt2, pair, device="cpu")

    full_overt = ctx + overt_target
    full_null = ctx + null_target
    ref = _reference_sequential(
        tiny_gpt2,
        [full_overt, full_null],
        [len(ctx), len(ctx)],
        [None, None],
        device="cpu",
    )

    diffs_o = [abs(x - y) for x, y in zip(paired.overt_target_log_probs,
                                           ref[0]["target_log_probs"])]
    diffs_n = [abs(x - y) for x, y in zip(paired.null_target_log_probs,
                                           ref[1]["target_log_probs"])]
    max_diff = max(diffs_o + diffs_n)
    assert max_diff <= _KV_CACHE_TOL, (
        f"paired-vs-sequential diff {max_diff} (tol {_KV_CACHE_TOL})"
    )
