"""Pseudo-log-likelihood (PLL) scoring for bidirectional masked LMs.

Implements Masked Language Model Scoring (Salazar et al. 2020,
"Masked Language Model Scoring", ACL) as a drop-in replacement for the
causal `batched_surprisal` scorer: `make_pll_scorer(mask_id, ...)`
returns a callable with the exact same signature and return type
(`List[BatchedSurprisalResult]`), so it can be passed anywhere
`batched_surprisal` is (e.g. `PerModelRunner.evaluate_checkpoint`).

Semantics:

  - Inputs are `[CLS] context... target...` sequences ([CLS] is
    prepended upstream as the BOS token). If `sep_id` is given, a [SEP]
    token is appended to each sequence before scoring; it participates
    as bidirectional context but is never itself scored, and all
    hotspot/target indexing into the ORIGINAL sequence is unchanged.
  - For each sequence and each TARGET position t (context positions are
    never masked), we build a copy with position t replaced by
    `mask_id`, run the masked LM, and gather
    `log P(original_token_t | rest of sequence)` from the log-softmax
    of the logits at position t. The sum over the target span is the
    PLL of the target given bidirectional context.
  - One forward row per (sequence, target position). Rows are batched
    across sequences AND positions into forwards of at most
    `max_masked_rows` rows, right-padded with `pad_id` + attention mask
    (pads get mask 0; the appended [SEP] gets mask 1).

Numerics: runs under `torch.no_grad()` in the model's native dtype
(fp32 for eval) — no `.half()` / autocast. Equivalence with a naive
one-row-per-forward reference loop is asserted in
`evaluation/tests/test_pll_surprisal.py`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from evaluation.core.batched_surprisal import BatchedSurprisalResult, _pad_batch

logger = logging.getLogger(__name__)


def make_pll_scorer(
    mask_id: int,
    sep_id: Optional[int] = None,
    max_masked_rows: int = 256,
) -> Callable:
    """Build a PLL scorer with the same interface as `batched_surprisal`.

    Args:
      mask_id: token id of the model's [MASK] token.
      sep_id: optional token id of [SEP]; when given it is appended to
              each sequence as extra (never-scored) context.
      max_masked_rows: cap on rows per forward pass. Each row is one
              masked copy of one sequence, so VRAM scales with
              `max_masked_rows * max_seq_len`.

    Returns:
      A callable `pll_surprisal(model, full_token_ids_list, context_lens,
      hotspot_indices=None, *, pad_id=0, device=None)
      -> List[BatchedSurprisalResult]`.
    """
    if max_masked_rows < 1:
        raise ValueError("max_masked_rows must be >= 1")

    def pll_surprisal(
        model,
        full_token_ids_list: Sequence[Sequence[int]],
        context_lens: Sequence[int],
        hotspot_indices: Optional[Sequence[Optional[int]]] = None,
        *,
        pad_id: int = 0,
        device: Optional[str] = None,
    ) -> List[BatchedSurprisalResult]:
        """Score a batch of (context + target) sequences by PLL.

        Same contract as `evaluation.core.batched_surprisal.batched_surprisal`,
        except each target log-prob is the masked-LM pseudo-log-likelihood
        log P(token_t | sequence with position t masked) rather than the
        causal left-to-right log-prob.
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

        # --- Enumerate one masked row per (sequence, target position) -------
        # rows[k] is the masked copy; row_meta[k] = (seq_index, target_pos).
        rows: List[List[int]] = []
        row_meta: List[Tuple[int, int]] = []
        for i, seq in enumerate(sequences):
            L = len(seq)
            context_len = int(context_lens[i])
            if context_len >= L:
                continue  # degenerate; handled at assembly time
            scored_seq = (seq + [sep_id]) if sep_id is not None else seq
            for t in range(context_len, L):
                row = list(scored_seq)
                row[t] = mask_id
                rows.append(row)
                row_meta.append((i, t))

        # --- Batched forwards over masked rows ------------------------------
        # per_seq_lps[i][t] = log P(seq[t] | masked context)
        per_seq_lps: Dict[int, Dict[int, float]] = {i: {} for i in range(len(sequences))}
        with torch.no_grad():
            for start in range(0, len(rows), max_masked_rows):
                chunk_rows = rows[start:start + max_masked_rows]
                chunk_meta = row_meta[start:start + max_masked_rows]
                input_ids, attention_mask = _pad_batch(chunk_rows, pad_id=pad_id)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (B, T, V)

                B = len(chunk_rows)
                pos = torch.tensor([t for (_, t) in chunk_meta],
                                   dtype=torch.long, device=device)
                tok = torch.tensor(
                    [sequences[i][t] for (i, t) in chunk_meta],
                    dtype=torch.long, device=device,
                )
                # Only the masked position of each row is needed: gather the
                # (B, V) logit slice there, log-softmax, pick the original id.
                masked_logits = logits[torch.arange(B, device=device), pos]  # (B, V)
                lps = torch.log_softmax(masked_logits, dim=-1)
                gathered = lps[torch.arange(B, device=device), tok]  # (B,)
                for (i, t), lp in zip(chunk_meta, gathered.tolist()):
                    per_seq_lps[i][t] = float(lp)

        # --- Assemble results aligned with input order -----------------------
        results: List[BatchedSurprisalResult] = []
        for i, seq in enumerate(sequences):
            L = len(seq)
            context_len = int(context_lens[i])
            if context_len >= L:
                # No target tokens at all (degenerate) — match the causal scorer.
                results.append(BatchedSurprisalResult(
                    target_log_probs=[],
                    target_token_ids=[],
                    hotspot_log_prob=None,
                ))
                continue

            target_positions = list(range(context_len, L))
            target_log_probs = [per_seq_lps[i][t] for t in target_positions]

            h = hotspot_indices[i]
            hotspot_lp: Optional[float] = None
            if h is not None and context_len <= h < L:
                # Already computed as part of the target span — reuse, don't
                # recompute.
                hotspot_lp = per_seq_lps[i][h]

            results.append(BatchedSurprisalResult(
                target_log_probs=target_log_probs,
                target_token_ids=[seq[t] for t in target_positions],
                hotspot_log_prob=hotspot_lp,
            ))

        return results

    return pll_surprisal
