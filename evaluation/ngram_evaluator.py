"""D12: N-gram fast-path evaluator.

Evaluates stimuli on an n-gram language model without any torch forward
pass — pure table lookups with Laplace smoothing. Outputs the same Stage-4
schema as the GPT-style evaluator, so n-gram results slot into the same
parquet tree under a separate architecture tag (`ngram_{n}`).

An n-gram model is stored as `NgramTable` — a pickled context→Counter
structure with vocab-size and smoothing-α metadata. The trainer in this
module is deliberately simple (and CPU-only): iterate the corpus, count
(n-1)-gram prefixes and their next tokens, save.

Usage:
    from evaluation.ngram_evaluator import NgramTable, evaluate_cell
    table = NgramTable.from_corpus(
        corpus_path=..., tokenizer=..., order=3,
        corpus_id="en_baseline", tokenizer_id=...,
    )
    table.save("...pkl")

    items, pairs = evaluate_cell(table, stimuli_cache, cell_id="ngram_3__en_baseline__rep00")
"""

from __future__ import annotations

import logging
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from evaluation.runners.per_model_runner import (
    CheckpointItemResult,
    CheckpointPairResult,
    _build_pair_results,
)
from evaluation.stimuli_cache import StimulusCache

logger = logging.getLogger(__name__)


# --- N-gram model -----------------------------------------------------------

@dataclass
class NgramTable:
    """Context → Counter(next_token) table with Laplace smoothing.

    For order=n, `context` is a tuple of (n-1) most recent token ids
    (or a shorter prefix at sequence start). Smoothing treats every
    vocab token as having α prior count, so unseen (context, next)
    pairs still get a valid probability.

    log P(w | c) = log((count[c][w] + α) / (Σ_w count[c][w] + α * V))
    """
    order: int                              # n in n-gram
    vocab_size: int
    smoothing_alpha: float
    corpus_id: str
    tokenizer_id: str
    # context (tuple of ints, len = order-1) → Counter {next_token_id: count}
    counts: Dict[Tuple[int, ...], Counter]
    # Cached context-level totals (context → Σ counts); filled lazily.
    _totals: Optional[Dict[Tuple[int, ...], int]] = None

    def __post_init__(self):
        if self._totals is None:
            self._totals = {c: sum(counter.values())
                             for c, counter in self.counts.items()}

    # --- Lookup ---------------------------------------------------------

    def log_prob(self, context: Sequence[int], next_token: int) -> float:
        """Natural-log conditional probability P(next | context).

        For order=n, uses the last (n-1) elements of context (shorter
        prefixes at sequence start use whatever is available — they
        live in the same table under their actual length).
        """
        prefix = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        counter = self.counts.get(prefix)
        total = self._totals.get(prefix, 0) if self._totals else 0
        if counter is None or total == 0:
            # Prefix never seen in training — back off to uniform + smoothing
            return math.log(
                self.smoothing_alpha / (self.smoothing_alpha * self.vocab_size)
            )
        numer = counter.get(next_token, 0) + self.smoothing_alpha
        denom = total + self.smoothing_alpha * self.vocab_size
        return math.log(numer / denom)

    def sum_log_prob(
        self,
        token_ids: Sequence[int],
        context_prefix: Sequence[int] = (),
    ) -> float:
        """Σ log P(w_i | context_prefix + w_{<i}) over a token sequence."""
        history = list(context_prefix)
        total = 0.0
        for tid in token_ids:
            total += self.log_prob(history, tid)
            history.append(tid)
        return total

    def per_token_log_prob(
        self,
        token_ids: Sequence[int],
        context_prefix: Sequence[int] = (),
    ) -> List[float]:
        """Per-token natural-log log-prob, given preceding context_prefix."""
        history = list(context_prefix)
        out = []
        for tid in token_ids:
            out.append(self.log_prob(history, tid))
            history.append(tid)
        return out

    # --- Training -------------------------------------------------------

    @classmethod
    def from_token_ids(
        cls,
        token_streams: Iterable[Iterable[int]],
        order: int,
        vocab_size: int,
        corpus_id: str,
        tokenizer_id: str,
        smoothing_alpha: float = 1.0,
    ) -> "NgramTable":
        """Build an n-gram table from an iterable of sequences (each a
        sentence or line, as an iterable of token ids).
        """
        if order < 1:
            raise ValueError(f"order must be ≥ 1, got {order}")
        counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)

        for stream in token_streams:
            ids = list(stream)
            for i, tid in enumerate(ids):
                start = max(0, i - (order - 1))
                prefix = tuple(ids[start:i])
                counts[prefix][tid] += 1

        return cls(
            order=order,
            vocab_size=vocab_size,
            smoothing_alpha=smoothing_alpha,
            corpus_id=corpus_id,
            tokenizer_id=tokenizer_id,
            counts=dict(counts),
        )

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        tokenizer,
        order: int,
        corpus_id: str,
        tokenizer_id: str,
        smoothing_alpha: float = 1.0,
        file_glob: str = "*.train",
    ) -> "NgramTable":
        """Train from a corpus file or directory."""
        corpus_path = Path(corpus_path)
        if corpus_path.is_dir():
            files = sorted(corpus_path.glob(file_glob))
            if not files:
                raise FileNotFoundError(
                    f"No files matching {file_glob!r} in {corpus_path}"
                )
        elif corpus_path.is_file():
            files = [corpus_path]
        else:
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

        if hasattr(tokenizer, "get_piece_size"):
            V = tokenizer.get_piece_size()
        elif hasattr(tokenizer, "vocab_size"):
            vs = tokenizer.vocab_size
            V = vs() if callable(vs) else vs
        else:
            raise TypeError(
                "Tokenizer must expose `.get_piece_size()` or `.vocab_size`"
            )
        if not isinstance(V, int):
            raise TypeError(f"vocab_size resolved to non-int: {V!r}")

        logger.info("Training %d-gram on %d file(s); V=%d; α=%s",
                    order, len(files), V, smoothing_alpha)

        def streams():
            for fp in files:
                with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        ids = [t for t in tokenizer.encode(line) if 0 <= t < V]
                        if ids:
                            yield ids

        return cls.from_token_ids(
            streams(),
            order=order,
            vocab_size=V,
            corpus_id=corpus_id,
            tokenizer_id=tokenizer_id,
            smoothing_alpha=smoothing_alpha,
        )

    # --- Persistence ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": "ngram.v1",
            "order": self.order,
            "vocab_size": self.vocab_size,
            "smoothing_alpha": self.smoothing_alpha,
            "corpus_id": self.corpus_id,
            "tokenizer_id": self.tokenizer_id,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NgramTable":
        if data.get("format_version") != "ngram.v1":
            raise ValueError(
                f"Unknown ngram format_version: {data.get('format_version')!r}"
            )
        return cls(
            order=data["order"],
            vocab_size=data["vocab_size"],
            smoothing_alpha=data["smoothing_alpha"],
            corpus_id=data["corpus_id"],
            tokenizer_id=data["tokenizer_id"],
            counts=data["counts"],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NgramTable":
        with open(path, "rb") as fh:
            return cls.from_dict(pickle.load(fh))


# --- Evaluation -------------------------------------------------------------

def evaluate_cell(
    table: NgramTable,
    stimuli: StimulusCache,
    *,
    cell_id: str,
    architecture: Optional[str] = None,
    intervention: str = "",
    rep: int = 0,
    checkpoint_step: int = 0,
    checkpoint_path: str = "<ngram>",
    unigram=None,
) -> Tuple[List[CheckpointItemResult], List[CheckpointPairResult]]:
    """Score every stimulus with the n-gram table; emit Stage-4 results.

    `architecture` defaults to `f"ngram_{table.order}"`. `checkpoint_step`
    defaults to 0 since n-gram models don't have gradient checkpoints.
    When `unigram` is supplied, SLOR is computed; otherwise the SLOR
    fields are None.
    """
    if architecture is None:
        architecture = f"ngram_{table.order}"

    item_results: List[CheckpointItemResult] = []
    for e in stimuli.rows:
        target_ids = e.target_token_ids
        context_ids = e.context_token_ids
        lps = table.per_token_log_prob(target_ids, context_prefix=context_ids)
        target_sum = float(sum(lps))
        target_n = len(lps)
        target_mean = (target_sum / target_n) if target_n else 0.0

        unigram_sum = None
        slor = None
        if unigram is not None and target_n > 0:
            unigram_sum = float(unigram.sum_log_prob(target_ids))
            slor = (target_sum - unigram_sum) / target_n

        # Hotspot: full_token_ids[hotspot_token_index] is the scored token.
        hotspot_lp: Optional[float] = None
        if e.context_len <= e.hotspot_token_index < e.full_len and target_n > 0:
            j = e.hotspot_token_index - e.context_len
            if 0 <= j < target_n:
                hotspot_lp = lps[j]

        item_results.append(CheckpointItemResult(
            cell_id=cell_id,
            architecture=architecture,
            intervention=intervention,
            rep=rep,
            checkpoint_step=checkpoint_step,
            checkpoint_path=checkpoint_path,
            item_id=e.item_id,
            category=e.category,
            condition=e.condition,
            pronoun_status=e.pronoun_status,
            language=e.language,
            target_sum_log_prob=target_sum,
            target_mean_log_prob=target_mean,
            target_n_tokens=target_n,
            target_unigram_sum_log_prob=unigram_sum,
            slor=slor,
            hotspot_log_prob=hotspot_lp,
            per_token_log_prob=lps,
            per_token_ids=list(target_ids),
        ))

    pair_results = _build_pair_results(item_results)
    return item_results, pair_results
