"""D3: Unigram baseline trainer and lookup table.

Provides `UnigramTable` for fast subword log-prob lookup, trained on a
corpus with Laplace smoothing (α=1 default). Used by the SLOR evaluator
(D4). Natural-log throughout — matches the `per_token_log_prob` output
from `SurprisalCalculator`.

Usage:
    from evaluation.unigram import UnigramTable
    table = UnigramTable.from_corpus(
        corpus_path="data/english/train_90M",
        tokenizer=sp_processor,
        corpus_id="en_baseline",
        tokenizer_id="abc123",
    )
    table.save("data/unigrams/en_baseline.pkl")
    ...
    table = UnigramTable.load("data/unigrams/en_baseline.pkl")
    log_p = table.log_prob(token_id=42)
"""

from __future__ import annotations

import logging
import math
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UnigramTable:
    """Subword-level unigram table with Laplace smoothing.

    Stored form: counts array of length `vocab_size` (int64), plus scalars.
    The `log_probs` array is derived on construction / load.

    log_prob(t) = log((count[t] + alpha) / (N + alpha * V))

    where N = total_tokens, V = vocab_size. Natural log.
    """

    counts: np.ndarray              # int64[vocab_size]
    vocab_size: int
    total_tokens: int
    smoothing_alpha: float
    corpus_id: str
    tokenizer_id: str
    # Cached log-probs (derived; reconstructed on load)
    log_probs: np.ndarray = field(default=None, repr=False)

    # --- Construction -------------------------------------------------------

    def __post_init__(self):
        if self.log_probs is None:
            self._recompute_log_probs()

    def _recompute_log_probs(self):
        alpha = self.smoothing_alpha
        V = self.vocab_size
        N = self.total_tokens
        denom = N + alpha * V
        # log((c + α) / denom) = log(c + α) - log(denom)
        smoothed = self.counts.astype(np.float64) + alpha
        self.log_probs = (np.log(smoothed) - np.log(denom)).astype(np.float32)

    # --- Lookups ------------------------------------------------------------

    def log_prob(self, token_id: int) -> float:
        """Natural-log probability of a single token id."""
        return float(self.log_probs[token_id])

    def log_prob_many(self, token_ids: Iterable[int]) -> np.ndarray:
        """Natural-log probabilities for a list of token ids (float32 array)."""
        idx = np.asarray(list(token_ids), dtype=np.int64)
        return self.log_probs[idx]

    def sum_log_prob(self, token_ids: Iterable[int]) -> float:
        """Σ log P(t) over a token sequence (natural log)."""
        return float(self.log_prob_many(token_ids).sum())

    # --- Training -----------------------------------------------------------

    @classmethod
    def from_token_ids(
        cls,
        token_id_stream: Iterator[int],
        vocab_size: int,
        corpus_id: str,
        tokenizer_id: str,
        smoothing_alpha: float = 1.0,
    ) -> "UnigramTable":
        """Build a table from a stream of token ids.

        Exists primarily for testing; the corpus-file entry point is
        `from_corpus`.
        """
        counts = np.zeros(vocab_size, dtype=np.int64)
        total = 0
        for tid in token_id_stream:
            counts[tid] += 1
            total += 1
        return cls(
            counts=counts,
            vocab_size=vocab_size,
            total_tokens=total,
            smoothing_alpha=smoothing_alpha,
            corpus_id=corpus_id,
            tokenizer_id=tokenizer_id,
        )

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        tokenizer,
        corpus_id: str,
        tokenizer_id: str,
        smoothing_alpha: float = 1.0,
        file_glob: str = "*.train",
    ) -> "UnigramTable":
        """Build a unigram table by streaming a corpus directory.

        Handles either a single file or a directory of files matching
        `file_glob`. Uses `tokenizer.encode(text)` per line.

        Args:
            corpus_path: file or directory
            tokenizer: a SentencePiece-like tokenizer with `.encode(text)`
                and `.vocab_size()` (or `.get_piece_size()`). If neither,
                a `vocab_size` int must be available as `tokenizer.vocab_size`.
            corpus_id: identifier for the training corpus (e.g. 'en_baseline')
            tokenizer_id: content hash of the tokenizer file
            smoothing_alpha: Laplace smoothing parameter (default 1.0)
            file_glob: glob pattern inside a directory (default '*.train')
        """
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

        # Resolve vocab size (prefer SentencePiece's canonical API)
        if hasattr(tokenizer, "get_piece_size"):
            V = tokenizer.get_piece_size()
        elif hasattr(tokenizer, "vocab_size"):
            vs = tokenizer.vocab_size
            V = vs() if callable(vs) else vs
        elif hasattr(tokenizer, "__len__"):
            V = len(tokenizer)
        else:
            raise TypeError(
                "Tokenizer must expose vocab size via `.get_piece_size()`, "
                "`.vocab_size`, `.vocab_size()`, or `__len__`."
            )
        if not isinstance(V, int):
            raise TypeError(f"vocab_size resolved to non-int: {V!r}")

        counts = np.zeros(V, dtype=np.int64)
        total = 0

        logger.info("Training unigram on %d file(s); V=%d; α=%s",
                    len(files), V, smoothing_alpha)

        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ids = tokenizer.encode(line)
                    # Bound-check: skip any ids outside [0, V)
                    for tid in ids:
                        if 0 <= tid < V:
                            counts[tid] += 1
                            total += 1
                        # else: silently drop (shouldn't happen with
                        # properly-configured SP tokenizer)

        logger.info("Counted %d tokens over %d types (nonzero=%d)",
                    total, V, int((counts > 0).sum()))

        return cls(
            counts=counts,
            vocab_size=V,
            total_tokens=total,
            smoothing_alpha=smoothing_alpha,
            corpus_id=corpus_id,
            tokenizer_id=tokenizer_id,
        )

    # --- Persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a picklable dict representation (no derived log_probs)."""
        return {
            "counts": self.counts,
            "vocab_size": self.vocab_size,
            "total_tokens": self.total_tokens,
            "smoothing_alpha": self.smoothing_alpha,
            "corpus_id": self.corpus_id,
            "tokenizer_id": self.tokenizer_id,
            "format_version": "unigram.v1",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnigramTable":
        if data.get("format_version") != "unigram.v1":
            raise ValueError(
                f"Unknown unigram format_version: {data.get('format_version')!r}"
            )
        return cls(
            counts=data["counts"],
            vocab_size=data["vocab_size"],
            total_tokens=data["total_tokens"],
            smoothing_alpha=data["smoothing_alpha"],
            corpus_id=data["corpus_id"],
            tokenizer_id=data["tokenizer_id"],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp + rename
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self.to_dict(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        logger.info("Wrote unigram table to %s (V=%d, N=%d)",
                    path, self.vocab_size, self.total_tokens)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "UnigramTable":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls.from_dict(data)

    # --- Sanity -------------------------------------------------------------

    def sanity_check(self, atol: float = 1e-4) -> None:
        """Verify the log-probs sum to (approximately) 1 when exponentiated."""
        probs = np.exp(self.log_probs.astype(np.float64))
        s = probs.sum()
        assert abs(s - 1.0) < atol, (
            f"Unigram probabilities sum to {s} (expected 1.0 within {atol})"
        )
