#!/usr/bin/env python3
"""D3: Train a subword-level unigram baseline on a training corpus.

One-off script that builds a `UnigramTable` for a given (corpus, tokenizer)
pair and serializes it as a pickle for fast loading at eval time. Used by
the SLOR evaluator (D4) with per-corpus (FIT-CLAMS) baselines.

Usage:
    python scripts/train_unigram.py \
        --corpus data/english/train_90M \
        --tokenizer tokenizers/exp0_baseline/tokenizer.model \
        --corpus_id en_baseline \
        --output data/unigrams/en_baseline.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path


def _tokenizer_id(tokenizer_path: Path) -> str:
    """Content-hash the tokenizer file for provenance (first 12 hex chars)."""
    h = hashlib.sha256()
    with open(tokenizer_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Train a unigram baseline.")
    parser.add_argument("--corpus", required=True,
                        help="Corpus file or directory.")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to SentencePiece tokenizer.model file.")
    parser.add_argument("--corpus_id", required=True,
                        help="Identifier for the corpus, e.g. 'en_baseline'.")
    parser.add_argument("--output", required=True,
                        help="Output pickle path.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Laplace smoothing α (default 1.0).")
    parser.add_argument("--file_glob", default="*.train",
                        help="Glob pattern for corpus files in directory mode.")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Verify probabilities sum to 1.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    import sentencepiece as spm
    from evaluation.unigram import UnigramTable

    tokenizer_path = Path(args.tokenizer)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))

    tok_id = _tokenizer_id(tokenizer_path)

    table = UnigramTable.from_corpus(
        corpus_path=args.corpus,
        tokenizer=tokenizer,
        corpus_id=args.corpus_id,
        tokenizer_id=tok_id,
        smoothing_alpha=args.alpha,
        file_glob=args.file_glob,
    )

    if args.sanity_check:
        table.sanity_check()

    out = Path(args.output)
    table.save(out)
    print(f"Wrote unigram table: {out}")
    print(f"  corpus_id={table.corpus_id}  tokenizer_id={table.tokenizer_id}")
    print(f"  V={table.vocab_size}  N={table.total_tokens}  α={table.smoothing_alpha}")


if __name__ == "__main__":
    main()
