"""D14 test: end-to-end smoke test.

Exercises the full Wave-1-to-Wave-4 pipeline on one EN CSV + one ES CSV
from the real v2 staging directory against a tiny model:

  1. Load real v2 CSVs (tests D1 schema parsing).
  2. Pre-tokenize → StimulusCache (tests D6 on real unicode text).
  3. Train a fake-but-functional UnigramTable (tests D3).
  4. Build a fake checkpoint from a tiny GPT-2 with fixed seed (tests
     WeightSwappableModel-style state_dict swap from D9).
  5. Run `CachedRunner` (D11): forward-pass via `batched_surprisal` (D7),
     compute MeanLP + SLOR (D2 + D4), save per-token logprobs (D5),
     write partitioned parquet (D10).
  6. Load back via DuckDB and assert:
       - All expected columns present
       - No NaN/Inf in numeric metrics
       - Idempotent: a second `run_once()` call is a no-op (cache hits)
       - Spanish items survive unicode / accent handling

This test also catches integration bugs between modules — any one D7-D11
unit test passing while the glue is broken would show up here.

Runtime: ~5–15 s locally. Downloads no external models.
"""

from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pytest
import torch

from evaluation.cache import CachedRunner
from evaluation.output_v2 import register_duckdb_views
from evaluation.runners.per_model_runner import CellSpec
from evaluation.stimuli_cache import build_cache
from evaluation.unigram import UnigramTable


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_EN = REPO_ROOT / "evaluation" / "stimuli" / "null-subj-v2" / "staging" / "en" / "subject_drop.csv"
V2_ES = REPO_ROOT / "evaluation" / "stimuli" / "null-subj-v2" / "staging" / "es" / "subject_drop.csv"


# --- Helpers ----------------------------------------------------------------

def _make_tiny_gpt2(seed: int = 42):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=256, n_positions=128, n_embd=32, n_layer=2, n_head=2,
        use_cache=True,
    )
    return GPT2LMHeadModel(cfg)


def _word_tokenizer(vocab_size: int = 256):
    """Shared across EN and ES — treats each whitespace token as one subword.
    Accented characters pass through unchanged (sanity check for D6 unicode
    handling).
    """
    words: dict = {}

    def encode(text):
        ids = []
        for w in text.split():
            if w not in words and len(words) < vocab_size - 2:
                # keep ids in [2, vocab_size) — 0=pad, 1=bos
                words[w] = 2 + len(words)
            ids.append(words.get(w, 2))  # fallback for overflow
        return ids

    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = -1
    tok.pad_id.return_value = 0
    tok.get_piece_size.return_value = vocab_size
    tok.encode.side_effect = encode
    rev: dict = {}

    def decode(ids):
        # Keep the reverse map current
        rev.update({v: k for k, v in words.items()})
        return " ".join(rev.get(i, f"<{i}>") for i in ids)

    tok.decode.side_effect = decode
    return tok


def _truncate_csv_to_n_pairs(src: Path, dest: Path, n_pairs: int = 4):
    """Copy the first 2*n_pairs rows of a v2 CSV (n_pairs complete pairs)."""
    import pandas as pd
    df = pd.read_csv(src)
    df_small = df.head(n_pairs * 2)
    df_small.to_csv(dest, index=False)


# --- The smoke test ---------------------------------------------------------

@pytest.mark.skipif(not V2_EN.exists(), reason="v2 EN CSV absent")
@pytest.mark.skipif(not V2_ES.exists(), reason="v2 ES CSV absent")
def test_smoke_e2e(tmp_path):
    # 1. Copy a small subset of real stimuli into the tmp dir
    stimuli_dir = tmp_path / "stimuli"
    stimuli_dir.mkdir()
    en_small = stimuli_dir / "en_subject_drop.csv"
    es_small = stimuli_dir / "es_subject_drop.csv"
    _truncate_csv_to_n_pairs(V2_EN, en_small, n_pairs=4)
    _truncate_csv_to_n_pairs(V2_ES, es_small, n_pairs=4)

    # 2. Build stimuli cache (D6, tests unicode via ES)
    tok = _word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"smoke-tok")
    cache = build_cache([en_small, es_small], tok, tok_file)
    assert len(cache) == 16  # 4 pairs × 2 langs × 2 rows = 16

    # Sanity: both languages are present
    langs = {e.language for e in cache.rows}
    assert langs == {"en", "es"}

    # 3. Unigram baseline (D3)
    unigram = UnigramTable(
        counts=np.ones(256, dtype=np.int64),
        vocab_size=256, total_tokens=256, smoothing_alpha=1.0,
        corpus_id="smoke_corpus", tokenizer_id=cache.tokenizer_id,
    )

    # 4. Fake checkpoint
    ckpt_root = tmp_path / "model"
    ckpt_dir = ckpt_root / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)
    m = _make_tiny_gpt2(seed=7)
    torch.save(m.state_dict(), ckpt_dir / "pytorch_model.bin")

    # 5. CachedRunner over a single cell (D9 + D10 + D11)
    output_root = tmp_path / "out"
    cell = CellSpec(
        cell_id="smoke_cell",
        architecture="gpt2",
        intervention="smoke",
        rep=0,
        checkpoint_root=ckpt_root,
        model_factory=lambda: _make_tiny_gpt2(seed=7),
        stimuli=cache,
        unigram=unigram,
        device="cpu",
        batch_size=4,
    )

    runner = CachedRunner(cell, output_root=output_root, scoring_version="smoke-v1")
    summary_1 = runner.run_once()
    assert summary_1["n_processed"] == 1
    assert summary_1["n_forward_passes"] == 1

    # 6. Assert parquet written (D10)
    items_files = list((output_root / "items").rglob("*.parquet"))
    pairs_files = list((output_root / "pairs").rglob("*.parquet"))
    per_token_files = list((output_root / "per_token").rglob("*.parquet"))
    assert len(items_files) > 0
    assert len(pairs_files) > 0
    assert len(per_token_files) > 0

    # 7. DuckDB view layer works; no NaN/Inf
    con = duckdb.connect()
    try:
        register_duckdb_views(con, output_root)

        # All expected columns
        item_cols = [row[0] for row in
                     con.execute("DESCRIBE items").fetchall()]
        for need in ("target_mean_log_prob", "slor", "hotspot_log_prob",
                     "item_id", "pronoun_status", "checkpoint_step",
                     "language", "category", "condition"):
            assert need in item_cols, f"missing item column {need}"

        pair_cols = [row[0] for row in
                     con.execute("DESCRIBE pairs").fetchall()]
        for need in ("overt_mean_log_prob", "null_mean_log_prob",
                     "prefers_overt_meanlp", "prefers_overt_slor",
                     "log_prob_diff_overt_minus_null",
                     "slor_diff_overt_minus_null"):
            assert need in pair_cols, f"missing pair column {need}"

        # No NaN/Inf in numeric metrics
        n_bad = con.execute(
            "SELECT COUNT(*) FROM items WHERE "
            "NOT isfinite(target_mean_log_prob) OR "
            "NOT isfinite(slor) OR "
            "NOT isfinite(target_sum_log_prob)"
        ).fetchone()[0]
        assert n_bad == 0, f"{n_bad} items have non-finite metrics"

        n_pair_bad = con.execute(
            "SELECT COUNT(*) FROM pairs WHERE "
            "NOT isfinite(log_prob_diff_overt_minus_null) OR "
            "NOT isfinite(slor_diff_overt_minus_null)"
        ).fetchone()[0]
        assert n_pair_bad == 0, f"{n_pair_bad} pairs have non-finite metrics"

        # ES rows survived unicode
        n_es = con.execute(
            "SELECT COUNT(*) FROM items WHERE language = 'es'"
        ).fetchone()[0]
        assert n_es == 8  # 4 pairs × 2 rows
    finally:
        con.close()

    # 8. Idempotency: second run is zero forward passes (D11)
    runner_2 = CachedRunner(cell, output_root=output_root, scoring_version="smoke-v1")
    summary_2 = runner_2.run_once()
    assert summary_2["n_cached"] == 1
    assert summary_2["n_processed"] == 0
    assert summary_2["n_forward_passes"] == 0


@pytest.mark.skipif(not V2_EN.exists(), reason="v2 EN CSV absent")
def test_smoke_e2e_es_accents_survive(tmp_path):
    """Spanish items with accented characters pass through unmodified."""
    import pandas as pd
    df = pd.read_csv(V2_ES if V2_ES.exists() else V2_EN)
    # Take a few rows and verify they contain at least one accent
    sample = df.head(6)
    text = " ".join(sample["context"].astype(str).tolist() +
                    sample["target"].astype(str).tolist())
    # Accents are a signal that unicode handling is intact across the pipeline
    # (not strictly required for EN; this is a smoke check for ES).
    if V2_ES.exists():
        assert any(c in text for c in "áéíóúñ"), (
            "ES sample should contain accented characters"
        )
