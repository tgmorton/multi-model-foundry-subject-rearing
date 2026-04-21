"""D3 test: UnigramTable training, lookup, and round-trip persistence."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from evaluation.unigram import UnigramTable


def test_laplace_math_on_known_counts():
    """Hand-verified Laplace smoothing: V=4, N=6, counts=[3,2,1,0], α=1."""
    counts = np.array([3, 2, 1, 0], dtype=np.int64)
    table = UnigramTable(
        counts=counts,
        vocab_size=4,
        total_tokens=6,
        smoothing_alpha=1.0,
        corpus_id="toy",
        tokenizer_id="toy_tok",
    )
    # log P(t) = log((c + 1) / (6 + 4)) = log((c+1)/10)
    expected = [
        np.log(4 / 10),  # count=3
        np.log(3 / 10),  # count=2
        np.log(2 / 10),  # count=1
        np.log(1 / 10),  # count=0 → smoothed to 1
    ]
    for i, want in enumerate(expected):
        assert table.log_prob(i) == pytest.approx(want, abs=1e-6)


def test_sum_log_prob_over_sequence():
    counts = np.array([3, 2, 1, 0], dtype=np.int64)
    table = UnigramTable(
        counts=counts, vocab_size=4, total_tokens=6, smoothing_alpha=1.0,
        corpus_id="toy", tokenizer_id="toy_tok",
    )
    s = table.sum_log_prob([0, 1, 2, 3])
    expected = sum(np.log((c + 1) / 10) for c in [3, 2, 1, 0])
    assert s == pytest.approx(expected, abs=1e-6)


def test_from_token_ids_counts_correctly():
    stream = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    table = UnigramTable.from_token_ids(
        iter(stream), vocab_size=4, corpus_id="toy", tokenizer_id="tok",
    )
    np.testing.assert_array_equal(table.counts, [1, 2, 3, 4])
    assert table.total_tokens == 10


def test_round_trip_persistence(tmp_path):
    counts = np.array([5, 3, 1, 0, 0], dtype=np.int64)
    original = UnigramTable(
        counts=counts, vocab_size=5, total_tokens=9, smoothing_alpha=1.0,
        corpus_id="roundtrip", tokenizer_id="tok123",
    )
    path = tmp_path / "unigram.pkl"
    original.save(path)

    loaded = UnigramTable.load(path)
    assert loaded.vocab_size == original.vocab_size
    assert loaded.total_tokens == original.total_tokens
    assert loaded.corpus_id == original.corpus_id
    assert loaded.tokenizer_id == original.tokenizer_id
    np.testing.assert_array_equal(loaded.counts, original.counts)
    np.testing.assert_allclose(loaded.log_probs, original.log_probs, atol=1e-9)


def test_probabilities_sum_to_one():
    """After smoothing, sum over vocab of exp(log_prob) equals 1."""
    counts = np.array([10, 5, 3, 2, 0, 0, 0], dtype=np.int64)
    table = UnigramTable(
        counts=counts, vocab_size=7, total_tokens=20, smoothing_alpha=1.0,
        corpus_id="toy", tokenizer_id="tok",
    )
    probs = np.exp(table.log_probs.astype(np.float64))
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    # Also run the sanity check method
    table.sanity_check()


def test_from_corpus_with_mock_tokenizer(tmp_path):
    """Stream a tiny two-line corpus through a mock tokenizer."""
    corpus = tmp_path / "corpus.train"
    corpus.write_text("a b c\nb c d\n")

    tok = MagicMock()
    tok.get_piece_size.return_value = 5  # ids 0..4
    mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
    tok.encode.side_effect = lambda text: [mapping[w] for w in text.split()]

    table = UnigramTable.from_corpus(
        corpus_path=corpus,
        tokenizer=tok,
        corpus_id="toy",
        tokenizer_id="tok",
        smoothing_alpha=1.0,
    )
    # Expected counts: a=1, b=2, c=2, d=1, (id=4 never seen)=0
    np.testing.assert_array_equal(table.counts, [1, 2, 2, 1, 0])
    assert table.total_tokens == 6


def test_rejects_unknown_format_version(tmp_path):
    path = tmp_path / "bad.pkl"
    import pickle
    with open(path, "wb") as fh:
        pickle.dump({"format_version": "unknown.v99"}, fh)
    with pytest.raises(ValueError, match="format_version"):
        UnigramTable.load(path)
