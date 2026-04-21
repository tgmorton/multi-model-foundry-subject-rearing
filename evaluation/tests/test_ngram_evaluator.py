"""D12 test: n-gram evaluator math + stimuli evaluation."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from evaluation.ngram_evaluator import NgramTable, evaluate_cell
from evaluation.stimuli_cache import build_cache
from evaluation.unigram import UnigramTable


# --- Math tests -------------------------------------------------------------

def test_bigram_log_prob_hand_computed():
    """With α=1, V=4: counts[(1,)] = {2: 3, 3: 1}, total=4.
    log P(2|1) = log((3+1)/(4+4)) = log(4/8) = log(0.5)
    log P(5|1) = log((0+1)/(4+4)) = log(1/8)  (unseen next token)
    """
    table = NgramTable.from_token_ids(
        token_streams=[[1, 2], [1, 2], [1, 2], [1, 3]],
        order=2, vocab_size=4,
        corpus_id="toy", tokenizer_id="tok",
        smoothing_alpha=1.0,
    )
    assert table.log_prob([1], 2) == pytest.approx(np.log(0.5), abs=1e-9)
    assert table.log_prob([1], 3) == pytest.approx(np.log(2 / 8), abs=1e-9)
    # Unseen next token after seen context
    assert table.log_prob([1], 0) == pytest.approx(np.log(1 / 8), abs=1e-9)


def test_unseen_context_backs_off_to_uniform_laplace():
    """For a context never seen in training, log P = log(1/V)."""
    table = NgramTable.from_token_ids(
        token_streams=[[1, 2, 3]], order=3, vocab_size=10,
        corpus_id="toy", tokenizer_id="tok", smoothing_alpha=1.0,
    )
    # Context (7, 9) was never in training
    lp = table.log_prob([7, 9], 3)
    assert lp == pytest.approx(np.log(1 / 10), abs=1e-9)


def test_sum_log_prob_over_sequence():
    table = NgramTable.from_token_ids(
        token_streams=[[1, 2, 3, 4, 5]], order=2, vocab_size=10,
        corpus_id="toy", tokenizer_id="tok", smoothing_alpha=1.0,
    )
    # P(w_1|w_0)*P(w_2|w_1)*...
    expected = sum(table.log_prob([1, 2, 3, 4][:i], t)
                    for i, t in enumerate([1, 2, 3, 4]))
    # sum_log_prob with empty context_prefix scores each token given
    # its left history within token_ids
    s = table.sum_log_prob([1, 2, 3, 4])
    assert s == pytest.approx(expected, abs=1e-9)


def test_round_trip_persistence(tmp_path):
    table = NgramTable.from_token_ids(
        token_streams=[[1, 2, 3, 2, 1]], order=3, vocab_size=5,
        corpus_id="rt", tokenizer_id="tok", smoothing_alpha=1.0,
    )
    p = tmp_path / "ng.pkl"
    table.save(p)
    loaded = NgramTable.load(p)
    assert loaded.order == table.order
    assert loaded.vocab_size == table.vocab_size
    assert loaded.counts == table.counts
    # Identical log_probs
    for context, next_tok in [([1], 2), ([2, 3], 2), ([7], 4)]:
        assert loaded.log_prob(context, next_tok) == pytest.approx(
            table.log_prob(context, next_tok), abs=1e-9
        )


# --- Evaluate on a stimulus cache -------------------------------------------

def _word_tokenizer():
    words: dict = {}

    def encode(text):
        ids = []
        for w in text.split():
            if w not in words:
                words[w] = 10 + len(words)
            ids.append(words[w])
        return ids

    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = -1
    tok.pad_id.return_value = 0
    tok.get_piece_size.return_value = 128
    tok.encode.side_effect = encode
    tok.decode.side_effect = lambda ids: "".join(f"<{i}>" for i in ids)
    return tok


def _v2_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 1, "context": "ana won prize .",
         "target": "she shows pride .",
         "hotspot_token": "shows", "hotspot_position": 1,
         "language": "en", "names": "ana", "generator": "test"},
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 0, "context": "ana won prize .",
         "target": "shows pride .",
         "hotspot_token": "shows", "hotspot_position": 0,
         "language": "en", "names": "ana", "generator": "test"},
    ])
    p = tmp_path / "stimuli.csv"
    df.to_csv(p, index=False)
    return p


def test_evaluate_cell_produces_item_and_pair_results(tmp_path):
    csv = _v2_csv(tmp_path)
    tok = _word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"tok")
    cache = build_cache([csv], tok, tok_file)

    # Train a trigram on a tiny corpus
    table = NgramTable.from_token_ids(
        token_streams=[[10, 11, 12, 13, 14]], order=3, vocab_size=128,
        corpus_id="toy", tokenizer_id=cache.tokenizer_id, smoothing_alpha=1.0,
    )

    items, pairs = evaluate_cell(table, cache, cell_id="ngram_3__toy__rep00")

    assert len(items) == 2
    assert len(pairs) == 1

    for r in items:
        assert r.architecture == "ngram_3"
        assert r.target_n_tokens > 0
        assert r.per_token_log_prob  # non-empty
        assert sum(r.per_token_log_prob) == pytest.approx(
            r.target_sum_log_prob, abs=1e-9
        )

    pair = pairs[0]
    assert pair.architecture == "ngram_3"
    assert pair.prefers_overt_meanlp in (True, False)


def test_evaluate_cell_with_unigram_produces_slor(tmp_path):
    csv = _v2_csv(tmp_path)
    tok = _word_tokenizer()
    tok_file = tmp_path / "tok.model"
    tok_file.write_bytes(b"tok")
    cache = build_cache([csv], tok, tok_file)

    table = NgramTable.from_token_ids(
        token_streams=[[10, 11, 12]], order=2, vocab_size=128,
        corpus_id="toy", tokenizer_id=cache.tokenizer_id,
    )
    unigram = UnigramTable(
        counts=np.ones(128, dtype=np.int64),
        vocab_size=128, total_tokens=128, smoothing_alpha=1.0,
        corpus_id="u", tokenizer_id=cache.tokenizer_id,
    )

    items, pairs = evaluate_cell(
        table, cache, cell_id="ngram_2__toy__rep00", unigram=unigram,
    )
    for r in items:
        assert r.slor is not None
    assert pairs[0].slor_diff_overt_minus_null is not None


def test_from_corpus_builds_counts(tmp_path):
    corpus = tmp_path / "c.train"
    corpus.write_text("a b c\na b a\n")

    tok = MagicMock()
    tok.get_piece_size.return_value = 5
    mapping = {"a": 0, "b": 1, "c": 2}
    tok.encode.side_effect = lambda text: [mapping[w] for w in text.split()]

    table = NgramTable.from_corpus(
        corpus_path=corpus, tokenizer=tok, order=2,
        corpus_id="toy", tokenizer_id="tok", smoothing_alpha=1.0,
    )
    # (a,) → {b: 2}
    # (b,) → {c: 1, a: 1}
    # (,)  → {a: 2, ...}   (empty-prefix counts = unigram counts)
    assert table.counts[(0,)] == {1: 2}
    assert table.counts[(1,)] == {2: 1, 0: 1}


def test_rejects_unknown_format_version(tmp_path):
    import pickle
    p = tmp_path / "bad.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"format_version": "bad.v1"}, fh)
    with pytest.raises(ValueError, match="format_version"):
        NgramTable.load(p)
