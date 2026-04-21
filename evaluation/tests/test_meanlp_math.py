"""D2 test: MeanLP output math.

Asserts that the SurprisalCalculator returns natural-log log-prob values
consistent with the surprisal values it already returns:
    log_prob = -surprisal * ln(2)
and that totals are sum / mean of the per-token values.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from evaluation.core.surprisal_calculator import (
    NullSubjectSurprisalCalculator,
    SurprisalCalculator,
)


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = 2
    tok.pad_id.return_value = 0
    # Encode maps words to a deterministic per-word id >=10
    def encode(text):
        return [10 + (ord(w[0]) % 50) for w in text.split()]
    tok.encode.side_effect = encode
    tok.decode.side_effect = lambda ids: " ".join(f"tok{i}" for i in ids)
    return tok


def _make_mock_model(vocab_size=100):
    """Model that returns deterministic, non-uniform logits."""
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])

    # Produce logits where P(token_id=k) decreases with k — predictable
    def forward(input_ids=None, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, vocab_size)
        # Add a small per-position bias so surprisals differ across positions
        for pos in range(seq_len):
            for v in range(vocab_size):
                logits[:, pos, v] = -0.01 * (v + pos)
        out = MagicMock()
        out.logits = logits
        return out

    model.side_effect = forward
    model.__call__ = forward
    return model


def test_mean_log_prob_equals_neg_surprisal_times_ln2():
    tok = _make_mock_tokenizer()
    model = _make_mock_model()
    calc = SurprisalCalculator(model, tok, device="cpu")

    result = calc.calculate_surprisal("the quick brown fox", context="a sentence")

    # mean_log_prob and total_log_prob must satisfy the identity
    ln2 = float(np.log(2))
    assert result["mean_log_prob"] == pytest.approx(
        -result["mean_surprisal"] * ln2, abs=1e-9
    )
    assert result["total_log_prob"] == pytest.approx(
        -result["total_surprisal"] * ln2, abs=1e-9
    )
    # log-probs are non-positive (log of probability ≤ 1)
    assert result["mean_log_prob"] <= 0
    assert result["total_log_prob"] <= 0


def test_per_token_log_prob_sum_equals_total():
    """D5: sum of per-token log-probs equals total_log_prob."""
    tok = _make_mock_tokenizer()
    model = _make_mock_model()
    calc = SurprisalCalculator(model, tok, device="cpu")

    result = calc.calculate_surprisal("one two three", context="context here")

    per_tok = result["per_token_log_prob"]
    assert isinstance(per_tok, list)
    assert len(per_tok) == result["tokens"]
    assert sum(per_tok) == pytest.approx(result["total_log_prob"], abs=1e-6)


def test_per_token_ids_match_token_count():
    """D5: per_token_ids has same length as per_token_log_prob."""
    tok = _make_mock_tokenizer()
    model = _make_mock_model()
    calc = SurprisalCalculator(model, tok, device="cpu")

    result = calc.calculate_surprisal("alpha beta", context="something")

    assert len(result["per_token_ids"]) == len(result["per_token_log_prob"])
    assert len(result["per_token_strings"]) == len(result["per_token_log_prob"])


def test_null_subject_pair_meanlp_fields_present():
    """D2: evaluate_null_subject_pair returns MeanLP + preference under MeanLP."""
    tok = _make_mock_tokenizer()
    model = _make_mock_model()
    calc = NullSubjectSurprisalCalculator(model, tok, device="cpu")

    result = calc.evaluate_null_subject_pair(
        context="marta won the award",
        overt_target="she shows pride",
        null_target="shows pride",
        hotspot="shows",
    )

    for field in (
        "overt_mean_log_prob",
        "null_mean_log_prob",
        "overt_total_log_prob",
        "null_total_log_prob",
        "log_prob_diff_overt_minus_null",
        "prefers_overt_meanlp",
        "overt_per_token_log_prob",
        "null_per_token_log_prob",
        "overt_per_token_ids",
        "null_per_token_ids",
        "overt_n_target_tokens",
        "null_n_target_tokens",
    ):
        assert field in result, f"missing {field}"

    # The two preference metrics agree by construction
    assert result["prefers_overt_meanlp"] == result["prefers_overt"]


def test_hotspot_log_prob_matches_surprisal_identity():
    """D2: hotspot log-prob is consistent with hotspot surprisal."""
    tok = _make_mock_tokenizer()
    model = _make_mock_model()
    calc = NullSubjectSurprisalCalculator(model, tok, device="cpu")

    result = calc.evaluate_null_subject_pair(
        context="marta won the award",
        overt_target="she shows pride",
        null_target="shows pride",
        hotspot="shows",
    )
    if "overt_hotspot_surprisal" in result and "overt_hotspot_log_prob" in result:
        ln2 = float(np.log(2))
        assert result["overt_hotspot_log_prob"] == pytest.approx(
            -result["overt_hotspot_surprisal"] * ln2, abs=1e-9
        )
