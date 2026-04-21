"""D5 test: per-token log-probs output (MORCELA-readiness).

Asserts that SurprisalCalculator.calculate_surprisal returns a per-target-token
log-probability list whose length equals the target token count, whose sum
equals total_log_prob, and which survives a round-trip through JSON so that
downstream parquet / cache layers can serialize it.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from evaluation.core.surprisal_calculator import (
    NullSubjectSurprisalCalculator,
    SurprisalCalculator,
)


def _make_tok():
    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = 2
    tok.pad_id.return_value = 0
    tok.encode.side_effect = lambda text: [10 + (ord(w[0]) % 50) for w in text.split()]
    tok.decode.side_effect = lambda ids: " ".join(f"tok{i}" for i in ids)
    return tok


def _make_model(vocab_size=100):
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])

    def forward(input_ids=None, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, vocab_size)
        for pos in range(seq_len):
            for v in range(vocab_size):
                logits[:, pos, v] = -0.01 * (v + pos)
        out = MagicMock()
        out.logits = logits
        return out

    model.side_effect = forward
    model.__call__ = forward
    return model


def test_per_token_log_prob_length_and_sum():
    calc = SurprisalCalculator(_make_model(), _make_tok(), device="cpu")
    result = calc.calculate_surprisal("alpha beta gamma delta", context="ctx")

    per_tok = result["per_token_log_prob"]
    n = result["tokens"]

    assert len(per_tok) == n, "per_token length must equal target token count"
    assert sum(per_tok) == pytest.approx(result["total_log_prob"], abs=1e-6)
    assert np.mean(per_tok) == pytest.approx(result["mean_log_prob"], abs=1e-6)


def test_per_token_log_prob_all_non_positive():
    """Log-probs are log of probabilities, so ≤ 0."""
    calc = SurprisalCalculator(_make_model(), _make_tok(), device="cpu")
    result = calc.calculate_surprisal("alpha beta gamma", context="ctx")
    for lp in result["per_token_log_prob"]:
        assert lp <= 0


def test_per_token_log_prob_json_serializable():
    """Downstream parquet / cache layers will serialize these — must round-trip."""
    calc = SurprisalCalculator(_make_model(), _make_tok(), device="cpu")
    result = calc.calculate_surprisal("alpha beta", context="ctx")
    payload = {
        "per_token_log_prob": result["per_token_log_prob"],
        "per_token_ids": result["per_token_ids"],
    }
    blob = json.dumps(payload)
    round_trip = json.loads(blob)
    assert round_trip["per_token_log_prob"] == pytest.approx(
        payload["per_token_log_prob"], abs=1e-12
    )
    assert round_trip["per_token_ids"] == payload["per_token_ids"]


def test_null_subject_pair_per_token_fields():
    """Pair evaluation carries through both overt and null per-token vectors."""
    calc = NullSubjectSurprisalCalculator(_make_model(), _make_tok(), device="cpu")
    result = calc.evaluate_null_subject_pair(
        context="the meeting started",
        overt_target="she arrived late",
        null_target="arrived late",
        hotspot="arrived",
    )
    assert len(result["overt_per_token_log_prob"]) == result["overt_n_target_tokens"]
    assert len(result["null_per_token_log_prob"]) == result["null_n_target_tokens"]
    # Null target is one word shorter than overt (pronoun dropped)
    assert result["overt_n_target_tokens"] >= result["null_n_target_tokens"]
    # Sum matches the aggregate total_log_prob for each side
    assert sum(result["overt_per_token_log_prob"]) == pytest.approx(
        result["overt_total_log_prob"], abs=1e-6
    )
    assert sum(result["null_per_token_log_prob"]) == pytest.approx(
        result["null_total_log_prob"], abs=1e-6
    )
