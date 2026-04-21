"""D4 test: SLOR math in NullSubjectEvaluator.

SLOR(s) = (Σ log P_M(w) - Σ log P_u(w)) / |s|     (natural log)

Asserts:
- When a unigram table is supplied, the evaluator emits overt_slor,
  null_slor, slor_diff_overt_minus_null, prefers_overt_slor.
- The math matches hand-computed values on controlled inputs.
- When unigram is None, SLOR fields are omitted (backwards compat).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from evaluation.core.surprisal_calculator import NullSubjectSurprisalCalculator
from evaluation.evaluators.null_subject_evaluator import NullSubjectEvaluator
from evaluation.unigram import UnigramTable


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


def _uniform_unigram(vocab_size: int) -> UnigramTable:
    """Unigram where every token has count 1 (pure uniform after smoothing)."""
    counts = np.ones(vocab_size, dtype=np.int64)
    return UnigramTable(
        counts=counts, vocab_size=vocab_size, total_tokens=vocab_size,
        smoothing_alpha=1.0, corpus_id="uniform_test", tokenizer_id="tok_test",
    )


def _stub_group(overt_target="she shows pride", null_target="shows pride",
                context="marta won the award"):
    """Build a DataFrame group mimicking a single v2 pair."""
    import pandas as pd
    return pd.DataFrame([
        {"item": 1, "item_group": "subject_drop", "condition": "subj_3sg",
         "language": "en", "schema_version": "v2",
         "pronoun_status": 1, "c_english": context, "target": overt_target,
         "hotspot_english": "shows", "hotspot_position": 1},
        {"item": 1, "item_group": "subject_drop", "condition": "subj_3sg",
         "language": "en", "schema_version": "v2",
         "pronoun_status": 0, "c_english": context, "target": null_target,
         "hotspot_english": "shows", "hotspot_position": 0},
    ])


def test_slor_fields_present_when_unigram_supplied():
    calc = NullSubjectSurprisalCalculator(_make_model(), _make_tok(), device="cpu")
    evaluator = NullSubjectEvaluator(calc, unigram=_uniform_unigram(100))

    result = evaluator.process_item_group(_stub_group())

    for field in (
        "overt_slor",
        "null_slor",
        "overt_unigram_sum_log_prob",
        "null_unigram_sum_log_prob",
        "slor_diff_overt_minus_null",
        "prefers_overt_slor",
        "unigram_corpus_id",
        "unigram_tokenizer_id",
    ):
        assert field in result, f"missing SLOR field: {field}"

    assert result["unigram_corpus_id"] == "uniform_test"


def test_slor_fields_absent_when_no_unigram():
    """Backwards compat: no unigram → no SLOR fields."""
    calc = NullSubjectSurprisalCalculator(_make_model(), _make_tok(), device="cpu")
    evaluator = NullSubjectEvaluator(calc)  # no unigram

    result = evaluator.process_item_group(_stub_group())
    for field in ("overt_slor", "null_slor", "slor_diff_overt_minus_null"):
        assert field not in result, f"unexpected SLOR field {field} present"


def test_slor_identity():
    """SLOR equals (model_sum - unigram_sum) / n_tokens for each side."""
    calc = NullSubjectSurprisalCalculator(_make_model(), _make_tok(), device="cpu")
    unigram = _uniform_unigram(100)
    evaluator = NullSubjectEvaluator(calc, unigram=unigram)

    result = evaluator.process_item_group(_stub_group())

    overt_n = result["overt_n_target_tokens"]
    null_n = result["null_n_target_tokens"]
    expected_overt_slor = (
        result["overt_total_log_prob"] - result["overt_unigram_sum_log_prob"]
    ) / overt_n
    expected_null_slor = (
        result["null_total_log_prob"] - result["null_unigram_sum_log_prob"]
    ) / null_n
    assert result["overt_slor"] == pytest.approx(expected_overt_slor, abs=1e-9)
    assert result["null_slor"] == pytest.approx(expected_null_slor, abs=1e-9)
    assert result["slor_diff_overt_minus_null"] == pytest.approx(
        expected_overt_slor - expected_null_slor, abs=1e-9
    )


def test_slor_uniform_unigram_matches_meanlp_minus_log_inv_V():
    """Sanity check: under a uniform unigram where every token has
    log_prob = log(1/V), SLOR = MeanLP - log(1/V)."""
    V = 100
    calc = NullSubjectSurprisalCalculator(_make_model(V), _make_tok(), device="cpu")
    unigram = _uniform_unigram(V)
    evaluator = NullSubjectEvaluator(calc, unigram=unigram)

    # Smoothing α=1, counts=1 each, N=V → P(t) = (1+1)/(V+V*1) = 1/V exactly
    expected_unigram_lp = float(np.log(1.0 / V))

    result = evaluator.process_item_group(_stub_group())
    # Overt side
    overt_expected = result["overt_mean_log_prob"] - expected_unigram_lp
    assert result["overt_slor"] == pytest.approx(overt_expected, abs=1e-6)
    # Null side
    null_expected = result["null_mean_log_prob"] - expected_unigram_lp
    assert result["null_slor"] == pytest.approx(null_expected, abs=1e-6)
