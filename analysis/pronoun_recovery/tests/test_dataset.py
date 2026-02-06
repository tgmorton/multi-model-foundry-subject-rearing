"""Tests for HuggingFace dataset construction."""
import pytest
from analysis.pronoun_recovery.constants import ALL_LABELS, LABEL_TO_ID, LABEL_NONE


class TestDatasetFormat:
    def test_label_set_size(self):
        assert len(ALL_LABELS) == 9

    def test_none_is_first_label(self):
        assert ALL_LABELS[0] == LABEL_NONE

    def test_label_to_id_bijection(self):
        assert len(LABEL_TO_ID) == 9
        for label, idx in LABEL_TO_ID.items():
            assert ALL_LABELS[idx] == label

    def test_synthetic_pair_to_word_labels(self):
        """Verify synthetic pair conversion produces correct word-level labels."""
        from analysis.pronoun_recovery.model.dataset import _synthetic_to_word_labels
        record = {
            "dropped": "Runs fast .",
            "verb_labels": [
                {"verb_idx": 0, "verb_text": "Runs", "label": "PRO.3sg", "original_pronoun": "She"}
            ],
        }
        result = _synthetic_to_word_labels(record)
        assert result is not None
        assert result["words"] == ["Runs", "fast", "."]
        assert result["labels"][0] == "PRO.3sg"
        assert result["labels"][1] == LABEL_NONE
        assert result["labels"][2] == LABEL_NONE
