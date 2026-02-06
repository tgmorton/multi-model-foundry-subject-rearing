"""Tests for sequence labeler (mocked model)."""
import pytest
from analysis.pronoun_recovery.constants import ALL_LABELS, NUM_LABELS, LABEL_NONE


class TestSequenceLabelerInterface:
    def test_label_count(self):
        assert NUM_LABELS == 9

    def test_all_labels_contains_none(self):
        assert LABEL_NONE in ALL_LABELS

    def test_all_labels_contains_pro_labels(self):
        for suffix in ["1sg", "2sg", "3sg", "1pl", "2pl", "3pl"]:
            assert f"PRO.{suffix}" in ALL_LABELS

    def test_imp_conj_in_labels(self):
        assert "IMP" in ALL_LABELS
        assert "CONJ" in ALL_LABELS
