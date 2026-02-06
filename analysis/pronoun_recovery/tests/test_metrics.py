"""Tests for validation metrics."""
from analysis.pronoun_recovery.validation.metrics import compute_annotation_metrics


class TestAnnotationMetrics:
    def test_perfect_match(self):
        pred = [{"id": "a", "markers": [{"label": "PRO.1sg", "position": 3}]}]
        gold = [{"id": "a", "markers": [{"label": "PRO.1sg", "position": 3}]}]
        m = compute_annotation_metrics(pred, gold)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.feature_accuracy == 1.0

    def test_no_predictions(self):
        gold = [{"id": "a", "markers": [{"label": "PRO.3sg", "position": 5}]}]
        m = compute_annotation_metrics([], gold)
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_label_mismatch(self):
        pred = [{"id": "a", "markers": [{"label": "PRO.3sg", "position": 4}]}]
        gold = [{"id": "a", "markers": [{"label": "PRO.1sg", "position": 3}]}]
        m = compute_annotation_metrics(pred, gold, position_tolerance=2)
        assert m.correct_insertions == 1  # position matched
        assert m.feature_accuracy == 0.0  # label wrong

    def test_empty_both(self):
        m = compute_annotation_metrics([], [])
        assert m.precision == 1.0
        assert m.recall == 1.0
