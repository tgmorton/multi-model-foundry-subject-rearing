"""
Tests for the tree-based null subject detector.

Uses blank spaCy models with manually set attributes (no model download).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spacy
from spacy.tokens import Doc

from analysis.pronoun_recovery.tree_detector.evaluator import (
    check_quality_gates,
    evaluate_predictions,
)
from analysis.pronoun_recovery.tree_detector.feature_extractor import (
    FEATURE_NAMES,
    ItalianVerbFeatureExtractor,
    VerbFeatureRow,
)
from analysis.pronoun_recovery.tree_detector.label_aligner import (
    _match_marker_to_verb,
    load_aligned_data,
    save_aligned_data,
)
from analysis.pronoun_recovery.tree_detector.trainer import (
    _binarise_labels,
    _identify_categorical_columns,
    prepare_data,
)

from .conftest import make_doc


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def nlp_it():
    return spacy.blank("it")


def _make_it_finite_verb_doc(nlp_it, words, pos, deps, heads, morphs, lemmas=None):
    """Helper to build Italian doc with finite verb morphology."""
    doc = make_doc(nlp_it, words, pos=pos, deps=deps, heads=heads, morphs=morphs, lemmas=lemmas)
    return doc


@pytest.fixture
def it_null_subject_doc(nlp_it):
    """'Vogliamo approvare' — finite verb with no subject (null subject, 1pl)."""
    return _make_it_finite_verb_doc(
        nlp_it,
        words=["Vogliamo", "approvare", "i", "regolamenti", "."],
        pos=["VERB", "VERB", "DET", "NOUN", "PUNCT"],
        deps=["ROOT", "xcomp", "det", "obj", "punct"],
        heads=[0, 0, 3, 1, 0],
        morphs=[
            {"VerbForm": "Fin", "Person": "1", "Number": "Plur", "Mood": "Ind", "Tense": "Pres"},
            {"VerbForm": "Inf"},
            None,
            None,
            None,
        ],
        lemmas=["volere", "approvare", "il", "regolamento", "."],
    )


@pytest.fixture
def it_overt_subject_doc(nlp_it):
    """'Noi vogliamo' — finite verb with overt subject."""
    return _make_it_finite_verb_doc(
        nlp_it,
        words=["Noi", "vogliamo", "questo", "."],
        pos=["PRON", "VERB", "PRON", "PUNCT"],
        deps=["nsubj", "ROOT", "obj", "punct"],
        heads=[1, 1, 1, 1],
        morphs=[
            {"Person": "1", "Number": "Plur", "PronType": "Prs"},
            {"VerbForm": "Fin", "Person": "1", "Number": "Plur", "Mood": "Ind", "Tense": "Pres"},
            None,
            None,
        ],
        lemmas=["noi", "volere", "questo", "."],
    )


@pytest.fixture
def it_weather_verb_doc(nlp_it):
    """'Piove forte' — weather verb (expletive context, not referential)."""
    return _make_it_finite_verb_doc(
        nlp_it,
        words=["Piove", "forte", "."],
        pos=["VERB", "ADV", "PUNCT"],
        deps=["ROOT", "advmod", "punct"],
        heads=[0, 0, 0],
        morphs=[
            {"VerbForm": "Fin", "Person": "3", "Number": "Sing", "Mood": "Ind", "Tense": "Pres"},
            None,
            None,
        ],
        lemmas=["piovere", "forte", "."],
    )


@pytest.fixture
def it_multi_clause_doc(nlp_it):
    """'Penso che abbiano ragione' — bridge verb + complement."""
    return _make_it_finite_verb_doc(
        nlp_it,
        words=["Penso", "che", "abbiano", "ragione", "."],
        pos=["VERB", "SCONJ", "VERB", "NOUN", "PUNCT"],
        deps=["ROOT", "mark", "ccomp", "obj", "punct"],
        heads=[0, 2, 0, 2, 0],
        morphs=[
            {"VerbForm": "Fin", "Person": "1", "Number": "Sing", "Mood": "Ind", "Tense": "Pres"},
            None,
            {"VerbForm": "Fin", "Person": "3", "Number": "Plur", "Mood": "Sub", "Tense": "Pres"},
            None,
            None,
        ],
        lemmas=["pensare", "che", "avere", "ragione", "."],
    )


@pytest.fixture
def extractor():
    """Feature extractor instance (uses get_default_annotators)."""
    return ItalianVerbFeatureExtractor(language="it")


# ── Feature Extractor Tests ──────────────────────────────────────────


class TestFeatureExtractor:
    """Tests for ItalianVerbFeatureExtractor."""

    def test_extracts_finite_verbs_only(self, nlp_it, extractor):
        """Should only extract VerbForm=Fin tokens."""
        doc = _make_it_finite_verb_doc(
            nlp_it,
            words=["Vogliamo", "approvare", "."],
            pos=["VERB", "VERB", "PUNCT"],
            deps=["ROOT", "xcomp", "punct"],
            heads=[0, 0, 0],
            morphs=[
                {"VerbForm": "Fin", "Person": "1", "Number": "Plur"},
                {"VerbForm": "Inf"},
                None,
            ],
            lemmas=["volere", "approvare", "."],
        )
        sent = list(doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) == 1
        assert rows[0].verb_text == "Vogliamo"

    def test_morph_label_suffix(self, it_null_subject_doc, extractor):
        """Should extract correct person/number suffix."""
        sent = list(it_null_subject_doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) >= 1
        assert rows[0].morph_label_suffix == "1pl"

    def test_weather_verb_feature(self, it_weather_verb_doc, extractor):
        """Weather verbs should have is_weather_verb=True."""
        sent = list(it_weather_verb_doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) == 1
        assert rows[0].features["is_weather_verb"] is True

    def test_overt_subject_feature(self, it_overt_subject_doc, extractor):
        """Verbs with overt nsubj should have has_overt_nsubj=True."""
        sent = list(it_overt_subject_doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) == 1
        assert rows[0].features["has_overt_nsubj"] is True

    def test_null_subject_features(self, it_null_subject_doc, extractor):
        """Null subject verbs should have has_overt_nsubj=False."""
        sent = list(it_null_subject_doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) >= 1
        assert rows[0].features["has_overt_nsubj"] is False

    def test_multi_clause_extracts_multiple(self, it_multi_clause_doc, extractor):
        """Multi-clause sentence should extract multiple finite verbs."""
        sent = list(it_multi_clause_doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) == 2

    def test_feature_names_complete(self, it_null_subject_doc, extractor):
        """All FEATURE_NAMES should be present in output."""
        sent = list(it_null_subject_doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) >= 1
        for name in FEATURE_NAMES:
            assert name in rows[0].features, f"Missing feature: {name}"

    def test_extract_doc(self, it_null_subject_doc, extractor):
        """extract_doc should process all sentences."""
        rows = extractor.extract_doc(it_null_subject_doc)
        assert len(rows) >= 1

    def test_verb_identity_features(self, nlp_it, extractor):
        """is_essere, is_avere, is_modal should be set correctly."""
        doc = _make_it_finite_verb_doc(
            nlp_it,
            words=["È", "importante", "."],
            pos=["AUX", "ADJ", "PUNCT"],
            deps=["ROOT", "acomp", "punct"],
            heads=[0, 0, 0],
            morphs=[
                {"VerbForm": "Fin", "Person": "3", "Number": "Sing"},
                None,
                None,
            ],
            lemmas=["essere", "importante", "."],
        )
        sent = list(doc.sents)[0]
        rows = extractor.extract_sentence(sent)
        assert len(rows) == 1
        assert rows[0].features["is_essere"] is True
        assert rows[0].features["is_avere"] is False
        assert rows[0].features["is_modal"] is False


# ── Label Aligner Tests ──────────────────────────────────────────────


class TestLabelAligner:
    """Tests for marker-to-verb alignment."""

    def test_match_marker_exact_offset(self):
        """Marker at exact verb char_offset should match."""
        row = VerbFeatureRow(
            token_idx=0, char_offset=0, verb_text="Vogliamo",
            verb_lemma="volere", person="1", number="Plur",
            morph_label_suffix="1pl",
        )
        marker = {"position": 0, "it_verb": "Vogliamo", "label": "PRO.1pl"}
        result = _match_marker_to_verb(marker, [row])
        assert result is not None
        assert result.token_idx == 0

    def test_match_marker_close_offset(self):
        """Marker within tolerance should still match."""
        row = VerbFeatureRow(
            token_idx=0, char_offset=2, verb_text="Vogliamo",
            verb_lemma="volere", person="1", number="Plur",
            morph_label_suffix="1pl",
        )
        marker = {"position": 0, "it_verb": "Vogliamo", "label": "PRO.1pl"}
        result = _match_marker_to_verb(marker, [row])
        assert result is not None

    def test_no_match_beyond_tolerance(self):
        """Marker too far from any verb should return None."""
        row = VerbFeatureRow(
            token_idx=5, char_offset=50, verb_text="ha",
            verb_lemma="avere", person="3", number="Sing",
            morph_label_suffix="3sg",
        )
        marker = {"position": 0, "it_verb": "Vogliamo", "label": "PRO.1pl"}
        result = _match_marker_to_verb(marker, [row])
        assert result is None

    def test_save_and_load_roundtrip(self):
        """Save and load should preserve data."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        y = np.array(["PRO.1pl", "NONE", "PRO.3sg"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_aligned_data(X, y, Path(tmpdir))
            X_loaded, y_loaded = load_aligned_data(Path(tmpdir))

        assert len(X_loaded) == 3
        np.testing.assert_array_equal(y_loaded, y)


# ── Trainer Tests ────────────────────────────────────────────────────


class TestTrainer:
    """Tests for training utilities."""

    def test_binarise_labels(self):
        """PRO.* -> 1, NONE -> 0."""
        y = np.array(["PRO.1pl", "NONE", "PRO.3sg", "NONE", "PRO.1sg"])
        y_bin = _binarise_labels(y)
        np.testing.assert_array_equal(y_bin, [1, 0, 1, 0, 1])

    def test_identify_categoricals(self):
        """Should find object and bool columns."""
        X = pd.DataFrame({
            "str_col": ["a", "b"],
            "int_col": [1, 2],
            "bool_col": [True, False],
            "float_col": [1.0, 2.0],
        })
        cats = _identify_categorical_columns(X)
        assert "str_col" in cats
        assert "bool_col" in cats
        assert "int_col" not in cats
        assert "float_col" not in cats

    def test_prepare_data_shapes(self):
        """Prepare data should split correctly."""
        n = 100
        X = pd.DataFrame({
            "f1": np.random.rand(n),
            "f2": np.random.choice(["a", "b", "c"], n),
        })
        y = np.array(["PRO.1pl"] * 30 + ["NONE"] * 70)

        X_train, X_test, y_train, y_test, encoder = prepare_data(
            X, y, test_fraction=0.2, seed=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert y_train.sum() + y_test.sum() == 30  # all positives accounted for


# ── Evaluator Tests ──────────────────────────────────────────────────


class TestEvaluator:
    """Tests for evaluation metrics."""

    def test_perfect_detection(self):
        """Perfect predictions should give F1=1.0."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_orig = np.array(["PRO.1pl", "NONE", "PRO.3sg", "NONE", "PRO.1sg"])

        metrics = evaluate_predictions(y_true, y_pred, y_orig, "test")
        assert metrics["detection_f1"] == 1.0
        assert metrics["feature_accuracy"] == 1.0

    def test_all_false_negatives(self):
        """Predicting all NONE should give F1=0."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        y_orig = np.array(["PRO.1pl", "PRO.3sg", "PRO.1sg"])

        metrics = evaluate_predictions(y_true, y_pred, y_orig, "test")
        assert metrics["detection_f1"] == 0.0

    def test_quality_gates_pass(self):
        """Gates should pass when metrics exceed thresholds."""
        metrics = {"detection_f1": 0.90, "feature_accuracy": 0.99}
        gates = check_quality_gates(metrics, min_detection_f1=0.80, min_feature_accuracy=0.95)
        assert gates["all_pass"] is True

    def test_quality_gates_fail(self):
        """Gates should fail when detection_f1 is below threshold."""
        metrics = {"detection_f1": 0.60, "feature_accuracy": 0.99}
        gates = check_quality_gates(metrics, min_detection_f1=0.80, min_feature_accuracy=0.95)
        assert gates["all_pass"] is False
        assert gates["detection_f1"]["pass"] is False

    def test_per_label_metrics(self):
        """Should report per-label recall."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        y_orig = np.array(["PRO.1pl", "NONE", "PRO.3sg", "NONE", "PRO.1sg"])

        metrics = evaluate_predictions(y_true, y_pred, y_orig, "test")
        assert "PRO.1pl" in metrics["per_label"]
        assert metrics["per_label"]["PRO.1pl"]["recall"] == 1.0
        assert metrics["per_label"]["PRO.3sg"]["recall"] == 0.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2x2."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_orig = np.array(["PRO.1pl", "NONE", "PRO.3sg", "NONE"])

        metrics = evaluate_predictions(y_true, y_pred, y_orig, "test")
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


# ── Config Tests ─────────────────────────────────────────────────────


class TestTreeDetectorConfig:
    """Tests for TreeDetectorConfig."""

    def test_config_loads(self):
        """TreeDetectorConfig should instantiate with required fields."""
        from analysis.pronoun_recovery.config import TreeDetectorConfig

        cfg = TreeDetectorConfig(
            aligned_data_path="/tmp/test.jsonl",
        )
        assert cfg.language == "it"
        assert cfg.classifier_type == "decision_tree"
        assert cfg.cv_folds == 5
        assert cfg.min_detection_f1 == 0.80

    def test_config_from_yaml(self, tmp_path):
        """Should load from YAML file."""
        import yaml

        from analysis.pronoun_recovery.config import TreeDetectorConfig, load_config

        yaml_content = {
            "aligned_data_path": "/tmp/test.jsonl",
            "output_path": str(tmp_path / "output"),
            "classifier_type": "gradient_boosted",
            "cv_folds": 3,
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        cfg = load_config(str(yaml_path), TreeDetectorConfig)
        assert cfg.classifier_type == "gradient_boosted"
        assert cfg.cv_folds == 3
