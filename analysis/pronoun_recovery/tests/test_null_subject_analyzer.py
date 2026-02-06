"""Tests for null subject analyzer."""
from analysis.pronoun_recovery.descriptive.context_classifier import classify_null_subject_context
from analysis.pronoun_recovery.descriptive.null_subject_analyzer import NullSubjectAnalyzer
from .conftest import make_doc


class TestContextClassifier:
    def test_imperative(self, nlp):
        doc = make_doc(nlp, words=["Run", "!"], pos=["VERB", "PUNCT"],
                       deps=["ROOT", "punct"], heads=[0, 0])
        result = classify_null_subject_context("IMP", doc[0], doc)
        assert result == "imperative"

    def test_conj_is_topic_continuity(self, nlp):
        doc = make_doc(nlp, words=["runs", "."], pos=["VERB", "PUNCT"],
                       deps=["ROOT", "punct"], heads=[0, 0])
        result = classify_null_subject_context("CONJ", doc[0], doc)
        assert result == "topic_continuity"

    def test_diary_drop(self, nlp):
        doc = make_doc(nlp, words=["Ran", "fast", "."],
                       pos=["VERB", "ADV", "PUNCT"],
                       deps=["ROOT", "advmod", "punct"],
                       heads=[0, 0, 0])
        result = classify_null_subject_context("PRO.1sg", doc[0], doc)
        assert result == "diary_drop"

    def test_other_fallback(self, nlp):
        doc = make_doc(nlp, words=["then", "runs", "."],
                       pos=["ADV", "VERB", "PUNCT"],
                       deps=["advmod", "ROOT", "punct"],
                       heads=[1, 1, 1])
        result = classify_null_subject_context("PRO.3sg", doc[1], doc)
        assert result == "other"


class TestNullSubjectAnalyzer:
    def test_overt_subject_counting(self, nlp):
        """Analyzer should count overt subject pronouns even without model."""
        doc = make_doc(
            nlp, words=["She", "runs", "."],
            pos=["PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "punct"],
            heads=[1, 1, 1],
            morphs=[{"Person": "3", "Number": "Sing"}, {"VerbForm": "Fin"}, None],
        )
        analyzer = NullSubjectAnalyzer(model=None)
        analyzer.process_doc(doc, genre="test")
        results = analyzer.get_results()
        assert results["overall"]["overt_subject_count"] == 1
        assert results["overall"]["null_subject_count"] == 0

    def test_null_rate_zero_without_model(self, nlp):
        doc = make_doc(
            nlp, words=["She", "runs", "."],
            pos=["PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "punct"],
            heads=[1, 1, 1],
            morphs=[{"Person": "3", "Number": "Sing"}, {"VerbForm": "Fin"}, None],
        )
        analyzer = NullSubjectAnalyzer(model=None)
        analyzer.process_doc(doc, genre="test")
        results = analyzer.get_results()
        assert results["overall"]["null_rate"] == 0.0

    def test_results_structure(self, nlp):
        analyzer = NullSubjectAnalyzer(model=None)
        doc = make_doc(
            nlp, words=["She", "runs", "."],
            pos=["PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "punct"],
            heads=[1, 1, 1],
            morphs=[{"Person": "3", "Number": "Sing"}, {"VerbForm": "Fin"}, None],
        )
        analyzer.process_doc(doc, genre="TestGenre")
        results = analyzer.get_results()
        assert "overall" in results
        assert "by_genre" in results
        assert "TestGenre" in results["by_genre"]
