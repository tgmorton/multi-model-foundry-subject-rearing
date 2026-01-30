"""Tests for §1.2 Expletive analyzer."""

from analysis.corpus_descriptives.analyzers.expletives import ExpletiveAnalyzer
from .conftest import make_doc


class TestExpletives:
    def test_weather_expletive(self, nlp):
        # "it rains"
        doc = make_doc(
            nlp,
            words=["it", "rains"],
            pos=["PRON", "VERB"],
            deps=["expl", "ROOT"],
            heads=[1, 1],
            lemmas=["it", "rain"],
        )
        analyzer = ExpletiveAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["class_counts"]["weather"] == 1

    def test_existential_there(self, nlp):
        # "there is a cat"
        doc = make_doc(
            nlp,
            words=["there", "is", "a", "cat"],
            pos=["PRON", "VERB", "DET", "NOUN"],
            deps=["expl", "ROOT", "det", "nsubj"],
            heads=[1, 1, 3, 1],
            lemmas=["there", "be", "a", "cat"],
        )
        analyzer = ExpletiveAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["class_counts"]["existential"] == 1

    def test_raising_expletive(self, nlp):
        # "it seems that..."
        doc = make_doc(
            nlp,
            words=["it", "seems"],
            pos=["PRON", "VERB"],
            deps=["expl", "ROOT"],
            heads=[1, 1],
            lemmas=["it", "seem"],
        )
        analyzer = ExpletiveAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["class_counts"]["raising"] == 1

    def test_verb_lemma_tracking(self, nlp):
        doc = make_doc(
            nlp,
            words=["it", "rains"],
            pos=["PRON", "VERB"],
            deps=["expl", "ROOT"],
            heads=[1, 1],
            lemmas=["it", "rain"],
        )
        analyzer = ExpletiveAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        verb_counts = results["overall"]["verb_counts"]
        assert len(verb_counts) == 1
        assert verb_counts[0]["class"] == "weather"
        assert verb_counts[0]["verb_lemma"] == "rain"
