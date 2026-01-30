"""Tests for §1.7 Negation analyzer."""

from analysis.corpus_descriptives.analyzers.negation import NegationAnalyzer
from .conftest import make_doc


class TestNegation:
    def test_preverbal_negation(self, nlp):
        # "he does not run" — "not" before "run"
        doc = make_doc(
            nlp,
            words=["he", "does", "not", "run"],
            pos=["PRON", "AUX", "PART", "VERB"],
            deps=["nsubj", "aux", "advmod", "ROOT"],
            heads=[3, 3, 3, 3],
            morphs=[None, None, "Polarity=Neg", "VerbForm=Fin"],
            lemmas=["he", "do", "not", "run"],
        )
        analyzer = NegationAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["position_counts"]["pre"] == 1
        assert results["overall"]["total_negations"] == 1

    def test_subject_crosstab(self, nlp):
        # "not run" (no subject)
        doc = make_doc(
            nlp,
            words=["not", "run"],
            pos=["PART", "VERB"],
            deps=["advmod", "ROOT"],
            heads=[1, 1],
            morphs=["Polarity=Neg", "VerbForm=Fin"],
            lemmas=["not", "run"],
        )
        analyzer = NegationAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ct = results["overall"]["cross_tab"]
        assert any(
            r["position"] == "pre" and r["subject_status"] == "none"
            for r in ct
        )

    def test_overt_subject_crosstab(self, nlp):
        doc = make_doc(
            nlp,
            words=["he", "not", "runs"],
            pos=["PRON", "PART", "VERB"],
            deps=["nsubj", "advmod", "ROOT"],
            heads=[2, 2, 2],
            morphs=[None, "Polarity=Neg", "VerbForm=Fin"],
            lemmas=["he", "not", "run"],
        )
        analyzer = NegationAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ct = results["overall"]["cross_tab"]
        assert any(
            r["position"] == "pre" and r["subject_status"] == "overt"
            for r in ct
        )

    def test_no_negation(self, nlp):
        doc = make_doc(
            nlp,
            words=["he", "runs"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
        )
        analyzer = NegationAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()
        assert results["overall"]["total_negations"] == 0
