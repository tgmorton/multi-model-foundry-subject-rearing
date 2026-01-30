"""Tests for §1.3 Verb Finiteness analyzer."""

from analysis.corpus_descriptives.analyzers.verb_finiteness import (
    VerbFinitenessAnalyzer,
)
from .conftest import make_doc


class TestVerbFiniteness:
    def test_finite_root(self, nlp):
        # "he runs"
        doc = make_doc(
            nlp,
            words=["he", "runs"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            morphs=[None, "VerbForm=Fin"],
        )
        analyzer = VerbFinitenessAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ct = results["overall"]["cross_tab"]
        assert len(ct) == 1
        assert ct[0]["verb_form"] == "Fin"
        assert ct[0]["clause_position"] == "ROOT"
        assert ct[0]["count"] == 1

    def test_infinitival_xcomp(self, nlp):
        # "wants to run"
        doc = make_doc(
            nlp,
            words=["wants", "to", "run"],
            pos=["VERB", "PART", "VERB"],
            deps=["ROOT", "mark", "xcomp"],
            heads=[0, 2, 0],
            morphs=["VerbForm=Fin", None, "VerbForm=Inf"],
        )
        analyzer = VerbFinitenessAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ct = results["overall"]["cross_tab"]
        forms = {(r["verb_form"], r["clause_position"]) for r in ct}
        assert ("Fin", "ROOT") in forms
        assert ("Inf", "xcomp") in forms

    def test_aux_counted(self, nlp):
        # "he has gone"
        doc = make_doc(
            nlp,
            words=["he", "has", "gone"],
            pos=["PRON", "AUX", "VERB"],
            deps=["nsubj", "ROOT", "xcomp"],
            heads=[1, 1, 1],
            morphs=[None, "VerbForm=Fin", "VerbForm=Part"],
        )
        analyzer = VerbFinitenessAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ct = results["overall"]["cross_tab"]
        assert any(r["verb_form"] == "Fin" and r["clause_position"] == "ROOT" for r in ct)

    def test_no_verbform_skipped(self, nlp):
        doc = make_doc(
            nlp,
            words=["dog"],
            pos=["NOUN"],
            deps=["ROOT"],
            heads=[0],
        )
        analyzer = VerbFinitenessAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()
        assert results["overall"]["cross_tab"] == []
