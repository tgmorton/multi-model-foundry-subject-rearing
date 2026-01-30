"""Tests for §1.4 Clause Structure analyzer."""

from analysis.corpus_descriptives.analyzers.clause_structure import (
    ClauseStructureAnalyzer,
)
from .conftest import make_doc


class TestClauseStructure:
    def test_overt_subject(self, nlp):
        # "he runs"
        doc = make_doc(
            nlp,
            words=["he", "runs"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            morphs=[None, "VerbForm=Fin"],
        )
        analyzer = ClauseStructureAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        fc = results["overall"]["finite_clauses"]
        assert len(fc) == 1
        assert fc[0]["clause_dep"] == "ROOT"
        assert fc[0]["subject_status"] == "overt"

    def test_no_subject(self, nlp):
        # "runs" (imperative-like)
        doc = make_doc(
            nlp,
            words=["runs"],
            pos=["VERB"],
            deps=["ROOT"],
            heads=[0],
            morphs=["VerbForm=Fin"],
        )
        analyzer = ClauseStructureAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        fc = results["overall"]["finite_clauses"]
        assert len(fc) == 1
        assert fc[0]["subject_status"] == "none"

    def test_expletive_subject(self, nlp):
        # "it rains"
        doc = make_doc(
            nlp,
            words=["it", "rains"],
            pos=["PRON", "VERB"],
            deps=["expl", "ROOT"],
            heads=[1, 1],
            morphs=[None, "VerbForm=Fin"],
        )
        analyzer = ClauseStructureAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        fc = results["overall"]["finite_clauses"]
        assert fc[0]["subject_status"] == "expletive"

    def test_xcomp_matrix_verb(self, nlp):
        # "wants to run"
        doc = make_doc(
            nlp,
            words=["wants", "to", "run"],
            pos=["VERB", "PART", "VERB"],
            deps=["ROOT", "mark", "xcomp"],
            heads=[0, 2, 0],
            morphs=["VerbForm=Fin", None, "VerbForm=Inf"],
            lemmas=["want", "to", "run"],
        )
        analyzer = ClauseStructureAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        xv = results["overall"]["xcomp_matrix_verbs"]
        assert len(xv) == 1
        assert xv[0]["verb_lemma"] == "want"
