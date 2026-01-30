"""Tests for §1.6 Relative Clauses analyzer."""

from analysis.corpus_descriptives.analyzers.relative_clauses import (
    RelativeClauseAnalyzer,
)
from .conftest import make_doc


class TestRelativeClauses:
    def test_subject_relative(self, nlp):
        # "the man who runs" — "runs" has dep=acl:relcl, "who" is nsubj of "runs"
        doc = make_doc(
            nlp,
            words=["the", "man", "who", "runs"],
            pos=["DET", "NOUN", "PRON", "VERB"],
            deps=["det", "ROOT", "nsubj", "acl:relcl"],
            heads=[1, 1, 3, 1],
            morphs=[None, None, "PronType=Rel", "VerbForm=Fin"],
            lemmas=["the", "man", "who", "run"],
        )
        analyzer = RelativeClauseAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["counts"]["subject"] == 1

    def test_object_relative(self, nlp):
        # "the book that I read" — "read" has dep=acl:relcl, "I" is nsubj
        doc = make_doc(
            nlp,
            words=["the", "book", "that", "I", "read"],
            pos=["DET", "NOUN", "PRON", "PRON", "VERB"],
            deps=["det", "ROOT", "obj", "nsubj", "acl:relcl"],
            heads=[1, 1, 4, 4, 1],
            morphs=[None, None, None, None, "VerbForm=Fin"],
            lemmas=["the", "book", "that", "I", "read"],
        )
        analyzer = RelativeClauseAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["counts"]["object"] == 1

    def test_acl_variant(self, nlp):
        # Some spaCy versions use "acl" instead of "acl:relcl"
        doc = make_doc(
            nlp,
            words=["man", "who", "runs"],
            pos=["NOUN", "PRON", "VERB"],
            deps=["ROOT", "nsubj", "acl"],
            heads=[0, 2, 0],
            morphs=[None, "PronType=Rel", "VerbForm=Fin"],
            lemmas=["man", "who", "run"],
        )
        analyzer = RelativeClauseAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()
        assert results["overall"]["total"] == 1
