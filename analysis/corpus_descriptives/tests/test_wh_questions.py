"""Tests for §1.5 Wh-Questions analyzer."""

from analysis.corpus_descriptives.analyzers.wh_questions import WhQuestionAnalyzer
from .conftest import make_doc


class TestWhQuestions:
    def test_subject_wh_root(self, nlp):
        # "who runs"
        doc = make_doc(
            nlp,
            words=["who", "runs"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            morphs=["PronType=Int", "VerbForm=Fin"],
            lemmas=["who", "run"],
        )
        analyzer = WhQuestionAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ec = results["overall"]["extraction_counts"]
        assert len(ec) == 1
        assert ec[0]["extraction_type"] == "subject"
        assert ec[0]["clause_level"] == "root"

    def test_object_wh_root(self, nlp):
        # "what did he see"
        doc = make_doc(
            nlp,
            words=["what", "did", "he", "see"],
            pos=["PRON", "AUX", "PRON", "VERB"],
            deps=["obj", "aux", "nsubj", "ROOT"],
            heads=[3, 3, 3, 3],
            morphs=["PronType=Int", None, None, None],
            lemmas=["what", "do", "he", "see"],
        )
        analyzer = WhQuestionAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ec = results["overall"]["extraction_counts"]
        assert len(ec) == 1
        assert ec[0]["extraction_type"] == "object"
        assert ec[0]["clause_level"] == "root"

    def test_adjunct_wh(self, nlp):
        # "where does he go"
        doc = make_doc(
            nlp,
            words=["where", "does", "he", "go"],
            pos=["ADV", "AUX", "PRON", "VERB"],
            deps=["advmod", "aux", "nsubj", "ROOT"],
            heads=[3, 3, 3, 3],
            morphs=["PronType=Int", None, None, None],
            lemmas=["where", "do", "he", "go"],
        )
        analyzer = WhQuestionAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ec = results["overall"]["extraction_counts"]
        assert ec[0]["extraction_type"] == "adjunct"

    def test_wh_detected_by_lemma(self, nlp):
        # "how" without PronType morph but lemma in set
        doc = make_doc(
            nlp,
            words=["how", "is", "it"],
            pos=["ADV", "AUX", "PRON"],
            deps=["advmod", "ROOT", "nsubj"],
            heads=[1, 1, 1],
            lemmas=["how", "be", "it"],
        )
        analyzer = WhQuestionAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()
        assert len(results["overall"]["extraction_counts"]) == 1
