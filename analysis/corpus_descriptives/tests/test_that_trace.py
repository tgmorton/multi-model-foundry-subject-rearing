"""Tests for §1.8 That-Trace Environments analyzer."""

from analysis.corpus_descriptives.analyzers.that_trace import ThatTraceAnalyzer
from .conftest import make_doc


class TestThatTrace:
    def test_bridge_verb_with_comp(self, nlp):
        # "I think that he runs"
        doc = make_doc(
            nlp,
            words=["I", "think", "that", "he", "runs"],
            pos=["PRON", "VERB", "SCONJ", "PRON", "VERB"],
            deps=["nsubj", "ROOT", "mark", "nsubj", "ccomp"],
            heads=[1, 1, 4, 4, 1],
            morphs=[None, "VerbForm=Fin", None, None, "VerbForm=Fin"],
            lemmas=["I", "think", "that", "he", "run"],
        )
        analyzer = ThatTraceAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["complementizer_rate"]["with_that"] == 1
        bv = results["overall"]["bridge_verb_counts"]
        assert any(v["verb_lemma"] == "think" for v in bv)

    def test_bridge_verb_without_comp(self, nlp):
        # "I think he runs"
        doc = make_doc(
            nlp,
            words=["I", "think", "he", "runs"],
            pos=["PRON", "VERB", "PRON", "VERB"],
            deps=["nsubj", "ROOT", "nsubj", "ccomp"],
            heads=[1, 1, 3, 1],
            morphs=[None, "VerbForm=Fin", None, "VerbForm=Fin"],
            lemmas=["I", "think", "he", "run"],
        )
        analyzer = ThatTraceAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["complementizer_rate"]["without_that"] == 1

    def test_non_bridge_verb_ignored(self, nlp):
        # "I eat that he runs" — "eat" not a bridge verb
        doc = make_doc(
            nlp,
            words=["I", "eat", "that", "he", "runs"],
            pos=["PRON", "VERB", "SCONJ", "PRON", "VERB"],
            deps=["nsubj", "ROOT", "mark", "nsubj", "ccomp"],
            heads=[1, 1, 4, 4, 1],
            morphs=[None, "VerbForm=Fin", None, None, "VerbForm=Fin"],
            lemmas=["I", "eat", "that", "he", "run"],
        )
        analyzer = ThatTraceAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        assert results["overall"]["bridge_verb_counts"] == []

    def test_subject_extraction(self, nlp):
        # "who do you think runs" — wh in matrix, no nsubj in ccomp
        doc = make_doc(
            nlp,
            words=["who", "do", "you", "think", "runs"],
            pos=["PRON", "AUX", "PRON", "VERB", "VERB"],
            deps=["nsubj", "aux", "nsubj", "ROOT", "ccomp"],
            heads=[3, 3, 3, 3, 3],
            morphs=["PronType=Int", None, None, "VerbForm=Fin", "VerbForm=Fin"],
            lemmas=["who", "do", "you", "think", "run"],
        )
        analyzer = ThatTraceAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ct = results["overall"]["cross_tab"]
        assert any(r["extraction_type"] == "subject" for r in ct)

    def test_object_extraction(self, nlp):
        # "what do you think he sees" — wh in matrix, nsubj "he" in ccomp
        doc = make_doc(
            nlp,
            words=["what", "do", "you", "think", "he", "sees"],
            pos=["PRON", "AUX", "PRON", "VERB", "PRON", "VERB"],
            deps=["obj", "aux", "nsubj", "ROOT", "nsubj", "ccomp"],
            heads=[3, 3, 3, 3, 5, 3],
            morphs=["PronType=Int", None, None, "VerbForm=Fin", None, "VerbForm=Fin"],
            lemmas=["what", "do", "you", "think", "he", "see"],
        )
        analyzer = ThatTraceAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        ct = results["overall"]["cross_tab"]
        assert any(r["extraction_type"] == "object" for r in ct)

    def test_interrogative_complement_filtered(self, nlp):
        # "I wonder who runs" — wh-word is dependent of embedded verb
        doc = make_doc(
            nlp,
            words=["I", "wonder", "who", "runs"],
            pos=["PRON", "VERB", "PRON", "VERB"],
            deps=["nsubj", "ROOT", "nsubj", "ccomp"],
            heads=[1, 1, 3, 1],
            morphs=[None, "VerbForm=Fin", "PronType=Int", "VerbForm=Fin"],
            lemmas=["I", "wonder", "who", "run"],
        )
        # "wonder" is not a bridge verb, so it would be filtered anyway
        # But let's test with a bridge verb
        doc2 = make_doc(
            nlp,
            words=["I", "know", "who", "runs"],
            pos=["PRON", "VERB", "PRON", "VERB"],
            deps=["nsubj", "ROOT", "nsubj", "ccomp"],
            heads=[1, 1, 3, 1],
            morphs=[None, "VerbForm=Fin", "PronType=Int", "VerbForm=Fin"],
            lemmas=["I", "know", "who", "run"],
        )
        analyzer = ThatTraceAnalyzer()
        analyzer.process_doc(doc2, "TestGenre")
        results = analyzer.get_results()

        # Should be filtered because "who" is a dependent of embedded "runs"
        assert results["overall"]["bridge_verb_counts"] == []
