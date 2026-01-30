"""Tests for §1.1 Pronoun Inventory analyzer."""

from analysis.corpus_descriptives.analyzers.pronoun_inventory import (
    PronounInventoryAnalyzer,
)
from .conftest import make_doc


class TestPronounInventory:
    def test_subject_pronoun(self, nlp):
        # "I run"
        doc = make_doc(
            nlp,
            words=["I", "run"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            morphs=[
                {"Person": "1", "Number": "Sing", "Case": "Nom"},
                {"VerbForm": "Fin"},
            ],
            lemmas=["I", "run"],
        )
        analyzer = PronounInventoryAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        entries = results["overall"]["entries"]
        assert len(entries) == 1
        assert entries[0]["function"] == "subject"
        assert entries[0]["person"] == "1"
        assert entries[0]["number"] == "Sing"
        assert entries[0]["lemma"] == "i"
        assert entries[0]["count"] == 1

    def test_object_pronoun(self, nlp):
        # "see him"
        doc = make_doc(
            nlp,
            words=["see", "him"],
            pos=["VERB", "PRON"],
            deps=["ROOT", "obj"],
            heads=[0, 0],
            morphs=[
                {"VerbForm": "Fin"},
                {"Person": "3", "Number": "Sing", "Case": "Acc"},
            ],
            lemmas=["see", "he"],
        )
        analyzer = PronounInventoryAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()

        entries = results["overall"]["entries"]
        assert len(entries) == 1
        assert entries[0]["function"] == "object"
        assert entries[0]["lemma"] == "he"

    def test_genre_breakdown(self, nlp):
        doc1 = make_doc(
            nlp,
            words=["I", "go"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            morphs=[{"Person": "1", "Number": "Sing"}, None],
            lemmas=["I", "go"],
        )
        doc2 = make_doc(
            nlp,
            words=["he", "runs"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            morphs=[{"Person": "3", "Number": "Sing"}, None],
            lemmas=["he", "run"],
        )
        analyzer = PronounInventoryAnalyzer()
        analyzer.process_doc(doc1, "CHILDES")
        analyzer.process_doc(doc2, "BNC")
        results = analyzer.get_results()

        assert "CHILDES" in results["by_genre"]
        assert "BNC" in results["by_genre"]
        assert len(results["overall"]["entries"]) == 2

    def test_non_argumental_pronoun_ignored(self, nlp):
        # Pronoun with dep=poss should not be counted
        doc = make_doc(
            nlp,
            words=["my", "book"],
            pos=["PRON", "NOUN"],
            deps=["poss", "ROOT"],
            heads=[1, 1],
            lemmas=["my", "book"],
        )
        analyzer = PronounInventoryAnalyzer()
        analyzer.process_doc(doc, "TestGenre")
        results = analyzer.get_results()
        assert results["overall"]["total_pronouns"] == 0
