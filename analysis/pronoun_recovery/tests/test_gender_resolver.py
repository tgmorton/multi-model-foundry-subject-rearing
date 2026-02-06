"""Tests for gender resolver."""
from analysis.pronoun_recovery.insertion.gender_resolver import GenderResolver
from .conftest import make_doc


class TestGenderResolver:
    def test_english_defaults_to_they(self, nlp):
        doc = make_doc(
            nlp, words=["runs", "."],
            pos=["VERB", "PUNCT"],
            deps=["ROOT", "punct"],
            heads=[0, 0],
            morphs=[{"VerbForm": "Fin"}, None],
        )
        resolver = GenderResolver(language="en")
        result = resolver.resolve(doc[0])
        assert result == "they"

    def test_italian_masculine_default(self, nlp_it):
        doc = make_doc(
            nlp_it, words=["parla", "."],
            pos=["VERB", "PUNCT"],
            deps=["ROOT", "punct"],
            heads=[0, 0],
            morphs=[{"VerbForm": "Fin"}, None],
        )
        resolver = GenderResolver(language="it")
        result = resolver.resolve(doc[0])
        assert result == "lui"

    def test_italian_feminine_from_verb_gender(self, nlp_it):
        doc = make_doc(
            nlp_it, words=["\u00e8", "andata", "."],
            pos=["AUX", "VERB", "PUNCT"],
            deps=["aux", "ROOT", "punct"],
            heads=[1, 1, 1],
            morphs=[
                {"VerbForm": "Fin"},
                {"VerbForm": "Part", "Gender": "Fem"},
                None,
            ],
        )
        resolver = GenderResolver(language="it")
        # The finite verb "\u00e8" should see the participle "andata" with Gender=Fem
        result = resolver.resolve(doc[0])
        # Since "\u00e8" is aux with head "andata" which has Gender=Fem
        assert result == "lei"
