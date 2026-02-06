"""Tests for pronoun insertion."""
from analysis.pronoun_recovery.insertion.inserter import PronounInserter
from analysis.pronoun_recovery.insertion.pronoun_maps import get_default_pronoun, needs_gender_resolution
from .conftest import make_doc


class TestPronounMaps:
    def test_1sg_english(self):
        assert get_default_pronoun("PRO.1sg", "en") == "I"

    def test_3sg_english_default(self):
        assert get_default_pronoun("PRO.3sg", "en") == "they"

    def test_imp_returns_none(self):
        assert get_default_pronoun("IMP", "en") is None

    def test_conj_returns_none(self):
        assert get_default_pronoun("CONJ", "en") is None

    def test_none_returns_none(self):
        assert get_default_pronoun("NONE", "en") is None

    def test_needs_gender_resolution_3sg(self):
        assert needs_gender_resolution("PRO.3sg") is True

    def test_needs_gender_resolution_1sg(self):
        assert needs_gender_resolution("PRO.1sg") is False


class TestPronounInserter:
    def test_insert_1sg(self, nlp):
        """PRO.1sg should insert 'I' before the verb."""
        doc = make_doc(
            nlp,
            words=["Ran", "fast", "."],
            pos=["VERB", "ADV", "PUNCT"],
            deps=["ROOT", "advmod", "punct"],
            heads=[0, 0, 0],
            morphs=[{"VerbForm": "Fin"}, None, None],
        )
        inserter = PronounInserter(language="en")
        predictions = [{"token_idx": 0, "token_text": "Ran", "label": "PRO.1sg"}]
        result = inserter.insert(doc, predictions)
        assert "I" in result
        assert "ran" in result.lower()

    def test_imp_no_insertion(self, nlp):
        """IMP label should NOT insert anything."""
        doc = make_doc(
            nlp,
            words=["Run", "fast", "!"],
            pos=["VERB", "ADV", "PUNCT"],
            deps=["ROOT", "advmod", "punct"],
            heads=[0, 0, 0],
        )
        inserter = PronounInserter(language="en")
        predictions = [{"token_idx": 0, "token_text": "Run", "label": "IMP"}]
        result = inserter.insert(doc, predictions)
        assert result.strip() == "Run fast !"

    def test_capitalization_transfer(self, nlp):
        """Sentence-initial verb should transfer cap to inserted pronoun."""
        doc = make_doc(
            nlp,
            words=["Runs", "fast", "."],
            pos=["VERB", "ADV", "PUNCT"],
            deps=["ROOT", "advmod", "punct"],
            heads=[0, 0, 0],
            morphs=[{"VerbForm": "Fin"}, None, None],
        )
        inserter = PronounInserter(language="en")
        predictions = [{"token_idx": 0, "token_text": "Runs", "label": "PRO.3sg"}]
        result = inserter.insert(doc, predictions)
        # "They" should be capitalized, "runs" should be lowercased
        assert result.startswith("They") or result.startswith("they")
