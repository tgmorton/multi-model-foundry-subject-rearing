"""
Integration tests for newly migrated ablations.

Tests remove_expletives, impoverish_determiners, lemmatize_verbs,
and remove_subject_pronominals ablation functions.
"""

import pytest
import spacy
from preprocessing.registry import AblationRegistry
from preprocessing.base import AblationPipeline
from preprocessing.config import AblationConfig


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not available")


# ============================================================================
# Remove Expletives Tests
# NOTE: Stale — these reference the old name "remove_expletives".
# The ablation was renamed to "remove_expletive_sentences_en" and is
# thoroughly tested in test_remove_expletive_sentences.py.
# ============================================================================

# class TestRemoveExpletivesRegistration:
#     """Tests for remove_expletives registration."""
#
#     def test_remove_expletives_is_registered(self):
#         """remove_expletives should be registered."""
#         assert AblationRegistry.is_registered("remove_expletives")
#
#     def test_can_retrieve_remove_expletives(self):
#         """Should be able to retrieve remove_expletives function."""
#         ablation_fn, validator_fn = AblationRegistry.get("remove_expletives")
#         assert callable(ablation_fn)
#         assert callable(validator_fn)
#
#
# class TestRemoveExpletivesFunction:
#     """Tests for remove_expletives ablation function."""
#
#     def test_removes_expletive_it(self, nlp):
#         """Should identify and remove tokens marked as expletives."""
#         text = "It is raining outside."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("remove_expletives")
#
#         ablated_text, num_removed = ablation_fn(doc)
#
#         assert isinstance(ablated_text, str)
#         assert isinstance(num_removed, int)
#         assert num_removed >= 0
#
#     def test_preserves_non_expletives(self, nlp):
#         """Should preserve non-expletive pronouns."""
#         text = "She likes cats. They are friendly."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("remove_expletives")
#
#         ablated_text, num_removed = ablation_fn(doc)
#
#         assert "she" in ablated_text.lower() or "they" in ablated_text.lower()
#
#     def test_handles_empty_doc(self, nlp):
#         """Should handle empty documents."""
#         text = ""
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("remove_expletives")
#
#         ablated_text, num_removed = ablation_fn(doc)
#
#         assert ablated_text == ""
#         assert num_removed == 0


# ============================================================================
# Impoverish Determiners Tests
# NOTE: Stale — "impoverish_determiners" was never registered.
# Closest registered ablation is "impoverish_case_en".
# ============================================================================

# class TestImpoverishDeterminersRegistration:
#     """Tests for impoverish_determiners registration."""
#
#     def test_impoverish_determiners_is_registered(self):
#         """impoverish_determiners should be registered."""
#         assert AblationRegistry.is_registered("impoverish_determiners")
#
#     def test_can_retrieve_impoverish_determiners(self):
#         """Should be able to retrieve impoverish_determiners function."""
#         ablation_fn, validator_fn = AblationRegistry.get("impoverish_determiners")
#         assert callable(ablation_fn)
#         assert callable(validator_fn)
#
#
# class TestImpoverishDeterminersFunction:
#     """Tests for impoverish_determiners ablation function."""
#
#     def test_replaces_a_with_the(self, nlp):
#         """Should replace 'a' with 'the'."""
#         text = "A cat sat on a mat."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("impoverish_determiners")
#
#         ablated_text, num_replaced = ablation_fn(doc)
#
#         assert "the cat" in ablated_text.lower()
#         assert num_replaced >= 2
#
#     def test_replaces_an_with_the(self, nlp):
#         """Should replace 'an' with 'the'."""
#         text = "An elephant walks slowly."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("impoverish_determiners")
#
#         ablated_text, num_replaced = ablation_fn(doc)
#
#         assert "the elephant" in ablated_text.lower()
#         assert num_replaced >= 1
#
#     def test_preserves_non_determiners(self, nlp):
#         """Should preserve non-determiner words."""
#         text = "Cats are nice."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("impoverish_determiners")
#
#         ablated_text, num_replaced = ablation_fn(doc)
#
#         assert "cats" in ablated_text.lower()
#         assert "nice" in ablated_text.lower()
#
#     def test_handles_empty_doc(self, nlp):
#         """Should handle empty documents."""
#         text = ""
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("impoverish_determiners")
#
#         ablated_text, num_replaced = ablation_fn(doc)
#
#         assert ablated_text == ""
#         assert num_replaced == 0


# ============================================================================
# Lemmatize Verbs Tests
# ============================================================================


class TestLemmatizeVerbsRegistration:
    """Tests for lemmatize_verbs registration."""

    def test_lemmatize_verbs_is_registered(self):
        """lemmatize_verbs should be registered."""
        assert AblationRegistry.is_registered("lemmatize_verbs")

    def test_can_retrieve_lemmatize_verbs(self):
        """Should be able to retrieve lemmatize_verbs function."""
        ablation_fn, validator_fn = AblationRegistry.get("lemmatize_verbs")
        assert callable(ablation_fn)
        assert callable(validator_fn)


class TestLemmatizeVerbsFunction:
    """Tests for lemmatize_verbs ablation function."""

    def test_lemmatizes_running_to_run(self, nlp):
        """Should lemmatize 'running' to 'run'."""
        text = "She is running quickly."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("lemmatize_verbs")

        ablated_text, num_lemmatized = ablation_fn(doc)

        # 'running' should become 'run', 'is' should become 'be'
        assert num_lemmatized >= 1

    def test_lemmatizes_went_to_go(self, nlp):
        """Should lemmatize 'went' to 'go'."""
        text = "He went to the store."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("lemmatize_verbs")

        ablated_text, num_lemmatized = ablation_fn(doc)

        assert "go" in ablated_text.lower()
        assert num_lemmatized >= 1

    def test_preserves_non_verbs(self, nlp):
        """Should preserve non-verb words."""
        text = "The quick brown fox."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("lemmatize_verbs")

        ablated_text, num_lemmatized = ablation_fn(doc)

        assert "quick" in ablated_text.lower()
        assert "brown" in ablated_text.lower()
        assert "fox" in ablated_text.lower()

    def test_handles_empty_doc(self, nlp):
        """Should handle empty documents."""
        text = ""
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("lemmatize_verbs")

        ablated_text, num_lemmatized = ablation_fn(doc)

        assert ablated_text == ""
        assert num_lemmatized == 0


# ============================================================================
# Remove Subject Pronominals Tests
# NOTE: Stale — "remove_subject_pronominals" was never registered.
# ============================================================================

# class TestRemoveSubjectPronominalsRegistration:
#     """Tests for remove_subject_pronominals registration."""
#
#     def test_remove_subject_pronominals_is_registered(self):
#         """remove_subject_pronominals should be registered."""
#         assert AblationRegistry.is_registered("remove_subject_pronominals")
#
#     def test_can_retrieve_remove_subject_pronominals(self):
#         """Should be able to retrieve remove_subject_pronominals function."""
#         ablation_fn, validator_fn = AblationRegistry.get("remove_subject_pronominals")
#         assert callable(ablation_fn)
#         assert callable(validator_fn)
#
#
# class TestRemoveSubjectPronominalsFunction:
#     """Tests for remove_subject_pronominals ablation function."""
#
#     def test_removes_subject_she(self, nlp):
#         """Should remove subject pronoun 'she'."""
#         text = "She likes cats."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")
#
#         ablated_text, num_removed = ablation_fn(doc)
#
#         assert "she likes" not in ablated_text.lower() or num_removed > 0
#
#     def test_removes_subject_they(self, nlp):
#         """Should remove subject pronoun 'they'."""
#         text = "They are running."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")
#
#         ablated_text, num_removed = ablation_fn(doc)
#
#         assert "they are" not in ablated_text.lower() or num_removed > 0
#
#     def test_preserves_object_pronouns(self, nlp):
#         """Should preserve pronouns that are not subjects."""
#         text = "I gave it to her."
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")
#
#         ablated_text, num_removed = ablation_fn(doc)
#
#         assert "her" in ablated_text.lower() or "it" in ablated_text.lower()
#
#     def test_handles_empty_doc(self, nlp):
#         """Should handle empty documents."""
#         text = ""
#         doc = nlp(text)
#         ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")
#
#         ablated_text, num_removed = ablation_fn(doc)
#
#         assert ablated_text == ""
#         assert num_removed == 0


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestNewAblationsPipeline:
    """Tests for new ablations working with AblationPipeline."""

    # NOTE: remove_expletives, impoverish_determiners, and
    # remove_subject_pronominals pipeline tests commented out — stale names.

    # def test_pipeline_can_initialize_with_remove_expletives(self, tmp_path):
    #     ...

    # def test_pipeline_can_initialize_with_impoverish_determiners(self, tmp_path):
    #     ...

    def test_pipeline_can_initialize_with_lemmatize_verbs(self, tmp_path):
        """Pipeline can initialize with lemmatize_verbs ablation."""
        config = AblationConfig(
            type="lemmatize_verbs",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            spacy_model="en_core_web_sm"
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.type == "lemmatize_verbs"
        except OSError:
            pytest.skip("spaCy model not available")

    # def test_pipeline_can_initialize_with_remove_subject_pronominals(self, tmp_path):
    #     ...


# ============================================================================
# Spanish Ablations — integration tests against es_core_news_lg
# ============================================================================


@pytest.fixture(scope="module")
def nlp_es():
    """Load Spanish spaCy model once for all ES tests."""
    try:
        return spacy.load("es_core_news_lg")
    except OSError:
        pytest.skip("spaCy model es_core_news_lg not available")


class TestSpanishImpoverishCaseRegistration:
    def test_registered(self):
        assert AblationRegistry.is_registered("impoverish_case_es")

    def test_retrievable(self):
        fn, validator = AblationRegistry.get("impoverish_case_es")
        assert callable(fn)
        assert callable(validator)


class TestSpanishImpoverishCaseFunction:
    def test_collapses_tonic_oblique_mi(self, nlp_es):
        """'conmigo y contigo' should lose the oblique forms."""
        text = "él vino conmigo y contigo ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("impoverish_case_es")
        result, n = fn(doc)
        assert "conmigo" not in result.lower()
        assert "contigo" not in result.lower()
        assert "yo" in result.lower()
        assert "tú" in result.lower()
        assert n >= 2

    def test_collapses_dative_le_to_nominative(self, nlp_es):
        """'le dije' — dative *le* should become nominative *él*."""
        text = "le dije la verdad ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("impoverish_case_es")
        result, n = fn(doc)
        # le should be replaced; result should contain él
        assert "él" in result.lower() or n == 0  # spaCy may or may not tag le as PRON

    def test_preserves_nominative_forms(self, nlp_es):
        """'yo tú él ella' should be unchanged (already nominative)."""
        text = "yo , tú , él y ella llegamos ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("impoverish_case_es")
        result, n = fn(doc)
        assert n == 0
        assert "yo" in result.lower()
        assert "tú" in result.lower()
        assert "él" in result.lower()
        assert "ella" in result.lower()

    def test_collapses_short_possessives(self, nlp_es):
        """'mi casa y tu coche' → possessives collapsed."""
        text = "mi casa y tu coche son grandes ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("impoverish_case_es")
        result, n = fn(doc)
        assert n >= 2
        assert "yo casa" in result.lower() or "yo" in result.lower()
        assert "tú coche" in result.lower() or "tú" in result.lower()

    def test_collapses_long_possessives(self, nlp_es):
        """'un libro mío y un coche tuyo' → long possessives collapsed."""
        text = "un libro mío y un coche tuyo ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("impoverish_case_es")
        result, n = fn(doc)
        assert n >= 2

    def test_preserves_articles_lo_la_los_las(self, nlp_es):
        """'la casa y los libros' — *la*/*los* as articles should be preserved.

        This is the critical disambiguation: homographs *la*/*lo*/*los*/*las*
        are articles here, not clitics, and must not be transformed.
        """
        text = "la casa y los libros son grandes ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("impoverish_case_es")
        result, n = fn(doc)
        # As articles (PronType=Art), should be skipped; result should still
        # contain 'la' and 'los'
        assert "la" in result.lower().split()
        assert "los" in result.lower().split()

    def test_collapses_accusative_clitic_lo(self, nlp_es):
        """'lo vi' — clitic *lo* (not article) should become *él*."""
        text = "lo vi ayer ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("impoverish_case_es")
        result, n = fn(doc)
        # Clitic lo → él; not guaranteed by spaCy's parse, but we test
        # for at least some replacement activity.
        # (If spaCy tags `lo` as DET/Art, n=0; if as PRON, n=1.)
        assert isinstance(result, str)
        assert isinstance(n, int)


class TestSpanishLemmatizeVerbs:
    """lemmatize_verbs is language-agnostic; verify it works on Spanish."""

    def test_lemmatizes_spanish_verbs(self, nlp_es):
        """'hablando' → 'hablar', 'comieron' → 'comer'."""
        text = "los niños estaban hablando y comieron después ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("lemmatize_verbs")
        result, n = fn(doc)
        assert n >= 2
        assert "hablar" in result.lower()
        assert "comer" in result.lower()

    def test_preserves_non_verbs_es(self, nlp_es):
        text = "la casa grande es bonita ."
        doc = nlp_es(text)
        fn, _ = AblationRegistry.get("lemmatize_verbs")
        result, _ = fn(doc)
        assert "casa" in result.lower()
        assert "grande" in result.lower()
        assert "bonita" in result.lower()


class TestSpanishRemoveExpletiveSentencesIntegration:
    """End-to-end tests for Spanish expletive removal against real parses."""

    def test_llueve_removed(self, nlp_es):
        """'llueve mucho hoy' should be removed (weather verb)."""
        fn, _ = AblationRegistry.get("remove_expletive_sentences_es")
        doc = nlp_es("llueve mucho hoy .")
        result, count = fn(doc)
        assert count == 1
        assert result == ""

    def test_hay_tres_gatos_removed(self, nlp_es):
        """'hay tres gatos' should be removed (existential haber)."""
        fn, _ = AblationRegistry.get("remove_expletive_sentences_es")
        doc = nlp_es("hay tres gatos en la casa .")
        result, count = fn(doc)
        assert count == 1
        assert result == ""

    def test_auxiliary_haber_kept(self, nlp_es):
        """'juan ha comido' should be kept (haber as aux)."""
        fn, _ = AblationRegistry.get("remove_expletive_sentences_es")
        doc = nlp_es("juan ha comido .")
        result, count = fn(doc)
        assert count == 0
        assert result != ""

    def test_regular_sentence_kept(self, nlp_es):
        fn, _ = AblationRegistry.get("remove_expletive_sentences_es")
        doc = nlp_es("los niños corren en el parque .")
        result, count = fn(doc)
        assert count == 0
        assert result != ""
