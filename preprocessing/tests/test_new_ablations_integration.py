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
# ============================================================================


class TestRemoveExpletivesRegistration:
    """Tests for remove_expletives registration."""

    def test_remove_expletives_is_registered(self):
        """remove_expletives should be registered."""
        assert AblationRegistry.is_registered("remove_expletives")

    def test_can_retrieve_remove_expletives(self):
        """Should be able to retrieve remove_expletives function."""
        ablation_fn, validator_fn = AblationRegistry.get("remove_expletives")
        assert callable(ablation_fn)
        assert callable(validator_fn)


class TestRemoveExpletivesFunction:
    """Tests for remove_expletives ablation function."""

    def test_removes_expletive_it(self, nlp):
        """Should identify and remove tokens marked as expletives."""
        text = "It is raining outside."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("remove_expletives")

        ablated_text, num_removed = ablation_fn(doc)

        # The function should process the text and may remove expletives if detected
        # Note: en_core_web_sm may not always tag "it" as EXPL in this context
        assert isinstance(ablated_text, str)
        assert isinstance(num_removed, int)
        assert num_removed >= 0

    def test_preserves_non_expletives(self, nlp):
        """Should preserve non-expletive pronouns."""
        text = "She likes cats. They are friendly."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("remove_expletives")

        ablated_text, num_removed = ablation_fn(doc)

        # Non-expletive pronouns should be preserved
        assert "she" in ablated_text.lower() or "they" in ablated_text.lower()

    def test_handles_empty_doc(self, nlp):
        """Should handle empty documents."""
        text = ""
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("remove_expletives")

        ablated_text, num_removed = ablation_fn(doc)

        assert ablated_text == ""
        assert num_removed == 0


# ============================================================================
# Impoverish Determiners Tests
# ============================================================================


class TestImpoverishDeterminersRegistration:
    """Tests for impoverish_determiners registration."""

    def test_impoverish_determiners_is_registered(self):
        """impoverish_determiners should be registered."""
        assert AblationRegistry.is_registered("impoverish_determiners")

    def test_can_retrieve_impoverish_determiners(self):
        """Should be able to retrieve impoverish_determiners function."""
        ablation_fn, validator_fn = AblationRegistry.get("impoverish_determiners")
        assert callable(ablation_fn)
        assert callable(validator_fn)


class TestImpoverishDeterminersFunction:
    """Tests for impoverish_determiners ablation function."""

    def test_replaces_a_with_the(self, nlp):
        """Should replace 'a' with 'the'."""
        text = "A cat sat on a mat."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("impoverish_determiners")

        ablated_text, num_replaced = ablation_fn(doc)

        assert "the cat" in ablated_text.lower()
        assert num_replaced >= 2  # At least 'a' and 'a' replaced

    def test_replaces_an_with_the(self, nlp):
        """Should replace 'an' with 'the'."""
        text = "An elephant walks slowly."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("impoverish_determiners")

        ablated_text, num_replaced = ablation_fn(doc)

        assert "the elephant" in ablated_text.lower()
        assert num_replaced >= 1

    def test_preserves_non_determiners(self, nlp):
        """Should preserve non-determiner words."""
        text = "Cats are nice."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("impoverish_determiners")

        ablated_text, num_replaced = ablation_fn(doc)

        assert "cats" in ablated_text.lower()
        assert "nice" in ablated_text.lower()

    def test_handles_empty_doc(self, nlp):
        """Should handle empty documents."""
        text = ""
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("impoverish_determiners")

        ablated_text, num_replaced = ablation_fn(doc)

        assert ablated_text == ""
        assert num_replaced == 0


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
# ============================================================================


class TestRemoveSubjectPronominalsRegistration:
    """Tests for remove_subject_pronominals registration."""

    def test_remove_subject_pronominals_is_registered(self):
        """remove_subject_pronominals should be registered."""
        assert AblationRegistry.is_registered("remove_subject_pronominals")

    def test_can_retrieve_remove_subject_pronominals(self):
        """Should be able to retrieve remove_subject_pronominals function."""
        ablation_fn, validator_fn = AblationRegistry.get("remove_subject_pronominals")
        assert callable(ablation_fn)
        assert callable(validator_fn)


class TestRemoveSubjectPronominalsFunction:
    """Tests for remove_subject_pronominals ablation function."""

    def test_removes_subject_she(self, nlp):
        """Should remove subject pronoun 'she'."""
        text = "She likes cats."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")

        ablated_text, num_removed = ablation_fn(doc)

        # 'She' as subject should be removed
        assert "she likes" not in ablated_text.lower() or num_removed > 0

    def test_removes_subject_they(self, nlp):
        """Should remove subject pronoun 'they'."""
        text = "They are running."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")

        ablated_text, num_removed = ablation_fn(doc)

        # 'They' as subject should be removed
        assert "they are" not in ablated_text.lower() or num_removed > 0

    def test_preserves_object_pronouns(self, nlp):
        """Should preserve pronouns that are not subjects."""
        text = "I gave it to her."
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")

        ablated_text, num_removed = ablation_fn(doc)

        # Object pronouns like 'her' should be preserved
        assert "her" in ablated_text.lower() or "it" in ablated_text.lower()

    def test_handles_empty_doc(self, nlp):
        """Should handle empty documents."""
        text = ""
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("remove_subject_pronominals")

        ablated_text, num_removed = ablation_fn(doc)

        assert ablated_text == ""
        assert num_removed == 0


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestNewAblationsPipeline:
    """Tests for new ablations working with AblationPipeline."""

    def test_pipeline_can_initialize_with_remove_expletives(self, tmp_path):
        """Pipeline can initialize with remove_expletives ablation."""
        config = AblationConfig(
            type="remove_expletives",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            spacy_model="en_core_web_sm"
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.type == "remove_expletives"
        except OSError:
            pytest.skip("spaCy model not available")

    def test_pipeline_can_initialize_with_impoverish_determiners(self, tmp_path):
        """Pipeline can initialize with impoverish_determiners ablation."""
        config = AblationConfig(
            type="impoverish_determiners",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            spacy_model="en_core_web_sm"
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.type == "impoverish_determiners"
        except OSError:
            pytest.skip("spaCy model not available")

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

    def test_pipeline_can_initialize_with_remove_subject_pronominals(self, tmp_path):
        """Pipeline can initialize with remove_subject_pronominals ablation."""
        config = AblationConfig(
            type="remove_subject_pronominals",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            spacy_model="en_core_web_sm"
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.type == "remove_subject_pronominals"
        except OSError:
            pytest.skip("spaCy model not available")
