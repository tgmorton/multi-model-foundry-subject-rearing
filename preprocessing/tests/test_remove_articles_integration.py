"""
Integration tests for remove_articles ablation.

Tests the complete pipeline with the remove_articles ablation registered
and verifies it produces expected output.
"""

import pytest
from pathlib import Path

from preprocessing import AblationPipeline, AblationConfig, AblationRegistry

# Import ablations module to trigger registration
# This must happen after importing preprocessing package
from preprocessing.ablations import remove_articles  # noqa: F401


class TestRemoveArticlesRegistration:
    """Test that remove_articles is properly registered."""

    def test_remove_articles_is_registered(self):
        """remove_articles ablation should be registered."""
        # Re-import to ensure registration (in case other tests cleared registry)
        from preprocessing.ablations import remove_articles  # noqa: F401, F811
        assert AblationRegistry.is_registered("remove_articles")

    def test_can_retrieve_remove_articles(self):
        """Can retrieve remove_articles ablation and validator."""
        # Re-import to ensure registration (in case other tests cleared registry)
        from preprocessing.ablations import remove_articles  # noqa: F401, F811
        ablation_fn, validator_fn = AblationRegistry.get("remove_articles")

        assert ablation_fn is not None
        assert validator_fn is not None


@pytest.mark.skipif(
    not pytest.importorskip("spacy", reason="spaCy not installed"),
    reason="Requires spaCy models"
)
class TestRemoveArticlesFunction:
    """Test the remove_articles ablation function directly."""

    def test_removes_articles_from_simple_text(self):
        """Removes articles from simple text."""
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not installed")

        from preprocessing.ablations.remove_articles import remove_articles_doc

        doc = nlp("The cat sat on a mat.")
        ablated_text, num_removed = remove_articles_doc(doc)

        # Should remove "The" and "a"
        assert num_removed == 2
        assert "the" not in ablated_text.lower() or "a" not in ablated_text.lower()

    def test_preserves_non_articles(self):
        """Preserves non-article determiners."""
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not installed")

        from preprocessing.ablations.remove_articles import remove_articles_doc

        # "This" and "that" are determiners but not articles
        doc = nlp("This cat likes that dog.")
        ablated_text, num_removed = remove_articles_doc(doc)

        # Should not remove "this" or "that"
        assert num_removed == 0
        assert "This" in ablated_text
        assert "that" in ablated_text

    def test_handles_empty_doc(self):
        """Handles empty document."""
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not installed")

        from preprocessing.ablations.remove_articles import remove_articles_doc

        doc = nlp("")
        ablated_text, num_removed = remove_articles_doc(doc)

        assert num_removed == 0
        assert ablated_text == ""


@pytest.mark.skipif(
    not pytest.importorskip("spacy", reason="spaCy not installed"),
    reason="Requires spaCy models"
)
class TestRemoveArticlesValidator:
    """Test the validation function."""

    def test_validator_passes_when_articles_removed(self):
        """Validator passes when articles are successfully removed."""
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not installed")

        from preprocessing.ablations.remove_articles import validate_article_removal

        original = "The cat sat on a mat."
        ablated = "cat sat on mat."

        result = validate_article_removal(original, ablated, nlp)

        assert result is True

    def test_validator_passes_when_no_articles_exist(self):
        """Validator passes when original has no articles."""
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model en_core_web_sm not installed")

        from preprocessing.ablations.remove_articles import validate_article_removal

        original = "Cat sat on mat."
        ablated = "Cat sat on mat."

        result = validate_article_removal(original, ablated, nlp)

        assert result is True


@pytest.mark.skipif(
    not pytest.importorskip("spacy", reason="spaCy not installed"),
    reason="Requires spaCy models and large test data"
)
class TestRemoveArticlesPipeline:
    """Integration tests for full pipeline with remove_articles."""

    def test_pipeline_can_initialize_with_remove_articles(self, tmp_path):
        """Can initialize pipeline with remove_articles ablation."""
        config = AblationConfig(
            type="remove_articles",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            seed=42
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.type == "remove_articles"
        except OSError:
            pytest.skip("spaCy model not available")

    def test_pipeline_processes_single_file(self, tmp_path):
        """Pipeline can process a single file with articles."""
        # Create input corpus
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        input_file = input_dir / "test.train"
        input_file.write_text("The cat sat on a mat.\nThe dog ran fast.\n")

        # Configure pipeline
        config = AblationConfig(
            type="remove_articles",
            input_path=input_dir,
            output_path=tmp_path / "output",
            seed=42,
            skip_validation=True  # Skip validation for faster test
        )

        try:
            pipeline = AblationPipeline(config)
            manifest = pipeline.process_corpus()

            # Check output was created
            output_file = tmp_path / "output" / "test.train"
            assert output_file.exists()

            # Check some articles were removed
            assert manifest.metadata.total_items_ablated > 0

            # Check manifest was saved
            manifest_file = tmp_path / "output" / "ABLATION_MANIFEST.json"
            assert manifest_file.exists()

        except OSError:
            pytest.skip("spaCy model not available")
