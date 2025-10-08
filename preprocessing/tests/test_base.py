"""
Unit tests for AblationPipeline base class.

Tests initialization, configuration, and core pipeline functionality.
Note: Full integration tests require spaCy models and are in separate files.
"""

import pytest
from pathlib import Path

from preprocessing.base import AblationPipeline
from preprocessing.config import AblationConfig
from preprocessing.registry import AblationRegistry


class TestAblationPipelineInit:
    """Tests for AblationPipeline initialization."""

    def test_cannot_init_without_registered_ablation(self, tmp_path):
        """Cannot initialize pipeline with unregistered ablation type."""
        config = AblationConfig(
            type="nonexistent_ablation",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output"
        )

        with pytest.raises(KeyError) as exc_info:
            AblationPipeline(config)

        assert "not found in registry" in str(exc_info.value)

    def test_init_with_registered_ablation(self, tmp_path, dummy_ablation_function):
        """Can initialize pipeline with registered ablation."""
        # Register a dummy ablation
        AblationRegistry.register("dummy", dummy_ablation_function)

        config = AblationConfig(
            type="dummy",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            spacy_model="en_core_web_sm"
        )

        # This will try to load spaCy model, so we expect it might fail
        # In a real test environment with spaCy installed, this would work
        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config == config
            assert pipeline.ablation_fn == dummy_ablation_function
        except OSError:
            # spaCy model not available in test environment
            pytest.skip("spaCy model not available")

    def test_stores_config(self, tmp_path, dummy_ablation_function):
        """Pipeline stores the configuration."""
        AblationRegistry.register("dummy", dummy_ablation_function)

        config = AblationConfig(
            type="dummy",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output"
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.type == "dummy"
            assert pipeline.config.input_path == config.input_path
        except OSError:
            pytest.skip("spaCy model not available")

    def test_retrieves_ablation_and_validator(self, tmp_path, dummy_ablation_function, dummy_validator_function):
        """Pipeline retrieves both ablation and validator functions."""
        AblationRegistry.register("dummy", dummy_ablation_function, dummy_validator_function)

        config = AblationConfig(
            type="dummy",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output"
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.ablation_fn == dummy_ablation_function
            assert pipeline.validation_fn == dummy_validator_function
        except OSError:
            pytest.skip("spaCy model not available")


class TestAblationPipelineProcessCorpus:
    """Tests for process_corpus method."""

    def test_process_corpus_fails_with_no_files(self, tmp_path, dummy_ablation_function, empty_corpus_dir):
        """process_corpus raises error when no .train files found."""
        AblationRegistry.register("dummy", dummy_ablation_function)

        config = AblationConfig(
            type="dummy",
            input_path=empty_corpus_dir,
            output_path=tmp_path / "output"
        )

        try:
            pipeline = AblationPipeline(config)

            with pytest.raises(ValueError) as exc_info:
                pipeline.process_corpus()

            assert "No '.train' files found" in str(exc_info.value)
        except OSError:
            pytest.skip("spaCy model not available")


class TestAblationPipelineConfiguration:
    """Tests for pipeline configuration handling."""

    def test_respects_seed_from_config(self, tmp_path, dummy_ablation_function):
        """Pipeline uses seed from configuration."""
        AblationRegistry.register("dummy", dummy_ablation_function)

        config = AblationConfig(
            type="dummy",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            seed=123
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.seed == 123
        except OSError:
            pytest.skip("spaCy model not available")

    def test_respects_chunk_size_from_config(self, tmp_path, dummy_ablation_function):
        """Pipeline uses chunk size from configuration."""
        AblationRegistry.register("dummy", dummy_ablation_function)

        config = AblationConfig(
            type="dummy",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            chunk_size=500
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.chunk_size == 500
        except OSError:
            pytest.skip("spaCy model not available")

    def test_respects_verbose_from_config(self, tmp_path, dummy_ablation_function):
        """Pipeline uses verbose setting from configuration."""
        AblationRegistry.register("dummy", dummy_ablation_function)

        config = AblationConfig(
            type="dummy",
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            verbose=True
        )

        try:
            pipeline = AblationPipeline(config)
            assert pipeline.config.verbose is True
        except OSError:
            pytest.skip("spaCy model not available")
