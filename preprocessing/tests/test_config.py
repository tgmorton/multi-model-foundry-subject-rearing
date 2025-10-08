"""
Unit tests for preprocessing configuration models.

Tests validation, path resolution, and seed setting for configuration models.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from preprocessing.config import (
    AblationConfig,
    ProvenanceMetadata,
    FileStatistics,
    ProvenanceManifest
)


class TestAblationConfig:
    """Tests for AblationConfig model."""

    def test_valid_minimal_config(self):
        """Valid minimal configuration should load successfully."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output"
        )

        assert config.type == "remove_articles"
        assert config.seed == 42  # Default
        assert config.chunk_size == 1000  # Default

    def test_custom_seed(self):
        """Can set custom random seed."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output",
            seed=123
        )

        assert config.seed == 123

    def test_custom_chunk_size(self):
        """Can set custom chunk size."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output",
            chunk_size=500
        )

        assert config.chunk_size == 500

    def test_chunk_size_must_be_positive(self):
        """Chunk size must be greater than 0."""
        with pytest.raises(ValidationError) as exc_info:
            AblationConfig(
                type="remove_articles",
                input_path="/tmp/input",
                output_path="/tmp/output",
                chunk_size=0
            )

        assert "chunk_size" in str(exc_info.value)

    def test_negative_chunk_size_rejected(self):
        """Negative chunk size should be rejected."""
        with pytest.raises(ValidationError):
            AblationConfig(
                type="remove_articles",
                input_path="/tmp/input",
                output_path="/tmp/output",
                chunk_size=-100
            )

    def test_optional_replacement_pool_dir(self):
        """Replacement pool directory is optional."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output"
        )

        assert config.replacement_pool_dir is None

    def test_replacement_pool_dir_can_be_set(self):
        """Can set replacement pool directory."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output",
            replacement_pool_dir="/tmp/pool"
        )

        assert config.replacement_pool_dir is not None

    def test_default_spacy_model(self):
        """Default spaCy model is en_core_web_sm."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output"
        )

        assert config.spacy_model == "en_core_web_sm"

    def test_custom_spacy_model(self):
        """Can set custom spaCy model."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output",
            spacy_model="en_core_web_trf"
        )

        assert config.spacy_model == "en_core_web_trf"

    def test_verbose_defaults_to_false(self):
        """Verbose mode defaults to False."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output"
        )

        assert config.verbose is False

    def test_skip_validation_defaults_to_false(self):
        """Skip validation defaults to False."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output"
        )

        assert config.skip_validation is False

    def test_additional_parameters(self):
        """Can include additional ablation-specific parameters."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output",
            parameters={"custom_param": "value", "threshold": 0.5}
        )

        assert config.parameters["custom_param"] == "value"
        assert config.parameters["threshold"] == 0.5

    def test_paths_converted_to_path_objects(self):
        """Path strings are converted to Path objects."""
        config = AblationConfig(
            type="remove_articles",
            input_path="/tmp/input",
            output_path="/tmp/output"
        )

        assert isinstance(config.input_path, Path)
        assert isinstance(config.output_path, Path)


class TestProvenanceMetadata:
    """Tests for ProvenanceMetadata model."""

    def test_create_from_environment(self):
        """Can create metadata from environment."""
        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type="remove_articles",
            random_seed=42,
            chunk_size=1000,
            device="cpu",
            spacy_version="3.7.5",
            spacy_model_name="en_core_web_sm",
            spacy_model_version="3.7.1"
        )

        assert metadata.ablation_type == "remove_articles"
        assert metadata.random_seed == 42
        assert metadata.chunk_size == 1000
        assert metadata.device == "cpu"
        assert metadata.spacy_version == "3.7.5"
        assert metadata.spacy_model_name == "en_core_web_sm"

    def test_timestamp_is_iso_format(self):
        """Timestamp is in ISO format with Z suffix."""
        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type="test",
            random_seed=42,
            chunk_size=1000,
            device="cpu",
            spacy_version="3.7.5",
            spacy_model_name="en_core_web_sm",
            spacy_model_version="3.7.1"
        )

        assert metadata.timestamp.endswith("Z")
        assert "T" in metadata.timestamp

    def test_hostname_and_platform_captured(self):
        """Hostname and platform are captured."""
        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type="test",
            random_seed=42,
            chunk_size=1000,
            device="cpu",
            spacy_version="3.7.5",
            spacy_model_name="en_core_web_sm",
            spacy_model_version="3.7.1"
        )

        assert metadata.hostname is not None
        assert metadata.platform is not None
        assert len(metadata.hostname) > 0


class TestFileStatistics:
    """Tests for FileStatistics model."""

    def test_valid_file_statistics(self):
        """Valid file statistics should load successfully."""
        stats = FileStatistics(
            file_name="test.train",
            original_tokens=1000,
            final_tokens=900,
            items_ablated=50,
            proportion_removed=0.1,
            processing_time_seconds=5.5
        )

        assert stats.file_name == "test.train"
        assert stats.original_tokens == 1000
        assert stats.final_tokens == 900

    def test_negative_tokens_rejected(self):
        """Negative token counts should be rejected."""
        with pytest.raises(ValidationError):
            FileStatistics(
                file_name="test.train",
                original_tokens=-100,
                final_tokens=900,
                items_ablated=50
            )

    def test_proportion_removed_bounded(self):
        """Proportion removed must be between 0 and 1."""
        with pytest.raises(ValidationError):
            FileStatistics(
                file_name="test.train",
                original_tokens=1000,
                final_tokens=900,
                items_ablated=50,
                proportion_removed=1.5  # > 1.0
            )


class TestProvenanceManifest:
    """Tests for ProvenanceManifest model."""

    def test_add_file_stats_updates_aggregates(self):
        """Adding file stats updates aggregate statistics."""
        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type="test",
            random_seed=42,
            chunk_size=1000,
            device="cpu",
            spacy_version="3.7.5",
            spacy_model_name="en_core_web_sm",
            spacy_model_version="3.7.1"
        )

        manifest = ProvenanceManifest(
            metadata=metadata,
            config={}
        )

        stats = FileStatistics(
            file_name="test.train",
            original_tokens=1000,
            final_tokens=900,
            items_ablated=50
        )

        manifest.add_file_stats(stats)

        assert manifest.metadata.total_files_processed == 1
        assert manifest.metadata.total_tokens_original == 1000
        assert manifest.metadata.total_tokens_final == 900
        assert manifest.metadata.total_items_ablated == 50

    def test_add_multiple_file_stats(self):
        """Can add multiple file stats and aggregates accumulate."""
        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type="test",
            random_seed=42,
            chunk_size=1000,
            device="cpu",
            spacy_version="3.7.5",
            spacy_model_name="en_core_web_sm",
            spacy_model_version="3.7.1"
        )

        manifest = ProvenanceManifest(
            metadata=metadata,
            config={}
        )

        stats1 = FileStatistics(
            file_name="file1.train",
            original_tokens=1000,
            final_tokens=900,
            items_ablated=50
        )
        stats2 = FileStatistics(
            file_name="file2.train",
            original_tokens=2000,
            final_tokens=1800,
            items_ablated=100
        )

        manifest.add_file_stats(stats1)
        manifest.add_file_stats(stats2)

        assert manifest.metadata.total_files_processed == 2
        assert manifest.metadata.total_tokens_original == 3000
        assert manifest.metadata.total_tokens_final == 2700
        assert manifest.metadata.total_items_ablated == 150

    def test_save_creates_manifest_file(self, tmp_path):
        """save() creates ABLATION_MANIFEST.json file."""
        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type="test",
            random_seed=42,
            chunk_size=1000,
            device="cpu",
            spacy_version="3.7.5",
            spacy_model_name="en_core_web_sm",
            spacy_model_version="3.7.1"
        )

        manifest = ProvenanceManifest(
            metadata=metadata,
            config={"type": "test"}
        )

        manifest_path = manifest.save(tmp_path)

        assert manifest_path.exists()
        assert manifest_path.name == "ABLATION_MANIFEST.json"

    def test_saved_manifest_is_valid_json(self, tmp_path):
        """Saved manifest is valid JSON."""
        import json

        metadata = ProvenanceMetadata.create_from_environment(
            ablation_type="test",
            random_seed=42,
            chunk_size=1000,
            device="cpu",
            spacy_version="3.7.5",
            spacy_model_name="en_core_web_sm",
            spacy_model_version="3.7.1"
        )

        manifest = ProvenanceManifest(
            metadata=metadata,
            config={"type": "test"}
        )

        manifest_path = manifest.save(tmp_path)

        # Should be able to parse as JSON
        with open(manifest_path) as f:
            data = json.load(f)

        assert data["metadata"]["ablation_type"] == "test"
        assert data["config"]["type"] == "test"
