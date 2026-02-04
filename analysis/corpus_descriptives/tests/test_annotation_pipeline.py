"""Integration tests for the annotation pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
import spacy

pl = pytest.importorskip("polars")

from analysis.corpus_descriptives.config import CorpusAnalysisConfig
from analysis.corpus_descriptives.pipeline import (
    CorpusAnnotationPipeline,
    generate_sentence_id,
)
from analysis.corpus_descriptives.annotators import get_default_annotators


def _spacy_model_available(model_name: str) -> bool:
    """Check if a spaCy model is installed."""
    try:
        spacy.load(model_name)
        return True
    except OSError:
        return False


class TestSentenceIdGeneration:
    """Tests for sentence ID generation."""

    def test_basic_id_format(self):
        """Test basic ID format."""
        sid = generate_sentence_id("train_90M", "CHILDES", 0, 0)
        assert sid == "train_90M_childes_000000_000000"

    def test_id_with_spaces(self):
        """Test ID normalization with spaces in genre."""
        sid = generate_sentence_id("test", "Simple Wikipedia", 1, 2)
        assert sid == "test_simple_wikipedia_000001_000002"

    def test_id_with_hyphens(self):
        """Test ID normalization with hyphens in genre."""
        sid = generate_sentence_id("test", "en-US", 10, 100)
        assert sid == "test_en_us_000010_000100"

    def test_id_uniqueness(self):
        """Test that different inputs produce different IDs."""
        id1 = generate_sentence_id("train", "CHILDES", 0, 0)
        id2 = generate_sentence_id("train", "CHILDES", 0, 1)
        id3 = generate_sentence_id("train", "CHILDES", 1, 0)
        id4 = generate_sentence_id("train", "BNC", 0, 0)
        id5 = generate_sentence_id("test", "CHILDES", 0, 0)

        ids = [id1, id2, id3, id4, id5]
        assert len(ids) == len(set(ids)), "IDs should be unique"


@pytest.fixture
def small_corpus(tmp_path):
    """Create a minimal corpus for integration testing."""
    # CHILDES-style file
    childes = tmp_path / "childes.train"
    childes.write_text(
        "*CHI:\tI want a cookie.\n"
        "*MOT:\tyou can not have a cookie.\n"
        "*CHI:\twhy not?\n"
        "*MOT:\tbecause it is raining.\n"
    )

    # Generic file (simple_wiki-style)
    wiki = tmp_path / "simple_wiki.train"
    wiki.write_text(
        "The cat that sat on the mat was happy.\n"
        "There is a dog in the park.\n"
        "I think that she runs fast.\n"
        "Who is the president?\n"
    )

    return tmp_path


@pytest.fixture
def annotation_config(small_corpus, tmp_path):
    """Config for annotation pipeline test."""
    output = tmp_path / "output"
    return CorpusAnalysisConfig(
        input_path=small_corpus,
        output_path=output,
        split_name="test",
        spacy_model="en_core_web_sm",
        spacy_batch_size=10,
        chunk_size=100,
        genre_map={
            "childes": "CHILDES",
            "simple_wiki": "SimpleWikipedia",
        },
        annotation_mode=True,
        layered_output=False,
    )


@pytest.fixture
def layered_annotation_config(small_corpus, tmp_path):
    """Config for layered annotation pipeline test."""
    output = tmp_path / "output_layered"
    return CorpusAnalysisConfig(
        input_path=small_corpus,
        output_path=output,
        split_name="test",
        spacy_model="en_core_web_sm",
        spacy_batch_size=10,
        chunk_size=100,
        genre_map={
            "childes": "CHILDES",
            "simple_wiki": "SimpleWikipedia",
        },
        annotation_mode=True,
        layered_output=True,
    )


@pytest.mark.skipif(
    not _spacy_model_available("en_core_web_sm"),
    reason="en_core_web_sm not installed",
)
class TestAnnotationPipeline:
    """Integration tests for CorpusAnnotationPipeline."""

    def test_pipeline_runs(self, annotation_config):
        """Test that annotation pipeline runs without error."""
        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=annotation_config,
            annotators=annotators,
            layered=False,
        )
        metadata = pipeline.run()

        assert "total_sentences" in metadata
        assert metadata["total_sentences"] > 0

    def test_parquet_output_created(self, annotation_config):
        """Test that Parquet file is created."""
        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=annotation_config,
            annotators=annotators,
            layered=False,
        )
        pipeline.run()

        output_dir = pipeline.output_dir
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_parquet_readable(self, annotation_config):
        """Test that output Parquet is readable with polars."""
        import polars as pl

        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=annotation_config,
            annotators=annotators,
            layered=False,
        )
        pipeline.run()

        output_dir = pipeline.output_dir
        parquet_file = output_dir / f"{annotation_config.split_name}.parquet"

        if parquet_file.exists():
            df = pl.read_parquet(parquet_file)
            assert df.height > 0
            assert "sentence_id" in df.columns
            assert "text" in df.columns
            assert "genre" in df.columns

    def test_childes_speaker_split(self, annotation_config):
        """Test that CHILDES is split by speaker role."""
        import polars as pl

        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=annotation_config,
            annotators=annotators,
            layered=False,
        )
        pipeline.run()

        output_dir = pipeline.output_dir
        parquet_file = output_dir / f"{annotation_config.split_name}.parquet"

        if parquet_file.exists():
            df = pl.read_parquet(parquet_file)
            genres = df["genre"].unique().to_list()

            # CHILDES should be split into child/adult
            assert "CHILDES_child" in genres or "CHILDES_adult" in genres
            assert "CHILDES" not in genres


@pytest.mark.skipif(
    not _spacy_model_available("en_core_web_sm"),
    reason="en_core_web_sm not installed",
)
class TestLayeredAnnotationPipeline:
    """Integration tests for layered output mode."""

    def test_layered_pipeline_runs(self, layered_annotation_config):
        """Test that layered annotation pipeline runs."""
        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=layered_annotation_config,
            annotators=annotators,
            layered=True,
        )
        metadata = pipeline.run()

        assert "total_sentences" in metadata
        assert metadata["total_sentences"] > 0

    def test_layered_directory_structure(self, layered_annotation_config):
        """Test that layered output creates correct directory structure."""
        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=layered_annotation_config,
            annotators=annotators,
            layered=True,
        )
        pipeline.run()

        output_dir = pipeline.output_dir

        # Check base directory
        base_dir = output_dir / "base"
        assert base_dir.exists()
        base_files = list(base_dir.glob("*.parquet"))
        assert len(base_files) >= 1

        # Check layers directory
        layers_dir = output_dir / "layers"
        assert layers_dir.exists()

    def test_layered_base_file_content(self, layered_annotation_config):
        """Test that base file contains expected columns."""
        import polars as pl

        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=layered_annotation_config,
            annotators=annotators,
            layered=True,
        )
        pipeline.run()

        output_dir = pipeline.output_dir
        base_file = output_dir / "base" / f"{layered_annotation_config.split_name}.parquet"

        if base_file.exists():
            df = pl.read_parquet(base_file)

            # Base should have identifiers and text
            assert "sentence_id" in df.columns
            assert "text" in df.columns
            assert "genre" in df.columns
            assert "tokens" in df.columns or df.height == 0

    def test_metadata_json_created(self, layered_annotation_config):
        """Test that metadata.json is created."""
        annotators = get_default_annotators()
        pipeline = CorpusAnnotationPipeline(
            config=layered_annotation_config,
            annotators=annotators,
            layered=True,
        )
        pipeline.run()

        output_dir = pipeline.output_dir
        metadata_path = output_dir / "metadata.json"

        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "split_name" in metadata
        assert "total_sentences" in metadata
        assert "annotators" in metadata
