"""Tests for the query interface."""

import tempfile
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

from analysis.corpus_descriptives.query import AnnotatedCorpus, load_layered_corpus


@pytest.fixture
def sample_annotations(tmp_path):
    """Create sample annotation Parquet file for testing."""
    data = {
        "sentence_id": [
            "test_childes_000000_000000",
            "test_childes_000000_000001",
            "test_bnc_000000_000000",
            "test_bnc_000000_000001",
        ],
        "text": [
            "I want a cookie.",
            "There is a cat.",
            "Who did that?",
            "The cat that sat was happy.",
        ],
        "genre": ["CHILDES_child", "CHILDES_adult", "BNC", "BNC"],
        "speaker": ["CHI", "MOT", None, None],
        "role": ["child", "adult", None, None],
        "n_tokens": [5, 5, 4, 7],
        # Boolean flags
        "has_null_subject": [False, False, False, False],
        "has_expletive": [False, True, False, False],
        "has_that_trace_context": [False, False, False, False],
        "has_that_trace_violation": [False, False, False, False],
        "has_wh_extraction": [False, False, True, False],
        "has_relative_clause": [False, False, False, True],
        "has_negation": [False, False, False, False],
        "is_fragment": [False, False, False, False],
        "is_imperative": [False, False, False, False],
    }

    df = pl.DataFrame(data)
    parquet_path = tmp_path / "test_annotations.parquet"
    df.write_parquet(parquet_path)

    return parquet_path


@pytest.fixture
def sample_layered_corpus(tmp_path):
    """Create sample layered corpus structure for testing."""
    # Create directory structure
    base_dir = tmp_path / "annotated_corpus" / "base"
    base_dir.mkdir(parents=True)

    layers_dir = tmp_path / "annotated_corpus" / "layers"
    layers_dir.mkdir(parents=True)

    # Base data
    base_data = {
        "sentence_id": ["s1", "s2", "s3"],
        "text": ["Hello.", "There is a cat.", "Who?"],
        "genre": ["CHILDES", "BNC", "BNC"],
        "n_tokens": [1, 5, 1],
    }
    base_df = pl.DataFrame(base_data)
    base_df.write_parquet(base_dir / "test.parquet")

    # Expletive layer
    expletive_dir = layers_dir / "expletives"
    expletive_dir.mkdir()
    expletive_data = {
        "sentence_id": ["s1", "s2", "s3"],
        "has_expletive": [False, True, False],
    }
    expletive_df = pl.DataFrame(expletive_data)
    expletive_df.write_parquet(expletive_dir / "test.parquet")

    return tmp_path / "annotated_corpus"


class TestAnnotatedCorpus:
    """Tests for AnnotatedCorpus query interface."""

    def test_init_from_file(self, sample_annotations):
        """Test initialization from single file."""
        corpus = AnnotatedCorpus(sample_annotations)
        assert corpus.df is not None

    def test_init_from_directory(self, tmp_path, sample_annotations):
        """Test initialization from directory."""
        # Move file into directory
        dir_path = tmp_path / "corpus_dir"
        dir_path.mkdir()
        import shutil
        shutil.copy(sample_annotations, dir_path / "annotations.parquet")

        corpus = AnnotatedCorpus(dir_path)
        assert corpus.df is not None

    def test_filter_by_genre(self, sample_annotations):
        """Test filtering by genre."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.filter(genre="BNC")

        assert result.height == 2
        assert all(r == "BNC" for r in result["genre"].to_list())

    def test_filter_by_flag(self, sample_annotations):
        """Test filtering by boolean flag."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.filter(has_expletive=True)

        assert result.height == 1
        assert result["text"][0] == "There is a cat."

    def test_filter_multiple_conditions(self, sample_annotations):
        """Test filtering by multiple conditions."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.filter(genre="BNC", has_wh_extraction=True)

        assert result.height == 1
        assert result["text"][0] == "Who did that?"

    def test_get_sentence_by_id(self, sample_annotations):
        """Test getting sentence by ID."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.get_sentence_by_id("test_childes_000000_000000")

        assert result is not None
        assert result["text"] == "I want a cookie."

    def test_get_sentence_by_id_not_found(self, sample_annotations):
        """Test getting non-existent sentence."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.get_sentence_by_id("nonexistent_id")

        assert result is None

    def test_get_sentences_by_ids(self, sample_annotations):
        """Test getting multiple sentences by IDs."""
        corpus = AnnotatedCorpus(sample_annotations)
        ids = ["test_childes_000000_000000", "test_bnc_000000_000000"]
        result = corpus.get_sentences_by_ids(ids)

        assert result.height == 2

    def test_count_by_genre(self, sample_annotations):
        """Test counting by genre."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.count_by_genre()

        assert result.height == 3  # CHILDES_child, CHILDES_adult, BNC

    def test_count_by_genre_with_property(self, sample_annotations):
        """Test counting property by genre."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.count_by_genre("has_expletive")

        # Only one sentence has expletive
        assert result.height >= 1

    def test_get_cooccurrence(self, sample_annotations):
        """Test co-occurrence counting."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.get_cooccurrence("has_expletive", "has_wh_extraction")

        assert isinstance(result, dict)
        # All combinations should be counted
        assert len(result) >= 1

    def test_summary(self, sample_annotations):
        """Test summary statistics."""
        corpus = AnnotatedCorpus(sample_annotations)
        summary = corpus.summary()

        assert "total_sentences" in summary
        assert summary["total_sentences"] == 4
        assert "total_tokens" in summary
        assert "property_counts" in summary

    def test_sample(self, sample_annotations):
        """Test random sampling."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.sample(n=2, seed=42)

        assert result.height == 2

    def test_get_that_trace_violations(self, sample_annotations):
        """Test getting that-trace violations."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.get_that_trace_violations()

        # Sample data has no violations
        assert result.height == 0

    def test_get_expletive_sentences(self, sample_annotations):
        """Test getting expletive sentences."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.get_expletive_sentences()

        assert result.height == 1

    def test_get_wh_questions(self, sample_annotations):
        """Test getting wh-questions."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.get_wh_questions()

        assert result.height == 1

    def test_get_relative_clauses(self, sample_annotations):
        """Test getting relative clauses."""
        corpus = AnnotatedCorpus(sample_annotations)
        result = corpus.get_relative_clauses()

        assert result.height == 1


class TestLoadLayeredCorpus:
    """Tests for load_layered_corpus function."""

    def test_load_layered(self, sample_layered_corpus):
        """Test loading layered corpus."""
        df = load_layered_corpus(sample_layered_corpus)

        # Should be a LazyFrame
        assert isinstance(df, pl.LazyFrame)

        # Collect and check
        result = df.collect()
        assert "sentence_id" in result.columns
        assert "text" in result.columns
        # Expletive layer should be joined
        assert "has_expletive" in result.columns

    def test_load_specific_layers(self, sample_layered_corpus):
        """Test loading only specific layers."""
        df = load_layered_corpus(sample_layered_corpus, layers=["expletives"])

        result = df.collect()
        assert "has_expletive" in result.columns
