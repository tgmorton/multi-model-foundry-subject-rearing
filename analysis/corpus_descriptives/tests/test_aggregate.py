"""Tests for the aggregation utilities."""

import json
import tempfile
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

from analysis.corpus_descriptives.aggregate import (
    load_annotations,
    compute_clause_structure_aggregates,
    compute_that_trace_aggregates,
    compute_pronoun_aggregates,
    compute_negation_aggregates,
    compute_expletive_aggregates,
    compute_verb_aggregates,
    compute_wh_extraction_aggregates,
    compute_relative_clause_aggregates,
    compute_argument_structure_aggregates,
    compute_complexity_aggregates,
    compute_all_aggregates,
    compute_summary_statistics,
)


@pytest.fixture
def sample_annotations_parquet(tmp_path):
    """Create sample annotation Parquet file with nested structures."""
    # Create sample data with nested structures matching schema
    data = {
        "sentence_id": ["s1", "s2", "s3"],
        "genre": ["CHILDES", "BNC", "BNC"],
        "n_tokens": [5, 6, 4],
        # Clause structure
        "clauses": [
            [{"clause_type": "ROOT", "subject_status": "overt", "is_finite": True,
              "verb_idx": 1, "verb_lemma": "want", "subject_idx": 0, "subject_lemma": "i",
              "subject_person": 1, "subject_number": "Sing", "subject_is_pronoun": True,
              "object_status": "overt", "object_idx": 3, "object_lemma": "cookie",
              "object_is_pronoun": False, "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"clause_type": "ROOT", "subject_status": "expletive", "is_finite": True,
              "verb_idx": 1, "verb_lemma": "be", "subject_idx": 0, "subject_lemma": "there",
              "subject_person": None, "subject_number": None, "subject_is_pronoun": False,
              "object_status": "none", "object_idx": None, "object_lemma": None,
              "object_is_pronoun": False, "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [],
        ],
        # That-trace
        "bridge_complements": [
            [],
            [{"matrix_verb_lemma": "think", "matrix_verb_idx": 1, "comp_present": True,
              "extraction_type": "none", "embedded_subject_status": "pronominal",
              "is_violation": False}],
            [],
        ],
        # Expletives
        "expletives": [
            [],
            [{"token_idx": 0, "lemma": "there", "expletive_class": "existential", "verb_lemma": "be"}],
            [],
        ],
        # Pronouns
        "pronouns": [
            [{"token_idx": 0, "lemma": "i", "function": "subject", "person": 1,
              "number": "Sing", "case": "Nom", "is_reflexive": False, "is_clitic": False,
              "head_verb_idx": 1}],
            [],
            [{"token_idx": 0, "lemma": "who", "function": "subject", "person": 3,
              "number": "Sing", "case": "Nom", "is_reflexive": False, "is_clitic": False,
              "head_verb_idx": 1}],
        ],
        # Negations
        "negations": [
            [],
            [],
            [{"token_idx": 2, "lemma": "not", "position": "pre", "subject_status": "overt"}],
        ],
        # Wh-extractions
        "wh_extractions": [
            [],
            [],
            [{"wh_idx": 0, "wh_lemma": "who", "extraction_type": "subject", "clause_level": "root"}],
        ],
        # Relative clauses
        "relative_clauses": [
            [],
            [{"verb_idx": 3, "rel_type": "subject", "relativizer_idx": 2, "relativizer_lemma": "that"}],
            [],
        ],
        # Verbs
        "verbs": [
            [{"token_idx": 1, "lemma": "want", "verb_form": "Fin", "clause_position": "ROOT",
              "tense": "Pres", "aspect": None, "mood": "Ind"}],
            [{"token_idx": 1, "lemma": "be", "verb_form": "Fin", "clause_position": "ROOT",
              "tense": "Pres", "aspect": None, "mood": "Ind"}],
            [{"token_idx": 1, "lemma": "do", "verb_form": "Fin", "clause_position": "ROOT",
              "tense": "Past", "aspect": None, "mood": "Ind"}],
        ],
        # Argument structure
        "argument_structure": [
            [{"verb_idx": 1, "verb_lemma": "want", "transitivity": "transitive",
              "subject_status": "overt", "subject_idx": 0, "subject_lemma": "i",
              "subject_is_pronoun": True, "subject_person": 1, "subject_number": "Sing",
              "object_status": "overt", "object_idx": 3, "object_lemma": "cookie",
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"verb_idx": 1, "verb_lemma": "be", "transitivity": "intransitive",
              "subject_status": "expletive", "subject_idx": 0, "subject_lemma": "there",
              "subject_is_pronoun": False, "subject_person": None, "subject_number": None,
              "object_status": "null", "object_idx": None, "object_lemma": None,
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [],
        ],
        # Complexity
        "complexity": [
            {"n_clauses": 1, "max_embedding_depth": 2, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 1, "max_embedding_depth": 2, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 1, "max_embedding_depth": 1, "has_coordination": False, "n_conjuncts": 0},
        ],
        # Flags
        "has_null_subject": [False, False, False],
        "has_null_object": [False, False, False],
        "has_overt_subject_pronoun": [True, False, True],
        "has_overt_object_pronoun": [False, False, False],
        "has_expletive": [False, True, False],
        "has_that_trace_context": [False, True, False],
        "has_that_trace_violation": [False, False, False],
        "has_wh_extraction": [False, False, True],
        "has_relative_clause": [False, True, False],
        "has_negation": [False, False, True],
        "has_ditransitive": [False, False, False],
        "is_imperative": [False, False, False],
        "is_fragment": [False, False, False],
    }

    df = pl.DataFrame(data)
    parquet_path = tmp_path / "annotations.parquet"
    df.write_parquet(parquet_path)

    return parquet_path


class TestLoadAnnotations:
    """Tests for load_annotations function."""

    def test_load_from_file(self, sample_annotations_parquet):
        """Test loading from single file."""
        df = load_annotations(sample_annotations_parquet)
        assert isinstance(df, pl.LazyFrame)

    def test_load_from_directory(self, sample_annotations_parquet, tmp_path):
        """Test loading from directory."""
        # Create directory with file
        dir_path = tmp_path / "annotations_dir"
        dir_path.mkdir()
        import shutil
        shutil.copy(sample_annotations_parquet, dir_path / "data.parquet")

        df = load_annotations(dir_path)
        assert isinstance(df, pl.LazyFrame)

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_annotations(tmp_path / "nonexistent.parquet")


class TestClauseStructureAggregates:
    """Tests for clause structure aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_clause_structure_aggregates(df)

        assert "overall" in result
        assert "by_genre" in result
        assert "finite_clauses" in result["overall"]

    def test_handles_empty(self, tmp_path):
        """Test handling empty data."""
        # Create empty DataFrame with proper schema
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema([
            ("genre", pa.string()),
            ("clauses", pa.list_(pa.struct([
                ("clause_type", pa.string()),
                ("subject_status", pa.string()),
                ("is_finite", pa.bool_()),
            ]))),
        ])
        empty_table = pa.table({"genre": [], "clauses": []}, schema=schema)
        pq.write_table(empty_table, tmp_path / "empty.parquet")

        df = load_annotations(tmp_path / "empty.parquet")
        result = compute_clause_structure_aggregates(df)

        assert result["overall"]["finite_clauses"] == []


class TestThatTraceAggregates:
    """Tests for that-trace aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_that_trace_aggregates(df)

        assert "overall" in result
        assert "bridge_verb_counts" in result["overall"]
        assert "complementizer_rate" in result["overall"]


class TestPronounAggregates:
    """Tests for pronoun aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_pronoun_aggregates(df)

        assert "overall" in result
        assert "entries" in result["overall"]
        assert "total_pronouns" in result["overall"]


class TestNegationAggregates:
    """Tests for negation aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_negation_aggregates(df)

        assert "overall" in result
        assert "position_counts" in result["overall"]
        assert "total_negations" in result["overall"]


class TestExpletiveAggregates:
    """Tests for expletive aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_expletive_aggregates(df)

        assert "overall" in result
        assert "class_counts" in result["overall"]
        assert "verb_counts" in result["overall"]


class TestVerbAggregates:
    """Tests for verb aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_verb_aggregates(df)

        assert "overall" in result
        assert "verb_form_counts" in result["overall"]
        assert "tense_counts" in result["overall"]


class TestWhExtractionAggregates:
    """Tests for wh-extraction aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_wh_extraction_aggregates(df)

        assert "overall" in result
        assert "extraction_type_counts" in result["overall"]
        assert "wh_lemma_counts" in result["overall"]


class TestRelativeClauseAggregates:
    """Tests for relative clause aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_relative_clause_aggregates(df)

        assert "overall" in result
        assert "rel_type_counts" in result["overall"]


class TestArgumentStructureAggregates:
    """Tests for argument structure aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_argument_structure_aggregates(df)

        assert "overall" in result
        assert "transitivity_counts" in result["overall"]
        assert "subject_status_counts" in result["overall"]


class TestComplexityAggregates:
    """Tests for complexity aggregation."""

    def test_computes_aggregates(self, sample_annotations_parquet):
        """Test basic aggregation."""
        df = load_annotations(sample_annotations_parquet)
        result = compute_complexity_aggregates(df)

        assert "overall" in result
        assert "mean_clauses" in result["overall"]
        assert "mean_depth" in result["overall"]
        assert "fragment_rate" in result["overall"]


class TestComputeAllAggregates:
    """Tests for compute_all_aggregates function."""

    def test_computes_all(self, sample_annotations_parquet):
        """Test computing all aggregates."""
        result = compute_all_aggregates(sample_annotations_parquet)

        expected_keys = {
            "clause_structure", "that_trace", "pronoun_inventory", "negation",
            "expletive", "verb", "wh_extraction", "relative_clause",
            "argument_structure", "complexity",
        }
        assert expected_keys == set(result.keys())

    def test_writes_output(self, sample_annotations_parquet, tmp_path):
        """Test writing output files."""
        output_dir = tmp_path / "output"
        result = compute_all_aggregates(sample_annotations_parquet, output_path=output_dir)

        # Check files were created
        assert (output_dir / "results.json").exists()
        assert (output_dir / "clause_structure.json").exists()

        # Check content is valid JSON
        with open(output_dir / "results.json") as f:
            data = json.load(f)
        assert "clause_structure" in data


class TestComputeSummaryStatistics:
    """Tests for summary statistics computation."""

    def test_computes_summary(self, sample_annotations_parquet):
        """Test summary statistics."""
        result = compute_summary_statistics(sample_annotations_parquet)

        assert "total_sentences" in result
        assert result["total_sentences"] == 3
        assert "total_tokens" in result
        assert "null_subject" in result
        assert "proportion" in result["null_subject"]
