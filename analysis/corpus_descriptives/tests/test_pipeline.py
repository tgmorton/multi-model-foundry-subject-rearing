"""Integration test for the full pipeline using small test data."""

import json
import tempfile
from pathlib import Path

import pytest

from analysis.corpus_descriptives.config import CorpusAnalysisConfig
from analysis.corpus_descriptives.pipeline import CorpusAnalysisPipeline


def _spacy_model_available(model_name: str) -> bool:
    """Check if a spaCy model is installed."""
    try:
        import spacy
        spacy.load(model_name)
        return True
    except OSError:
        return False


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
def pipeline_config(small_corpus, tmp_path):
    """Config for integration test pipeline."""
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
    )


@pytest.mark.skipif(
    not _spacy_model_available("en_core_web_sm"),
    reason="en_core_web_sm not installed",
)
class TestPipelineIntegration:
    def test_pipeline_runs(self, pipeline_config):
        pipeline = CorpusAnalysisPipeline(pipeline_config)
        results = pipeline.run()

        assert len(results) == 8
        for name in pipeline_config.analyses:
            assert name in results

    def test_output_files_created(self, pipeline_config):
        pipeline = CorpusAnalysisPipeline(pipeline_config)
        pipeline.run()

        out = Path(pipeline_config.output_path)
        assert (out / "results.json").exists()
        assert (out / "metadata.json").exists()
        for name in pipeline_config.analyses:
            assert (out / f"{name}.json").exists()
            assert (out / f"{name}.csv").exists()

    def test_genre_breakdown(self, pipeline_config):
        pipeline = CorpusAnalysisPipeline(pipeline_config)
        results = pipeline.run()

        # Each analyzer should have by_genre data
        for name, data in results.items():
            assert "by_genre" in data
            assert "overall" in data

    def test_csv_has_split_column(self, pipeline_config):
        import csv

        pipeline = CorpusAnalysisPipeline(pipeline_config)
        pipeline.run()

        out = Path(pipeline_config.output_path)
        for name in pipeline_config.analyses:
            csv_path = out / f"{name}.csv"
            if csv_path.stat().st_size > 0:
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        assert "split" in row
                        assert row["split"] == "test"
                        break

    def test_childes_speaker_split(self, pipeline_config):
        """CHILDES genre should be split into CHILDES_child and CHILDES_adult."""
        pipeline = CorpusAnalysisPipeline(pipeline_config)
        results = pipeline.run()

        # Check that at least one analyzer has CHILDES_child and CHILDES_adult
        # but NOT bare "CHILDES" in by_genre
        for name, data in results.items():
            genres = set(data.get("by_genre", {}).keys())
            # The test corpus has both CHI and MOT lines, so both should appear
            assert "CHILDES_child" in genres, (
                f"{name}: expected CHILDES_child in genres, got {genres}"
            )
            assert "CHILDES_adult" in genres, (
                f"{name}: expected CHILDES_adult in genres, got {genres}"
            )
            assert "CHILDES" not in genres, (
                f"{name}: bare 'CHILDES' should not appear after speaker split"
            )


# === Italian CHILDES Cleaner Tests ===


class TestItalianCHILDESCleaner:
    """Tests for Italian CHILDES boundary detection and age extraction."""

    def test_italian_corpus_boundary(self):
        """Test Italian corpus boundary detection (4 equal signs each side)."""
        from analysis.corpus_descriptives.line_cleaners import CHILDESCleaner

        cleaner = CHILDESCleaner()
        text, meta = cleaner("= = = = Antelmi = = = =")
        assert text == ""
        assert cleaner._current_corpus == "Antelmi"

    def test_italian_file_boundary(self):
        """Test Italian file boundary detection (3 equal signs, .cha filename)."""
        from analysis.corpus_descriptives.line_cleaners import CHILDESCleaner

        cleaner = CHILDESCleaner()
        text, meta = cleaner("= = = 030409.cha = = =")
        assert text == ""
        # 03 years, 04 months = 40 months
        assert cleaner._current_age == 40
        assert cleaner._current_age_source == "filename"

    def test_italian_age_extraction(self):
        """Test age extraction from Italian CHILDES filenames."""
        from analysis.corpus_descriptives.line_cleaners import CHILDESCleaner

        # 01 years, 06 months = 18 months
        assert CHILDESCleaner._parse_italian_age("010600") == 18
        # 02 years, 11 months = 35 months
        assert CHILDESCleaner._parse_italian_age("021100") == 35
        # 03 years, 04 months = 40 months
        assert CHILDESCleaner._parse_italian_age("030409") == 40
        # Non-matching string
        assert CHILDESCleaner._parse_italian_age("readme") is None

    def test_italian_boundary_state_flow(self):
        """Test that corpus and file boundaries update state correctly."""
        from analysis.corpus_descriptives.line_cleaners import CHILDESCleaner

        cleaner = CHILDESCleaner()

        # First: corpus boundary
        cleaner("= = = = Antelmi = = = =")
        assert cleaner._current_corpus == "Antelmi"

        # Then: file boundary within that corpus
        cleaner("= = = 020300.cha = = =")
        assert cleaner._current_age == 27  # 2*12 + 3
        assert cleaner._current_corpus == "Antelmi"  # corpus unchanged

        # Utterance line should get the metadata
        text, meta = cleaner("*CHI:\tvoglio il latte.")
        assert text == "voglio il latte."
        assert meta["child_age_months"] == 27
        assert meta["childes_corpus"] == "Antelmi"
        assert meta["role"] == "child"

    def test_english_boundary_still_works(self):
        """Test that English CHILDES boundaries still work."""
        from analysis.corpus_descriptives.line_cleaners import CHILDESCleaner

        cleaner = CHILDESCleaner()
        text, meta = cleaner("= = = childes/Bloom/Peter/020100a.cha = = =")
        assert text == ""
