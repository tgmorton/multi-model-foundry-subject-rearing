"""Tests for the corpus_analysis module."""

import json
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

from analysis.corpus_descriptives.corpus_analysis import (
    corpus_overview,
    phenomenon_rates_by_genre,
    cooccurrence_null_subject_that_trace,
    cooccurrence_negation_null_subject,
    cooccurrence_expletive_depth,
    childes_child_vs_adult,
    childes_speaker_distribution,
    childes_pronoun_distribution,
    childes_person_distribution,
    pronoun_distribution,
    subject_pronoun_distribution,
    subject_pronoun_rate,
    case_marked_pronoun_analysis,
    that_trace_inventory,
    clause_structure_crosstab,
    that_trace_crosstab,
    expletive_inventory,
    expletive_by_class,
    complexity_distribution,
    depth_distribution,
    negation_position_distribution,
    verb_by_subject_status,
    verb_null_subject_rate,
    bridge_verb_null_rates,
    weather_verb_null_rates,
    tense_aspect_mood_distribution,
    verbal_morphology_paradigm,
    person_number_tense_distribution,
    morphology_manipulation_coverage,
    register_comparison,
    register_by_genre,
    manipulation_coverage,
    manipulation_overlap_matrix,
    evaluation_context_frequency,
    language_agnostic_analysis,
    cross_linguistic_comparison,
    wh_question_distribution,
    h5_distal_effects,
    h6a_expletive_analysis,
    null_subject_by_embedding_depth,
    run_all_analyses,
)


@pytest.fixture
def sample_df():
    """Build a minimal annotated DataFrame matching the Parquet schema."""
    return pl.DataFrame({
        "sentence_id": [
            "test_childes_child_000000_000000",
            "test_childes_child_000000_000001",
            "test_childes_adult_000000_000000",
            "test_childes_adult_000000_000001",
            "test_bnc_000000_000000",
            "test_bnc_000000_000001",
            "test_gutenberg_000000_000000",
        ],
        "split": ["test"] * 7,
        "genre": [
            "CHILDES_child", "CHILDES_child",
            "CHILDES_adult", "CHILDES_adult",
            "BNC", "BNC",
            "Gutenberg",
        ],
        "speaker": ["CHI", "CHI", "MOT", "MOT", None, None, None],
        "role": ["child", "child", "adult", "adult", None, None, None],
        "file_name": ["childes"] * 4 + ["bnc"] * 2 + ["gutenberg"],
        "sent_idx": [0, 1, 0, 1, 0, 1, 0],
        "text": [
            "I want cookie.",
            "why not?",
            "you can not have a cookie.",
            "because it is raining.",
            "There is a dog in the park.",
            "I think that she runs fast.",
            "The cat that sat on the mat was happy.",
        ],
        "n_tokens": [4, 3, 7, 5, 8, 7, 9],
        # Clauses
        "clauses": [
            [{"clause_type": "ROOT", "verb_idx": 1, "verb_lemma": "want",
              "is_finite": True, "subject_status": "overt", "subject_idx": 0,
              "subject_lemma": "i", "subject_person": 1, "subject_number": "Sing",
              "subject_is_pronoun": True, "object_status": "overt", "object_idx": 2,
              "object_lemma": "cookie", "object_is_pronoun": False,
              "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [],  # fragment
            [{"clause_type": "ROOT", "verb_idx": 2, "verb_lemma": "have",
              "is_finite": True, "subject_status": "overt", "subject_idx": 0,
              "subject_lemma": "you", "subject_person": 2, "subject_number": "Sing",
              "subject_is_pronoun": True, "object_status": "overt", "object_idx": 5,
              "object_lemma": "cookie", "object_is_pronoun": False,
              "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"clause_type": "ROOT", "verb_idx": 2, "verb_lemma": "rain",
              "is_finite": True, "subject_status": "expletive", "subject_idx": 1,
              "subject_lemma": "it", "subject_person": 3, "subject_number": "Sing",
              "subject_is_pronoun": False, "object_status": "none", "object_idx": None,
              "object_lemma": None, "object_is_pronoun": False,
              "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"clause_type": "ROOT", "verb_idx": 1, "verb_lemma": "be",
              "is_finite": True, "subject_status": "expletive", "subject_idx": 0,
              "subject_lemma": "there", "subject_person": None, "subject_number": None,
              "subject_is_pronoun": False, "object_status": "none", "object_idx": None,
              "object_lemma": None, "object_is_pronoun": False,
              "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"clause_type": "ROOT", "verb_idx": 1, "verb_lemma": "think",
              "is_finite": True, "subject_status": "overt", "subject_idx": 0,
              "subject_lemma": "i", "subject_person": 1, "subject_number": "Sing",
              "subject_is_pronoun": True, "object_status": "none", "object_idx": None,
              "object_lemma": None, "object_is_pronoun": False,
              "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False},
             {"clause_type": "ccomp", "verb_idx": 5, "verb_lemma": "run",
              "is_finite": True, "subject_status": "overt", "subject_idx": 4,
              "subject_lemma": "she", "subject_person": 3, "subject_number": "Sing",
              "subject_is_pronoun": True, "object_status": "none", "object_idx": None,
              "object_lemma": None, "object_is_pronoun": False,
              "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"clause_type": "ROOT", "verb_idx": 7, "verb_lemma": "be",
              "is_finite": True, "subject_status": "overt", "subject_idx": 1,
              "subject_lemma": "cat", "subject_person": 3, "subject_number": "Sing",
              "subject_is_pronoun": False, "object_status": "none", "object_idx": None,
              "object_lemma": None, "object_is_pronoun": False,
              "iobject_status": "none", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
        ],
        # Bridge complements
        "bridge_complements": [
            [], [], [], [],
            [],
            [{"matrix_verb_lemma": "think", "matrix_verb_idx": 1,
              "comp_present": True, "extraction_type": "none",
              "embedded_subject_status": "pronominal", "is_violation": False}],
            [],
        ],
        # Expletives
        "expletives": [
            [], [], [],
            [{"token_idx": 1, "lemma": "it", "expletive_class": "weather", "verb_lemma": "rain"}],
            [{"token_idx": 0, "lemma": "there", "expletive_class": "existential", "verb_lemma": "be"}],
            [], [],
        ],
        # Pronouns
        "pronouns": [
            [{"token_idx": 0, "lemma": "i", "function": "subject", "person": 1,
              "number": "Sing", "case": "Nom", "is_reflexive": False,
              "is_clitic": False, "head_verb_idx": 1}],
            [],
            [{"token_idx": 0, "lemma": "you", "function": "subject", "person": 2,
              "number": "Sing", "case": "Nom", "is_reflexive": False,
              "is_clitic": False, "head_verb_idx": 2}],
            [],
            [],
            [{"token_idx": 0, "lemma": "i", "function": "subject", "person": 1,
              "number": "Sing", "case": "Nom", "is_reflexive": False,
              "is_clitic": False, "head_verb_idx": 1},
             {"token_idx": 4, "lemma": "she", "function": "subject", "person": 3,
              "number": "Sing", "case": "Nom", "is_reflexive": False,
              "is_clitic": False, "head_verb_idx": 5}],
            [],
        ],
        # Negations
        "negations": [
            [], [],
            [{"token_idx": 3, "lemma": "not", "position": "pre", "subject_status": "overt"}],
            [], [], [], [],
        ],
        # Wh-extractions
        "wh_extractions": [
            [],
            [{"wh_idx": 0, "wh_lemma": "why", "extraction_type": "adjunct", "clause_level": "root"}],
            [], [], [], [], [],
        ],
        # Relative clauses
        "relative_clauses": [
            [], [], [], [], [], [],
            [{"verb_idx": 3, "rel_type": "subject", "relativizer_idx": 2, "relativizer_lemma": "that"}],
        ],
        # Verbs (with person/number for Phase 3)
        "verbs": [
            [{"token_idx": 1, "lemma": "want", "verb_form": "Fin",
              "clause_position": "ROOT", "tense": "Pres", "aspect": None, "mood": "Ind",
              "person": 1, "number": "Sing"}],
            [],
            [{"token_idx": 2, "lemma": "have", "verb_form": "Fin",
              "clause_position": "ROOT", "tense": "Pres", "aspect": None, "mood": "Ind",
              "person": 2, "number": "Sing"}],
            [{"token_idx": 2, "lemma": "rain", "verb_form": "Fin",
              "clause_position": "ROOT", "tense": "Pres", "aspect": "Prog", "mood": "Ind",
              "person": 3, "number": "Sing"}],
            [{"token_idx": 1, "lemma": "be", "verb_form": "Fin",
              "clause_position": "ROOT", "tense": "Pres", "aspect": None, "mood": "Ind",
              "person": 3, "number": "Sing"}],
            [{"token_idx": 1, "lemma": "think", "verb_form": "Fin",
              "clause_position": "ROOT", "tense": "Pres", "aspect": None, "mood": "Ind",
              "person": 1, "number": "Sing"},
             {"token_idx": 5, "lemma": "run", "verb_form": "Fin",
              "clause_position": "ccomp", "tense": "Pres", "aspect": None, "mood": "Ind",
              "person": 3, "number": "Sing"}],
            [{"token_idx": 3, "lemma": "sit", "verb_form": "Fin",
              "clause_position": "acl:relcl", "tense": "Past", "aspect": None, "mood": "Ind",
              "person": 3, "number": "Sing"},
             {"token_idx": 7, "lemma": "be", "verb_form": "Fin",
              "clause_position": "ROOT", "tense": "Past", "aspect": None, "mood": "Ind",
              "person": 3, "number": "Sing"}],
        ],
        # Argument structure
        "argument_structure": [
            [{"verb_idx": 1, "verb_lemma": "want", "transitivity": "transitive",
              "subject_status": "overt", "subject_idx": 0, "subject_lemma": "i",
              "subject_is_pronoun": True, "subject_person": 1, "subject_number": "Sing",
              "object_status": "overt", "object_idx": 2, "object_lemma": "cookie",
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [],
            [{"verb_idx": 2, "verb_lemma": "have", "transitivity": "transitive",
              "subject_status": "overt", "subject_idx": 0, "subject_lemma": "you",
              "subject_is_pronoun": True, "subject_person": 2, "subject_number": "Sing",
              "object_status": "overt", "object_idx": 5, "object_lemma": "cookie",
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"verb_idx": 2, "verb_lemma": "rain", "transitivity": "intransitive",
              "subject_status": "expletive", "subject_idx": 1, "subject_lemma": "it",
              "subject_is_pronoun": False, "subject_person": 3, "subject_number": "Sing",
              "object_status": "null", "object_idx": None, "object_lemma": None,
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"verb_idx": 1, "verb_lemma": "be", "transitivity": "intransitive",
              "subject_status": "expletive", "subject_idx": 0, "subject_lemma": "there",
              "subject_is_pronoun": False, "subject_person": None, "subject_number": None,
              "object_status": "null", "object_idx": None, "object_lemma": None,
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"verb_idx": 1, "verb_lemma": "think", "transitivity": "transitive",
              "subject_status": "overt", "subject_idx": 0, "subject_lemma": "i",
              "subject_is_pronoun": True, "subject_person": 1, "subject_number": "Sing",
              "object_status": "clausal", "object_idx": None, "object_lemma": None,
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
            [{"verb_idx": 7, "verb_lemma": "be", "transitivity": "intransitive",
              "subject_status": "overt", "subject_idx": 1, "subject_lemma": "cat",
              "subject_is_pronoun": False, "subject_person": 3, "subject_number": "Sing",
              "object_status": "null", "object_idx": None, "object_lemma": None,
              "object_is_pronoun": False, "iobject_status": "null", "iobject_idx": None,
              "iobject_lemma": None, "iobject_is_pronoun": False}],
        ],
        # Topic info
        "topic_info": [
            {"root_subject_lemma": "i", "root_subject_is_pronoun": True, "root_subject_person": 1},
            {"root_subject_lemma": None, "root_subject_is_pronoun": False, "root_subject_person": None},
            {"root_subject_lemma": "you", "root_subject_is_pronoun": True, "root_subject_person": 2},
            {"root_subject_lemma": "it", "root_subject_is_pronoun": True, "root_subject_person": 3},
            {"root_subject_lemma": "there", "root_subject_is_pronoun": True, "root_subject_person": None},
            {"root_subject_lemma": "i", "root_subject_is_pronoun": True, "root_subject_person": 1},
            {"root_subject_lemma": "cat", "root_subject_is_pronoun": False, "root_subject_person": 3},
        ],
        # Complexity
        "complexity": [
            {"n_clauses": 1, "max_embedding_depth": 2, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 1, "max_embedding_depth": 1, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 1, "max_embedding_depth": 2, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 1, "max_embedding_depth": 2, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 1, "max_embedding_depth": 3, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 2, "max_embedding_depth": 3, "has_coordination": False, "n_conjuncts": 0},
            {"n_clauses": 2, "max_embedding_depth": 4, "has_coordination": False, "n_conjuncts": 0},
        ],
        # Boolean flags
        "has_null_subject": [False, False, False, False, False, False, False],
        "has_null_object": [False, False, False, False, False, False, False],
        "has_overt_subject_pronoun": [True, False, True, False, False, True, False],
        "has_overt_object_pronoun": [False, False, False, False, False, False, False],
        "has_expletive": [False, False, False, True, True, False, False],
        "has_that_trace_context": [False, False, False, False, False, True, False],
        "has_that_trace_violation": [False, False, False, False, False, False, False],
        "has_wh_extraction": [False, True, False, False, False, False, False],
        "has_relative_clause": [False, False, False, False, False, False, True],
        "has_negation": [False, False, True, False, False, False, False],
        "has_ditransitive": [False, False, False, False, False, False, False],
        "is_imperative": [False, False, False, False, False, False, False],
        "is_fragment": [False, True, False, False, False, False, False],
    })


# === Section 1: Corpus Overview ===


class TestCorpusOverview:
    def test_returns_dataframe(self, sample_df):
        result = corpus_overview(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_expected_columns(self, sample_df):
        result = corpus_overview(sample_df)
        assert "genre" in result.columns
        assert "n_sentences" in result.columns
        assert "total_tokens" in result.columns
        assert "mean_sent_length" in result.columns

    def test_correct_counts(self, sample_df):
        result = corpus_overview(sample_df)
        total = result["n_sentences"].sum()
        assert total == 7

    def test_genre_breakdown(self, sample_df):
        result = corpus_overview(sample_df)
        genres = result["genre"].to_list()
        assert "BNC" in genres
        assert "Gutenberg" in genres


# === Section 2: Phenomenon Rates ===


class TestPhenomenonRates:
    def test_returns_dataframe(self, sample_df):
        result = phenomenon_rates_by_genre(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_rate_columns(self, sample_df):
        result = phenomenon_rates_by_genre(sample_df)
        assert any("_rate" in c for c in result.columns)

    def test_rates_between_0_and_1(self, sample_df):
        result = phenomenon_rates_by_genre(sample_df)
        rate_cols = [c for c in result.columns if c.endswith("_rate")]
        for col in rate_cols:
            vals = result[col].to_list()
            for v in vals:
                if v is not None:
                    assert 0.0 <= v <= 1.0, f"{col}: {v} out of range"


# === Section 3: Co-occurrence ===


class TestCooccurrenceNullSubjectThatTrace:
    def test_returns_dict(self, sample_df):
        result = cooccurrence_null_subject_that_trace(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = cooccurrence_null_subject_that_trace(sample_df)
        assert "baseline_null_subject_rate" in result
        assert "that_trace_total" in result
        assert "enrichment_ratio" in result

    def test_baseline_correct(self, sample_df):
        result = cooccurrence_null_subject_that_trace(sample_df)
        # No null subjects in sample data
        assert result["baseline_null_subject_rate"] == 0.0


class TestCooccurrenceNegation:
    def test_returns_dataframe(self, sample_df):
        result = cooccurrence_negation_null_subject(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_genre_column(self, sample_df):
        result = cooccurrence_negation_null_subject(sample_df)
        assert "genre" in result.columns
        assert "ALL" in result["genre"].to_list()


class TestCooccurrenceExpletiveDepth:
    def test_returns_dict(self, sample_df):
        result = cooccurrence_expletive_depth(sample_df)
        assert isinstance(result, dict)

    def test_expletive_depth_exists(self, sample_df):
        result = cooccurrence_expletive_depth(sample_df)
        assert result["expletive_mean_depth"] is not None
        assert result["non_expletive_mean_depth"] is not None


# === Section 4: CHILDES ===


class TestChildesAnalyses:
    def test_child_vs_adult(self, sample_df):
        result = childes_child_vs_adult(sample_df)
        assert isinstance(result, pl.DataFrame)
        roles = result["role"].to_list()
        assert "child" in roles
        assert "adult" in roles

    def test_speaker_distribution(self, sample_df):
        result = childes_speaker_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)
        speakers = result["speaker"].to_list()
        assert "CHI" in speakers
        assert "MOT" in speakers

    def test_pronoun_distribution(self, sample_df):
        result = childes_pronoun_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)
        assert "role" in result.columns
        assert "lemma" in result.columns
        assert "count" in result.columns

    def test_person_distribution(self, sample_df):
        result = childes_person_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)
        if not result.is_empty():
            assert "proportion" in result.columns


# === Section 5: Pronoun Distribution ===


class TestPronounDistribution:
    def test_full_distribution(self, sample_df):
        result = pronoun_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)
        assert "genre" in result.columns
        assert "lemma" in result.columns

    def test_subject_only(self, sample_df):
        result = subject_pronoun_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)
        # All should be subject function
        if not result.is_empty():
            assert result.height > 0


# === Section 6: That-Trace ===


class TestThatTraceInventory:
    def test_returns_dict(self, sample_df):
        result = that_trace_inventory(sample_df)
        assert isinstance(result, dict)

    def test_has_bridge_verbs(self, sample_df):
        result = that_trace_inventory(sample_df)
        assert "bridge_verb_counts" in result
        assert "complementizer_rate" in result

    def test_think_is_bridge_verb(self, sample_df):
        result = that_trace_inventory(sample_df)
        if result["bridge_verb_counts"]:
            verbs = [v["matrix_verb_lemma"] for v in result["bridge_verb_counts"]]
            assert "think" in verbs


# === Section 7: Expletives ===


class TestExpletiveAnalyses:
    def test_inventory(self, sample_df):
        result = expletive_inventory(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_by_class(self, sample_df):
        result = expletive_by_class(sample_df)
        assert isinstance(result, pl.DataFrame)
        if not result.is_empty():
            classes = result["expletive_class"].to_list()
            assert "existential" in classes or "weather" in classes


# === Section 8: Complexity ===


class TestComplexity:
    def test_distribution(self, sample_df):
        result = complexity_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)
        assert "mean_clauses" in result.columns
        assert "mean_depth" in result.columns

    def test_depth_distribution(self, sample_df):
        result = depth_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)
        assert "depth" in result.columns
        assert "count" in result.columns


# === Section 9: Verb Analyses ===


class TestVerbAnalyses:
    def test_verb_by_subject(self, sample_df):
        result = verb_by_subject_status(sample_df)
        assert isinstance(result, pl.DataFrame)
        assert "verb_lemma" in result.columns
        assert "subject_status" in result.columns

    def test_null_rate(self, sample_df):
        result = verb_null_subject_rate(sample_df, min_occurrences=1)
        assert isinstance(result, pl.DataFrame)
        assert "null_rate" in result.columns

    def test_bridge_verbs(self, sample_df):
        result = bridge_verb_null_rates(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_weather_verbs(self, sample_df):
        result = weather_verb_null_rates(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_tense_aspect_mood(self, sample_df):
        result = tense_aspect_mood_distribution(sample_df)
        assert isinstance(result, dict)
        assert "tense" in result
        assert "aspect" in result
        assert "mood" in result
        assert "verb_form" in result
        for key, frame in result.items():
            assert isinstance(frame, pl.DataFrame)


# === Section 10: Register ===


class TestRegister:
    def test_comparison(self, sample_df):
        result = register_comparison(sample_df)
        assert isinstance(result, pl.DataFrame)
        registers = result["register"].to_list()
        assert "spoken" in registers or "written" in registers

    def test_by_genre(self, sample_df):
        result = register_by_genre(sample_df)
        assert isinstance(result, pl.DataFrame)
        assert "register" in result.columns
        assert "genre" in result.columns


# === Section 11: Language-Agnostic ===


class TestLanguageAgnostic:
    def test_returns_dict(self, sample_df):
        result = language_agnostic_analysis(sample_df, "English")
        assert isinstance(result, dict)
        assert result["language"] == "English"

    def test_has_rates(self, sample_df):
        result = language_agnostic_analysis(sample_df)
        rate_keys = [k for k in result if k.endswith("_rate")]
        assert len(rate_keys) > 0

    def test_cross_linguistic(self, sample_df):
        en = language_agnostic_analysis(sample_df, "English")
        it = language_agnostic_analysis(sample_df, "Italian")
        result = cross_linguistic_comparison(en, it)
        assert isinstance(result, pl.DataFrame)
        assert "English" in result.columns
        assert "Italian" in result.columns


# === Research Hypotheses ===


class TestHypotheses:
    def test_h5(self, sample_df):
        result = h5_distal_effects(sample_df)
        assert isinstance(result, dict)

    def test_h6a(self, sample_df):
        result = h6a_expletive_analysis(sample_df)
        assert isinstance(result, dict)
        assert "total_expletive_sentences" in result
        assert result["total_expletive_sentences"] == 2

    def test_null_by_depth(self, sample_df):
        result = null_subject_by_embedding_depth(sample_df)
        assert isinstance(result, pl.DataFrame)
        assert "depth" in result.columns
        assert "null_subject_rate" in result.columns


# === Phase 1 New Functions ===


class TestSubjectPronounRate:
    def test_returns_dict(self, sample_df):
        result = subject_pronoun_rate(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = subject_pronoun_rate(sample_df)
        assert "overall_rate" in result
        assert "total_finite_verbs" in result
        assert "by_genre" in result

    def test_rate_is_valid(self, sample_df):
        result = subject_pronoun_rate(sample_df)
        rate = result.get("overall_rate")
        if rate is not None:
            assert 0.0 <= rate <= 1.0


class TestNegationPositionDistribution:
    def test_returns_dict(self, sample_df):
        result = negation_position_distribution(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = negation_position_distribution(sample_df)
        assert "total_negations" in result
        assert "position_distribution" in result
        assert "by_genre" in result


class TestWhQuestionDistribution:
    def test_returns_dict(self, sample_df):
        result = wh_question_distribution(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = wh_question_distribution(sample_df)
        assert "total_wh_questions" in result
        assert "by_extraction_type" in result
        assert "by_wh_lemma" in result


# === Phase 2 New Functions ===


class TestClauseStructureCrosstab:
    def test_returns_dataframe(self, sample_df):
        result = clause_structure_crosstab(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_expected_columns(self, sample_df):
        result = clause_structure_crosstab(sample_df)
        if not result.is_empty():
            assert "clause_type" in result.columns
            assert "subject_status" in result.columns
            assert "count" in result.columns


class TestCaseMarkedPronounAnalysis:
    def test_returns_dict(self, sample_df):
        result = case_marked_pronoun_analysis(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = case_marked_pronoun_analysis(sample_df)
        assert "total_pronouns" in result
        assert "by_case" in result
        assert "case_neutralization_coverage" in result


class TestThatTraceCrosstab:
    def test_returns_dataframe(self, sample_df):
        result = that_trace_crosstab(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_columns_if_not_empty(self, sample_df):
        result = that_trace_crosstab(sample_df)
        if not result.is_empty():
            assert "comp_present" in result.columns
            assert "extraction_type" in result.columns


class TestManipulationCoverage:
    def test_returns_dict(self, sample_df):
        result = manipulation_coverage(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = manipulation_coverage(sample_df)
        assert "total_sentences" in result
        assert "total_tokens" in result
        assert "manipulations" in result

    def test_manipulations_has_pronoun_removal(self, sample_df):
        result = manipulation_coverage(sample_df)
        manips = result.get("manipulations", {})
        assert "pronoun_removal" in manips
        assert "expletive_removal" in manips


class TestManipulationOverlapMatrix:
    def test_returns_dataframe(self, sample_df):
        result = manipulation_overlap_matrix(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_manipulation_column(self, sample_df):
        result = manipulation_overlap_matrix(sample_df)
        assert "manipulation" in result.columns


class TestEvaluationContextFrequency:
    def test_returns_dict(self, sample_df):
        result = evaluation_context_frequency(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = evaluation_context_frequency(sample_df)
        assert "total_tokens" in result
        assert "categories" in result

    def test_categories_have_count(self, sample_df):
        result = evaluation_context_frequency(sample_df)
        categories = result.get("categories", {})
        for name, data in categories.items():
            assert "count" in data
            assert "is_rare" in data


# === Phase 3 Verbal Morphology ===


class TestVerbalMorphologyParadigm:
    def test_returns_dict(self, sample_df):
        result = verbal_morphology_paradigm(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = verbal_morphology_paradigm(sample_df)
        assert "total_finite_verbs" in result
        assert "paradigm_cells" in result


class TestPersonNumberTenseDistribution:
    def test_returns_dataframe(self, sample_df):
        result = person_number_tense_distribution(sample_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_columns_if_not_empty(self, sample_df):
        result = person_number_tense_distribution(sample_df)
        if not result.is_empty():
            assert "person" in result.columns
            assert "number" in result.columns
            assert "count" in result.columns


class TestMorphologyManipulationCoverage:
    def test_returns_dict(self, sample_df):
        result = morphology_manipulation_coverage(sample_df)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sample_df):
        result = morphology_manipulation_coverage(sample_df)
        assert "total_finite_verbs" in result
        assert "english_enrichment_coverage" in result
        assert "italian_impoverishment_coverage" in result


# === Run All ===


class TestRunAll:
    def test_run_all(self, sample_df):
        result = run_all_analyses(sample_df)
        assert isinstance(result, dict)
        assert len(result) > 10

    def test_run_all_with_output(self, sample_df, tmp_path):
        output_dir = tmp_path / "analysis_output"
        result = run_all_analyses(sample_df, output_dir=output_dir)

        assert output_dir.exists()
        combined = output_dir / "corpus_analysis_results.json"
        assert combined.exists()

        with open(combined) as f:
            data = json.load(f)
        assert "corpus_overview" in data

    def test_run_all_on_lazy(self, sample_df):
        result = run_all_analyses(sample_df.lazy())
        assert isinstance(result, dict)
        assert len(result) > 10
