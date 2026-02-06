#!/usr/bin/env python
"""
Run corpus descriptive analyses on annotated Parquet data.

Usage
-----
    # Single Parquet file (non-layered output)
    python analysis/scripts/run_corpus_descriptives.py \\
        data/annotated_corpus/test_10M.parquet \\
        --output analysis/output/corpus_descriptives/data/test_10M

    # Layered directory
    python analysis/scripts/run_corpus_descriptives.py \\
        data/annotated_corpus/ \\
        --layered \\
        --output analysis/output/corpus_descriptives/data/train_90M

    # Specific analyses only
    python analysis/scripts/run_corpus_descriptives.py \\
        data/annotated_corpus/test_10M.parquet \\
        --analyses overview phenomenon_rates childes \\
        --output analysis/output/corpus_descriptives/data/test_10M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl


def _load_data(path: Path, layered: bool) -> pl.LazyFrame:
    """Load annotations from file or layered directory."""
    if layered:
        from analysis.corpus_descriptives.query import load_layered_corpus
        return load_layered_corpus(path)
    elif path.is_file():
        return pl.scan_parquet(path)
    elif path.is_dir():
        return pl.scan_parquet(str(path / "**" / "*.parquet"))
    else:
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def _print_df(df: pl.DataFrame, max_rows: int = 30) -> None:
    with pl.Config(tbl_rows=max_rows, tbl_cols=20, fmt_str_lengths=40):
        print(df)


def _print_dict(d: dict, indent: int = 2) -> None:
    print(json.dumps(d, indent=indent, default=str))


def run_overview(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import corpus_overview
    _print_section("1. Corpus Overview")
    result = corpus_overview(lf)
    _print_df(result)
    return {"corpus_overview": result.to_dicts()}


def run_phenomenon_rates(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import phenomenon_rates_by_genre
    _print_section("2. Phenomenon Rates by Genre")
    result = phenomenon_rates_by_genre(lf)
    _print_df(result)
    return {"phenomenon_rates_by_genre": result.to_dicts()}


def run_cooccurrence(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        cooccurrence_null_subject_that_trace,
        cooccurrence_negation_null_subject,
        cooccurrence_expletive_depth,
    )
    results = {}

    _print_section("3a. Co-occurrence: Null Subject <-> That-Trace")
    r = cooccurrence_null_subject_that_trace(lf)
    _print_dict(r)
    results["cooccurrence_null_subject_that_trace"] = r

    _print_section("3b. Co-occurrence: Negation <-> Null Subject")
    r = cooccurrence_negation_null_subject(lf)
    _print_df(r)
    results["cooccurrence_negation_null_subject"] = r.to_dicts()

    _print_section("3c. Co-occurrence: Expletive <-> Embedding Depth")
    r = cooccurrence_expletive_depth(lf)
    _print_dict(r)
    results["cooccurrence_expletive_depth"] = r

    return results


def run_childes(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        childes_child_vs_adult,
        childes_speaker_distribution,
        childes_pronoun_distribution,
        childes_person_distribution,
    )
    results = {}

    _print_section("4a. CHILDES: Child vs Adult")
    r = childes_child_vs_adult(lf)
    _print_df(r)
    results["childes_child_vs_adult"] = r.to_dicts()

    _print_section("4b. CHILDES: Speaker Distribution")
    r = childes_speaker_distribution(lf)
    _print_df(r)
    results["childes_speaker_distribution"] = r.to_dicts()

    _print_section("4c. CHILDES: Pronoun Distribution by Role")
    r = childes_pronoun_distribution(lf)
    _print_df(r, max_rows=40)
    results["childes_pronoun_distribution"] = r.to_dicts()

    _print_section("4d. CHILDES: Person Distribution by Role")
    r = childes_person_distribution(lf)
    _print_df(r)
    results["childes_person_distribution"] = r.to_dicts()

    return results


def run_pronouns(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        subject_pronoun_distribution,
    )
    _print_section("5. Subject Pronoun Distribution")
    r = subject_pronoun_distribution(lf)
    _print_df(r, max_rows=40)
    return {"subject_pronoun_distribution": r.to_dicts()}


def run_that_trace(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import that_trace_inventory
    _print_section("6. That-Trace Environment Inventory")
    r = that_trace_inventory(lf)
    _print_dict(r)
    return {"that_trace_inventory": r}


def run_expletives(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        expletive_inventory,
        expletive_by_class,
    )
    results = {}

    _print_section("7a. Expletive Inventory (full)")
    r = expletive_inventory(lf)
    _print_df(r, max_rows=40)
    results["expletive_inventory"] = r.to_dicts()

    _print_section("7b. Expletive by Class")
    r = expletive_by_class(lf)
    _print_df(r)
    results["expletive_by_class"] = r.to_dicts()

    return results


def run_complexity(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        complexity_distribution,
        depth_distribution,
    )
    results = {}

    _print_section("8a. Complexity Distribution by Genre")
    r = complexity_distribution(lf)
    _print_df(r)
    results["complexity_distribution"] = r.to_dicts()

    _print_section("8b. Embedding Depth Distribution")
    r = depth_distribution(lf)
    _print_df(r, max_rows=50)
    results["depth_distribution"] = r.to_dicts()

    return results


def run_verbs(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        verb_null_subject_rate,
        bridge_verb_null_rates,
        weather_verb_null_rates,
        tense_aspect_mood_distribution,
    )
    results = {}

    _print_section("9a. Verbs with Highest Null-Subject Rate")
    r = verb_null_subject_rate(lf)
    _print_df(r.head(30))
    results["verb_null_subject_rate_top30"] = r.head(30).to_dicts()

    _print_section("9b. Bridge Verb Null-Subject Rates")
    r = bridge_verb_null_rates(lf)
    _print_df(r)
    results["bridge_verb_null_rates"] = r.to_dicts()

    _print_section("9c. Weather Verb Null-Subject Rates")
    r = weather_verb_null_rates(lf)
    _print_df(r)
    results["weather_verb_null_rates"] = r.to_dicts()

    _print_section("9d. Tense/Aspect/Mood Distribution")
    tam = tense_aspect_mood_distribution(lf)
    for key, frame in tam.items():
        print(f"\n--- {key} ---")
        _print_df(frame)
        results[f"tam_{key}"] = frame.to_dicts()

    return results


def run_register(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        register_comparison,
        register_by_genre,
    )
    results = {}

    _print_section("10a. Register Comparison (Spoken vs Written)")
    r = register_comparison(lf)
    _print_df(r)
    results["register_comparison"] = r.to_dicts()

    _print_section("10b. Register × Genre Detail")
    r = register_by_genre(lf)
    _print_df(r)
    results["register_by_genre"] = r.to_dicts()

    return results


def run_hypotheses(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        h5_distal_effects,
        h6a_expletive_analysis,
        null_subject_by_embedding_depth,
    )
    results = {}

    _print_section("H5: Distal Effects (That-Trace <-> Null Subjects)")
    r = h5_distal_effects(lf)
    _print_dict(r)
    results["h5_distal_effects"] = r

    _print_section("H6a: Expletive Analysis")
    r = h6a_expletive_analysis(lf)
    _print_dict(r)
    results["h6a_expletive_analysis"] = r

    _print_section("Null Subject Rate by Embedding Depth")
    r = null_subject_by_embedding_depth(lf)
    _print_df(r)
    results["null_subject_by_depth"] = r.to_dicts()

    return results


def run_subject_pronoun_rate(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import subject_pronoun_rate
    _print_section("Subject Pronoun Rate (§2.1)")
    r = subject_pronoun_rate(lf)
    _print_dict(r)
    return {"subject_pronoun_rate": r}


def run_negation_position(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import negation_position_distribution
    _print_section("Negation Position Distribution (§8.2)")
    r = negation_position_distribution(lf)
    _print_dict(r)
    return {"negation_position_distribution": r}


def run_wh_questions(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import wh_question_distribution
    _print_section("Wh-Question Distribution (§10)")
    r = wh_question_distribution(lf)
    _print_dict(r)
    return {"wh_question_distribution": r}


def run_clause_crosstab(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import clause_structure_crosstab
    _print_section("Clause Structure Cross-tabulation (§1.3)")
    r = clause_structure_crosstab(lf)
    _print_df(r, max_rows=50)
    return {"clause_structure_crosstab": r.to_dicts()}


def run_case_pronouns(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import case_marked_pronoun_analysis
    _print_section("Case-Marked Pronoun Analysis (§2.3)")
    r = case_marked_pronoun_analysis(lf)
    _print_dict(r)
    return {"case_marked_pronoun_analysis": r}


def run_that_trace_crosstab(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import that_trace_crosstab
    _print_section("That-Trace Cross-tabulation (§4.4)")
    r = that_trace_crosstab(lf)
    _print_df(r, max_rows=50)
    return {"that_trace_crosstab": r.to_dicts()}


def run_manipulation(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        manipulation_coverage,
        manipulation_overlap_matrix,
    )
    results = {}

    _print_section("Manipulation Coverage (§7)")
    r = manipulation_coverage(lf)
    _print_dict(r)
    results["manipulation_coverage"] = r

    _print_section("Manipulation Overlap Matrix (§7)")
    r = manipulation_overlap_matrix(lf)
    _print_df(r)
    results["manipulation_overlap_matrix"] = r.to_dicts()

    return results


def run_evaluation_contexts(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import evaluation_context_frequency
    _print_section("Evaluation Context Frequency (§9)")
    r = evaluation_context_frequency(lf)
    _print_dict(r)
    return {"evaluation_context_frequency": r}


def run_morphology(lf: pl.LazyFrame) -> dict:
    from analysis.corpus_descriptives.corpus_analysis import (
        verbal_morphology_paradigm,
        person_number_tense_distribution,
        morphology_manipulation_coverage,
    )
    results = {}

    _print_section("Verbal Morphology Paradigm (§6.1)")
    r = verbal_morphology_paradigm(lf)
    _print_dict(r)
    results["verbal_morphology_paradigm"] = r

    _print_section("Person × Number × Tense Distribution (§6.2)")
    r = person_number_tense_distribution(lf)
    _print_df(r, max_rows=50)
    results["person_number_tense_distribution"] = r.to_dicts()

    _print_section("Morphology Manipulation Coverage (§6.3)")
    r = morphology_manipulation_coverage(lf)
    _print_dict(r)
    results["morphology_manipulation_coverage"] = r

    return results


ANALYSIS_RUNNERS = {
    "overview": run_overview,
    "phenomenon_rates": run_phenomenon_rates,
    "cooccurrence": run_cooccurrence,
    "childes": run_childes,
    "pronouns": run_pronouns,
    "that_trace": run_that_trace,
    "expletives": run_expletives,
    "complexity": run_complexity,
    "verbs": run_verbs,
    "register": run_register,
    "hypotheses": run_hypotheses,
    "subject_pronoun_rate": run_subject_pronoun_rate,
    "negation_position": run_negation_position,
    "wh_questions": run_wh_questions,
    "clause_crosstab": run_clause_crosstab,
    "case_pronouns": run_case_pronouns,
    "that_trace_crosstab": run_that_trace_crosstab,
    "manipulation": run_manipulation,
    "evaluation_contexts": run_evaluation_contexts,
    "morphology": run_morphology,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run corpus descriptive analyses on annotated Parquet data.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to Parquet file or annotated corpus directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for JSON results (default: print only)",
    )
    parser.add_argument(
        "--layered",
        action="store_true",
        help="Input is a layered corpus directory (base/ + layers/)",
    )
    parser.add_argument(
        "--analyses",
        nargs="*",
        choices=list(ANALYSIS_RUNNERS.keys()),
        default=None,
        help="Specific analyses to run (default: all)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading annotations from: {args.input_path}")
    lf = _load_data(args.input_path, args.layered)

    # Quick schema check
    print(f"Columns available: {lf.columns}")

    # Run analyses
    analyses = args.analyses or list(ANALYSIS_RUNNERS.keys())
    all_results = {}

    for name in analyses:
        runner = ANALYSIS_RUNNERS[name]
        try:
            results = runner(lf)
            all_results.update(results)
        except Exception as e:
            print(f"\nWARNING: Analysis '{name}' failed: {e}", file=sys.stderr)

    # Write combined output
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

        for key, data in all_results.items():
            path = args.output / f"{key}.json"
            path.write_text(json.dumps(data, indent=2, default=str))

        combined = args.output / "corpus_analysis_results.json"
        combined.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
