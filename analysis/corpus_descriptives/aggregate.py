"""
Aggregation utilities for computing summary statistics from Parquet annotations.

Replicates the output of original analyzers by computing aggregates from
the sentence-level annotation files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl


def load_annotations(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """
    Load annotations from Parquet file(s) as a lazy frame.

    Args:
        path: Path to Parquet file or directory with partitioned data
        columns: Optional list of columns to load (for efficiency)

    Returns:
        Polars LazyFrame for lazy evaluation
    """
    path = Path(path)

    if path.is_file():
        return pl.scan_parquet(path, allow_missing_columns=True)
    elif path.is_dir():
        # Scan all parquet files in directory
        pattern = str(path / "**" / "*.parquet")
        return pl.scan_parquet(pattern, allow_missing_columns=True)
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")


def compute_clause_structure_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute clause structure aggregates matching original analyzer output.

    Returns dict with "overall" and "by_genre" keys.
    """
    # Explode clauses and aggregate
    clauses_df = (
        df.select(["genre", "clauses"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .unnest("clauses")
        .collect()
    )

    if clauses_df.is_empty():
        return {"overall": {"finite_clauses": []}, "by_genre": {}}

    # Finite clauses aggregation
    finite_df = clauses_df.filter(pl.col("is_finite") == True)

    overall_finite = (
        finite_df
        .group_by(["clause_type", "subject_status"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    by_genre = {}
    for genre in clauses_df["genre"].unique().to_list():
        genre_finite = (
            finite_df
            .filter(pl.col("genre") == genre)
            .group_by(["clause_type", "subject_status"])
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
            .to_dicts()
        )
        by_genre[genre] = {
            "finite_clauses": [
                {"clause_dep": r["clause_type"], "subject_status": r["subject_status"], "count": r["count"]}
                for r in genre_finite
            ]
        }

    return {
        "overall": {
            "finite_clauses": [
                {"clause_dep": r["clause_type"], "subject_status": r["subject_status"], "count": r["count"]}
                for r in overall_finite
            ]
        },
        "by_genre": by_genre,
    }


def compute_that_trace_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute that-trace aggregates matching original analyzer output.
    """
    complements_df = (
        df.select(["genre", "bridge_complements"])
        .explode("bridge_complements")
        .filter(pl.col("bridge_complements").is_not_null())
        .unnest("bridge_complements")
        .collect()
    )

    if complements_df.is_empty():
        return {
            "overall": {
                "bridge_verb_counts": [],
                "complementizer_rate": {"with_that": 0, "without_that": 0},
                "cross_tab": [],
            },
            "by_genre": {},
        }

    # Bridge verb counts
    overall_verbs = (
        complements_df
        .group_by("matrix_verb_lemma")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    # Complementizer rate
    comp_counts = complements_df.group_by("comp_present").agg(pl.count().alias("count")).to_dicts()
    comp_rate = {"with_that": 0, "without_that": 0}
    for r in comp_counts:
        if r["comp_present"]:
            comp_rate["with_that"] = r["count"]
        else:
            comp_rate["without_that"] = r["count"]

    # Cross-tab: comp_present × extraction_type
    cross_tab = (
        complements_df
        .group_by(["comp_present", "extraction_type"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    # Violations count
    violations = complements_df.filter(pl.col("is_violation") == True).height

    return {
        "overall": {
            "bridge_verb_counts": [
                {"verb_lemma": r["matrix_verb_lemma"], "count": r["count"]}
                for r in overall_verbs
            ],
            "complementizer_rate": comp_rate,
            "cross_tab": [
                {"comp_present": r["comp_present"], "extraction_type": r["extraction_type"], "count": r["count"]}
                for r in cross_tab
            ],
            "violations_count": violations,
        },
        "by_genre": {},  # Can be extended similarly
    }


def compute_pronoun_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute pronoun inventory aggregates.
    """
    pronouns_df = (
        df.select(["genre", "pronouns", "n_tokens"])
        .explode("pronouns")
        .filter(pl.col("pronouns").is_not_null())
        .unnest("pronouns")
        .collect()
    )

    if pronouns_df.is_empty():
        return {"overall": {"entries": [], "total_pronouns": 0}, "by_genre": {}}

    # Aggregate by function, person, number, case, lemma
    overall = (
        pronouns_df
        .group_by(["function", "person", "number", "case", "lemma"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    total_pronouns = pronouns_df.height

    return {
        "overall": {
            "entries": [
                {
                    "function": r["function"],
                    "person": r["person"],
                    "number": r["number"],
                    "case": r["case"],
                    "lemma": r["lemma"],
                    "count": r["count"],
                }
                for r in overall
            ],
            "total_pronouns": total_pronouns,
        },
        "by_genre": {},
    }


def compute_negation_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute negation aggregates.
    """
    negations_df = (
        df.select(["genre", "negations"])
        .explode("negations")
        .filter(pl.col("negations").is_not_null())
        .unnest("negations")
        .collect()
    )

    if negations_df.is_empty():
        return {"overall": {"position_counts": {}, "total_negations": 0}, "by_genre": {}}

    # Position counts
    position_counts = (
        negations_df
        .group_by("position")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    return {
        "overall": {
            "position_counts": {r["position"]: r["count"] for r in position_counts},
            "total_negations": negations_df.height,
        },
        "by_genre": {},
    }


def compute_expletive_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute expletive aggregates.
    """
    expletives_df = (
        df.select(["genre", "expletives"])
        .explode("expletives")
        .filter(pl.col("expletives").is_not_null())
        .unnest("expletives")
        .collect()
    )

    if expletives_df.is_empty():
        return {"overall": {"class_counts": {}, "verb_counts": {}, "total_expletives": 0}, "by_genre": {}}

    # Class counts
    class_counts = (
        expletives_df
        .group_by("expletive_class")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Verb counts
    verb_counts = (
        expletives_df
        .group_by("verb_lemma")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(20)
        .to_dicts()
    )

    return {
        "overall": {
            "class_counts": {r["expletive_class"]: r["count"] for r in class_counts},
            "verb_counts": [{"verb_lemma": r["verb_lemma"], "count": r["count"]} for r in verb_counts],
            "total_expletives": expletives_df.height,
        },
        "by_genre": {},
    }


def compute_verb_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute verb aggregates including tense/aspect/mood distributions.
    """
    verbs_df = (
        df.select(["genre", "verbs"])
        .explode("verbs")
        .filter(pl.col("verbs").is_not_null())
        .unnest("verbs")
        .collect()
    )

    if verbs_df.is_empty():
        return {"overall": {"verb_form_counts": {}, "tense_counts": {}, "total_verbs": 0}, "by_genre": {}}

    # Verb form distribution
    verb_form_counts = (
        verbs_df
        .group_by("verb_form")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Tense distribution
    tense_counts = (
        verbs_df
        .filter(pl.col("tense").is_not_null())
        .group_by("tense")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Aspect distribution
    aspect_counts = (
        verbs_df
        .filter(pl.col("aspect").is_not_null())
        .group_by("aspect")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Mood distribution
    mood_counts = (
        verbs_df
        .filter(pl.col("mood").is_not_null())
        .group_by("mood")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    return {
        "overall": {
            "verb_form_counts": {r["verb_form"]: r["count"] for r in verb_form_counts},
            "tense_counts": {r["tense"]: r["count"] for r in tense_counts},
            "aspect_counts": {r["aspect"]: r["count"] for r in aspect_counts},
            "mood_counts": {r["mood"]: r["count"] for r in mood_counts},
            "total_verbs": verbs_df.height,
        },
        "by_genre": {},
    }


def compute_wh_extraction_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute wh-extraction aggregates.
    """
    wh_df = (
        df.select(["genre", "wh_extractions"])
        .explode("wh_extractions")
        .filter(pl.col("wh_extractions").is_not_null())
        .unnest("wh_extractions")
        .collect()
    )

    if wh_df.is_empty():
        return {"overall": {"extraction_type_counts": {}, "wh_lemma_counts": {}, "total_extractions": 0}, "by_genre": {}}

    # Extraction type distribution
    extraction_types = (
        wh_df
        .group_by("extraction_type")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Wh-lemma distribution
    wh_lemmas = (
        wh_df
        .group_by("wh_lemma")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    # Clause level distribution
    clause_levels = (
        wh_df
        .group_by("clause_level")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    return {
        "overall": {
            "extraction_type_counts": {r["extraction_type"]: r["count"] for r in extraction_types},
            "wh_lemma_counts": [{"wh_lemma": r["wh_lemma"], "count": r["count"]} for r in wh_lemmas],
            "clause_level_counts": {r["clause_level"]: r["count"] for r in clause_levels},
            "total_extractions": wh_df.height,
        },
        "by_genre": {},
    }


def compute_relative_clause_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute relative clause aggregates.
    """
    relcl_df = (
        df.select(["genre", "relative_clauses"])
        .explode("relative_clauses")
        .filter(pl.col("relative_clauses").is_not_null())
        .unnest("relative_clauses")
        .collect()
    )

    if relcl_df.is_empty():
        return {"overall": {"rel_type_counts": {}, "total_relative_clauses": 0}, "by_genre": {}}

    # Relative type distribution (subject vs object)
    rel_types = (
        relcl_df
        .group_by("rel_type")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Relativizer distribution
    relativizers = (
        relcl_df
        .filter(pl.col("relativizer_lemma").is_not_null())
        .group_by("relativizer_lemma")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    return {
        "overall": {
            "rel_type_counts": {r["rel_type"]: r["count"] for r in rel_types},
            "relativizer_counts": [{"relativizer": r["relativizer_lemma"], "count": r["count"]} for r in relativizers],
            "total_relative_clauses": relcl_df.height,
        },
        "by_genre": {},
    }


def compute_argument_structure_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute argument structure aggregates.
    """
    arg_df = (
        df.select(["genre", "argument_structure"])
        .explode("argument_structure")
        .filter(pl.col("argument_structure").is_not_null())
        .unnest("argument_structure")
        .collect()
    )

    if arg_df.is_empty():
        return {"overall": {"transitivity_counts": {}, "subject_status_counts": {}, "total_verbs": 0}, "by_genre": {}}

    # Transitivity distribution
    transitivity = (
        arg_df
        .group_by("transitivity")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Subject status distribution
    subject_status = (
        arg_df
        .group_by("subject_status")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    # Object status distribution
    object_status = (
        arg_df
        .group_by("object_status")
        .agg(pl.count().alias("count"))
        .to_dicts()
    )

    return {
        "overall": {
            "transitivity_counts": {r["transitivity"]: r["count"] for r in transitivity},
            "subject_status_counts": {r["subject_status"]: r["count"] for r in subject_status},
            "object_status_counts": {r["object_status"]: r["count"] for r in object_status},
            "total_verbs": arg_df.height,
        },
        "by_genre": {},
    }


def compute_complexity_aggregates(
    df: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Compute complexity aggregates.
    """
    # Extract complexity struct fields
    complexity_df = (
        df.select([
            "genre",
            pl.col("complexity").struct.field("n_clauses").alias("n_clauses"),
            pl.col("complexity").struct.field("max_embedding_depth").alias("max_depth"),
            pl.col("complexity").struct.field("has_coordination").alias("has_coordination"),
            pl.col("complexity").struct.field("n_conjuncts").alias("n_conjuncts"),
            "is_fragment",
            "is_imperative",
        ])
        .collect()
    )

    if complexity_df.is_empty():
        return {"overall": {}, "by_genre": {}}

    # Clause count distribution
    clause_dist = (
        complexity_df
        .group_by("n_clauses")
        .agg(pl.count().alias("count"))
        .sort("n_clauses")
        .to_dicts()
    )

    # Embedding depth distribution
    depth_dist = (
        complexity_df
        .group_by("max_depth")
        .agg(pl.count().alias("count"))
        .sort("max_depth")
        .to_dicts()
    )

    # Fragment and imperative rates
    total = complexity_df.height
    fragment_count = complexity_df.filter(pl.col("is_fragment") == True).height
    imperative_count = complexity_df.filter(pl.col("is_imperative") == True).height
    coordination_count = complexity_df.filter(pl.col("has_coordination") == True).height

    return {
        "overall": {
            "clause_count_distribution": [{"n_clauses": r["n_clauses"], "count": r["count"]} for r in clause_dist],
            "depth_distribution": [{"depth": r["max_depth"], "count": r["count"]} for r in depth_dist],
            "mean_clauses": complexity_df["n_clauses"].mean(),
            "mean_depth": complexity_df["max_depth"].mean(),
            "fragment_count": fragment_count,
            "fragment_rate": fragment_count / total if total else 0,
            "imperative_count": imperative_count,
            "imperative_rate": imperative_count / total if total else 0,
            "coordination_rate": coordination_count / total if total else 0,
            "total_sentences": total,
        },
        "by_genre": {},
    }


def compute_all_aggregates(
    annotations_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute all aggregates from annotation files.

    Args:
        annotations_path: Path to annotations directory or Parquet file
        output_path: Optional path to write aggregate JSON files

    Returns:
        Dict mapping analyzer names to their aggregate results
    """
    df = load_annotations(annotations_path)

    results = {
        "clause_structure": compute_clause_structure_aggregates(df),
        "that_trace": compute_that_trace_aggregates(df),
        "pronoun_inventory": compute_pronoun_aggregates(df),
        "negation": compute_negation_aggregates(df),
        "expletive": compute_expletive_aggregates(df),
        "verb": compute_verb_aggregates(df),
        "wh_extraction": compute_wh_extraction_aggregates(df),
        "relative_clause": compute_relative_clause_aggregates(df),
        "argument_structure": compute_argument_structure_aggregates(df),
        "complexity": compute_complexity_aggregates(df),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, data in results.items():
            json_path = output_path / f"{name}.json"
            json_path.write_text(json.dumps(data, indent=2, default=str))

        # Combined results
        combined_path = output_path / "results.json"
        combined_path.write_text(json.dumps(results, indent=2, default=str))

    return results


def compute_summary_statistics(
    annotations_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Compute high-level summary statistics from annotations.

    Returns counts and proportions for key phenomena.
    """
    df = load_annotations(annotations_path)

    # Collect basic stats
    stats_df = df.select([
        pl.count().alias("total_sentences"),
        pl.col("n_tokens").sum().alias("total_tokens"),
        pl.col("has_null_subject").sum().alias("null_subject_count"),
        pl.col("has_expletive").sum().alias("expletive_count"),
        pl.col("has_that_trace_violation").sum().alias("that_trace_violation_count"),
        pl.col("has_wh_extraction").sum().alias("wh_extraction_count"),
        pl.col("has_relative_clause").sum().alias("relative_clause_count"),
        pl.col("has_negation").sum().alias("negation_count"),
        pl.col("is_fragment").sum().alias("fragment_count"),
        pl.col("is_imperative").sum().alias("imperative_count"),
    ]).collect()

    row = stats_df.to_dicts()[0]
    total = row["total_sentences"]

    return {
        "total_sentences": total,
        "total_tokens": row["total_tokens"],
        "null_subject": {
            "count": row["null_subject_count"],
            "proportion": row["null_subject_count"] / total if total else 0,
        },
        "expletive": {
            "count": row["expletive_count"],
            "proportion": row["expletive_count"] / total if total else 0,
        },
        "that_trace_violation": {
            "count": row["that_trace_violation_count"],
            "proportion": row["that_trace_violation_count"] / total if total else 0,
        },
        "wh_extraction": {
            "count": row["wh_extraction_count"],
            "proportion": row["wh_extraction_count"] / total if total else 0,
        },
        "relative_clause": {
            "count": row["relative_clause_count"],
            "proportion": row["relative_clause_count"] / total if total else 0,
        },
        "negation": {
            "count": row["negation_count"],
            "proportion": row["negation_count"] / total if total else 0,
        },
        "fragment": {
            "count": row["fragment_count"],
            "proportion": row["fragment_count"] / total if total else 0,
        },
        "imperative": {
            "count": row["imperative_count"],
            "proportion": row["imperative_count"] / total if total else 0,
        },
    }
