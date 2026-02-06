"""
Corpus descriptive analyses on layered sentence-level annotations.

Implements the analysis plan for characterising the training corpus before
ablation experiments.  Every public function accepts a polars LazyFrame
(or an AnnotatedCorpus) and returns a polars DataFrame or a plain dict
suitable for JSON serialisation.

Sections
--------
1.  Corpus overview
2.  Phenomenon rates by genre
3.  Targeted co-occurrence analyses
4.  CHILDES developmental analysis
5.  Pronoun distribution
6.  That-trace environment inventory
7.  Expletive inventory
8.  Sentence complexity distribution
9.  Verb lemma analysis
10. Cross-genre comparison (spoken vs written)
11. Language-agnostic / Italian comparison hooks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from .constants import (
    ENGLISH_BRIDGE_VERBS,
    ITALIAN_BRIDGE_VERBS,
    WEATHER_VERBS,
    WH_LEMMAS_EN,
    WH_LEMMAS_IT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_lazy(df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return df


SPOKEN_GENRES = frozenset({
    "CHILDES", "CHILDES_child", "CHILDES_adult",
    "BNC", "Switchboard",
})
WRITTEN_GENRES = frozenset({
    "Gutenberg", "SimpleWikipedia", "OpenSubtitles",
})

ALL_FLAG_COLUMNS = [
    "has_null_subject",
    "has_null_subject_finite",
    "has_null_subject_nonfinite",
    "has_null_object",
    "has_overt_subject_pronoun",
    "has_overt_object_pronoun",
    "has_expletive",
    "has_that_trace_context",
    "has_that_trace_violation",
    "has_wh_extraction",
    "has_relative_clause",
    "has_negation",
    "has_ditransitive",
    "is_imperative",
    "is_fragment",
]


# ---------------------------------------------------------------------------
# 1. Corpus overview
# ---------------------------------------------------------------------------

def corpus_overview(df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
    """Basic sentence/token counts by genre and speaker role."""
    lf = _ensure_lazy(df)
    return (
        lf
        .group_by(["genre", "role"])
        .agg([
            pl.count().alias("n_sentences"),
            pl.col("n_tokens").sum().alias("total_tokens"),
            pl.col("n_tokens").mean().alias("mean_sent_length"),
        ])
        .sort(["genre", "role"])
        .collect()
    )


# ---------------------------------------------------------------------------
# 2. Phenomenon rates by genre
# ---------------------------------------------------------------------------

def phenomenon_rates_by_genre(
    df: Union[pl.DataFrame, pl.LazyFrame],
    phenomena: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Compute the proportion of sentences exhibiting each boolean flag, per genre."""
    lf = _ensure_lazy(df)
    if phenomena is None:
        phenomena = ALL_FLAG_COLUMNS

    # Only use columns that actually exist in the data
    available = set(lf.collect_schema().names())
    phenomena = [p for p in phenomena if p in available]

    aggs = [pl.col(p).mean().alias(f"{p}_rate") for p in phenomena]
    aggs.append(pl.count().alias("n_sentences"))

    return (
        lf
        .group_by("genre")
        .agg(aggs)
        .sort("genre")
        .collect()
    )


# ---------------------------------------------------------------------------
# 3. Targeted co-occurrence analyses
# ---------------------------------------------------------------------------

def cooccurrence_null_subject_that_trace(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §3a – Do null subjects co-occur with that-trace contexts?

    Returns baseline null-subject rate, rate within that-trace contexts,
    and enrichment ratio.
    """
    lf = _ensure_lazy(df)

    baseline_null = (
        lf.select(pl.col("has_null_subject").mean().alias("rate"))
        .collect().item()
    )

    tt = lf.filter(pl.col("has_that_trace_context") == True)
    tt_total = tt.select(pl.count()).collect().item()
    if tt_total == 0:
        return {
            "baseline_null_subject_rate": baseline_null,
            "that_trace_total": 0,
            "that_trace_with_null_subject": 0,
            "null_rate_in_that_trace": None,
            "enrichment_ratio": None,
        }

    tt_with_null = (
        tt.filter(pl.col("has_null_subject") == True)
        .select(pl.count())
        .collect().item()
    )
    null_rate_in_tt = tt_with_null / tt_total
    enrichment = null_rate_in_tt / baseline_null if baseline_null > 0 else None

    return {
        "baseline_null_subject_rate": baseline_null,
        "that_trace_total": tt_total,
        "that_trace_with_null_subject": tt_with_null,
        "null_rate_in_that_trace": null_rate_in_tt,
        "enrichment_ratio": enrichment,
    }


def cooccurrence_negation_null_subject(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §3b – Does negation affect null-subject rate?

    Returns null-subject rate in negated vs non-negated sentences, overall
    and by genre.
    """
    lf = _ensure_lazy(df)

    col_order = ["genre", "has_negation", "null_subject_rate", "n_sentences"]

    overall = (
        lf.group_by("has_negation")
        .agg([
            pl.col("has_null_subject").mean().alias("null_subject_rate"),
            pl.count().alias("n_sentences"),
        ])
        .with_columns(pl.lit("ALL").alias("genre"))
        .select(col_order)
    )

    by_genre = (
        lf.group_by(["genre", "has_negation"])
        .agg([
            pl.col("has_null_subject").mean().alias("null_subject_rate"),
            pl.count().alias("n_sentences"),
        ])
        .select(col_order)
    )

    return pl.concat([overall, by_genre]).sort(["genre", "has_negation"]).collect()


def cooccurrence_expletive_depth(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §3c – Are expletives simple surface constructions or also deeply embedded?

    Returns mean depth for expletive vs non-expletive sentences and
    depth distribution for expletive sentences.
    """
    lf = _ensure_lazy(df)

    depth_col = pl.col("complexity").struct.field("max_embedding_depth")

    expletive_depth = (
        lf.filter(pl.col("has_expletive") == True)
        .select(depth_col.alias("depth"))
        .collect()
    )
    non_expletive_depth = (
        lf.filter(pl.col("has_expletive") == False)
        .select(depth_col.mean().alias("mean_depth"))
        .collect().item()
    )

    if expletive_depth.is_empty():
        return {
            "expletive_mean_depth": None,
            "non_expletive_mean_depth": non_expletive_depth,
            "expletive_depth_distribution": [],
        }

    dist = (
        expletive_depth
        .group_by("depth")
        .agg(pl.count().alias("count"))
        .sort("depth")
        .to_dicts()
    )

    return {
        "expletive_mean_depth": expletive_depth["depth"].mean(),
        "non_expletive_mean_depth": non_expletive_depth,
        "expletive_depth_distribution": dist,
    }


# ---------------------------------------------------------------------------
# 4. CHILDES developmental analysis
# ---------------------------------------------------------------------------

def childes_child_vs_adult(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §4a – Core comparison: child (CHI) vs adult (MOT, FAT, …) production.

    Returns key phenomenon rates and complexity metrics by role.
    """
    lf = _ensure_lazy(df)
    childes = lf.filter(
        pl.col("genre").str.starts_with("CHILDES")
    )

    available = set(childes.columns)
    aggs: list = [
        pl.count().alias("n_sentences"),
        pl.col("n_tokens").sum().alias("total_tokens"),
        pl.col("n_tokens").mean().alias("mean_sent_length"),
    ]

    for flag in [
        "has_null_subject", "has_overt_subject_pronoun",
        "has_expletive", "is_fragment", "is_imperative",
    ]:
        if flag in available:
            aggs.append(pl.col(flag).mean().alias(f"{flag}_rate"))

    if "complexity" in available:
        aggs.extend([
            pl.col("complexity").struct.field("n_clauses").mean().alias("mean_clauses"),
            pl.col("complexity").struct.field("max_embedding_depth").mean().alias("mean_depth"),
        ])

    return (
        childes
        .group_by("role")
        .agg(aggs)
        .sort("role")
        .collect()
    )


def childes_speaker_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §4b – Distribution of speaker codes in CHILDES data.

    Also reports unique file names for checking whether age metadata is encoded.
    """
    lf = _ensure_lazy(df)
    childes = lf.filter(pl.col("genre").str.starts_with("CHILDES"))

    return (
        childes
        .group_by("speaker")
        .agg([
            pl.count().alias("n_sentences"),
            pl.col("n_tokens").sum().alias("total_tokens"),
        ])
        .sort("n_sentences", descending=True)
        .collect()
    )


def childes_pronoun_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §4c – Pronoun distribution in child vs adult CHILDES speech.

    Returns top pronoun lemmas by role with person breakdown.
    """
    lf = _ensure_lazy(df)
    childes = lf.filter(pl.col("genre").str.starts_with("CHILDES"))

    return (
        childes
        .select(["role", "pronouns"])
        .explode("pronouns")
        .filter(pl.col("pronouns").is_not_null())
        .select([
            "role",
            pl.col("pronouns").struct.field("lemma").alias("lemma"),
            pl.col("pronouns").struct.field("person").alias("person"),
            pl.col("pronouns").struct.field("function").alias("function"),
        ])
        .group_by(["role", "lemma", "person", "function"])
        .agg(pl.count().alias("count"))
        .sort(["role", "count"], descending=[False, True])
        .collect()
    )


def childes_person_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Person distribution (1st/2nd/3rd) by speaker role in CHILDES."""
    lf = _ensure_lazy(df)
    childes = lf.filter(pl.col("genre").str.starts_with("CHILDES"))

    raw = (
        childes
        .select(["role", "pronouns"])
        .explode("pronouns")
        .filter(pl.col("pronouns").is_not_null())
        .select([
            "role",
            pl.col("pronouns").struct.field("person").alias("person"),
        ])
        .group_by(["role", "person"])
        .agg(pl.count().alias("count"))
        .collect()
    )

    if raw.is_empty():
        return raw

    return raw.with_columns(
        (pl.col("count") / pl.col("count").sum().over("role")).alias("proportion")
    ).sort(["role", "person"])


# ---------------------------------------------------------------------------
# 5. Pronoun distribution analysis
# ---------------------------------------------------------------------------

def pronoun_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §5 – Full pronoun inventory by genre, function, person, number, lemma.
    """
    lf = _ensure_lazy(df)
    return (
        lf
        .select(["genre", "pronouns"])
        .explode("pronouns")
        .filter(pl.col("pronouns").is_not_null())
        .select([
            "genre",
            pl.col("pronouns").struct.field("function").alias("function"),
            pl.col("pronouns").struct.field("person").alias("person"),
            pl.col("pronouns").struct.field("number").alias("number"),
            pl.col("pronouns").struct.field("lemma").alias("lemma"),
        ])
        .group_by(["genre", "function", "person", "number", "lemma"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
        .collect()
    )


def subject_pronoun_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Subject pronouns specifically — the targets for ablation."""
    lf = _ensure_lazy(df)
    return (
        lf
        .select(["genre", "pronouns"])
        .explode("pronouns")
        .filter(pl.col("pronouns").is_not_null())
        .filter(pl.col("pronouns").struct.field("function") == "subject")
        .select([
            "genre",
            pl.col("pronouns").struct.field("person").alias("person"),
            pl.col("pronouns").struct.field("number").alias("number"),
            pl.col("pronouns").struct.field("lemma").alias("lemma"),
        ])
        .group_by(["genre", "person", "number", "lemma"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
        .collect()
    )


def case_marked_pronoun_analysis(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §2.3 / H6c – Frequency by pronoun × case (nominative vs accusative).

    Count case-ambiguous forms (you, it) and calculate manipulation coverage
    for case neutralization.
    """
    lf = _ensure_lazy(df)

    pronouns = (
        lf
        .select(["genre", "pronouns"])
        .explode("pronouns")
        .filter(pl.col("pronouns").is_not_null())
        .select([
            "genre",
            pl.col("pronouns").struct.field("lemma").alias("lemma"),
            pl.col("pronouns").struct.field("case").alias("case"),
            pl.col("pronouns").struct.field("function").alias("function"),
        ])
        .collect()
    )

    if pronouns.is_empty():
        return {
            "total_pronouns": 0,
            "by_case": [],
            "by_lemma_case": [],
            "case_ambiguous_count": 0,
            "case_marked_count": 0,
            "case_neutralization_coverage": None,
        }

    by_case = (
        pronouns
        .group_by("case")
        .agg(pl.count().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
        .sort("count", descending=True)
        .to_dicts()
    )

    by_lemma_case = (
        pronouns
        .group_by(["lemma", "case"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    # Case-ambiguous forms in English: you, it
    ambiguous_lemmas = {"you", "it"}
    case_ambiguous = pronouns.filter(pl.col("lemma").is_in(ambiguous_lemmas)).height
    case_marked = pronouns.filter(~pl.col("lemma").is_in(ambiguous_lemmas)).height

    total = pronouns.height
    coverage = case_marked / total if total > 0 else None

    return {
        "total_pronouns": total,
        "by_case": by_case,
        "by_lemma_case": by_lemma_case,
        "case_ambiguous_count": case_ambiguous,
        "case_marked_count": case_marked,
        "case_neutralization_coverage": coverage,
    }


def subject_pronoun_rate(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §2.1 – Compute % of finite verbs with pronominal subjects.

    Key statistic for H2 pronoun manipulation. Returns overall rate and
    breakdown by genre.
    """
    lf = _ensure_lazy(df)

    # Count finite verbs from clauses
    finite_verbs = (
        lf
        .select(["genre", "clauses"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .filter(pl.col("clauses").struct.field("is_finite") == True)
        .select([
            "genre",
            pl.col("clauses").struct.field("subject_is_pronoun").alias("subject_is_pronoun"),
        ])
        .collect()
    )

    if finite_verbs.is_empty():
        return {
            "overall_rate": None,
            "total_finite_verbs": 0,
            "total_pronominal_subjects": 0,
            "by_genre": [],
        }

    total_finite = finite_verbs.height
    total_pronominal = finite_verbs.filter(pl.col("subject_is_pronoun") == True).height
    overall_rate = total_pronominal / total_finite if total_finite > 0 else None

    by_genre = (
        finite_verbs
        .group_by("genre")
        .agg([
            pl.count().alias("finite_verbs"),
            (pl.col("subject_is_pronoun") == True).sum().alias("pronominal_subjects"),
        ])
        .with_columns(
            (pl.col("pronominal_subjects") / pl.col("finite_verbs")).alias("rate")
        )
        .sort("genre")
        .to_dicts()
    )

    return {
        "overall_rate": overall_rate,
        "total_finite_verbs": total_finite,
        "total_pronominal_subjects": total_pronominal,
        "by_genre": by_genre,
    }


# ---------------------------------------------------------------------------
# 6. That-trace environment inventory
# ---------------------------------------------------------------------------

def that_trace_inventory(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §6 – Detailed inventory of that-trace environments.

    Returns bridge verb distribution, complementizer rate, extraction types,
    and violation count.
    """
    lf = _ensure_lazy(df)
    tt = lf.filter(pl.col("has_that_trace_context") == True)

    complements = (
        tt
        .select("bridge_complements")
        .explode("bridge_complements")
        .filter(pl.col("bridge_complements").is_not_null())
        .select([
            pl.col("bridge_complements").struct.field("matrix_verb_lemma").alias("matrix_verb_lemma"),
            pl.col("bridge_complements").struct.field("comp_present").alias("comp_present"),
            pl.col("bridge_complements").struct.field("extraction_type").alias("extraction_type"),
            pl.col("bridge_complements").struct.field("is_violation").alias("is_violation"),
        ])
        .collect()
    )

    if complements.is_empty():
        return {
            "total_contexts": 0,
            "bridge_verb_counts": [],
            "complementizer_rate": None,
            "extraction_types": [],
            "violation_count": 0,
        }

    bridge_verbs = (
        complements
        .group_by("matrix_verb_lemma")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    comp_rate = complements["comp_present"].mean()

    extraction_types = (
        complements
        .group_by("extraction_type")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )

    violation_count = complements.filter(pl.col("is_violation") == True).height

    return {
        "total_contexts": complements.height,
        "bridge_verb_counts": bridge_verbs,
        "complementizer_rate": comp_rate,
        "extraction_types": extraction_types,
        "violation_count": violation_count,
    }


def clause_structure_crosstab(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §1.3 – Cross-tab: clause_type × subject_status.

    Reads from existing clauses nested column. Categories:
    - Clause types: ROOT, ccomp, advcl, xcomp, acl:relcl, other
    - Subject status: overt, expletive, none (null)
    """
    lf = _ensure_lazy(df)

    clauses = (
        lf
        .select(["genre", "clauses"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .select([
            "genre",
            pl.col("clauses").struct.field("clause_type").alias("clause_type"),
            pl.col("clauses").struct.field("subject_status").alias("subject_status"),
            pl.col("clauses").struct.field("is_finite").alias("is_finite"),
        ])
        .collect()
    )

    if clauses.is_empty():
        return pl.DataFrame()

    # Create cross-tabulation
    crosstab = (
        clauses
        .group_by(["clause_type", "subject_status", "is_finite"])
        .agg(pl.count().alias("count"))
        .sort(["clause_type", "subject_status", "is_finite"])
    )

    return crosstab


def that_trace_crosstab(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §4.4 – Critical cross-tab: complementizer × extraction_type.

    +Comp/Subject extraction should be ~0 in English (violation).
    Returns the cross-tabulation with genre breakdown.
    """
    lf = _ensure_lazy(df)

    complements = (
        lf
        .filter(pl.col("has_that_trace_context") == True)
        .select(["genre", "bridge_complements"])
        .explode("bridge_complements")
        .filter(pl.col("bridge_complements").is_not_null())
        .select([
            "genre",
            pl.col("bridge_complements").struct.field("comp_present").alias("comp_present"),
            pl.col("bridge_complements").struct.field("extraction_type").alias("extraction_type"),
        ])
        .collect()
    )

    if complements.is_empty():
        return pl.DataFrame()

    # Create cross-tabulation: comp_present × extraction_type × genre
    crosstab = (
        complements
        .group_by(["comp_present", "extraction_type", "genre"])
        .agg(pl.count().alias("count"))
        .sort(["comp_present", "extraction_type", "genre"])
    )

    return crosstab


# ---------------------------------------------------------------------------
# 7. Expletive inventory
# ---------------------------------------------------------------------------

def expletive_inventory(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §7 – Expletive distribution by genre, class, and verb lemma.
    """
    lf = _ensure_lazy(df)
    return (
        lf
        .filter(pl.col("has_expletive") == True)
        .select(["genre", "expletives"])
        .explode("expletives")
        .filter(pl.col("expletives").is_not_null())
        .select([
            "genre",
            pl.col("expletives").struct.field("expletive_class").alias("expletive_class"),
            pl.col("expletives").struct.field("verb_lemma").alias("verb_lemma"),
        ])
        .group_by(["genre", "expletive_class", "verb_lemma"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .collect()
    )


def expletive_by_class(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Expletive counts by genre and class (weather / existential / raising)."""
    lf = _ensure_lazy(df)
    return (
        lf
        .filter(pl.col("has_expletive") == True)
        .select(["genre", "expletives"])
        .explode("expletives")
        .filter(pl.col("expletives").is_not_null())
        .select([
            "genre",
            pl.col("expletives").struct.field("expletive_class").alias("expletive_class"),
        ])
        .group_by(["genre", "expletive_class"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
        .collect()
    )


# ---------------------------------------------------------------------------
# 8. Sentence complexity distribution
# ---------------------------------------------------------------------------

def complexity_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §8 – Clause count, embedding depth, and coordination rate by genre.
    """
    lf = _ensure_lazy(df)
    return (
        lf
        .select([
            "genre",
            pl.col("complexity").struct.field("n_clauses").alias("n_clauses"),
            pl.col("complexity").struct.field("max_embedding_depth").alias("max_depth"),
            pl.col("complexity").struct.field("has_coordination").alias("has_coordination"),
        ])
        .group_by("genre")
        .agg([
            pl.col("n_clauses").mean().alias("mean_clauses"),
            pl.col("max_depth").mean().alias("mean_depth"),
            pl.col("max_depth").max().alias("max_depth_observed"),
            pl.col("has_coordination").mean().alias("coordination_rate"),
            pl.count().alias("n_sentences"),
        ])
        .sort("genre")
        .collect()
    )


def depth_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Embedding depth histogram (count of sentences at each depth)."""
    lf = _ensure_lazy(df)
    return (
        lf
        .select([
            "genre",
            pl.col("complexity").struct.field("max_embedding_depth").alias("depth"),
        ])
        .group_by(["genre", "depth"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "depth"])
        .collect()
    )


def negation_position_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §8.2 – Token position of negation relative to verb (preverbal/postverbal).

    Uses existing negations struct which has position field.
    """
    lf = _ensure_lazy(df)

    negations = (
        lf
        .filter(pl.col("has_negation") == True)
        .select(["genre", "negations"])
        .explode("negations")
        .filter(pl.col("negations").is_not_null())
        .select([
            "genre",
            pl.col("negations").struct.field("position").alias("position"),
            pl.col("negations").struct.field("lemma").alias("lemma"),
        ])
        .collect()
    )

    if negations.is_empty():
        return {
            "total_negations": 0,
            "position_distribution": [],
            "by_genre": [],
        }

    position_dist = (
        negations
        .group_by("position")
        .agg(pl.count().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
        .sort("count", descending=True)
        .to_dicts()
    )

    by_genre = (
        negations
        .group_by(["genre", "position"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
        .to_dicts()
    )

    return {
        "total_negations": negations.height,
        "position_distribution": position_dist,
        "by_genre": by_genre,
    }


# ---------------------------------------------------------------------------
# 9. Verb lemma analysis
# ---------------------------------------------------------------------------

def verb_by_subject_status(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §9 – Which verb lemmas appear with null vs overt subjects?

    Only finite clauses are included.
    """
    lf = _ensure_lazy(df)
    return (
        lf
        .select(["genre", "clauses"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .select([
            "genre",
            pl.col("clauses").struct.field("verb_lemma").alias("verb_lemma"),
            pl.col("clauses").struct.field("subject_status").alias("subject_status"),
            pl.col("clauses").struct.field("is_finite").alias("is_finite"),
        ])
        .filter(pl.col("is_finite") == True)
        .group_by(["genre", "verb_lemma", "subject_status"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
        .collect()
    )


def verb_null_subject_rate(
    df: Union[pl.DataFrame, pl.LazyFrame],
    min_occurrences: int = 10,
) -> pl.DataFrame:
    """
    §9 – Verbs ranked by null-subject rate (finite clauses, minimum frequency filter).
    """
    lf = _ensure_lazy(df)
    return (
        lf
        .select("clauses")
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .select([
            pl.col("clauses").struct.field("verb_lemma").alias("verb_lemma"),
            pl.col("clauses").struct.field("subject_status").alias("subject_status"),
            pl.col("clauses").struct.field("is_finite").alias("is_finite"),
        ])
        .filter(pl.col("is_finite") == True)
        .group_by("verb_lemma")
        .agg([
            (pl.col("subject_status") == "none").sum().alias("null_count"),
            pl.count().alias("total_count"),
        ])
        .filter(pl.col("total_count") >= min_occurrences)
        .with_columns(
            (pl.col("null_count") / pl.col("total_count")).alias("null_rate")
        )
        .sort("null_rate", descending=True)
        .collect()
    )


def bridge_verb_null_rates(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Null-subject rate specifically for bridge verbs."""
    all_rates = verb_null_subject_rate(df, min_occurrences=1)
    return all_rates.filter(
        pl.col("verb_lemma").is_in(list(ENGLISH_BRIDGE_VERBS))
    )


def weather_verb_null_rates(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Null-subject rate specifically for weather verbs."""
    all_rates = verb_null_subject_rate(df, min_occurrences=1)
    return all_rates.filter(
        pl.col("verb_lemma").is_in(list(WEATHER_VERBS))
    )


def tense_aspect_mood_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, pl.DataFrame]:
    """Tense, aspect, and mood distributions across the corpus."""
    lf = _ensure_lazy(df)
    verbs = (
        lf
        .select(["genre", "verbs"])
        .explode("verbs")
        .filter(pl.col("verbs").is_not_null())
        .select([
            "genre",
            pl.col("verbs").struct.field("tense").alias("tense"),
            pl.col("verbs").struct.field("aspect").alias("aspect"),
            pl.col("verbs").struct.field("mood").alias("mood"),
            pl.col("verbs").struct.field("verb_form").alias("verb_form"),
        ])
        .collect()
    )

    tense = (
        verbs
        .filter(pl.col("tense").is_not_null())
        .group_by(["genre", "tense"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
    )

    aspect = (
        verbs
        .filter(pl.col("aspect").is_not_null())
        .group_by(["genre", "aspect"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
    )

    mood = (
        verbs
        .filter(pl.col("mood").is_not_null())
        .group_by(["genre", "mood"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
    )

    verb_form = (
        verbs
        .group_by(["genre", "verb_form"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
    )

    return {
        "tense": tense,
        "aspect": aspect,
        "mood": mood,
        "verb_form": verb_form,
    }


def verbal_morphology_paradigm(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §6.1 – Person × Number paradigm showing distinct vs syncretic forms.

    Returns the agreement paradigm completeness for the corpus.
    """
    lf = _ensure_lazy(df)

    verbs = (
        lf
        .select("verbs")
        .explode("verbs")
        .filter(pl.col("verbs").is_not_null())
        .select([
            pl.col("verbs").struct.field("verb_form").alias("verb_form"),
            pl.col("verbs").struct.field("person").alias("person"),
            pl.col("verbs").struct.field("number").alias("number"),
            pl.col("verbs").struct.field("tense").alias("tense"),
            pl.col("verbs").struct.field("lemma").alias("lemma"),
        ])
        .filter(pl.col("verb_form") == "Fin")  # Only finite verbs
        .collect()
    )

    if verbs.is_empty():
        return {
            "total_finite_verbs": 0,
            "paradigm_cells": [],
            "distinct_forms_per_paradigm": None,
            "syncretism_rate": None,
        }

    # Cross-tab person × number
    paradigm = (
        verbs
        .filter(pl.col("person").is_not_null())
        .filter(pl.col("number").is_not_null())
        .group_by(["person", "number"])
        .agg(pl.count().alias("count"))
        .sort(["person", "number"])
        .to_dicts()
    )

    # Count unique surface forms (lemma + tense combinations)
    # This approximates paradigm richness
    unique_forms = (
        verbs
        .filter(pl.col("person").is_not_null())
        .filter(pl.col("number").is_not_null())
        .group_by(["person", "number", "tense"])
        .agg(pl.col("lemma").n_unique().alias("unique_lemmas"))
        .height
    )

    # Theoretical cells (6 = 3 persons × 2 numbers)
    theoretical_cells = 6
    syncretism = 1 - (unique_forms / theoretical_cells) if theoretical_cells > 0 else None

    return {
        "total_finite_verbs": verbs.height,
        "paradigm_cells": paradigm,
        "distinct_forms_per_paradigm": unique_forms,
        "syncretism_rate": syncretism,
    }


def person_number_tense_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §6.2 – Frequency of Person × Number × Tense cells.

    Returns detailed distribution of agreement cells in the corpus.
    """
    lf = _ensure_lazy(df)

    return (
        lf
        .select("verbs")
        .explode("verbs")
        .filter(pl.col("verbs").is_not_null())
        .select([
            pl.col("verbs").struct.field("verb_form").alias("verb_form"),
            pl.col("verbs").struct.field("person").alias("person"),
            pl.col("verbs").struct.field("number").alias("number"),
            pl.col("verbs").struct.field("tense").alias("tense"),
        ])
        .filter(pl.col("verb_form") == "Fin")
        .filter(pl.col("person").is_not_null())
        .filter(pl.col("number").is_not_null())
        .group_by(["person", "number", "tense"])
        .agg(pl.count().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
        .sort(["person", "number", "tense"])
        .collect()
    )


def morphology_manipulation_coverage(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §6.3 – % verbs affected by enrichment (EN) / impoverishment (IT).

    Calculates what proportion of verbs would be affected by morphological
    manipulation (adding rich agreement to English or impoverishing Italian).
    """
    lf = _ensure_lazy(df)

    verbs = (
        lf
        .select("verbs")
        .explode("verbs")
        .filter(pl.col("verbs").is_not_null())
        .select([
            pl.col("verbs").struct.field("verb_form").alias("verb_form"),
            pl.col("verbs").struct.field("person").alias("person"),
            pl.col("verbs").struct.field("number").alias("number"),
            pl.col("verbs").struct.field("tense").alias("tense"),
        ])
        .filter(pl.col("verb_form") == "Fin")
        .collect()
    )

    if verbs.is_empty():
        return {
            "total_finite_verbs": 0,
            "english_enrichment_coverage": None,
            "italian_impoverishment_coverage": None,
            "by_person_number": [],
        }

    total_finite = verbs.height

    # For English enrichment: verbs that are NOT 3sg (since 3sg already has -s)
    # These would receive new morphological marking
    non_3sg = verbs.filter(
        ~((pl.col("person") == 3) & (pl.col("number") == "Sing"))
    ).height
    english_coverage = non_3sg / total_finite if total_finite else None

    # For Italian impoverishment: all verbs with person/number marking
    # would be affected (collapsed to single form)
    marked = verbs.filter(
        pl.col("person").is_not_null() & pl.col("number").is_not_null()
    ).height
    italian_coverage = marked / total_finite if total_finite else None

    # Breakdown by person × number
    by_pn = (
        verbs
        .filter(pl.col("person").is_not_null())
        .filter(pl.col("number").is_not_null())
        .group_by(["person", "number"])
        .agg([
            pl.count().alias("count"),
        ])
        .with_columns(
            (pl.col("count") / total_finite).alias("proportion")
        )
        .sort(["person", "number"])
        .to_dicts()
    )

    return {
        "total_finite_verbs": total_finite,
        "english_enrichment_coverage": english_coverage,
        "italian_impoverishment_coverage": italian_coverage,
        "by_person_number": by_pn,
    }


# ---------------------------------------------------------------------------
# 10. Cross-genre comparison: spoken vs written
# ---------------------------------------------------------------------------

def _add_register(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add a ``register`` column (spoken/written) based on genre."""
    return lf.with_columns(
        pl.when(pl.col("genre").is_in(list(SPOKEN_GENRES)))
        .then(pl.lit("spoken"))
        .otherwise(pl.lit("written"))
        .alias("register")
    )


def register_comparison(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §10 – Key phenomena by spoken vs written register.
    """
    lf = _add_register(_ensure_lazy(df))
    available = set(lf.collect_schema().names())

    aggs: list = [pl.count().alias("n_sentences")]
    for flag in [
        "has_null_subject", "has_overt_subject_pronoun", "has_expletive",
        "has_that_trace_context", "is_fragment", "is_imperative",
    ]:
        if flag in available:
            aggs.append(pl.col(flag).mean().alias(f"{flag}_rate"))

    if "complexity" in available:
        aggs.append(
            pl.col("complexity").struct.field("max_embedding_depth")
            .mean().alias("mean_depth")
        )

    return lf.group_by("register").agg(aggs).sort("register").collect()


def register_by_genre(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Detailed phenomenon rates within register × genre."""
    lf = _add_register(_ensure_lazy(df))
    available = set(lf.collect_schema().names())

    aggs: list = [pl.count().alias("n_sentences")]
    for flag in [
        "has_null_subject", "has_expletive", "is_fragment",
    ]:
        if flag in available:
            aggs.append(pl.col(flag).mean().alias(f"{flag}_rate"))

    return (
        lf
        .group_by(["register", "genre"])
        .agg(aggs)
        .sort(["register", "has_null_subject_rate" if "has_null_subject" in available else "n_sentences"],
              descending=[False, True])
        .collect()
    )


# ---------------------------------------------------------------------------
# 10.5 Manipulation feasibility (§7)
# ---------------------------------------------------------------------------

def manipulation_coverage(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §7 – Tokens/sentences affected by each manipulation.

    Calculates the proportion of corpus affected by each experimental
    manipulation (pronoun removal, expletive removal, etc.).
    """
    lf = _ensure_lazy(df)

    # Get total counts
    totals = lf.select([
        pl.count().alias("total_sentences"),
        pl.col("n_tokens").sum().alias("total_tokens"),
    ]).collect()
    total_sentences = totals["total_sentences"].item()
    total_tokens = totals["total_tokens"].item()

    coverage: Dict[str, Any] = {
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "manipulations": {},
    }

    # Pronoun removal (sentences with subject pronouns)
    pron_sents = lf.filter(pl.col("has_overt_subject_pronoun") == True)
    pron_count = pron_sents.select(pl.count()).collect().item()
    pron_tokens = pron_sents.select(pl.col("n_tokens").sum()).collect().item() or 0
    coverage["manipulations"]["pronoun_removal"] = {
        "sentences_affected": pron_count,
        "sentence_rate": pron_count / total_sentences if total_sentences else None,
        "tokens_in_affected_sentences": pron_tokens,
        "token_rate": pron_tokens / total_tokens if total_tokens else None,
    }

    # Expletive removal
    expl_sents = lf.filter(pl.col("has_expletive") == True)
    expl_count = expl_sents.select(pl.count()).collect().item()
    expl_tokens = expl_sents.select(pl.col("n_tokens").sum()).collect().item() or 0
    coverage["manipulations"]["expletive_removal"] = {
        "sentences_affected": expl_count,
        "sentence_rate": expl_count / total_sentences if total_sentences else None,
        "tokens_in_affected_sentences": expl_tokens,
        "token_rate": expl_tokens / total_tokens if total_tokens else None,
    }

    # Case neutralization (sentences with any pronouns that have case marking)
    # This approximates the effect
    obj_pron_sents = lf.filter(pl.col("has_overt_object_pronoun") == True)
    obj_count = obj_pron_sents.select(pl.count()).collect().item()
    subj_count = pron_count  # Already computed
    # Union of sentences with subject or object pronouns
    either_pron = lf.filter(
        (pl.col("has_overt_subject_pronoun") == True) |
        (pl.col("has_overt_object_pronoun") == True)
    )
    either_count = either_pron.select(pl.count()).collect().item()
    either_tokens = either_pron.select(pl.col("n_tokens").sum()).collect().item() or 0
    coverage["manipulations"]["case_neutralization"] = {
        "sentences_affected": either_count,
        "sentence_rate": either_count / total_sentences if total_sentences else None,
        "tokens_in_affected_sentences": either_tokens,
        "token_rate": either_tokens / total_tokens if total_tokens else None,
    }

    return coverage


def manipulation_overlap_matrix(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """
    §7 – Pairwise overlap between manipulations.

    Returns a matrix showing proportion of sentences affected by manipulation A
    that are also affected by manipulation B.
    """
    lf = _ensure_lazy(df)

    # Define manipulation filters
    manipulations = {
        "pronoun_removal": pl.col("has_overt_subject_pronoun") == True,
        "expletive_removal": pl.col("has_expletive") == True,
        "negation_present": pl.col("has_negation") == True,
        "that_trace_context": pl.col("has_that_trace_context") == True,
    }

    # Compute overlap matrix
    rows = []
    for name_a, filter_a in manipulations.items():
        sents_a = lf.filter(filter_a)
        count_a = sents_a.select(pl.count()).collect().item()

        row = {"manipulation": name_a, "total": count_a}
        for name_b, filter_b in manipulations.items():
            if count_a > 0:
                overlap_count = sents_a.filter(filter_b).select(pl.count()).collect().item()
                row[f"overlap_{name_b}"] = overlap_count / count_a
            else:
                row[f"overlap_{name_b}"] = None
        rows.append(row)

    return pl.DataFrame(rows)


def evaluation_context_frequency(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §9 – Map 18 evaluation stimulus types to corpus proxies.

    Flags rare categories (<100 per 90M tokens).
    """
    lf = _ensure_lazy(df)

    total_tokens = lf.select(pl.col("n_tokens").sum()).collect().item() or 0
    RARE_THRESHOLD = 100  # per 90M tokens

    contexts: Dict[str, Any] = {
        "total_tokens": total_tokens,
        "categories": {},
    }

    # Helper to compute count and flag rarity
    def add_context(name: str, count: int) -> None:
        # Scale threshold to actual corpus size
        scaled_threshold = RARE_THRESHOLD * (total_tokens / 90_000_000) if total_tokens else 0
        contexts["categories"][name] = {
            "count": count,
            "per_million": count / (total_tokens / 1_000_000) if total_tokens else None,
            "is_rare": count < scaled_threshold,
        }

    # 3sg/3pl subject drop proxy: finite clause with 3rd person pronominal subject
    clauses_3p = (
        lf
        .select("clauses")
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .filter(pl.col("clauses").struct.field("is_finite") == True)
        .filter(pl.col("clauses").struct.field("subject_person") == 3)
        .filter(pl.col("clauses").struct.field("subject_is_pronoun") == True)
        .select(pl.count())
        .collect().item()
    )
    add_context("3p_subject_pronoun", clauses_3p)

    # 1sg/2sg subject proxy
    clauses_12 = (
        lf
        .select("clauses")
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .filter(pl.col("clauses").struct.field("is_finite") == True)
        .filter(pl.col("clauses").struct.field("subject_person").is_in([1, 2]))
        .filter(pl.col("clauses").struct.field("subject_is_pronoun") == True)
        .select(pl.count())
        .collect().item()
    )
    add_context("1p_2p_subject_pronoun", clauses_12)

    # Subordinate clause with pronominal subject (ccomp)
    ccomp_pron = (
        lf
        .select("clauses")
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .filter(pl.col("clauses").struct.field("clause_type") == "ccomp")
        .filter(pl.col("clauses").struct.field("subject_is_pronoun") == True)
        .select(pl.count())
        .collect().item()
    )
    add_context("subordinate_pronominal_subject", ccomp_pron)

    # Expletive with raising verbs
    raising_expl = (
        lf
        .filter(pl.col("has_expletive") == True)
        .select("expletives")
        .explode("expletives")
        .filter(pl.col("expletives").is_not_null())
        .filter(pl.col("expletives").struct.field("expletive_class") == "raising")
        .select(pl.count())
        .collect().item()
    )
    add_context("expletive_raising", raising_expl)

    # Expletive with existential
    exist_expl = (
        lf
        .filter(pl.col("has_expletive") == True)
        .select("expletives")
        .explode("expletives")
        .filter(pl.col("expletives").is_not_null())
        .filter(pl.col("expletives").struct.field("expletive_class") == "existential")
        .select(pl.count())
        .collect().item()
    )
    add_context("expletive_existential", exist_expl)

    # That-trace environments (from §4)
    tt_count = lf.filter(pl.col("has_that_trace_context") == True).select(pl.count()).collect().item()
    add_context("that_trace_context", tt_count)

    # Wh-extraction
    wh_count = lf.filter(pl.col("has_wh_extraction") == True).select(pl.count()).collect().item()
    add_context("wh_extraction", wh_count)

    # Relative clauses
    rel_count = lf.filter(pl.col("has_relative_clause") == True).select(pl.count()).collect().item()
    add_context("relative_clause", rel_count)

    return contexts


# ---------------------------------------------------------------------------
# 11. Language-agnostic / Italian comparison hooks
# ---------------------------------------------------------------------------

def language_agnostic_analysis(
    df: Union[pl.DataFrame, pl.LazyFrame],
    language_label: str = "English",
) -> Dict[str, Any]:
    """
    §11 – Core analyses that work identically on EN and IT annotations.

    Returns a dict of key rates that can be compared cross-linguistically.
    """
    lf = _ensure_lazy(df)
    available = set(lf.collect_schema().names())

    stats: Dict[str, Any] = {"language": language_label}

    for flag in [
        "has_null_subject", "has_expletive", "has_that_trace_context",
        "is_fragment", "is_imperative",
    ]:
        if flag in available:
            val = lf.select(pl.col(flag).mean()).collect().item()
            stats[f"{flag}_rate"] = val

    # Subject pronoun person distribution
    if "pronouns" in available:
        person_dist = (
            lf
            .select("pronouns")
            .explode("pronouns")
            .filter(pl.col("pronouns").is_not_null())
            .filter(pl.col("pronouns").struct.field("function") == "subject")
            .select(pl.col("pronouns").struct.field("person").alias("person"))
            .group_by("person")
            .agg(pl.count().alias("count"))
            .collect()
            .to_dicts()
        )
        stats["subject_pronoun_person_distribution"] = person_dist

    return stats


def cross_linguistic_comparison(
    en_stats: Dict[str, Any],
    it_stats: Dict[str, Any],
) -> pl.DataFrame:
    """
    Build a comparison table from two ``language_agnostic_analysis`` results.
    """
    metrics = [
        ("Null subject rate", "has_null_subject_rate"),
        ("Overt expletive rate", "has_expletive_rate"),
        ("That-trace context rate", "has_that_trace_context_rate"),
        ("Fragment rate", "is_fragment_rate"),
        ("Imperative rate", "is_imperative_rate"),
    ]

    rows = []
    for label, key in metrics:
        en_val = en_stats.get(key)
        it_val = it_stats.get(key)
        diff = (it_val - en_val) if (en_val is not None and it_val is not None) else None
        rows.append({
            "Metric": label,
            "English": en_val,
            "Italian": it_val,
            "Difference": diff,
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 12. Wh-question distribution (§10)
# ---------------------------------------------------------------------------

def wh_question_distribution(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    §10 – Subject vs object vs embedded wh-question counts.

    Filters by has_wh_extraction flag and uses extraction_type from wh_extractions.
    """
    lf = _ensure_lazy(df)

    wh_data = (
        lf
        .filter(pl.col("has_wh_extraction") == True)
        .select(["genre", "wh_extractions"])
        .explode("wh_extractions")
        .filter(pl.col("wh_extractions").is_not_null())
        .select([
            "genre",
            pl.col("wh_extractions").struct.field("extraction_type").alias("extraction_type"),
            pl.col("wh_extractions").struct.field("wh_lemma").alias("wh_lemma"),
            pl.col("wh_extractions").struct.field("clause_level").alias("clause_level"),
        ])
        .collect()
    )

    if wh_data.is_empty():
        return {
            "total_wh_questions": 0,
            "by_extraction_type": [],
            "by_clause_level": [],
            "by_wh_lemma": [],
            "by_genre": [],
        }

    by_extraction = (
        wh_data
        .group_by("extraction_type")
        .agg(pl.count().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
        .sort("count", descending=True)
        .to_dicts()
    )

    by_clause_level = (
        wh_data
        .group_by("clause_level")
        .agg(pl.count().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("proportion")
        )
        .sort("count", descending=True)
        .to_dicts()
    )

    by_lemma = (
        wh_data
        .group_by("wh_lemma")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(20)
        .to_dicts()
    )

    by_genre = (
        wh_data
        .group_by(["genre", "extraction_type"])
        .agg(pl.count().alias("count"))
        .sort(["genre", "count"], descending=[False, True])
        .to_dicts()
    )

    return {
        "total_wh_questions": wh_data.height,
        "by_extraction_type": by_extraction,
        "by_clause_level": by_clause_level,
        "by_wh_lemma": by_lemma,
        "by_genre": by_genre,
    }


# ---------------------------------------------------------------------------
# Research hypothesis analyses
# ---------------------------------------------------------------------------

def h5_distal_effects(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    H5: Does manipulating pronoun distribution affect that-trace judgments?

    Provides the corpus-level co-occurrence baseline.
    """
    return cooccurrence_null_subject_that_trace(df)


def h6a_expletive_analysis(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> Dict[str, Any]:
    """
    H6a: Do expletives signal overt-subject requirements?

    Returns expletive distribution by class and co-occurrence with null subjects.
    """
    lf = _ensure_lazy(df)

    by_class = expletive_by_class(lf)

    # Co-occurrence: expletive sentences that also have null subjects
    expl_sents = lf.filter(pl.col("has_expletive") == True)
    total_expl = expl_sents.select(pl.count()).collect().item()
    expl_with_null = (
        expl_sents.filter(pl.col("has_null_subject") == True)
        .select(pl.count()).collect().item()
    )

    return {
        "by_class": by_class.to_dicts(),
        "total_expletive_sentences": total_expl,
        "expletive_with_null_subject": expl_with_null,
        "null_rate_in_expletive_sents": expl_with_null / total_expl if total_expl else None,
    }


def null_subject_by_embedding_depth(
    df: Union[pl.DataFrame, pl.LazyFrame],
) -> pl.DataFrame:
    """Are null subjects more common in embedded clauses?"""
    lf = _ensure_lazy(df)
    return (
        lf
        .select([
            pl.col("has_null_subject"),
            pl.col("complexity").struct.field("max_embedding_depth").alias("depth"),
        ])
        .group_by("depth")
        .agg([
            pl.col("has_null_subject").mean().alias("null_subject_rate"),
            pl.count().alias("n_sentences"),
        ])
        .sort("depth")
        .collect()
    )


# ---------------------------------------------------------------------------
# Convenience: run all analyses
# ---------------------------------------------------------------------------

def run_all_analyses(
    df: Union[pl.DataFrame, pl.LazyFrame],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute every analysis and return a results dict.

    If *output_dir* is provided, individual JSON files are written there.
    """
    lf = _ensure_lazy(df)
    results: Dict[str, Any] = {}

    # 1. Overview
    results["corpus_overview"] = corpus_overview(lf).to_dicts()

    # 2. Phenomenon rates
    results["phenomenon_rates_by_genre"] = phenomenon_rates_by_genre(lf).to_dicts()

    # 3. Co-occurrences
    results["cooccurrence_null_subject_that_trace"] = cooccurrence_null_subject_that_trace(lf)
    results["cooccurrence_negation_null_subject"] = cooccurrence_negation_null_subject(lf).to_dicts()
    results["cooccurrence_expletive_depth"] = cooccurrence_expletive_depth(lf)

    # 4. CHILDES
    results["childes_child_vs_adult"] = childes_child_vs_adult(lf).to_dicts()
    results["childes_speaker_distribution"] = childes_speaker_distribution(lf).to_dicts()
    results["childes_pronoun_distribution"] = childes_pronoun_distribution(lf).to_dicts()
    results["childes_person_distribution"] = childes_person_distribution(lf).to_dicts()

    # 5. Pronouns
    results["subject_pronoun_distribution"] = subject_pronoun_distribution(lf).to_dicts()
    results["subject_pronoun_rate"] = subject_pronoun_rate(lf)
    results["case_marked_pronoun_analysis"] = case_marked_pronoun_analysis(lf)

    # 6. That-trace
    results["that_trace_inventory"] = that_trace_inventory(lf)
    clause_crosstab = clause_structure_crosstab(lf)
    results["clause_structure_crosstab"] = clause_crosstab.to_dicts() if not clause_crosstab.is_empty() else []
    tt_crosstab = that_trace_crosstab(lf)
    results["that_trace_crosstab"] = tt_crosstab.to_dicts() if not tt_crosstab.is_empty() else []

    # 7. Expletives
    results["expletive_by_class"] = expletive_by_class(lf).to_dicts()

    # 8. Complexity
    results["complexity_distribution"] = complexity_distribution(lf).to_dicts()
    results["negation_position_distribution"] = negation_position_distribution(lf)

    # 9. Verbs
    results["verb_null_subject_rate"] = verb_null_subject_rate(lf).to_dicts()
    results["bridge_verb_null_rates"] = bridge_verb_null_rates(lf).to_dicts()
    results["weather_verb_null_rates"] = weather_verb_null_rates(lf).to_dicts()
    tam = tense_aspect_mood_distribution(lf)
    results["tense_distribution"] = tam["tense"].to_dicts()
    results["aspect_distribution"] = tam["aspect"].to_dicts()
    results["mood_distribution"] = tam["mood"].to_dicts()
    results["verb_form_distribution"] = tam["verb_form"].to_dicts()

    # 9.5 Verbal morphology (§6)
    results["verbal_morphology_paradigm"] = verbal_morphology_paradigm(lf)
    pnt_dist = person_number_tense_distribution(lf)
    results["person_number_tense_distribution"] = pnt_dist.to_dicts() if not pnt_dist.is_empty() else []
    results["morphology_manipulation_coverage"] = morphology_manipulation_coverage(lf)

    # 10. Register
    results["register_comparison"] = register_comparison(lf).to_dicts()

    # 10.5 Manipulation feasibility
    results["manipulation_coverage"] = manipulation_coverage(lf)
    results["manipulation_overlap_matrix"] = manipulation_overlap_matrix(lf).to_dicts()
    results["evaluation_context_frequency"] = evaluation_context_frequency(lf)

    # 11. Language-agnostic
    results["language_agnostic"] = language_agnostic_analysis(lf)

    # 12. Wh-questions
    results["wh_question_distribution"] = wh_question_distribution(lf)

    # Research hypotheses
    results["h5_distal_effects"] = h5_distal_effects(lf)
    results["h6a_expletive_analysis"] = h6a_expletive_analysis(lf)
    results["null_subject_by_depth"] = null_subject_by_embedding_depth(lf).to_dicts()

    # Write output
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, data in results.items():
            path = output_dir / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, default=str))

        combined = output_dir / "corpus_analysis_results.json"
        combined.write_text(json.dumps(results, indent=2, default=str))

    return results
