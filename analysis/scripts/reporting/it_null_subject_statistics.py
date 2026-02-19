#!/usr/bin/env python
"""
Compute descriptive statistics about null subjects in Italian.

Joins annotation layers (clause_structure, complexity) with tree detector
output to produce a comprehensive report on null subject distribution.

Usage
-----
    PYTHONPATH=. python -m analysis.scripts.reporting.it_null_subject_statistics \
        --annotated-dir data/output/it_local_test/annotated_corpus/ \
        --output data/output/it_local_test/it_null_subject_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)

_NULL_STATUSES = ["none", "imperative"]

_BASE_COLS = ["sentence_id", "genre", "role", "n_tokens"]


# ---------------------------------------------------------------------------
# Data loading (file-by-file to avoid OOM)
# ---------------------------------------------------------------------------

def _load_base(corpus_path: Path) -> pl.DataFrame:
    """Load base layer with minimal columns."""
    base_dir = corpus_path / "base"
    return pl.read_parquet(
        sorted(base_dir.glob("*.parquet")),
        columns=_BASE_COLS,
    )


def _iter_layer_files(corpus_path: Path, layer_name: str) -> List[Path]:
    """Return sorted list of parquet files for a given layer."""
    layer_dir = corpus_path / "layers" / layer_name
    if not layer_dir.exists():
        return []
    return sorted(f for f in layer_dir.glob("*.parquet") if f.stem != "_backup")


def _load_clause_data(corpus_path: Path) -> pl.DataFrame:
    """Load and explode clause_structure layer, returning finite clauses."""
    files = _iter_layer_files(corpus_path, "clause_structure")
    if not files:
        return pl.DataFrame()

    chunks = []
    for f in files:
        df = pl.read_parquet(f)
        chunk = (
            df.select(["sentence_id", "clauses"])
            .explode("clauses")
            .filter(pl.col("clauses").is_not_null())
            .select([
                "sentence_id",
                pl.col("clauses").struct.field("clause_type").alias("clause_type"),
                pl.col("clauses").struct.field("subject_status").alias("subject_status"),
                pl.col("clauses").struct.field("is_finite").alias("is_finite"),
                pl.col("clauses").struct.field("verb_lemma").alias("verb_lemma"),
            ])
            .filter(pl.col("is_finite") == True)
            .with_columns(
                pl.col("subject_status").is_in(_NULL_STATUSES).alias("is_null"),
            )
        )
        chunks.append(chunk)

    return pl.concat(chunks)


def _load_tree_detector(corpus_path: Path) -> pl.DataFrame:
    """Load tree detector layer."""
    files = _iter_layer_files(corpus_path, "tree_detector")
    if not files:
        return pl.DataFrame()

    return pl.read_parquet(
        files,
        columns=["sentence_id", "null_subject_detections", "n_null_subjects",
                  "has_null_subject_tree"],
    )


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_annotation_stats(
    base: pl.DataFrame,
    clauses: pl.DataFrame,
) -> Dict[str, Any]:
    """Compute annotation-based statistics."""
    stats: Dict[str, Any] = {}

    if clauses.is_empty():
        return {"annotation": {"error": "No clause_structure data found"}}

    # Join genre
    clauses_with_genre = clauses.join(
        base.select(["sentence_id", "genre"]),
        on="sentence_id",
        how="left",
    )

    total_finite = clauses_with_genre.height
    total_null = clauses_with_genre.filter(pl.col("is_null")).height
    null_rate = total_null / total_finite if total_finite > 0 else 0.0

    stats["total_finite_clauses"] = total_finite
    stats["total_null_subject_clauses"] = total_null
    stats["null_subject_rate"] = round(null_rate, 4)

    # By genre
    by_genre = (
        clauses_with_genre
        .group_by("genre")
        .agg([
            pl.len().alias("n_finite"),
            pl.col("is_null").sum().alias("n_null"),
        ])
        .with_columns(
            (pl.col("n_null") / pl.col("n_finite")).alias("null_rate"),
        )
        .sort("genre")
    )
    stats["by_genre"] = by_genre.to_dicts()

    # Imperative rate
    n_imperative = clauses_with_genre.filter(
        pl.col("subject_status") == "imperative"
    ).height
    stats["imperative_rate"] = round(n_imperative / total_finite, 4) if total_finite > 0 else 0.0
    stats["n_imperative"] = n_imperative

    # By clause type
    by_clause = (
        clauses_with_genre
        .group_by("clause_type")
        .agg([
            pl.len().alias("n_finite"),
            pl.col("is_null").sum().alias("n_null"),
        ])
        .with_columns(
            (pl.col("n_null") / pl.col("n_finite")).alias("null_rate"),
        )
        .sort("clause_type")
    )
    stats["by_clause_type"] = by_clause.to_dicts()

    # Subject status distribution
    status_dist = (
        clauses_with_genre
        .group_by("subject_status")
        .agg(pl.len().alias("count"))
        .sort("subject_status")
    )
    stats["subject_status_distribution"] = status_dist.to_dicts()

    return {"annotation": stats}


def compute_tree_detector_stats(
    base: pl.DataFrame,
    tree: pl.DataFrame,
) -> Dict[str, Any]:
    """Compute tree-detector-based statistics."""
    stats: Dict[str, Any] = {}

    if tree.is_empty():
        return {"tree_detector": {"error": "No tree_detector data found"}}

    total_sents = tree.height
    sents_with_null = tree.filter(pl.col("has_null_subject_tree")).height
    stats["total_sentences"] = total_sents
    stats["sentences_with_null_subject"] = sents_with_null
    stats["sentence_null_rate"] = round(sents_with_null / total_sents, 4) if total_sents > 0 else 0.0

    # Total detections
    total_detections = tree.select(pl.col("n_null_subjects").sum()).item()
    stats["total_null_subject_detections"] = total_detections

    # By label — explode detections
    tree_with_genre = tree.join(
        base.select(["sentence_id", "genre"]),
        on="sentence_id",
        how="left",
    )

    exploded = (
        tree_with_genre
        .filter(pl.col("has_null_subject_tree"))
        .explode("null_subject_detections")
        .filter(pl.col("null_subject_detections").is_not_null())
        .select([
            "genre",
            pl.col("null_subject_detections").struct.field("label").alias("label"),
            pl.col("null_subject_detections").struct.field("confidence").alias("confidence"),
        ])
    )

    if not exploded.is_empty():
        # By label
        by_label = (
            exploded
            .group_by("label")
            .agg([
                pl.len().alias("count"),
                pl.col("confidence").mean().alias("mean_confidence"),
            ])
            .sort("count", descending=True)
        )
        stats["by_label"] = by_label.to_dicts()

        # By genre
        by_genre = (
            exploded
            .group_by("genre")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        stats["by_genre"] = by_genre.to_dicts()

        # Confidence distribution (quartiles)
        confidences = exploded["confidence"]
        stats["confidence_distribution"] = {
            "min": round(confidences.min(), 4),
            "q25": round(confidences.quantile(0.25), 4),
            "median": round(confidences.median(), 4),
            "q75": round(confidences.quantile(0.75), 4),
            "max": round(confidences.max(), 4),
            "mean": round(confidences.mean(), 4),
        }

    return {"tree_detector": stats}


def compute_cross_layer_stats(
    clauses: pl.DataFrame,
    tree: pl.DataFrame,
) -> Dict[str, Any]:
    """Compute cross-layer agreement statistics."""
    stats: Dict[str, Any] = {}

    if clauses.is_empty() or tree.is_empty():
        return {"cross_layer": {"error": "Missing clause or tree data"}}

    # Sentences with annotation-based null subjects
    null_sents_annotation = (
        clauses
        .filter(pl.col("is_null"))
        .select("sentence_id")
        .unique()
    )

    # Sentences with tree-detected null subjects
    null_sents_tree = (
        tree
        .filter(pl.col("has_null_subject_tree"))
        .select("sentence_id")
    )

    n_ann = null_sents_annotation.height
    n_tree = null_sents_tree.height

    # Overlap
    both = null_sents_annotation.join(null_sents_tree, on="sentence_id", how="inner")
    n_both = both.height

    stats["annotation_null_sentences"] = n_ann
    stats["tree_null_sentences"] = n_tree
    stats["both_null_sentences"] = n_both
    stats["agreement_rate"] = round(n_both / n_ann, 4) if n_ann > 0 else 0.0
    stats["tree_only"] = n_tree - n_both
    stats["annotation_only"] = n_ann - n_both

    return {"cross_layer": stats}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute Italian null subject statistics."
    )
    parser.add_argument(
        "--annotated-dir",
        type=str,
        required=True,
        help="Path to annotated corpus directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: annotated-dir/it_null_subject_stats.json)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    corpus_path = Path(args.annotated_dir)

    # Load data
    logger.info("Loading base layer...")
    base = _load_base(corpus_path)
    logger.info("  %d sentences", base.height)

    logger.info("Loading clause_structure layer...")
    clauses = _load_clause_data(corpus_path)
    logger.info("  %d finite clauses", clauses.height if not clauses.is_empty() else 0)

    logger.info("Loading tree_detector layer...")
    tree = _load_tree_detector(corpus_path)
    logger.info("  %d sentences", tree.height if not tree.is_empty() else 0)

    # Compute statistics
    all_stats: Dict[str, Any] = {}
    all_stats.update(compute_annotation_stats(base, clauses))
    all_stats.update(compute_tree_detector_stats(base, tree))
    all_stats.update(compute_cross_layer_stats(clauses, tree))

    # Output
    output_path = Path(args.output) if args.output else corpus_path / "it_null_subject_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_stats, indent=2, default=str))
    logger.info("Statistics written to %s", output_path)

    # Print summary
    ann = all_stats.get("annotation", {})
    td = all_stats.get("tree_detector", {})
    cl = all_stats.get("cross_layer", {})

    print("\n=== Italian Null Subject Statistics ===")
    if "total_finite_clauses" in ann:
        print(f"Finite clauses:      {ann['total_finite_clauses']:,}")
        print(f"Null subject rate:   {ann['null_subject_rate']:.1%}")
        print(f"Imperative rate:     {ann['imperative_rate']:.1%}")
    if "total_null_subject_detections" in td:
        print(f"Tree detections:     {td['total_null_subject_detections']:,}")
        print(f"Sentence null rate:  {td['sentence_null_rate']:.1%}")
    if "agreement_rate" in cl:
        print(f"Cross-layer agree:   {cl['agreement_rate']:.1%}")


if __name__ == "__main__":
    main()
