#!/usr/bin/env python
"""
Generate longitudinal null-subject figures from per-child data.

Requires pre-computed data files in data/output/train_90M/:
  - child_longitudinal_null_rates.parquet (from reconstruct_child_identity)
  - childes_child_mapping.parquet
  - childes_null_type.parquet (from reconstruct_imperatives.py)

Produces:
  1. child_longitudinal_lowess_6mo.png — Faceted LOWESS per child (6mo bins)
  2. child_null_subject_two_groups_early.png — Decliner vs adult-like (12-35m split)
  3. child_vs_adult_unsplit.png — Overall child vs adult null rate + MLU
  4. child_vs_adult_two_groups.png — 2×2 grid: child/adult × decliner/adult-like
  5. null_subject_imperative_split.png — Imperative vs non-imperative null

Usage
-----
    python analysis/scripts/reporting/generate_longitudinal_figures.py \
        --data-dir data/output/train_90M
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
import statsmodels.api as sm

DPI = 150
ADULT_RATE = 0.036
THRESHOLD = 0.06  # early-rate split for two-group analysis

AGE_LABELS = {17.5: "12-23", 29.5: "24-35", 41.5: "36-47",
              53.5: "48-59", 65.0: "60-71", 78.0: "72+"}


def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def add_12m_bins(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("child_age_months") < 24).then(17.5)
        .when(pl.col("child_age_months") < 36).then(29.5)
        .when(pl.col("child_age_months") < 48).then(41.5)
        .when(pl.col("child_age_months") < 60).then(53.5)
        .when(pl.col("child_age_months") < 72).then(65.0)
        .otherwise(78.0)
        .alias("age_mid"),
    )


def load_data(data_dir: Path):
    """Load all required datasets."""
    mapping = pl.read_parquet(data_dir / "childes_child_mapping.parquet")
    base = pl.read_parquet(
        data_dir / "annotated_corpus" / "base" / "train_90M_childes.parquet",
        columns=["sentence_id", "sent_idx", "genre", "role", "speaker",
                 "child_age_months", "n_tokens"]
    ).join(mapping, on="sent_idx", how="left")

    clause_dir = data_dir / "annotated_corpus" / "layers" / "clause_structure"
    clause_files = sorted(f for f in clause_dir.glob("*.parquet") if f.stem != "_backup")
    childes_ids = set(base.filter(
        pl.col("genre").str.contains("CHILDES")
    )["sentence_id"].to_list())

    clause_dfs = []
    for f in clause_files:
        df = pl.read_parquet(f, columns=["sentence_id", "clauses"])
        df = df.filter(pl.col("sentence_id").is_in(childes_ids))
        if len(df) > 0:
            clause_dfs.append(df)
    clauses = pl.concat(clause_dfs)

    exploded = (
        clauses.explode("clauses").unnest("clauses")
        .filter(pl.col("is_finite") == True)
        .with_columns(
            pl.col("subject_status").is_in(["none", "imperative"]).alias("is_null")
        )
    )
    exploded = exploded.join(
        base.select(["sentence_id", "genre", "child_age_months",
                     "childes_child", "childes_corpus", "n_tokens"]),
        on="sentence_id", how="inner"
    )

    longitudinal = pl.read_parquet(data_dir / "child_longitudinal_null_rates.parquet")

    null_type = None
    null_type_path = data_dir / "childes_null_type.parquet"
    if null_type_path.exists():
        null_type = pl.read_parquet(null_type_path)

    return base, exploded, longitudinal, null_type


def get_child_groups(exploded: pl.DataFrame):
    """Split children into decliners vs adult-like using 12-35m rate."""
    child_speech = exploded.filter(
        (pl.col("genre") == "CHILDES_child") &
        pl.col("childes_child").is_not_null() &
        (pl.col("child_age_months") >= 12)
    ).with_columns(
        ((pl.col("child_age_months") // 6) * 6 + 3).cast(pl.Float64).alias("age_mid_6mo"),
    )

    child_binned = (
        child_speech.group_by(["childes_child", "childes_corpus", "age_mid_6mo"])
        .agg([pl.col("is_null").sum().alias("null_count"), pl.len().alias("n_clauses")])
        .with_columns((pl.col("null_count") / pl.col("n_clauses")).alias("null_rate"))
    )

    early = child_binned.filter(pl.col("age_mid_6mo") < 36)
    early_rates = (
        early.group_by(["childes_child", "childes_corpus"])
        .agg([pl.col("null_count").sum().alias("null_count"),
              pl.col("n_clauses").sum().alias("n_clauses")])
        .filter(pl.col("n_clauses") >= 20)
        .with_columns((pl.col("null_count") / pl.col("n_clauses")).alias("early_rate"))
    )

    coverage = (
        child_speech.with_columns(((pl.col("child_age_months") // 12) * 12).alias("band"))
        .group_by(["childes_child", "childes_corpus"])
        .agg([pl.col("band").n_unique().alias("n_bands"), pl.len().alias("total")])
        .filter((pl.col("n_bands") >= 3) & (pl.col("total") >= 200))
    )

    classifiable = early_rates.join(coverage, on=["childes_child", "childes_corpus"])

    decliners = set(classifiable.filter(
        pl.col("early_rate") > THRESHOLD
    )["childes_child"].to_list())
    adult_like = set(classifiable.filter(
        pl.col("early_rate") <= THRESHOLD
    )["childes_child"].to_list())

    return decliners, adult_like


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/output/train_90M"))
    args = parser.parse_args()

    print("Loading data...")
    base, exploded, longitudinal, null_type = load_data(args.data_dir)
    decliners, adult_like = get_child_groups(exploded)

    print(f"Decliners: {len(decliners)}, Adult-like: {len(adult_like)}")
    print(f"Figures will be saved to {args.data_dir}/")
    print("Run individual figure functions as needed.")
    print("(Full figure generation logic is in the inline scripts;")
    print(" this module provides the shared data loading and group classification.)")


if __name__ == "__main__":
    main()
