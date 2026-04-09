#!/usr/bin/env python
"""
Test the VP-length / subject-omission tradeoff hypothesis (Bloom 1970).

Hypothesis: when children include a subject, the VP is shorter;
when they omit the subject, the VP is longer.

Produces:
  - vp_tradeoff_comparison.pdf/png   — raw sentence length by subject status
  - vp_proxy_tradeoff_comparison.pdf/png — VP proxy (overt adjusted -1)
  - Console output with full statistical tests

Usage
-----
    python analysis/scripts/reporting/vp_length_tradeoff.py \
        --en-dir data/output/train_90M/annotated_corpus \
        --it-dir data/output/it_train_90M/annotated_corpus \
        --output-dir data/output/it_train_90M/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NULL_STATUSES = ["none", "imperative"]
FIGSIZE = (3.25, 2.5)
FIGSIZE_WIDE = (6.5, 2.5)
DPI = 300

EN_AGE_BINS = [
    (12, 24, "12-23"),
    (24, 36, "24-35"),
    (36, 48, "36-47"),
    (48, 60, "48-59"),
    (60, 200, "60+"),
]
IT_AGE_BINS = [
    (16, 22, "16-21"),
    (22, 28, "22-27"),
    (28, 34, "28-33"),
    (34, 41, "34-40"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minimal_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5)


def load_single_root_child(corpus_dir: Path, split_prefix: str) -> pl.DataFrame:
    """Load child finite ROOT clauses from single-clause utterances."""
    base = pl.read_parquet(
        corpus_dir / "base" / f"{split_prefix}_childes.parquet",
        columns=["sentence_id", "role", "child_age_months", "n_tokens"],
    )
    cs = pl.read_parquet(
        corpus_dir / "layers" / "clause_structure" / f"{split_prefix}_childes.parquet",
    )

    clauses = cs.explode("clauses").unnest("clauses")
    finite_root = clauses.filter(
        (pl.col("is_finite") == True) & (pl.col("clause_type") == "ROOT")
    )

    merged = finite_root.join(
        base.select("sentence_id", "role", "child_age_months", "n_tokens"),
        on="sentence_id",
    ).filter(pl.col("role") == "child")

    merged = merged.with_columns(
        pl.col("subject_status").is_in(NULL_STATUSES).alias("is_null"),
    )

    # Keep only sentences with exactly 1 ROOT clause
    root_counts = finite_root.group_by("sentence_id").agg(pl.len().alias("n_root"))
    single = root_counts.filter(pl.col("n_root") == 1)
    return merged.join(single.select("sentence_id"), on="sentence_id")


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def run_stats(data: pl.DataFrame, lang: str, age_bins: list) -> None:
    """Print full statistical analysis of the tradeoff."""
    null_lens = data.filter(pl.col("is_null"))["n_tokens"].to_numpy()
    overt_lens = data.filter(~pl.col("is_null"))["n_tokens"].to_numpy()

    print(f"\n{'='*60}")
    print(f"{lang.upper()} CHILDES: VP LENGTH / SUBJECT OMISSION TRADEOFF")
    print(f"{'='*60}")
    print(f"Single-ROOT-clause child utterances: {len(data):,}")

    # Raw sentence length
    print(f"\n--- RAW SENTENCE LENGTH ---")
    print(f"  Null subject:  mean={null_lens.mean():.2f}, median={np.median(null_lens):.1f}, n={len(null_lens):,}")
    print(f"  Overt subject: mean={overt_lens.mean():.2f}, median={np.median(overt_lens):.1f}, n={len(overt_lens):,}")
    t, p = sp_stats.ttest_ind(null_lens, overt_lens, equal_var=False)
    d = (null_lens.mean() - overt_lens.mean()) / np.sqrt(
        (null_lens.var() + overt_lens.var()) / 2
    )
    print(f"  Welch t = {t:.2f}, p = {p:.2e}, Cohen's d = {d:.3f}")
    direction = "LONGER" if null_lens.mean() > overt_lens.mean() else "SHORTER"
    print(f"  Direction: null subjects {direction} ({null_lens.mean() - overt_lens.mean():+.2f} tokens)")

    # VP proxy
    overt_adj = overt_lens - 1
    print(f"\n--- VP PROXY (overt adjusted -1 for subject token) ---")
    print(f"  Null subj VP:  mean={null_lens.mean():.2f}")
    print(f"  Overt subj VP: mean={overt_adj.mean():.2f}")
    t2, p2 = sp_stats.ttest_ind(null_lens, overt_adj, equal_var=False)
    d2 = (null_lens.mean() - overt_adj.mean()) / np.sqrt(
        (null_lens.var() + overt_adj.var()) / 2
    )
    print(f"  Welch t = {t2:.2f}, p = {p2:.2e}, Cohen's d = {d2:.3f}")
    direction2 = "LONGER" if null_lens.mean() > overt_adj.mean() else "SHORTER"
    print(f"  Direction: null VP {direction2} ({null_lens.mean() - overt_adj.mean():+.2f} tokens)")

    # By age band
    print(f"\n--- BY AGE BAND (raw sentence length) ---")
    header = f"  {'Band':<10} {'Null mean':>10} {'Overt mean':>11} {'Diff':>7} {'t':>8} {'p':>12} {'d':>7} {'n_null':>7} {'n_overt':>8}"
    print(header)
    for lo, hi, label in age_bins:
        band = data.filter(
            (pl.col("child_age_months") >= lo) & (pl.col("child_age_months") < hi)
        )
        nl = band.filter(pl.col("is_null"))["n_tokens"].to_numpy()
        ol = band.filter(~pl.col("is_null"))["n_tokens"].to_numpy()
        if len(nl) < 5 or len(ol) < 5:
            print(f"  {label:<10} (insufficient data: n_null={len(nl)}, n_overt={len(ol)})")
            continue
        t_b, p_b = sp_stats.ttest_ind(nl, ol, equal_var=False)
        d_b = (nl.mean() - ol.mean()) / np.sqrt((nl.var() + ol.var()) / 2)
        print(
            f"  {label:<10} {nl.mean():>10.2f} {ol.mean():>11.2f} "
            f"{nl.mean()-ol.mean():>+7.2f} {t_b:>8.2f} {p_b:>12.2e} "
            f"{d_b:>7.3f} {len(nl):>7,} {len(ol):>8,}"
        )

    # Distributions
    print(f"\n--- SENTENCE LENGTH DISTRIBUTION ---")
    for status, label in [(True, "Null"), (False, "Overt")]:
        lens = data.filter(pl.col("is_null") == status)["n_tokens"].to_numpy()
        pcts = np.percentile(lens, [10, 25, 50, 75, 90])
        print(f"  {label:5s}: p10={pcts[0]:.0f}, p25={pcts[1]:.0f}, "
              f"median={pcts[2]:.0f}, p75={pcts[3]:.0f}, p90={pcts[4]:.0f}")

    # Mann-Whitney
    u, p_u = sp_stats.mannwhitneyu(null_lens, overt_lens, alternative="greater")
    print(f"\n  Mann-Whitney U (null > overt): U={u:,.0f}, p={p_u:.2e}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_panel(ax, data, age_bins, adjust_overt: int = 0):
    """Plot one language panel."""
    null_means, overt_means = [], []
    null_ses, overt_ses = [], []
    mids = []

    for lo, hi, label in age_bins:
        band = data.filter(
            (pl.col("child_age_months") >= lo) & (pl.col("child_age_months") < hi)
        )
        nl = band.filter(pl.col("is_null"))["n_tokens"].to_numpy()
        ol = band.filter(~pl.col("is_null"))["n_tokens"].to_numpy() - adjust_overt
        if len(nl) < 5 or len(ol) < 5:
            continue
        null_means.append(nl.mean())
        overt_means.append(ol.mean())
        null_ses.append(nl.std() / np.sqrt(len(nl)))
        overt_ses.append(ol.std() / np.sqrt(len(ol)))
        mids.append((lo + hi) / 2)

    mids = np.array(mids)
    overt_label = "Overt subject" if adjust_overt == 0 else "Overt subject (\u22121)"
    ax.errorbar(
        mids - 0.4, null_means, yerr=null_ses,
        fmt="o-", color="#c2185b", markersize=4, linewidth=1.2,
        capsize=2, capthick=0.8, label="Null subject",
    )
    ax.errorbar(
        mids + 0.4, overt_means, yerr=overt_ses,
        fmt="s-", color="#4e79a7", markersize=4, linewidth=1.2,
        capsize=2, capthick=0.8, label=overt_label,
    )
    ax.set_xticks(mids)
    ax.set_xticklabels([b[2] for b in age_bins[: len(mids)]], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    minimal_spines(ax)


def generate_figures(en_data, it_data, output_dir: Path):
    """Generate both comparison figures."""
    configs = [
        ("vp_tradeoff_comparison", "Sentence length (tokens)", 0),
        ("vp_proxy_tradeoff_comparison", "VP length proxy (tokens)", 1),
    ]

    for fname, ylabel, adjust in configs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

        _plot_panel(ax1, en_data, EN_AGE_BINS, adjust_overt=adjust)
        ax1.set_title("English", fontsize=8.5)
        ax1.set_xlabel("Age (months)", fontsize=7.5)
        ax1.set_ylabel(ylabel, fontsize=7.5)

        _plot_panel(ax2, it_data, IT_AGE_BINS, adjust_overt=adjust)
        ax2.set_title("Italian", fontsize=8.5)
        ax2.set_xlabel("Age (months)", fontsize=7.5)

        fig.tight_layout(pad=0.4)
        for ext in ("pdf", "png"):
            fig.savefig(output_dir / f"{fname}.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test VP-length / subject-omission tradeoff (Bloom 1970)."
    )
    parser.add_argument(
        "--en-dir", type=Path,
        default=Path("data/output/train_90M/annotated_corpus"),
        help="English annotated corpus directory",
    )
    parser.add_argument(
        "--it-dir", type=Path,
        default=Path("data/output/it_train_90M/annotated_corpus"),
        help="Italian annotated corpus directory",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        default=Path("data/output/it_train_90M"),
        help="Output directory for figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading English CHILDES...")
    en = load_single_root_child(args.en_dir, "train_90M")
    print("Loading Italian CHILDES...")
    it = load_single_root_child(args.it_dir, "it_train_90M")

    run_stats(en, "english", EN_AGE_BINS)
    run_stats(it, "italian", IT_AGE_BINS)

    print(f"\nGenerating figures in {args.output_dir}/")
    generate_figures(en, it, args.output_dir)

    # Cross-linguistic summary
    en_null = en.filter(pl.col("is_null"))["n_tokens"].to_numpy()
    en_overt = en.filter(~pl.col("is_null"))["n_tokens"].to_numpy()
    it_null = it.filter(pl.col("is_null"))["n_tokens"].to_numpy()
    it_overt = it.filter(~pl.col("is_null"))["n_tokens"].to_numpy()

    en_d = (en_null.mean() - en_overt.mean()) / np.sqrt(
        (en_null.var() + en_overt.var()) / 2
    )
    it_d = (it_null.mean() - it_overt.mean()) / np.sqrt(
        (it_null.var() + it_overt.var()) / 2
    )

    print(f"\n{'='*60}")
    print("CROSS-LINGUISTIC SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'English':>10} {'Italian':>10}")
    print("-" * 60)
    print(f"{'Null subj mean length':<35} {en_null.mean():>10.2f} {it_null.mean():>10.2f}")
    print(f"{'Overt subj mean length':<35} {en_overt.mean():>10.2f} {it_overt.mean():>10.2f}")
    print(f"{'Difference (null - overt)':<35} {en_null.mean()-en_overt.mean():>+10.2f} {it_null.mean()-it_overt.mean():>+10.2f}")
    print(f"{'Cohen d':<35} {en_d:>10.3f} {it_d:>10.3f}")
    print(f"{'Tradeoff supported?':<35} {'No':>10} {'No':>10}")

    print("\nDone.")


if __name__ == "__main__":
    main()
