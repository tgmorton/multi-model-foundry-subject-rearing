"""
Verb-frame planning analysis (Thomas 2026-08-24): the speaker-planner
window — full history + forward context exactly to the subject's
DEPENDENCY-HEAD verb ("matrix verb" = whatever the graph says).

Questions answered:
1. What does the window look like? head-gap distribution (pieces ≈
   words×1.3): adjacency rate, intervening-material lengths, inversion
   (verb-before-pronoun) rate — by genre and person.
2. How does surprisal change with verb distance? Curves for DYN-incl
   (through head), DYN-excl (to head, exclusive), and fixed references —
   plus the verb's own information contribution (VX→V delta) and the
   intervening material's contribution (L-only→VX delta) by gap bin.
3. Where do the DYN configs sit in the locality landscape? Saturation,
   tie mass, rank agreement vs the causal ensemble / fixed-R candidates /
   the bidirectional ceiling.

Inputs: grid dirs L250RV, L250RVX, L64RV + the locality long table.
Outputs: data/recoverability/analysis/locality/verbframe_metrics.csv,
figures/verbframe_*.png, printed summary for the report.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("data/recoverability")
GRID = ROOT / "train_90M" / "external_bert_wwm" / "grid"
OUT = ROOT / "analysis" / "locality"
GAP_BINS = [(-99, -1, "verb-before"), (0, 0, "adjacent"),
            (1, 1, "gap 1"), (2, 2, "gap 2"), (3, 4, "gap 3-4"),
            (5, 8, "gap 5-8"), (9, 16, "gap 9-16"), (17, 999, "gap 17+")]


def load_cfg(name: str, col: str) -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(GRID / name / "*.parquet")):
        df = pq.read_table(f).to_pandas()
        df["genre"] = Path(f).stem
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d[col] = -d.logprob_sum
    keep = ["genre", "line_idx", "token_i", col]
    if "head_gap" in d.columns:
        keep += ["head_gap", "head_n_pieces", "form", "person"]
    return d[keep]


def gap_label(g):
    if pd.isna(g):
        return None
    for lo, hi, name in GAP_BINS:
        if lo <= g <= hi:
            return name
    return None


def main() -> None:
    v = load_cfg("L250RV", "surp_V")
    vx = load_cfg("L250RVX", "surp_VX")[["genre", "line_idx", "token_i", "surp_VX"]]
    v64 = load_cfg("L64RV", "surp_V64")[["genre", "line_idx", "token_i", "surp_V64"]]
    df = v.merge(vx, on=["genre", "line_idx", "token_i"]).merge(
        v64, on=["genre", "line_idx", "token_i"])

    # references from the locality long table
    long = pd.read_parquet(OUT / "locality_long.parquet",
                           columns=["genre", "line_idx", "token_i",
                                    "L", "R", "surp", "surp_form__clean",
                                    "surp_ceiling"])
    for L, R, name in ((250, 0, "surp_L0"), (250, 1, "surp_R1"),
                       (250, 2, "surp_R2")):
        ref = long[(long.L == L) & (long.R == R)][
            ["genre", "line_idx", "token_i", "surp"]].rename(
            columns={"surp": name})
        df = df.merge(ref, on=["genre", "line_idx", "token_i"])
    refs = long[(long.L == 250) & (long.R == 0)][
        ["genre", "line_idx", "token_i", "surp_form__clean", "surp_ceiling"]]
    df = df.merge(refs, on=["genre", "line_idx", "token_i"])
    df["gap_bin"] = df.head_gap.map(gap_label)
    print(f"assembled: {len(df):,} instances")

    # ---- 1. window shape ----
    print("\n== head-gap distribution (wordpieces between pronoun end and "
          "head start) ==")
    g = df.head_gap.dropna()
    print(f"aligned heads: {len(g):,}/{len(df):,} "
          f"({len(g)/len(df):.1%}); verb-before-pronoun: {(g<0).mean():.1%}; "
          f"adjacent: {(g==0).mean():.1%}; median gap: {g.median():.0f}; "
          f"p90: {g.quantile(.9):.0f}; p99: {g.quantile(.99):.0f}")
    print("\nby genre: " + ", ".join(
        f"{gn}: med={sub.head_gap.median():.0f} adj={(sub.head_gap==0).mean():.0%}"
        for gn, sub in df.groupby("genre")))
    print("by person: " + ", ".join(
        f"{int(p)}: med={sub.head_gap.median():.0f}"
        for p, sub in df.groupby("person") if p in (1, 2, 3)))

    # ---- 2. surprisal vs distance + decompositions ----
    order = [b[2] for b in GAP_BINS]
    tab = (df.groupby("gap_bin")
           .agg(n=("surp_V", "size"),
                V=("surp_V", "median"), VX=("surp_VX", "median"),
                L_only=("surp_L0", "median"),
                verb_contrib=("surp_V", lambda s: np.nan),  # filled below
                )
           .reindex(order))
    # per-instance deltas, medianed by bin (paired, not median-of-medians)
    df["verb_contrib"] = df.surp_VX - df.surp_V        # info the verb adds
    df["interv_contrib"] = df.surp_L0 - df.surp_VX     # info of intervening material
    for colname in ("verb_contrib", "interv_contrib"):
        tab[colname] = df.groupby("gap_bin")[colname].median().reindex(order)
    tab["tie_V"] = df.groupby("gap_bin").surp_V.apply(
        lambda s: (s < 0.1).mean()).reindex(order)
    print("\n== surprisal by verb distance (medians, nats) ==")
    print(tab.round(3).to_string())

    # ---- 3. landscape placement ----
    rows = []
    for col, label in (("surp_V", "DYN-incl 250:V"),
                       ("surp_VX", "DYN-excl 250:VX"),
                       ("surp_V64", "DYN-incl 64:V"),
                       ("surp_R1", "fixed 250:1"),
                       ("surp_R2", "fixed 250:2"),
                       ("surp_L0", "backward-only 250:0")):
        s = df[col].dropna()
        rows.append({
            "config": label, "median": s.median(),
            "pct_lt_01": (s < 0.1).mean(),
            "rho_clean": df[col].corr(df.surp_form__clean, method="spearman"),
            "rho_R1": df[col].corr(df.surp_R1, method="spearman"),
            "rho_ceiling": df[col].corr(df.surp_ceiling, method="spearman"),
        })
    met = pd.DataFrame(rows)
    print("\n== landscape placement ==")
    print(met.round(3).to_string(index=False))
    met.to_csv(OUT / "verbframe_metrics.csv", index=False)
    tab.to_csv(OUT / "verbframe_by_gap.csv")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    sub = df[df.head_gap.between(0, 20)]
    med = sub.groupby("head_gap")[["surp_V", "surp_VX", "surp_L0"]].median()
    ax.plot(med.index, med.surp_V, marker="o", label="DYN-incl (verb visible)")
    ax.plot(med.index, med.surp_VX, marker="s", label="DYN-excl")
    ax.plot(med.index, med.surp_L0, marker="^", label="backward-only")
    ax.set_xlabel("verb distance (pieces)"); ax.set_ylabel("median surprisal")
    ax.legend(); ax.set_title("Surprisal vs verb distance")
    ax = axes[1]
    med2 = sub.groupby("head_gap")[["verb_contrib", "interv_contrib"]].median()
    ax.plot(med2.index, med2.verb_contrib, marker="o", label="verb's contribution (VX−V)")
    ax.plot(med2.index, med2.interv_contrib, marker="s",
            label="intervening material (L-only−VX)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("verb distance (pieces)"); ax.set_ylabel("median Δ surprisal")
    ax.legend(); ax.set_title("Information decomposition")
    ax = axes[2]
    df.head_gap.clip(-2, 25).hist(bins=28, ax=ax)
    ax.set_xlabel("verb distance (pieces, clipped)"); ax.set_title("Gap distribution")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "verbframe.png", dpi=150)
    print(f"\nwrote {OUT}/verbframe_metrics.csv, verbframe_by_gap.csv, "
          "figures/verbframe.png")


if __name__ == "__main__":
    main()
