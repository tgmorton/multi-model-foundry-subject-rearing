"""
Causal-vs-BERT backward-depth comparison (Thomas 2026-08-23): does a
causal LM, trained on the backward view alone, derive more from added
backward context than BERT, whose pretraining always had the forward
view (the verb) available?

Design: pretrained gpt2-medium scored the frozen 100K locality sample at
exactly the BERT backward-only depths L in {1,2,4,8,16,32,64,125,250,500}
(R structurally 0). Paired per-instance comparison against BERT's L:0
rows from the locality long table.

Signatures tested:
1. Depth-utilization profile — share of the model's own total backward
   gain (L=1 -> 500) captured by depth L. Scale-free; if BERT's profile
   sits above gpt2m's at small L (or its total gain is a smaller multiple
   of its L=1 surprisal), BERT extracts less from deep history.
2. Marginal paired gain per depth doubling (median of surp(L_prev)-surp(L)).
3. Rank sharpening — rho(config, in-house causal ensemble) vs L for both
   models; BERT climbed 0.35 -> 0.54, still rising at L=500.
4. Cross-model rank agreement at matched L, and person decomposition.

Outputs: analysis/locality/causal_backward_metrics.csv, _by_person.csv,
figures/causal_backward.png, printed summary for the report.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("data/recoverability")
GRID = ROOT / "train_90M" / "external_gpt2m_causal" / "grid"
OUT = ROOT / "analysis" / "locality"
DEPTHS = [1, 2, 4, 8, 16, 32, 64, 125, 250, 500]


def load_gpt2m() -> pd.DataFrame:
    frames = []
    for L in DEPTHS:
        for f in glob.glob(str(GRID / f"L{L}R0" / "*.parquet")):
            df = pq.read_table(
                f, columns=["line_idx", "token_i", "form", "person",
                            "logprob_sum"]).to_pandas()
            df["genre"] = Path(f).stem
            df["L"] = L
            frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d["surp"] = -d.logprob_sum
    return d.drop(columns=["logprob_sum"])


def main() -> None:
    g = load_gpt2m()
    print(f"gpt2m grid: {len(g):,} rows "
          f"({g.L.nunique()} depths x {len(g)//g.L.nunique():,} instances)")

    long = pd.read_parquet(
        OUT / "locality_long.parquet",
        columns=["genre", "line_idx", "token_i", "L", "R", "surp",
                 "surp_form__clean", "surp_ceiling"])
    b = long[(long.R == 0) & (long.L.isin(DEPTHS))][
        ["genre", "line_idx", "token_i", "L", "surp"]].rename(
        columns={"surp": "surp_bert"})
    refs = long[(long.L == 250) & (long.R == 0)][
        ["genre", "line_idx", "token_i", "surp_form__clean"]]

    df = g.merge(b, on=["genre", "line_idx", "token_i", "L"], how="inner")
    df = df.merge(refs, on=["genre", "line_idx", "token_i"], how="left")
    print(f"paired: {len(df):,} rows")

    # wide per-instance tables for paired depth deltas
    wg = df.pivot_table(index=["genre", "line_idx", "token_i"],
                        columns="L", values="surp")
    wb = df.pivot_table(index=["genre", "line_idx", "token_i"],
                        columns="L", values="surp_bert")

    rows = []
    for model, w in (("gpt2m_causal", wg), ("bert_L:0", wb)):
        total_gain = (w[1] - w[500]).median()
        for i, L in enumerate(DEPTHS):
            s = w[L].dropna()
            sub = df[df.L == L]
            col = "surp" if model == "gpt2m_causal" else "surp_bert"
            rows.append({
                "model": model, "L": L,
                "median": s.median(),
                "pct_lt_01": (s < 0.1).mean(),
                # share of the model's own total backward gain captured by L
                "depth_share": float((w[1] - w[L]).median() / total_gain)
                               if total_gain else np.nan,
                # paired marginal gain from the previous depth
                "marginal_gain": float((w[DEPTHS[i - 1]] - w[L]).median())
                                 if i else np.nan,
                "rho_clean": sub[col].corr(sub.surp_form__clean,
                                           method="spearman"),
                "rho_cross_model": sub.surp.corr(sub.surp_bert,
                                                 method="spearman"),
            })
    met = pd.DataFrame(rows)
    print("\n== backward-depth profiles (paired, same instances) ==")
    print(met.round(3).to_string(index=False))
    met.to_csv(OUT / "causal_backward_metrics.csv", index=False)

    # person decomposition
    pers = (df[df.person.isin([1, 2, 3])]
            .groupby(["person", "L"])[["surp", "surp_bert"]].median()
            .round(3))
    print("\n== median surprisal by person ==")
    print(pers.to_string())
    pers.to_csv(OUT / "causal_backward_by_person.csv")

    # headline numbers for the report
    for model, w in (("gpt2m_causal", wg), ("bert_L:0", wb)):
        deep = (w[16] - w[500]).median()   # gain from beyond-local history
        total = (w[1] - w[500]).median()
        print(f"\n{model}: total gain L1->500 = {total:.3f} nats; "
              f"gain from L16->500 (deep history) = {deep:.3f} nats "
              f"({deep / total:.1%} of total)")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    mg = met[met.model == "gpt2m_causal"]
    mb = met[met.model == "bert_L:0"]
    ax = axes[0]
    ax.plot(mg.L, mg["median"], marker="o", label="gpt2-medium (causal)")
    ax.plot(mb.L, mb["median"], marker="s", label="BERT L:0")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("backward depth L (model's own pieces)")
    ax.set_ylabel("median surprisal (nats)")
    ax.legend(); ax.set_title("Backward-only depth curves")
    ax = axes[1]
    ax.plot(mg.L, mg.depth_share, marker="o", label="gpt2-medium (causal)")
    ax.plot(mb.L, mb.depth_share, marker="s", label="BERT L:0")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("backward depth L"); ax.set_ylabel("share of own total gain")
    ax.legend(); ax.set_title("Depth-utilization profile (scale-free)")
    ax = axes[2]
    ax.plot(mg.L, mg.rho_clean, marker="o", label="gpt2-medium (causal)")
    ax.plot(mb.L, mb.rho_clean, marker="s", label="BERT L:0")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("backward depth L")
    ax.set_ylabel("rho vs in-house causal ensemble")
    ax.legend(); ax.set_title("Rank sharpening with depth")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "causal_backward.png", dpi=150)
    print(f"\nwrote {OUT}/causal_backward_metrics.csv, "
          "causal_backward_by_person.csv, figures/causal_backward.png")


if __name__ == "__main__":
    main()
