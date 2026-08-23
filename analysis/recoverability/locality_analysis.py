"""
BERT context-locality analysis (Thomas's directive 2026-08-24).

Assembles the (L, R) grid scorings of the frozen 100K sample into the
locality landscape: how much context — symmetric, backward-only,
forward-only, and asymmetric (the "speaker model" region L>>R>0) — the
masked-recovery measure needs, where it saturates, and how each config's
RANKING relates to the causal ensemble and to the full-context ceiling.

Inputs:
  data/recoverability/train_90M/external_bert_wwm/grid/L{l}R{r}/<genre>.parquet
  data/recoverability/locality_sample/sample_index.parquet
  data/recoverability/analysis/instance_measures/<genre>.parquet  (clean)
  data/recoverability/train_90M/external_gpt2_medium/instances/<genre>.parquet

Outputs (under data/recoverability/analysis/locality/):
  locality_long.parquet      instance x config long table
  config_metrics.csv         one row per (L, R) with all metrics
  figures/*.png              saturation curves, asymmetry, heatmaps,
                             convergence, person decomposition
  LOCALITY_REPORT.md         the write-up with headline numbers
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("data/recoverability")
GRID = ROOT / "train_90M" / "external_bert_wwm" / "grid"
OUT = ROOT / "analysis" / "locality"
GENRES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles",
          "simple_wiki", "switchboard"]
_CFG = re.compile(r"L(\d+)R(\d+)$")
PERSON_GROUP = {1.0: "1st", 2.0: "2nd", 3.0: "3rd"}


def load_long() -> pd.DataFrame:
    frames = []
    for cfg_dir in sorted(GRID.iterdir()):
        m = _CFG.search(cfg_dir.name)
        if not m:
            continue
        L, R = int(m.group(1)), int(m.group(2))
        for f in glob.glob(str(cfg_dir / "*.parquet")):
            genre = Path(f).stem
            df = pq.read_table(f).to_pandas()
            df["L"], df["R"], df["genre"] = L, R, genre
            frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["surp"] = -long.logprob_sum
    return long


def load_refs() -> pd.DataFrame:
    refs = []
    for g in GENRES:
        clean = pq.read_table(
            ROOT / "analysis" / "instance_measures" / f"{g}.parquet",
            columns=["line_idx", "token_i", "surp_form__clean"]).to_pandas()
        g2m = pq.read_table(
            ROOT / "train_90M" / "external_gpt2_medium" / "instances"
            / f"{g}.parquet",
            columns=["line_idx", "token_i", "gpt2_medium__logprob_sum"]
        ).to_pandas()
        g2m["surp_g2m"] = -g2m.gpt2_medium__logprob_sum
        r = clean.merge(g2m[["line_idx", "token_i", "surp_g2m"]],
                        on=["line_idx", "token_i"], how="outer")
        r["genre"] = g
        refs.append(r)
    return pd.concat(refs, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    long = load_long()
    refs = load_refs()
    long = long.merge(refs, on=["genre", "line_idx", "token_i"], how="left")
    ceiling = (long[(long.L == 250) & (long.R == 250)]
               [["genre", "line_idx", "token_i", "surp"]]
               .rename(columns={"surp": "surp_ceiling"}))
    long = long.merge(ceiling, on=["genre", "line_idx", "token_i"], how="left")
    long.to_parquet(OUT / "locality_long.parquet")
    print(f"long table: {len(long):,} rows, "
          f"{long[['L','R']].drop_duplicates().shape[0]} configs")

    rows = []
    for (L, R), g in long.groupby(["L", "R"]):
        s = g.surp.dropna()
        row = {"L": L, "R": R, "n": len(s),
               "median": s.median(), "mean": s.mean(), "sd": s.std(),
               "pct_lt_001": (s < 0.01).mean(),
               "pct_lt_01": (s < 0.1).mean(),
               "pct_lt_05": (s < 0.5).mean(),
               "p90": s.quantile(0.9),
               "rho_clean": g.surp.corr(g.surp_form__clean, method="spearman"),
               "rho_g2m": g.surp.corr(g.surp_g2m, method="spearman"),
               "rho_ceiling": g.surp.corr(g.surp_ceiling, method="spearman")}
        for pval, pname in PERSON_GROUP.items():
            sub = g[g.person == pval].surp.dropna()
            row[f"median_{pname}"] = sub.median() if len(sub) else np.nan
            row[f"pct01_{pname}"] = (sub < 0.1).mean() if len(sub) else np.nan
        rows.append(row)
    met = pd.DataFrame(rows).sort_values(["L", "R"])
    met.to_csv(OUT / "config_metrics.csv", index=False)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sym = met[met.L == met.R].sort_values("L")
    lonly = met[met.R == 0].sort_values("L")
    ronly = met[met.L == 0].sort_values("R")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for d, lbl, mk in ((sym, "symmetric L=R", "o"),
                       (lonly, "backward only (R=0)", "s"),
                       (ronly, "forward only (L=0)", "^")):
        x = d.L if lbl != "forward only (L=0)" else d.R
        ax.plot(np.maximum(x, 0.5), d["median"], marker=mk, label=lbl)
    ax.set_xscale("log"); ax.set_xlabel("context (wordpieces)")
    ax.set_ylabel("median surprisal (nats)"); ax.legend(); ax.set_title("Saturation")
    ax = axes[1]
    for d, lbl, mk in ((sym, "symmetric", "o"), (lonly, "backward", "s"),
                       (ronly, "forward", "^")):
        x = d.L if lbl != "forward" else d.R
        ax.plot(np.maximum(x, 0.5), d.pct_lt_01, marker=mk, label=lbl)
    ax.set_xscale("log"); ax.set_ylabel("fraction < 0.1 nats (tie mass)")
    ax.set_xlabel("context (wordpieces)"); ax.legend(); ax.set_title("Tie mass")
    ax = axes[2]
    for d, lbl, mk in ((sym, "symmetric", "o"), (lonly, "backward", "s"),
                       (ronly, "forward", "^")):
        x = d.L if lbl != "forward" else d.R
        ax.plot(np.maximum(x, 0.5), d.rho_clean, marker=mk, label=lbl)
    ax.set_xscale("log"); ax.set_ylabel("Spearman vs causal clean ensemble")
    ax.set_xlabel("context (wordpieces)"); ax.legend(); ax.set_title("Rank agreement")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "locality_curves.png", dpi=150)

    # asymmetric heatmaps over the cross region
    cross = met[(met.L.isin([16, 32, 64, 250])) & (met.R > 0) & (met.L != met.R) |
                ((met.L.isin([16, 64, 250])) & (met.R.isin([1, 2, 4, 8, 16, 32, 64])))]
    for col, title in (("median", "median surprisal"),
                       ("pct_lt_01", "tie mass (<0.1 nats)"),
                       ("rho_clean", "rho vs causal ensemble")):
        piv = met[met.R > 0].pivot_table(index="L", columns="R", values=col)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(piv.values, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("R (forward)"); ax.set_ylabel("L (backward)")
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7)
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(OUT / "figures" / f"heatmap_{col}.png", dpi=150)

    # person decomposition on backward-only
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for pname in ("1st", "2nd", "3rd"):
        ax.plot(np.maximum(lonly.L, 0.5), lonly[f"median_{pname}"],
                marker="o", label=pname)
    ax.set_xscale("log"); ax.set_xlabel("backward context (wordpieces, R=0)")
    ax.set_ylabel("median surprisal (nats)"); ax.legend()
    ax.set_title("Locality need by person")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "person_backward.png", dpi=150)

    print(met[["L", "R", "median", "pct_lt_01", "rho_clean",
               "rho_ceiling"]].to_string(index=False))
    print(f"\nwrote {OUT}/config_metrics.csv + figures/")


if __name__ == "__main__":
    main()
