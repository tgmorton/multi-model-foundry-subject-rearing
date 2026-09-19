"""Locality-variant comparison (Thomas 2026-09-18/19):
BERT vs RoBERTa, truncation vs masked-context windows.

Questions:
1. Artifact check: does BERT's non-monotonic short-L backward behavior
   (L=4 worse than L=2 under truncation) survive when the input keeps
   its natural full-window length and context is MASKED instead?
2. Cross-mode anchor: masked 250:250 must reproduce truncated 250:250
   (nothing masked) — validates comparability.
3. Does RoBERTa reproduce BERT's landscape (saturation from forward
   context, backward non-saturation, speaker-agreement profile)?

Inputs: external_{bert_wwm,roberta}/{grid,grid_mask} on the frozen
sample; clean-ensemble reference from locality_long.parquet.
Outputs: analysis/locality/variant_metrics.csv,
figures/variant_comparison.png, printed summary.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("data/recoverability")
OUT = ROOT / "analysis" / "locality"
GRIDS = {
    ("bert", "trunc"): ROOT / "train_90M/external_bert_wwm/grid",
    ("bert", "mask"): ROOT / "train_90M/external_bert_wwm/grid_mask",
    ("roberta", "trunc"): ROOT / "train_90M/external_roberta/grid",
    ("roberta", "mask"): ROOT / "train_90M/external_roberta/grid_mask",
}


def load_grid(root: Path, model: str, mode: str) -> pd.DataFrame:
    frames = []
    for cfg_dir in sorted(root.glob("L*R*")):
        L, R = cfg_dir.name[1:].split("R")
        if R in ("V", "VX"):
            continue
        for f in glob.glob(str(cfg_dir / "*.parquet")):
            d = pq.read_table(f, columns=["line_idx", "token_i",
                                          "logprob_sum"]).to_pandas()
            d["genre"] = Path(f).stem
            d["L"], d["R"] = int(L), int(R)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["surp"] = -df.logprob_sum
    df["model"], df["mode"] = model, mode
    return df.drop(columns=["logprob_sum"])


def main() -> None:
    refs = pd.read_parquet(
        OUT / "locality_long.parquet",
        columns=["genre", "line_idx", "token_i", "L", "R",
                 "surp_form__clean"])
    clean = refs[(refs.L == 250) & (refs.R == 0)][
        ["genre", "line_idx", "token_i", "surp_form__clean"]]

    frames = [load_grid(p, m, md) for (m, md), p in GRIDS.items()
              if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(clean, on=["genre", "line_idx", "token_i"], how="left")
    print(f"assembled {len(df):,} rows over "
          f"{df.groupby(['model', 'mode']).ngroups} (model, mode) sets")

    met = (df.groupby(["model", "mode", "L", "R"])
           .agg(n=("surp", "size"), median=("surp", "median"),
                tie=("surp", lambda s: (s < 0.1).mean()))
           .reset_index())
    rho = (df.groupby(["model", "mode", "L", "R"])
           .apply(lambda g: g.surp.corr(g.surp_form__clean,
                                        method="spearman"))
           .rename("rho_clean").reset_index())
    met = met.merge(rho, on=["model", "mode", "L", "R"])
    met.to_csv(OUT / "variant_metrics.csv", index=False)

    # ---- headline tests ----
    print("\n== 1. short-L backward non-monotonicity (truncation artifact?) ==")
    for model in ("bert", "roberta"):
        for mode in ("trunc", "mask"):
            b = met[(met.model == model) & (met["mode"] == mode)
                    & (met.R == 0) & (met.L.isin([1, 2, 4, 8]))]
            if not len(b):
                continue
            curve = b.sort_values("L")["median"].round(3).tolist()
            mono = all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))
            print(f"  {model}/{mode}: L=1,2,4,8 medians {curve} "
                  f"{'MONOTONE' if mono else 'NON-MONOTONE'}")

    print("\n== 2. cross-mode anchor (250:250) ==")
    for model in ("bert", "roberta"):
        a = met[(met.model == model) & (met.L == 250) & (met.R == 250)]
        for _, r in a.iterrows():
            print(f"  {model}/{r['mode']}: median {r['median']:.4f} "
                  f"tie {r.tie:.3f} rho {r.rho_clean:.3f}")

    print("\n== 3. landscape (L=250 backward / R=250 forward / ceiling) ==")
    for model in ("bert", "roberta"):
        for mode in ("trunc", "mask"):
            sub = met[(met.model == model) & (met["mode"] == mode)]
            for (L, R, tag) in ((250, 0, "bwd250"), (0, 250, "fwd250")):
                row = sub[(sub.L == L) & (sub.R == R)]
                if len(row):
                    r = row.iloc[0]
                    print(f"  {model}/{mode} {tag}: median {r['median']:.3f} "
                          f"tie {r.tie:.3f} rho_clean {r.rho_clean:.3f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TEAL, SIENNA, GREY, DARK = "#2C6E63", "#B0562B", "#8A8878", "#3A3A35"
    style = {("bert", "trunc"): (DARK, "-"), ("bert", "mask"): (DARK, "--"),
             ("roberta", "trunc"): (SIENNA, "-"),
             ("roberta", "mask"): (SIENNA, "--")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for (model, mode), (c, ls) in style.items():
        sub = met[(met.model == model) & (met["mode"] == mode)]
        bwd = sub[(sub.R == 0) & (sub.L > 0)].sort_values("L")
        fwd = sub[(sub.L == 0) & (sub.R > 0)].sort_values("R")
        lab = f"{model} {mode}"
        if len(bwd):
            axes[0].plot(bwd.L, bwd["median"], ls, color=c, marker="o",
                         ms=3, label=lab)
            axes[2].plot(bwd.L, bwd.rho_clean, ls, color=c, marker="o",
                         ms=3, label=lab)
        if len(fwd):
            axes[1].plot(fwd.R, fwd["median"], ls, color=c, marker="s",
                         ms=3, label=lab)
    for ax, t, xl in ((axes[0], "Backward-only family", "L (backward)"),
                      (axes[1], "Forward-only family", "R (forward)"),
                      (axes[2], "Backward speaker-agreement", "L (backward)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
        ax.set_title(t)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("median surprisal (nats)")
    axes[2].set_ylabel("ρ vs causal clean ensemble")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "variant_comparison.png", dpi=150)
    print(f"\nwrote {OUT}/variant_metrics.csv, figures/variant_comparison.png")


if __name__ == "__main__":
    main()
