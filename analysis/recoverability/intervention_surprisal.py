"""How interventions change each pronoun's circumstances (Thomas 2026-09-28).

Paired, per-instance: the SAME baseline-identified subject pronoun, scored
by the selection rater (BERT 250:1) in baseline context and in each
intervention's edited context.

Questions:
 1. Validity: baseline-via-intervention-path must reproduce the existing
    L250R1 locality-grid scores (identity check on the machinery).
 2. Shift: does each intervention make pronouns more or less recoverable?
    (licensing prediction: lemmatize raises surprisal — the verb no longer
    predicts the subject; enrich lowers it — agreement re-marks person.)
 3. Rank stability: does "most recoverable" survive the intervention?
    rho(baseline, intervention) and decile migration — the direct test of
    D4's assumption that baseline-ranked deciles mean the same thing in
    every intervention.
 4. Where: shift conditioned on whether the head verb was edited, and by
    person/number.
 5. remove_expletive: fraction of sample pronouns on removed lines.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("data/recoverability")
# Defaults = the BERT 250:1 run (2026-09-28); --scorer/--grid/--out switch
# raters (RoBERTa ±250: external_roberta_bi, external_roberta/grid/L250R250).
IVROOT = ROOT / "train_90M/external_bert_wwm/intervention"
OUT = ROOT / "analysis/intervention"
GRID = ROOT / "train_90M/external_bert_wwm/grid/L250R1"
RATER = "BERT 250:1"
FLAGS = ROOT / "analysis/intervention/sample_contraction_flags.parquet"
IVS = ["lemmatize_verbs", "enrich_verbal_morphology", "impoverish_case",
       "remove_expletive_sentences"]
KEY = ["genre", "line_idx", "token_i"]


def load(iv: str) -> pd.DataFrame:
    fr = []
    for f in glob.glob(str(IVROOT / iv / "*.parquet")):
        if f.endswith(".line_removed.parquet"):
            continue
        d = pq.read_table(f).to_pandas()
        d["genre"] = Path(f).stem
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["surp"] = -d.logprob_sum
    bad = d.surp.isna()
    if bad.any():
        by = d[bad].genre.value_counts().to_dict()
        print(f"  WARNING {iv}: {bad.sum():,} NaN scores dropped {by}")
        d = d[~bad]
    return d


def removed(iv: str) -> pd.DataFrame:
    fr = [pq.read_table(f).to_pandas().assign(genre=Path(f).name.split(".")[0])
          for f in glob.glob(str(IVROOT / iv / "*.line_removed.parquet"))]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=KEY)


CLITICS = {"'s", "'re", "'ve", "'m", "'ll", "'d",
           "\u2019s", "\u2019re", "\u2019ve", "\u2019m", "\u2019ll", "\u2019d"}


def contraction_flags() -> pd.DataFrame:
    """Per sample instance: is the pronoun followed by a clitic ("they've")?

    The selection rater's R1 window is ONE wordpiece, so for a contracted
    pronoun it sees only the apostrophe — baseline surprisal is inflated
    ~2.7x, and lemmatize/enrich (which un-contract) expose the verb. Every
    shift must be read split by this flag. spaCy's rule tokenizer only
    (identical token indices to the trf parse; checked 100% form match).
    """
    cache = FLAGS  # shared across raters: depends only on the sample text
    if cache.exists():
        return pd.read_parquet(cache)[KEY + ["contracted"]]
    import spacy
    nlp = spacy.blank("en")
    rows = []
    for f in sorted((ROOT / "locality_sample").glob("*.parquet")):
        if f.stem == "sample_index":
            continue
        s = pq.read_table(f).to_pandas()
        want, lines = set(s.line_idx), {}
        with open(Path("data/raw/train_90M") / f"{f.stem}.train") as fh:
            for i, ln in enumerate(fh):
                if i in want:
                    lines[i] = ln.rstrip("\n\r")
        for li, ti in zip(s.line_idx, s.token_i):
            d = nlp(lines.get(li, ""))
            nxt = d[ti + 1].text.lower() if ti + 1 < len(d) else ""
            rows.append((f.stem, li, ti, nxt in CLITICS))
    t = pd.DataFrame(rows, columns=KEY + ["contracted"])
    t.to_parquet(cache)
    return t


def main() -> None:
    import argparse
    global IVROOT, OUT, GRID, RATER
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="external_bert_wwm",
                    help="scorer dir under train_90M/ holding intervention/<IV>/")
    ap.add_argument("--grid", default="external_bert_wwm/grid/L250R1",
                    help="baseline grid under train_90M/ for the identity check")
    ap.add_argument("--out", default="analysis/intervention",
                    help="output dir under data/recoverability/")
    ap.add_argument("--rater", default="BERT 250:1", help="label for the figure")
    a = ap.parse_args()
    IVROOT = ROOT / "train_90M" / a.scorer / "intervention"
    GRID, OUT, RATER = ROOT / "train_90M" / a.grid, ROOT / a.out, a.rater
    OUT.mkdir(parents=True, exist_ok=True)
    flags = contraction_flags()
    base = load("baseline").rename(columns={"surp": "surp_base"})

    # 1. identity check vs the rater's baseline grid
    grid = GRID
    if grid.exists():
        g = pd.concat([pq.read_table(f).to_pandas().assign(genre=Path(f).stem)
                       for f in glob.glob(str(grid / "*.parquet"))])
        g["surp_grid"] = -g.logprob_sum
        j = base.merge(g[KEY + ["surp_grid"]], on=KEY)
        diff = (j.surp_base - j.surp_grid).abs()
        q = diff.quantile([.5, .99, .999]).round(4).tolist()
        print(f"[1] identity: n={len(j):,}  |Δ| p50/p99/p99.9 {q}  "
              f"max {diff.max():.3f}  >0.5 nats: {(diff > .5).sum()}  "
              f"rho {j.surp_base.corr(j.surp_grid, method='spearman'):.6f}")

    base["dec_base"] = pd.qcut(base.surp_base.rank(method="first"), 10,
                               labels=False)
    rows, rows_p, mig = [], [], []
    for iv in IVS:
        d = load(iv)
        j = base[KEY + ["surp_base", "dec_base", "person", "number"]].merge(
            d[KEY + ["surp", "head_edited", "form_edited", "form", "n_line_edits"]],
            on=KEY).merge(flags, on=KEY, how="left")
        rem = removed(iv)
        j["delta"] = j.surp - j.surp_base
        j["dec_iv"] = pd.qcut(j.surp.rank(method="first"), 10, labels=False)
        pron_edited = (j.form_edited != j.form).mean()
        rows.append({
            "intervention": iv, "n_paired": len(j),
            "n_line_removed": len(rem),
            "frac_line_removed": len(rem) / max(len(j) + len(rem), 1),
            "frac_any_edit_on_line": (j.n_line_edits > 0).mean(),
            "frac_head_edited": j.head_edited.mean(),
            "frac_pronoun_edited": pron_edited,
            "median_base": j.surp_base.median(),
            "median_iv": j.surp.median(),
            "median_delta": j.delta.median(),
            "mean_delta": j.delta.mean(),
            "rho_rank": j.surp_base.corr(j.surp, method="spearman"),
            "decile0_retained": ((j.dec_base == 0) & (j.dec_iv == 0)).sum()
                                / max((j.dec_base == 0).sum(), 1),
            "same_decile": (j.dec_base == j.dec_iv).mean(),
            "within_1_decile": ((j.dec_base - j.dec_iv).abs() <= 1).mean(),
            "frac_contracted": j.contracted.mean(),
            "median_delta_uncontracted": j[~j.contracted].delta.median(),
            "mean_delta_uncontracted": j[~j.contracted].delta.mean(),
            "rho_uncontracted": j[~j.contracted].surp_base.corr(
                j[~j.contracted].surp, method="spearman"),
            "median_delta_contracted": j[j.contracted].delta.median(),
        })
        for ct, sub in j.groupby("contracted"):
            rows_p.append({"intervention": iv, "slice": f"contracted={ct}",
                           "n": len(sub), "median_delta": sub.delta.median(),
                           "rho": sub.surp_base.corr(sub.surp, method="spearman")})
        for he, sub in j.groupby("head_edited"):
            rows_p.append({"intervention": iv, "slice": f"head_edited={he}",
                           "n": len(sub), "median_delta": sub.delta.median(),
                           "rho": sub.surp_base.corr(sub.surp, method="spearman")})
        for p, sub in j[j.person.isin([1, 2, 3])].groupby("person"):
            rows_p.append({"intervention": iv, "slice": f"person={int(p)}",
                           "n": len(sub), "median_delta": sub.delta.median(),
                           "rho": sub.surp_base.corr(sub.surp, method="spearman")})
        for g, sub in j.groupby("genre"):
            rows_p.append({"intervention": iv, "slice": f"genre={g}",
                           "n": len(sub), "median_delta": sub.delta.median(),
                           "rho": sub.surp_base.corr(sub.surp, method="spearman")})
        for (fm, he), sub in j[~j.contracted].groupby(["form", "head_edited"]):
            if len(sub) >= 300:
                rows_p.append({"intervention": iv,
                               "slice": f"uncontracted,form={fm},head_edited={he}",
                               "n": len(sub), "median_delta": sub.delta.median(),
                               "rho": sub.surp_base.corr(sub.surp, method="spearman")})
        for dcl, sub in j.groupby("dec_base"):
            rows_p.append({"intervention": iv, "slice": f"dec_base={int(dcl)}",
                           "n": len(sub), "median_delta": sub.delta.median(),
                           "rho": sub.surp_base.corr(sub.surp, method="spearman")})
        m = pd.crosstab(j.dec_base, j.dec_iv, normalize="index")
        m["intervention"] = iv
        mig.append(m.reset_index())
        j.to_parquet(OUT / f"paired_{iv}.parquet")

    summ = pd.DataFrame(rows)
    sl = pd.DataFrame(rows_p)
    summ.to_csv(OUT / "intervention_summary.csv", index=False)
    sl.to_csv(OUT / "intervention_slices.csv", index=False)
    pd.concat(mig).to_csv(OUT / "decile_migration.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n[2-3] per-intervention summary:")
    print(summ.round(3).to_string(index=False))
    print("\n[4] slices:")
    print(sl.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TEAL, SIENNA, GREY = "#2C6E63", "#B0562B", "#8A8878"
    ivs = [iv for iv in IVS if (OUT / f"paired_{iv}.parquet").exists()]
    migd = pd.read_csv(OUT / "decile_migration.csv")
    fig, axes = plt.subplots(3, len(ivs), figsize=(3.8 * len(ivs), 11))
    for c, iv in enumerate(ivs):
        j = pd.read_parquet(OUT / f"paired_{iv}.parquet")
        ax = axes[0][c]
        lo, hi = 0, np.quantile(j.surp_base, .99)
        from matplotlib.colors import LogNorm
        h = ax.hist2d(j.surp_base.clip(lo, hi), j.surp.clip(lo, hi), bins=70,
                      cmap="Greys", norm=LogNorm(), cmin=1)
        ax.plot([lo, hi], [lo, hi], color=SIENNA, lw=1)
        ax.set_title(iv.replace("_", " "), fontsize=10)
        ax.set_xlabel("baseline surprisal (nats)")
        if c == 0: ax.set_ylabel("surprisal under intervention")
        ax = axes[1][c]
        for ct, col, lab in ((False, GREY, "uncontracted"),
                             (True, TEAL, "contracted (they've, it's)")):
            s = j[j.contracted == ct].delta.clip(-8, 8)
            if len(s): ax.hist(s, bins=80, alpha=.6, color=col, label=f"{lab} (n={len(s):,})",
                               density=True)
        ax.axvline(0, color="#666", lw=.8)
        ax.set_xlabel("Δ surprisal (intervention − baseline)")
        if c == 0: ax.set_ylabel("density")
        ax.legend(fontsize=7)
        ax = axes[2][c]
        mm = (migd[migd.intervention == iv].set_index("dec_base")
              .drop(columns="intervention"))
        mm = mm[[str(k) for k in range(10)]].to_numpy()
        ax.imshow(mm, cmap="Greys", vmin=0, vmax=1, origin="lower")
        for a in range(10):
            ax.text(a, a, f"{mm[a, a]:.0%}", ha="center", va="center",
                    fontsize=6, color=SIENNA)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.tick_params(labelsize=7)
        ax.set_xlabel("decile under intervention (0 = most recoverable)")
        if c == 0: ax.set_ylabel("baseline decile")
    fig.suptitle(f"Same pronoun, different circumstances: {RATER} surprisal "
                 "in baseline vs intervention context", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "intervention_surprisal.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT}/intervention_surprisal.png + CSVs")


if __name__ == "__main__":
    main()
