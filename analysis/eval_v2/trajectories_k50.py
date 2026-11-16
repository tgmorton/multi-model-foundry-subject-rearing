#!/usr/bin/env python3
"""PDF: single-model trajectories at 50% removal, random vs most vs least recoverable first.

One line per trained model, nothing pooled across models. Original contexts, with the
enrichment runs on the fixed v2 stimuli. Reads the tables nopron_priming_report.py caches.

    python analysis/eval_v2/trajectories_k50.py --out trajectories_k50.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nopron_priming_report as R  # noqa: E402
import priming_deep_report as D  # noqa: E402

ARMS = ["rand", "robbi", "robbianti"]
CATS = ["subject_drop", "subject_drop_no_agreement", "expletive", "conjunction",
        "embedded_drop", "control", "extraction", "object_drop"]
PERS = ["subj_1sg", "subj_1pl", "subj_2sg", "subj_2pl", "subj_3sg", "subj_3pl"]


def load():
    v1 = pd.read_parquet(R.CACHE / f"{R.ORIG}__agg.parquet")
    v2 = pd.read_parquet(R.CACHE / f"{R.ORIG_V2}__agg.parquet")
    d = pd.concat([v1[~v1.run.str.contains("_enrichvm-")], v2[v2.run.str.contains("_enrichvm-")]], ignore_index=True)
    d = d[d.run.str.contains(r"pdrop2_(?:rand|robbi|robbianti)50_")]
    d = R.parse_runs(d)
    tok = pd.read_parquet(R.CACHE / "tokens.parquet")
    d = d.merge(tok, on=["run", "checkpoint_step"], validate="m:1")
    d = d[d.tokens_seen > 0]
    d["mtok"] = d.tokens_seen / 1e6
    assert d.groupby("iv").run.nunique().eq(3).all() and d.iv.nunique() == 5
    return d


def panel(ax, g):
    for arm in ARMS:
        x = g[g.arm == arm].groupby("mtok").prefers_overt_meanlp.mean().sort_index()
        ax.plot(x.index, x.values, color=D.ARM_C[arm], lw=1.3, label=D.ARM_LABEL[arm])
    ax.axhline(0.5, color="#b8b8b8", lw=0.7, zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([1, 10, 100, 1000], ["1M", "10M", "100M", "1B"])
    ax.minorticks_off()
    ax.tick_params(labelsize=6.8)


def grid(pdf, plt, d, rows, title, lede, name):
    """rows: list of (label, frame). Columns are the five training conditions."""
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(11, 8.5), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.83, bottom=0.11 if n > 1 else 0.2, hspace=0.16, wspace=0.07)
    fig.text(0.045, 0.955, title, fontsize=14.5, weight="bold", va="top")
    fig.text(0.045, 0.915, lede, fontsize=8.8, color="#555", va="top")
    for r, (label, f) in enumerate(rows):
        for c, iv in enumerate(R.IVS):
            ax = axes[r][c]
            panel(ax, f[f.iv == iv])
            if r == 0:
                ax.set_title(R.IV_LABEL[iv], fontsize=9, weight="bold")
            if c == 0:
                ax.set_ylabel(label, fontsize=7.8)
    h, l = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.6, frameon=False, bbox_to_anchor=(0.54, 0.012 if n > 1 else 0.08))
    for line in leg.get_lines():
        line.set_linewidth(3)
    fig.text(0.54, 0.052 if n > 1 else 0.14, "tokens seen", fontsize=8.2, ha="center")
    fig.text(0.955, 0.012, name, fontsize=7, color="#888", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    plt = D.setup_mpl()
    from matplotlib.backends.backend_pdf import PdfPages
    d = load()
    note = ("One line per trained model (one seed each), 50% of subject pronouns removed. No averaging across models. "
            "y = P(prefer overt).")
    subj = d[d.category.isin(R.SUBJ)]
    sub = subj.groupby(["run", "arm", "iv", "mtok", "condition"]).prefers_overt_meanlp.mean().reset_index()
    per = lambda f: f.groupby(["run", "arm", "iv", "mtok"]).prefers_overt_meanlp.mean().reset_index()  # noqa: E731
    with PdfPages(a.out) as pdf:
        grid(pdf, plt, d, [("P(prefer overt)", per(sub))], "Subject drop at 50% removal, every training condition",
             note + " Subject-drop targets, all persons.", "1")
        grid(pdf, plt, d, [(R.CAT_SHORT.get(c, c), per(d[d.category == c])) for c in CATS[:4]],
             "By construction (1 of 2)", note, "2")
        grid(pdf, plt, d, [(R.CAT_SHORT.get(c, c), per(d[d.category == c])) for c in CATS[4:]],
             "By construction (2 of 2)", note, "3")
        grid(pdf, plt, d, [(c.replace("subj_", ""), per(sub[sub.condition == c])) for c in PERS],
             "Subject drop by target person", note, "4")
    print("wrote", a.out, d.run.nunique(), "models", d.groupby("run").checkpoint_step.nunique().describe()[["min", "max"]].to_dict())


if __name__ == "__main__":
    main()
