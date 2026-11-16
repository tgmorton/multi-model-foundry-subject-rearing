#!/usr/bin/env python3
"""PDF: first BERT-v2 results on the matched set, next to gpt2_large on the same cells.

32 runs: 27 baseline cells plus all-removed in each training condition. BERT items are
pseudo-log-likelihood scored. Reads the per-checkpoint table bert_end.py caches in
report_cache/bert_matched_v2__agg2.parquet; gpt2_large comes from the report's matched table.

    python analysis/eval_v2/bert_v2_matched_plots.py --out bert_v2_matched.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nopron_priming_report as R  # noqa: E402
import priming_deep_report as D  # noqa: E402

ARM = re.compile(r"pdrop2_(?P<arm>[a-z]+?)(?P<k>\d+)_(?P<iv>[a-z]+)-")
PERS = ["subj_1sg", "subj_1pl", "subj_2sg", "subj_2pl", "subj_3sg", "subj_3pl"]
MODEL_C = {"BERT-v2": "#2a78d6", "gpt2_large": "#eb6834"}


def parse(d):
    m = d.run.str.extract(ARM)
    d = pd.concat([d.reset_index(drop=True), m.reset_index(drop=True)], axis=1).dropna(subset=["arm"])
    d["k"] = d.k.astype(int)
    return d


def endstate(d, last=3):
    a = d[d.checkpoint_step > 0]
    ls = a.groupby("run").checkpoint_step.apply(lambda s: sorted(s.unique())[-last:]).rename("ls")
    a = a.merge(ls, left_on="run", right_index=True)
    a = a[[s in l for s, l in zip(a.checkpoint_step, a.ls)]]
    return a.groupby(["run", "arm", "k", "iv", "category", "condition"]).prefers_overt_meanlp.mean().reset_index()


def per_run(e):
    return e[e.category.isin(R.SUBJ)].groupby(["run", "arm", "k", "iv"]).prefers_overt_meanlp.mean().reset_index()


def order(r):
    w = r[(r.iv == "base") & r.arm.isin(["robbi", "robbianti"])].pivot_table(index="k", columns="arm", values="prefers_overt_meanlp")
    return (w["robbi"] - w["robbianti"]).dropna()


def p_arms(doc, b, g):
    fig, box = doc.page("BERT-v2, matched set", "The arms in BERT and gpt2_large",
                        "Overt preference on subject-drop items at end of training, baseline training condition, fixed stimuli. "
                        "Each point is one run. BERT holds near ceiling far longer than gpt2_large.")
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.2}, wspace=0.1)
    axes = []
    for j, (name, v) in enumerate([("BERT-v2", b), ("gpt2_large", g)]):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        v = v[v.iv == "base"]
        for arm in ("rand", "robbi", "robbianti"):
            x = v[v.arm == arm].sort_values("k")
            ax.plot(x.k, x.prefers_overt_meanlp, color=D.ARM_C[arm], marker="o", ms=4.5, mec="white", mew=0.8, label=D.ARM_LABEL[arm])
        x = v[v.arm == "all"]
        ax.scatter(x.k, x.prefers_overt_meanlp, color=D.ARM_C["all"], marker="D", s=38, edgecolor="white", lw=0.8, zorder=3, label=D.ARM_LABEL["all"])
        ax.set_ylim(0, 1)
        ax.set_xticks([10, 30, 50, 70, 90, 100])
        ax.set_xlabel("subject pronouns removed (%)")
        ax.set_title(name)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("P(prefer overt), end of training")
            ax.legend(loc="lower left", fontsize=7.2)
    eb, eg = order(b), order(g)
    doc.bottom_notes(fig, [f"Order effect, mean over 9 levels: BERT {eb.mean():+.3f}, gpt2_large {eg.mean():+.3f}.",
                           f"At 90% removal: BERT {eb[90]:+.3f}, gpt2_large {eg[90]:+.3f}."], top=0.135, ncol=2, title="Read")
    doc.save(fig)


def p_order(doc, b, g):
    eb, eg = order(b), order(g)
    fig, box = doc.page("BERT-v2, matched set", "The order effect by removal level",
                        "Most-recoverable-first minus least-recoverable-first, baseline training condition.",
                        notes=[f"Mean: BERT {eb.mean():+.3f}, gpt2_large {eg.mean():+.3f}.",
                               "Differences of two single-seed runs, so each point is noisy.",
                               "BERT is pseudo-log-likelihood scored, so its log-probabilities are not on gpt2_large's scale. "
                               "Preference is comparable; the size of a gap may not be."])
    ax = fig.add_subplot(fig.add_gridspec(1, 1, **box)[0])
    for name, e in (("BERT-v2", eb), ("gpt2_large", eg)):
        ax.plot(e.index, e.values, color=MODEL_C[name], marker="o", ms=5, mec="white", mew=0.8, label=name)
    D.zero(ax)
    ax.set_xticks(range(10, 100, 10))
    ax.set_xlabel("subject pronouns removed (%)")
    ax.set_ylabel("Δ P(prefer overt): most minus least recoverable first")
    ax.legend(loc="upper left", fontsize=7.6)
    doc.save(fig)


def p_groups(doc, eb, eg):
    fig, box = doc.page("BERT-v2, matched set", "By person and by construction",
                        "End of training, baseline training condition, mean over the 27 arm and removal cells. One bar per model family.")
    gs = fig.add_gridspec(2, 1, **box, hspace=0.45)

    def bars(ax, tb, tg, labels):
        x = np.arange(len(labels))
        ax.bar(x - 0.19, tb, 0.36, color=MODEL_C["BERT-v2"], label="BERT-v2")
        ax.bar(x + 0.19, tg, 0.36, color=MODEL_C["gpt2_large"], label="gpt2_large")
        ax.set_xticks(x, labels, size=7.4)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(prefer overt)", size=7.8)

    b, g = eb[(eb.iv == "base") & (eb.arm != "all")], eg[(eg.iv == "base") & (eg.arm != "all")]
    sub = lambda e: e[e.category.isin(R.SUBJ)].groupby(["run", "condition"]).prefers_overt_meanlp.mean().groupby("condition").mean().reindex(PERS)  # noqa: E731
    ax = fig.add_subplot(gs[0])
    bars(ax, sub(b), sub(g), [c.replace("subj_", "") for c in PERS])
    ax.set_title("Subject drop, by target person")
    ax.legend(loc="upper center", ncol=2, fontsize=7.4, bbox_to_anchor=(0.5, 1.28))
    cats = [c for c in R.CATS if c in set(b.category)]
    cat = lambda e: e.groupby(["run", "category"]).prefers_overt_meanlp.mean().groupby("category").mean().reindex(cats)  # noqa: E731
    ax = fig.add_subplot(gs[1])
    bars(ax, cat(b), cat(g), [R.CAT_SHORT.get(c, c) for c in cats])
    ax.set_title("By construction")
    doc.save(fig)


def p_traj(doc, d, tok):
    x = d[(d.iv == "base") & d.category.isin(R.SUBJ) & d.arm.isin(["rand", "robbi", "robbianti"]) & d.k.isin([50, 90])]
    x = x.merge(tok, on=["run", "checkpoint_step"])
    x = x[x.tokens_seen > 0]
    fig, box = doc.page("BERT-v2, matched set", "Trajectories at 50% and 90% removal",
                        "One line per trained model, nothing averaged across models. Subject-drop targets, baseline training condition.")
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.14}, wspace=0.08)
    for j, k in enumerate((50, 90)):
        ax = fig.add_subplot(gs[j])
        for arm in ("rand", "robbi", "robbianti"):
            t = x[(x.k == k) & (x.arm == arm)].groupby("tokens_seen").prefers_overt_meanlp.mean().sort_index()
            ax.plot(t.index / 1e6, t.values, color=D.ARM_C[arm], lw=1.3, label=D.ARM_LABEL[arm])
        ax.axhline(0.5, color="#b8b8b8", lw=0.7, zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(f"{k}% removed")
        ax.set_xlabel("tokens seen (millions)")
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("P(prefer overt)")
            ax.legend(loc="lower right", fontsize=7.2)
    doc.save(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    bd = parse(pd.read_parquet(R.CACHE / "bert_matched_v2__agg2.parquet"))
    gd = parse(pd.read_parquet(R.CACHE / f"{R.ORIG}__agg.parquet"))
    gd = gd[gd.iv == "base"]
    eb, eg = endstate(bd), endstate(gd)
    b, g = per_run(eb), per_run(eg)
    s3 = R.s3c()
    tok = pd.concat([R.get_parquet(s3, R.ORIG_V2, "checkpoints", r, ["checkpoint_step", "tokens_seen"]).assign(run=r)
                     for r in sorted(bd[(bd.iv == "base") & bd.k.isin([50, 90])].run.unique())], ignore_index=True)
    plt = D.setup_mpl()
    D.FOOT = "BERT-v2 tranche 1 · matched set, fixed stimuli · gpt2_large for comparison"
    doc = D.Doc(a.out, plt)
    p_arms(doc, b, g)
    p_order(doc, b, g)
    p_groups(doc, eb, eg)
    p_traj(doc, bd, tok)
    doc.close()
    print("wrote", a.out, doc.n, "pages", b.run.nunique(), "BERT runs", g.run.nunique(), "gpt2 runs")


if __name__ == "__main__":
    main()
