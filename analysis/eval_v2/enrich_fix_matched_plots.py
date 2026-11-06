#!/usr/bin/env python3
"""PDF: the enrichment fix on the matched set (original contexts), wave2 and production.

Reads the table enrich_fix_compare.py writes. Runs may still be arriving, so each
page states its coverage, and baseline is restricted to the same (arm, removal)
cells as the scored enrichment runs, which keeps a partial comparison like for like.

    python analysis/eval_v2/enrich_fix_compare.py --out-csv matched.csv
    python analysis/eval_v2/enrich_fix_matched_plots.py --csv matched.csv --out enrich_fix_matched.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import enrich_fix_compare as C  # noqa: E402
import nopron_priming_report as R  # noqa: E402
import priming_deep_report as D  # noqa: E402

PERS = ["subj_1sg", "subj_1pl", "subj_2sg", "subj_2pl", "subj_3sg", "subj_3pl"]
CMP = {"baseline": "#008300", "enrich, old items": D.YELLOW, "enrich, fixed items": D.VIOLET}
WAVE2_RUNS, PROD_RUNS = 28, 28


def frames(d):
    """{series: run-level subject-drop frame} for one cohort, plus the run count."""
    s = d[d.category.isin(C.SUBJ)]
    fixed = s[(s.iv == "enrichvm") & (s.bench == C.V2)]
    done = fixed.run.unique()
    old = s[(s.iv == "enrichvm") & (s.bench == C.V1) & s.run.isin(done)]
    cells = fixed[["arm", "k"]].drop_duplicates()
    base = s[(s.iv == "base") & (s.bench == C.V1)].merge(cells, on=["arm", "k"])
    return {"baseline": base, "enrich, old items": old, "enrich, fixed items": fixed}, len(done)


def run_means(f):
    return f.groupby(["run", "arm", "k"])[["prefers_overt_meanlp", "log_prob_diff_overt_minus_null"]].mean().reset_index()


def person_table(fr, col):
    return pd.DataFrame({k: v.groupby(["run", "condition"])[col].mean().groupby("condition").mean()
                         for k, v in fr.items()}).reindex(PERS)


def p_arms(doc, fr, n):
    runs = {k: run_means(v) for k, v in fr.items()}
    fig, box = doc.page(
        "Matched set, wave2", "The arms on the original contexts",
        f"Overt preference on subject-drop items at end of training, per selection arm. Covers {n} of {WAVE2_RUNS} enrichment "
        "runs; baseline shows the same cells, and old and fixed are the same runs. Each point is one run.")
    gs = fig.add_gridspec(1, 3, **{**box, "bottom": 0.2}, wspace=0.12)
    axes = []
    for j, (name, v) in enumerate(runs.items()):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        for arm in ("rand", "robbi", "robbianti"):
            x = v[v.arm == arm].sort_values("k")
            if len(x):
                ax.plot(x.k, x.prefers_overt_meanlp, color=D.ARM_C[arm], marker="o", ms=4.5, mec="white", mew=0.8,
                        label=D.ARM_LABEL[arm])
        x = v[v.arm == "all"]
        if len(x):
            ax.scatter(x.k, x.prefers_overt_meanlp, color=D.ARM_C["all"], marker="D", s=38, edgecolor="white", lw=0.8,
                       zorder=3, label=D.ARM_LABEL["all"])
        ax.set_ylim(0, 1)
        ax.set_xticks([10, 30, 50, 70, 90, 100])
        ax.set_xlabel("subject pronouns removed (%)")
        ax.set_title(name)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("P(prefer overt), end of training")
            ax.legend(loc="lower left", fontsize=7.2)
    arm = {k: v.groupby("arm").prefers_overt_meanlp.mean() for k, v in runs.items()}
    notes = []
    for a in ("all", "robbianti", "rand", "robbi"):
        if a in arm["enrich, fixed items"].index:
            notes.append(f"{D.ARM_LABEL[a]}: old {arm['enrich, old items'][a]:.3f}, fixed {arm['enrich, fixed items'][a]:.3f}, "
                         f"baseline {arm['baseline'][a]:.3f}.")
    doc.bottom_notes(fig, notes, top=0.135, ncol=2, title="Read")
    doc.save(fig)


def p_order(doc, fr, n):
    runs = {k: run_means(v) for k, v in fr.items()}

    def effect(v):
        w = v[v.arm.isin(["robbi", "robbianti"])].pivot_table(index="k", columns="arm", values="prefers_overt_meanlp")
        return (w["robbi"] - w["robbianti"]).dropna() if {"robbi", "robbianti"} <= set(w.columns) else pd.Series(dtype=float)
    eff = {k: effect(v) for k, v in runs.items()}
    n_levels = len(eff["enrich, fixed items"])
    fig, box = doc.page(
        "Matched set, wave2", "The order effect on the original contexts",
        f"Most-recoverable-first minus least-recoverable-first at each removal level with both arms scored ({n_levels} of 9 "
        f"levels so far; {n} of {WAVE2_RUNS} runs).",
        notes=[f"Mean over those levels: baseline {eff['baseline'].mean():+.3f}, old items {eff['enrich, old items'].mean():+.3f}, "
               f"fixed items {eff['enrich, fixed items'].mean():+.3f}." if n_levels else "No removal level has both arms yet.",
               "Differences of two single-seed runs, so each point is noisy."])
    gs = fig.add_gridspec(1, 1, **box)
    ax = fig.add_subplot(gs[0])
    for name, e in eff.items():
        if len(e):
            ax.plot(e.index, e.values, color=CMP[name], marker="o", ms=5, mec="white", mew=0.8, label=name)
    D.zero(ax)
    ax.set_xticks(range(10, 100, 10))
    ax.set_xlabel("subject pronouns removed (%)")
    ax.set_ylabel("Δ P(prefer overt): most minus least recoverable first")
    ax.legend(loc="upper left", fontsize=7.6)
    doc.save(fig)


def p_person(doc, fr, n, cohort, total, title, lede_tail, notes):
    p = person_table(fr, "prefers_overt_meanlp")
    g = person_table(fr, "log_prob_diff_overt_minus_null")
    fig, box = doc.page(f"Matched set, {cohort}", title, f"Subject-drop items at end of training, by target person. Covers {n} of "
                        f"{total} enrichment runs. {lede_tail}", notes=notes(p, g))
    gs = fig.add_gridspec(2, 1, **box, hspace=0.42)
    lab = [c.replace("subj_", "") for c in PERS]
    for r, (t, ylab, ttl) in enumerate([(p, "P(prefer overt)", "Overt preference"),
                                        (g, "overt − null log-prob (nats/token)", "Log-prob gap")]):
        ax = fig.add_subplot(gs[r])
        x = np.arange(len(PERS))
        for k, name in enumerate(t.columns):
            ax.bar(x + (k - 1) * 0.27, t[name], 0.25, color=CMP[name], label=name)
        ax.set_xticks(x, lab)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(ylab, size=7.8)
        ax.set_title(ttl)
        if r == 0:
            ax.set_ylim(0, 1)
            ax.legend(loc="upper center", ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, 1.2))
        else:
            D.zero(ax)
    doc.save(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    plt = D.setup_mpl()
    D.FOOT = "gpt2_large · enrichment null-agreement fix · matched set (original contexts)"
    doc = D.Doc(a.out, plt)
    w, nw = frames(d[d.cohort == "wave2"])
    pr, np_ = frames(d[d.cohort == "production"])
    p_arms(doc, w, nw)
    p_order(doc, w, nw)
    p_person(doc, w, nw, "wave2", WAVE2_RUNS, "By person, wave2", "Baseline shows the same cells.",
             lambda p, g: [f"All persons: baseline {p['baseline'].mean():.3f}, old {p['enrich, old items'].mean():.3f}, "
                           f"fixed {p['enrich, fixed items'].mean():.3f}.",
                           f"Log-prob gap: baseline {g['baseline'].mean():+.3f}, old {g['enrich, old items'].mean():+.3f}, "
                           f"fixed {g['enrich, fixed items'].mean():+.3f}."])
    p_person(doc, pr, np_, "production", PROD_RUNS, "By person, production (no pronouns removed)",
             "These models saw almost no subjectless marked verbs in training.",
             lambda p, g: [f"All persons: baseline {p['baseline'].mean():.3f}, old {p['enrich, old items'].mean():.3f}, "
                           f"fixed {p['enrich, fixed items'].mean():.3f}.",
                           "A marked verb with no subject is rare in this corpus, so fixed null items are unlikely and overt "
                           "preference rises."])
    doc.close()
    print(f"wrote {a.out} ({doc.n} pages); wave2 {nw}/{WAVE2_RUNS}, production {np_}/{PROD_RUNS}")


if __name__ == "__main__":
    main()
