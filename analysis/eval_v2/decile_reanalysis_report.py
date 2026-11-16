#!/usr/bin/env python3
"""PDF: reanalysis of the decile result (unmodified reference, rater-free grading, trajectories).

Reads the deck export tables, so every number in the PDF is a number the deck gets:

  arg_decile_by_order.csv, arg_decile_order_minus_reference.csv      (deck_exports_argument.py --only arg)
  arg_rater_free_{by_decile,slopes,agreement,crosstab}.csv           (rater_free_regrade.py)
  arg_decile_trajectories{,_minus_random}.csv                        (deck_exports_decile_trajectories.py)

    python analysis/eval_v2/decile_reanalysis_report.py --exports <main>/analysis/eval_v2/figures/deck_exports \\
        --out <main>/analysis/eval_v2/figures/eval_recoverability/decile_reanalysis.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import priming_deep_report as D  # noqa: E402

SECTION = "Decile result · reanalysis"
MOST, LEAST, RAND, ALL = "most recoverable first", "least recoverable first", "random removal", "all removed"
NONE = "none (unmodified corpus, Study 1 runs)"
COL = {MOST: D.BLUE, LEAST: D.ORANGE, RAND: D.CTRL, NONE: D.AQUA, ALL: D.INK}
NAT, MAT, NOP = "natural position", "position matched across deciles", "pronoun-free contexts"
ARCHS = ["GPT-2 large", "LSTM"]
PREF = "share of pairs preferring the overt form"


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k].isin(v) if isinstance(v, (list, tuple, set)) else df[k] == v).to_numpy()
    return df[m]


def sg(x: float) -> str:
    return "0.00" if abs(x) < 0.005 else f"{x:+.2f}"


def settles(s: pd.Series, ok) -> float | None:
    """First index from which `ok` holds for the rest of the series."""
    m = np.asarray(ok, bool)
    i = len(m)
    while i > 0 and m[i - 1]:
        i -= 1
    return s.index[i] if i < len(m) else None


def decile_axis(ax, label=True):
    ax.set_xlim(-0.4, 9.4)
    ax.set_xticks(range(10))
    ax.grid(axis="x", visible=False)
    if label:
        ax.set_xlabel("decile of the test item  (0 = most recoverable)")


def line(ax, g, color, ls="-", lw=1.9, ms=3.4, label=None, band=True, mark_removed=False):
    g = g.sort_values("decile")
    if band:
        ax.fill_between(g.decile, g.lower, g.upper, color=color, alpha=0.13, lw=0)
    ax.plot(g.decile, g["mean"], color=color, ls=ls, lw=lw, label=label)
    if mark_removed and "decile_removed" in g:
        r = g.decile_removed.astype(bool).to_numpy()
        ax.plot(g.decile[r], g["mean"][r], ls="", marker="o", ms=ms + 1.2, color=color, mec="white", mew=0.8)
        ax.plot(g.decile[~r], g["mean"][~r], ls="", marker="o", ms=ms + 1.2, mfc="white", mec=color, mew=1.2)
    else:
        ax.plot(g.decile, g["mean"], ls="", marker="o", ms=ms, color=color)


# ---------------------------------------------------------------------------------------------
def p_summary(doc, T):
    by, sl, ag, gd = T["by"], T["sl"], T["ag"], T["gd"]
    un = sel(by, arm=NONE, test_set=[NAT, MAT])
    rho = sel(ag, test_set=NAT).set_index("architecture")

    def s(arch, arm, grading):
        r = sel(sl, architecture=arch, test_set=NAT, arm=arm, percent="mean of 10-90%", grading=grading).iloc[0]
        return r.slope_per_decile

    def g(arch, arm, grading, dec):
        return sel(gd, architecture=arch, test_set=NAT, arm=arm, grading=grading, decile=dec).ordered_minus_random.iloc[0]

    ratio = np.mean([abs(s(a, o, "learner's own grade alone") / s(a, o, "rater grade alone")) for a in ARCHS for o in (MOST, LEAST)])
    have_traj = T["tr"] is not None
    col1 = [
        ("The question", [
            "The decile test sets grade each pronoun with the selection rater, and the same rater decided which pronouns "
            "were removed from training. The decile result (models drop pronouns where training dropped them) is read off "
            "the rater's deciles. It could therefore owe something to the rater's own way of grading, and it had no "
            "reference model with nothing removed.",
        ]),
        ("What was added", [
            "1. Models trained on the unmodified corpus, scored on the same items: six seeds each for GPT-2 large and LSTM.",
            "2. A second grading of every test item that does not use the rater: the surprisal of the pronoun under those "
            "unmodified models themselves, cut into ten equal bins.",
            ("3. The natural-position set scored along training for nine models per architecture "
             "(unmodified, all removed, and each removal order at 50% and 80%).") if have_traj else
            "3. By-decile scoring along training is not in this version.",
        ]),
        ("Not covered", [
            "BERT: there are no unmodified BERT models of the new recipe to serve as a reference, and a masked model has "
            "no left-to-right probability for the pronoun to grade items with.",
        ]),
    ]
    col2 = [
        ("What came out", [
            f"With nothing removed there is no tilt. Unmodified models prefer the overt form on {un['mean'].min():.2f} to "
            f"{un['mean'].max():.2f} of pairs in every decile (lowest in decile 9). The tilt in the removal models is "
            "produced by the removal.",
            f"The learner's own grading agrees only moderately with the rater's (rank correlation "
            f"{rho.loc['GPT-2 large'].spearman_own_vs_rater_surprisal:.2f} for GPT-2 large, "
            f"{rho.loc['LSTM'].spearman_own_vs_rater_surprisal:.2f} for LSTM), so it is a real second opinion on the items.",
            f"The tilt is still there under the learner's own grading, in the right direction for both orders, in all three "
            f"test sets and both architectures. GPT-2 large, most recoverable first, against random removal: "
            f"{g('GPT-2 large', MOST, 'learner' + chr(39) + 's own decile', 0):+.2f} in the learner's decile 0 and "
            f"{g('GPT-2 large', MOST, 'learner' + chr(39) + 's own decile', 9):+.2f} in its decile 9 "
            f"(rater deciles: {g('GPT-2 large', MOST, 'rater decile', 0):+.2f} and {g('GPT-2 large', MOST, 'rater decile', 9):+.2f}).",
            f"It is smaller under the learner's grading: the slope per decile is about {ratio:.0%} of the slope under the "
            "rater's grading. With both gradings in one model each keeps a slope and the rater's is the larger, as it "
            "should be, since the rater's grading is what decided the removals.",
        ]),
    ] + ([("Along training", [
            "The preference for the overt form is learned first, in all deciles at once, by roughly 100M tokens. The "
            "tilt develops afterwards: it reaches half its final size between about 70M and 1B tokens, depending on the "
            "model. Least recoverable first separates earlier than most recoverable first in both architectures.",
        ])] if have_traj else []) + [
        ("Reading", [
            "The decile result does not depend on grading test items with the rater that chose the removals. It does not "
            "show that the learners share the rater's notion of recoverability; it shows that what they learned to drop "
            "is also ordered by their own sense of how predictable a pronoun is.",
        ]),
    ]
    doc.text_page(SECTION, "Does the decile result depend on the rater?",
                  "No. Models with nothing removed show no tilt across deciles, and the tilt in the removal models survives "
                  "when the test items are graded by the learner itself instead of by the rater.", [col1, col2])


def p_unmodified(doc, T):
    from matplotlib.lines import Line2D
    by = sel(T["by"], test_set=NAT)
    un = sel(by, arm=NONE)
    notes = [
        "Green: models trained on the unmodified corpus (mean of six seeds). They sit at "
        f"{un['mean'].min():.2f} to {un['mean'].max():.2f} in every decile; only decile 9 dips.",
        "Grey: random removal lowers the preference in every decile, a little more toward decile 9.",
        "Blue and orange: the two orders pull the curve down at opposite ends. Filled points are deciles whose pronouns "
        "that model's training corpus lost.",
        "Bands are 95% intervals from resampling items. One run per removal cell.",
    ]
    fig, box = doc.page(SECTION, "With nothing removed there is no tilt",
                        "Preference for the overt form by decile of the test item, natural-position set, end state. The "
                        "unmodified models give the reference the decile result did not have.", notes)
    gs = fig.add_gridspec(2, 2, **{**box, "bottom": 0.185}, wspace=0.10, hspace=0.36)
    for i, arch in enumerate(ARCHS):
        for j, k in enumerate((50, 80)):
            ax = fig.add_subplot(gs[i, j])
            line(ax, sel(un, architecture=arch), COL[NONE], lw=1.7)
            for arm in (RAND, MOST, LEAST):
                line(ax, sel(by, architecture=arch, arm=arm, percent=k), COL[arm], mark_removed=arm != RAND)
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color=D.MUTED, lw=0.8, zorder=1)
            decile_axis(ax, label=i == 1)
            ax.set_title(f"{arch} · {k}% removed")
            if j == 0:
                ax.set_ylabel("share preferring overt")
            else:
                ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=COL[NONE], lw=1.8, marker="o", ms=4, label="unmodified corpus"),
                        Line2D([], [], color=COL[RAND], lw=1.8, marker="o", ms=4, label="random removal"),
                        Line2D([], [], color=COL[MOST], lw=1.8, marker="o", ms=4, label=MOST),
                        Line2D([], [], color=COL[LEAST], lw=1.8, marker="o", ms=4, label=LEAST)],
               loc="lower left", bbox_to_anchor=(0.07, 0.065), ncol=2, fontsize=7.8, handlelength=2.0, columnspacing=1.4)
    doc.save(fig)


def p_reference(doc, T):
    from matplotlib.lines import Line2D
    ref = sel(T["ref"], test_set=NAT)
    fig, box = doc.page(SECTION, "The tilt against either reference",
                        "Ordered removal minus a reference model, by decile, paired by item. Solid: minus random removal at "
                        "the same percent. Dashed: minus the unmodified model. Natural-position set, end state.")
    gs = fig.add_gridspec(2, 3, **{**box, "bottom": 0.335}, wspace=0.08, hspace=0.36)
    lo, hi = ref["mean"].min() - 0.04, ref["mean"].max() + 0.04
    for i, arch in enumerate(ARCHS):
        for j, k in enumerate((30, 50, 80)):
            ax = fig.add_subplot(gs[i, j])
            for arm in (MOST, LEAST):
                line(ax, sel(ref, architecture=arch, percent=k, arm=f"{arm} minus the unmodified model"), COL[arm],
                     ls=(0, (4, 2)), lw=1.5, band=False, ms=2.6)
                line(ax, sel(ref, architecture=arch, percent=k, arm=f"{arm} minus random removal"), COL[arm],
                     mark_removed=True)
            D.zero(ax)
            ax.set_ylim(lo, hi)
            decile_axis(ax, label=i == 1)
            ax.set_title(f"{arch} · {k}% removed")
            if j == 0:
                ax.set_ylabel("Δ share preferring overt")
            else:
                ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=COL[MOST], lw=1.9, label=f"{MOST}, minus random"),
                        Line2D([], [], color=COL[MOST], lw=1.5, ls=(0, (4, 2)), label="minus unmodified"),
                        Line2D([], [], color=COL[LEAST], lw=1.9, label=f"{LEAST}, minus random"),
                        Line2D([], [], color=COL[LEAST], lw=1.5, ls=(0, (4, 2)), label="minus unmodified"),
                        Line2D([], [], color=D.INK2, ls="", marker="o", ms=5, label="decile removed in training"),
                        Line2D([], [], color=D.INK2, ls="", marker="o", ms=5, mfc="white", label="decile kept")],
               loc="upper left", bbox_to_anchor=(0.07, 0.285), ncol=6, fontsize=7.4, handlelength=2.0, columnspacing=1.1)

    def v(arch, arm, refname, k, dec):
        return sel(ref, architecture=arch, percent=k, arm=f"{arm} minus {refname}", decile=dec)["mean"].iloc[0]

    def rl(arch, k, dec):
        b = sel(T["by"], test_set=NAT, architecture=arch, decile=dec)
        return sel(b, arm=RAND, percent=k)["mean"].iloc[0] - sel(b, arm=NONE)["mean"].iloc[0]
    doc.bottom_notes(fig, [
        "Against the unmodified model every curve lies at or below zero: removal lowers the preference everywhere, and "
        "lowers it most where that order removed pronouns.",
        f"GPT-2 large at 50%, minus unmodified: most-first {sg(v('GPT-2 large', MOST, 'the unmodified model', 50, 0))} "
        f"in decile 0 and {sg(v('GPT-2 large', MOST, 'the unmodified model', 50, 8))} in decile 8; least-first "
        f"{sg(v('GPT-2 large', LEAST, 'the unmodified model', 50, 0))} in decile 0 and "
        f"{sg(v('GPT-2 large', LEAST, 'the unmodified model', 50, 9))} in decile 9.",
        "Against random removal the kept deciles sit at or above zero: an ordered model keeps more of its preference there "
        "than a random model that lost the same number of pronouns.",
        f"The gap between solid and dashed lines is the random model's own loss. It grows toward decile 9 (GPT-2 large "
        f"at 80%: {sg(rl('GPT-2 large', 80, 0))} in decile 0, {sg(rl('GPT-2 large', 80, 9))} in decile 9), so against the "
        "unmodified model the least-first tilt is steeper and the most-first tilt shallower. The direction is the same "
        "against both references.",
    ], 0.235)
    doc.save(fig)


def p_agreement(doc, T):
    from matplotlib.colors import LinearSegmentedColormap
    ct, ag = sel(T["ct"], test_set=NAT), T["ag"]
    a = ag.set_index(["architecture", "test_set"])
    notes = [
        "Each row is one rater decile; the cells show where its items fall among the learner's own deciles. Agreement "
        "would put the mass on the diagonal.",
        "The diagonal is visible but wide. Rank correlation between the two surprisals: " + "; ".join(
            f"{arch} {a.loc[(arch, NAT)].spearman_own_vs_rater_surprisal:.2f}" for arch in ARCHS) + ".",
        "Share of items within one decile of each other: " + "; ".join(
            f"{arch} {a.loc[(arch, NAT)].share_within_one_decile:.0%}" for arch in ARCHS) +
        ". Chance for ten equal bins is 28%.",
        "The other two sets give the same picture (correlations " +
        f"{ag.spearman_own_vs_rater_surprisal.min():.2f} to {ag.spearman_own_vs_rater_surprisal.max():.2f}).",
        "The learner reads left to right and grades from the preceding text only. The rater also sees what follows the "
        "pronoun. Some of the disagreement comes from that.",
    ]
    fig, box = doc.page(SECTION, "Two gradings of the same items",
                        "The learner's own grade: surprisal of the pronoun at its position under models trained on the "
                        "unmodified corpus (mean of six seeds), cut into ten equal bins. Natural-position set.", notes)
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.30}, wspace=0.22)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#a9c4e8", "#2a5ba3", "#14366d"])
    for j, arch in enumerate(ARCHS):
        ax = fig.add_subplot(gs[0, j])
        M = sel(ct, architecture=arch).pivot(index="rater_decile", columns="own_decile", values="share_of_rater_decile")
        M = M.reindex(index=range(10), columns=range(10)).to_numpy()
        ax.imshow(M, cmap=cmap, vmin=0, vmax=0.36, aspect="equal")
        for r in range(10):
            for c in range(10):
                ax.text(c, r, f"{M[r, c] * 100:.0f}", ha="center", va="center", size=6.3,
                        color="white" if M[r, c] > 0.2 else D.INK2)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticks(np.arange(-0.5, 10), minor=True)
        ax.set_yticks(np.arange(-0.5, 10), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(arch)
        ax.set_xlabel("learner's own decile")
        if j == 0:
            ax.set_ylabel("rater decile")
    fig.text(box["left"], 0.235, "Cells: percent of the rater decile's items. Decile 0 = most recoverable (rater) or most "
             "predictable to the learner (own).", family=D.BODY, size=8.6, color=D.MUTED, va="top")
    doc.save(fig)


def p_owngrade(doc, T):
    from matplotlib.lines import Line2D
    gd = sel(T["gd"], test_set=NAT).rename(columns={"ordered_minus_random": "mean"})
    own, rat = "learner's own decile", "rater decile"

    def v(arch, arm, grading, dec):
        return sel(gd, architecture=arch, arm=arm, grading=grading, decile=dec)["mean"].iloc[0]
    notes = [
        "Solid lines sort the items by the rater's grade, dashed lines by the learner's own grade. Both are the same models "
        "and the same items; only the grouping of items into deciles changes.",
        f"Most recoverable first (blue) drops more than random where pronouns are most predictable under either grading: "
        f"GPT-2 large {v('GPT-2 large', MOST, own, 0):+.2f} in the learner's decile 0 against "
        f"{v('GPT-2 large', MOST, own, 9):+.2f} in its decile 9.",
        f"Least recoverable first (orange) is the mirror image: {sg(v('GPT-2 large', LEAST, own, 0))} in the learner's "
        f"decile 0 against {sg(v('GPT-2 large', LEAST, own, 9))} in decile 9.",
        "The dashed curves are flatter than the solid ones. The rater's grading sorts the items more sharply by what was "
        "removed, as expected.",
        "Mean over the nine removal levels (10-90%). Bands: 95% intervals from resampling items.",
    ]
    fig, box = doc.page(SECTION, "The tilt under the learner's own grading",
                        "Ordered removal minus random removal at the same percent, by decile of the test item, with the "
                        "deciles defined by the rater or by the learner. Natural-position set, end state.", notes)
    gs = fig.add_gridspec(2, 1, **{**box, "bottom": 0.195}, hspace=0.36)
    lo, hi = min(gd.lower.min(), -0.05) - 0.02, max(gd.upper.max(), 0.05) + 0.02
    for i, arch in enumerate(ARCHS):
        ax = fig.add_subplot(gs[i, 0])
        for arm in (MOST, LEAST):
            line(ax, sel(gd, architecture=arch, arm=arm, grading=rat), COL[arm])
            g = sel(gd, architecture=arch, arm=arm, grading=own).sort_values("decile")
            ax.fill_between(g.decile, g.lower, g.upper, color=COL[arm], alpha=0.10, lw=0)
            ax.plot(g.decile, g["mean"], color=COL[arm], ls=(0, (4, 2)), lw=1.7, marker="o", ms=4, mfc="white", mew=1.1)
        D.zero(ax)
        ax.set_ylim(lo, hi)
        decile_axis(ax, label=i == 1)
        if i == 1:
            ax.set_xlabel("decile of the test item under that grading  (0 = most recoverable / most predictable)")
        ax.set_title(arch)
        ax.set_ylabel("ordered minus random\n(Δ share preferring overt)")
    fig.legend(handles=[Line2D([], [], color=COL[MOST], lw=1.9, marker="o", ms=4, label=f"{MOST} · rater deciles"),
                        Line2D([], [], color=COL[MOST], lw=1.7, ls=(0, (4, 2)), marker="o", ms=4, mfc="white",
                               label="learner's own deciles"),
                        Line2D([], [], color=COL[LEAST], lw=1.9, marker="o", ms=4, label=f"{LEAST} · rater deciles"),
                        Line2D([], [], color=COL[LEAST], lw=1.7, ls=(0, (4, 2)), marker="o", ms=4, mfc="white",
                               label="learner's own deciles")],
               loc="lower left", bbox_to_anchor=(0.07, 0.06), ncol=2, fontsize=7.6, handlelength=2.6, columnspacing=1.6)
    doc.save(fig)


def p_slopes(doc, T):
    from matplotlib.lines import Line2D
    sl = sel(T["sl"], percent="mean of 10-90%")
    G = ["rater grade alone", "learner's own grade alone", "rater grade, own grade held fixed",
         "learner's own grade, rater grade held fixed"]
    GL = ["sorted by\nrater", "sorted by\nlearner", "rater, same\nlearner grade", "learner, same\nrater grade"]
    fig, box = doc.page(SECTION, "The same items sorted two ways",
                        "Each item has one number: how much more the ordered-removal model drops its pronoun than the "
                        "random-removal model. The slope is how that number changes per decile when items are sorted by "
                        "the rater's grade, or by the learner's own grade. Positive: more dropping toward decile 0. Mean "
                        "over 10-90% removed; bars are 95% intervals from resampling items.")
    gs = fig.add_gridspec(2, 3, **{**box, "bottom": 0.36}, wspace=0.08, hspace=0.30)
    lim = max(abs(sl.lower.min()), abs(sl.upper.max())) + 0.006
    for i, arch in enumerate(ARCHS):
        for j, ts in enumerate((NAT, MAT, NOP)):
            ax = fig.add_subplot(gs[i, j])
            for m, arm in enumerate((MOST, LEAST)):
                for x0, gname in enumerate(G):
                    r = sel(sl, architecture=arch, test_set=ts, arm=arm, grading=gname).iloc[0]
                    x = x0 + (m - 0.5) * 0.26
                    ax.plot([x, x], [r.lower, r.upper], color=COL[arm], lw=2.2, solid_capstyle="round")
                    ax.plot([x], [r.slope_per_decile], marker="o", ms=6.5, color=COL[arm], mec="white", mew=1.1)
                    ax.annotate(f"{r.slope_per_decile:+.3f}".replace("0.", "."),
                                (x, r.upper if r.slope_per_decile > 0 else r.lower),
                                xytext=(0, 4 if r.slope_per_decile > 0 else -4), textcoords="offset points", ha="center",
                                va="bottom" if r.slope_per_decile > 0 else "top", size=6.3, color=D.INK2)
            D.zero(ax)
            ax.axvline(1.5, color=D.RULE, lw=0.8)
            ax.set_ylim(-lim, lim)
            ax.set_xlim(-0.55, 3.55)
            ax.set_xticks(range(4), GL if i == 1 else [""] * 4, fontsize=7.2)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{arch} · {ts}", fontsize=9)
            if j == 0:
                ax.set_ylabel("slope per decile")
            else:
                ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=COL[a_], lw=2.2, marker="o", ms=6, mec="white", label=a_) for a_ in (MOST, LEAST)],
               loc="upper left", bbox_to_anchor=(0.07, 0.285), ncol=2, fontsize=8)
    al = sel(sl, grading=G[0]).slope_per_decile.abs()
    ol = sel(sl, grading=G[1]).slope_per_decile.abs()
    pf = sel(sl, grading=G[3])
    ok = int(((pf.lower > 0) | (pf.upper < 0)).sum())
    doc.bottom_notes(fig, [
        f"Left pair in each panel: the items sorted one way at a time. \"Sorted by rater\" is the original decile result. "
        f"\"Sorted by learner\" is the check: the rater has no say in the sorting, and the slope has the same sign in all "
        f"{len(ol)} comparisons ({ol.min():.3f} to {ol.max():.3f} in size, against {al.min():.3f} to {al.max():.3f}).",
        f"The learner's grade matters among items the rater grades alike in {ok} of {len(pf)} comparisons (interval "
        "excludes zero). So it is not picking up the rater's grade second-hand.",
        "Right pair: the two grades are correlated, so the left pair could be one effect seen twice. Here each grade is "
        "tested among items that have the same value of the other. Both keep a slope; the rater's is the larger everywhere "
        "except the pronoun-free set for LSTM, where they are about equal.",
        "Position-matched and pronoun-free sets agree with the natural-position set, so neither line position nor a "
        "repeated pronoun in the context carries this.",
    ], 0.235)
    doc.save(fig)


def p_bypercent(doc, T):
    from matplotlib.lines import Line2D
    sl = sel(T["sl"], test_set=NAT)
    sl = sl[sl.percent != "mean of 10-90%"].copy()
    sl["k"] = sl.percent.str.rstrip("%").astype(int)
    notes = [
        "The slope grows with the percent removed under both gradings and for both orders. Only most recoverable first "
        "under the rater's grading levels off or eases back at 90%.",
        "The learner's own grading (dashed) follows the rater's (solid) at every percent, at a smaller size.",
        "At 10% the slopes are near zero for both orders: removing one tenth leaves little trace on this measure at the "
        "end of training.",
        "Each point is one model against the random-removal model of the same percent. Bands: 95% intervals from "
        "resampling items.",
    ]
    fig, box = doc.page(SECTION, "The same check at each percent removed",
                        "Slope of (ordered minus random) per decile step, at each removal level, under each grading alone. "
                        "Natural-position set, end state.", notes)
    gs = fig.add_gridspec(2, 1, **{**box, "bottom": 0.195}, hspace=0.36)
    lim = max(abs(sl.lower.min()), abs(sl.upper.max())) + 0.004
    for i, arch in enumerate(ARCHS):
        ax = fig.add_subplot(gs[i, 0])
        for arm in (MOST, LEAST):
            for gname, ls, mfc in (("rater grade alone", "-", None), ("learner's own grade alone", (0, (4, 2)), "white")):
                g = sel(sl, architecture=arch, arm=arm, grading=gname).sort_values("k")
                ax.fill_between(g.k, g.lower, g.upper, color=COL[arm], alpha=0.11, lw=0)
                ax.plot(g.k, g.slope_per_decile, color=COL[arm], ls=ls, lw=1.8, marker="o", ms=4,
                        mfc=mfc or COL[arm], mew=1.1)
        D.zero(ax)
        ax.set_ylim(-lim, lim)
        ax.set_xticks(range(10, 100, 10))
        ax.grid(axis="x", visible=False)
        ax.set_title(arch)
        ax.set_ylabel("slope per decile")
        if i == 1:
            ax.set_xlabel("percent of subject pronouns removed from training")
    fig.legend(handles=[Line2D([], [], color=COL[MOST], lw=1.9, marker="o", ms=4, label=f"{MOST} · rater grading"),
                        Line2D([], [], color=COL[MOST], lw=1.7, ls=(0, (4, 2)), marker="o", ms=4, mfc="white",
                               label="learner's own grading"),
                        Line2D([], [], color=COL[LEAST], lw=1.9, marker="o", ms=4, label=f"{LEAST} · rater grading"),
                        Line2D([], [], color=COL[LEAST], lw=1.7, ls=(0, (4, 2)), marker="o", ms=4, mfc="white",
                               label="learner's own grading")],
               loc="lower left", bbox_to_anchor=(0.07, 0.06), ncol=2, fontsize=7.6, handlelength=2.6, columnspacing=1.6)
    doc.save(fig)


# ---------------------------------------------------------------------------------------------
def seq10():
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("seq", ["#b9cfee", "#6a99d6", "#2a5ba3", "#102a55"])
    return [cm(x) for x in np.linspace(0, 1, 10)]


def p_traj(doc, T, arch):
    from matplotlib.lines import Line2D
    tr = sel(T["tr"], architecture=arch)
    tr = tr[tr.tokens_seen > 0]
    panels = [(NONE, 0, "unmodified corpus"), (RAND, 50, "random · 50%"), (MOST, 50, "most recoverable first · 50%"),
              (LEAST, 50, "least recoverable first · 50%"), (ALL, 100, "all removed"), (RAND, 80, "random · 80%"),
              (MOST, 80, "most recoverable first · 80%"), (LEAST, 80, "least recoverable first · 80%")]
    fig, box = doc.page(SECTION, f"Along training, by decile: {arch}",
                        "Preference for the overt form on the natural-position set at each scored checkpoint, one line per "
                        "decile of the test item (rater grading). Light = decile 0, most recoverable; dark = decile 9.")
    gs = fig.add_gridspec(2, 4, **{**box, "bottom": 0.375}, wspace=0.08, hspace=0.40)
    cols = seq10()
    for n, (arm, k, title) in enumerate(panels):
        ax = fig.add_subplot(gs[n // 4, n % 4])
        g = sel(tr, arm=arm, percent=k)
        for dc in range(10):
            d = sel(g, decile=dc).sort_values("tokens_seen")
            ax.plot(d.tokens_seen / 1e6, d["mean"], color=cols[dc], lw=1.5)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=D.MUTED, lw=0.8, zorder=1)
        D.tokens_axis(ax, label=n // 4 == 1)
        ax.set_title(title, fontsize=8.6)
        if n % 4 == 0:
            ax.set_ylabel("share preferring overt")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=cols[0], lw=2, label="decile 0 (most recoverable)"),
                        Line2D([], [], color=cols[4], lw=2, label="decile 4"),
                        Line2D([], [], color=cols[9], lw=2, label="decile 9 (least recoverable)")],
               loc="upper left", bbox_to_anchor=(0.07, 0.325), ncol=3, fontsize=8)
    return fig


def spread(tr, arch, arm, k, tok_lo=None):
    """Decile 0 minus decile 9 preference along training for one model."""
    g = sel(tr, architecture=arch, arm=arm, percent=k)
    p = g.pivot_table(index="tokens_seen", columns="decile", values="mean")
    return p[0] - p[9]


def traj_notes(T, arch):
    tr = T["tr"]
    tr = tr[tr.tokens_seen > 0]
    out = []
    un = sel(tr, architecture=arch, arm=NONE).pivot_table(index="tokens_seen", columns="decile", values="mean")
    first = settles(un, un.min(axis=1) > 0.9)
    out.append(f"Unmodified: all ten deciles rise together and every one stays above 0.90 from "
               f"{D.fmt_tok(first / 1e6) if first else 'the end'} tokens on. Decile 9 is the last to get there.")
    for arm, word in ((MOST, "the deciles rise together and the light lines fall away afterwards"),
                      (LEAST, "the dark lines stop lower during the rise and then decline")):
        s = spread(tr, arch, arm, 80)
        e = s.iloc[-3:].mean()
        half = settles(s, s * np.sign(e) >= abs(e) / 2)
        out.append(f"{arm.capitalize()}, 80%: {word}. Decile 0 minus decile 9 ends at {e:+.2f} and stays beyond half of "
                   f"that from about {D.fmt_tok(half / 1e6)} tokens.")
    r = spread(tr, arch, RAND, 80).iloc[-3:].mean()
    a = sel(tr, architecture=arch, arm=ALL).pivot_table(index="tokens_seen", columns="decile", values="mean")
    out.append(f"Random removal lowers every decile, decile 9 most (decile 0 minus decile 9 ends at {r:+.2f} at 80%). All "
               f"removed ends between {a.iloc[-3:].mean().min():.2f} and {a.iloc[-3:].mean().max():.2f} in every decile.")
    out.append("The dip before about 30M tokens is a word-frequency effect: the models do not yet use the pronoun to predict "
               "the verb, and the measure only asks whether the pronoun is a commoner token than its neighbours.")
    out.append("One run per removal cell; the unmodified panel is the mean of two seeds. Lines are means over about 288 "
               "items per decile; 26 checkpoints per run.")
    return out


def p_tilt_time(doc, T):
    from matplotlib.lines import Line2D
    trd = T["trd"]
    trd = trd[trd.tokens_seen > 0]
    gaps = {}
    for arch in ARCHS:
        for k in (50, 80):
            for arm in (MOST, LEAST):
                g = sel(trd, architecture=arch, percent=k, arm=f"{arm} minus random removal")
                m = g.groupby([g.decile_removed.astype(bool), "tokens_seen"])["mean"].mean()
                gap = m[False] - m[True]
                e = gap.iloc[-3:].mean()
                gaps[(arch, k, arm)] = (e, settles(gap, gap >= e / 2))
    T["gaps"] = gaps

    def when(arch):
        return ", ".join(f"{D.fmt_tok(gaps[(arch, k, arm)][1] / 1e6)} ({'most' if arm == MOST else 'least'}-first {k}%)"
                         for k in (50, 80) for arm in (MOST, LEAST))
    notes = [
        "Solid: deciles that model's training corpus lost. Dashed: deciles it kept. Both are differences from the "
        "random-removal model of the same percent at the same point in training.",
        "The tilt comes late. Solid and dashed run together while the preference for the overt form is being learned "
        "(to roughly 100M tokens) and part afterwards, earlier for least recoverable first.",
        f"Kept minus removed stays beyond half its final size from: GPT-2 large {when('GPT-2 large')}; LSTM {when('LSTM')}.",
        "Final size of kept minus removed: " + "; ".join(
            f"{arch} " + ", ".join(f"{gaps[(arch, k, arm)][0]:+.2f}" for k in (50, 80) for arm in (MOST, LEAST))
            for arch in ARCHS) + " (same order as above).",
        "Before about 30M tokens the models know only word frequencies, and the measure there reflects how common each "
        "pronoun is, not subject drop. Runs are aligned by position in the checkpoint schedule. One run per cell, no "
        "intervals drawn.",
    ]
    fig, box = doc.page(SECTION, "When the tilt appears",
                        "Ordered removal minus random removal along training, averaged over the deciles the order removed "
                        "and over the deciles it kept. Natural-position set.", notes)
    gs = fig.add_gridspec(2, 2, **{**box, "bottom": 0.195}, wspace=0.10, hspace=0.38)
    lo, hi = -0.45, 0.25
    for i, arch in enumerate(ARCHS):
        for j, k in enumerate((50, 80)):
            ax = fig.add_subplot(gs[i, j])
            for arm in (MOST, LEAST):
                g = sel(trd, architecture=arch, percent=k, arm=f"{arm} minus random removal")
                for rem, ls in ((True, "-"), (False, (0, (4, 2)))):
                    m = g[g.decile_removed.astype(bool) == rem].groupby("tokens_seen")["mean"].mean()
                    ax.plot(m.index / 1e6, m.values, color=COL[arm], ls=ls, lw=1.8)
                    lo, hi = min(lo, m.min() - 0.03), max(hi, m.max() + 0.03)
            D.zero(ax)
            D.tokens_axis(ax, label=i == 1)
            ax.set_title(f"{arch} · {k}% removed")
            if j == 0:
                ax.set_ylabel("ordered minus random\n(Δ share preferring overt)")
            else:
                ax.set_yticklabels([])
    for ax in fig.axes:
        ax.set_ylim(lo, hi)
    fig.legend(handles=[Line2D([], [], color=COL[MOST], lw=1.9, label=f"{MOST}: removed deciles"),
                        Line2D([], [], color=COL[MOST], lw=1.9, ls=(0, (4, 2)), label="kept deciles"),
                        Line2D([], [], color=COL[LEAST], lw=1.9, label=f"{LEAST}: removed deciles"),
                        Line2D([], [], color=COL[LEAST], lw=1.9, ls=(0, (4, 2)), label="kept deciles")],
               loc="lower left", bbox_to_anchor=(0.07, 0.06), ncol=2, fontsize=7.6, handlelength=2.6, columnspacing=1.6)
    doc.save(fig)


def p_methods(doc, T):
    ag = T["ag"]
    n = sel(ag, architecture="GPT-2 large").set_index("test_set").n_items
    traj = T["tr"]
    col1 = [
        ("Models", [
            "Removal models: baseline grammar condition, one run per cell (random, most recoverable first, least "
            "recoverable first at 10-90%, plus all removed), GPT-2 large and LSTM. End state is the final checkpoint.",
            "Unmodified models: the Study 1 baseline runs at the winning hyperparameter setting, six full-length seeds per "
            "architecture, final checkpoint. Their per-item scores are averaged over seeds.",
        ]),
        ("Test items", [
            f"The three decile-graded sets drawn from the held-out 10M-word test corpus: natural position ({n[NAT]:,} "
            f"pairs), position matched across deciles ({n[MAT]:,}), pronoun-free contexts ({n[NOP]:,}). Each pair is the "
            "same passage with and without one subject pronoun.",
            "Rater decile: the selection rater's surprisal for the pronoun, cut at the thresholds frozen on the training "
            "corpus. Decile 0 is most recoverable.",
        ]),
        ("The learner's own grade", [
            "Log-probability of the pronoun at its own position, given the text before it, under the unmodified models of "
            "the same architecture. Averaged over seeds, negated, ranked within each test set and cut into ten "
            "equal-count bins.",
        ]),
    ]
    col2 = [
        ("Statistics", [
            "Preference: share of pairs where the version with the pronoun has the higher mean log-probability per token.",
            "Ordered minus random: difference in that 0/1 preference between an ordered-removal model and the "
            "random-removal model at the same percent, item by item.",
            "Slope: least-squares slope of that difference on the decile (0-9), over items; with both gradings as "
            "predictors for the held-fixed version. Intervals: 95%, from resampling items 1,000 times (300 for the "
            "per-percent rows).",
        ]),
        ("Limits", [
            "One run per removal cell, so intervals describe items, not seeds.",
            "The learner's grade uses left context only; the rater uses both sides.",
            "The unmodified models are near ceiling on the preference itself, so they cannot show a tilt on that measure "
            "even if they had one; their graded surprisal is what the second grading uses.",
            "No BERT: no unmodified reference under the new recipe and no left-to-right pronoun probability.",
        ] + (["Trajectories: nine models per architecture, 26 checkpoints each, natural-position set only."]
             if traj is not None else [])),
        ("Files", [
            "Tables behind every figure: analysis/eval_v2/figures/deck_exports/arg_decile_*.csv and arg_rater_free_*.csv. "
            "Scripts (uncommitted, analysis/eval_v2/): rater_free_regrade.py, deck_exports_argument.py, "
            "deck_exports_decile_trajectories.py, decile_reanalysis_report.py.",
        ]),
    ]
    doc.text_page(SECTION, "How this was done", "Models, items, the second grading, and what the intervals mean.", [col1, col2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rd = lambda n: pd.read_csv(a.exports / n) if (a.exports / n).exists() else None  # noqa: E731
    T = dict(by=rd("arg_decile_by_order.csv"), ref=rd("arg_decile_order_minus_reference.csv"),
             gd=rd("arg_rater_free_by_decile.csv"), sl=rd("arg_rater_free_slopes.csv"), ag=rd("arg_rater_free_agreement.csv"),
             ct=rd("arg_rater_free_crosstab.csv"), tr=rd("arg_decile_trajectories.csv"),
             trd=rd("arg_decile_trajectories_minus_random.csv"))
    plt = D.setup_mpl()
    D.FOOT = "decile-graded test-set eval · baseline condition, GPT-2 large and LSTM · decile_reanalysis_report.py"
    doc = D.Doc(a.out, plt)
    p_summary(doc, T)
    p_unmodified(doc, T)
    p_reference(doc, T)
    p_agreement(doc, T)
    p_owngrade(doc, T)
    p_slopes(doc, T)
    p_bypercent(doc, T)
    if T["tr"] is not None:
        for arch in ARCHS:
            fig = p_traj(doc, T, arch)
            doc.bottom_notes(fig, traj_notes(T, arch), 0.275)
            doc.save(fig)
        p_tilt_time(doc, T)
    p_methods(doc, T)
    doc.close()
    print(f"wrote {a.out} ({doc.n} pages)")


if __name__ == "__main__":
    main()
