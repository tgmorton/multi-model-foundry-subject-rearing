#!/usr/bin/env python3
"""PDF: how recoverable are the subjects that children and adults leave out (CHILDES, train_90M)?

Reads sites_scored.parquet and decile_distributions.csv (analyze.py) and funnel counts from null_sites.parquet.

    python analysis/recoverability/natural_drop/report.py --dir <site dir> --out <pdf>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval_v2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import priming_deep_report as D  # noqa: E402

SECTION = "Natural drops · recoverability"
FORMS = ["i", "you", "he", "she", "it", "we", "they"]
SHOW = {"i": "I", "you": "you", "he": "he", "she": "she", "it": "it", "we": "we", "they": "they"}
ROLES = {"child": "children", "adult": "adults"}
POSS = {"child": "children's", "adult": "adults'"}
CLS = {"inflected": "verb form rules out an imperative", "stative": "want / think / know / need ...", "bare": "bare verb"}
C_OVERT, C_INF, C_STA, C_BARE = D.CTRL, D.BLUE, D.ORANGE, D.AQUA
CC = {"overt": C_OVERT, "inflected": C_INF, "stative": C_STA, "bare": C_BARE}
BANDS = ["under 2", "2", "3", "4", "5 and over"]
BAND_TICK = ["<2", "2", "3", "4", "5+"]
MAIN = ["inflected", "stative"]


def ci(x, f=np.mean, n=800, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = f(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def pct(x):
    return f"{100 * x:.0f}%"


def grp(d, role, cls):
    if cls == "overt":
        return d[(d.role == role) & (d.kind == "overt")]
    if cls == "dropped":
        return d[(d.role == role) & (d.kind == "null") & d.verb_class.isin(MAIN)]
    return d[(d.role == role) & (d.kind == "null") & (d.verb_class == cls)]


# ---------------------------------------------------------------------------------------------
def p_summary(doc, d, sites, S):
    col1 = [
        ("The question", [
            "When a child or an adult in the corpus leaves out a subject, how recoverable was it? The study orders its "
            "removals by a rater's surprisal for each overt pronoun. This puts the subjects that speakers actually "
            "dropped on the same scale.",
        ]),
        ("What was done", [
            f"Sites: finite clauses with no subject in CHILDES (the corpus null-subject annotation), {S['n_all']:,} in all. "
            f"After removing formulas (thank you), parse errors and unusable tokens, {S['n_main']:,} remain where an "
            "imperative reading is ruled out by the verb form or unlikely for the verb; a further "
            f"{S['n_bare']:,} bare-verb sites are kept apart.",
            "At each site a subject slot is opened where an overt subject would stand, and the selection rater "
            "(the one that ranked the corpus) is asked what belongs there, with the same 250 pieces of text on "
            "each side that it had for the ranking.",
            f"Comparison: {S['n_overt']:,} overt pronoun subjects of the same speakers, re-scored with the same code. "
            f"The re-scoring reproduces the stored ranking (correlation {S['corr']:.3f}, same decile for {pct(S['same_dec'])}).",
        ]),
        ("The measure", [
            "The pronoun that was dropped is not known. The measure is the probability of the rater's best pronoun "
            "guess: if that guess is right, the dropped pronoun's surprisal is minus its log, and whatever was meant, "
            "the surprisal is at least that. It is placed on the ranking's frozen deciles (0 = most recoverable). "
            "For overt pronouns the same best-guess measure is computed, so the two are compared like for like.",
        ]),
    ]
    col2 = [
        ("What came out", S["findings"]),
        ("Reading", S["reading"]),
    ]
    doc.text_page(SECTION, "How recoverable are the subjects speakers leave out?", S["lede"], [col1, col2])


def p_sites(doc, d, sites, S):
    fig, _ = doc.page(SECTION, "Which sites, and what the rater says belongs there",
                      "A dropped subject is a finite clause with no subject whose verb form cannot be an imperative (goes, "
                      "fell, is, got, can't), or whose verb is rarely one (want, think, know). Bare verbs that could be "
                      "imperatives are kept apart. Examples are random draws; the bracket is the opened slot.")
    y0 = 0.80
    fig.text(0.055, y0, "SITES", family=D.HEAD, size=7.8, color=D.MUTED, weight="demibold", va="top")
    rows = [("finite clauses with no subject", S["n_all"]), ("  formulas (thank you, excuse me)", -S["n_formula"]),
            ("  a noun or pronoun stands right before the verb (parse error)", -S["n_nominal"]),
            ("  clitic verb attached to the word before (here's, where's)", -S["n_glued"]),
            ("  unusable tokens, speaker role unknown", -S["n_other"]),
            ("  here / there or a question word before the verb (inverted order)", -S["n_inverted"]),
            ("kept", S["n_main"] + S["n_bare"])]
    y = y0 - 0.03
    for lab, v in rows:
        fig.text(0.055, y, lab, family=D.BODY, size=9.4, color=D.INK, va="top")
        fig.text(0.44, y, f"{v:,}", family=D.BODY, size=9.4, color=D.INK, va="top", ha="right")
        y -= 0.024
    y -= 0.012
    fig.text(0.055, y, "KEPT, BY SPEAKER AND VERB", family=D.HEAD, size=7.8, color=D.MUTED, weight="demibold", va="top")
    y -= 0.03
    fig.text(0.30, y, "children", family=D.HEAD, size=8, color=D.MUTED, va="top", ha="right")
    fig.text(0.44, y, "adults", family=D.HEAD, size=8, color=D.MUTED, va="top", ha="right")
    y -= 0.024
    for cls in ("inflected", "stative", "bare"):
        fig.text(0.055, y, CLS[cls], family=D.BODY, size=9.4, color=D.INK, va="top")
        for x, role in ((0.30, "child"), (0.44, "adult")):
            fig.text(x, y, f"{len(grp(d, role, cls)):,}", family=D.BODY, size=9.4, color=D.INK, va="top", ha="right")
        y -= 0.024
    fig.text(0.055, y, "overt pronoun subjects re-scored", family=D.BODY, size=9.4, color=D.INK, va="top")
    for x, role in ((0.30, "child"), (0.44, "adult")):
        fig.text(x, y, f"{len(grp(d, role, 'overt')):,}", family=D.BODY, size=9.4, color=D.INK, va="top", ha="right")
    y -= 0.045
    fig.text(0.055, y, "FIRST IN THE UTTERANCE", family=D.HEAD, size=7.8, color=D.MUTED, weight="demibold", va="top")
    y -= 0.03
    for lab, cls in (("dropped subjects", "dropped"), ("overt pronoun subjects", "overt")):
        fig.text(0.055, y, lab, family=D.BODY, size=9.4, color=D.INK, va="top")
        for x, role in ((0.30, "child"), (0.44, "adult")):
            fig.text(x, y, pct(grp(d, role, cls).first_in_utterance.mean()), family=D.BODY, size=9.4, color=D.INK, va="top", ha="right")
        y -= 0.024

    y -= 0.03
    fig.text(0.055, y, "A CORRECTION TO THE EARLIER CORPUS NUMBERS", family=D.HEAD, size=7.8, color=D.MUTED, weight="demibold", va="top")
    y -= 0.028
    import textwrap
    note = ("The February table that split null subjects into imperative and non-imperative took clause rows from two "
            "annotation runs whose sentence ids collide. Rebuilt from the matching files, finite non-imperative nulls by "
            f"that rule number {S['n_old_rule']:,} in CHILDES, not 61,062; and the rule calls every subjectless main clause "
            "with a verb an imperative, so \"goes there\" and \"want that\" were counted as imperatives. The sites here "
            "are classed by verb form instead. The corpus report's null rates (5.2% child, 3.6% adult) do not depend on "
            "that table.")
    fig.text(0.055, y, "\n".join(textwrap.wrap(note, 62)), family=D.BODY, size=9.0, color=D.INK, va="top", linespacing=1.38)

    rng = np.random.default_rng(11)
    x0, y = 0.50, y0
    for role in ("child", "adult"):
        fig.text(x0, y, f"EXAMPLES, {ROLES[role].upper()}", family=D.HEAD, size=7.8, color=D.MUTED, weight="demibold", va="top")
        y -= 0.028
        g = grp(d, role, "dropped")
        g = g[g.line_text.str.len() < 62]
        for r in g.iloc[rng.permutation(len(g))[:9]].itertuples():
            ln = r.scored_line.split("\t", 1)[-1].strip()
            text = ln.replace("it ", "[   ] ", 1) if ln.startswith("it ") else r.line_text.split("\t", 1)[-1].strip()
            if not ln.startswith("it "):
                text = (r.before + " [   ] " + r.after).strip()
            guesses = sorted(((getattr(r, f"p_{f}"), SHOW[f]) for f in FORMS), reverse=True)[:2]
            gtxt = ", ".join(f"{w} {p:.2f}" for p, w in guesses)
            fig.text(x0, y, text[:58], family=D.MONO, size=7.4, color=D.INK, va="top")
            fig.text(0.945, y, gtxt, family=D.MONO, size=7.4, color=D.INK2, va="top", ha="right")
            y -= 0.0215
        y -= 0.02
    doc.save(fig)


def p_deciles(doc, d, dist, S):
    from matplotlib.lines import Line2D
    notes = S["notes_deciles"]
    fig, box = doc.page(SECTION, "Dropped subjects sit at the less recoverable end",
                        "Share of sites in each decile of the corpus ranking, by the rater's best pronoun guess at the slot. "
                        "Decile 0 holds the pronouns the rater recovers almost with certainty; decile 9 the ones it can "
                        "barely guess. Grey: overt pronoun subjects of the same speakers, measured the same way.", notes)
    gs = fig.add_gridspec(2, 1, **{**box, "bottom": 0.185}, hspace=0.36)
    for i, role in enumerate(("child", "adult")):
        ax = fig.add_subplot(gs[i, 0])
        w = 0.27
        for k, cls in enumerate(("overt", "inflected", "stative")):
            g = grp(d, role, cls)
            sh = g.dec_best.value_counts(normalize=True).reindex(range(10), fill_value=0.0)
            ax.bar(np.arange(10) + (k - 1) * w, sh.to_numpy(), width=w - 0.03, color=CC[cls])
        ax.set_xticks(range(10))
        ax.set_xlim(-0.6, 9.6)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("share of sites")
        ax.set_title(f"{ROLES[role].capitalize()}")
        if i == 1:
            ax.set_xlabel("decile of the best-guess surprisal  (0 = most recoverable)")
    fig.legend(handles=[Line2D([], [], color=C_OVERT, lw=6, label="overt pronoun subjects"),
                        Line2D([], [], color=C_INF, lw=6, label=f"dropped: {CLS['inflected']}"),
                        Line2D([], [], color=C_STA, lw=6, label=f"dropped: {CLS['stative']}")],
               loc="lower left", bbox_to_anchor=(0.07, 0.06), ncol=1, fontsize=7.8, handlelength=1.4)
    doc.save(fig)


def p_expected(doc, d, dist, S):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, "On the ranking's own deciles",
                        "The ranking's deciles are defined on the surprisal of the pronoun that was actually there. For overt "
                        "pronouns that is known (grey). For dropped subjects it is estimated: each site inherits the decile "
                        "distribution of overt pronouns with the same best-guess decile.", S["notes_expected"])
    gs = fig.add_gridspec(2, 1, **{**box, "bottom": 0.185}, hspace=0.36)
    for i, role in enumerate(("child", "adult")):
        ax = fig.add_subplot(gs[i, 0])
        t = dist[dist.role == role]
        o = t[t.verb_class == "overt"].set_index("decile").share_expected
        ax.plot(o.index, o.to_numpy(), color=C_OVERT, lw=2.2, marker="o", ms=4)
        for cls in MAIN:
            e = t[t.verb_class == cls].set_index("decile").share_expected
            ax.plot(e.index, e.to_numpy(), color=CC[cls], lw=2.0, marker="o", ms=4)
        ax.axhline(0.1, color=D.MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.set_xticks(range(10))
        ax.set_xlim(-0.4, 9.4)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("share of sites")
        ax.set_title(ROLES[role].capitalize())
        if i == 1:
            ax.set_xlabel("decile of the corpus ranking  (0 = most recoverable)")
    fig.legend(handles=[Line2D([], [], color=C_OVERT, lw=2.2, marker="o", ms=4, label="overt pronoun subjects (actual)"),
                        Line2D([], [], color=C_INF, lw=2, marker="o", ms=4, label=f"dropped, estimated: {CLS['inflected']}"),
                        Line2D([], [], color=C_STA, lw=2, marker="o", ms=4, label=f"dropped, estimated: {CLS['stative']}"),
                        Line2D([], [], color=D.MUTED, lw=0.8, ls=(0, (2, 2)), label="one tenth (whole-corpus share per decile)")],
               loc="lower left", bbox_to_anchor=(0.07, 0.045), ncol=2, fontsize=7.6, handlelength=2.2, columnspacing=1.6)
    doc.save(fig)


def p_position(doc, d, S):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, "Not a matter of where in the utterance",
                        "Dropped subjects are more often the first thing in the utterance than overt pronouns are, and a "
                        "first word has less text before it. The comparison is repeated within each position. Bars: "
                        "share of sites where the rater's best guess has probability 0.9 or more; lines are 95% intervals.",
                        S["notes_position"])
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.30}, wspace=0.12)
    for j, role in enumerate(("child", "adult")):
        ax = fig.add_subplot(gs[0, j])
        for p, first in enumerate((True, False)):
            for k, cls in enumerate(("overt", "inflected", "stative")):
                g = grp(d, role, cls)
                x = (g[g.first_in_utterance == first].p_best >= 0.9).astype(float)
                pos = p + (k - 1) * 0.26
                ax.bar(pos, x.mean(), width=0.23, color=CC[cls])
                lo, hi = ci(x)
                ax.plot([pos, pos], [lo, hi], color=D.INK2, lw=1.0)
                ax.annotate(f"{x.mean():.2f}", (pos, hi), xytext=(0, 3), textcoords="offset points", ha="center", size=7, color=D.INK2)
        ax.set_xticks([0, 1], ["first in the utterance", "later in the utterance"])
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.set_title(ROLES[role].capitalize())
        if j == 0:
            ax.set_ylabel("share with best guess at 0.9 or more")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=C_OVERT, lw=6, label="overt pronoun subjects"),
                        Line2D([], [], color=C_INF, lw=6, label=f"dropped: {CLS['inflected']}"),
                        Line2D([], [], color=C_STA, lw=6, label=f"dropped: {CLS['stative']}")],
               loc="upper left", bbox_to_anchor=(0.07, 0.235), ncol=3, fontsize=7.8, handlelength=1.4)
    doc.save(fig)


def p_which(doc, d, S):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, "Which pronoun is missing",
                        "The rater's best guess for each dropped subject, beside the pronouns the same speakers did say. "
                        "Share of sites.", S["notes_which"])
    gs = fig.add_gridspec(2, 1, **{**box, "bottom": 0.185}, hspace=0.36)
    for i, role in enumerate(("child", "adult")):
        ax = fig.add_subplot(gs[i, 0])
        w = 0.27
        series = [("overt", grp(d, role, "overt").form), ("inflected", grp(d, role, "inflected").best), ("stative", grp(d, role, "stative").best)]
        for k, (cls, s) in enumerate(series):
            sh = s.value_counts(normalize=True).reindex(FORMS, fill_value=0.0)
            ax.bar(np.arange(len(FORMS)) + (k - 1) * w, sh.to_numpy(), width=w - 0.03, color=CC[cls])
        ax.set_xticks(range(len(FORMS)), [SHOW[f] for f in FORMS])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("share of sites")
        ax.set_title(ROLES[role].capitalize())
    fig.legend(handles=[Line2D([], [], color=C_OVERT, lw=6, label="overt pronoun subjects (the pronoun said)"),
                        Line2D([], [], color=C_INF, lw=6, label=f"dropped, rater's best guess: {CLS['inflected']}"),
                        Line2D([], [], color=C_STA, lw=6, label=f"dropped, rater's best guess: {CLS['stative']}")],
               loc="lower left", bbox_to_anchor=(0.07, 0.06), ncol=1, fontsize=7.8, handlelength=1.4)
    doc.save(fig)


def p_byform(doc, d, S):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, "Pronoun by pronoun",
                        "The same comparison within each pronoun: overt sites by the pronoun that was said, dropped sites by "
                        "the rater's best guess. Share of sites where the best guess has probability 0.9 or more; lines are "
                        "95% intervals. Dropped sites of both main classes together.", S["notes_byform"])
    gs = fig.add_gridspec(2, 1, **{**box, "bottom": 0.185}, hspace=0.36)
    for i, role in enumerate(("child", "adult")):
        ax = fig.add_subplot(gs[i, 0])
        o, n = grp(d, role, "overt"), grp(d, role, "dropped")
        for k, f in enumerate(FORMS):
            for off, x, col in ((-0.2, (o[o.best == f].p_best >= 0.9).astype(float), C_OVERT),
                                (0.2, (n[n.best == f].p_best >= 0.9).astype(float), D.VIOLET)):
                if len(x) < 20:
                    continue
                ax.bar(k + off, x.mean(), width=0.36, color=col)
                lo, hi = ci(x)
                ax.plot([k + off, k + off], [lo, hi], color=D.INK2, lw=1.0)
                ax.annotate(f"{x.mean():.2f}", (k + off, hi), xytext=(0, 3), textcoords="offset points", ha="center", size=6.8, color=D.INK2)
        ax.set_xticks(range(len(FORMS)), [SHOW[f] for f in FORMS])
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("share at 0.9 or more")
        ax.set_title(ROLES[role].capitalize())
        if i == 1:
            ax.set_xlabel("the rater's best guess at the slot")
    fig.legend(handles=[Line2D([], [], color=C_OVERT, lw=6, label="overt pronoun subjects"),
                        Line2D([], [], color=D.VIOLET, lw=6, label="dropped subjects")],
               loc="lower left", bbox_to_anchor=(0.07, 0.07), ncol=2, fontsize=7.8, handlelength=1.4)
    doc.save(fig)


def p_two_reasons(doc, d, S):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, "Two ways a dropped subject can be hard to recover",
                        "Left: is a personal pronoun expected at the slot at all (probability 0.9 or more on the seven "
                        "pronouns together)? Right: among the sites where it is, does the rater know which pronoun (best "
                        "guess takes 0.9 or more of the pronoun probability)? Lines are 95% intervals.", S["notes_two"])
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.30}, wspace=0.14)
    for j, (title, fn) in enumerate((("A pronoun is expected", lambda g: (g.mass >= 0.9).astype(float)),
                                     ("Which pronoun is clear, given one is expected",
                                      lambda g: (g[g.mass >= 0.9].p_best / g[g.mass >= 0.9].mass >= 0.9).astype(float)))):
        ax = fig.add_subplot(gs[0, j])
        for p, role in enumerate(("child", "adult")):
            for k, cls in enumerate(("overt", "inflected", "stative")):
                x = fn(grp(d, role, cls))
                pos = p + (k - 1) * 0.26
                ax.bar(pos, x.mean(), width=0.23, color=CC[cls])
                lo, hi = ci(x)
                ax.plot([pos, pos], [lo, hi], color=D.INK2, lw=1.0)
                ax.annotate(f"{x.mean():.2f}", (pos, hi), xytext=(0, 3), textcoords="offset points", ha="center", size=7, color=D.INK2)
        ax.set_xticks([0, 1], ["children", "adults"])
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.4)
        if j == 0:
            ax.set_ylabel("share of sites")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=C_OVERT, lw=6, label="overt pronoun subjects"),
                        Line2D([], [], color=C_INF, lw=6, label=f"dropped: {CLS['inflected']}"),
                        Line2D([], [], color=C_STA, lw=6, label=f"dropped: {CLS['stative']}")],
               loc="upper left", bbox_to_anchor=(0.07, 0.235), ncol=3, fontsize=7.8, handlelength=1.4)
    doc.save(fig)


def p_age(doc, d, S):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, "By the child's age",
                        "Share of sites where the rater's best guess has probability 0.9 or more, by the age of the child in "
                        "the recording. Left: the child's own speech. Right: the adults talking to a child of that age. "
                        "Bands are 95% intervals from resampling sites.", S["notes_age"])
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.30}, wspace=0.12)
    for j, role in enumerate(("child", "adult")):
        ax = fig.add_subplot(gs[0, j])
        for cls in ("overt", "inflected", "stative"):
            g = grp(d, role, cls)
            m, lo, hi, xs = [], [], [], []
            for k, b in enumerate(BANDS):
                x = (g[g.age_band == b].p_best >= 0.9).astype(float)
                if len(x) < 30:
                    continue
                l, h = ci(x)
                xs.append(k), m.append(x.mean()), lo.append(l), hi.append(h)
            ax.fill_between(xs, lo, hi, color=CC[cls], alpha=0.14, lw=0)
            ax.plot(xs, m, color=CC[cls], lw=2, marker="o", ms=4.5)
        ax.set_xticks(range(len(BANDS)), BAND_TICK)
        ax.set_xlim(-0.3, len(BANDS) - 0.7)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("age of the child (years)")
        ax.set_title("Children's own speech" if role == "child" else "Adults speaking to the child")
        if j == 0:
            ax.set_ylabel("share with best guess at 0.9 or more")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=C_OVERT, lw=2, marker="o", ms=4, label="overt pronoun subjects"),
                        Line2D([], [], color=C_INF, lw=2, marker="o", ms=4, label=f"dropped: {CLS['inflected']}"),
                        Line2D([], [], color=C_STA, lw=2, marker="o", ms=4, label=f"dropped: {CLS['stative']}")],
               loc="upper left", bbox_to_anchor=(0.07, 0.235), ncol=3, fontsize=7.8, handlelength=2.0)
    doc.save(fig)


def p_slot(doc, d, S):
    from matplotlib.patches import Patch
    fig, box = doc.page(SECTION, "What the rater puts in the slot",
                        "Mean probability the rater gives, at the opened slot, to a personal pronoun, to that / this / there, "
                        "to a question word, and to everything else (nouns, names, other words).", S["notes_slot"])
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.30}, wspace=0.12)
    parts = [("personal pronoun", D.BLUE), ("that, this, there", D.AQUA), ("what, who", D.YELLOW), ("anything else", D.RULE)]
    for j, role in enumerate(("child", "adult")):
        ax = fig.add_subplot(gs[0, j])
        for k, cls in enumerate(("overt", "inflected", "stative", "bare")):
            g = grp(d, role, cls)
            v = [g.mass.mean(), (g.x_that + g.x_this + g.x_there).mean(), (g.x_what + g.x_who).mean()]
            v.append(max(0.0, 1 - sum(v)))
            b = 0.0
            for (lab, col), val in zip(parts, v):
                ax.bar(k, val, bottom=b, width=0.62, color=col, edgecolor="white", linewidth=1.5)
                if val > 0.06:
                    ax.text(k, b + val / 2, f"{val:.2f}", ha="center", va="center", size=7.4, color="white" if col != D.RULE else D.INK2)
                b += val
        ax.set_xticks(range(4), ["overt\npronoun", "dropped:\ninflected", "dropped:\nwant, think ...", "dropped:\nbare verb"], fontsize=7.6)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.set_title(ROLES[role].capitalize())
        if j == 0:
            ax.set_ylabel("mean probability at the slot")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Patch(color=c, label=lab) for lab, c in parts], loc="upper left", bbox_to_anchor=(0.07, 0.225), ncol=4,
               fontsize=7.8)
    doc.save(fig)


def p_methods(doc, d, S):
    col1 = [
        ("Sites", [
            "Source: the clause_structure layer of the annotated train_90M corpus, CHILDES files only; finite clauses with "
            "subject_status none. Each sentence is mapped back to the raw corpus line (99.4% of sentences found) so the "
            "rater sees exactly the text it ranked, including the speaker tag.",
            "Slot: before the first auxiliary or verb of the clause, and before a preceding just / really / never-type "
            "adverb. A placeholder pronoun is inserted there, the line is re-tokenized, and the placeholder is masked. "
            "Sites right after here, there or a question word are left out (inverted order, nothing dropped).",
            "Classes: inflected = the first verbal element differs from its base form or is a modal (goes, fell, is, "
            "got, can't); stative = base form of want, need, like, love, know, think, guess, mean, have and similar; "
            "bare = any other base form. Children = the target child (CHI); adults = the other coded adult speakers.",
        ]),
        ("Scoring", [
            "The rater is the one that ranked the corpus (bidirectional, 250 pieces of the raw stream on each side). "
            "Probabilities are summed over upper- and lower-case and leading-space variants of each pronoun.",
            f"Check: for the {S['n_overt']:,} overt pronouns the re-scored surprisal matches the stored ranking "
            f"(correlation {S['corr']:.3f}; same decile for {pct(S['same_dec'])}).",
        ]),
    ]
    col2 = [
        ("Limits", [
            "The dropped word is unknown. The best-guess measure is a floor on its surprisal, so the deciles shown are, "
            "if anything, too favourable to the dropped subjects.",
            "The estimated deciles assume a dropped pronoun is as likely to be the rater's best guess as an overt pronoun "
            "with the same best-guess probability. Nothing checks that.",
            "The missing subject need not be a personal pronoun. Where the rater prefers that, there or a noun, the "
            "pronoun probabilities are low and the site lands in a high decile.",
            "The rater learned English from text where subjects are almost always present. Its difficulty at a slot "
            "measures how far the text around the slot pins the subject down, not what the speaker and hearer knew "
            "from the situation.",
            "Site detection rests on an automatic parse of child speech. The classes were checked by reading samples, "
            "not by annotation; the bare-verb class in particular mixes imperatives with real drops.",
            "Overt pronouns: a random sample per speaker group from the pronoun subjects the rater ranked.",
        ]),
        ("Files", [
            "data/recoverability/analysis/natural_drop/: null_sites.parquet, overt_sites.parquet, scores.parquet, "
            "sites_scored.parquet, decile_distributions.csv, childes_null_type_regenerated.parquet. Scripts "
            "(uncommitted): analysis/recoverability/natural_drop/{build_sites,score_sites,analyze,report}.py.",
        ]),
    ]
    doc.text_page(SECTION, "How this was done", "Sites, scoring, and what the measure can and cannot say.", [col1, col2])


# ---------------------------------------------------------------------------------------------
def stats(d, dist, sites, extra):
    S = dict(extra)
    kept = sites[~sites.nominal_before & ~sites.glued & sites.verb_class.isin(["inflected", "stative", "bare"]) & sites.role.isin(["child", "adult"])]
    S["n_all"] = len(sites)
    S["n_formula"] = int((sites.verb_class == "formula").sum())
    rest = sites[sites.verb_class != "formula"]
    S["n_nominal"] = int(rest.nominal_before.sum())
    rest = rest[~rest.nominal_before]
    S["n_glued"] = int(rest.glued.sum())
    S["n_other"] = len(rest[~rest.glued]) - len(kept)
    nd = d[d.kind == "null"]
    S["n_main"] = int(nd.verb_class.isin(MAIN).sum())
    S["n_bare"] = int((nd.verb_class == "bare").sum())
    S["n_inverted"] = len(kept) - len(nd)
    S["n_overt"] = int((d.kind == "overt").sum())

    def hi(role, cls):
        return float((grp(d, role, cls).p_best >= 0.9).mean())

    def med(role, cls):
        return float(grp(d, role, cls).p_best.median())

    def low_half(role, cls):
        return float((grp(d, role, cls).dec_best <= 4).mean())

    def exp_low(role, cls):
        t = dist[(dist.role == role) & (dist.verb_class == cls)]
        return float(t[t.decile <= 4].share_expected.sum())

    def exp_top3(role, cls):
        t = dist[(dist.role == role) & (dist.verb_class == cls)]
        return float(t[t.decile >= 7].share_expected.sum())

    S["hi"], S["med"], S["low_half"], S["exp_low"] = hi, med, low_half, exp_low
    acc = {r: float((grp(d, r, "overt").best == grp(d, r, "overt").form).mean()) for r in ROLES}
    S["lede"] = (f"Less recoverable than the pronouns they do say. The rater's best guess reaches probability 0.9 at "
                 f"{pct(hi('child', 'dropped'))} of children's dropped subjects and {pct(hi('adult', 'dropped'))} of adults', against "
                 f"{pct(hi('child', 'overt'))} and {pct(hi('adult', 'overt'))} of their overt pronoun subjects.")
    S["findings"] = [
        f"Overt pronoun subjects are easy for the rater: its best guess is the pronoun that was said for {pct(acc['child'])} "
        f"(children) and {pct(acc['adult'])} (adults) of them, with a median probability of {med('child', 'overt'):.2f} and "
        f"{med('adult', 'overt'):.2f}.",
        f"Dropped subjects are harder. Median best-guess probability: children {med('child', 'inflected'):.2f} where the verb "
        f"form rules out an imperative and {med('child', 'stative'):.2f} with want / think / know; adults "
        f"{med('adult', 'inflected'):.2f} and {med('adult', 'stative'):.2f}.",
        f"On the ranking's deciles, {pct(low_half('child', 'dropped'))} of children's and {pct(low_half('adult', 'dropped'))} of "
        f"adults' dropped subjects fall in the more recoverable half (deciles 0-4) by the best guess, against "
        f"{pct(low_half('child', 'overt'))} and {pct(low_half('adult', 'overt'))} of overt pronouns measured the same way.",
        f"Children and adults differ little. The larger difference is the kind of drop: drops with want, think or know, "
        f"mostly first person, come closest to overt pronouns ({pct(hi('child', 'stative'))} of children's and "
        f"{pct(hi('adult', 'stative'))} of adults' at 0.9 or more); the rest, mostly third person, are far from them "
        f"({pct(hi('child', 'inflected'))} and {pct(hi('adult', 'inflected'))}).",
    ]
    S["reading"] = [
        "Speakers do not leave out the subjects that the text makes most predictable. By the rater's measure a natural "
        "drop looks like a pronoun from the upper deciles, which is the end the least-recoverable-first order removes. "
        "The rater reads text only, so this says the drops are not licensed by the surrounding words; it says nothing "
        "about what the situation made obvious.",
    ]

    def dline(role):
        o, i, s = grp(d, role, "overt"), grp(d, role, "inflected"), grp(d, role, "stative")
        return (f"{ROLES[role].capitalize()}: deciles 0-4 hold {pct((o.dec_best <= 4).mean())} of overt pronouns, "
                f"{pct((i.dec_best <= 4).mean())} of inflected-verb drops and {pct((s.dec_best <= 4).mean())} of want / think drops. "
                f"Deciles 8-9 hold {pct((o.dec_best >= 8).mean())}, {pct((i.dec_best >= 8).mean())} and {pct((s.dec_best >= 8).mean())}.")
    S["notes_deciles"] = [
        "Grey bars lean left: for an overt pronoun the rater's best guess is usually near certain.",
        "Blue bars lean right: where the verb form shows a subject is missing, the rater is often unsure which.",
        dline("child"), dline("adult"),
        "The measure is a floor for the dropped subjects (the best guess may be wrong), so the true picture is no "
        "further left than this.",
    ]
    S["notes_expected"] = [
        "Grey is not flat at one tenth because CHILDES pronouns are not a tenth of every decile of the whole corpus.",
        f"Estimated share in deciles 0-4: children {pct(exp_low('child', 'inflected'))} (inflected) and "
        f"{pct(exp_low('child', 'stative'))} (want / think), against {pct(exp_low('child', 'overt'))} of their overt pronouns; "
        f"adults {pct(exp_low('adult', 'inflected'))} and {pct(exp_low('adult', 'stative'))}, against {pct(exp_low('adult', 'overt'))}.",
        f"Estimated share in deciles 7-9: children {pct(exp_top3('child', 'inflected'))} and {pct(exp_top3('child', 'stative'))} "
        f"against {pct(exp_top3('child', 'overt'))}; adults {pct(exp_top3('adult', 'inflected'))} and "
        f"{pct(exp_top3('adult', 'stative'))} against {pct(exp_top3('adult', 'overt'))}.",
        "This is the scale the removal orders use. A learner that dropped what these speakers drop would be removing "
        "mostly from the upper deciles.",
        "The estimate assumes a dropped pronoun is as likely to be the rater's best guess as an overt one with the same "
        "best-guess probability.",
    ]

    def pos_line(role):
        o, i = grp(d, role, "overt"), grp(d, role, "inflected")
        f = lambda g, first: (g[g.first_in_utterance == first].p_best >= 0.9).mean()  # noqa: E731
        return (f"{ROLES[role].capitalize()}, first in the utterance: {f(o, True):.2f} overt against {f(i, True):.2f} dropped "
                f"(inflected). Later: {f(o, False):.2f} against {f(i, False):.2f}.")
    S["notes_position"] = [
        f"First in the utterance: {pct(grp(d, 'child', 'dropped').first_in_utterance.mean())} of children's dropped subjects and "
        f"{pct(grp(d, 'child', 'overt').first_in_utterance.mean())} of their overt pronouns; adults "
        f"{pct(grp(d, 'adult', 'dropped').first_in_utterance.mean())} and {pct(grp(d, 'adult', 'overt').first_in_utterance.mean())}.",
        pos_line("child"), pos_line("adult"),
        "The gap between overt and dropped holds in both positions, so it is not produced by drops being utterance-initial.",
    ]

    def top_forms(s, k=3):
        v = s.value_counts(normalize=True)
        return ", ".join(f"{SHOW[f]} {pct(v[f])}" for f in v.index[:k])
    S["notes_which"] = [
        f"Children say: {top_forms(grp(d, 'child', 'overt').form)}. Their inflected-verb drops are guessed as "
        f"{top_forms(grp(d, 'child', 'inflected').best)}; their want / think drops as {top_forms(grp(d, 'child', 'stative').best)}.",
        f"Adults say: {top_forms(grp(d, 'adult', 'overt').form)}. Inflected-verb drops: {top_forms(grp(d, 'adult', 'inflected').best)}; "
        f"want / think drops: {top_forms(grp(d, 'adult', 'stative').best)}.",
        "Drops with an inflected verb are mostly third person (goes, looks, is, fell) or got. Third-person referents are "
        "the ones a reader of the text has to work out, which is why these are the hard ones.",
        "Drops with want, think or know are first or second person: the speaker or the hearer, who are always available.",
        "These are the rater's guesses, not annotations of what was meant.",
    ]

    def form_line(role):
        o, n = grp(d, role, "overt"), grp(d, role, "dropped")
        out = []
        for f in FORMS:
            a_, b_ = o[o.best == f].p_best, n[n.best == f].p_best
            if len(a_) >= 20 and len(b_) >= 20:
                out.append(f"{SHOW[f]} {(a_ >= 0.9).mean():.2f} / {(b_ >= 0.9).mean():.2f}")
        return f"{ROLES[role].capitalize()} (overt / dropped): " + ", ".join(out) + "."
    S["notes_byform"] = [
        form_line("child"), form_line("adult"),
        "The gap is there within every pronoun, so it is not produced by drops being mostly third person or mostly I.",
        "Sites are grouped by the rater's best guess on both sides, so an overt site is counted under the pronoun the "
        "rater preferred, which is the pronoun said for most of them.",
    ]

    def two(role, cls):
        g = grp(d, role, cls)
        h = g[g.mass >= 0.9]
        return (g.mass >= 0.9).mean(), (h.p_best / h.mass >= 0.9).mean()
    S["notes_two"] = [
        "A pronoun is expected at " + ", ".join(f"{two(r, c)[0]:.2f} of {POSS[r]} {lab}" for r in ROLES for c, lab in
                                                 (("overt", "overt sites"), ("inflected", "inflected-verb drops"))) + ".",
        "Where a pronoun is expected, which one is clear at " + ", ".join(
            f"{two(r, c)[1]:.2f} of {POSS[r]} {lab}" for r in ROLES for c, lab in
            (("overt", "overt sites"), ("inflected", "inflected-verb drops"), ("stative", "want / think drops"))) + ".",
        "Both steps are harder at a dropped subject. Even where the rater is sure a pronoun is missing, it is less sure "
        "which one than it is for a pronoun that was said.",
        "So the result is not only that some missing subjects were never pronouns.",
    ]

    def age_line(role, cls, label):
        g = grp(d, role, cls)
        v = [(b, (g[g.age_band == b].p_best >= 0.9).mean(), int((g.age_band == b).sum())) for b in BANDS]
        v = [x for x in v if x[2] >= 30]
        return f"{label}: " + (", ".join(f"{b} {m:.2f}" for b, m, _ in v) if v else "too few sites") + "."
    S["notes_age"] = [
        age_line("child", "overt", "Children's overt pronouns"), age_line("child", "inflected", "Children's inflected-verb drops"),
        age_line("child", "stative", "Children's want / think drops"),
        "Number of children's dropped subjects by age: " + ", ".join(
            f"{b} {int((grp(d, 'child', 'dropped').age_band == b).sum()):,}" for b in BANDS) + ".",
        "For inflected-verb drops the gap to overt pronouns is there at every age, in the child's speech and in the "
        "adults'. Everything rises a little with age.",
    ]

    def slot_line(role):
        return (f"{ROLES[role].capitalize()}: a personal pronoun gets {grp(d, role, 'overt').mass.mean():.2f} of the probability "
                f"at overt sites, {grp(d, role, 'inflected').mass.mean():.2f} at inflected-verb drops, "
                f"{grp(d, role, 'stative').mass.mean():.2f} at want / think drops and {grp(d, role, 'bare').mass.mean():.2f} at bare verbs.")
    S["notes_slot"] = [
        slot_line("child"), slot_line("adult"),
        "At inflected-verb drops part of the probability goes to that, this, there and to nouns: the missing subject is "
        "often not a personal pronoun at all (\"[that] goes there\", \"[there] is two\").",
        "At bare verbs the rater mostly does not expect a subject pronoun. Many of these are imperatives or fragments, "
        "which is why the class is kept out of the main comparison.",
    ]
    return S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_parquet(a.dir / "sites_scored.parquet")
    dist = pd.read_csv(a.dir / "decile_distributions.csv")
    sites = pd.read_parquet(a.dir / "null_sites.parquet")
    reg = pd.read_parquet(a.dir / "childes_null_type_regenerated.parquet", columns=["is_finite", "null_type"])
    o = d[d.kind == "overt"]
    from analyze import decile  # noqa: E402
    extra = dict(corr=float(np.corrcoef(o.surp_exact, o.s_robbi)[0, 1]), same_dec=float(np.mean(decile(o.surp_exact) == o.d_robbi)),
                 n_old_rule=int((reg.is_finite & (reg.null_type == "non_imperative_null")).sum()))
    S = stats(d, dist, sites, extra)
    plt = D.setup_mpl()
    D.FOOT = "CHILDES (train_90M) · subjects left out by children and adults, scored by the selection rater · natural_drop/report.py"
    doc = D.Doc(a.out, plt)
    p_summary(doc, d, sites, S)
    p_sites(doc, d, sites, S)
    p_deciles(doc, d, dist, S)
    p_expected(doc, d, dist, S)
    p_position(doc, d, S)
    p_which(doc, d, S)
    p_byform(doc, d, S)
    p_slot(doc, d, S)
    p_two_reasons(doc, d, S)
    p_age(doc, d, S)
    p_methods(doc, d, S)
    doc.close()
    print(f"wrote {a.out} ({doc.n} pages)")


if __name__ == "__main__":
    main()
