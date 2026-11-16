#!/usr/bin/env python3
"""PDF: where the pronouns in our eval stimuli sit on the corpus recoverability scale.

Inputs, all under data/recoverability/analysis/eval_stimuli/ in the main checkout:
  robbi__matched_v2__baseline.parquet, robbi__nopron_v2__baseline.parquet
      per-pronoun scores from analysis/recoverability/eval_stimuli/score_eval_stimuli.py
  gpt2l_base_items_endstate.parquet
      item-level end-state of the 28 baseline gpt2_large models (matched-v1 pairs)
and the corpus tables in data/recoverability/analysis/survey/ (population_features,
linesep_test_joined).

    python analysis/eval_v2/eval_recoverability_report.py --data <main>/data/recoverability/analysis \\
        --out eval_recoverability_baseline.pdf
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

KEY = ["category", "condition", "item_id"]
SUBJ = ["subject_drop", "subject_drop_no_agreement"]
PERS = ["subj_1sg", "subj_1pl", "subj_2sg", "subj_2pl", "subj_3sg", "subj_3pl"]
CATS = ["subject_drop", "subject_drop_no_agreement", "expletive", "embedded_drop", "conjunction",
        "object_drop", "extraction", "control"]
CAT_LABEL = {**D.SHORT, "object_drop": "object drop (it / them)", "extraction": "extraction (that)",
             "control": "control (him)", "embedded_drop": "embedded", "conjunction": "conjunction"}
BANDS = ["d0–1", "d2–4", "d5–7", "d8–9"]
BAND_C = ["#cfe0f5", "#8fb6e6", "#4a86cf", "#1c4f94"]          # one hue, light = most recoverable
LAYOUT_C = {"two_lines": "#8a8878", "one_line": "#2a78d6"}
BAND3 = ["d1–4", "d5–7", "d8–9"]
BAND3_C = ["#8fb6e6", "#4a86cf", "#1c4f94"]


def band(dec):
    return pd.cut(dec, [-1, 1, 4, 7, 9], labels=BANDS)


def crit(d, layout="one_line", render="natural"):
    return d[(d.role == "critical") & (d.render == render) & (d.layout == layout)]


def decile_counts(x):
    return x.decile.value_counts().reindex(range(10), fill_value=0)


def p_position(doc, a):
    s2, s1 = crit(a[a.category.isin(SUBJ)], "two_lines"), crit(a[a.category.isin(SUBJ)], "one_line")
    fig, box = doc.page(
        "Step 1 · baseline stimuli, rater robbi", "Our subject pronouns on the corpus decile scale",
        "The 288 subject-drop items, each scored with the code that ranked the corpus. Decile 0 is the most recoverable tenth "
        "of corpus pronouns, decile 9 the least. The same items are scored two ways: the corpus way (each sentence its own "
        "line, lines joined with no separator), and as one line of running text.",
        notes=[f"Scored the corpus way, all {len(s2)} sit in decile 9 (median {s2.surp.median():.1f} nats).",
               f"As running text they spread over deciles 1 to 9 (median {s1.surp.median():.2f} nats, mean decile "
               f"{s1.decile.mean():.1f}).",
               "The difference is the line join. The pronoun starts its line, so its token has no leading space, and the "
               "rater does not expect that form after a full stop.",
               "Both are truecased. Left lowercased as the learners see them, the items score less recoverable still "
               "(page 3)."])
    gs = fig.add_gridspec(2, 1, **box, hspace=0.5)
    for r, (x, lay, ttl) in enumerate([(s2, "two_lines", "Scored the corpus way: target sentence on its own line"),
                                       (s1, "one_line", "Scored as running text: context and target on one line")]):
        ax = fig.add_subplot(gs[r])
        c = decile_counts(x)
        ax.bar(c.index, c.values, 0.72, color=LAYOUT_C[lay])
        for k, v in c.items():
            if v:
                ax.text(k, v + 3, str(v), ha="center", size=7.6, color=D.INK2)
        ax.set_xticks(range(10))
        ax.set_ylim(0, 320)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("items")
        ax.set_title(ttl)
        if r:
            ax.set_xlabel("recoverability decile  (0 = most recoverable, 9 = least)")
    doc.save(fig)


def p_categories(doc, a):
    c = crit(a)
    t = pd.crosstab(c.category, band(c.decile)).reindex(CATS).reindex(columns=BANDS, fill_value=0)
    share = t.div(t.sum(axis=1), axis=0)
    med = c.groupby("category").surp.median()
    fig, box = doc.page(
        "Step 1 · baseline stimuli, rater robbi", "Every construction, scored as running text",
        "The element that alternates in each pair: the subject pronoun, the expletive, the object pronoun, 'that', or the "
        "resumptive 'him'. Bands group the deciles; the corpus has 20%, 30%, 30% and 20% of its pronouns in them. The "
        "corpus deciles were defined on subject pronouns only, so the last three rows are off that scale's population.",
        notes=["Subject-drop items fall mostly in the middle and upper bands. Almost none reach the most recoverable "
               "fifth of the corpus.",
               "Embedded and expletive pronouns are the most recoverable. 'Him' in control is not recoverable at all, "
               "as expected for a resumptive pronoun English does not allow.",
               "Median surprisal in nats is printed at the right of each bar."])
    ax = fig.add_subplot(fig.add_gridspec(1, 1, **{**box, "left": 0.2, "right": 0.6, "top": box["top"] - 0.03})[0])
    y = np.arange(len(CATS))[::-1]
    left = np.zeros(len(CATS))
    for b, col in zip(BANDS, BAND_C):
        v = share[b].values
        ax.barh(y, v, 0.62, left=left, color=col, label=b, edgecolor="white", linewidth=1.5)
        for yy, l, w, n in zip(y, left, v, t[b].values):
            if w > 0.055:
                ax.text(l + w / 2, yy, str(n), ha="center", va="center", size=7.6,
                        color="white" if col in BAND_C[2:] else D.INK)
        left += v
    for yy, cat in zip(y, CATS):
        ax.text(1.012, yy, f"{med[cat]:.2f}", va="center", size=7.8, color=D.INK2)
    ax.set_yticks(y, [CAT_LABEL.get(k, k) for k in CATS])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of items (counts inside the bars)")
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), title="decile band (left = most recoverable)",
              title_fontsize=7.6)
    doc.save(fig)


def p_person(doc, a, b):
    ca, cb = crit(a[a.category.isin(SUBJ)]), crit(b[b.category.isin(SUBJ)])
    lower = crit(a[a.category.isin(SUBJ)], "one_line", "as_is")
    fig, box = doc.page(
        "Step 1 · baseline stimuli, rater robbi", "By person, and what the context does",
        "Subject-drop items as running text. Top: the original contexts. Bottom: the pronoun-free contexts, where the "
        "context no longer mentions the referent with a pronoun. The target sentences are identical in both.",
        notes=[f"Original contexts: mean decile {ca.decile.mean():.1f}. Pronoun-free contexts: {cb.decile.mean():.1f}.",
               "Rewriting only the context moved 1st- and 2nd-person targets by about three deciles. 3rd-person contexts "
               "were mostly unchanged, and so are their scores.",
               "So the context is a usable handle: the same target pair can be placed in different bands.",
               f"Lowercased as the learners see them, the original items have mean decile {lower.decile.mean():.1f}: the "
               "rater is cased and expects a capital after a full stop."])
    gs = fig.add_gridspec(2, 1, **{**box, "top": box["top"] - 0.045}, hspace=0.45)
    for r, (x, ttl) in enumerate([(ca, "Original contexts"), (cb, "Pronoun-free contexts")]):
        ax = fig.add_subplot(gs[r])
        t = pd.crosstab(x.condition, band(x.decile)).reindex(PERS).reindex(columns=BANDS, fill_value=0)
        xx = np.arange(len(PERS))
        bottom = np.zeros(len(PERS))
        for bnd, col in zip(BANDS, BAND_C):
            ax.bar(xx, t[bnd].values, 0.6, bottom=bottom, color=col, label=bnd, edgecolor="white", linewidth=1.5)
            bottom += t[bnd].values
        ax.set_xticks(xx, [p.replace("subj_", "") for p in PERS])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("items (48 per person)")
        ax.set_title(ttl)
        if r == 0:
            ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.3), title="decile band (light = most recoverable)",
                      title_fontsize=7.6)
    doc.save(fig)


def p_corpus(doc, pop, ls):
    li, rest = ls[ls.char0], ls[~ls.char0]
    fig, box = doc.page(
        "Corpus reference", "Line-initial pronouns in the corpus ranking",
        "A 100K-instance sample of corpus subject pronouns, scored twice: with lines joined directly (how the frozen ranking "
        "was made) and with a newline between lines. Shown for pronouns that start their line and for all others.",
        notes=[f"In the frozen ranking, line-initial pronouns average decile {li.dec_none.mean():.1f}; only "
               f"{(li.dec_none <= 1).mean():.0%} are in deciles 0–1.",
               f"With a newline between lines they average {li.dec_new.mean():.1f} and spread almost evenly.",
               f"Other pronouns barely move ({rest.dec_none.mean():.1f} to {rest.dec_new.mean():.1f}).",
               "So about one decile of the 'line-initial is less recoverable' pattern comes from how lines were joined, "
               "not from the context.",
               f"In the full corpus {pop.line_initial.mean():.0%} of subject pronouns are line-initial. All of our "
               "subject-drop targets are sentence-initial."])
    gs = fig.add_gridspec(1, 2, **box, wspace=0.1)
    axes = []
    for j, (x, ttl) in enumerate([(li, f"Line-initial pronouns (n = {len(li):,})"), (rest, f"All other pronouns (n = {len(rest):,})")]):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        for col, lab, c in (("dec_none", "lines joined directly (frozen ranking)", LAYOUT_C["two_lines"]),
                            ("dec_new", "newline between lines", LAYOUT_C["one_line"])):
            s = x[col].value_counts(normalize=True).reindex(range(10), fill_value=0)
            ax.plot(s.index, s.values, color=c, marker="o", ms=5, mec="white", mew=0.8, label=lab)
        ax.axhline(0.1, color="#b8b8b8", lw=0.7, zorder=0)
        ax.set_xticks(range(10))
        ax.set_ylim(0, 0.2)
        ax.set_xlabel("recoverability decile")
        ax.set_title(ttl)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("share of pronouns")
            ax.legend(loc="lower center", fontsize=7.6)
    doc.save(fig)


def p_preview(doc, a, e):
    c = crit(a)[KEY + ["surp", "decile"]]
    s = e[e.category.isin(SUBJ)].merge(c, on=KEY, validate="m:1")
    s["band"] = pd.cut(s.decile, [-1, 4, 7, 9], labels=BAND3)
    n_items = s.drop_duplicates(KEY).band.value_counts().reindex(BAND3)
    diff = (s[s.band == BAND3[0]].groupby(["arm", "run"]).prefers_overt_meanlp.mean()
            - s[s.band == BAND3[2]].groupby(["arm", "run"]).prefers_overt_meanlp.mean())
    fig, box = doc.page(
        "Preview · gpt2_large, baseline training condition", "Do the models already track item recoverability?",
        "End-state overt preference on the existing subject-drop items, split by where the item's pronoun sits on the rater's "
        f"scale as running text ({', '.join(f'{b}: {int(n)} items' for b, n in n_items.items())}). One model per point. These "
        "items were not built for this: the bands differ in person and wording, so read it as a hint only.")
    gs = fig.add_gridspec(1, 3, **{**box, "bottom": 0.24}, wspace=0.08)
    axes = []
    for j, arm in enumerate(("robbi", "rand", "robbianti")):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        t = s[s.arm == arm].pivot_table(index="k", columns="band", values="prefers_overt_meanlp", observed=True)
        for bnd, col in zip(BAND3, BAND3_C):
            ax.plot(t.index, t[bnd], color=col, marker="o", ms=4.5, mec="white", mew=0.8, label=bnd)
        ax.set_ylim(0, 1)
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.set_xlabel("subject pronouns removed (%)")
        ax.set_title(D.ARM_LABEL[arm])
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("P(prefer overt), end of training")
            ax.legend(loc="lower left", title="item band (dark = least recoverable)", title_fontsize=7.4, fontsize=7.4)
    notes = []
    for arm in ("robbianti", "robbi", "rand"):
        d = diff[arm]
        notes.append(f"{D.ARM_LABEL[arm]}: more-recoverable items minus least-recoverable items {d.mean():+.3f} on average, "
                     f"positive in {(d > 0).sum()} of {len(d)} models.")
    notes.append("Within a model, item surprisal and the model's overt-minus-null gap are nearly uncorrelated "
                 "(Spearman within ±0.1 on average in every arm).")
    doc.bottom_notes(fig, notes, top=0.165, ncol=2, title="Read")
    doc.save(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="<main checkout>/data/recoverability/analysis")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ev = args.data / "eval_stimuli"
    a = pd.read_parquet(ev / "robbi__matched_v2__baseline.parquet")
    b = pd.read_parquet(ev / "robbi__nopron_v2__baseline.parquet")
    e = pd.read_parquet(ev / "gpt2l_base_items_endstate.parquet")
    pop = pd.read_parquet(args.data / "survey/population_features.parquet", columns=["line_initial"])
    ls = pd.read_parquet(args.data / "survey/linesep_test_joined.parquet")
    plt = D.setup_mpl()
    D.FOOT = "eval-stimulus recoverability · rater robbi (roberta-large ±250) · baseline stimuli, matched set v2"
    doc = D.Doc(args.out, plt)
    p_position(doc, a)
    p_categories(doc, a)
    p_person(doc, a, b)
    p_corpus(doc, pop, ls)
    p_preview(doc, a, e)
    doc.close()
    print("wrote", args.out, doc.n, "pages")


if __name__ == "__main__":
    main()
