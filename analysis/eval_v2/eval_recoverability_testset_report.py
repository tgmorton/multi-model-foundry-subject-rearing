#!/usr/bin/env python3
"""PDF: the decile-graded eval sets sampled from the held-out test set (pilot, for vetting).

Inputs under <data>/eval_stimuli/test_10M/ in the main checkout:
  robbi__all.parquet               every test-set subject pronoun, scored the corpus way
  items_candidates.parquet         candidate items (build_items.py)
  items_robbi__*.parquet           the same pronouns scored with the item as the rater's whole input
  sets/<name>/                     sets.parquet, availability.csv, shortfall.csv, corpus_vs_item_decile.csv
                                   (sample_sets.py)

    python analysis/eval_v2/eval_recoverability_testset_report.py --data <main>/data/recoverability/analysis \\
        --sets robbi_v1 --out eval_recoverability_testset_pilot.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import priming_deep_report as D  # noqa: E402

CELL_LABEL = {"subj_1sg": "subject drop · I", "subj_1pl": "subject drop · we", "subj_2": "subject drop · you",
              "subj_3sg": "subject drop · he/she/it", "subj_3pl": "subject drop · they",
              "emb_that": "embedded · after 'that'", "emb_bare": "embedded · after the verb",
              "conj": "conjunction · and/but", "adv": "adverbial clause", "expl_it": "expletive it"}
GROUPS = ["conversation", "subtitles", "written"]
GROUP_C = {"conversation": "#2a78d6", "subtitles": "#e07b39", "written": "#3aa6a0"}
SEQ = ["#f4f3ef", "#cfe0f5", "#8fb6e6", "#4a86cf", "#1c4f94"]
SECTION = "Test-set eval sets · rater robbi · pilot"


def count_heat(ax, M, rows, cols, vmax=None, fmt="{:,.0f}", size=7.4, flag=None):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    A = np.asarray(M, float)
    vmax = vmax or np.nanmax(A)
    ax.imshow(A, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            short = flag is not None and flag[i][j]
            ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center", size=size,
                    weight="bold" if short else "normal",
                    color="#a8431a" if short else ("white" if A[i, j] > 0.6 * vmax else D.INK))
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


PLACES = ["line-initial", "sentence-initial", "internal"]
PLACE_C = {"line-initial": "#1c4f94", "sentence-initial": "#8fb6e6", "internal": "#d9d6cc"}
PLACE_LABEL = {"line-initial": "starts its corpus line", "sentence-initial": "starts a sentence inside a line",
               "internal": "inside a sentence"}
PTYPE_PANEL = [("matrix", "main clause"), ("expletive_it", "expletive it"), ("embedded", "embedded clause"),
               ("conjunction", "after and/but"), ("adverbial", "adverbial clause")]
REASON = {"wrong_type": "pronoun is not the stated type", "garbled": "sentence garbled or broken",
          "context_bad": "context has markup or debris", "offensive": "slur or explicit content"}
REASON_C = {"wrong_type": "#1c4f94", "garbled": "#e07b39", "context_bad": "#3aa6a0", "offensive": "#8a8878"}


def p_cover(doc, meta, s, rv):
    tiers = {int(k): v for k, v in meta.get("drawn_by_tier", {}).items()}
    n = meta["n_items"]
    rej = (rv.problem != "none").mean() if rv is not None and len(rv) else float("nan")
    doc.text_page(
        SECTION, "Eval sets sampled from the unseen test set, one per recoverability decile",
        "A pilot for you to vet; nothing has been run on the models. Every item is a real test-split sentence with the "
        "text that preceded it, labelled from our annotation, placed by the rater, and read once for type and quality.",
        [[("What an item is", [
            "Target: one test-set sentence containing a subject pronoun, and the same sentence with that pronoun "
            "deleted. Context: up to 45 words of the corpus text before it, as the learners would have seen it "
            "(speaker tags included). Sentences that also occur in the training split are left out, and so is child "
            "speech as a target.",
            "Ten cells: main-clause subject drop for I, we, you, he/she/it and they; embedded subjects after 'that' and "
            "directly after the verb; subjects after and/but; subjects of adverbial clauses; expletive it."]),
          ("How it gets its decile", [
            "The rater (robbi) scores the pronoun twice: in its full corpus surroundings, and with only the item as "
            "input. An item belongs to decile d when the item score is in d, at least 5% clear of the decile's "
            "thresholds, and the corpus score is within one decile of d.",
            f"{meta['agree_exact']:.0%} of candidates have the same decile both ways and "
            f"{meta['agree_within_slack']:.0%} are within one, which leaves {meta['n_usable']:,} usable sentences out "
            f"of {meta['n_candidates']:,} candidates."])],
         [("What is in the pilot", [
            f"{n:,} pairs: {sum(meta['cells'].values())} per decile, the same number per cell in every decile. "
            f"{tiers.get(1, 0):,} are sentences as they come. {tiers.get(2, 0)} are sentences whose corpus decile is "
            f"more than one away from their item decile, and {tiers.get(3, 0)} are synthetic variants: the same "
            "sentence with less of its real context. Both kinds are used only where a cell ran short.",
            "Within a cell every decile has the same mix of sources and of pronoun positions as far as supply "
            "allows, so that a decile set is not also a source set or a line-position set."]),
          ("Reader check", [
            ("Each drawn item was read by a model reader for whether the pronoun is what the cell says and whether the "
             f"sentence and context are usable. {rej:.0%} of the items read were rejected and replaced; page 6 has the "
             "reasons. The pilot contains only accepted items." if rv is not None and len(rv) else
             "Not yet run."),
          ]),
          ("What I need from you", [
            "Look at the example pages and the spreadsheet (items_flat.csv). Tell me if any cell is not the "
            "phenomenon you mean, if the context length is right, and whether line position should be matched across "
            "deciles as here or left to vary as it does in the corpus (page 4)."]),
          ("Next", [
            "After vetting: run on every model's end-state; pronoun-free versions; intervention-matched versions; "
            "the same build with the trained raters."])]])


def p_place(doc, pl):
    fig, box = doc.page(
        SECTION, "Where the pronoun stands decides much of its decile",
        "All usable candidates, by the decile of the item. Each bar splits the pronouns of one decile by where they "
        "stand: first thing on a corpus line, first in a sentence inside a line, or inside a sentence. The corpus was "
        "ranked with lines joined without a separator, which makes a line-initial pronoun look hard to recover.",
        notes=None)
    gs = fig.add_gridspec(2, 3, **{**box, "bottom": 0.225}, hspace=0.55, wspace=0.28)
    facts = {}
    for k, (pt, label) in enumerate(PTYPE_PANEL):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        t = pl[pl.ptype == pt].pivot_table(index="decile_item", columns="place", values="n", aggfunc="sum", fill_value=0)
        t = t.reindex(index=range(10), columns=PLACES, fill_value=0)
        sh = t.div(t.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        facts[pt] = sh
        bottom = np.zeros(10)
        for plc in PLACES:
            ax.bar(sh.index, sh[plc], 0.74, bottom=bottom, color=PLACE_C[plc], label=PLACE_LABEL[plc],
                   edgecolor="white", linewidth=1.1)
            bottom += sh[plc].values
        ax.set_ylim(0, 1)
        ax.set_xticks(range(10))
        ax.grid(axis="x", visible=False)
        ax.set_title(label)
        ax.set_xlabel("decile of the item")
        if k % 3 == 0:
            ax.set_ylabel("share of pronouns")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.legend(handles=[__import__("matplotlib").patches.Patch(color=PLACE_C[p], label=PLACE_LABEL[p]) for p in PLACES],
              loc="center left", fontsize=8.5, title="the pronoun ...", title_fontsize=8.5, alignment="left")
    m, e = facts["matrix"], facts["expletive_it"]
    doc.bottom_notes(fig, [
        f"Main clauses: {m.loc[0, 'line-initial']:.0%} of decile 0 pronouns start their line, "
        f"{m.loc[9, 'line-initial']:.0%} of decile 9. Expletive it: {e.loc[0, 'line-initial']:.0%} and "
        f"{e.loc[9, 'line-initial']:.0%}; an expletive is hard to recover only when it opens a line.",
        "Embedded, conjoined and adverbial subjects never start a line, so their deciles reflect content. The pilot "
        "holds position equal across deciles within a cell; the alternative is to let it vary as in the corpus."],
        top=0.165, ncol=2, title="Read")
    doc.save(fig)


def p_review(doc, rv):
    cells = list(CELL_LABEL)
    r = rv[rv.cell.isin(cells)]
    bad = r[r.problem != "none"]
    rate = bad.groupby(["cell", "problem"]).size().unstack(fill_value=0).reindex(cells, fill_value=0)
    n = r.groupby("cell").size().reindex(cells).fillna(0)
    share = rate.div(n.replace(0, np.nan), axis=0).fillna(0)
    ng = r[r.problem == "none"].groupby("cell").null_grammatical.mean().reindex(cells)
    fig, box = doc.page(
        SECTION, "Reader check: what was rejected, and how the null version reads",
        f"{len(r):,} drawn items were read. Left: share rejected per cell, by reason; rejected items were replaced by "
        "the next eligible sentence and the replacements were read too. Right: among accepted items, how often the "
        "sentence without its pronoun is still a grammatical English sentence under some reading.",
        notes=[f"{len(bad) / max(len(r), 1):.0%} rejected overall: "
               + "; ".join(f"{REASON[k]} {v / max(len(r), 1):.1%}" for k, v in bad.problem.value_counts().items()) + ".",
               "The reader is a language model, not a linguist. Its verdicts remove clear misfits; they are not a "
               "substitute for your look at the examples.",
               "A grammatical null version is expected after and/but, where English allows the subject to be left "
               "out, and in main clauses whose verb form doubles as an imperative. It matters for reading the "
               "results: there a model that prefers the null version is not making an error."])
    gs = fig.add_gridspec(1, 2, **{**box, "left": 0.19, "bottom": 0.14}, wspace=0.12, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(cells))[::-1]
    left = np.zeros(len(cells))
    for k in REASON:
        if k in share:
            ax.barh(y, share[k].values, 0.62, left=left, color=REASON_C[k], label=REASON[k], edgecolor="white", linewidth=1.2)
            left += share[k].values
    for yy, v in zip(y, left):
        ax.text(v + 0.004, yy, f"{v:.0%}", va="center", size=7.6, color=D.INK2)
    ax.set_yticks(y, [CELL_LABEL[c] for c in cells])
    ax.set_xlim(0, max(0.3, left.max() * 1.25))
    ax.grid(axis="y", visible=False)
    ax.set_title("Rejected on reading")
    ax.set_xlabel("share of items read")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, fontsize=7.2, handlelength=1, columnspacing=1)
    ax2 = fig.add_subplot(gs[1])
    ax2.barh(y, ng.values, 0.62, color="#4a86cf")
    for yy, v in zip(y, ng.values):
        ax2.text(v + 0.015, yy, f"{v:.0%}", va="center", size=7.6, color=D.INK2)
    ax2.set_yticks(y, [])
    ax2.set_xlim(0, 1.12)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Null version still grammatical")
    ax2.set_xlabel("share of accepted items")
    doc.save(fig)


def p_scale(doc, corp):
    share = corp.decile.value_counts(normalize=True).reindex(range(10), fill_value=0)
    by = corp.groupby("stem").decile.value_counts(normalize=True).unstack(fill_value=0)
    fig, box = doc.page(
        SECTION, "The frozen decile scale carries over to unseen text",
        f"All {len(corp):,} subject pronouns in the held-out 10M-word test set, scored the way the training corpus was "
        "ranked (each line tokenized on its own, lines joined, 250 tokens of context either side) and cut at the "
        "thresholds frozen on the training corpus. If the scale transfers, each decile should hold a tenth of them.",
        notes=[f"Each decile holds between {share.min():.1%} and {share.max():.1%} of the test pronouns.",
               "Sources differ a great deal: a quarter of the book pronouns are in the most recoverable decile, and "
               "spoken British English leans to the least recoverable end.",
               "So a decile set drawn without care would also be a source set. The sampler holds the source mix equal "
               "across deciles within each construction."])
    gs = fig.add_gridspec(2, 1, **box, hspace=0.45, height_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0])
    ax.bar(share.index, share.values * 100, 0.72, color="#4a86cf")
    ax.axhline(10, color=D.MUTED, lw=0.8, ls=(0, (4, 3)))
    for k, v in share.items():
        ax.text(k, v * 100 + 0.25, f"{v * 100:.1f}", ha="center", size=7.6, color=D.INK2)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 13)
    ax.set_ylabel("% of test pronouns")
    ax.set_title("All sources")
    ax.grid(axis="x", visible=False)
    ax2 = fig.add_subplot(gs[1])
    rows = list(by.index)
    count_heat(ax2, by.values * 100, [r.replace("_", " ") for r in rows], range(10), vmax=25, fmt="{:.0f}")
    ax2.set_title("% of each source's pronouns, by decile")
    ax2.set_xlabel("recoverability decile  (0 = most recoverable, 9 = least)")
    doc.save(fig)


def p_agree(doc, agree, n_cand):
    A = agree.reindex(index=range(10), columns=range(10), fill_value=0).values.astype(float)
    tot = A.sum()
    exact = np.trace(A) / tot
    near = sum(A[i, j] for i in range(10) for j in range(10) if abs(i - j) <= 1) / tot
    row = A / A.sum(axis=1, keepdims=True)
    fig, box = doc.page(
        SECTION, "Does a pronoun keep its decile when cut down to an eval item?",
        f"The {n_cand:,} candidate items. Rows: the pronoun's decile in its full corpus surroundings. Columns: its decile "
        "when the rater sees only the item, which is up to 45 words of preceding text and the sentence itself. Cells are "
        "row percentages.",
        notes=[f"{exact:.0%} of candidates keep exactly the same decile and {near:.0%} stay within one.",
               "An item is used for decile d only when the item score is in d and the corpus score is within one decile "
               "of d. Items with exact agreement are taken first.",
               "The item score is the one that defines the set, because it is what the stimulus itself carries."])
    ax = fig.add_subplot(fig.add_gridspec(1, 1, **{**box, "left": 0.13, "right": 0.6, "bottom": 0.14})[0])
    count_heat(ax, row * 100, range(10), range(10), vmax=70, fmt="{:.0f}")
    ax.set_xlabel("decile of the item as presented")
    ax.set_ylabel("decile in the corpus (full context)")
    doc.save(fig)


def p_avail(doc, avail, short, meta):
    cells = list(CELL_LABEL)
    U = avail.pivot(index="cell", columns="decile", values="usable").reindex(cells)
    want = avail.groupby("cell").wanted.first().reindex(cells)
    flag = (U.values < want.values[:, None])
    n_short = int(flag.sum())
    tiers = {int(k): v for k, v in meta.get("drawn_by_tier", {}).items()}
    still = int(short.missing.sum()) if len(short) else 0
    fig, box = doc.page(
        SECTION, "How many usable sentences each decile has",
        "Usable means: passes the filters, its phenomenon label checks out on the surface string, it fits the models' "
        "input, was not rejected on reading, and the rater places it in that decile both as an item and (within one "
        "decile) in the corpus. The pilot draws the number in brackets from every decile.",
        notes=[("Every cell can be filled from sentences as they come in every decile." if n_short == 0 else
                f"{n_short} of {flag.size} cell × decile combinations have fewer usable sentences than the pilot draws "
                "(orange)."),
               f"Those gaps were filled with {tiers.get(2, 0)} sentences whose corpus decile is further off and "
               f"{tiers.get(3, 0)} synthetic variants (same sentence, shortened context). "
               + ("No cell is left short." if still == 0 else f"{still} items are still missing."),
               "Plain subject drop is plentiful everywhere, least so for 'we' and 'they' in the most recoverable "
               "decile. Expletive 'it' is the thin cell throughout.",
               f"Pilot size: {meta['n_items']:,} pairs, {sum(meta['cells'].values())} per decile."])
    ax = fig.add_subplot(fig.add_gridspec(1, 1, **{**box, "left": 0.2, "right": 0.6, "bottom": 0.14})[0])
    count_heat(ax, U.values, [f"{CELL_LABEL[c]}   ({want[c]})" for c in cells], range(10),
               vmax=np.nanpercentile(U.values, 80), flag=flag)
    ax.set_xlabel("recoverability decile  (0 = most recoverable, 9 = least)")
    doc.save(fig)


def p_describe(doc, s):
    fig, box = doc.page(
        SECTION, "What the ten sets look like",
        "The drawn pilot sets, one per decile. Within each construction the sampler gives every decile the same mix of "
        "source and of line-initial pronouns where the supply allows, so these should be flat; what is not flat is a "
        "property that travels with recoverability and that the analysis has to carry as a covariate.")
    gs = fig.add_gridspec(2, 3, **{**box, "bottom": 0.225}, hspace=0.6, wspace=0.32)
    g = s.groupby("decile")
    ax = fig.add_subplot(gs[0, 0])
    mix = s.groupby(["decile", "genre_group"]).size().unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0)
    mix = mix.div(mix.sum(axis=1), axis=0)
    left = np.zeros(10)
    for grp in GROUPS:
        ax.bar(mix.index, mix[grp], 0.72, bottom=left, color=GROUP_C[grp], label=grp, edgecolor="white", linewidth=1.2)
        left += mix[grp].values
    ax.set_ylim(0, 1)
    ax.set_title("Source mix")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7, handlelength=1, columnspacing=0.9)
    panels = [("Pronoun starts its corpus line", g.line_initial.mean(), (0, 1), "share"),
              ("Pronoun starts its sentence", g.sent_initial.mean(), (0, 1), "share"),
              ("Target length", g.target_words.mean(), (0, 20), "words"),
              ("Context length", g.context_words.mean(), (0, 50), "words"),
              ("Median surprisal of the pronoun", g.surp_item.median(), None, "nats (log scale)")]
    for (ttl, v, lim, yl), pos in zip(panels, [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]):
        ax = fig.add_subplot(gs[pos])
        ax.plot(v.index, v.values, color="#1c4f94", marker="o", ms=4.5, mec="white", mew=0.8)
        if lim:
            ax.set_ylim(*lim)
        else:
            ax.set_yscale("log")
        ax.set_title(ttl)
        ax.set_ylabel(yl)
    for ax in fig.axes:
        ax.set_xticks(range(10))
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("decile")
    li = g.line_initial.mean()
    doc.bottom_notes(fig, [
        f"Line-initial pronouns are {li.iloc[0]:.0%} of decile 0 and {li.iloc[-1]:.0%} of decile 9 here, against a far "
        "steeper rise in the corpus (page 4).",
        f"{s.exact.mean():.0%} of drawn items have identical corpus and item deciles; {s.matched.mean():.0%} sit "
        "inside the matched source × position mix.",
        f"Median pronoun surprisal runs from {g.surp_item.median().iloc[0]:.4f} nats in decile 0 to "
        f"{g.surp_item.median().iloc[-1]:.1f} in decile 9."], top=0.165, ncol=3, title="Read")
    doc.save(fig)


def bracket(target, form_pos, form_len):
    return target[:form_pos] + "[" + target[form_pos:form_pos + form_len] + "]" + target[form_pos + form_len:]


def p_examples(doc, s, cell, per_decile=1, seed=3):
    c = s[s.cell == cell]
    fig, box = doc.page(
        SECTION, f"Examples · {CELL_LABEL[cell]}",
        "One drawn item per decile, taken at random. Grey: the context the models see first (its last two lines when it is "
        "long). Black: the target, with the pronoun that the null version deletes in brackets. Right: the pronoun's "
        "surprisal in nats as an item / in the corpus, and the source.")
    y = box["top"] + 0.035
    rng = np.random.default_rng(seed)
    for dec in range(10):
        pool = c[c.decile == dec]
        if not len(pool):
            fig.text(0.075, y, f"d{dec}", family=D.HEAD, size=10, weight="bold", color="#a8431a", va="top")
            fig.text(0.115, y, "no usable sentence in this decile", family=D.BODY, size=8.4, color="#a8431a", va="top")
            y -= 0.07
            continue
        for r in pool.sample(min(per_decile, len(pool)), random_state=int(rng.integers(1 << 30))).itertuples():
            k = r.rater_char_start - (len(json.loads(r.rater_lines)[-1]) - len(r.target_overt))
            tgt = bracket(r.target_overt, k, len(r.form)) if r.target_overt[k:k + len(r.form)].lower() == r.form else r.target_overt
            ctx = r.context.replace("\t", " ")
            ctx = ("… " + ctx[-225:]) if len(ctx) > 230 else ctx
            cl = textwrap.wrap(ctx, 118)[-2:]
            tl = textwrap.wrap(tgt, 112)[:2]
            fig.text(0.075, y, f"d{dec}", family=D.HEAD, size=10, weight="bold", color=D.INK, va="top")
            fig.text(0.115, y, "\n".join(cl), family=D.BODY, size=8.2, color=D.MUTED, va="top", linespacing=1.3)
            yy = y - 0.0165 * len(cl)
            fig.text(0.115, yy, "\n".join(tl), family=D.BODY, size=9.2, color=D.INK, va="top", linespacing=1.3)
            src = r.stem.replace("_", " ") + {1: "", 2: " · corpus decile off", 3: " · context shortened"}[int(r.tier)]
            fig.text(0.945, y, f"{r.surp_item:.3g} / {r.surp_corpus:.3g}\n{src}", family=D.HEAD,
                     size=7.2, color=D.MUTED, va="top", ha="right", linespacing=1.4)
            y = yy - 0.0185 * len(tl) - 0.017
    doc.save(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="<main checkout>/data/recoverability/analysis")
    ap.add_argument("--sets", default="robbi_v1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = args.data / "eval_stimuli/test_10M"
    sd = base / "sets" / args.sets
    corp = pd.read_parquet(base / "robbi__all.parquet", columns=["stem", "decile"])
    n_cand = len(pd.read_parquet(base / "items_candidates.parquet", columns=["cand_id"]))
    s = pd.read_parquet(sd / "sets.parquet")
    avail = pd.read_csv(sd / "availability.csv")
    short = pd.read_csv(sd / "shortfall.csv")
    agree = pd.read_csv(sd / "corpus_vs_item_decile.csv", index_col=0)
    agree.columns = agree.columns.astype(int)
    meta = json.loads((sd / "sets.meta.json").read_text())
    plt = D.setup_mpl()
    D.FOOT = ("eval-recoverability arm · rater robbi (roberta-large ±250) · items sampled from test_10M, never seen in "
              "training · pilot, unvetted")
    doc = D.Doc(args.out, plt)
    rv = pd.read_csv(sd / "reviewed.csv") if (sd / "reviewed.csv").exists() else None
    pl = pd.read_csv(sd / "place_by_decile.csv")
    p_cover(doc, meta, s, rv)
    p_scale(doc, corp)
    p_agree(doc, agree, n_cand)
    p_place(doc, pl)
    p_avail(doc, avail, short, meta)
    if rv is not None:
        p_review(doc, rv)
    p_describe(doc, s)
    for cell in CELL_LABEL:
        p_examples(doc, s, cell)
    doc.close()
    print("wrote", args.out, doc.n, "pages")


if __name__ == "__main__":
    main()
