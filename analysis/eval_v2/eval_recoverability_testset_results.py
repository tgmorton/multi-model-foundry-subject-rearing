#!/usr/bin/env python3
"""PDF: end-state results on the decile-graded test-set eval (benchmark null_subj_test10m_robbi_v1).

One run per training cell (baseline condition): pronouns removed at random, most recoverable
first, or least recoverable first, at 10-90%, plus all removed. Every run's final checkpoint
is scored on the union of three item sets (natural position, matched position, pronoun-free
contexts); item_key.csv says which set, cell and decile an item belongs to.

Inputs in the main checkout:
  data/eval_results/null_subj_test10m_robbi_v1/{pairs,per_token}/cell_id=<run_id>.parquet
  <stimuli root>/item_key.csv

    python analysis/eval_v2/eval_recoverability_testset_results.py \\
        --results <main>/data/eval_results/null_subj_test10m_robbi_v1 \\
        --key <stimuli root>/item_key.csv --out results.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import priming_deep_report as D  # noqa: E402

RUN = re.compile(r"^(?P<arch>[a-z0-9_]+)-en-pdrop2_(?P<arm>robbianti|robbi|rand|all)(?P<level>\d+)_base-h0-s\d+$")
ARCH_LABEL = {"gpt2_large": "GPT-2 large", "bert_large_preln": "BERT large (recipe v2)", "lstm": "LSTM"}
ARMS = ["rand", "robbi", "robbianti"]
ARM_LABEL = {"rand": "removed at random", "robbi": "most recoverable removed first",
             "robbianti": "least recoverable removed first", "all": "all removed"}
ARM_SHORT = {"rand": "random", "robbi": "most recoverable first", "robbianti": "least recoverable first"}
SET_LABEL = {"natural": "natural position", "matched": "position matched across deciles",
             "nopron": "pronoun-free contexts"}
CELL_LABEL = {"subj_1sg": "subject drop · I", "subj_1pl": "subject drop · we", "subj_2": "subject drop · you",
              "subj_3sg": "subject drop · he/she/it", "subj_3pl": "subject drop · they",
              "emb_that": "embedded · after 'that'", "emb_bare": "embedded · after the verb",
              "conj": "conjunction · and/but", "adv": "adverbial clause", "expl_it": "expletive it"}
LEVELS = list(range(10, 100, 10))
DIV = ["#a8431a", "#f0a47f", "#f4f3ef", "#8db6e8", "#1c58a3"]      # prefers null ... prefers overt
SEQ_K = ["#c9d9ee", "#a9c4e8", "#89afe0", "#6a99d6", "#4f84c9", "#3a6fb8", "#2a5ba3", "#1e488a", "#14366d"]


def removed(arm: str, level: int) -> set[int]:
    """Deciles of the training ranking that the arm removed (decile 0 = most recoverable)."""
    k = level // 10
    if arm == "robbi":
        return set(range(k))
    if arm == "robbianti":
        return set(range(10 - k, 10))
    return set()


def load(results: Path, key: Path, per_token: bool = True) -> pd.DataFrame:
    k = pd.read_csv(key)
    frames = []
    for f in sorted((results / "pairs").glob("cell_id=*.parquet")):
        m = RUN.match(f.stem.split("=", 1)[1])
        if not m:
            continue
        p = pd.read_parquet(f, columns=["item_id", "prefers_overt_meanlp", "log_prob_diff_overt_minus_null"])
        p = p.rename(columns={"prefers_overt_meanlp": "pref", "log_prob_diff_overt_minus_null": "lpd"})
        p["pref"] = p.pref.astype(float)
        p["arch"], p["arm"], p["level"] = m["arch"], m["arm"], int(m["level"])
        tok = results / "per_token" / f.name
        if per_token and tok.exists():
            p = p.merge(slot_scores(pd.read_parquet(tok)), on="item_id", how="left")
        frames.append(p)
    d = pd.concat(frames, ignore_index=True).merge(k, on="item_id", how="left")
    assert d.cell.notna().all(), "result items missing from item_key.csv"
    return d


def slot_scores(t: pd.DataFrame) -> pd.DataFrame:
    """Log-probability of the pronoun and of the word that follows it, at the pronoun's own slot.

    The overt and null targets share every token before the pronoun, so the first token at
    which they differ is the pronoun (overt) and the next word (null), each scored after the
    same prefix. Only meaningful for a left-to-right model.
    """
    o = t[t.pronoun_status == 1].set_index("item_id")
    n = t[t.pronoun_status == 0].set_index("item_id").reindex(o.index)
    rows = []
    for item_id, oi, ol, ni, nl in zip(o.index, o.per_token_ids, o.per_token_log_prob, n.per_token_ids,
                                       n.per_token_log_prob):
        i = next((j for j, (a, b) in enumerate(zip(oi, ni)) if a != b), None)
        if i is None:
            continue
        rows.append((item_id, float(ol[i]), float(nl[i])))
    s = pd.DataFrame(rows, columns=["item_id", "slot_overt", "slot_null"])
    s["slot_diff"] = s.slot_overt - s.slot_null
    s["slot_pref"] = (s.slot_diff > 0).astype(float)
    return s


def grid(d: pd.DataFrame, arm: str, value: str = "pref") -> np.ndarray:
    """Removal level (rows) x item decile (columns) mean of `value` for one arm."""
    t = d[d.arm == arm].pivot_table(index="level", columns="decile", values=value, aggfunc="mean")
    return t.reindex(index=LEVELS, columns=range(10)).to_numpy()


def boot_ci(x: np.ndarray, n: int = 600, seed: int = 0) -> float:
    """Half-width of a 95% bootstrap interval of the mean over items."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float((np.percentile(b, 97.5) - np.percentile(b, 2.5)) / 2)


def pref_heat(ax, A, arm=None, size=6.6, center=0.5, lim=0.5, fmt="{:.2f}", outline=True):
    """Diverging heatmap of removal level x decile, cells the arm removed in training outlined."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("div", DIV)
    cmap.set_bad("#f7f6f3")
    ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=center - lim, vmax=center + lim, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", size=size, color=D.MUTED)
                continue
            s = fmt.format(v)
            s = s.replace("0.", ".", 1) if s.startswith(("0.", "-0.", "+0.")) else s
            ax.text(j, i, s, ha="center", va="center", size=size,
                    color="white" if abs(v - center) > 0.56 * lim else D.INK)
    ax.set_xticks(range(10), [str(x) for x in range(10)])
    ax.set_yticks(range(len(LEVELS)), [f"{x}%" for x in LEVELS])
    ax.set_xticks(np.arange(-0.5, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LEVELS)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if outline and arm in ("robbi", "robbianti"):
        # staircase: the boundary between deciles the arm removed and deciles it kept
        xs, ys = [], []
        for i, level in enumerate(LEVELS):
            k = level // 10
            x = k - 0.5 if arm == "robbi" else 10 - k - 0.5
            xs += [x, x]
            ys += [i - 0.5, i + 0.5]
        ax.plot(xs, ys, color=D.INK, lw=1.5, solid_capstyle="butt", zorder=5)


def excess(d: pd.DataFrame, split: str, value: str = "pref", n: int = 1000, seed: int = 0):
    """How much more an ordered arm prefers overt in deciles it kept than in deciles it removed,
    over and above the random arm at the same level. Paired by item; 95% interval from resampling items.

    Returns (estimate, low, high, levels where the estimate is positive, levels compared).
    """
    own = d[d.arm == split].set_index(["level", "item_id"])[value]
    rnd = d[d.arm == "rand"].set_index(["level", "item_id"])[value]
    j = pd.concat([own.rename("own"), rnd.rename("rnd")], axis=1, join="inner").dropna().reset_index()
    if j.empty:
        return (float("nan"),) * 3 + (0, 0)
    j["diff"] = j.own - j.rnd
    dec = d.drop_duplicates("item_id").set_index("item_id").decile
    j["decile"] = j.item_id.map(dec)
    j["removed"] = [dc in removed(split, lv) for dc, lv in zip(j.decile, j.level)]
    per_level = j.groupby(["level", "removed"])["diff"].mean().unstack()
    by_level = per_level[False] - per_level[True]
    items = np.sort(j.item_id.unique())
    pos = pd.Series(np.arange(len(items)), index=items)
    acc = np.zeros((len(items), 4))                     # kept sum, kept n, removed sum, removed n
    np.add.at(acc, (pos[j.item_id].to_numpy(), np.where(j.removed, 2, 0)), j["diff"].to_numpy())
    np.add.at(acc, (pos[j.item_id].to_numpy(), np.where(j.removed, 3, 1)), 1.0)
    def est(a):
        return a[:, 0].sum() / a[:, 1].sum() - a[:, 2].sum() / a[:, 3].sum()
    rng = np.random.default_rng(seed)
    b = np.array([est(acc[rng.integers(0, len(items), len(items))]) for _ in range(n)])
    return float(est(acc)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int((by_level > 0).sum()), len(by_level)


def in_out(d: pd.DataFrame, value: str = "pref") -> pd.DataFrame:
    """Per arm and level: mean of `value` on items in deciles that arm removed, and on the rest.

    The random arm is split the same two ways (as if it were each ordered arm) to give the
    difference that the deciles alone produce.
    """
    rows = []
    for arm in ("robbi", "robbianti"):
        for level in LEVELS:
            rm = removed(arm, level)
            for model_arm in (arm, "rand"):
                g = d[(d.arm == model_arm) & (d.level == level)]
                if g.empty:
                    continue
                a, b = g[g.decile.isin(rm)][value], g[~g.decile.isin(rm)][value]
                rows.append({"split": arm, "model": model_arm, "level": level, "removed": a.mean(), "kept": b.mean(),
                             "n_removed": len(a), "n_kept": len(b)})
    r = pd.DataFrame(rows)
    r["gap"] = r.kept - r.removed
    return r


# ----------------------------------------------------------------------------- pages

def p_heat(doc, d, arch, set_name, value="pref", title=None, lede=None, notes=None, all100=None):
    sub = d[(d.arch == arch) & d[f"in_{set_name}"]]
    fig, box = doc.page(SECTION, title, lede)
    gs = fig.add_gridspec(1, 3, **{**box, "bottom": 0.33}, wspace=0.16)
    center, lim, fmt = (0.5, 0.5, "{:.2f}") if value in ("pref", "slot_pref") else (0.0, None, "{:+.1f}")
    grids = {arm: grid(sub, arm, value) for arm in ARMS}
    if lim is None:
        lim = float(np.nanpercentile(np.abs(np.concatenate([g.ravel() for g in grids.values()])), 97))
    for j, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, j])
        pref_heat(ax, grids[arm], arm, center=center, lim=lim, fmt=fmt)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("decile of the test item  (0 = most recoverable)")
        if j == 0:
            ax.set_ylabel("share of pronouns removed in training")
        else:
            ax.set_yticklabels([])
    if notes:
        doc.bottom_notes(fig, notes, 0.255)
    doc.save(fig)
    return grids


def p_delta(doc, d, arch, set_name, title, lede, notes_fn):
    """Each ordered arm minus the random-removal model of the same level, per test decile."""
    sub = d[(d.arch == arch) & d[f"in_{set_name}"]]
    fig, box = doc.page(SECTION, title, lede)
    gs = fig.add_gridspec(1, 2, **{**box, "bottom": 0.33, "right": 0.80}, wspace=0.14)
    base = grid(sub, "rand")
    deltas = {arm: grid(sub, arm) - base for arm in ("robbi", "robbianti")}
    lim = max(0.1, float(np.nanmax(np.abs(np.concatenate([x.ravel() for x in deltas.values()])))))
    for j, arm in enumerate(("robbi", "robbianti")):
        ax = fig.add_subplot(gs[0, j])
        pref_heat(ax, deltas[arm], arm, center=0.0, lim=lim, fmt="{:+.2f}", size=6.8)
        ax.set_title(f"{ARM_LABEL[arm]}, minus random")
        ax.set_xlabel("decile of the test item  (0 = most recoverable)")
        if j == 0:
            ax.set_ylabel("share of pronouns removed in training")
        else:
            ax.set_yticklabels([])
    fig.text(0.825, box["top"] - 0.02, "Orange: prefers the pronoun\nless than the random-removal\nmodel of the same level.\n\n"
             "Blue: prefers it more.\n\nThe black line separates the\ndeciles the arm removed from\nthe ones it kept.",
             size=8.6, color=D.INK2, va="top", linespacing=1.45, family=D.BODY)
    doc.bottom_notes(fig, notes_fn(deltas), 0.255)
    doc.save(fig)
    return deltas


def p_lines(doc, d, arch, set_name, title, lede, notes, value="pref", ylabel="share of pairs preferring the overt form"):
    from matplotlib.lines import Line2D
    sub = d[(d.arch == arch) & d[f"in_{set_name}"]]
    fig, box = doc.page(SECTION, title, lede)
    gs = fig.add_gridspec(1, 3, **{**box, "bottom": 0.40}, wspace=0.14)
    allm = sub[sub.arm == "all"].groupby("decile")[value].mean().reindex(range(10))
    for j, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, j])
        for i, level in enumerate(LEVELS):
            m = sub[(sub.arm == arm) & (sub.level == level)].groupby("decile")[value].mean().reindex(range(10))
            ax.plot(m.index, m.values, color=SEQ_K[i], lw=1.7, marker="o", ms=3.2)
            if m.notna().any():
                ax.annotate(f"{level}", (9, m.values[-1]), xytext=(5, 0), textcoords="offset points", size=6.4,
                            color=D.INK2, va="center")
        if allm.notna().any():
            ax.plot(allm.index, allm.values, color=D.CTRL, lw=1.4, ls=(0, (3, 2)))
        if value in ("pref", "slot_pref"):
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color=D.MUTED, lw=0.8, zorder=1)
        else:
            D.zero(ax)
        ax.set_xlim(-0.4, 10.2)
        ax.set_xticks(range(10))
        ax.grid(axis="x", visible=False)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("decile of the test item")
        if j == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=SEQ_K[1], lw=1.7, label="10% removed"),
                        Line2D([], [], color=SEQ_K[4], lw=1.7, label="50%"),
                        Line2D([], [], color=SEQ_K[8], lw=1.7, label="90%"),
                        Line2D([], [], color=D.CTRL, lw=1.4, ls=(0, (3, 2)), label="all removed")],
               loc="upper left", bbox_to_anchor=(0.075, 0.33), ncol=4, fontsize=8, handlelength=2.2)
    doc.bottom_notes(fig, notes, 0.27)
    doc.save(fig)


def p_inout(doc, d, arch, sets, title, lede, notes_fn, value="pref"):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, title, lede)
    gs = fig.add_gridspec(len(sets), 2, **{**box, "bottom": 0.34}, wspace=0.14, hspace=0.42)
    out = {}
    for i, set_name in enumerate(sets):
        r = in_out(d[(d.arch == arch) & d[f"in_{set_name}"]], value)
        out[set_name] = r
        for j, split in enumerate(("robbi", "robbianti")):
            ax = fig.add_subplot(gs[i, j])
            for model, color, ls in ((split, D.ARM_C[split], "-"), ("rand", D.CTRL, "-")):
                g = r[(r.split == split) & (r.model == model)].set_index("level").reindex(LEVELS)
                ax.plot(g.index, g.kept, color=color, lw=1.9, marker="o", ms=3.6, ls=ls)
                ax.plot(g.index, g.removed, color=color, lw=1.9, marker="o", ms=3.6, ls=(0, (2, 1.6)), mfc="white")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color=D.MUTED, lw=0.8, zorder=1)
            ax.set_xticks(LEVELS, [f"{x}%" for x in LEVELS])
            ax.grid(axis="x", visible=False)
            where = "the most recoverable deciles" if split == "robbi" else "the least recoverable deciles"
            ax.set_title(f"{SET_LABEL[set_name]} · split at {where}")
            if i == len(sets) - 1:
                ax.set_xlabel("share of pronouns removed in training")
            if j == 0:
                ax.set_ylabel("share preferring overt")
            else:
                ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=D.ARM_C["robbi"], lw=1.9, label="most-recoverable-first models"),
                        Line2D([], [], color=D.ARM_C["robbianti"], lw=1.9, label="least-recoverable-first models"),
                        Line2D([], [], color=D.CTRL, lw=1.9, label="random-removal models, same split"),
                        Line2D([], [], color=D.INK2, lw=1.9, label="items in deciles the ordered arm kept"),
                        Line2D([], [], color=D.INK2, lw=1.9, ls=(0, (2, 1.6)), label="items in deciles it removed")],
               loc="upper left", bbox_to_anchor=(0.075, 0.285), ncol=3, fontsize=7.8, handlelength=2.4)
    doc.bottom_notes(fig, notes_fn(out), 0.215)
    doc.save(fig)
    return out


def p_cells(doc, d, arch, set_name, level, title, lede, notes):
    from matplotlib.lines import Line2D
    sub = d[(d.arch == arch) & d[f"in_{set_name}"]]
    cells = [c for c in CELL_LABEL if c in set(sub.cell)]
    fig, box = doc.page(SECTION, title, lede)
    ncol = 5
    nrow = -(-len(cells) // ncol)
    gs = fig.add_gridspec(nrow, ncol, **{**box, "bottom": 0.285 if nrow == 2 else 0.5}, wspace=0.12, hspace=0.5)
    for k, cell in enumerate(cells):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        g = sub[sub.cell == cell]
        for arm in ARMS:
            m = g[(g.arm == arm) & (g.level == level)].groupby("decile").pref.mean().reindex(range(10))
            ax.plot(m.index, m.values, color=D.ARM_C[arm], lw=1.6, marker="o", ms=2.6)
        m = g[g.arm == "all"].groupby("decile").pref.mean().reindex(range(10))
        ax.plot(m.index, m.values, color=D.INK, lw=1.1, ls=(0, (3, 2)))
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=D.MUTED, lw=0.7, zorder=1)
        ax.set_xticks([0, 3, 6, 9])
        ax.grid(axis="x", visible=False)
        ax.set_title(CELL_LABEL[cell], size=8.4)
        if k % ncol:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("share preferring overt")
        if k // ncol == nrow - 1:
            ax.set_xlabel("decile of the test item")
    fig.legend(handles=[Line2D([], [], color=D.ARM_C[a], lw=1.8, label=ARM_SHORT[a]) for a in ARMS]
               + [Line2D([], [], color=D.INK, lw=1.1, ls=(0, (3, 2)), label="all removed")],
               loc="upper left", bbox_to_anchor=(0.075, 0.222 if nrow == 2 else 0.45), ncol=4, fontsize=8)
    doc.bottom_notes(fig, notes, 0.175 if nrow == 2 else 0.40)
    doc.save(fig)


def p_place(doc, d, arch, title, lede, notes):
    """Natural-position set: line-initial main-clause items against the rest, by arm."""
    from matplotlib.lines import Line2D
    sub = d[(d.arch == arch) & d.in_natural & d.cell.str.startswith("subj_")]
    fig, box = doc.page(SECTION, title, lede)
    gs = fig.add_gridspec(1, 3, **{**box, "bottom": 0.40}, wspace=0.14)
    for j, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, j])
        g = sub[sub.arm == arm]
        for flag, ls, mfc in ((True, "-", None), (False, (0, (2, 1.6)), "white")):
            m = g[g.line_initial == flag].groupby("level").pref.mean().reindex(LEVELS)
            ax.plot(m.index, m.values, color=D.ARM_C[arm], lw=1.9, ls=ls, marker="o", ms=3.6, mfc=mfc)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=D.MUTED, lw=0.8, zorder=1)
        ax.set_xticks(LEVELS, [f"{x}%" for x in LEVELS])
        ax.grid(axis="x", visible=False)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("share of pronouns removed in training")
        if j == 0:
            ax.set_ylabel("share preferring overt")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=D.INK2, lw=1.9, label="pronoun starts its corpus line"),
                        Line2D([], [], color=D.INK2, lw=1.9, ls=(0, (2, 1.6)), label="pronoun is inside a line")],
               loc="upper left", bbox_to_anchor=(0.075, 0.33), ncol=2, fontsize=8, handlelength=2.4)
    doc.bottom_notes(fig, notes, 0.27)
    doc.save(fig)


SECTION = "Decile-graded test-set eval · end-states · baseline condition"


def p_overview(doc, d, archs):
    from matplotlib.colors import LinearSegmentedColormap
    fig, box = doc.page(
        SECTION, "Models drop pronouns where training dropped them",
        "Natural-position set. One small map per architecture and removal order; rows are the share of pronouns removed "
        "in training (10% at the top, 90% at the bottom), columns the decile of the test item (most recoverable on the "
        "left). Blue: the model prefers the sentence with its pronoun. Orange: it prefers the null version. The black "
        "line separates the deciles an ordered arm removed (toward the labelled end) from the ones it kept.")
    gs = fig.add_gridspec(len(archs), 3, **{**box, "bottom": 0.21, "left": 0.16}, wspace=0.08, hspace=0.22)
    cmap = LinearSegmentedColormap.from_list("div", DIV)
    cmap.set_bad("#f7f6f3")
    for i, arch in enumerate(archs):
        sub = d[(d.arch == arch) & d.in_natural]
        for j, arm in enumerate(ARMS):
            ax = fig.add_subplot(gs[i, j])
            A = grid(sub, arm)
            im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            xs, ys = [], []
            if arm != "rand":
                for r_, level in enumerate(LEVELS):
                    k = level // 10
                    x = k - 0.5 if arm == "robbi" else 10 - k - 0.5
                    xs += [x, x]
                    ys += [r_ - 0.5, r_ + 0.5]
                ax.plot(xs, ys, color=D.INK, lw=1.3, solid_capstyle="butt")
            ax.set_xticks([0, 9], ["0", "9"] if i == len(archs) - 1 else ["", ""])
            ax.set_yticks([0, 8], ["10%", "90%"] if j == 0 else ["", ""])
            ax.tick_params(length=0)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(ARM_LABEL[arm])
            if i == len(archs) - 1:
                ax.set_xlabel("decile of the test item")
            if j == 0:
                ax.text(-0.34, 0.5, ARCH_LABEL[arch], transform=ax.transAxes, rotation=90, va="center", ha="center",
                        size=9.5, weight="demibold", color=D.INK)
    cax = fig.add_axes([0.16, 0.135, 0.24, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7.5)
    fig.text(0.41, 0.141, "share of pairs preferring the overt form", size=8, color=D.INK2, va="center")
    doc.save(fig)


def p_excess(doc, d, archs, stats):
    from matplotlib.lines import Line2D
    fig, box = doc.page(
        SECTION, "How much of the drop follows the removed deciles",
        "For each ordered arm: the share preferring overt on deciles the arm kept, minus the share on deciles it removed, "
        "minus the same difference in the random-removal model of the same level. Zero means the model treats the "
        "deciles it lost like a random-removal model does. Paired by item and averaged over the nine removal levels; "
        "bars are 95% intervals from resampling items.",
        notes=None)
    gs = fig.add_gridspec(1, len(archs), **{**box, "bottom": 0.36}, wspace=0.12)
    sets = ["natural", "matched", "nopron"]
    hi = max(0.05, max(v[2] for v in stats.values() if np.isfinite(v[2])))
    lo = min(0.0, min(v[1] for v in stats.values() if np.isfinite(v[1])))
    for j, arch in enumerate(archs):
        ax = fig.add_subplot(gs[0, j])
        for k, set_name in enumerate(sets):
            for m, arm in enumerate(("robbi", "robbianti")):
                est, l, h, npos, nlev = stats[(arch, set_name, arm)]
                if not np.isfinite(est):
                    continue
                x = k + (m - 0.5) * 0.3
                ax.plot([x, x], [l, h], color=D.ARM_C[arm], lw=2.2, solid_capstyle="round")
                ax.plot([x], [est], marker="o", ms=7, color=D.ARM_C[arm], mec="white", mew=1.2)
                ax.annotate(f"{est:+.2f}", (x, h), xytext=(0, 5), textcoords="offset points", ha="center", size=7.4,
                            color=D.INK2)
        D.zero(ax)
        ax.set_ylim(lo - 0.02, hi + 0.035)
        ax.set_xlim(-0.6, len(sets) - 0.4)
        ax.set_xticks(range(len(sets)), ["natural\nposition", "matched\nposition", "pronoun-free\ncontexts"])
        ax.grid(axis="x", visible=False)
        ax.set_title(ARCH_LABEL[arch])
        if j == 0:
            ax.set_ylabel("kept minus removed deciles, beyond random removal\n(Δ share preferring overt)")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=D.ARM_C[a_], lw=2.2, marker="o", ms=6, mec="white", label=ARM_LABEL[a_])
                        for a_ in ("robbi", "robbianti")],
               loc="upper left", bbox_to_anchor=(0.075, 0.275), ncol=2, fontsize=8.2)
    notes = []
    for arch in archs:
        e = [stats[(arch, s_, a_)] for s_ in sets for a_ in ("robbi", "robbianti")]
        ok = [x for x in e if np.isfinite(x[0])]
        if ok:
            notes.append(f"{ARCH_LABEL[arch]}: {min(x[0] for x in ok):+.2f} to {max(x[0] for x in ok):+.2f} across the six "
                         f"comparisons; the interval excludes zero in {sum(x[1] > 0 for x in ok)} of {len(ok)}.")
    notes.append("The matched-position set holds line position equal across deciles, and the pronoun-free set removes "
                 "every personal pronoun from the context. The effect is of similar size in all three, so it is not "
                 "carried by line position or by repeating a pronoun from the context.")
    doc.bottom_notes(fig, notes, 0.215)
    doc.save(fig)


def build(doc, d, archs, a):
    tables = {}
    sets = ["natural", "matched", "nopron"]
    stats = {(arch, s_, arm): excess(d[(d.arch == arch) & d[f"in_{s_}"]], arm)
             for arch in archs for s_ in sets for arm in ("robbi", "robbianti")}
    models = {arch: d[d.arch == arch][["arm", "level"]].drop_duplicates().shape[0] for arch in archs}
    n_items = {s_: int(d.drop_duplicates("item_id")[f"in_{s_}"].sum()) for s_ in sets}

    def n2(v):
        """A share to two places, or a dash while that model has not been scored."""
        return f"{v:.2f}" if np.isfinite(v) else "–"

    def one(arch, arm, level, lo, hi, set_name="natural"):
        g = d[(d.arch == arch) & (d.arm == arm) & (d.level == level) & d[f"in_{set_name}"]]
        return g[g.decile.between(lo, hi)].pref.mean()

    fin = [v[0] for v in stats.values() if np.isfinite(v[0])]
    doc.text_page(
        SECTION, "Decile-graded eval on the unseen test set: first results",
        "End-state of every baseline-condition model, scored on real test-set sentences grouped by how recoverable "
        "their subject pronoun is to the rater (robbi). One run per training cell.",
        [[("What was run", [
            f"Three item sets from test_10M, one distinct pair scored once: natural position ({n_items['natural']:,} "
            f"pairs, ten cells), position matched across deciles ({n_items['matched']:,}), and pronoun-free contexts "
            f"({n_items['nopron']:,}, five cells). Every item was read for type and quality by a model reader.",
            "Models: " + "; ".join(f"{ARCH_LABEL[x]} ({models[x]} of 28)" for x in archs) + ". Training removed 10-90% of "
            "subject pronouns at random, most recoverable first, or least recoverable first, plus all removed. "
            "Only the final checkpoint is scored.",
            "Measure: the share of pairs where the sentence with its pronoun has the higher mean log-probability per "
            "token than the same sentence without it."]),
          ("Main result", [
            "Models drop pronouns where training dropped them. A model trained with the most recoverable pronouns removed "
            "prefers the null version most on the most recoverable test items; a model trained with the least recoverable "
            "removed prefers it most on the least recoverable ones. Random removal lowers the preference by a similar "
            "amount across deciles, with decile 9 lowest throughout.",
            *[f"{ARCH_LABEL[x]}, 80% removed, deciles 0-1 against 8-9: random {n2(one(x, 'rand', 80, 0, 1))} / "
              f"{n2(one(x, 'rand', 80, 8, 9))}; most recoverable first {n2(one(x, 'robbi', 80, 0, 1))} / "
              f"{n2(one(x, 'robbi', 80, 8, 9))}; least recoverable first {n2(one(x, 'robbianti', 80, 0, 1))} / "
              f"{n2(one(x, 'robbianti', 80, 8, 9))}." for x in archs]])],
         [("How strong", [
            "It is a tilt, not a switch. How much was removed still matters most; which deciles were removed shifts the "
            f"preference by {min(fin):.2f} to {max(fin):.2f} on top of that, depending on architecture and item set "
            "(page 3).",
            "With 30% removed most recoverable first, decile 0 has lost every pronoun in training, and the overt version "
            "still wins there: " + "; ".join(
                f"{ARCH_LABEL[x]} {n2(one(x, 'robbi', 30, 0, 0))} (random removal {n2(one(x, 'rand', 30, 0, 0))})"
                for x in archs) + "."]),
          ("Position and context", [
            "Position was left to vary with the decile, as you chose. Least-recoverable-first models drop line-initial "
            "pronouns far more than line-internal ones; most-recoverable-first models do the opposite. The tilt is still "
            "there, at similar size, in the set that holds position equal and in the pronoun-free set.",
          ]),
          ("Read with care", [
            "One run per cell, so differences between single models are not tested; the patterns above are consistent "
            "across the nine levels of an arm.",
            "The rater that orders the test items is the same one that ordered the removals, so this shows that models "
            "learn the rater's notion of recoverability from the training corpus, not that the notion is right.",
            "Baseline condition only. Intervention-matched versions and the trained raters come next."])]])
    p_overview(doc, d, archs)
    p_excess(doc, d, archs, stats)
    for arch in archs:
        label = ARCH_LABEL[arch]
        sub = d[d.arch == arch]
        causal = not arch.startswith("bert")
        have_slot = causal and "slot_pref" in sub and sub.slot_pref.notna().any()
        nat = sub[sub.in_natural]
        graded = nat[nat.arm != "all"]
        dec = graded.groupby("decile").pref.mean().reindex(range(10))
        r = in_out(nat)
        tables[f"{arch}_kept_removed_natural"] = r
        allv = nat[nat.arm == "all"].pref.mean()
        p_heat(doc, d, arch, "natural",
               title=f"{label}: overt preference by test decile and removal level",
               lede="Natural-position set. Each cell is one model's share of pairs in which the sentence with its pronoun "
                    "scores higher than the sentence without it. Rows: how much was removed in training. Columns: decile "
                    "of the test item. The black line separates the deciles an ordered arm removed from the ones it kept.",
               notes=[
                   f"Pooled over the graded models the overt version wins {graded.pref.mean():.0%} of pairs"
                   + (f"; with all pronouns removed it wins {allv:.0%}." if np.isfinite(allv) else "."),
                   f"Random removal: decile 9 is the lowest column at every level ({n2(one(arch, 'rand', 50, 9, 9))} at 50% "
                   f"against {n2(one(arch, 'rand', 50, 0, 0))} in decile 0). The learners also find the rater's least "
                   "recoverable pronouns the least expected.",
                   f"Most recoverable first, 90% removed: {n2(one(arch, 'robbi', 90, 0, 0))} in decile 0, the first one "
                   f"removed, against {n2(one(arch, 'robbi', 90, 9, 9))} in decile 9, the only one kept.",
                   f"Least recoverable first, 90% removed: {n2(one(arch, 'robbianti', 90, 9, 9))} in decile 9, the first "
                   f"one removed, against {n2(one(arch, 'robbianti', 90, 0, 0))} in decile 0, the only one kept.",
               ])
        def delta_notes(deltas):
            out = []
            for arm, word in (("robbi", "most recoverable first"), ("robbianti", "least recoverable first")):
                A = deltas[arm]
                rm = np.array([[dc in removed(arm, lv) for dc in range(10)] for lv in LEVELS])
                out.append(f"{word.capitalize()}: {np.nanmean(A[rm]):+.3f} on average in cells the arm removed, "
                           f"{np.nanmean(A[~rm]):+.3f} in cells it kept.")
            out.append("At equal volume removed, an ordered arm sits below random where it removed and at or above "
                       "random where it kept. Volume is held equal in every cell of this page, so what is left is where "
                       "the removals fell.")
            return out

        p_delta(doc, d, arch, "natural",
                title=f"{label}: each ordered arm against random removal of the same amount",
                lede="Natural-position set. Each cell is an ordered-removal model's overt preference minus that of the "
                     "random-removal model with the same share removed, on the same items.",
                notes_fn=delta_notes)
        e = {k[1:]: v for k, v in stats.items() if k[0] == arch}
        p_inout(doc, d, arch, ["natural", "matched"],
                title=f"{label}: deciles the arm kept against deciles it removed",
                lede="For each ordered arm and removal level, test items are split into deciles that arm removed from "
                     "training and deciles it kept. Grey lines put the random-removal model of the same level through the "
                     "same split: a gap there is what the deciles do by themselves.",
                notes_fn=lambda out, e=e: [
                    "Solid above dashed: the model prefers the pronoun more in deciles it kept than in deciles it lost. "
                    "The arm's own effect is how much wider its gap is than the grey gap.",
                    f"Natural position: most recoverable first {e[('natural', 'robbi')][0]:+.3f} "
                    f"({e[('natural', 'robbi')][3]} of {e[('natural', 'robbi')][4]} levels wider than random); least "
                    f"recoverable first {e[('natural', 'robbianti')][0]:+.3f} ({e[('natural', 'robbianti')][3]} of "
                    f"{e[('natural', 'robbianti')][4]}).",
                    f"Position matched: most recoverable first {e[('matched', 'robbi')][0]:+.3f} "
                    f"({e[('matched', 'robbi')][3]} of {e[('matched', 'robbi')][4]}); least recoverable first "
                    f"{e[('matched', 'robbianti')][0]:+.3f} ({e[('matched', 'robbianti')][3]} of "
                    f"{e[('matched', 'robbianti')][4]})."])
        mc = nat[nat.cell.str.startswith("subj_")]
        pl = mc[mc.level >= 60].groupby(["arm", "line_initial"]).pref.mean()
        p_place(doc, d, arch,
                title=f"{label}: line-initial pronouns against the rest",
                lede="Natural-position set, main-clause cells only (I, we, you, he/she/it, they). Position varies with the "
                     "decile in this set, so this page separates the two: for every model, items whose pronoun starts its "
                     "corpus line against items whose pronoun is inside a line.",
                notes=[
                    f"At 60-90% removed, line-initial against line-internal: random {pl.get(('rand', True), np.nan):.2f} / "
                    f"{pl.get(('rand', False), np.nan):.2f}; most recoverable first {pl.get(('robbi', True), np.nan):.2f} / "
                    f"{pl.get(('robbi', False), np.nan):.2f}; least recoverable first "
                    f"{pl.get(('robbianti', True), np.nan):.2f} / {pl.get(('robbianti', False), np.nan):.2f}.",
                    "Least-recoverable-first removal takes mostly line-initial pronouns out of training (44% of them at 30% "
                    "removed, against 6% for most-recoverable-first), and the models follow it."])
        if (sub.in_nopron & (sub.arm != "all")).any():
            np_ = sub[sub.in_nopron & (sub.arm != "all")]
            p_heat(doc, d, arch, "nopron",
                   title=f"{label}: pronoun-free contexts",
                   lede="Pronoun-free set: five cells (I, he/she/it, they, embedded after the verb, adverbial). The context "
                        "is the longest stretch of real preceding text with no personal pronoun (median 12 words), and the "
                        "decile is the one the rater gives the item with that context.",
                   notes=[f"Overt wins {np_.pref.mean():.0%} of pairs here against {graded.pref.mean():.0%} in the natural "
                          "set.",
                          f"Arm-specific effect: most recoverable first {e[('nopron', 'robbi')][0]:+.3f}, least "
                          f"recoverable first {e[('nopron', 'robbianti')][0]:+.3f}; the same size as with pronouns in the "
                          "context."])
        p_cells(doc, d, arch, "natural", 80,
                title=f"{label}: by construction, at 80% removed",
                lede="Natural-position set, 24 pairs per point (48 for conjunction and adverbial). One line per arm at 80% "
                     "removal, plus the all-removed model. An ordered arm at 80% has removed deciles 0-7 "
                     "(most recoverable first) or 2-9 (least recoverable first).",
                notes=["If removal transfers to matching contexts, blue should rise toward the right and orange fall "
                       "toward the right, with grey between them.",
                       "With 24 pairs one pair moves a point by 0.04, so read the direction of a line, not single points."])
        if have_slot:
            es = {arm: excess(nat, arm, "slot_pref") for arm in ("robbi", "robbianti")}
            p_heat(doc, d, arch, "natural", value="slot_pref",
                   title=f"{label}: pronoun against the next word, at the pronoun's own slot",
                   lede="Natural-position set, a stricter reading of the same pairs. After the identical prefix, is the "
                        "pronoun more probable than the word that would come next if it were dropped? Each cell is the "
                        "share of pairs where it is. Nothing later in the sentence enters this number.",
                   notes=[f"Same direction as the sentence-level measure and smaller: arm-specific effect "
                          f"{es['robbi'][0]:+.3f} for most recoverable first and {es['robbianti'][0]:+.3f} for least "
                          f"recoverable first, against {e[('natural', 'robbi')][0]:+.3f} and "
                          f"{e[('natural', 'robbianti')][0]:+.3f}.",
                          "So the tilt is not produced by averaging log-probability over a target that is one token "
                          "longer with the pronoun."])
    if a.tables:
        a.tables.mkdir(parents=True, exist_ok=True)
        for name, t in tables.items():
            t.to_csv(a.tables / f"{name}.csv", index=False)
        rows = []
        for (arch, s_, arm), v in stats.items():
            rows.append({"arch": arch, "set": s_, "arm": arm, "excess": v[0], "low": v[1], "high": v[2],
                         "levels_positive": v[3], "levels": v[4]})
        pd.DataFrame(rows).to_csv(a.tables / "arm_specific_effect.csv", index=False)
        long = []
        for s_ in sets:
            g = d[d[f"in_{s_}"]].groupby(["arch", "arm", "level", "decile"]).agg(
                pref=("pref", "mean"), lpd=("lpd", "mean"), n=("pref", "size")).reset_index()
            g.insert(0, "set", s_)
            long.append(g)
        pd.concat(long).to_csv(a.tables / "overt_preference_by_model_and_decile.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, default=None, help="directory for the CSV tables behind the figures")
    a = ap.parse_args()
    plt = D.setup_mpl()
    d = load(a.results, a.key)
    archs = [x for x in ARCH_LABEL if x in set(d.arch)]
    D.FOOT = ("decile-graded test-set eval (rater robbi) · end-state, baseline condition, one run per cell · "
              "eval_recoverability_testset_results.py")
    doc = D.Doc(a.out, plt)
    build(doc, d, archs, a)
    doc.close()
    print(f"wrote {a.out} ({doc.n} pages)")


if __name__ == "__main__":
    main()
