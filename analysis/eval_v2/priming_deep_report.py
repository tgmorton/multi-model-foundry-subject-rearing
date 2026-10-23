#!/usr/bin/env python3
"""Deep-dive PDF on cross-construction priming in the wave2 gpt2_large cohort
(140 runs × 97 checkpoints × 4,608 prime–target pairings).

Reads the item-level tables written by priming_deep_data.py (run it first). Every
number in the text is computed here, so the prose follows the data on a rebuild.

    python analysis/eval_v2/priming_deep_data.py
    python analysis/eval_v2/priming_deep_report.py --out priming_deep.pdf
"""

from __future__ import annotations

import argparse
import difflib
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nopron_priming_report as R  # noqa: E402
import priming_deep_data as P  # noqa: E402

# Palette: the dataviz reference instance (validated: adjacent CVD ΔE ≥ 9.1, the
# first three also all-pairs). Grey is reserved for controls / reference series.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET = "#6250d6"
INK, INK2, MUTED, RULE, GRID, CTRL = "#1d1d1b", "#3d3c38", "#6b6a64", "#cfcdc6", "#ecebe6", "#9a9990"
# Arrows (→ ↔) are missing from Avenir Next and Charter: fall back per glyph to DejaVu Sans.
HEAD, BODY, MONO = ["Avenir Next", "DejaVu Sans"], ["Charter", "DejaVu Sans"], ["Menlo"]

REL = ["same word", "same features", "other pronoun", "non-pronoun"]
REL_C = {"same word": BLUE, "same features": ORANGE, "other pronoun": AQUA, "non-pronoun": CTRL}
REL_LABEL = {"same word": "same word  (she → she)",
             "same features": "same person/number/gender, other case  (him → he)",
             "other pronoun": "different pronoun  (it → she)",
             "non-pronoun": "non-pronoun element  (that → she)"}
ARMS = ["rand", "robbi", "robbianti", "all"]
ARM_C = {"rand": CTRL, "robbi": BLUE, "robbianti": ORANGE, "all": INK}
ARM_LABEL = {"rand": "random", "robbi": "most recoverable first", "robbianti": "least recoverable first",
             "all": "all removed (100%)"}
IV_C = dict(zip(R.IVS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
CATS = ["subject_drop", "subject_drop_no_agreement", "expletive", "conjunction", "embedded_drop",
        "control", "extraction", "object_drop"]
SHORT = {**R.CAT_SHORT, "subject_drop_no_agreement": "subj drop (no agr)"}
WORDS = ["i", "we", "you", "he", "him", "she", "it", "they", "them", "that"]
SUBJC = ["subject_drop", "subject_drop_no_agreement", "expletive"]
VALS = ["dpref", "dlpd", "pref_o", "pref_n", "pref_u", "lpd_o", "lpd_n", "lpd_u"]
EFF = "priming effect, Δ log-prob (nats/token)"
EFFP = "priming effect, Δ P(prefer overt)"
FOOT = "gpt2_large wave2 · cross-construction priming · stimuli unvetted · built from priming_deep_data.py tables"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    meta = pd.read_parquet(P.OUT / "meta.parquet")
    end = R.parse_runs(pd.read_parquet(P.OUT / "end.parquet"))
    end["cond_dir"] = end.iv.map(P.IV_DIR)
    e = end.merge(meta, on=["cond_dir", "pid"], validate="m:1")
    for m in P.MEAS:
        e[f"d{m}"] = e[f"{m}_o"] - e[f"{m}_n"]
    tok = pd.read_parquet(R.CACHE / "tokens.parquet").rename(columns={"checkpoint_step": "step"})
    tok = tok.sort_values(["run", "step"])
    tok["idx"] = tok.groupby("run").cumcount()
    T = tok.groupby("idx").tokens_seen.median() / 1e6
    tr = pd.read_parquet(P.OUT / "traj.parquet").merge(tok[["run", "step", "idx"]], on=["run", "step"], validate="m:1")
    for m in ("pref", "lpd"):
        tr[f"d{m}"] = tr[f"{m}_o"] - tr[f"{m}_n"]
    for c in VALS:
        tr[f"{c}_w"] = tr[c] * tr.n
    side = pd.read_parquet(P.OUT / "side.parquet").merge(tok[["run", "step", "idx"]], on=["run", "step"], validate="m:1")
    for c in SIDE_VALS:
        side[f"{c}_w"] = side[c] * side.n
    return meta, e, tr, T, side


SIDE_VALS = ["null_lpd", "overt_lpd", "null_pref", "overt_pref"]
PGROUPS = ["subj drop 3rd", "subj drop 1st/2nd", "expletive", "conjunction", "embedded_drop", "control", "extraction",
           "object_drop"]
PG_SHORT = {"subj drop 3rd": "subj drop, 3rd p.", "subj drop 1st/2nd": "subj drop, 1st/2nd p.", **SHORT}


def sagg(side, mask=None, keys=()):
    """Run × checkpoint (× keys) means of the reference-relative sides, weighted by pairings."""
    d = side if mask is None else side[mask]
    g = d.groupby(["run", "idx", *keys], observed=True)
    out = g[[f"{c}_w" for c in SIDE_VALS]].sum().div(g.n.sum(), axis=0)
    out.columns = SIDE_VALS
    return out.reset_index()


def tagg(tr, mask=None, keys=()):
    """Run × checkpoint (× keys) means, weighted by pairing counts."""
    d = tr if mask is None else tr[mask]
    g = d.groupby(["run", "idx", *keys], observed=True)
    s = g[[f"{c}_w" for c in VALS]].sum().div(g.n.sum(), axis=0)
    s.columns = VALS
    return R.parse_runs(s.reset_index())


def masks(df):
    """Named subsets, defined identically on the end-state and trajectory frames."""
    rel = df.relation.astype(str)
    tc, pc = df.target_category.astype(str), df.prime_category.astype(str)
    tw, pw = df.target_word.astype(str), df.prime_word.astype(str)
    return {
        "same word": rel == "same word",
        "same features": rel == "same features",
        "other pronoun": rel == "other pronoun",
        "non-pronoun": rel == "non-pronoun",
        "subject cluster": tc.isin(SUBJC) & pc.isin(SUBJC) & (rel == "other pronoun"),
        "object cluster": (tc == "object_drop") & (pc == "object_drop") & rel.isin(["other pronoun", "non-pronoun"]),
        "he/she competition": tw.isin(["he", "she"]) & pw.isin(["he", "she"]) & (tw != pw),
    }


def run_ci(x):
    return R.ci(x)


def item_boot(e, mask, value="dlpd", n=400, seed=0):
    """95% CI of a subset mean when TARGET ITEMS are resampled (pairing means over runs)."""
    d = e[mask]
    it = d.groupby(["cond_dir", "pid", "target_category", "target_condition", "target_item_id"])[value].mean().reset_index()
    it["t"] = it.target_category + "|" + it.target_condition + "|" + it.target_item_id.astype(str)
    s = it.groupby("t")[value].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    b = s["sum"].to_numpy()[idx].sum(1) / s["size"].to_numpy()[idx].sum(1)
    return np.percentile(b, [2.5, 97.5])


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": HEAD, "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
        "axes.titlesize": 10, "axes.titleweight": "demibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.titlepad": 8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.8,
        "pdf.fonttype": 42, "lines.linewidth": 2, "lines.solid_capstyle": "round", "patch.linewidth": 0,
    })
    return plt


class Doc:
    def __init__(self, path, plt):
        from matplotlib.backends.backend_pdf import PdfPages
        self.pdf, self.plt, self.n = PdfPages(path), plt, 0
        self.ref = {}   # page name -> number, for cross-references

    def page(self, section, title, lede, notes=None, notes_title="What to see"):
        fig = self.plt.figure(figsize=(11, 8.5))
        self.n += 1
        fig.text(0.055, 0.955, section.upper(), family=HEAD, size=7.8, color=MUTED, weight="demibold")
        fig.text(0.055, 0.935, title, family=HEAD, size=17, weight="bold", color=INK, va="top")
        lines = textwrap.wrap(lede, 128)
        fig.text(0.055, 0.885, "\n".join(lines), family=BODY, size=10.5, color=INK2, va="top", linespacing=1.45)
        rule_y = 0.885 - 0.0245 * len(lines) - 0.012
        fig.add_artist(self.plt.Line2D([0.055, 0.945], [rule_y, rule_y], color=RULE, lw=0.8))
        fig.text(0.055, 0.03, FOOT, family=HEAD, size=6.8, color=MUTED)
        fig.text(0.945, 0.03, str(self.n), family=HEAD, size=7.5, color=MUTED, ha="right")
        top = rule_y - 0.06
        right = 0.945
        if notes:
            right = 0.635
            y = rule_y - 0.035
            fig.text(0.69, y, notes_title.upper(), family=HEAD, size=7.6, color=MUTED, weight="demibold", va="top")
            y -= 0.032
            for b in notes:
                wl = textwrap.wrap(b, 43)
                fig.text(0.69, y, "—", family=BODY, size=9.2, color=MUTED, va="top")
                fig.text(0.707, y, "\n".join(wl), family=BODY, size=9.2, color=INK, va="top", linespacing=1.38)
                y -= 0.0195 * len(wl) + 0.014
        return fig, dict(left=0.075, right=right, bottom=0.095, top=top)

    def bottom_notes(self, fig, notes, top, ncol=2, title="What to see"):
        """Notes as columns under the figure, for pages that need the full width."""
        fig.text(0.075, top, title.upper(), family=HEAD, size=7.6, color=MUTED, weight="demibold", va="top")
        w = 0.87 / ncol
        per = -(-len(notes) // ncol)
        for j in range(ncol):
            y = top - 0.03
            for b in notes[j * per:(j + 1) * per]:
                wl = textwrap.wrap(b, int(140 / ncol))
                fig.text(0.075 + j * w, y, "—", family=BODY, size=9.0, color=MUTED, va="top")
                fig.text(0.092 + j * w, y, "\n".join(wl), family=BODY, size=9.0, color=INK, va="top", linespacing=1.35)
                y -= 0.019 * len(wl) + 0.012

    def text_page(self, section, title, lede, columns):
        """Columns of (heading, [paragraphs])."""
        fig, _ = self.page(section, title, lede)
        ncol = len(columns)
        w = 0.89 / ncol
        for j, blocks in enumerate(columns):
            x, y = 0.055 + j * w, 0.80
            for head, paras in blocks:
                if head:
                    fig.text(x, y, head.upper(), family=HEAD, size=7.8, color=MUTED, weight="demibold", va="top")
                    y -= 0.028
                for p in paras:
                    wl = textwrap.wrap(p, int(118 / ncol * 1.02))
                    fig.text(x, y, "\n".join(wl), family=BODY, size=9.6, color=INK, va="top", linespacing=1.4)
                    y -= 0.0205 * len(wl) + 0.013
                y -= 0.012
        self.save(fig)

    def save(self, fig):
        self.pdf.savefig(fig)
        self.plt.close(fig)

    def close(self):
        self.pdf.close()


def tokens_axis(ax, label=True):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_xscale("log")
    ax.set_xlim(0.45, 4500)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}M" if v < 1000 else f"{v / 1000:g}B"))
    if label:
        ax.set_xlabel("training tokens (log scale)")


def band(ax, df, col, color, T, label=None, ls="-", lw=2.0, alpha=0.15):
    g = df[df.idx > 0].groupby("idx")[col]
    m, e = g.mean(), g.apply(run_ci)
    x = T.loc[m.index].to_numpy()
    ax.fill_between(x, m - e, m + e, color=color, alpha=alpha, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label)
    return pd.Series(m.to_numpy(), index=x)


def zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=MUTED, lw=0.8, zorder=1)


def heat(ax, M, rows, cols, lim, fmt="{:+.2f}", size=7.2, annotate=True, nan_text="–"):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#a8431a", "#f0a47f", "#f4f3ef", "#8db6e8", "#1c58a3"])
    cmap.set_bad("#f7f6f3")
    A = np.asarray(M, float)
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            if not annotate and not np.isnan(v):
                continue
            ax.text(j, i, nan_text if np.isnan(v) else fmt.format(v), ha="center", va="center", size=size,
                    color=MUTED if np.isnan(v) else ("white" if abs(v) > 0.55 * lim else INK))
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def bars(ax, labels, means, errs, colors, horizontal=False, fmt="{:+.3f}", label_vals=True):
    pos = np.arange(len(labels))
    if horizontal:
        ax.barh(pos, means, xerr=errs, color=colors, height=0.62, error_kw={"lw": 0.8, "ecolor": INK2, "capsize": 2})
        ax.set_yticks(pos, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        if label_vals:
            for p, m, er in zip(pos, means, errs):
                er = er if np.isfinite(er) else 0
                x = m + er if m >= 0 else 0
                ax.annotate(fmt.format(m), (x, p), xytext=(4, 0), textcoords="offset points",
                            va="center", ha="left", size=7.4, color=INK2)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo, hi + 0.18 * (hi - lo))
    else:
        ax.bar(pos, means, yerr=errs, color=colors, width=0.66, error_kw={"lw": 0.8, "ecolor": INK2, "capsize": 2})
        ax.set_xticks(pos, labels)
        ax.grid(axis="x", visible=False)
    return pos


def bracket(over: str, null: str) -> str:
    """The overt string with the words the null string lacks in [brackets]."""
    a, b = over.split(), null.split()
    out = []
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        seg = a[i1:i2]
        out.append(f"[{' '.join(seg)}]" if tag in ("delete", "replace") and seg else " ".join(seg))
    return " ".join(s for s in out if s)


# ---------------------------------------------------------------------------
# Summary numbers (used across pages, incl. the cover)
# ---------------------------------------------------------------------------
def summarise(e, tr, T):
    S = {}
    per = e.groupby(["run", "arm", "k", "iv"])[["dpref", "dlpd", "dhot", "pref_u", "pref_o", "pref_n"]].mean().reset_index()
    S["per"] = per
    S["n_runs"] = per.run.nunique()
    S["n_pairings"] = e.pid.nunique()
    S["dpref"], S["dlpd"] = per.dpref.mean(), per.dlpd.mean()
    S["dpref_ci"], S["dlpd_ci"] = run_ci(per.dpref), run_ci(per.dlpd)
    S["pos_runs"] = int((per.dlpd > 0).sum())
    S["pos_runs_pref"] = int((per.dpref > 0).sum())
    init = tagg(tr, tr.idx == 0)
    S["init_dlpd"] = init.dlpd.mean()
    S["init_dpref"] = init.dpref.mean()
    M = masks(e)
    rel = {}
    for r in REL + ["subject cluster", "object cluster", "he/she competition"]:
        d = e[M[r]]
        pr = d.groupby("run")[["dlpd", "dpref"]].mean()
        rel[r] = dict(share=M[r].mean(), dlpd=pr.dlpd.mean(), dpref=pr.dpref.mean(),
                      ci=run_ci(pr.dlpd), boot=item_boot(e, M[r]))
    tot = e.dlpd.mean()
    for r in REL:
        rel[r]["contrib"] = rel[r]["share"] * e[M[r]].dlpd.mean() / tot
    S["rel"] = rel
    # position diagnostic: prime "it" by source, targets without "it"
    x = e[(e.prime_word == "it") & (e.target_word != "it")]
    subjt = x.target_category.isin(["subject_drop", "subject_drop_no_agreement", "conjunction", "embedded_drop"])
    S["it_expl_subj"] = x[subjt & (x.prime_category == "expletive")].groupby("run").dlpd.mean().mean()
    S["it_obj_subj"] = x[subjt & (x.prime_category == "object_drop")].groupby("run").dlpd.mean().mean()
    S["it_obj_obj"] = x[(x.target_category == "object_drop") & (x.prime_category == "object_drop")].groupby("run").dlpd.mean().mean()
    # rearing regression (selection arms only; the all-removed arm has a single level)
    import statsmodels.formula.api as smf
    q = per[per.arm != "all"]
    fit = smf.ols('dlpd ~ C(arm, Treatment("rand")) + pu + C(iv)', q.rename(columns={"pref_u": "pu"})).fit()
    S["fit"] = fit
    S["arm_means"] = per.groupby("arm")[["dlpd", "dpref", "pref_u"]].mean()
    S["iv_means"] = per.groupby("iv")[["dlpd", "dpref", "pref_u"]].mean()
    # emergence timing (run bootstrap)
    S["timing"] = {}
    Mt = masks(tr)
    for r in ["same word", "same features", "other pronoun", "subject cluster", "object cluster"]:
        S["timing"][r] = emergence(tagg(tr, Mt[r]), T)
    S["timing"]["all"] = emergence(tagg(tr), T)
    # reliability
    runs = sorted(e.run.unique())
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(20):
        h = set(rng.permutation(runs)[: len(runs) // 2])
        a = e[e.run.isin(h)].groupby(["cond_dir", "pid"]).dlpd.mean()
        b = e[~e.run.isin(h)].groupby(["cond_dir", "pid"]).dlpd.mean()
        rs.append(np.corrcoef(a, b.loc[a.index])[0, 1])
    S["split_r"] = float(np.mean(rs))
    return S


def side_end(side, targets):
    """End-state (last 3 checkpoints) reference-relative sides per run × prime group, for some targets."""
    d = sagg(side, side.target_category.isin(targets) & (side.idx >= side.idx.max() - 2), ["prime_group"])
    return d.groupby(["run", "prime_group"])[SIDE_VALS].mean().reset_index()


def summarise_sides(S, side):
    se = side_end(side, R.SUBJ).groupby("prime_group")[SIDE_VALS].mean()
    S["side_subj"] = se
    S["side_3rd_null"], S["side_3rd_overt"] = se.at["subj drop 3rd", "null_lpd"], se.at["subj drop 3rd", "overt_lpd"]
    S["side_expl_null"] = se.at["expletive", "null_lpd"]
    S["side_conj_null"] = se.at["conjunction", "null_lpd"]


def emergence(df, T, n=300, seed=1, col="dlpd"):
    """Tokens at which the run-mean curve first reaches 50% / 90% of its end-state (in the
    direction of the end-state sign), with run-bootstrap CIs."""
    W = df.pivot(index="run", columns="idx", values=col).sort_index(axis=1)
    A = W.to_numpy()
    rng = np.random.default_rng(seed)

    def times(mat):
        m = pd.Series(np.nanmean(mat, 0), index=W.columns).rolling(3, center=True, min_periods=1).mean()
        fin = m.iloc[-3:].mean()
        sg = 1.0 if fin >= 0 else -1.0
        out = []
        for f in (0.5, 0.9):
            hit = m[(m.index > 0) & (m * sg >= f * abs(fin))]
            out.append(T.loc[hit.index[0]] if len(hit) else np.nan)
        return out
    point = times(A)
    boot = np.array([times(A[rng.integers(0, len(A), len(A))]) for _ in range(n)])
    return {"t50": point[0], "t90": point[1], "t50_ci": np.nanpercentile(boot[:, 0], [2.5, 97.5]),
            "t90_ci": np.nanpercentile(boot[:, 1], [2.5, 97.5]),
            "final": np.nanmean(A[:, -3:])}


def fmt_tok(v):
    return f"{v:.0f}M" if v < 1000 else f"{v / 1000:.1f}B"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def p_cover(doc, S):
    rel, tm = S["rel"], S["timing"]
    fig = doc.plt.figure(figsize=(11, 8.5))
    doc.n += 1
    fig.text(0.055, 0.90, "CROSS-CONSTRUCTION PRIMING · GPT2_LARGE WAVE2", family=HEAD, size=8.5, color=MUTED, weight="demibold")
    fig.text(0.055, 0.875, "What gets primed, and when", family=HEAD, size=30, weight="bold", color=INK, va="top")
    fig.text(0.055, 0.80, textwrap.fill(
        f"{S['n_runs']} runs × 97 checkpoints × {S['n_pairings']:,} prime–target pairings "
        f"({S['n_runs'] * 97 * S['n_pairings'] * 2 / 1e6:.0f}M scored minimal pairs). Each target minimal pair "
        "is scored after the same prime item in its overt and its null version; the priming effect is the "
        "difference. Stimuli are unvetted (pronoun-free contexts, built 2026-10-21).", 120),
        family=BODY, size=10.5, color=INK2, va="top", linespacing=1.45)
    fig.add_artist(doc.plt.Line2D([0.055, 0.945], [0.715, 0.715], color=RULE, lw=0.8))
    fit = S["fit"]
    b_rob = fit.params['C(arm, Treatment("rand"))[T.robbi]']
    b_anti = fit.params['C(arm, Treatment("rand"))[T.robbianti]']
    t_rob = fit.tvalues['C(arm, Treatment("rand"))[T.robbi]']
    sw, op = rel["same word"], rel["other pronoun"]
    pts = [
        ("Real, and learned",
         f"An overt prime raises the target's overt preference relative to a null prime in {S['pos_runs']}/{S['n_runs']} runs: "
         f"+{S['dlpd']:.3f} nats/token (±{S['dlpd_ci']:.3f}), or +{S['dpref']:.3f} in P(prefer overt). At initialization it is "
         f"{S['init_dlpd']:+.4f}: nothing in the stimuli alone produces it."),
        ("Mostly the word repeating",
         f"Pairings where the prime's overt element is the target's own pronoun are {sw['share']:.0%} of pairings but carry "
         f"{sw['contrib']:.0%} of the pooled effect (+{sw['dlpd']:.3f} vs +{op['dlpd']:.3f} for a different pronoun)."),
        ("Then grammatical function",
         f"Subject-position elements prime overt subjects; object-position elements prime objects. Expletive 'it' raises other "
         f"subject pronouns by +{S['it_expl_subj']:.3f}, object 'it' by {S['it_obj_subj']:+.3f}, while object 'it' primes "
         f"overt objects by +{S['it_obj_obj']:.3f}. Subject drop and expletives prime each other in both directions."),
        ("Null primes do real work",
         f"Against an object-drop reference, a dropped 3rd-person subject pushes the next subject-drop target "
         f"+{S['side_3rd_null']:.3f} nats/token toward null, while its overt version adds only {S['side_3rd_overt']:+.3f}. "
         f"Dropped expletives push toward null too (+{S['side_expl_null']:.3f}). Coordination drop, which English allows, "
         f"does not ({S['side_conj_null']:+.3f}). The split holds under four different reference primes."),
        ("Competition too",
         f"he ↔ she primes negatively ({rel['he/she competition']['dlpd']:+.3f}); I / we / you / they facilitate each other."),
        ("Early, then stable",
         f"Flat to 16M tokens; lexical priming reaches half size by {fmt_tok(tm['same word']['t50'])}, the abstract subject "
         f"cluster by {fmt_tok(tm['subject cluster']['t50'])}; most of it is in place by ~{fmt_tok(tm['subject cluster']['t90'])} "
         "and stable for the remaining ~3B tokens. Interference involving embedded drop is the exception: it keeps growing."),
        ("Rearing sets primeability",
         f"At matched removal volume, recoverability-ordered removal roughly doubles priming over random removal "
         f"(+{b_rob:.3f} / +{b_anti:.3f} nats, t = {t_rob:.0f}), beyond what the models' overt preference explains. The change "
         "is in subject priming; object priming is untouched. Lemmatized verbs are the most primeable intervention."),
        ("Stable items, few of them",
         f"Pairing-level effects correlate r = {S['split_r']:.2f} between random halves of the runs. Error bars across runs "
         "treat the items as fixed; resampling items gives wider intervals (shown where it matters)."),
    ]
    y = 0.68
    for head, body in pts:
        fig.text(0.055, y, head, family=HEAD, size=11, weight="demibold", color=INK, va="top")
        wl = textwrap.wrap(body, 108)
        fig.text(0.255, y, "\n".join(wl), family=BODY, size=9.8, color=INK, va="top", linespacing=1.4)
        y -= max(0.04, 0.0215 * len(wl) + 0.022)
    fig.text(0.055, 0.03, FOOT, family=HEAD, size=6.8, color=MUTED)
    doc.save(fig)


def p_design(doc, meta):
    mb = split_prime(meta)
    other = set(P.PRON) - {"it", "she"}
    clean = mb[(mb.target_category == "subject_drop") & (mb.prime_category == "expletive") & (mb.target_word == "she") &
               np.array([not (set(f"{a} {b}".split()) & other) for a, b in zip(mb.prime_o, mb.target_n)])]
    b = clean.iloc[0]
    fig, box = doc.page(
        "1 · Reading the data", "Anatomy of one pairing",
        "Every target minimal pair is scored twice: after the overt version of a prime item and after its null version. "
        "The prime item comes from a different construction (or a different item of the same one) and shares no content "
        "word or name with the target.")
    rows = [
        ("PRIME, OVERT VERSION", BLUE, bracket(b.prime_o, b.prime_n)),
        ("PRIME, NULL VERSION", ORANGE, b.prime_n),
        ("TARGET CONTEXT", INK2, b.tctx),
        ("TARGET, OVERT", INK2, bracket(b.target_o, b.target_n)),
        ("TARGET, NULL", INK2, b.target_n),
    ]
    fig.text(0.075, 0.775, "Scored string = prime (one version) + target context + target (one version). "
                           "Four strings per pairing; the target pair is compared within each prime version.",
             family=HEAD, size=7.8, color=MUTED, va="top")
    y = 0.74
    for lab, col, txt in rows:
        fig.text(0.075, y, lab, family=HEAD, size=8, weight="demibold", color=col, va="top")
        fig.text(0.25, y, "\n".join(textwrap.wrap(txt, 92)), family=MONO, size=8.8, color=INK, va="top", linespacing=1.45)
        y -= 0.046
    fig.text(0.25, y + 0.008, f"prime: expletive ({b.prime_condition}) · target: subject drop ({b.target_condition}) · "
                              f"overt elements: '{b.prime_word}' → '{b.target_word}' (different words, both subjects)",
             family=HEAD, size=7.8, color=MUTED, va="top")
    cols = [
        ("The measures", [
            "For each scored context the model gives the target's overt and null versions a mean log-probability per token. "
            "Δ log-prob = overt − null (nats/token); P(prefer overt) = 1 when the overt version wins.",
            "Priming effect = the measure after the overt prime − the same measure after the null prime, for the same "
            "target and the same prime item. It is a paired, within-item difference: the target text, the prime's content "
            "and the position are held fixed; only the prime's overt element differs.",
            "The two prime versions differ by one word, so the non-pronoun pairings (extraction 'that' primes on pronoun "
            "targets) double as a control for 'one more word in the context'.",
        ]),
        ("Relations between prime and target", [
            "same word — the overt prime adds exactly the target's pronoun (she → she).",
            "same features — same person, number and gender, other case (him → he, them → they).",
            "different pronoun — any other pronoun pair (it → she, we → you).",
            "non-pronoun — either side is the extraction complementizer 'that'.",
            "Words are read off each run's own condition, so impoverish-case items count him → he as 'same word'.",
        ]),
        ("Unprimed, end-state, uncertainty", [
            "Unprimed = the same target scored with its own pronoun-free context only (the matched-nopron eval rides along "
            "in every cell). Primed items carry a whole extra discourse, so unprimed is not a neutral baseline.",
            "End-state = mean of each run's last three checkpoints (step > 0). Trajectories align runs by checkpoint index "
            "(token counts agree within 5%).",
            "Error bars: 95% CI across the 140 runs unless marked 'items'. Runs share the items, so these treat items as "
            "fixed; item-resampled intervals are wider.",
        ]),
    ]
    w = 0.29
    for j, (head, paras) in enumerate(cols):
        x, yy = 0.075 + j * (w + 0.015), 0.43
        fig.text(x, yy, head.upper(), family=HEAD, size=7.8, weight="demibold", color=MUTED, va="top")
        yy -= 0.03
        for ptxt in paras:
            wl = textwrap.wrap(ptxt, 50)
            fig.text(x, yy, "\n".join(wl), family=BODY, size=8.9, color=INK, va="top", linespacing=1.36)
            yy -= 0.0185 * len(wl) + 0.011
    doc.save(fig)


def p_headline(doc, S, e, tr, T):
    per = S["per"]
    fig, box = doc.page(
        "2 · Headline", f"Every model is primed, and only after training",
        f"Per-run end-state priming effect on both measures. All {S['n_runs']} runs are above zero; the untrained "
        f"checkpoint is at zero. The spread across runs is mostly rearing condition (section 6), not noise.",
        notes=[f"Δ log-prob: +{S['dlpd']:.4f} ± {S['dlpd_ci']:.4f} nats/token; {S['pos_runs']}/{S['n_runs']} runs > 0.",
               f"Δ P(prefer overt): +{S['dpref']:.4f} ± {S['dpref_ci']:.4f}; {S['pos_runs_pref']}/{S['n_runs']} runs > 0. "
               "The binary measure is a coarsened view of the same shift: only items near the overt/null boundary can flip.",
               f"These average over pairings. Averaging over target constructions instead (as the v2 report did) gives "
               f"+{e.groupby(['run', 'target_category']).dpref.mean().groupby('run').mean().mean():.3f}: subject drop has "
               "three times the pairings and is primed more.",
               f"Untrained checkpoint: {S['init_dlpd']:+.5f} nats, {S['init_dpref']:+.4f} in preference. The paired design "
               "cancels everything except what the model has learned about the prime.",
               "The long right tail is the all-removed and least-recoverable-first runs: models that rarely saw "
               "overt subjects are the most swayed by one in context.",
               f"Hotspot log-prob (the critical token only) is positive on average (+{per.dhot.mean():.3f}) but noisy: "
               f"{int((per.dhot > 0).sum())}/{S['n_runs']} runs > 0. It is not used further."])
    gs = fig.add_gridspec(2, 2, **box, hspace=0.55, wspace=0.28)
    for j, (col, lab) in enumerate([("dlpd", EFF), ("dpref", EFFP)]):
        ax = fig.add_subplot(gs[0, j])
        ax.hist([per[per.arm == arm][col] for arm in ARMS], bins=np.linspace(0, per[col].max() * 1.05, 28),
                color=[ARM_C[arm] for arm in ARMS], label=[ARM_LABEL[arm] for arm in ARMS], stacked=True,
                edgecolor="white", lw=0.6)
        zero(ax, "x")
        ax.axvline(per[col].mean(), color=INK, lw=1, ls=(0, (3, 2)))
        ax.set_xlabel(lab)
        ax.set_ylabel("runs")
        ax.set_title("Δ log-prob" if col == "dlpd" else "Δ P(prefer overt)")
        if j == 0:
            ax.legend(loc="upper right", fontsize=7)
    ax = fig.add_subplot(gs[1, :])
    a = tagg(tr)
    band(ax, a, "dlpd", INK, T, label="Δ log-prob (nats/token)")
    zero(ax)
    tokens_axis(ax)
    ax.set_ylabel(EFF)
    ax.set_title("The same effect over training (mean of 140 runs, 95% CI)")
    ax.annotate("init: 0", (0.5, 0), xytext=(0.6, 0.012), size=7.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    doc.save(fig)


def p_relation(doc, S, e):
    rel = S["rel"]
    fig, box = doc.page(
        "3 · What is primed", "Most of the effect is the pronoun repeating",
        "The same pooled effect, split by how the prime's overt element relates to the target's. Repetition of the very word "
        "dominates; a different pronoun still helps, a non-pronoun element does not.",
        notes=[f"Same word: {rel['same word']['share']:.1%} of pairings, {rel['same word']['contrib']:.0%} of the pooled effect.",
               f"Different pronoun: {rel['other pronoun']['share']:.0%} of pairings, {rel['other pronoun']['contrib']:.0%} of the effect "
               f"at a sixth of the per-pairing size (+{rel['other pronoun']['dlpd']:.3f}).",
               f"Same features (him → he): +{rel['same features']['dlpd']:.3f}, twice a different pronoun. Shared person/number/gender "
               "carries over across case.",
               f"Non-pronoun ('that' ↔ pronoun): {rel['non-pronoun']['dlpd']:+.4f}. One extra word in the prime does not by itself "
               "push the model toward overt targets; this is the design's length control.",
               "Item-resampled intervals (thin) are much wider than run intervals (thick): the 4,608 pairings are a modest "
               "sample of possible stimuli, and generalization to new items should be argued from them."])
    box = {**box, "left": 0.2}
    gs = fig.add_gridspec(2, 1, **box, hspace=0.35, height_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0])
    short = {"same word": "same word\n(she → she)", "same features": "same features\n(him → he)",
             "other pronoun": "different pronoun\n(it → she)", "non-pronoun": "non-pronoun\n(that → she)"}
    m = [rel[r]["dlpd"] for r in REL]
    pos = np.arange(len(REL))
    for p_, r in zip(pos, REL):
        lo, hi = rel[r]["boot"]
        ax.plot([lo, hi], [p_, p_], color=INK2, lw=0.8, zorder=3)
    bars(ax, [short[r] for r in REL], m, [rel[r]["ci"] for r in REL], [REL_C[r] for r in REL], horizontal=True)
    zero(ax, "x")
    ax.set_xlabel(EFF)
    ax.set_title("Effect per pairing (capped bar: runs CI; thin line: items CI)")
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    hdr = ["relation", "share of\npairings", "Δ P(prefer\novert)", "Δ log-prob\n(nats/token)", "share of\npooled effect"]
    cells = [[r, f"{rel[r]['share']:.1%}", f"{rel[r]['dpref']:+.3f}", f"{rel[r]['dlpd']:+.3f}", f"{rel[r]['contrib']:+.0%}"] for r in REL]
    tb = ax2.table(cellText=cells, colLabels=hdr, loc="upper left", cellLoc="right", colLoc="right",
                   colWidths=[0.28, 0.18, 0.18, 0.18, 0.18])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.4)
    tb.scale(1, 1.7)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_linewidth(0.6)
        c.get_text().set_fontfamily(HEAD)
        if i == 0:
            c.get_text().set_color(MUTED)
            c.get_text().set_fontsize(7.4)
            c.set_height(c.get_height() * 1.6)
        if j == 0:
            c._loc = "left"
            c.get_text().set_horizontalalignment("left")
    ax2.set_title("Composition of the pooled effect", loc="left")
    doc.save(fig)


def p_words(doc, S, e):
    w = e.groupby(["run", "prime_word", "target_word"]).dlpd.mean().groupby(["prime_word", "target_word"]).mean().unstack()
    w = w.reindex(index=WORDS, columns=WORDS)
    n = e.groupby(["prime_word", "target_word"]).size().unstack().reindex(index=WORDS, columns=WORDS) / S["n_runs"]
    w = w.where(n >= 5)
    lim = 0.35
    fig, box = doc.page(
        "3 · What is primed", "Which word primes which",
        "Mean priming effect for each pair of overt elements (rows: the word the overt prime adds; columns: the target's "
        "pronoun). Cells with fewer than 5 pairings per run are blank. The colour scale is capped at ±0.35.",
        notes=[f"The diagonal dominates. 'i' → 'i' is {w.at['i', 'i']:.2f} nats/token: the pronoun-free contexts never "
               "mention the speaker, so a first-person prime is the only evidence that a speaker exists.",
               f"First and second person and 'they' form a facilitating block (e.g. i → we {w.at['i', 'we']:+.2f}, "
               f"you → we {w.at['you', 'we']:+.2f}, they → you {w.at['they', 'you']:+.2f}): discourse-participant / plural "
               "pronouns license one another.",
               f"he and she compete: he → she {w.at['he', 'she']:+.3f}, she → he {w.at['she', 'he']:+.3f}. A prime about one "
               "3sg referent makes an overt pronoun of the other gender less likely, not more.",
               f"Case is not a wall: him → he {w.at['him', 'he']:+.3f}, he → him {w.at['he', 'him']:+.3f}.",
               f"'it' is a mild general facilitator for subject pronouns, and 'that' primes only 'that' "
               f"({w.at['that', 'that']:+.2f})."])
    gs = fig.add_gridspec(1, 1, **box)
    ax = fig.add_subplot(gs[0])
    im = heat(ax, w.to_numpy(), WORDS, WORDS, lim, fmt="{:+.2f}", size=7.3)
    ax.set_xlabel("target's overt pronoun")
    ax.set_ylabel("prime's overt element")
    for i in range(len(WORDS)):
        ax.add_patch(doc.plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(EFF, size=7.5)
    cb.outline.set_visible(False)
    doc.save(fig)


def p_position(doc, S, e):
    x = e[(e.prime_word == "it") & (e.target_word != "it")].copy()
    x["tpos"] = np.where(x.target_category == "object_drop", "object targets ('them')", "subject targets (he/she/I/we/you/they)")
    x = x[x.target_category.isin(["subject_drop", "subject_drop_no_agreement", "conjunction", "embedded_drop", "object_drop"])]
    src = {"expletive": "expletive\nsubject 'it'", "subject_drop": "referential\nsubject 'it'",
           "subject_drop_no_agreement": "referential\nsubject 'it'", "object_drop": "object 'it'"}
    x["src"] = x.prime_category.map(src)
    g = x.groupby(["run", "src", "tpos"]).dlpd.mean().reset_index()
    srcs = list(dict.fromkeys(src.values()))
    tposs = ["subject targets (he/she/I/we/you/they)", "object targets ('them')"]
    # target he / she panel
    y = e[e.target_word.isin(["he", "she"])].copy()
    def rel_hs(r):
        if r.prime_word == r.target_word:
            return "same pronoun"
        if r.prime_word in ("him", "her"):
            same = P.PRON[r.prime_word][2] == P.PRON[r.target_word][2]
            return "same gender, object\ncase (him → he)" if same else "other gender, object\ncase (him → she)"
        if r.prime_word in ("he", "she"):
            return "other gender,\nsubject (she → he)"
        if r.prime_word == "it":
            return "it, subject position" if r.prime_category != "object_drop" else "it, object position"
        if r.prime_word in ("i", "we", "you", "they", "them"):
            return "1st / 2nd person,\nplural"
        return "that"
    y["kind"] = [rel_hs(r) for r in y[["prime_word", "target_word", "prime_category"]].itertuples(index=False)]
    order = ["same pronoun", "same gender, object\ncase (him → he)", "it, subject position", "1st / 2nd person,\nplural",
             "that", "it, object position", "other gender, object\ncase (him → she)", "other gender,\nsubject (she → he)"]
    gy = y.groupby(["run", "kind"]).dlpd.mean().reset_index()
    fig, box = doc.page(
        "3 · What is primed", "Under the word: grammatical function",
        "Holding the word fixed separates position from identity. The same word 'it' primes overt subjects only when it "
        "sits in subject position, and primes overt objects only from object position.",
        notes=[f"Expletive 'it' → other subject pronouns: +{S['it_expl_subj']:.3f}. It is non-referential, so this is not "
               "about a referent: an overt subject of any kind makes the next subject more likely to be overt.",
               f"Object 'it' → subject pronouns: {S['it_obj_subj']:+.3f}, essentially zero; object 'it' → overt object "
               f"'them': +{S['it_obj_obj']:.3f}.",
               "This is the cleanest evidence in the set for structural rather than lexical priming. It is also the "
               "classic pro-drop link: in null-subject languages expletive and referential subjects drop together.",
               "Right panel: for he/she targets the same pronoun helps most, the same gender in object case next, subject 'it' "
               "and other persons a little; object 'it' does nothing, and the other gender hurts in either case."])
    box = {**box, "left": 0.09}
    gs = fig.add_gridspec(1, 2, **box, wspace=0.55, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0])
    wdt = 0.36
    for k, (tp, col) in enumerate(zip(tposs, [BLUE, ORANGE])):
        mm = [g[(g.src == s) & (g.tpos == tp)].dlpd.mean() for s in srcs]
        ee = [run_ci(g[(g.src == s) & (g.tpos == tp)].dlpd) for s in srcs]
        ax.bar(np.arange(len(srcs)) + (k - 0.5) * wdt, mm, wdt * 0.92, yerr=ee, color=col, label=tp,
               error_kw={"lw": 0.8, "ecolor": INK2, "capsize": 2})
    ax.set_xticks(range(len(srcs)), srcs, size=7.6)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax.grid(axis="x", visible=False)
    zero(ax)
    ax.set_ylabel(EFF)
    ax.set_title("Prime 'it', by where it sits")
    ax.legend(loc="upper left", fontsize=7.2)
    ax2 = fig.add_subplot(gs[1])
    mm = [gy[gy.kind == k].dlpd.mean() for k in order]
    ee = [run_ci(gy[gy.kind == k].dlpd) for k in order]
    cols = [BLUE, ORANGE, AQUA, AQUA, CTRL, CTRL, VIOLET, VIOLET]
    bars(ax2, order, mm, ee, cols, horizontal=True)
    zero(ax2, "x")
    ax2.set_xlabel(EFF)
    ax2.set_title("Targets with 'he' or 'she': what helps")
    doc.save(fig)


def p_person(doc, S, e):
    s = e[e.target_category.isin(R.SUBJ) & e.prime_category.isin(R.SUBJ)]
    M = s.groupby(["run", "target_condition", "prime_condition"]).dlpd.mean().groupby(["target_condition", "prime_condition"]).mean().unstack()
    M = M.reindex(index=R.PERSONS, columns=R.PERSONS)
    Mp = s.groupby(["run", "target_condition", "prime_condition"]).dpref.mean().groupby(["target_condition", "prime_condition"]).mean().unstack()
    Mp = Mp.reindex(index=R.PERSONS, columns=R.PERSONS)
    lab = [c.replace("subj_", "") for c in R.PERSONS]
    fig, box = doc.page(
        "3 · What is primed", "Person and number inside subject drop",
        "Subject-drop targets (both agreement types) after subject-drop primes, by the person/number of each. 2sg and 2pl "
        "share 'you', so their 2×2 block is all same-word.",
        notes=[f"The diagonal again: 1sg {M.at['subj_1sg', 'subj_1sg']:.2f}, 2pl/2sg block {M.loc[['subj_2sg', 'subj_2pl'], ['subj_2sg', 'subj_2pl']].to_numpy().mean():.2f}, "
               f"1pl {M.at['subj_1pl', 'subj_1pl']:.2f}, 3pl {M.at['subj_3pl', 'subj_3pl']:.2f}, 3sg {M.at['subj_3sg', 'subj_3sg']:.2f}. "
               "3sg is lowest because it mixes he, she and it.",
               "Off the diagonal every cell is positive: within subject drop, any overt subject pronoun in the prime "
               "raises the overt target. The 1st/2nd-person block is the warmest off-diagonal region.",
               "Null primes from 1st/2nd-person items read as imperatives in pronoun-free contexts, so these columns may "
               "contrast 'overt subject' with 'imperative' rather than with 'dropped subject'. The 3sg/3pl columns are the "
               "clean drop contrast.",
               "Right: the binary measure tells the same story at smaller scale."])
    gs = fig.add_gridspec(1, 2, **box, wspace=0.35)
    for j, (A, lim, f, t) in enumerate([(M, 0.5, "{:+.2f}", "Δ log-prob (nats/token)"), (Mp, 0.3, "{:+.2f}", "Δ P(prefer overt)")]):
        ax = fig.add_subplot(gs[j])
        heat(ax, A.to_numpy(), lab, [x + ("†" if x[0] in "12" else "") for x in lab], lim, fmt=f, size=7.4)
        ax.set_xlabel("prime person/number (†: null prime imperative-like)")
        ax.set_ylabel("target person/number")
        ax.set_title(t)
    doc.save(fig)


def p_constructions(doc, S, e):
    def mat(d, col="dlpd"):
        g = d.groupby(["run", "target_category", "prime_category"])[col].mean().groupby(["target_category", "prime_category"]).mean().unstack()
        return g.reindex(index=CATS, columns=CATS)
    A = mat(e)
    abstract = e[e.relation.isin(["other pronoun", "non-pronoun"])]
    B = mat(abstract)
    nB = abstract.groupby(["target_category", "prime_category"]).size().unstack().reindex(index=CATS, columns=CATS) / S["n_runs"]
    B = B.where(nB >= 5)
    lab = [SHORT[c] for c in CATS]
    fig, box = doc.page(
        "4 · Across constructions", "The construction × construction map",
        "Left: every pairing. Right: only pairings whose overt elements are different words with different features, which "
        "removes repetition and leaves the cross-construction part. Blank cells have no such pairings.",
        )
    notes = ["Left, the diagonal is bright because same-construction primes usually repeat the pronoun.",
               f"Right, a subject block survives: subject drop ↔ expletive in both directions (e.g. expletive → subj drop "
               f"{B.at['subject_drop', 'expletive']:+.3f}, subj drop → expletive {B.at['expletive', 'subject_drop']:+.3f}).",
               f"Object drop primes object drop without word repetition ({B.at['object_drop', 'object_drop']:+.3f}), and "
               f"conjunction primes conjunction ({B.at['conjunction', 'conjunction']:+.3f}).",
               f"Embedded-drop targets turn negative after conjunction, control and embedded primes "
               f"({B.at['embedded_drop', 'conjunction']:+.3f}, {B.at['embedded_drop', 'control']:+.3f}, {B.at['embedded_drop', 'embedded_drop']:+.3f}): "
               "these are mostly he ↔ she pairs, the gender competition of the previous pages.",
               "Extraction ('that') and control sit apart: they neither prime nor are primed across constructions."]
    box = {**box, "left": 0.13, "right": 0.95, "bottom": 0.385}
    gs = fig.add_gridspec(1, 2, **box, wspace=0.32)
    for j, (M, t) in enumerate([(A, "All pairings"), (B, "Different words only")]):
        ax = fig.add_subplot(gs[j])
        heat(ax, M.to_numpy(), lab, lab, 0.2, fmt="{:+.2f}", size=6.6)
        ax.set_xticklabels(lab, rotation=40, ha="right", size=7.2)
        ax.set_yticklabels(lab, size=7.2)
        ax.set_xlabel("prime construction")
        if j == 0:
            ax.set_ylabel("target construction")
        ax.set_title(t)
    doc.bottom_notes(fig, notes, top=0.235, ncol=3)
    doc.save(fig)


def p_potency(doc, S, e):
    ab = e[e.relation.isin(["other pronoun", "non-pronoun"])]
    pot = ab.groupby(["run", "prime_category"]).dlpd.mean().reset_index()
    sus = ab.groupby(["run", "target_category"]).dlpd.mean().reset_index()
    cell = ab.groupby(["run", "target_category", "prime_category"]).dlpd.mean().groupby(["target_category", "prime_category"]).mean()
    pairs = []
    for i, a in enumerate(CATS):
        for b in CATS[i + 1:]:
            if (a, b) in cell.index and (b, a) in cell.index:
                pairs.append((a, b, cell[(b, a)], cell[(a, b)]))   # (a primes b), (b primes a)
    fig, box = doc.page(
        "4 · Across constructions", "Which constructions prime, which get primed",
        "Different-word pairings only. Potency = a prime construction's mean effect over all targets; susceptibility = a "
        "target construction's mean effect over all primes. Right: each construction pair in both directions.",
        notes=["Subject drop and expletive primes are the potent ones; object drop and conjunction primes are weaker, and "
               "control and embedded primes push the wrong way (they mostly carry 'him' / 'he' against she-targets).",
               "Susceptibility splits the same way: subject-drop and expletive targets move, embedded targets move backwards.",
               "Points near the diagonal are symmetric pairs: when A primes B, B primes A about as much. The subject "
               "cluster (subject drop ↔ expletive, the two subject-drop sets) sits at the top right; the embedded pairs "
               "sit at the bottom left.",
               "Potency and susceptibility here are averages over a construction's prime words, so they inherit the "
               "word-level structure of section 3; treat them as summaries, not as construction properties."])
    box = {**box, "left": 0.15}
    gs = fig.add_gridspec(2, 2, **box, wspace=0.45, hspace=0.45, width_ratios=[1, 1.05])
    for k, (d, col, t) in enumerate([(pot, "prime_category", "Prime potency"), (sus, "target_category", "Target susceptibility")]):
        ax = fig.add_subplot(gs[k, 0])
        order = d.groupby(col).dlpd.mean().reindex(CATS).sort_values(ascending=False)
        bars(ax, [SHORT[c] for c in order.index], order.to_numpy(),
             [run_ci(d[d[col] == c].dlpd) for c in order.index], [BLUE] * len(order), horizontal=True)
        zero(ax, "x")
        ax.set_title(t)
        ax.tick_params(axis="y", labelsize=7.4)
        if k == 1:
            ax.set_xlabel(EFF)
    ax = fig.add_subplot(gs[:, 1])
    hot = sorted([q for q in pairs if max(abs(q[2]), abs(q[3])) > 0.035], key=lambda q: -q[3])
    for a, b, ab_, ba in pairs:
        ax.scatter(ab_, ba, s=34, color=BLUE, edgecolor="white", lw=1, zorder=3)
    for i, (a, b, ab_, ba) in enumerate(hot):
        ax.annotate(f"{SHORT[a]} ↔ {SHORT[b]}", (ab_, ba), xytext=(-60, -14 - 13 * i) if ba > 0 else (8, 0),
                    textcoords="offset points", size=6.6, color=INK2, ha="right" if ba > 0 else "left", va="center",
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.6) if ba > 0 else None)
    lim = max(0.05, max(max(abs(p[2]), abs(p[3])) for p in pairs) * 1.15)
    ax.plot([-lim, lim], [-lim, lim], color=RULE, lw=0.8)
    zero(ax)
    zero(ax, "x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("A primes B")
    ax.set_ylabel("B primes A")
    ax.set_title("Symmetry of construction pairs")
    doc.save(fig)


def p_ceiling(doc, S, e):
    g = e.groupby(["run", "target_condition", "target_category"])[["pref_u", "dpref", "dlpd"]].mean().reset_index()
    g = g.groupby(["target_condition", "target_category"])[["pref_u", "dpref", "dlpd"]].mean().reset_index()
    fig, box = doc.page(
        "4 · Across constructions", "Ceilings: why the binary measure undersells some targets",
        "Each point is a target condition: its unprimed overt preference against its priming effect. The binary measure can "
        "only move items near the overt/null boundary, so targets near 0 or 1 look unprimed on it.",
        notes=["Left: binary effects peak in the middle of the preference range and vanish near the ends, as a ceiling would "
               "produce.",
               "Right: the continuous measure separates real insensitivity from ceiling. Object-drop targets (near 1.0) are "
               "still primed in log-prob; embedded-drop targets (≈0.9) are genuinely flat.",
               "Control targets sit near the floor (the model prefers the bare infinitive) and move little on either measure.",
               "This is why the decomposition pages use Δ log-prob: it is not truncated at the extremes."])
    gs = fig.add_gridspec(1, 2, **box, wspace=0.3)
    cat_c = {c: (BLUE if c in SUBJC else ORANGE if c in ("conjunction", "embedded_drop") else AQUA if c == "object_drop" else CTRL) for c in CATS}
    for j, (col, lab) in enumerate([("dpref", EFFP), ("dlpd", EFF)]):
        ax = fig.add_subplot(gs[j])
        for r in g.itertuples(index=False):
            ax.scatter(r.pref_u, getattr(r, col), s=40, color=cat_c[r.target_category], edgecolor="white", lw=1, zorder=3)
            ax.annotate(r.target_condition.replace("subj_", ""), (r.pref_u, getattr(r, col)), xytext=(4, 2),
                        textcoords="offset points", size=6.6, color=INK2)
        zero(ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("unprimed P(prefer overt), end-state")
        ax.set_ylabel(lab)
        ax.set_title(["Binary: Δ P(prefer overt)", "Continuous: Δ log-prob"][j])
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=c, label=l, markersize=6) for c, l in
          [(BLUE, "subject drop / expletive"), (ORANGE, "conjunction / embedded"), (AQUA, "object drop"), (CTRL, "control / extraction")]]
    fig.axes[-1].legend(handles=hs, loc="lower left", fontsize=7)
    doc.save(fig)


def p_levels(doc, S, e):
    g = e.groupby(["run", "target_category"])[["pref_u", "pref_n", "pref_o"]].mean().reset_index()
    g["mid"] = (g.pref_o + g.pref_n) / 2 - g.pref_u
    fig, box = doc.page(
        "4 · Across constructions", "Levels, and why the direction is not identified",
        "Left: each target's overt preference unprimed, after a null prime and after an overt prime. Right: how far the "
        "primed pair as a whole sits from unprimed — the effect of adding a whole prime discourse, before the prime's "
        "overt/null status matters.",
        notes=["The overt − null gap (the priming effect) is small next to the shift that any prime discourse causes, and "
               "that shift has a different sign per construction: up for subject drop and expletives (+0.09 to +0.12), down "
               "for extraction (−0.09) and embedded (−0.04), near zero elsewhere.",
               "So 'after an overt prime vs unprimed' and 'after a null prime vs unprimed' cannot be read as the two "
               "directions of priming; both are dominated by the discourse shift.",
               "The next two pages get part of the way with a reference prime instead. The neutral root (prime context "
               "with no prime sentence, 4,608 pairs, built and unscored) would settle it; scoring only the end-state "
               "checkpoints (3 per run) is a small job."])
    box = {**box, "bottom": 0.16}
    gs = fig.add_gridspec(1, 2, **box, wspace=0.45, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(CATS))
    for k, (col, c, lab) in enumerate([("pref_u", CTRL, "unprimed"), ("pref_n", ORANGE, "after null prime"), ("pref_o", BLUE, "after overt prime")]):
        m = g.groupby("target_category")[col].mean().reindex(CATS)
        er = g.groupby("target_category")[col].apply(run_ci).reindex(CATS)
        ax.bar(x + (k - 1) * 0.27, m, 0.25, yerr=er, color=c, label=lab, error_kw={"lw": 0.7, "ecolor": INK2, "capsize": 1.5})
    ax.set_xticks(x, [SHORT[c] for c in CATS], rotation=30, ha="right", size=7.6)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("end-state P(prefer overt)")
    ax.legend(loc="upper left", ncol=3, fontsize=7.2, bbox_to_anchor=(0, 1.08))
    ax2 = fig.add_subplot(gs[1])
    m = g.groupby("target_category").mid.mean().reindex(CATS)
    er = g.groupby("target_category").mid.apply(run_ci).reindex(CATS)
    bars(ax2, [SHORT[c] for c in CATS], m.to_numpy(), er.to_numpy(), [BLUE if v > 0 else ORANGE for v in m], horizontal=True, fmt="{:+.3f}")
    zero(ax2, "x")
    ax2.set_xlabel("primed midpoint − unprimed, Δ P(prefer overt)")
    ax2.set_title("Shift from adding a prime discourse")
    doc.save(fig)


def p_traj_overview(doc, S, tr, T):
    a = tagg(tr)
    mu = a[a.idx > 0].groupby("idx")[["pref_u", "dpref", "dlpd"]].mean()
    pk = mu.pref_u.idxmax()
    dp_pk = mu.dpref.idxmax()
    fig, box = doc.page(
        "5 · Training dynamics", "Priming over training",
        "Top: the target's overt preference unprimed, after a null prime and after an overt prime, through training. Bottom: "
        "the priming effect itself on both measures. Mean of 140 runs with 95% CI; x is tokens on a log scale.",
        notes=[f"Nothing before 16M tokens. The effect rises from ~30M, reaches half its final size by "
               f"{fmt_tok(S['timing']['all']['t50'])} and 90% by {fmt_tok(S['timing']['all']['t90'])}, then holds for the "
               "remaining ~3B tokens.",
               f"Overt preference itself peaks at {mu.pref_u.max():.2f} around {fmt_tok(T[pk])} and then falls (the removal "
               f"conditions take hold). The priming gap opens on the way down, between ~250M and ~600M tokens.",
               f"On the binary measure the effect peaks at {mu.dpref.max():+.3f} near {fmt_tok(T[dp_pk])} and eases to "
               f"{mu.dpref.iloc[-3:].mean():+.3f}; in log-prob it holds. Fewer items sit near the decision boundary late in "
               "training, so the same shift flips fewer of them.",
               "Unprimed drifts below both primed curves late in training: the extra discourse itself raises overt "
               f"preference for subject-drop targets (page {doc.ref['levels']}).",
               "Checkpoints double in token count up to 262M (16M, 33M, 66M, 131M, 262M) and are dense after, so onset "
               "timing below 262M has factor-of-two resolution."])
    gs = fig.add_gridspec(2, 2, **box, hspace=0.5, wspace=0.3, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    for col, c, lab in [("pref_u", CTRL, "unprimed"), ("pref_n", ORANGE, "after null prime"), ("pref_o", BLUE, "after overt prime")]:
        band(ax, a, col, c, T, label=lab)
    tokens_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(prefer overt)")
    ax.legend(loc="lower right", ncol=3)
    ax.set_title("Overt preference, three ways")
    for j, (col, lab, t) in enumerate([("dlpd", EFF, "Effect, Δ log-prob"), ("dpref", EFFP, "Effect, Δ P(prefer overt)")]):
        ax = fig.add_subplot(gs[1, j])
        band(ax, a, col, INK, T)
        zero(ax)
        tokens_axis(ax)
        ax.set_ylabel(lab, size=7.6)
        ax.set_title(t)
    doc.save(fig)


def p_traj_relation(doc, S, tr, T):
    Mt = masks(tr)
    curves = {r: tagg(tr, Mt[r]) for r in REL}
    tm = S["timing"]
    fig, box = doc.page(
        "5 · Training dynamics", "Lexical priming first, structure second",
        "Left: the effect for each prime–target relation over training. Right: the lexical and the two structural components "
        "rescaled to their own end-state, so their timing can be compared directly.",
        notes=[f"Same word: half size by {fmt_tok(tm['same word']['t50'])} (95% CI {fmt_tok(tm['same word']['t50_ci'][0])}–"
               f"{fmt_tok(tm['same word']['t50_ci'][1])}), 90% by {fmt_tok(tm['same word']['t90'])}. It overshoots and settles "
               "about 10% lower.",
               f"Subject cluster without repetition (subject drop / expletive ↔ each other, different pronouns): half by "
               f"{fmt_tok(tm['subject cluster']['t50'])} ({fmt_tok(tm['subject cluster']['t50_ci'][0])}–"
               f"{fmt_tok(tm['subject cluster']['t50_ci'][1])}), 90% by {fmt_tok(tm['subject cluster']['t90'])}. It lags the "
               "lexical effect and approaches its end-state more slowly, without overshooting.",
               f"Object cluster: half by {fmt_tok(tm['object cluster']['t50'])}; it overshoots near 450M, dips to ~80% and "
               "recovers by ~1B.",
               "The order fits copying (induction-style) mechanisms forming first, with priming of grammatical function "
               "building on top. It is timing evidence only; it does not show one depends on the other."])
    gs = fig.add_gridspec(1, 2, **box, wspace=0.3)
    ax = fig.add_subplot(gs[0])
    for r in REL:
        band(ax, curves[r], "dlpd", REL_C[r], T, label=REL_LABEL[r].split("  ")[0])
    zero(ax)
    tokens_axis(ax)
    ax.set_ylabel(EFF)
    ax.legend(loc="upper left", fontsize=7.2)
    ax.set_title("By relation")
    ax2 = fig.add_subplot(gs[1])
    for r, c, ls in [("same word", BLUE, "-"), ("subject cluster", AQUA, "-"), ("object cluster", ORANGE, "--")]:
        d = tagg(tr, Mt[r])
        fin = d[d.idx >= d.idx.max() - 2].dlpd.mean()
        d = d.assign(norm=d.dlpd / fin)
        band(ax2, d, "norm", c, T, label=f"{r} (end-state +{fin:.3f})", ls=ls, alpha=0.1)
    ax2.axhline(1, color=RULE, lw=0.8)
    ax2.axhline(0.5, color=RULE, lw=0.8, ls=":")
    zero(ax2)
    tokens_axis(ax2)
    ax2.set_ylim(-0.15, 1.35)
    ax2.set_ylabel("fraction of own end-state effect")
    ax2.legend(loc="upper left", fontsize=7.2)
    ax2.set_title("Timing, rescaled")
    doc.save(fig)


def p_traj_targets(doc, S, tr, T):
    rel = tr.relation.astype(str)
    tcat = tr.target_category.astype(str)
    over = {}
    for c in CATS:
        msk = (tcat == c) & (rel == "same word")
        if msk.any():
            mu = tagg(tr, msk).groupby("idx").dlpd.mean()
            over[c] = (mu.max(), mu.iloc[-3:].mean())
    fall = [c for c, (pk, en) in over.items() if en < 0.7 * pk]
    fig, box = doc.page(
        "5 · Training dynamics", "Each target construction over training",
        "Priming effect per target construction, split into same-word pairings and all other pairings. Same y-scale "
        "throughout; n = pairings per run in each split.",
        notes=["Subject-drop targets carry the largest same-word curves (I/we/you repeat) and also the largest "
               "different-word curves.",
               "Same-word priming overshoots and then falls back by a third or more for "
               + ", ".join(f"{SHORT[c]} ({over[c][0]:.2f} → {over[c][1]:.2f})" for c in fall)
               + "; for control and extraction it keeps rising to the end. Repetition is not one mechanism with one "
               "time course.",
               "Embedded targets turn negative for different words late in training: the he/she competition grows "
               "rather than fades.",
               "Extraction targets are primed only by 'that' (same word), i.e. only from their own construction."])
    gs = fig.add_gridspec(2, 4, **box, hspace=0.55, wspace=0.28)
    lim = None
    axes = []
    for k, c in enumerate(CATS):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        axes.append(ax)
        for sel, col, lab in [(rel == "same word", BLUE, "same word"), (rel != "same word", AQUA, "other")]:
            msk = (tcat == c) & sel
            if msk.sum() == 0:
                continue
            d = tagg(tr, msk)
            npr = int(tr[msk & (tr.idx == 96)].n.sum() / tr.run.nunique())
            band(ax, d, "dlpd", col, T, label=f"{lab} (n={npr})", lw=1.6)
        zero(ax)
        tokens_axis(ax, label=k >= 4)
        ax.set_title(SHORT[c], size=9)
        ax.legend(loc="upper left", fontsize=6.4)
        if k % 4 == 0:
            ax.set_ylabel("Δ log-prob", size=7.6)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    doc.save(fig)


def p_traj_rearing(doc, S, tr, T, key):
    a = tagg(tr)
    groups = ARMS if key == "arm" else R.IVS
    col_of = ARM_C if key == "arm" else IV_C
    lab_of = ARM_LABEL if key == "arm" else R.IV_LABEL
    if key == "arm":
        title, lede = ("Selection arm over training",
                       "Priming effect and unprimed overt preference over training, by selection arm (all removal levels and "
                       "interventions pooled within arm).")
        notes = ["The arms separate as soon as the effect forms (~300M) and keep their order to the end: random lowest, "
                 "both recoverability-ordered arms about twice as high, all-removed highest.",
                 "They separate in the same window (~300–600M) in which their overt preferences diverge: priming is set "
                 "while the model is learning how often, and where, subjects are dropped.",
                 "Unprimed preference orders the arms differently (most-recoverable-first highest, least-recoverable-first "
                 "low), so the priming order is not simply the inverse of the preference order.",
                 "There is no 0%-removal run in the cohort, so how primeable a model trained on intact English is remains "
                 "unmeasured; low-level random removal is the nearest proxy."]
    else:
        title, lede = ("Intervention over training",
                       "Priming effect and unprimed overt preference over training, by morphosyntactic intervention (all "
                       "arms and levels pooled within intervention).")
        notes = ["Lemmatized verbs separate from ~500M tokens on: with agreement morphology removed, the verb no longer "
                 "signals the subject, and overt-vs-null decisions lean more on the discourse.",
                 "Enriched verbal morphology is slightly above baseline, impoverished case slightly above, removing "
                 "expletive sentences indistinguishable from baseline.",
                 "Lemmatized verbs also have the lowest unprimed preference; section 7 separates the two."]
    fig, box = doc.page("5 · Training dynamics", title, lede, notes=notes)
    gs = fig.add_gridspec(2, 1, **box, hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    for g_ in groups:
        d = a[a[key] == g_]
        band(ax, d, "dlpd", col_of[g_], T, label=lab_of[g_], alpha=0.12)
        band(ax2, d, "pref_u", col_of[g_], T, label=lab_of[g_], alpha=0.12)
    zero(ax)
    tokens_axis(ax, label=False)
    tokens_axis(ax2)
    ax.set_ylabel(EFF, size=7.6)
    ax2.set_ylabel("unprimed P(prefer overt)")
    ax2.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=2 if key == "arm" else 3, fontsize=7.2)
    ax.set_title("Priming effect")
    ax2.set_title("Unprimed overt preference")
    doc.save(fig)


def p_dose(doc, S, e):
    per = S["per"]
    fig, box = doc.page(
        "6 · Rearing conditions", "Dose: how much is removed matters less than how",
        "End-state priming effect and unprimed preference against the share of subject pronouns removed, per selection "
        "arm (each point = mean of the 5 interventions). The all-removed arm is the single point at 100%.",
        notes=["Random removal: priming stays flat across 10–90% removal even though unprimed preference falls from "
               f"{per[(per.arm == 'rand') & (per.k == 10)].pref_u.mean():.2f} to {per[(per.arm == 'rand') & (per.k == 90)].pref_u.mean():.2f}.",
               "Recoverability-ordered removal: already about twice random at 10% removal, rising mildly with dose for "
               "least-recoverable-first.",
               "So primeability is set by which pronouns are removed, i.e. by whether overt/null in training covaries with "
               "context, more than by how many.",
               "Interpretation (hypothesis): random drop is context-free noise, so a model learns a base rate; ordered drop "
               "makes overtness depend on the context, which is exactly the dependence that priming probes."])
    gs = fig.add_gridspec(1, 2, **box, wspace=0.3)
    for j, (col, lab, ylim) in enumerate([("dlpd", EFF, None), ("pref_u", "unprimed P(prefer overt)", (0, 1))]):
        ax = fig.add_subplot(gs[j])
        for arm in ARMS:
            d = per[per.arm == arm].groupby("k")[col].agg(["mean", run_ci]).reset_index()
            mk = "o" if arm != "all" else "D"
            ax.errorbar(d.k, d["mean"], yerr=d["run_ci"], color=ARM_C[arm], marker=mk, ms=5, lw=2 if arm != "all" else 0,
                        elinewidth=0.8, capsize=2, label=ARM_LABEL[arm], mec="white", mew=1)
        ax.set_xlabel("subject pronouns removed (%)")
        ax.set_xticks([10, 30, 50, 70, 90, 100])
        ax.set_ylabel(lab)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(["Priming effect", "Unprimed overt preference"][j])
        if j == 0:
            ax.legend(loc="upper left", fontsize=7.2)
    doc.save(fig)


def p_scatter(doc, S, e):
    per, fit = S["per"], S["fit"]
    co = fit.params
    ci_ = fit.conf_int()
    rows = [("most recoverable first", 'C(arm, Treatment("rand"))[T.robbi]'),
            ("least recoverable first", 'C(arm, Treatment("rand"))[T.robbianti]'),
            ("impoverish case", "C(iv)[T.impcase]"), ("lemmatize verbs", "C(iv)[T.lemverb]"),
            ("enrich verbal morph", "C(iv)[T.enrichvm]"), ("remove expletives", "C(iv)[T.rmexpl]"),
            ("unprimed P(overt), per +1.0", "pu")]
    fig, box = doc.page(
        "6 · Rearing conditions", "Arm effect is not a preference effect",
        "Each point is a run: unprimed overt preference against priming effect. If primeability were just a by-product of a "
        "weaker overt preference, the arms would fall on one line. They do not.",
        notes=[f"Within each arm the slope is shallow; between arms there is a vertical offset. Fitting "
               f"effect ~ arm + intervention + unprimed preference (126 runs, the all-removed arm excluded):",
               *[f"{lab}: {co[k]:+.4f} [{ci_.loc[k, 0]:+.4f}, {ci_.loc[k, 1]:+.4f}]" for lab, k in rows],
               f"R² = {fit.rsquared:.2f}. Reference: random arm, baseline intervention.",
               "The CIs treat runs as independent and items as fixed; with one seed per cell, arm and level are not "
               "separable from seed variance within a cell."])
    gs = fig.add_gridspec(1, 1, **box)
    ax = fig.add_subplot(gs[0])
    for arm in ARMS:
        d = per[per.arm == arm]
        ax.scatter(d.pref_u, d.dlpd, s=30 if arm != "all" else 46, color=ARM_C[arm], edgecolor="white", lw=1,
                   marker="o" if arm != "all" else "D", label=ARM_LABEL[arm], zorder=3)
        if arm != "all":
            b = np.polyfit(d.pref_u, d.dlpd, 1)
            xs = np.linspace(d.pref_u.min(), d.pref_u.max(), 20)
            ax.plot(xs, np.polyval(b, xs), color=ARM_C[arm], lw=1.4, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("unprimed P(prefer overt), end-state")
    ax.set_ylabel(EFF)
    ax.legend(loc="upper right")
    doc.save(fig)


def p_components(doc, S, e):
    M = masks(e)
    comps = ["same word", "other pronoun", "subject cluster", "object cluster", "he/she competition"]
    comp_c = [BLUE, AQUA, YELLOW, MAGENTA, VIOLET]
    tab = {}
    for c in comps:
        for key in ("arm", "iv"):
            d = e[M[c]].groupby(["run", key]).dlpd.mean().reset_index()
            for g_, v in d.groupby(key).dlpd:
                tab[(c, g_)] = (v.mean(), run_ci(v))
    t = lambda c, g_: tab[(c, g_)][0]  # noqa: E731
    fig, box = doc.page(
        "6 · Rearing conditions", "Which part of priming rearing changes",
        "End-state effect of each component by selection arm (top) and intervention (bottom). Components: repetition, "
        "different pronoun, the two structural clusters, and he/she competition.",
        notes=[f"Ordered removal raises the subject components: subject cluster {t('subject cluster', 'rand'):+.3f} (random) → "
               f"{t('subject cluster', 'robbi'):+.3f} / {t('subject cluster', 'robbianti'):+.3f}, different pronoun "
               f"{t('other pronoun', 'rand'):+.3f} → {t('other pronoun', 'robbi'):+.3f} / {t('other pronoun', 'robbianti'):+.3f}, "
               f"repetition {t('same word', 'rand'):+.3f} → {t('same word', 'robbi'):+.3f} / {t('same word', 'robbianti'):+.3f}.",
               f"Object-cluster priming does not move across arms ({t('object cluster', 'rand'):+.3f} / "
               f"{t('object cluster', 'robbi'):+.3f} / {t('object cluster', 'robbianti'):+.3f}). Only subject pronouns were "
               "removed, so this is a built-in control: the rearing effect is specific to what was manipulated.",
               f"he/she competition is strongest under random removal ({t('he/she competition', 'rand'):+.3f}), weaker under "
               f"ordered removal ({t('he/she competition', 'robbi'):+.3f} / {t('he/she competition', 'robbianti'):+.3f}), and "
               f"positive in the all-removed runs ({t('he/she competition', 'all'):+.3f}, wide CI): with no subject pronouns in "
               "training, any pronoun in context is evidence for one.",
               f"Among interventions, lemmatized verbs raise the subject cluster most ({t('subject cluster', 'lemverb'):+.3f} vs "
               f"{t('subject cluster', 'base'):+.3f} baseline) and weaken competition; remove-expletives looks like baseline."])
    gs = fig.add_gridspec(2, 1, **box, hspace=0.5)
    for r, (key, groups, labs) in enumerate([("arm", ARMS, ARM_LABEL), ("iv", R.IVS, R.IV_LABEL)]):
        ax = fig.add_subplot(gs[r])
        w = 0.8 / len(comps)
        for k, (c, col) in enumerate(zip(comps, comp_c)):
            m = [tab[(c, g_)][0] for g_ in groups]
            er = [tab[(c, g_)][1] for g_ in groups]
            ax.bar(np.arange(len(groups)) + (k - (len(comps) - 1) / 2) * w, m, w * 0.9, yerr=er, color=col,
                   label=c, error_kw={"lw": 0.7, "ecolor": INK2, "capsize": 1.5})
        ax.set_xticks(range(len(groups)), [labs[g_] for g_ in groups], size=8)
        ax.grid(axis="x", visible=False)
        zero(ax)
        ax.set_ylabel(EFF, size=7.4)
        ax.set_title("By selection arm" if key == "arm" else "By intervention")
        if r == 0:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.4)
            ax.legend(loc="upper left", ncol=5, fontsize=7.2)
    doc.save(fig)


def p_grid(doc, S, e):
    per = S["per"]
    G = per.pivot_table(index="iv", columns="arm", values="dlpd").reindex(index=R.IVS, columns=ARMS)
    holds = [iv for iv in R.IVS if G.at[iv, "rand"] < min(G.at[iv, "robbi"], G.at[iv, "robbianti"]) < G.at[iv, "all"]]
    breaks = [R.IV_LABEL[iv] for iv in R.IVS if iv not in holds]
    anti_ge = sum(G.at[iv, "robbianti"] >= G.at[iv, "robbi"] - 1e-9 for iv in R.IVS)
    top = all(G[a].idxmax() == "lemverb" for a in ARMS)
    fig, box = doc.page(
        "6 · Rearing conditions", "Intervention × arm",
        "End-state priming effect for every intervention and selection arm (removal levels pooled; the all-removed arm is one "
        "run per intervention). Values are nats/token (left) and P(prefer overt) (right).",
        notes=[f"Random < both ordered arms in every intervention. Ordered < all-removed in {len(holds)} of 5"
               + (f"; not in {', '.join(breaks)}, whose single all-removed run is lower." if breaks else "."),
               ("The lemmatize-verbs row is highest in every column" if top else "The lemmatize-verbs row is highest in most columns")
               + "; remove-expletives tracks baseline except for its all-removed run.",
               f"Least-recoverable-first ≥ most-recoverable-first in {anti_ge} of 5 rows, while their unprimed preferences "
               f"differ by {S['arm_means'].at['robbi', 'pref_u'] - S['arm_means'].at['robbianti', 'pref_u']:.2f}: the two "
               "ordered arms are similarly primeable despite very different overt rates.",
               "The all-removed column is one run per cell; read it as indicative."])
    box = {**box, "left": 0.15, "bottom": 0.14}
    gs = fig.add_gridspec(1, 2, **box, wspace=0.5)
    for j, (col, lim, f) in enumerate([("dlpd", 0.12, "{:.3f}"), ("dpref", 0.06, "{:.3f}")]):
        ax = fig.add_subplot(gs[j])
        M = per.pivot_table(index="iv", columns="arm", values=col).reindex(index=R.IVS, columns=ARMS)
        from matplotlib.colors import LinearSegmentedColormap
        cm = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#8db6e8", "#1c58a3"])
        im = ax.imshow(M.to_numpy(), cmap=cm, vmin=0, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                v = M.iat[i, k]
                ax.text(k, i, f.format(v), ha="center", va="center", size=8, color="white" if v > 0.6 * lim else INK)
        ax.set_xticks(range(len(ARMS)), ["random", "most rec.\nfirst", "least rec.\nfirst", "all\nremoved"], size=7.6)
        ax.set_yticks(range(len(R.IVS)), [R.IV_LABEL[i].replace("enrich verbal morph", "enrich verb morph") for i in R.IVS], size=7.6)
        ax.set_xticks(np.arange(-0.5, len(ARMS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(R.IVS)), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(["Δ log-prob (nats/token)", "Δ P(prefer overt)"][j])
    doc.save(fig)


def p_reliability(doc, S, e):
    import statsmodels.formula.api as smf
    runs = sorted(e.run.unique())
    h = set(np.random.default_rng(3).permutation(runs)[: len(runs) // 2])
    a = e[e.run.isin(h)].groupby(["cond_dir", "pid"]).dlpd.mean()
    b = e[~e.run.isin(h)].groupby(["cond_dir", "pid"]).dlpd.mean().loc[a.index]
    relc = e.drop_duplicates(["cond_dir", "pid"]).set_index(["cond_dir", "pid"]).relation.loc[a.index]
    it = e.groupby(["cond_dir", "pid", "target_category", "target_item_id", "target_condition", "prime_category",
                    "prime_item_id", "prime_condition", "relation"]).dlpd.mean().reset_index()
    it["tkey"] = it.target_category + "|" + it.target_condition + "|" + it.target_item_id.astype(str)
    it["pkey"] = it.prime_category + "|" + it.prime_condition + "|" + it.prime_item_id.astype(str)
    models = [("relation", "C(relation)"), ("target item", "C(tkey)"), ("prime item", "C(pkey)"),
              ("target + prime item", "C(tkey) + C(pkey)"),
              ("relation + construction pair", "C(relation) + C(target_category) * C(prime_category)")]
    r2 = [(lab, smf.ols(f"dlpd ~ {f}", it).fit()) for lab, f in models]
    fig, box = doc.page(
        "7 · Items", "Item effects are stable; the items decide the pattern",
        "Left: each pairing's effect in one random half of the runs against the other half. Right: how much of the "
        "pairing-to-pairing variation simple descriptions explain (pairing means over runs).",
        notes=[f"Split-half r = {S['split_r']:.3f} (mean of 20 random splits). Which pairings prime strongly is almost "
               "entirely a property of the stimuli, shared by all 140 models.",
               "That is partly because the models are not independent subjects: they share architecture, hyper-parameters "
               "and most of their training text. Run-level error bars are therefore narrow for a structural reason.",
               f"The relation alone explains {r2[0][1].rsquared:.0%}; relation plus construction pair "
               f"{r2[4][1].rsquared:.0%}. Prime identity explains more than target identity "
               f"(adj. R² {r2[2][1].rsquared_adj:.2f} vs {r2[1][1].rsquared_adj:.2f}): what the prime contains matters more "
               "than which target it precedes.",
               "Generalization to new stimuli should be argued from items, not runs."])
    gs = fig.add_gridspec(1, 2, **box, wspace=0.42, width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0])
    for r in REL[::-1]:
        m = (relc == r).to_numpy()
        ax.scatter(a.to_numpy()[m], b.to_numpy()[m], s=5, color=REL_C[r], alpha=0.55, lw=0, label=r, rasterized=True)
    lim = (min(a.min(), b.min()) - 0.05, max(a.max(), b.max()) + 0.05)
    ax.plot(lim, lim, color=RULE, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("pairing effect, half A of runs")
    ax.set_ylabel("pairing effect, half B of runs")
    ax.set_title(f"4,608 pairings × 5 conditions, r = {np.corrcoef(a, b)[0, 1]:.3f}")
    ax.legend(loc="upper left", markerscale=3, fontsize=7)
    ax2 = fig.add_subplot(gs[1])
    labs = [lab for lab, _ in r2]
    ax2.barh(np.arange(len(r2)), [m.rsquared for _, m in r2], color=CTRL, height=0.6, label="R²")
    ax2.barh(np.arange(len(r2)), [max(0, m.rsquared_adj) for _, m in r2], color=BLUE, height=0.6, label="adjusted R²")
    ax2.set_yticks(range(len(r2)), labs, size=7.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("share of pairing variance explained")
    ax2.legend(loc="lower right")
    ax2.set_title("What explains pairing differences")
    doc.save(fig)


def split_prime(meta, cond_dir="baseline"):
    """Per pairing: the prime part (prime context + prime sentence) in its overt and null
    versions, and the target's own context, by stripping the target context (from the
    matched pronoun-free root) off the composite."""
    root = R.REPO / "evaluation/stimuli/null-subj-v2-matched-nopron-v1" / cond_dir / "en"
    src = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(root.glob("*.csv"))], ignore_index=True)
    tctx = src[src.pronoun_status == 1].set_index(["category", "condition", "item_id"]).context
    m = meta[meta.cond_dir == cond_dir].copy()
    m["tctx"] = [tctx.get((a, b, c), "") for a, b, c in zip(m.target_category, m.target_condition, m.target_item_id)]

    def strip(ctx, t):
        return ctx[: len(ctx) - len(t)].strip() if t and ctx.endswith(t) else ctx
    m["prime_o"] = [strip(c, t) for c, t in zip(m.ctx_o, m.tctx)]
    m["prime_n"] = [strip(c, t) for c, t in zip(m.ctx_n, m.tctx)]
    return m.set_index("pid")


def p_examples(doc, S, e, meta):
    ab = e[e.relation.isin(["other pronoun", "non-pronoun"]) & (e.cond_dir == "baseline")]
    g = ab.groupby("pid").dlpd.mean().sort_values()
    mb = split_prime(meta)
    nb = ab.run.nunique()
    fig, box = doc.page(
        "7 · Items", "The strongest and weakest pairings without repetition",
        f"Baseline-condition pairings whose overt elements are different words, ranked by mean effect over the {nb} baseline "
        "runs. [Brackets] mark the word the overt prime adds; the target pair is shown with its overt word bracketed.")
    y = 0.80
    for head, ids in [("STRONGEST FACILITATION (OVERT PRIME → MORE OVERT TARGET)", g.index[::-1][:5]),
                      ("STRONGEST INTERFERENCE (OVERT PRIME → LESS OVERT TARGET)", g.index[:5])]:
        fig.text(0.075, y, head, family=HEAD, size=7.8, weight="demibold", color=MUTED, va="top")
        y -= 0.03
        for pid in ids:
            r = mb.loc[pid]
            fig.text(0.075, y, f"{g[pid]:+.3f}", family=HEAD, size=9, weight="demibold", color=INK, va="top")
            fig.text(0.135, y, f"{SHORT[r.prime_category]} ({r.prime_condition}) → {SHORT[r.target_category]} "
                               f"({r.target_condition})    '{r.prime_word}' → '{r.target_word}'",
                     family=HEAD, size=7.6, color=MUTED, va="top")
            fig.text(0.135, y - 0.019, "prime   " + textwrap.shorten(bracket(r.prime_o, r.prime_n), 140, placeholder=" …"),
                     family=MONO, size=7.2, color=INK, va="top")
            fig.text(0.135, y - 0.036, "then    " + textwrap.shorten(f"{r.tctx}  {bracket(r.target_o, r.target_n)}", 140,
                                                                     placeholder=" …"),
                     family=MONO, size=7.2, color=INK2, va="top")
            y -= 0.058
        y -= 0.012
    fig.text(0.075, 0.075, "\n".join(textwrap.wrap(
        "Reading: the top two facilitating pairs share a sentence template (\"where is X ? NAME V-ed [it] into ...\"), a "
        "structural overlap that the no-shared-lemma filter does not catch. The interfering pairs are other-gender primes "
        "(she → he, him → she): a prime about one referent lowers an overt pronoun for a referent of the other gender.", 150)),
        family=BODY, size=8.6, color=INK2, va="bottom", linespacing=1.35)
    doc.save(fig)


def p_prime_items(doc, S, e, meta):
    ab = e[e.relation.isin(["other pronoun", "non-pronoun"]) & (e.cond_dir == "baseline")]
    ab = ab.assign(pkey=ab.prime_category + "|" + ab.prime_condition + "|" + ab.prime_item_id.astype(str))
    g = ab.groupby("pkey").agg(eff=("dlpd", "mean"), n=("pid", "nunique")).query("n >= 3").sort_values("eff")
    mb = split_prime(meta).reset_index()
    mb["pkey"] = mb.prime_category + "|" + mb.prime_condition + "|" + mb.prime_item_id.astype(str)
    first = mb.drop_duplicates("pkey").set_index("pkey")
    fig, box = doc.page(
        "7 · Items", "Prime items ranked by potency",
        "Each prime item's mean effect over all of its targets that have a different overt word (baseline runs; items used "
        "for at least 3 such targets). The prime is shown in its overt version, [element] bracketed.")
    y = 0.80
    for head, keys in [("MOST POTENT PRIMES", g.index[::-1][:8]), ("INTERFERING PRIMES", g.index[:8])]:
        fig.text(0.075, y, head, family=HEAD, size=7.8, weight="demibold", color=MUTED, va="top")
        y -= 0.028
        for k in keys:
            r = first.loc[k]
            fig.text(0.075, y, f"{g.at[k, 'eff']:+.3f}", family=HEAD, size=8.6, weight="demibold", color=INK, va="top")
            fig.text(0.13, y, f"{SHORT[r.prime_category]} · {r.prime_condition} · '{r.prime_word}' · "
                              f"{int(g.at[k, 'n'])} targets", family=HEAD, size=7.2, color=MUTED, va="top")
            fig.text(0.13, y - 0.017, textwrap.shorten(bracket(r.prime_o, r.prime_n), 150, placeholder=" …"),
                     family=MONO, size=7.1, color=INK, va="top")
            y -= 0.04
        y -= 0.018
    doc.save(fig)


# --- which side does the work (reference-relative) ---------------------------
def p_sides(doc, S, e, side, T, meta):
    se = side_end(side, R.SUBJ)
    groups = [g for g in PGROUPS if g != "object_drop"]
    m = se.groupby("prime_group")[SIDE_VALS].mean()
    c = se.groupby("prime_group")[SIDE_VALS].agg(run_ci)
    ref_eff = e[e.target_category.isin(R.SUBJ) & (e.prime_category == "object_drop")].groupby("run").dlpd.mean().mean()
    tl = {g: emergence(sagg(side, side.target_category.isin(R.SUBJ) & (side.prime_group == g)), T, col="null_lpd")
          for g in ["subj drop 3rd", "expletive"]}
    tl_ov = emergence(sagg(side, side.target_category.isin(R.SUBJ) & (side.prime_group == "expletive")), T, col="overt_lpd")
    mb = split_prime(meta)
    emb = mb[mb.prime_category == "embedded_drop"].drop_duplicates("prime_item_id")
    emb_share = np.mean([bool(set(t.split()) & set(P.PRON)) for t in emb.prime_n])
    v = lambda g, k: m.at[g, k]  # noqa: E731
    fig, box = doc.page(
        "4 · Across constructions", "Which side does the work: null primes vs overt primes",
        "Subject-drop targets. Each prime's null version is compared with the same target after an object-drop prime's null "
        "version, and its overt version with the object-drop prime's overt version. Object-drop primes leave subject targets "
        f"in place (their own overt − null effect here: {ref_eff:+.3f}), so they serve as the reference.",
        notes=[f"Clean 3rd-person subject drop: toward null {v('subj drop 3rd', 'null_lpd'):+.3f}, toward overt "
               f"{v('subj drop 3rd', 'overt_lpd'):+.3f}. Nearly all of its priming effect is the null version making the "
               "target less overt: null-subject priming.",
               f"Expletive: toward null {v('expletive', 'null_lpd'):+.3f}, toward overt {v('expletive', 'overt_lpd'):+.3f}. A "
               "dropped expletive ('seems that ...') also pushes the next subject toward null.",
               f"1st/2nd-person subject drop: both sides ({v('subj drop 1st/2nd', 'null_lpd'):+.3f} / "
               f"{v('subj drop 1st/2nd', 'overt_lpd'):+.3f}). The overt side is mostly I/we/you repeating; the null side may "
               "be imperative priming.",
               f"Conjunction ({v('conjunction', 'null_lpd'):+.3f}), control and extraction do not push toward null. A subject "
               "dropped inside a coordination is ordinary English, and it does not prime null subjects.",
               f"Embedded primes look anti-null ({v('embedded_drop', 'null_lpd'):+.3f}) because {emb_share:.0%} of them keep an "
               "overt matrix subject in both versions ('i believe that ∅ plays the guitar'); that pronoun is what the "
               "target sees.",
               f"Over training the null side forms first: half size by {fmt_tok(tl['subj drop 3rd']['t50'])} (subject drop, "
               f"3rd) and {fmt_tok(tl['expletive']['t50'])} (expletive). The expletive's overt side comes much later "
               f"({fmt_tok(tl_ov['t50'])}) and is still growing at the end.",
               "Assumption: apart from the dropped element, a prime's text affects the target as the reference's text does. "
               f"Page {doc.n + 3} checks it against three other references."])
    gs = fig.add_gridspec(3, 2, **{**box, "left": 0.17}, wspace=0.42, hspace=0.55, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(groups))
    for k, (col, colr, lab) in enumerate([("null_lpd", ORANGE, "toward null"), ("overt_lpd", BLUE, "toward overt")]):
        ax.barh(y + (k - 0.5) * 0.38, [m.at[g, col] for g in groups], 0.36, xerr=[c.at[g, col] for g in groups],
                color=colr, label=lab, error_kw={"lw": 0.8, "ecolor": INK2, "capsize": 2})
    ax.set_yticks(y, [PG_SHORT[g] for g in groups])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    zero(ax, "x")
    ax.set_xlabel("shift of the target, Δ log-prob (nats/token)")
    ax.set_title("End-state, by prime construction")
    ax.legend(loc="upper left", fontsize=7)
    for r, g in enumerate(["subj drop 3rd", "subj drop 1st/2nd", "expletive"]):
        a2 = fig.add_subplot(gs[r, 1])
        d = sagg(side, side.target_category.isin(R.SUBJ) & (side.prime_group == g))
        band(a2, d, "null_lpd", ORANGE, T, label="toward null", lw=1.7)
        band(a2, d, "overt_lpd", BLUE, T, label="toward overt", lw=1.7)
        zero(a2)
        tokens_axis(a2, label=r == 2)
        a2.set_title(PG_SHORT[g] + " primes, over training", size=9)
        a2.set_ylim(-0.04, 0.16)
        if r == 0:
            a2.legend(loc="upper left", fontsize=7)
    doc.save(fig)


def p_sides_all(doc, S, side):
    d = sagg(side, side.idx >= side.idx.max() - 2, ["target_category", "prime_group"])
    d = d.groupby(["run", "target_category", "prime_group"])[SIDE_VALS].mean().reset_index()
    Mn = d.groupby(["target_category", "prime_group"]).null_lpd.mean().unstack().reindex(index=CATS, columns=PGROUPS)
    Mo = d.groupby(["target_category", "prime_group"]).overt_lpd.mean().unstack().reindex(index=CATS, columns=PGROUPS)
    fig, box = doc.page(
        "4 · Across constructions", "Both sides, every construction",
        "The same decomposition for every target. Left: how far each prime's null version pushes the target toward null; "
        "right: how far its overt version pushes it toward overt. Reference: object drop (extraction for object-drop "
        "targets). All pairings, so the overt side includes word repetition; the null side never repeats a word.")
    notes = [f"The null side carries the subject block: subj drop ← subj drop 3rd {Mn.at['subject_drop', 'subj drop 3rd']:+.3f}, "
             f"expletive ← subj drop 3rd {Mn.at['expletive', 'subj drop 3rd']:+.3f}, subj drop ← expletive "
             f"{Mn.at['subject_drop', 'expletive']:+.3f}, subj drop (no agr) ← expletive {Mn.at['subject_drop_no_agreement', 'expletive']:+.3f}.",
             f"Object-drop targets are primed on both sides by object-drop primes (null {Mn.at['object_drop', 'object_drop']:+.3f}, "
             f"overt {Mo.at['object_drop', 'object_drop']:+.3f}).",
             f"Same-construction overt sides are large where the word repeats (control ← control {Mo.at['control', 'control']:+.3f}, "
             f"extraction ← extraction {Mo.at['extraction', 'extraction']:+.3f}); their null sides are small "
             f"({Mn.at['control', 'control']:+.3f}, {Mn.at['extraction', 'extraction']:+.3f}): repetition, not structure.",
             "The embedded column is negative on the null side and positive on the overt side for most targets: the overt "
             "matrix subject both of its versions carry.",
             f"Where a row has opposite signs on the two sides in nearly every column (extraction: null side "
             f"{Mn.loc['extraction'].min():+.2f} to {Mn.loc['extraction'].max():+.2f}, overt side {Mo.loc['extraction'].min():+.2f} "
             f"to {Mo.loc['extraction'].max():+.2f}), the reference's own text shifts that target and the split there "
             "reflects the reference, not the primes. Read the subject rows.",
             "'ref' marks each row's reference cell (zero by construction)."]
    box = {**box, "left": 0.14, "right": 0.95, "bottom": 0.4}
    gs = fig.add_gridspec(1, 2, **box, wspace=0.32)
    cols = [PG_SHORT[g] for g in PGROUPS]
    for j, (M, t) in enumerate([(Mn, "Toward null (null version vs reference)"), (Mo, "Toward overt (overt version vs reference)")]):
        ax = fig.add_subplot(gs[j])
        heat(ax, M.to_numpy(), [SHORT[c] for c in CATS], cols, 0.2, fmt="{:+.2f}", size=6.6, nan_text="ref")
        ax.set_xticklabels(cols, rotation=40, ha="right", size=7.2)
        ax.set_yticklabels([SHORT[c] for c in CATS], size=7.2)
        ax.set_xlabel("prime construction")
        if j == 0:
            ax.set_ylabel("target construction")
        ax.set_title(t)
    doc.bottom_notes(fig, notes, top=0.235, ncol=3)
    doc.save(fig)


def p_sides_robust(doc, S, side):
    se = side_end(side, R.SUBJ)
    refs = ["object_drop", "conjunction", "control", "extraction"]
    groups = ["subj drop 3rd", "expletive", "subj drop 1st/2nd"]
    res = {}
    for g in groups:
        a = se[se.prime_group == g].set_index("run")[SIDE_VALS]
        for r in refs:
            d = a - (se[se.prime_group == r].set_index("run")[SIDE_VALS] if r != "object_drop" else 0)
            res[(g, r)] = (d.null_lpd.mean(), run_ci(d.null_lpd), d.overt_lpd.mean(), run_ci(d.overt_lpd))
    rng = lambda g, k: (min(res[(g, r)][k] for r in refs), max(res[(g, r)][k] for r in refs))  # noqa: E731
    n3, o3 = rng("subj drop 3rd", 0), rng("subj drop 3rd", 2)
    ne, oe = rng("expletive", 0), rng("expletive", 2)
    fig, box = doc.page(
        "4 · Across constructions", "Is the reference to blame?",
        "If a prime's text, apart from its dropped element, simply made targets less overt than the reference's text, that "
        "offset would move the two sides in opposite directions and could fake a null-side effect. Changing the reference "
        "changes the offset. Subject-drop targets, end-state.",
        notes=[f"Clean 3rd-person subject drop under four references with unrelated texts: toward null {n3[0]:+.3f} to "
               f"{n3[1]:+.3f}, toward overt {o3[0]:+.3f} to {o3[1]:+.3f}. The null side dominates under every reference.",
               f"Expletive: toward null {ne[0]:+.3f} to {ne[1]:+.3f}, toward overt {oe[0]:+.3f} to {oe[1]:+.3f}. Both sides "
               "contribute; the null-side lead seen against object drop is partly the reference.",
               "1st/2nd-person subject drop keeps its overt-side lead (I/we/you repetition) under every reference.",
               "What remains possible: a shift shared by all four reference constructions relative to subject-drop primes. "
               "The neutral root (prime context without the prime sentence) is the test that rules it out."])
    gs = fig.add_gridspec(1, 3, **{**box, "left": 0.12}, wspace=0.12)
    lab = {"object_drop": "object drop", "conjunction": "conjunction", "control": "control", "extraction": "extraction"}
    axes = []
    for j, g in enumerate(groups):
        ax = fig.add_subplot(gs[j], sharex=axes[0] if axes else None)
        axes.append(ax)
        for i, r in enumerate(refs):
            nm, nc, om, oc = res[(g, r)]
            ax.plot([nm - nc, nm + nc], [i - 0.12, i - 0.12], color=ORANGE, lw=1.4)
            ax.scatter(nm, i - 0.12, s=42, color=ORANGE, edgecolor="white", lw=1, zorder=3, label="toward null" if i == 0 else None)
            ax.plot([om - oc, om + oc], [i + 0.12, i + 0.12], color=BLUE, lw=1.4)
            ax.scatter(om, i + 0.12, s=42, color=BLUE, edgecolor="white", lw=1, zorder=3, label="toward overt" if i == 0 else None)
        ax.set_yticks(range(len(refs)), [lab[r] for r in refs] if j == 0 else [""] * len(refs))
        ax.set_ylim(len(refs) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        zero(ax, "x")
        ax.set_title(PG_SHORT[g] + " primes", size=9.5)
        ax.set_xlabel("shift, Δ log-prob (nats/token)")
        if j == 0:
            ax.set_ylabel("reference prime construction")
            ax.legend(loc="lower right", fontsize=7)
    axes[0].set_xlim(-0.01, 0.17)
    doc.save(fig)


# --- cross-construction priming over training --------------------------------
KEY_CELLS = [("subject_drop", "expletive"), ("expletive", "subject_drop"),
             ("subject_drop_no_agreement", "subject_drop"), ("subject_drop", "subject_drop_no_agreement"),
             ("object_drop", "object_drop"), ("conjunction", "conjunction"),
             ("embedded_drop", "control"), ("subject_drop", "object_drop")]


def cell_frames(tr):
    dw = tr.relation.astype(str).isin(["other pronoun", "non-pronoun"])
    keys = ["target_category", "prime_category"]
    cw = tagg(tr, dw, keys)
    ca = tagg(tr, None, keys)
    for d in (cw, ca):
        d["target_category"] = d.target_category.astype(str)
        d["prime_category"] = d.prime_category.astype(str)
    return cw, ca


def p_cross_snapshots(doc, S, cw, T):
    snaps = [33, 131, 277, 400, 1000, None]
    last = cw.idx.max()
    mats, labels = [], []
    for t in snaps:
        if t is None:
            sel, lab = cw.idx >= last - 2, f"end ({fmt_tok(T.iloc[-1])})"
        else:
            i = int((T - t).abs().idxmin())
            sel, lab = cw.idx == i, fmt_tok(T[i])
        M = (cw[sel].groupby(["run", "target_category", "prime_category"]).dlpd.mean()
             .groupby(["target_category", "prime_category"]).mean().unstack().reindex(index=CATS, columns=CATS))
        mats.append(M)
        labels.append(lab)
    track = [("expletive", "subject_drop"), ("subject_drop", "expletive"), ("object_drop", "object_drop"),
             ("embedded_drop", "embedded_drop"), ("embedded_drop", "control")]
    series = {k: " → ".join(f"{M.at[k]:+.2f}" for M in mats) for k in track}
    fig, box = doc.page(
        "5 · Training dynamics", "The construction map forming over training",
        "Cross-construction priming (different-word pairings only, as on page "
        f"{doc.ref['constructions']}) at six points in training. Same colour scale throughout; '–' = no such pairings.")
    notes = [f"{SHORT[t]} ← {SHORT[pr]}: {series[(t, pr)]}" for t, pr in track]
    notes = ["Five cells across the six snapshots:"] + notes + [
        "Nothing is organized before 131M. The subject block forms between 277M and 1B, one direction at a time: "
        "expletive ← subj drop is under way by 400M, subj drop ← expletive only after it.",
        "The embedded row changes sign. Early on (131M–430M) embedded targets are primed positively by embedded and "
        "conjunction primes; by the end the same cells are interference. The map is not just sharpening: part of it "
        "reverses."]
    box = {**box, "left": 0.13, "right": 0.88, "bottom": 0.41}
    gs = fig.add_gridspec(2, 3, **box, wspace=0.12, hspace=0.32)
    lab_ = [SHORT[c] for c in CATS]
    im = None
    for k, (M, lab) in enumerate(zip(mats, labels)):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        im = heat(ax, M.to_numpy(), lab_, lab_, 0.15, annotate=False, size=6)
        ax.set_title(lab, size=9)
        if k % 3 == 0:
            ax.set_yticklabels(lab_, size=6.6)
        else:
            ax.set_yticklabels([])
        if k // 3 == 1:
            ax.set_xticklabels(lab_, rotation=50, ha="right", size=6.4)
        else:
            ax.set_xticklabels([])
    cb = fig.colorbar(im, ax=fig.axes[-6:], shrink=0.7, pad=0.02)
    cb.set_label(EFF + "\nrows: target · columns: prime", size=7.2)
    cb.outline.set_visible(False)
    doc.bottom_notes(fig, notes, top=0.25, ncol=3)
    doc.save(fig)


def p_cross_cells(doc, S, cw, ca, T):
    timing = {k: emergence(cw[(cw.target_category == k[0]) & (cw.prime_category == k[1])], T) for k in KEY_CELLS[:7]}
    fig, box = doc.page(
        "5 · Training dynamics", "Key cross-construction links over training",
        "Eight construction pairs (target ← prime). Teal: different-word pairings only; grey: all pairings, i.e. with "
        "repetition. The last panel is a control: object-drop primes on subject-drop targets.",
        notes=[*[f"{SHORT[t]} ← {SHORT[pr]}: half size by {fmt_tok(timing[(t, pr)]['t50'])}, end-state "
                 f"{timing[(t, pr)]['final']:+.3f}." for t, pr in KEY_CELLS[:7]],
               "The subject-cluster links reach half size between 350M and 490M; the object link is earliest. The control "
               "cell (last panel) stays flat throughout.",
               "Repetition (grey above teal) rides on the same time course rather than a separate one.",
               "Two links change character. Object drop ← object drop overshoots near 450M and recovers. Embedded ← "
               "control is mildly positive around 131M, then becomes interference by ~700M."])
    gs = fig.add_gridspec(2, 4, **box, hspace=0.55, wspace=0.3)
    axes = []
    for k, (t, pr) in enumerate(KEY_CELLS):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        axes.append(ax)
        sel = lambda d: d[(d.target_category == t) & (d.prime_category == pr)]  # noqa: E731
        band(ax, sel(ca), "dlpd", CTRL, T, label="all pairings", lw=1.3, alpha=0.1)
        band(ax, sel(cw), "dlpd", AQUA, T, label="different words", lw=1.8)
        zero(ax)
        tokens_axis(ax, label=k >= 4)
        ax.set_title(f"{SHORT[t]}\n← {SHORT[pr]}", size=8.2)
        if k == 0:
            ax.legend(loc="upper left", fontsize=6.6)
        if k % 4 == 0:
            ax.set_ylabel("Δ log-prob", size=7.6)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    doc.save(fig)


def p_cross_timing(doc, S, cw, T):
    fin = (cw[cw.idx >= cw.idx.max() - 2].groupby(["run", "target_category", "prime_category"]).dlpd.mean()
           .groupby(["target_category", "prime_category"]).mean())
    cells = [k for k, v in fin.items() if abs(v) >= 0.03]
    rows = []
    for t, pr in cells:
        em = emergence(cw[(cw.target_category == t) & (cw.prime_category == pr)], T)
        rows.append((t, pr, em["t50"], *em["t50_ci"], fin[(t, pr)]))
    df = pd.DataFrame(rows, columns=["t", "p", "t50", "lo", "hi", "fin"]).sort_values("t50")
    sw = S["timing"]["same word"]["t50"]
    fac, intf = df[df.fin > 0], df[df.fin < 0]
    fig, box = doc.page(
        "5 · Training dynamics", "When each cross-construction link forms",
        "Every construction pair whose different-word priming ends at |Δ| ≥ 0.03 nats/token: the training point at which it "
        "reaches half its end-state, with a 95% run-bootstrap interval. Blue facilitates, violet interferes.",
        notes=[f"{len(df)} links pass the threshold: {len(fac)} facilitating, {len(intf)} interfering. "
               f"{int(((df.t50 >= 300) & (df.t50 <= 550)).sum())} of them reach half size between 300M and 550M.",
               f"Median half-size point: {fmt_tok(fac.t50.median())} for facilitating links"
               + (f", {fmt_tok(intf.t50.median())} for interfering ones." if len(intf) else "."),
               f"Same-word repetition (dashed line) is at half size by {fmt_tok(sw)}; every different-word link is later "
               "or level with it." if (df.t50 >= sw - 1e-9).all() else
               f"Same-word repetition (dashed line) is at half size by {fmt_tok(sw)}; most different-word links come later.",
               "Most links form in one window: cross-construction priming arrives largely as a system, not one "
               "construction pair at a time.",
               "The exceptions are late interference involving embedded drop (" + ", ".join(
                   f"{SHORT[r.t]} ← {SHORT[r.p]} {fmt_tok(r.t50)}" for r in df[df.t50 > 550].itertuples())
               + "), which keeps strengthening to the end of training.",
               "Intervals are over runs; checkpoint spacing (~5% in tokens here) sets the floor on their width."])
    gs = fig.add_gridspec(1, 1, **{**box, "left": 0.3})
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(df))
    for i, r in enumerate(df.itertuples()):
        col = BLUE if r.fin > 0 else VIOLET
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.4, alpha=0.6)
        ax.scatter(r.t50, i, s=40, color=col, edgecolor="white", lw=1, zorder=3)
        ax.annotate(f"{r.fin:+.3f}", (r.hi, i), xytext=(5, 0), textcoords="offset points", va="center", size=7, color=INK2)
    ax.axvline(sw, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.annotate("same-word repetition", (sw, -0.5), xytext=(4, -2), textcoords="offset points", ha="left",
                va="top", size=7, color=MUTED)
    ax.set_yticks(y, [f"{SHORT[r.t]} ← {SHORT[r.p]}" for r in df.itertuples()], size=7.6)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([200, 300, 500, 1000, 2000, 4000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_tok(v)))
    ax.set_xlim(180, 4500)
    ax.set_xlabel("training tokens at half of end-state (log scale)")
    ax.set_title("Half-size point per link (target ← prime)")
    doc.save(fig)




# --- appendix: every target × prime combination over training -----------------
def _grid_fig(doc, title, lede, legend):
    """A large appendix page (16 × 12 in) with an 8 × 8 panel grid: rows = target, columns = prime."""
    fig = doc.plt.figure(figsize=(16, 12))
    doc.n += 1
    fig.text(0.04, 0.965, "9 · APPENDIX", family=HEAD, size=8.5, color=MUTED, weight="demibold")
    fig.text(0.04, 0.95, title, family=HEAD, size=20, weight="bold", color=INK, va="top")
    lines = textwrap.wrap(lede, 190)
    fig.text(0.04, 0.915, "\n".join(lines), family=BODY, size=11, color=INK2, va="top", linespacing=1.4)
    top = min(0.835, 0.915 - 0.0215 * len(lines) - 0.05)
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.975, 0.955), ncol=len(legend), fontsize=9, frameon=False)
    fig.text(0.04, 0.015, FOOT, family=HEAD, size=7.5, color=MUTED)
    fig.text(0.975, 0.015, str(doc.n), family=HEAD, size=8, color=MUTED, ha="right")
    gs = fig.add_gridspec(8, 8, left=0.115, right=0.975, bottom=0.065, top=top, wspace=0.12, hspace=0.18)
    fig.text(0.545, top + 0.027, "PRIME CONSTRUCTION", family=HEAD, size=8, color=MUTED, weight="demibold", ha="center")
    return fig, gs


def _grid_axis(fig, gs, i, j, rows, cols, ylim, T):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax = fig.add_subplot(gs[i, j])
    ax.set_xscale("log")
    ax.set_xlim(0.45, 4500)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}M" if v < 1000 else f"{v / 1000:g}B"))
    ax.tick_params(labelsize=6.2, length=2)
    if i < 7:
        ax.set_xticklabels([])
    if j > 0:
        ax.set_yticklabels([])
    if i == 0:
        ax.set_title(cols[j], size=8.6, loc="center", pad=5)
    if j == 0:
        ax.set_ylabel(rows[i], size=8.2, rotation=0, ha="right", va="center", labelpad=8)
    zero(ax)
    return ax


def p_grid_cells(doc, cw, ca, T):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend = [Line2D([], [], color=AQUA, lw=2, label="different words"),
              Line2D([], [], color=CTRL, lw=1.4, label="all pairings (with repetition)"),
              Patch(facecolor="#e8f1fb", label="different-word end-state ≥ +0.03"),
              Patch(facecolor="#efecfa", label="≤ −0.03")]
    fig, gs = _grid_fig(
        doc, "Every cross-construction combination over training",
        "Priming effect (Δ log-prob, nats/token) for each target construction (rows) after each prime construction (columns), "
        "mean of 140 runs with 95% CI, tokens on a log scale. Same y-scale in every panel. Numbers: end-state for "
        "different-word pairings and, in brackets, all pairings. Diagonal = same construction, different items.",
        legend)
    lab = [SHORT[c] for c in CATS]
    ylim = (-0.1, 0.27)
    for i, t in enumerate(CATS):
        for j, pr in enumerate(CATS):
            ax = _grid_axis(fig, gs, i, j, lab, lab, ylim, T)
            dw = cw[(cw.target_category == t) & (cw.prime_category == pr)]
            al = ca[(ca.target_category == t) & (ca.prime_category == pr)]
            band(ax, al, "dlpd", CTRL, T, lw=1.1, alpha=0.1)
            txt = ""
            if len(dw):
                band(ax, dw, "dlpd", AQUA, T, lw=1.6, alpha=0.18)
                v = dw[dw.idx >= dw.idx.max() - 2].dlpd.mean()
                if v >= 0.03:
                    ax.set_facecolor("#e8f1fb")
                elif v <= -0.03:
                    ax.set_facecolor("#efecfa")
                txt = f"{v:+.3f}"
            va = al[al.idx >= al.idx.max() - 2].dlpd.mean()
            txt = (txt + f" ({va:+.3f})") if txt else f"– ({va:+.3f})"
            ax.text(0.04, 0.94, txt, transform=ax.transAxes, size=6.6, color=INK2, va="top", family=HEAD)
            ax.grid(color=GRID, lw=0.4)
    fig.text(0.012, 0.45, "TARGET CONSTRUCTION", rotation=90, family=HEAD, size=8, color=MUTED, weight="demibold", va="center")
    doc.save(fig)


def beyond_element_share(meta, cond_dir, cats=R.SUBJ):
    """Share of items in `cats` whose overt and null versions differ by more than one inserted word (e.g. a verb
    that changes agreement along with the pronoun)."""
    mb = split_prime(meta, cond_dir)
    tg = mb[mb.target_category.isin(cats)].drop_duplicates(["target_category", "target_item_id", "target_condition"])

    def more(o, n):
        ops = [op for op in difflib.SequenceMatcher(a=o.split(), b=n.split(), autojunk=False).get_opcodes() if op[0] != "equal"]
        return not (len(ops) == 1 and ops[0][0] == "delete" and ops[0][2] - ops[0][1] == 1)
    return float(np.mean([more(o, n) for o, n in zip(tg.target_o, tg.target_n)]))


def side_cells(side):
    d = R.parse_runs(sagg(side, None, ["target_category", "prime_group"]))
    d["target_category"] = d.target_category.astype(str)
    d["prime_group"] = d.prime_group.astype(str)
    return d


def side_grid_ylim(d):
    """One y-range for the pooled grid and every per-intervention grid, so the pages compare directly."""
    m = d[d.idx > 0].groupby(["iv", "target_category", "prime_group", "idx"])[["null_lpd", "overt_lpd"]].mean()
    lo, hi = float(np.nanmin(m.to_numpy())), float(np.nanmax(m.to_numpy()))
    return (np.floor(lo * 1.1 * 20) / 20, np.ceil(hi * 1.1 * 20) / 20)


def p_grid_sides(doc, d, T, ylim, iv=None, meta=None):
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color=ORANGE, lw=2, label="toward null (null version vs reference)"),
              Line2D([], [], color=BLUE, lw=2, label="toward overt (overt version vs reference)")]
    if iv is not None:
        pooled = d
        d = d[d.iv == iv]

        def head(x):
            e_ = x[(x.idx >= x.idx.max() - 2) & x.target_category.isin(R.SUBJ) & (x.prime_group == "subj drop 3rd")]
            return e_.null_lpd.mean(), e_.overt_lpd.mean()
        (n_, o_), (pn, po) = head(d), head(pooled)
        title = f"Null and overt sides: {R.IV_LABEL[iv]}"
        lede = (f"As page {doc.ref['grid_sides']}, for the {d.run.nunique()} {R.IV_LABEL[iv]} runs only (all arms and removal "
                f"levels). Clean 3rd-person subject drop on subject-drop targets, end-state: toward null {n_:+.3f}, toward "
                f"overt {o_:+.3f} (all interventions: {pn:+.3f} / {po:+.3f}). Same y-scale as page {doc.ref['grid_sides']} "
                "and the other intervention pages.")
        share = beyond_element_share(meta, P.IV_DIR[iv]) if meta is not None else 0.0
        if share > 0.1:
            lede += (f" CAUTION: in this condition {share:.0%} of subject-drop items change verb agreement between versions "
                     "(overt 'we enjoyimus the breeze', null 'enjoyikt the breeze'; the same holds in matched-v1), so the "
                     "subject-drop rows and columns here mix the pronoun with agreement morphology and are not comparable "
                     "with the other pages.")
    else:
        title = "Every combination, split into null and overt sides"
        lede = ("For each target (rows) and prime construction (columns; subject-drop primes of both agreement types pooled "
                "and split into clean 3rd-person drop and imperative-like 1st/2nd person): how far the prime's null version "
                "pushes the target toward null, and its overt version toward overt, relative to the same target after the "
                "reference prime (object drop; extraction for object-drop targets). 'ref' = the row's reference. Same "
                "y-scale throughout.")
    fig, gs = _grid_fig(doc, title, lede, legend)
    for i, t in enumerate(CATS):
        for j, g in enumerate(PGROUPS):
            ax = _grid_axis(fig, gs, i, j, [SHORT[c] for c in CATS], [PG_SHORT[x] for x in PGROUPS], ylim, T)
            x = d[(d.target_category == t) & (d.prime_group == g)]
            ax.grid(color=GRID, lw=0.4)
            if not len(x):
                ax.text(0.5, 0.5, "ref", transform=ax.transAxes, ha="center", va="center", size=9, color=MUTED)
                continue
            band(ax, x, "null_lpd", ORANGE, T, lw=1.5, alpha=0.15)
            band(ax, x, "overt_lpd", BLUE, T, lw=1.5, alpha=0.15)
            e_ = x[x.idx >= x.idx.max() - 2]
            ax.text(0.04, 0.94, f"null {e_.null_lpd.mean():+.3f} · overt {e_.overt_lpd.mean():+.3f}", transform=ax.transAxes,
                    size=6.2, color=INK2, va="top", family=HEAD)
    fig.text(0.012, 0.45, "TARGET CONSTRUCTION", rotation=90, family=HEAD, size=8, color=MUTED, weight="demibold", va="center")
    doc.save(fig)



def p_caveats(doc, S):
    doc.text_page(
        "8 · Caveats and next steps", "What this does and does not show",
        "The priming design is unvetted and its baseline is incomplete. These are the limits that matter for reading the "
        "results above, and the cheapest ways to close them.",
        [[("What is solid", [
            "An overt element in the preceding discourse raises the probability of an overt target pronoun, in every model, "
            "learned during the first few hundred million tokens.",
            "Most of it is lexical repetition, and the rest is organized by grammatical function (subject vs object), with "
            "expletive and referential subjects linked in both directions.",
            "Selection arm and intervention change how primeable a model is, in a way not reducible to its overt rate.",
        ]), ("What is not identified", [
            f"Direction, fully. Against reference primes (pages {doc.ref['sides']}–{doc.ref['sides_all']}) the clean "
            "subject-drop effect sits on the null side under four different references, but a shift shared by all of "
            "them cannot be excluded. The neutral root is built (4,608 pairs) and would settle it in one end-state pass.",
            "Null-prime type. 1st/2nd-person null primes read as imperatives in pronoun-free contexts; their columns "
            "contrast overt with imperative, not with dropped subjects.",
        ])],
         [("Design limits", [
             "Prime and target share no content lemma or name, but function words and pronouns are free to repeat, and "
             "repetition is most of the effect. A 'structural' claim should rest on different-word pairings (section 4) or "
             "on the 'it' position contrast.",
             "Primes come from the target's own eval set (one per construction per target), so construction pairs are "
             "unevenly populated by word pairs; a construction's potency inherits its words.",
             f"Same-construction primes can share the target's sentence template (page {doc.ref['examples']}). The strongest "
             "different-word effects may be template priming rather than function priming.",
             "One seed per cell, and runs share their training text: run-level CIs are narrow by construction.",
             "Enriched verbal morphology: most subject-drop items also change verb agreement between versions "
             "('we enjoyimus' / 'enjoyikt'), as in matched-v1.",
             "Pronoun-free contexts remove the speaker, so first-person targets and primes are odd discourse.",
         ]), ("Next steps (cheap to expensive)", [
             "Score the neutral root on the last 3 checkpoints of each run to identify direction.",
             "A no-repetition priming set: noun-phrase subjects as overt primes ('the farmer left' vs 'left') would test "
             "overt-subject priming with no pronoun to copy.",
             "A human-vetted pass over the 576 prime items, starting with the potent and interfering ones on page "
             f"{doc.n}.",
             "Repeat on BERT-v2 once tranche 1 finishes.",
         ])]])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    plt = setup_mpl()
    meta, e, tr, T, side = load()
    S = summarise(e, tr, T)
    summarise_sides(S, side)
    doc = Doc(a.out, plt)
    p_cover(doc, S)
    p_design(doc, meta)
    p_headline(doc, S, e, tr, T)
    p_relation(doc, S, e)
    p_words(doc, S, e)
    p_position(doc, S, e)
    p_person(doc, S, e)
    p_constructions(doc, S, e)
    doc.ref["constructions"] = doc.n
    p_potency(doc, S, e)
    p_ceiling(doc, S, e)
    p_levels(doc, S, e)
    doc.ref["levels"] = doc.n
    p_sides(doc, S, e, side, T, meta)
    doc.ref["sides"] = doc.n
    p_sides_all(doc, S, side)
    p_sides_robust(doc, S, side)
    doc.ref["sides_all"] = doc.n
    p_traj_overview(doc, S, tr, T)
    p_traj_relation(doc, S, tr, T)
    p_traj_targets(doc, S, tr, T)
    cw, ca = cell_frames(tr)
    p_cross_snapshots(doc, S, cw, T)
    p_cross_cells(doc, S, cw, ca, T)
    p_cross_timing(doc, S, cw, T)
    p_traj_rearing(doc, S, tr, T, "arm")
    p_traj_rearing(doc, S, tr, T, "iv")
    p_dose(doc, S, e)
    p_scatter(doc, S, e)
    p_components(doc, S, e)
    p_grid(doc, S, e)
    p_reliability(doc, S, e)
    p_examples(doc, S, e, meta)
    doc.ref["examples"] = doc.n
    p_prime_items(doc, S, e, meta)
    p_caveats(doc, S)
    p_grid_cells(doc, cw, ca, T)
    sd = side_cells(side)
    ylim = side_grid_ylim(sd)
    p_grid_sides(doc, sd, T, ylim)
    doc.ref["grid_sides"] = doc.n
    for iv in R.IVS:
        p_grid_sides(doc, sd, T, ylim, iv, meta)
    doc.close()
    print(f"wrote {a.out} ({doc.n} pages)")


if __name__ == "__main__":
    main()
