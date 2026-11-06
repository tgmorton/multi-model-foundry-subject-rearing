#!/usr/bin/env python3
"""PDF: what the enrichment null-agreement fix changed (docs/ENRICH_NULL_AGREEMENT_FIX.md).

Pronoun-free and priming results for the 28 wave2 enrichment runs, scored on the
old (v1) and fixed (v2) items, against the 28 baseline-intervention runs. Needs the
priming_deep tables (priming_deep_data.py) for v1 and the same tables built from
the v2 benchmark under report_cache/priming_deep_v2.

    python analysis/eval_v2/enrich_fix_plots.py --out enrich_fix_impact.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nopron_priming_report as R  # noqa: E402
import priming_deep_data as P  # noqa: E402
import priming_deep_report as D  # noqa: E402

D.FOOT = "gpt2_large wave2 · enrichment null-agreement fix · pronoun-free and priming sets; matched set pending"
SUBJ = R.SUBJ
PERS = ["subj_1sg", "subj_1pl", "subj_2sg", "subj_2pl", "subj_3sg", "subj_3pl"]
# Validated all-pairs for three series; these hues are not used for the arms, so colour keeps one meaning.
CMP = {"baseline": "#008300", "enrich, old items": D.YELLOW, "enrich, fixed items": D.VIOLET}
LEVELS = list(range(10, 100, 10))
SIDE_GROUPS = ["subj drop 3rd", "subj drop 1st/2nd", "expletive"]


def load_all():
    """{'baseline' | 'enrich, old items' | 'enrich, fixed items': (end, traj, side)}, and the token axis."""
    meta1, e1, tr1, T, s1 = D.load()
    P.OUT = R.CACHE / "priming_deep_v2"
    meta2, e2, tr2, _, s2 = D.load()

    def iv_of(df):
        return df.run.str.extract(R.RUN_RE)["iv"]
    s1, s2 = s1.assign(iv=iv_of(s1)), s2.assign(iv=iv_of(s2))
    tr1 = tr1.assign(iv=iv_of(tr1))
    out = {"baseline": (e1[e1.iv == "base"], tr1[tr1.iv == "base"], s1[s1.iv == "base"]),
           "enrich, old items": (e1[e1.iv == "enrichvm"], tr1[tr1.iv == "enrichvm"], s1[s1.iv == "enrichvm"]),
           "enrich, fixed items": (e2, tr2.assign(iv="enrichvm"), s2)}
    return out, T, meta2, s1


def unprimed(e):
    """One row per target: the unprimed (pronoun-free) score rides on every pairing of that target."""
    return e.drop_duplicates(["run", "target_category", "target_condition", "target_item_id"])


def subject_runs(e):
    u = unprimed(e[e.target_category.isin(SUBJ)])
    return u.groupby(["run", "arm", "k"])[["pref_u", "lpd_u"]].mean().reset_index()


def order_effect(v):
    w = v[v.arm.isin(["robbi", "robbianti"])].pivot_table(index="k", columns="arm", values="pref_u")
    return w.robbi - w.robbianti


def p_dose(doc, data):
    runs = {k: subject_runs(v[0]) for k, v in data.items()}
    oe = {k: order_effect(v).mean() for k, v in runs.items()}
    allr = {k: v[v.arm == "all"].pref_u.mean() for k, v in runs.items()}
    fig, box = doc.page(
        "Order effect", "With the items fixed, the arms separate like baseline again",
        "Overt preference on pronoun-free subject-drop items against the share of pronouns removed, per selection arm. Each "
        "point is one run. The old items squeezed the arms together; on the fixed items they spread back out.")
    gs = fig.add_gridspec(1, 3, **{**box, "top": box["top"] - 0.02, "bottom": 0.2}, wspace=0.12)
    axes = []
    for j, (name, v) in enumerate(runs.items()):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        for arm in ("rand", "robbi", "robbianti"):
            d = v[v.arm == arm].sort_values("k")
            ax.plot(d.k, d.pref_u, color=D.ARM_C[arm], marker="o", ms=4.5, mec="white", mew=0.8, label=D.ARM_LABEL[arm])
        d = v[v.arm == "all"]
        ax.scatter(d.k, d.pref_u, color=D.ARM_C["all"], marker="D", s=38, edgecolor="white", lw=0.8, zorder=3,
                   label=D.ARM_LABEL["all"])
        ax.set_ylim(0, 1)
        ax.set_xticks([10, 30, 50, 70, 90, 100])
        ax.set_xlabel("subject pronouns removed (%)")
        ax.set_title(name)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("P(prefer overt), end of training")
            ax.legend(loc="lower left", fontsize=7.2)
    doc.bottom_notes(fig, [
        f"Mean gap between most- and least-recoverable-first: baseline {oe['baseline']:+.3f}, old items "
        f"{oe['enrich, old items']:+.3f}, fixed items {oe['enrich, fixed items']:+.3f}.",
        f"All pronouns removed: old items {allr['enrich, old items']:.3f}, fixed items {allr['enrich, fixed items']:.3f}, "
        f"baseline {allr['baseline']:.3f}.",
        "One seed per point, so single points are noisy; read the shape of each arm."], top=0.135, ncol=3, title="Read")
    doc.save(fig)
    return oe


def p_order(doc, data):
    runs = {k: subject_runs(v[0]) for k, v in data.items()}
    fig, box = doc.page(
        "Order effect", "The order effect was compressed by the old items, not by enrichment",
        "Most-recoverable-first minus least-recoverable-first at each matched removal level. This is the pilot's key contrast. "
        "The earlier finding that enrichment compresses it does not survive the fix.",
        notes=["Fixed-item enrichment tracks baseline across the removal range, rather than sitting below it.",
               "The compression came from the least-recoverable-first arm, where old items inflated overt preference most.",
               "Differences of two single-seed runs, so each point is noisy. The summary on the right averages the 9 levels.",
               "Bars show mean ± 95% CI across removal levels."])
    gs = fig.add_gridspec(1, 2, **box, wspace=0.3, width_ratios=[1.7, 1])
    ax = fig.add_subplot(gs[0])
    means, errs = [], []
    for name, v in runs.items():
        d = order_effect(v)
        ax.plot(d.index, d.values, color=CMP[name], marker="o", ms=5, mec="white", mew=0.8, label=name)
        means.append(d.mean())
        errs.append(R.ci(d.values))
    D.zero(ax)
    ax.set_xticks(LEVELS)
    ax.set_xlabel("subject pronouns removed (%)")
    ax.set_ylabel("Δ P(prefer overt): most minus least recoverable first")
    ax.legend(loc="upper left", fontsize=7.6)
    ax.set_title("By removal level")
    ax2 = fig.add_subplot(gs[1])
    D.bars(ax2, [n.replace("enrich, ", "") for n in runs], means, errs, [CMP[n] for n in runs], horizontal=True)
    D.zero(ax2, "x")
    ax2.set_xlabel("mean order effect")
    ax2.set_title("Averaged over levels")
    doc.save(fig)


def p_person(doc, data):
    def tab(col):
        return pd.DataFrame({k: D_by(v[0], col) for k, v in data.items()}).reindex(PERS)

    def D_by(e, col):
        u = unprimed(e[e.target_category.isin(SUBJ)])
        return u.groupby(["run", "target_condition"])[col].mean().groupby("target_condition").mean()

    def sets(e):
        u = unprimed(e[e.target_category.isin(SUBJ)])
        return u.groupby(["run", "target_category"]).pref_u.mean().groupby("target_category").mean()
    p, g = tab("pref_u"), tab("lpd_u")
    st = pd.DataFrame({k: sets(v[0]) for k, v in data.items()}).reindex(SUBJ)
    fixed, base = p["enrich, fixed items"], p["baseline"]
    fig, box = doc.page(
        "By person", "The fix is not a uniform shift",
        "Pronoun-free subject-drop items at end of training, by the person of the target. The old items changed the verb for "
        "every person except 3sg, and in different ways, so each person moves differently.",
        notes=["3sg preference barely moves, because almost all its null forms were already right. Its gap falls only because "
               "a few items with plain-English nulls ('named the dog milo') were repaired; every other 3sg item scores "
               "exactly as before.",
               f"Enrichment is now less overt than baseline for 3rd person (3pl {fixed['subj_3pl']:.3f} vs "
               f"{base['subj_3pl']:.3f}, 3sg {fixed['subj_3sg']:.3f} vs {base['subj_3sg']:.3f}), the predicted direction. "
               "For 1st and 2nd person it is still more overt.",
               "Pronoun-free contexts never mention a speaker, so read the 1st-person bars with care; the matched set will "
               "settle them.",
               f"The two subject-drop sets now agree ({st.at[SUBJ[0], 'enrich, fixed items']:.3f} and "
               f"{st.at[SUBJ[1], 'enrich, fixed items']:.3f}; old items {st.at[SUBJ[0], 'enrich, old items']:.3f} and "
               f"{st.at[SUBJ[1], 'enrich, old items']:.3f})."])
    gs = fig.add_gridspec(2, 1, **box, hspace=0.42)
    lab = [c.replace("subj_", "") for c in PERS]
    for r, (t, ylab, title) in enumerate([(p, "P(prefer overt)", "Overt preference"),
                                          (g, "overt − null log-prob (nats/token)", "Log-prob gap")]):
        ax = fig.add_subplot(gs[r])
        x = np.arange(len(PERS))
        for k, name in enumerate(t.columns):
            ax.bar(x + (k - 1) * 0.27, t[name], 0.25, color=CMP[name], label=name)
        ax.set_xticks(x, lab)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(ylab, size=7.8)
        ax.set_title(title)
        if r == 0:
            ax.set_ylim(0, 1)
            ax.legend(loc="upper center", ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, 1.2))
        else:
            D.zero(ax)
    doc.save(fig)


def p_training(doc, data, T):
    fig, box = doc.page(
        "Over training", "On fixed items, least-recoverable-first keeps falling",
        "Overt preference on pronoun-free subject-drop items through training, by selection arm. Mean of the runs in each arm "
        "with 95% CI; removal levels pooled within arm. All-removed is a single run.")
    gs = fig.add_gridspec(1, 3, **{**box, "bottom": 0.2}, wspace=0.12)
    curves = {}
    axes = []
    for j, (name, (e, tr, s)) in enumerate(data.items()):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        a = D.tagg(tr, tr.target_category.astype(str).isin(SUBJ) & ((tr.prime_category.astype(str) == "object_drop")))
        curves[name] = a
        for arm in D.ARMS:
            d = a[a.arm == arm]
            if arm == "all":
                g = d[d.idx > 0].groupby("idx").pref_u.mean()
                ax.plot(T.loc[g.index], g, color=D.ARM_C[arm], lw=1.6, label=D.ARM_LABEL[arm])
            else:
                D.band(ax, d, "pref_u", D.ARM_C[arm], T, label=D.ARM_LABEL[arm], alpha=0.12)
        D.tokens_axis(ax)
        ax.set_ylim(0, 1)
        ax.set_title(name)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("P(prefer overt)")
            ax.legend(loc="lower left", fontsize=7)
    late = {n: c[c.idx >= c.idx.max() - 2].groupby("arm").pref_u.mean() for n, c in curves.items()}
    doc.bottom_notes(fig, [
        "The arms separate after about 100M tokens. On old items that stops by 300M; on fixed items "
        "least-recoverable-first is still falling at the end.",
        f"With fixed items, least-recoverable-first ends at {late['enrich, fixed items']['robbianti']:.2f} (old items "
        f"{late['enrich, old items']['robbianti']:.2f}, baseline {late['baseline']['robbianti']:.2f}).",
        "Unprimed scores only; the y-axis is the same in all panels."], top=0.135, ncol=3, title="Read")
    doc.save(fig)


def p_sides(doc, data, T):
    res = {}
    for name, (e, tr, s) in data.items():
        d = D.side_end(s, SUBJ)
        res[name] = (d.groupby("prime_group")[D.SIDE_VALS].mean(), d.groupby("prime_group")[D.SIDE_VALS].agg(D.run_ci))
    m = {k: v[0] for k, v in res.items()}
    fig, box = doc.page(
        "Priming", "The enrichment priming reversal is gone",
        "Subject-drop targets. How far each prime's null version pushes the target toward null, and its overt version toward "
        "overt, relative to the same target after an object-drop prime. Top: end of training. Bottom: over training for clean "
        "3rd-person subject-drop primes.",
        notes=[f"Clean 3rd-person primes on old items: toward null {m['enrich, old items'].at['subj drop 3rd', 'null_lpd']:+.3f}, "
               f"toward overt {m['enrich, old items'].at['subj drop 3rd', 'overt_lpd']:+.3f}. Fixed items: "
               f"{m['enrich, fixed items'].at['subj drop 3rd', 'null_lpd']:+.3f} and "
               f"{m['enrich, fixed items'].at['subj drop 3rd', 'overt_lpd']:+.3f}. The null side leads again, as at baseline "
               f"({m['baseline'].at['subj drop 3rd', 'null_lpd']:+.3f}, {m['baseline'].at['subj drop 3rd', 'overt_lpd']:+.3f}).",
               f"Dropped expletives go the same way: {m['enrich, old items'].at['expletive', 'null_lpd']:+.3f} to "
               f"{m['enrich, fixed items'].at['expletive', 'null_lpd']:+.3f} toward null.",
               "So null-subject priming holds under enrichment too. For clean 3rd-person primes it is smaller than baseline; "
               "for expletives it is about the same.",
               "These compare against one reference prime, so a text offset can shift both sides; see pages 12–14 of the "
               "priming report."])
    gs = fig.add_gridspec(2, 2, **{**box, "left": 0.135}, hspace=0.5, wspace=0.14, height_ratios=[1, 1])
    for j, (col, title) in enumerate([("null_lpd", "Toward null"), ("overt_lpd", "Toward overt")]):
        ax = fig.add_subplot(gs[0, j])
        y = np.arange(len(SIDE_GROUPS))
        for k, name in enumerate(CMP):
            mm = [res[name][0].at[g, col] for g in SIDE_GROUPS]
            ee = [res[name][1].at[g, col] for g in SIDE_GROUPS]
            ax.barh(y + (k - 1) * 0.27, mm, 0.25, xerr=ee, color=CMP[name], label=name,
                    error_kw={"lw": 0.7, "ecolor": D.INK2, "capsize": 1.5})
        ax.set_yticks(y, [D.PG_SHORT[g] for g in SIDE_GROUPS] if j == 0 else [""] * len(y), size=7.6)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        D.zero(ax, "x")
        ax.set_xlim(-0.05, 0.24)
        ax.set_xlabel("shift, Δ log-prob (nats/token)")
        ax.set_title(title)
        a2 = fig.add_subplot(gs[1, j])
        for name, (e, tr, s) in data.items():
            d = D.sagg(s, s.target_category.isin(SUBJ) & (s.prime_group == "subj drop 3rd"))
            D.band(a2, d, col, CMP[name], T, label=name, lw=1.7, alpha=0.12)
        D.zero(a2)
        D.tokens_axis(a2)
        a2.set_ylim(-0.06, 0.17)
        a2.set_title(title + ", clean 3rd-person primes", size=9)
        if j:
            a2.tick_params(labelleft=False)
        else:
            a2.legend(loc="upper left", fontsize=7)
    doc.save(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    plt = D.setup_mpl()
    data, T, meta2, all_sides = load_all()
    doc = D.Doc(a.out, plt)
    p_dose(doc, data)
    p_order(doc, data)
    p_person(doc, data)
    p_training(doc, data, T)
    p_sides(doc, data, T)
    # the page-36 grid, on the fixed items
    side = data["enrich, fixed items"][2]
    sd = D.side_cells(pd.concat([data["baseline"][2], side], ignore_index=True))
    ylim = D.side_grid_ylim(D.side_cells(all_sides))     # the report's scale, so the pages compare directly
    doc.ref["grid_sides"] = "32 of the priming report"
    D.p_grid_sides(doc, sd, T, ylim, "enrichvm", meta2, ref_iv="base")
    doc.close()
    print(f"wrote {a.out} ({doc.n} pages)")


if __name__ == "__main__":
    main()
