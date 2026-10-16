#!/usr/bin/env python3
"""End state by removal decile with one panel per selection arm and one line
per intervention: the interventions compared within most-recoverable-first,
random, and least-recoverable-first. A second figure collapses baseline and
interventions into one line per arm (mean over the conditions every arm has).

Input is an end-state CSV already written by wave2_endstate.py
(endstate_all*.csv: arm, k, iv, hp, endstate) or wave2_endstate_latest.py
(endstate_latest_*.csv: arm, k, iv, pref, finished). Runs still training
(finished == False) are drawn hollow.

  python analysis/eval_v2/wave2_arm_by_intervention.py --csv endstate_latest_gpt2_large_h0.csv \
      --label gpt2_large --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:  # run as a script from analysis/eval_v2/ ...
    from wave2_endstate import ARM_STYLE, SIENNA
except ImportError:  # ... or imported as analysis.eval_v2.wave2_arm_by_intervention
    from analysis.eval_v2.wave2_endstate import ARM_STYLE, SIENNA

ARMS = [("robbi", "Most recoverable removed first"),
        ("rand", "Random removal (control)"),
        ("robbianti", "Least recoverable removed first")]
# Categorical slots 1-5 in fixed order (validated: CVD + normal-vision floors
# pass); markers carry identity too, since three hues are <3:1 on white.
IVS = [("base", "baseline", "#2a78d6", "o"),
       ("impcase", "impoverish case", "#eb6834", "s"),
       ("lemverb", "lemmatize verbs", "#1baf7a", "^"),
       ("enrichvm", "enrich verbal morph", "#eda100", "D"),
       ("rmexpl", "remove expletives", "#e87ba4", "v")]
INK, MUTED, GRID = "#1c2127", "#5b636c", "#e3e5e3"


def load(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    if "endstate" in d.columns:            # wave2_endstate.py output: finished runs only
        d = d[d.hp == 0].rename(columns={"endstate": "pref"})
        d["finished"] = True
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--label", required=True, help="architecture label for the title")
    ap.add_argument("--note", default="", help="extra subtitle text")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = load(a.csv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    anchors = d[d.arm == "all100"]
    for ax, (arm, title) in zip(axes, ARMS):
        sub = d[(d.arm == arm) & (d.k < 100)]
        for j, (iv, lab, col, mk) in enumerate(IVS):
            s = sub[sub.iv == iv].groupby("k").agg(pref=("pref", "mean"), finished=("finished", "all")).sort_index()
            if len(s):
                ax.plot(s.index, s.pref, color=col, lw=2, label=lab, zorder=2)
                done, live = s[s.finished], s[~s.finished]
                ax.plot(done.index, done.pref, linestyle="none", marker=mk, ms=6, color=col,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
                ax.plot(live.index, live.pref, linestyle="none", marker=mk, ms=6,
                        markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.4, zorder=3)
            anc = anchors[anchors.iv == iv]
            if len(anc):   # all-removed reference, same for every arm; jittered so stars don't stack
                ax.plot([100 + (j - 2) * 1.6], [anc.pref.mean()], marker="*", ms=11, color=col,
                        markeredgecolor="white", markeredgewidth=0.6, linestyle="none", zorder=3)
        ax.axhline(0.5, color=GRID, lw=1, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xticks([10, 30, 50, 70, 90, 100])
        ax.set_xticklabels(["10", "30", "50", "70", "90", "all"])
        ax.set_xlabel("% of subject pronouns removed (decile)")
    axes[0].set_ylabel("end-state overt-subject preference")
    axes[0].set_ylim(0, 1)
    present = set(d.iv)
    h = [plt.Line2D([], [], color=col, lw=2, marker=mk, ms=6, markeredgecolor="white")
         for iv, _, col, mk in IVS if iv in present]
    l = [lab for iv, lab, _, _ in IVS if iv in present]
    extra = [plt.Line2D([], [], marker="*", ms=10, color=MUTED, linestyle="none")]
    lab_extra = ["at 'all' = every subject pronoun removed (per intervention)"]
    if (~d.finished).any():
        extra.append(plt.Line2D([], [], marker="o", ms=6, markerfacecolor="white", markeredgecolor=MUTED,
                                linestyle="none"))
        lab_extra.append("hollow = still training (latest checkpoints so far)")
    fig.legend(h + extra, l + lab_extra, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Interventions compared within each selection arm — {a.label}"
                 + (f"  ({a.note})" if a.note else ""), y=1.02, fontsize=12)
    fig.tight_layout()
    out = a.out / f"arm_by_intervention_{a.label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    combined_figure(d, a.label, a.note, a.out, plt)


def combined_figure(d: pd.DataFrame, label: str, note: str, out_dir: Path, plt) -> None:
    """One line per arm: mean over baseline + interventions. Only conditions
    present in all three arms count, so every arm averages the same set."""
    c = (d.groupby(["arm", "k", "iv"])
         .agg(pref=("pref", "mean"), finished=("finished", "all")).reset_index())
    arms = [arm for arm, _ in ARMS]
    conds = [iv for iv, *_ in IVS if all(((c.arm == arm) & (c.iv == iv)).any() for arm in arms)]
    m = (c[c.iv.isin(conds)].groupby(["arm", "k"])
         .agg(pref=("pref", "mean"), n=("iv", "nunique"), finished=("finished", "all")).reset_index())
    m.to_csv(out_dir / f"arm_combined_{label}.csv", index=False)
    names = {iv: lab for iv, lab, *_ in IVS}

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for arm in ("robbi", "rand", "robbianti"):
        col, ls, mk, lab = ARM_STYLE[arm]
        s = m[(m.arm == arm) & (m.k < 100)].sort_values("k")
        ax.plot(s.k, s.pref, color=col, linestyle=ls, lw=2, zorder=2)
        done, live = s[s.finished], s[~s.finished]
        ax.plot(done.k, done.pref, linestyle="none", marker=mk, ms=6, color=col,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.plot(live.k, live.pref, linestyle="none", marker=mk, ms=6, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=1.4, zorder=3)
        print(f"  {arm:10s} " + " ".join(f"{v:.2f}" for v in s.pref))
    anc = m[m.arm == "all100"]
    if len(anc):
        ax.plot([100], [anc.pref.mean()], marker="*", ms=14, color=SIENNA, linestyle="none", zorder=3)
    ax.axhline(0.5, color=GRID, lw=1, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xticks([10, 30, 50, 70, 90, 100])
    ax.set_xticklabels(["10", "30", "50", "70", "90", "all"])
    ax.set_xlabel("% of subject pronouns removed (decile)")
    ax.set_ylabel("end-state overt-subject preference")
    h = [plt.Line2D([], [], color=ARM_STYLE[arm][0], linestyle=ARM_STYLE[arm][1], lw=2,
                    marker=ARM_STYLE[arm][2], ms=6, markeredgecolor="white") for arm in ("robbi", "rand", "robbianti")]
    l = [ARM_STYLE[arm][3] for arm in ("robbi", "rand", "robbianti")]
    h.append(plt.Line2D([], [], marker="*", ms=12, color=SIENNA, linestyle="none"))
    l.append("all subject pronouns removed")
    if (~m.finished).any():
        h.append(plt.Line2D([], [], marker="o", ms=6, markerfacecolor="white", markeredgecolor=MUTED,
                            linestyle="none"))
        l.append("hollow = includes runs still training")
    ax.legend(h, l, loc="lower left", fontsize=8.5, frameon=False, handlelength=3.2)
    ax.set_title(f"Baseline + interventions averaged ({len(conds)} conditions) — {label}"
                 + (f"\n{note}" if note else ""), fontsize=10, color=INK)
    print("  conditions:", ", ".join(names[iv] for iv in conds))
    fig.tight_layout()
    out = out_dir / f"arm_combined_{label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
