#!/usr/bin/env python3
"""SCIL-paper-style figures from null-subj-v2 eval results.

Reproduces the visual conventions of subject-drop/analysis/paper_figures/scil
(white panels, % y-axis, log checkpoint axis, end-of-first-epoch line,
binomial CI ribbons, bottom legend) over the v2 wave's wider phenomenon set:
one learning-curve panel per category, plus an end-state grid.

Run after `AWS_PROFILE=nrp python scripts/pull_eval_results.py`:
    python analysis/eval_v2/scil_style_figures.py [--archs gpt2_small gpt2_medium]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/eval_results/null_subj_v2"
FIGS = REPO / "analysis/eval_v2/figures"

CONDS = ["baseline", "remove_expletive_sentences", "impoverish_case",
         "lemmatize_verbs", "enrich_verbal_morphology"]
LABELS = {"baseline": "Baseline",
          "remove_expletive_sentences": "Remove Expletives",
          "impoverish_case": "Impoverish Case",
          "lemmatize_verbs": "Lemmatize Verbs",
          "enrich_verbal_morphology": "Enrich Verbal Morphology"}
# Match the SCIL palette ordering (tab10): blue/orange/green/red/purple.
COLORS = dict(zip(CONDS, plt.cm.tab10.colors[:5]))
CATS = ["subject_drop", "subject_drop_no_agreement", "expletive",
        "object_drop", "embedded_drop", "extraction", "conjunction",
        "control"]
CAT_TITLES = {
    "subject_drop": "Subject Drop", "subject_drop_no_agreement":
    "Subject Drop (No Agreement)", "expletive": "Expletives",
    "object_drop": "Object Drop", "embedded_drop": "Embedded Drop",
    "extraction": "Extraction", "conjunction": "Conjunction",
    "control": "Control (null grammatical)",
}
EPOCH1_STEP = 1044  # first checkpoint inside epoch 1 (sidecar-measured)


def _style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="0.90", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.4")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.02, 1.02)


def _log_ticks(ax):
    ax.set_xscale("log")
    ticks = [1, 11, 101, 1001, 10001]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "10", "100", "1K", "10K"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def learning_curves(pairs: pd.DataFrame, arch: str) -> Path:
    d = pairs[pairs.architecture == arch]
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    for ax, cat in zip(axes.flat, CATS):
        sub_c = d[d.category == cat]
        for cond in CONDS:
            sub = sub_c[sub_c.intervention == cond]
            if sub.empty:
                continue
            g = sub.groupby("checkpoint_step").prefers_overt_meanlp
            p, n = g.mean(), g.size()
            se = np.sqrt(p * (1 - p) / n)
            x = p.index + 1
            ax.plot(x, p.values, color=COLORS[cond], lw=1.6,
                    label=LABELS[cond])
            ax.fill_between(x, p - 1.96 * se, p + 1.96 * se,
                            color=COLORS[cond], alpha=0.15, lw=0)
        ax.axvline(EPOCH1_STEP, color="0.45", lw=0.9)
        ax.text(EPOCH1_STEP * 1.15, 0.96, "end of first epoch",
                color="0.45", fontsize=7.5, va="top")
        ax.axhline(0.5, color="0.6", ls=":", lw=1)
        _style_axis(ax)
        _log_ticks(ax)
        ax.set_title(CAT_TITLES[cat], loc="left", fontweight="bold",
                     fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Training Checkpoint")
    for ax in axes[:, 0]:
        ax.set_ylabel("Overt Preference")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=10)
    fig.suptitle(f"Overt Preference by Phenomenon — {arch} (en, h0-s42)",
                 fontweight="bold", x=0.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    out = FIGS / f"scil_learning_curves_{arch}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(f"{out}.png")


def endstate(pairs: pd.DataFrame, archs: list[str]) -> Path:
    finals = pairs.groupby("cell_id").checkpoint_step.transform("max")
    fin = pairs[pairs.checkpoint_step == finals]
    fig, axes = plt.subplots(len(archs), 1, figsize=(11, 4 * len(archs)),
                             sharex=True, squeeze=False)
    width = 0.15
    xs = np.arange(len(CATS))
    for ax, arch in zip(axes[:, 0], archs):
        d = fin[fin.architecture == arch]
        for k, cond in enumerate(CONDS):
            sub = d[d.intervention == cond]
            g = sub.groupby("category").prefers_overt_meanlp
            p = g.mean().reindex(CATS)
            n = g.size().reindex(CATS)
            se = np.sqrt(p * (1 - p) / n)
            ax.bar(xs + (k - 2) * width, p.values, width=width,
                   color=COLORS[cond], label=LABELS[cond],
                   yerr=1.96 * se.values, error_kw={"lw": 0.8, "capsize": 2})
        ax.axhline(0.5, color="0.5", ls="--", lw=1)
        _style_axis(ax)
        ax.set_title(f"End-State Overt Preference — {arch}", loc="left",
                     fontweight="bold", fontsize=12)
        ax.set_ylabel("Overt Preference")
    axes[-1, 0].set_xticks(xs, [CAT_TITLES[c] for c in CATS], rotation=20,
                           ha="right", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = FIGS / "scil_endstate_by_phenomenon"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(f"{out}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["gpt2_small", "gpt2_medium"])
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    pairs = pd.concat([pd.read_parquet(f)
                       for f in (DATA / "pairs").glob("*.parquet")])
    avail = sorted(pairs.architecture.unique())
    print(f"architectures with data: {avail}")
    for arch in args.archs:
        if arch not in avail:
            print(f"  skip {arch} (no data yet)")
            continue
        print(" ", learning_curves(pairs, arch))
    print(" ", endstate(pairs, [a for a in args.archs if a in avail]))


if __name__ == "__main__":
    main()
