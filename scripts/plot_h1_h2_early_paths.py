#!/usr/bin/env python3
"""Plot H2 early paths and H1-vs-H2 early-start trajectories.

The input is the tidy aggregate emitted by ``plot_foundry_trajectories.py``.
H1 and H2 are represented by their first (epoch <= 2) segments.  The exact
pre-training initialization rows are added at ``tokens_seen == 0`` for both
series; those rows are independent of HP rank and are therefore replicated
from the initialization benchmark onto the plotted HP rank.  Scores are
binary overt-preference means from normalized likelihood comparisons.  Cell
means are the independent units for the displayed SE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "analysis/eval_v2/figures/foundry_trajectories/early_vs_continuation_aggregates.csv"
DEFAULT_OUT = REPO / "analysis/eval_v2/figures/foundry_trajectories/h1_h2_early_paths"
ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm", "mamba_370m"]
CONDITIONS = [
    "baseline", "remove_expletive_sentences", "impoverish_case",
    "lemmatize_verbs", "enrich_verbal_morphology",
]
COND_LABELS = {
    "baseline": "Baseline", "remove_expletive_sentences": "Remove expletives",
    "impoverish_case": "Impoverish case", "lemmatize_verbs": "Lemmatize verbs",
    "enrich_verbal_morphology": "Enrich verbal morphology",
}
COND_COLORS = dict(zip(CONDITIONS, plt.cm.tab10.colors[:5]))
HP_COLORS = {1: "#ff7f0e", 2: "#2ca02c"}
CATEGORIES = [
    "subject_drop", "subject_drop_no_agreement", "expletive", "object_drop",
    "embedded_drop", "extraction", "conjunction", "control",
]
CATEGORY_LABELS = {
    "subject_drop": "Subject drop",
    "subject_drop_no_agreement": "Subject drop (no agreement)",
    "expletive": "Expletive", "object_drop": "Object drop",
    "embedded_drop": "Embedded drop", "extraction": "Extraction",
    "conjunction": "Conjunction", "control": "Control",
}


def _x_ticks(max_tokens: float) -> tuple[list[float], list[str]]:
    vals = [0.0, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    vals = [v for v in vals if v == 0 or v <= max_tokens * 1.08]
    labels = ["0" if v == 0 else (f"{v/1e9:.0f}B" if v >= 1e9 else
                                   f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
              for v in vals]
    return list(np.log10(np.asarray(vals) + 1)), labels


def _cell_means(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Collapse duplicate item/checkpoint rows before estimating across cells."""
    group_keys = keys + ["cell_id", "seed"]
    if "source" in d.columns:
        group_keys.append("source")
    return (d.groupby(group_keys, as_index=False)["preference"]
            .mean())


def _binned(d: pd.DataFrame, group_cols: list[str], n_bins: int = 28) -> pd.DataFrame:
    """Log-bin positive token coordinates while preserving the zero point."""
    if d.empty:
        return d
    out = []
    for arch, q in d.groupby("architecture", sort=False):
        q = q.copy()
        positive = q.loc[q.tokens_seen > 0, "tokens_seen"]
        if positive.empty:
            q["bin_center"] = 0.0
            out.append(q)
            continue
        edges = np.unique(np.geomspace(float(positive.min()), float(positive.max()), n_bins + 1))
        if len(edges) < 2:
            q["bin_center"] = 0.0
            out.append(q)
            continue
        q["bin_index"] = np.digitize(q.tokens_seen.to_numpy(), edges, right=False)
        centers = {0: 0.0}
        for i in range(1, len(edges)):
            centers[i] = float(np.sqrt(edges[i - 1] * edges[i]))
        q["bin_center"] = q.bin_index.map(centers)
        out.append(q)
    return pd.concat(out, ignore_index=True)


def _frame(ax, title: str, ticks: tuple[list[float], list[str]], max_x: float,
           xlabel: bool = False) -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="0.55", ls=":", lw=1)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel("P(overt preferred)")
    ax.set_xlim(-0.1, max_x + 0.03)
    ax.set_xticks(*ticks)
    if xlabel:
        ax.set_xlabel("Tokens seen (log scale; log1p display keeps 0 visible)")


def _series(ax, d: pd.DataFrame, label: str, color, key: str = "bin_center") -> None:
    if d.empty:
        return
    s = (d.groupby(key)["preference"].agg(mean="mean", sd="std", n="count")
         .reset_index().sort_values(key))
    x = np.log10(s[key].to_numpy() + 1)
    y = s["mean"].to_numpy()
    se = s["sd"].fillna(0).to_numpy() / np.sqrt(s["n"].to_numpy())
    ax.fill_between(x, np.clip(y - se, 0, 1), np.clip(y + se, 0, 1),
                    color=color, alpha=0.16, lw=0)
    unit_col = "cell_id" if "cell_id" in d.columns else "seed"
    if "source" in d.columns:
        n_train = d.loc[~d.source.eq("initialization"), unit_col].nunique()
        n_init = d.loc[d.source.eq("initialization"), unit_col].nunique()
        label = f"{label} (train n={n_train}; init n={n_init})"
    else:
        label = f"{label} (n={d[unit_col].nunique()})"
    ax.plot(x, y, color=color, lw=2.0, label=label)


def _with_initialization(d: pd.DataFrame, hp_rank: int,
                        stage_segment: str) -> pd.DataFrame:
    """Attach exact checkpoint -1 rows to an early-path HP rank.

    Initialization is shared across HP ranks and interventions.  The source
    aggregate can contain duplicate bookkeeping rows for the same
    architecture/condition/seed/category, so collapse those before assigning
    the target HP rank and stage label.
    """
    init = d[d.source.eq("initialization")].copy()
    if init.empty:
        return d
    # Remove the original h0 initialization rows before appending their
    # explicitly relabeled target-HP copies; otherwise each point would be
    # counted twice at tokens_seen == 0.
    base = d[~d.source.eq("initialization")].copy()
    # At initialization the model state is shared across interventions.  The
    # binary preference values are identical across the available condition
    # files; canonicalize by architecture/seed/category, then materialize all
    # five condition labels so every plotted condition starts at the same
    # token-zero point even when an older export omitted a redundant file.
    keys = ["architecture", "seed", "category", "tokens_seen"]
    init = init.groupby(keys, as_index=False)["preference"].mean()
    conditions = pd.DataFrame({"intervention": CONDITIONS})
    init["_join"] = 1
    conditions["_join"] = 1
    init = init.merge(conditions, on="_join", how="inner").drop(columns="_join")
    init["hp_rank"] = hp_rank
    init["stage_segment"] = stage_segment
    init["cell_id"] = (
        init["architecture"].astype(str) + "-en-" +
        init["intervention"].astype(str) + "-h" + str(hp_rank) +
        "-init-s" + init["seed"].astype(str)
    )
    init["source"] = "initialization"
    return pd.concat([base, init], ignore_index=True)


def plot_h2(d: pd.DataFrame, out: Path) -> list[Path]:
    """One eight-category figure per architecture, H2 fixed at hp rank 2."""
    q = d[(d.hp_rank == 2) & (d.stage_segment == "continuation early") &
          d.category.isin(CATEGORIES) & d.intervention.isin(CONDITIONS)].copy()
    q = pd.concat([
        q,
        d[d.source.eq("initialization") &
          d.category.isin(CATEGORIES) & d.intervention.isin(CONDITIONS)],
    ], ignore_index=True)
    q = _with_initialization(q, hp_rank=2, stage_segment="continuation early")
    q = _cell_means(q, ["architecture", "intervention", "category", "tokens_seen"])
    q = _binned(q, ["architecture", "intervention", "category"])
    outputs = []
    for arch in ARCHES:
        z = q[q.architecture == arch]
        if z.empty:
            continue
        max_tokens = float(z.bin_center.max())
        ticks = _x_ticks(max_tokens)
        fig, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True, sharey=True)
        for i, (ax, cat) in enumerate(zip(axes.flat, CATEGORIES)):
            for cond in CONDITIONS:
                _series(ax, z[(z.category == cat) & (z.intervention == cond)],
                        COND_LABELS[cond], COND_COLORS[cond])
            _frame(ax, CATEGORY_LABELS[cat], ticks, np.log10(max_tokens + 1), i >= 6)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.01), fontsize=9)
        fig.suptitle(f"{arch}: H2 early path (h2; continuation segment through epoch 2)\n"
                     "Includes exact checkpoint −1 at 0 tokens; mean ± 1 SE across seed×condition cells",
                     fontsize=15, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        path = out / f"h2_early_path_{arch}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        z[z.architecture == arch].to_csv(path.with_suffix(".csv"), index=False)
        outputs.append(path)
    return outputs


def plot_h1_h2(d: pd.DataFrame, out: Path) -> list[Path]:
    """One eight-category figure per architecture comparing H1 and H2."""
    h1 = d[(d.hp_rank == 1) &
           d.stage_segment.isin(["early-only", "continuation early"])].copy()
    h1["series"] = "H1"
    h2 = d[(d.hp_rank == 2) &
           d.stage_segment.isin(["early-only", "continuation early"])].copy()
    h2["series"] = "H2"
    if h1.empty or h2.empty:
        return []
    init = d[d.source.eq("initialization") &
             d.category.isin(CATEGORIES) & d.intervention.isin(CONDITIONS)].copy()
    h1 = pd.concat([h1, init], ignore_index=True)
    h2 = pd.concat([h2, init], ignore_index=True)
    h1 = _with_initialization(h1, hp_rank=1, stage_segment="continuation early")
    h2 = _with_initialization(h2, hp_rank=2, stage_segment="continuation early")
    h1["series"] = "H1"
    h2["series"] = "H2"
    q = pd.concat([h1, h2], ignore_index=True)
    q = _cell_means(q, ["architecture", "series", "category", "tokens_seen"])
    # Average intervention conditions within seed before computing the SE so
    # the independent unit is the seed, not the five manipulations.
    q = (q.groupby(["architecture", "series", "category", "seed", "tokens_seen", "source"],
                   as_index=False)["preference"].mean())
    q = _binned(q, ["architecture", "series", "category"])
    colors = {"H1": HP_COLORS[1], "H2": HP_COLORS[2]}
    outputs = []
    for arch in ARCHES:
        z = q[q.architecture == arch]
        if z.empty:
            continue
        max_tokens = float(z.bin_center.max())
        ticks = _x_ticks(max_tokens)
        fig, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True, sharey=True)
        for i, (ax, cat) in enumerate(zip(axes.flat, CATEGORIES)):
            for series in ("H1", "H2"):
                _series(ax, z[(z.category == cat) & (z.series == series)],
                        series, colors[series])
            _frame(ax, CATEGORY_LABELS[cat], ticks, np.log10(max_tokens + 1), i >= 6)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.01), fontsize=9)
        fig.suptitle(f"{arch}: aggregate early starts, H1 vs H2\n"
                     "Includes exact checkpoint −1 at 0 tokens; interventions averaged within seed",
                     fontsize=15, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        path = out / f"h1_vs_h2_early_{arch}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        z[z.architecture == arch].to_csv(path.with_suffix(".csv"), index=False)
        outputs.append(path)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.input)
    d["hp_rank"] = pd.to_numeric(d["hp_rank"], errors="coerce")
    d["tokens_seen"] = pd.to_numeric(d["tokens_seen"], errors="coerce")
    d["seed"] = pd.to_numeric(d["seed"], errors="coerce")
    d = d[d.architecture.isin(ARCHES) & d.category.isin(CATEGORIES)].dropna(subset=["tokens_seen", "preference"])
    h2 = plot_h2(d, args.out)
    comparison = plot_h1_h2(d, args.out)
    print(f"H2 figures: {len(h2)}; H1-vs-H2 figures: {len(comparison)}")
    print("H2 cells by architecture:", d[(d.hp_rank == 2) & (d.stage_segment == "continuation early")].groupby("architecture").cell_id.nunique().to_dict())
    print("H1 cells by architecture:", d[(d.hp_rank == 1) & (d.stage_segment == "early-only")].groupby("architecture").cell_id.nunique().to_dict())


if __name__ == "__main__":
    main()
