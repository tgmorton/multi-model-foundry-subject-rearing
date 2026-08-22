#!/usr/bin/env python3
"""Plot Foundry trajectories by baseline HP, intervention, and run stage.

The script consumes the local null-subj-v2 pair/checkpoint exports.  It first
reduces each evaluation file to one binary overt-preference score per
category/checkpoint, then averages equally across cells at a common
``tokens_seen`` value.  This keeps seeds, HP ranks, and interventions from
being implicitly weighted by item counts.

Outputs are written below ``analysis/eval_v2/figures/foundry_trajectories``:

* ``baseline_by_hyperparameter_<arch>.png`` — baseline only, h0–h4,
  averaged across seeds;
* ``intervention_collapsed_<arch>.png`` — five intervention lines,
  collapsed across HP ranks and seeds;
* ``early_vs_continuation_<arch>.png`` — early-only starts, the early segment
  of continuation runs, and their late continuation segment.

Checkpoint -1 is taken from ``null_subj_v2_init`` and placed at exactly
0 tokens.  The plotted x coordinate is log10(tokens_seen + 1), solely so the
zero-token initialization point can coexist with the log-scaled trajectory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
TRAIN_PAIRS = REPO / "data/eval_results/null_subj_v2/pairs"
TRAIN_CKPTS = REPO / "data/eval_results/null_subj_v2/checkpoints"
INIT_PAIRS = REPO / "data/eval_results/null_subj_v2_init/pairs"
OUT = REPO / "analysis/eval_v2/figures/foundry_trajectories"

ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm", "mamba_370m"]
CONDITIONS = [
    "baseline",
    "remove_expletive_sentences",
    "impoverish_case",
    "lemmatize_verbs",
    "enrich_verbal_morphology",
]
COND_LABELS = {
    "baseline": "Baseline",
    "remove_expletive_sentences": "Remove expletives",
    "impoverish_case": "Impoverish case",
    "lemmatize_verbs": "Lemmatize verbs",
    "enrich_verbal_morphology": "Enrich verbal morphology",
}
CATEGORIES = [
    "subject_drop",
    "subject_drop_no_agreement",
    "expletive",
    "object_drop",
    "embedded_drop",
    "extraction",
    "conjunction",
    "control",
]
CATEGORY_LABELS = {
    "subject_drop": "Subject drop",
    "subject_drop_no_agreement": "Subject drop (no agreement)",
    "expletive": "Expletive",
    "object_drop": "Object drop",
    "embedded_drop": "Embedded drop",
    "extraction": "Extraction",
    "conjunction": "Conjunction",
    "control": "Control",
}
HP_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}
COND_COLORS = dict(zip(CONDITIONS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
STAGE_COLORS = {
    "early-only": "#1f77b4",
    "continuation early": "#ff7f0e",
    "late continuation": "#d62728",
}
STAGE_LABELS = {
    "early-only": "Early-only starts",
    "continuation early": "Continuation starts",
    "late continuation": "Late continuation",
}


def _cell_stage(ckpt: pd.DataFrame) -> str:
    """Classify a cell from the resolved checkpoint epoch metadata."""
    return "early-only" if int(ckpt["epoch"].max()) <= 2 else "continuation"


def _aggregate_eval(path: Path, ckpt: pd.DataFrame, source: str) -> pd.DataFrame:
    """Reduce one pair parquet to per-category/token binary preference."""
    cols = [
        "cell_id",
        "architecture",
        "intervention",
        "rep",
        "checkpoint_step",
        "category",
        "prefers_overt_meanlp",
    ]
    d = pd.read_parquet(path, columns=cols)
    d = d.dropna(subset=["prefers_overt_meanlp"])
    if d.empty:
        return pd.DataFrame()
    # A checkpoint sidecar is the authority for token and epoch coordinates.
    m = ckpt[["checkpoint_step", "tokens_seen", "epoch", "hp_rank", "seed"]].drop_duplicates("checkpoint_step")
    d = d.merge(m, on="checkpoint_step", how="left", validate="many_to_one")
    d = d.dropna(subset=["tokens_seen"])
    if d.empty:
        return pd.DataFrame()
    # First average item-level binary preferences within each cell/category/
    # token.  Checkpoints 0 and 1 can share a token coordinate, so collapsing
    # them here avoids a duplicated vertical segment on the token axis.
    g = (
        d.groupby(
            [
                "cell_id",
                "architecture",
                "intervention",
                "hp_rank",
                "seed",
                "category",
                "tokens_seen",
                "epoch",
            ],
            as_index=False,
        )["prefers_overt_meanlp"]
        .mean()
        .rename(columns={"prefers_overt_meanlp": "preference"})
    )
    g["source"] = source
    return g


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training trajectories and exact checkpoint -1 initialization rows."""
    train_rows: list[pd.DataFrame] = []
    cells: list[dict] = []
    for pair_path in sorted(TRAIN_PAIRS.glob("*.parquet")):
        cell_id = pair_path.stem.removeprefix("cell_id=")
        ckpt_path = TRAIN_CKPTS / pair_path.name
        if not ckpt_path.exists():
            continue
        ckpt = pd.read_parquet(ckpt_path, columns=["checkpoint_step", "tokens_seen", "epoch", "hp_rank", "seed"])
        if ckpt.empty:
            continue
        rows = _aggregate_eval(pair_path, ckpt, "training")
        if rows.empty:
            continue
        train_rows.append(rows)
        first = rows.iloc[0]
        cells.append(
            {
                "cell_id": cell_id,
                "architecture": str(first["architecture"]),
                "intervention": str(first["intervention"]),
                "hp_rank": int(first["hp_rank"]),
                "seed": int(first["seed"]),
                "stage": _cell_stage(ckpt),
            }
        )
    train = pd.concat(train_rows, ignore_index=True)
    cell_meta = pd.DataFrame(cells).drop_duplicates("cell_id")
    train = train.merge(cell_meta[["cell_id", "stage"]], on="cell_id", how="left", validate="many_to_one")
    train["stage_segment"] = np.where(
        train["stage"].eq("early-only"),
        "early-only",
        np.where(train["epoch"] <= 2, "continuation early", "late continuation"),
    )

    # Use one h0 initialization file per architecture/intervention/seed.  The
    # initialization state is independent of HP, so it can be replicated for
    # baseline HP curves without counting the same state five times in the
    # collapsed intervention curves.
    init_rows: list[pd.DataFrame] = []
    init_pat = re.compile(r"^cell_id=(?P<arch>.+)-en-(?P<int>.+)-h(?P<hp>\d+)-s(?P<seed>\d+)\.parquet$")
    chosen: dict[tuple[str, str, int], Path] = {}
    for p in sorted(INIT_PAIRS.glob("*.parquet")):
        m = init_pat.match(p.name)
        if not m or int(m.group("hp")) != 0:
            continue
        key = (m.group("arch"), m.group("int"), int(m.group("seed")))
        chosen.setdefault(key, p)
    for (arch, intervention, seed), p in chosen.items():
        d = pd.read_parquet(
            p,
            columns=["cell_id", "architecture", "intervention", "rep", "checkpoint_step", "category", "prefers_overt_meanlp"],
        )
        d = d[d["checkpoint_step"] == -1].dropna(subset=["prefers_overt_meanlp"])
        if d.empty:
            continue
        d["seed"] = int(seed)
        g = (
            d.groupby(["architecture", "intervention", "seed", "category"], as_index=False)["prefers_overt_meanlp"]
            .mean()
            .rename(columns={"prefers_overt_meanlp": "preference"})
        )
        g["cell_id"] = g["architecture"] + "-en-" + g["intervention"] + "-init-s" + g["seed"].astype(str)
        g["hp_rank"] = 0
        g["checkpoint_step"] = -1
        g["tokens_seen"] = 0.0
        g["epoch"] = -1
        g["source"] = "initialization"
        g["stage"] = "early-only"
        g["stage_segment"] = "early-only"
        init_rows.append(g)
    init = pd.concat(init_rows, ignore_index=True) if init_rows else pd.DataFrame()
    return train, init


def _with_init_for_baseline(train: pd.DataFrame, init: pd.DataFrame) -> pd.DataFrame:
    """Replicate each init state across the baseline HP ranks present for it."""
    b = train[train["intervention"].eq("baseline")]
    ranks = b[["architecture", "seed", "hp_rank"]].drop_duplicates()
    i = init[init["intervention"].eq("baseline")].drop(columns=["hp_rank"])
    i = i.merge(ranks, on=["architecture", "seed"], how="inner")
    return pd.concat([b, i], ignore_index=True)


def _x_ticks(max_tokens: float) -> tuple[list[float], list[str]]:
    vals = [0.0, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    vals = [v for v in vals if v == 0 or v <= max_tokens * 1.1]
    labels = []
    for v in vals:
        if v == 0:
            labels.append("0")
        elif v >= 1e9:
            labels.append(f"{v / 1e9:.0f}B")
        elif v >= 1e6:
            labels.append(f"{v / 1e6:.0f}M")
        else:
            labels.append(f"{v / 1e3:.0f}K")
    return list(np.log10(np.asarray(vals) + 1)), labels


def _plot_frame(ax, title: str, xlabel: bool, ticks: tuple[list[float], list[str]], max_x: float) -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="0.55", ls=":", lw=1)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel("P(overt preferred)")
    ax.set_xlim(-0.1, max_x + 0.03)
    ax.set_xticks(*ticks)
    if xlabel:
        ax.set_xlabel("Tokens seen (log scale; log1p display keeps 0 visible)")


def _draw_series(ax, q: pd.DataFrame, key: str, label: str, color: str) -> None:
    if q.empty:
        return
    s = (
        q.groupby("tokens_seen")
        .agg(mean=("preference", "mean"), std=("preference", "std"), n_cells=("cell_id", "nunique"))
        .reset_index()
        .sort_values("tokens_seen")
    )
    x = np.log10(s["tokens_seen"].to_numpy() + 1)
    y = s["mean"].to_numpy()
    sd = s["std"].fillna(0).to_numpy()
    ax.plot(x, y, color=color, lw=1.8, label=f"{label} (n={q['cell_id'].nunique()})")
    ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=color, alpha=0.13, lw=0)


def _figure(data: pd.DataFrame, arch: str, mode: str, out: Path) -> None:
    d = data[data["architecture"].eq(arch)].copy()
    if d.empty:
        return
    max_tokens = float(d["tokens_seen"].max())
    ticks = _x_ticks(max_tokens)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=True, sharey=True)
    axes = axes.ravel()
    if mode == "baseline":
        title = f"{arch}: baseline trajectories by HP, mean ± SD across seeds"
        groups = [(hp, f"h{hp}", HP_COLORS[hp]) for hp in sorted(d["hp_rank"].dropna().unique())]
    elif mode == "intervention":
        title = f"{arch}: trajectories collapsed across HP and seeds"
        groups = [(c, COND_LABELS[c], COND_COLORS[c]) for c in CONDITIONS]
    else:
        title = f"{arch}: early starts and late continuations"
        groups = [(s, STAGE_LABELS[s], STAGE_COLORS[s]) for s in STAGE_COLORS]
    fig.suptitle(
        title + "\nBinary normalized-likelihood preference; checkpoint −1 is the exact initialization at 0 tokens",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    for i, (ax, category) in enumerate(zip(axes, CATEGORIES)):
        qcat = d[d["category"].eq(category)]
        for key, label, color in groups:
            if mode == "baseline":
                q = qcat[qcat["hp_rank"].eq(int(key))]
            elif mode == "intervention":
                q = qcat[qcat["intervention"].eq(key)]
            else:
                q = qcat[qcat["stage_segment"].eq(key)]
            _draw_series(ax, q, str(key), label, color)
        _plot_frame(ax, CATEGORY_LABELS[category], i >= 6, ticks, np.log10(max_tokens + 1))
    axes[0].legend(loc="lower right", frameon=True, fontsize=8, title="Series")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train, init = _load_data()
    train = train[train["architecture"].isin(ARCHES) & train["intervention"].isin(CONDITIONS)].copy()
    init = init[init["architecture"].isin(ARCHES) & init["intervention"].isin(CONDITIONS)].copy()
    # Do not let initialization evaluations for states without a corresponding
    # local trajectory inflate the collapsed plots.
    init_keys = train[["architecture", "intervention", "seed"]].drop_duplicates()
    init = init.drop(columns=["stage", "stage_segment"]).merge(
        init_keys, on=["architecture", "intervention", "seed"], how="inner"
    )
    stage_keys = train[["architecture", "intervention", "seed", "stage"]].drop_duplicates()
    init = init.drop(columns=["stage"], errors="ignore").merge(
        stage_keys, on=["architecture", "intervention", "seed"], how="inner"
    )
    init["stage_segment"] = np.where(init["stage"].eq("early-only"), "early-only", "continuation early")

    baseline = _with_init_for_baseline(train, init)
    intervention = pd.concat([train, init], ignore_index=True)
    # For stage plots, initialization is represented once per stage group; this
    # makes the start of continuation runs visible without HP duplication.
    stage = pd.concat([train, init], ignore_index=True)
    stage = stage[stage["stage_segment"].isin(STAGE_COLORS)].copy()

    baseline.to_csv(args.out / "baseline_by_hyperparameter_aggregates.csv", index=False)
    intervention.to_csv(args.out / "intervention_collapsed_aggregates.csv", index=False)
    stage.to_csv(args.out / "early_vs_continuation_aggregates.csv", index=False)
    counts = (
        train[["cell_id", "architecture", "stage"]]
        .drop_duplicates()
        .groupby(["architecture", "stage"])
        .size()
        .rename("cells")
        .reset_index()
    )
    counts.to_csv(args.out / "run_stage_counts.csv", index=False)

    for arch in ARCHES:
        _figure(baseline, arch, "baseline", args.out / f"baseline_by_hyperparameter_{arch}.png")
        _figure(intervention, arch, "intervention", args.out / f"intervention_collapsed_{arch}.png")
        _figure(stage, arch, "stage", args.out / f"early_vs_continuation_{arch}.png")

    readme = args.out / "README.md"
    readme.write_text(
        """# Foundry trajectory figures

These figures use the local `null_subj_v2` pair/checkpoint exports.

- `baseline_by_hyperparameter_<arch>.png`: baseline only, h0–h4, equal-weighted across seeds.
- `intervention_collapsed_<arch>.png`: five conditions, collapsed across HP ranks and seeds.
- `early_vs_continuation_<arch>.png`: early-only starts, early segments of continuation runs, and late continuation.

Scores are item-level binary preferences from normalized likelihood comparisons, averaged first within each cell/category/checkpoint and then equally across cells at shared `tokens_seen`. Initialization is checkpoint -1 at exactly 0 tokens; the x-axis uses `log10(tokens_seen + 1)` only to display that zero on a log-like axis. A cell is classified as early-only when its resolved checkpoint metadata ends by epoch 2; otherwise it is a continuation cell.

See `run_stage_counts.csv` for the actual cell counts used.
"""
    )
    print(f"Wrote trajectory figures and aggregates to {args.out}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
