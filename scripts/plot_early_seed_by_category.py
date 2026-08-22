#!/usr/bin/env python3
"""Plot baseline-condition early-only seed trajectories by evaluation category."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "analysis/eval_v2/figures/foundry_trajectories"
INPUT = FIG_DIR / "early_vs_continuation_aggregates.csv"
ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm", "mamba_370m"]
CATEGORIES = [
    "subject_drop", "subject_drop_no_agreement", "expletive", "object_drop",
    "embedded_drop", "extraction", "conjunction", "control",
]
LABELS = {
    "subject_drop": "Subject drop",
    "subject_drop_no_agreement": "Subject drop (no agreement)",
    "expletive": "Expletive",
    "object_drop": "Object drop",
    "embedded_drop": "Embedded drop",
    "extraction": "Extraction",
    "conjunction": "Conjunction",
    "control": "Control",
}


def _attach_initialization(d: pd.DataFrame, full: pd.DataFrame,
                           hp_rank: int) -> pd.DataFrame:
    """Attach checkpoint −1 for the seeds actually present at this HP rank."""
    if d.empty or "source" not in full.columns:
        return d
    seeds = set(d.seed.astype(int).unique())
    init = full[full.source.eq("initialization") &
                full.seed.astype(int).isin(seeds)].copy()
    if init.empty:
        return d
    init = (init.groupby(["architecture", "seed", "category", "tokens_seen"],
                         as_index=False)["preference"].mean())
    init["intervention"] = "baseline"
    init["hp_rank"] = hp_rank
    init["stage_segment"] = "early-only"
    init["source"] = "initialization"
    init["cell_id"] = (
        init["architecture"].astype(str) + "-en-baseline-h" +
        str(hp_rank) + "-init-s" + init["seed"].astype(str)
    )
    base = d[~d.source.eq("initialization")].copy()
    return pd.concat([base, init], ignore_index=True)


def _ticks(max_tokens: float) -> tuple[list[float], list[str]]:
    vals = [0.0, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    vals = [v for v in vals if v == 0 or v <= max_tokens * 1.05]
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


def _bin_category_trajectories(d: pd.DataFrame, n_bins: int = 24) -> pd.DataFrame:
    """Collapse baseline cells within seed/category, then log-bin."""
    seed_tok = (
        d.groupby(["architecture", "seed", "category", "tokens_seen"], as_index=False)["preference"]
        .mean()
        .sort_values(["architecture", "seed", "category", "tokens_seen"])
    )
    out = []
    for arch, q in seed_tok.groupby("architecture", sort=False):
        q = q.copy()
        positive = q.loc[q.tokens_seen > 0, "tokens_seen"]
        edges = np.unique(np.geomspace(float(positive.min()), float(positive.max()), n_bins + 1))
        centers = {0: 0.0}
        for i in range(1, len(edges)):
            centers[i] = float(np.sqrt(edges[i - 1] * edges[i]))
        q["bin_index"] = np.digitize(q.tokens_seen.to_numpy(), edges, right=False)
        q["bin_center"] = q.bin_index.map(centers)
        q = (
            q.groupby(["architecture", "seed", "category", "bin_index", "bin_center"], as_index=False)["preference"]
            .mean()
        )
        out.append(q)
    return pd.concat(out, ignore_index=True)


def _plot_arch(d: pd.DataFrame, arch: str, out: Path,
               title_prefix: str = "baseline") -> None:
    q = d[d.architecture.eq(arch)].copy()
    q = _bin_category_trajectories(q)
    max_tokens = float(q.bin_center.max())
    xticks, xlabels = _ticks(max_tokens)
    seeds = sorted(int(x) for x in q.seed.unique())
    palette = plt.cm.tab20(np.linspace(0, 1, max(2, len(seeds))))
    seed_colors = dict(zip(seeds, palette))
    arch_color = plt.cm.Dark2.colors[ARCHES.index(arch)]
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=True, sharey=True)
    axes = axes.ravel()
    fig.suptitle(
        f"{arch}: {title_prefix} early starts by seed and evaluation category (n={len(seeds)} seeds)\n"
        "Thin lines = individual seeds; heavy line = mean ± 1 SE across seeds",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    for ax, category in zip(axes, CATEGORIES):
        z = q[q.category.eq(category)]
        for seed, s in z.groupby("seed", sort=True):
            ax.plot(np.log10(s.bin_center.to_numpy() + 1), s.preference,
                    color=seed_colors[int(seed)], lw=0.8, alpha=0.48)
        stats = (
            z.groupby("bin_center")["preference"]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
            .sort_values("bin_center")
        )
        x = np.log10(stats.bin_center.to_numpy() + 1)
        y = stats["mean"].to_numpy()
        se = stats["sd"].fillna(0).to_numpy() / np.sqrt(stats["n"].to_numpy())
        ax.fill_between(x, np.clip(y - se, 0, 1), np.clip(y + se, 0, 1), color=arch_color, alpha=0.22, lw=0)
        ax.plot(x, y, color=arch_color, lw=2.3)
        ax.set_title(LABELS[category], loc="left", fontweight="bold", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color="0.55", ls=":", lw=1)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("P(overt preferred)")
        ax.set_xticks(xticks, xlabels)
        ax.set_xlim(-0.1, np.log10(max_tokens + 1) + 0.03)
    for ax in axes[-2:]:
        ax.set_xlabel("Tokens seen (log scale; log1p display keeps 0 visible)")
    handles = [plt.Line2D([0], [0], color=seed_colors[s], lw=1.5, label=f"s{s}") for s in seeds]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=8, title="Seed")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    q.to_csv(out.with_suffix(".csv"), index=False)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hp-rank", type=int, choices=range(5),
                    help="Also write a separate baseline plot for this HP rank.")
    args = ap.parse_args()
    d = pd.read_csv(INPUT)
    d = d[d.intervention.eq("baseline") & d.architecture.isin(ARCHES) &
          d.category.isin(CATEGORIES)].copy()
    d["architecture"] = d.architecture.astype(str)
    d["seed"] = d.seed.astype(int)
    d["tokens_seen"] = d.tokens_seen.astype(float)
    if args.hp_rank is None:
        q = d[d.stage_segment.eq("early-only")].copy()
        for arch in ARCHES:
            _plot_arch(q, arch, FIG_DIR / f"early_starts_by_seed_category_{arch}.png")
    else:
        q = d[(d.hp_rank.eq(args.hp_rank)) &
              d.stage_segment.isin(["early-only", "continuation early"])].copy()
        q = _attach_initialization(q, d, args.hp_rank)
        for arch in ARCHES:
            _plot_arch(
                q, arch,
                FIG_DIR / f"early_starts_by_seed_category_h{args.hp_rank}_{arch}.png",
                title_prefix=f"baseline h{args.hp_rank}",
            )
    readme = FIG_DIR / "README.md"
    text = readme.read_text() if readme.exists() else "# Foundry trajectory figures\n"
    addition = (
        "\n## Category-resolved early starts\n\n"
        "- `early_starts_by_seed_category_<arch>.png`: eight evaluation-category panels per architecture for the baseline condition, with individual early-only seed trajectories and mean ±1 SE.\n"
    )
    if "## Category-resolved early starts" not in text:
        readme.write_text(text.rstrip() + "\n" + addition)
    print(f"Wrote category-resolved early-start figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
