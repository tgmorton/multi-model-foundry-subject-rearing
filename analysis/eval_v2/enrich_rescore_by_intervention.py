#!/usr/bin/env python3
"""PDF: the grammar-intervention wave (no pronoun removal) with enrichment on the fixed items.

The enrichment condition's null items had lost verb agreement in the first matched stimuli
(docs/ENRICH_NULL_AGREEMENT_FIX.md). The runs were re-scored on the fixed items (benchmark
null_subj_v2_condition_matched_v2); the other four conditions are unchanged and stay on
null_subj_v2_condition_matched_v1. This puts the five conditions side by side for every
architecture, with the old enrichment scores kept as a faint reference.

Runs used: the full-length seeds at the winning hyperparameter setting (h0, 30 epochs), so
intervals are across seeds.

    python analysis/eval_v2/enrich_rescore_by_intervention.py --data <main>/data/eval_results \\
        --out enrichment_rescore_by_intervention.pdf --export <dir for deck CSVs>
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

V1, V2 = "null_subj_v2_condition_matched_v1", "null_subj_v2_condition_matched_v2"
ARCHS = ["gpt2_small", "gpt2_medium", "gpt2_large", "lstm", "mamba_370m"]
ARCH_LABEL = {"gpt2_small": "GPT-2 small", "gpt2_medium": "GPT-2 medium", "gpt2_large": "GPT-2 large",
              "lstm": "LSTM", "mamba_370m": "Mamba (370M)"}
CONDS = ["baseline", "impoverish_case", "lemmatize_verbs", "remove_expletive_sentences", "enrich_verbal_morphology"]
# series shown: the five conditions, enrichment on the fixed items, plus enrichment on the old items
SERIES = ["baseline", "impoverish_case", "lemmatize_verbs", "remove_expletive_sentences", "enrich_fixed", "enrich_old"]
LABEL = {"baseline": "baseline", "impoverish_case": "impoverished case", "lemmatize_verbs": "lemmatized verbs",
         "remove_expletive_sentences": "expletive sentences removed", "enrich_fixed": "enriched verbal morphology",
         "enrich_old": "enrichment, old items (superseded)"}
SHORT = {"baseline": "baseline", "impoverish_case": "impov. case", "lemmatize_verbs": "lemm. verbs",
         "remove_expletive_sentences": "expl. removed", "enrich_old": "enrich., old items", "enrich_fixed": "enrich., fixed"}
COLOR = {"baseline": D.CTRL, "impoverish_case": D.BLUE, "lemmatize_verbs": D.AQUA,
         "remove_expletive_sentences": D.YELLOW, "enrich_fixed": D.ORANGE, "enrich_old": D.ORANGE}
SUBJ = ("subject_drop", "subject_drop_no_agreement")
OVERALL_EXCLUDE = ("extraction", "object_drop")
GROUPS = ["subject_drop", "subject_drop_no_agreement", "expletive", "conjunction", "embedded_drop", "control",
          "extraction", "object_drop"]
GROUP_LABEL = {"subject_drop": "subject drop", "subject_drop_no_agreement": "subject drop, no agreement",
               "expletive": "expletive", "conjunction": "conjunction", "embedded_drop": "embedded",
               "control": "control", "extraction": "extraction", "object_drop": "object drop"}
RUN = re.compile(r"^(?P<arch>[a-z0-9_]+)-en-(?P<cond>[a-z_]+)-h(?P<hp>\d+)-s(?P<seed>\d+)$")
SECTION = "Grammar interventions across architectures · enrichment on the fixed items"
FULL_TOKENS = 3.0e9
EPOCH_TOKENS = 128e6


def tval(n: int) -> float:
    """Two-sided 95% t multiplier for n runs."""
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    except ImportError:
        return {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}.get(n, 1.96)


def ci(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return tval(len(x)) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def load_run(data: Path, bench: str, run: str) -> pd.DataFrame | None:
    """Per checkpoint: tokens seen, preference on subject-drop targets, overall, and per item group."""
    f = data / bench / "pairs" / f"cell_id={run}.parquet"
    c = data / bench / "checkpoints" / f"cell_id={run}.parquet"
    if not f.exists() or not c.exists():
        return None
    p = pd.read_parquet(f, columns=["checkpoint_step", "category", "prefers_overt_meanlp"])
    p["pref"] = p.prefers_overt_meanlp.astype(float)
    g = p.pivot_table(index="checkpoint_step", columns="category", values="pref", aggfunc="mean")
    g["subj"] = p[p.category.isin(SUBJ)].groupby("checkpoint_step").pref.mean()
    g["overall"] = p[~p.category.isin(OVERALL_EXCLUDE)].groupby("checkpoint_step").pref.mean()
    ck = pd.read_parquet(c, columns=["checkpoint_step", "tokens_seen"]).set_index("checkpoint_step").tokens_seen
    g["tokens"] = ck.reindex(g.index)
    return g.reset_index()


def collect(data: Path) -> pd.DataFrame:
    rows = []
    v1 = {f.stem.split("=", 1)[1] for f in (data / V1 / "pairs").glob("cell_id=*.parquet")}
    v2 = {f.stem.split("=", 1)[1] for f in (data / V2 / "pairs").glob("cell_id=*.parquet")}
    for run in sorted(v1 | v2):
        m = RUN.match(run)
        if not m or m["arch"] not in ARCHS or m["cond"] not in CONDS or int(m["hp"]) != 0:
            continue
        todo = []
        if m["cond"] == "enrich_verbal_morphology":
            todo += [("enrich_old", V1)] if run in v1 else []
            todo += [("enrich_fixed", V2)] if run in v2 else []
        elif run in v1:
            todo.append((m["cond"], V1))
        for series, bench in todo:
            g = load_run(data, bench, run)
            if g is None or g.tokens.max() < FULL_TOKENS:
                continue
            g["arch"], g["series"], g["seed"] = m["arch"], series, int(m["seed"])
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def endstate(d: pd.DataFrame, cols, last: int = 3) -> pd.DataFrame:
    """Mean of each run's last `last` checkpoints (the project's end-state convention)."""
    out = []
    for (arch, series, seed), g in d.groupby(["arch", "series", "seed"]):
        g = g[g.checkpoint_step > 0].sort_values("checkpoint_step").tail(last)
        out.append({"arch": arch, "series": series, "seed": seed, **g[cols].mean().to_dict()})
    return pd.DataFrame(out)


def trajectory(d: pd.DataFrame, value: str) -> pd.DataFrame:
    """Across-seed mean and 95% interval at each checkpoint (step > 0), per architecture and series."""
    g = d[d.checkpoint_step > 0].groupby(["arch", "series", "checkpoint_step"])
    t = g.agg(tokens=("tokens", "median"), mean=(value, "mean"), half=(value, ci), n=(value, "count")).reset_index()
    t["lower"], t["upper"] = t["mean"] - t.half, t["mean"] + t.half
    return t


# ----------------------------------------------------------------------------- pages

def p_endstate(doc, es, archs, value, title, lede, notes):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, title, lede)
    gs = fig.add_gridspec(1, len(archs), **{**box, "bottom": 0.40}, wspace=0.10)
    order = ["baseline", "impoverish_case", "lemmatize_verbs", "remove_expletive_sentences", "enrich_old", "enrich_fixed"]
    for j, arch in enumerate(archs):
        ax = fig.add_subplot(gs[0, j])
        a = es[es.arch == arch]
        for k, s in enumerate(order):
            v = a[a.series == s][value].to_numpy()
            if not len(v):
                ax.text(k, 0.06, "not yet\nre-scored", ha="center", va="bottom", size=6.4, color=D.MUTED, linespacing=1.2)
                continue
            old = s == "enrich_old"
            rng = np.random.default_rng(k)
            ax.scatter(k + rng.uniform(-0.13, 0.13, len(v)), v, s=9, color=COLOR[s], alpha=0.25 if old else 0.38,
                       linewidths=0, zorder=2)
            h = ci(v)
            ax.plot([k, k], [v.mean() - h, v.mean() + h], color=COLOR[s], lw=2.0, alpha=0.55 if old else 1,
                    solid_capstyle="round", zorder=3)
            ax.plot([k], [v.mean()], marker="o", ms=7.5, mfc="white" if old else COLOR[s], mec=COLOR[s],
                    mew=1.6 if old else 1.2, zorder=4)
            ax.annotate(f"{v.mean():.2f}", (k, min(v.mean() + (h if np.isfinite(h) else 0), 0.97)), xytext=(0, 6),
                        textcoords="offset points", ha="center", size=6.8, color=D.INK2)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=D.MUTED, lw=0.8, zorder=1)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_xticks(range(len(order)), [SHORT[s] for s in order], fontsize=6.6, rotation=40, ha="right",
                      rotation_mode="anchor")
        ax.grid(axis="x", visible=False)
        ax.set_title(ARCH_LABEL[arch])
        if j == 0:
            ax.set_ylabel("share of pairs preferring the overt form\n(end state, mean of the last three checkpoints)")
        else:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], color=D.INK2, lw=2, marker="o", ms=7, mfc=D.INK2, label="mean of the seeds, 95% interval across seeds"),
                        Line2D([], [], color=D.INK2, lw=0, marker="o", ms=3.5, alpha=0.45, label="one seed"),
                        Line2D([], [], color=D.ORANGE, lw=2, marker="o", ms=7, mfc="white", alpha=0.7,
                               label="enrichment on the old items (superseded)")],
               loc="upper left", bbox_to_anchor=(0.075, 0.27), ncol=3, fontsize=7.8)
    doc.bottom_notes(fig, notes, 0.21)
    doc.save(fig)


def p_traj(doc, d, archs, value, title, lede, notes):
    from matplotlib.lines import Line2D
    fig, box = doc.page(SECTION, title, lede)
    ncol = 3
    nrow = -(-len(archs) // ncol)
    gs = fig.add_gridspec(nrow, ncol, **{**box, "bottom": 0.21}, wspace=0.12, hspace=0.36)
    t = trajectory(d, value)
    for k, arch in enumerate(archs):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        a = t[t.arch == arch]
        for s in ["baseline", "impoverish_case", "lemmatize_verbs", "remove_expletive_sentences", "enrich_old", "enrich_fixed"]:
            g = a[a.series == s].sort_values("tokens")
            if g.empty:
                continue
            x = g.tokens / 1e6
            old = s == "enrich_old"
            if not old:
                ax.fill_between(x, g.lower, g.upper, color=COLOR[s], alpha=0.13, lw=0)
            ax.plot(x, g["mean"], color=COLOR[s], lw=1.3 if old else 1.9, ls=(0, (2, 1.6)) if old else "-",
                    alpha=0.75 if old else 1)
        ax.axvline(EPOCH_TOKENS / 1e6, color=D.MUTED, lw=0.7, ls=(0, (1, 2)))
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=D.MUTED, lw=0.8, zorder=1)
        D.tokens_axis(ax, label=k // ncol == nrow - 1 or k + ncol >= len(archs))
        ax.set_title(ARCH_LABEL[arch])
        if k % ncol == 0:
            ax.set_ylabel("share preferring overt")
        else:
            ax.set_yticklabels([])
    ax = fig.add_subplot(gs[nrow - 1, ncol - 1]) if len(archs) % ncol else None
    handles = [Line2D([], [], color=COLOR[s], lw=1.9, label=LABEL[s]) for s in SERIES[:5]]
    handles += [Line2D([], [], color=D.ORANGE, lw=1.3, ls=(0, (2, 1.6)), alpha=0.75, label=LABEL["enrich_old"]),
                Line2D([], [], color=D.MUTED, lw=0.7, ls=(0, (1, 2)), label="end of the first epoch")]
    if ax is not None:
        ax.axis("off")
        ax.legend(handles=handles, loc="center left", fontsize=8.2, handlelength=2.4)
    else:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.17), ncol=4, fontsize=7.8)
    doc.bottom_notes(fig, notes, 0.135)
    doc.save(fig)


def p_groups(doc, es, archs, title, lede, notes):
    """Enrichment by item group: fixed items, old items, and baseline, per architecture."""
    fig, box = doc.page(SECTION, title, lede)
    ncol = 3
    nrow = -(-len(archs) // ncol)
    gs = fig.add_gridspec(nrow, ncol, **{**box, "bottom": 0.20, "left": 0.20}, wspace=0.30, hspace=0.42)
    cols = ["baseline", "impoverish_case", "lemmatize_verbs", "remove_expletive_sentences", "enrich_old", "enrich_fixed"]
    for k, arch in enumerate(archs):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        a = es[es.arch == arch].groupby("series")[GROUPS].mean()
        M = np.array([[a.loc[s, g] if s in a.index else np.nan for s in cols] for g in GROUPS])
        from eval_recoverability_testset_results import DIV
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("div", DIV)
        cmap.set_bad("#f7f6f3")
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}".replace("0.", ".", 1), ha="center", va="center", size=6.4,
                        color=D.MUTED if np.isnan(v) else ("white" if abs(v - 0.5) > 0.28 else D.INK))
        ax.set_xticks(range(len(cols)), [SHORT[s] for s in cols], rotation=38, ha="right", fontsize=6.4)
        ax.set_yticks(range(len(GROUPS)), [GROUP_LABEL[g] for g in GROUPS] if k % ncol == 0 else [""] * len(GROUPS), fontsize=7)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(GROUPS)), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(ARCH_LABEL[arch])
    doc.bottom_notes(fig, notes, 0.115)
    doc.save(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="directory holding the two benchmark folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export", type=Path, default=None, help="write the plotted series as CSV here")
    a = ap.parse_args()
    plt = D.setup_mpl()
    d = collect(a.data)
    archs = [x for x in ARCHS if x in set(d.arch)]
    es = endstate(d, ["subj", "overall"] + GROUPS)
    seeds = es.groupby(["arch", "series"]).seed.nunique().unstack()
    print(seeds.to_string())
    m = es.groupby(["arch", "series"]).subj.mean().unstack()
    D.FOOT = ("grammar-intervention wave · condition-matched eval · full-length seeds at the winning setting (h0) · "
              "enrich_rescore_by_intervention.py")
    doc = D.Doc(a.out, plt)

    def f(arch, s):
        return f"{m.loc[arch, s]:.2f}" if s in m.columns and np.isfinite(m.loc[arch, s]) else "–"

    done = [x for x in archs if "enrich_fixed" in m.columns and np.isfinite(m.loc[x, "enrich_fixed"])]
    pending = [x for x in archs if x not in done]
    doc.text_page(
        SECTION, "Enrichment re-scored, next to the other interventions",
        "The grammar-intervention wave: five training conditions, no pronoun removal. Enrichment is now scored on the "
        "items whose null version keeps verb agreement; the other four conditions were never affected.",
        [[("What changed", [
            "In the first condition-matched stimuli the null version of an enrichment item had lost its verb agreement, "
            "so 'overt against null' was partly 'agreeing against non-agreeing'. The fix restores agreement on the null "
            "side. Overt sentences are untouched, and on them the new scores match the August ones to 0.001 nats per token.",
            "Re-scored: " + ", ".join(ARCH_LABEL[x] for x in done) + "."
            + (f" Not yet: {', '.join(ARCH_LABEL[x] for x in pending)} (needs a GPU; queued)." if pending else "")]),
          ("What is plotted", [
            "Runs: the full-length seeds at the winning hyperparameter setting, "
            + ", ".join((f"{int(seeds.loc[x].min())} to {int(seeds.loc[x].max())}" if seeds.loc[x].min() != seeds.loc[x].max()
                         else f"{int(seeds.loc[x].max())}") + f" for {ARCH_LABEL[x]}" for x in archs)
            + " per condition. Intervals are across seeds.",
            "Measure: share of pairs where the sentence with its subject pronoun scores higher than the same sentence "
            "without it, on the subject-drop items unless a page says otherwise. End state is the mean of a run's last "
            "three checkpoints.",
            "The old-items enrichment scores are kept in outline or dashed, as a reference for what the fix moved."])],
         [("Subject-drop items, end state", [
            *[f"{ARCH_LABEL[x]}: baseline {f(x, 'baseline')}, impoverished case {f(x, 'impoverish_case')}, lemmatized "
              f"verbs {f(x, 'lemmatize_verbs')}, expletives removed {f(x, 'remove_expletive_sentences')}, enrichment "
              f"{f(x, 'enrich_fixed')} (old items {f(x, 'enrich_old')})." for x in archs]]),
          ("Read with care", [
            "A handful of GPT-2 medium checkpoints are damaged on disk and missing from the re-score; none belongs to "
            "the seeds plotted here.",
            "The old BERT runs are not shown: they predate the recipe fix and are not valid."])]])
    p_endstate(doc, es, archs, "subj",
               "Where each intervention ends, by architecture",
               "Subject-drop items. One panel per architecture; each column is a training condition. Enrichment appears "
               "twice: on the old items (outline) and on the fixed items (filled).",
               [f"Fixed minus old enrichment: " + "; ".join(
                    f"{ARCH_LABEL[x]} {m.loc[x, 'enrich_fixed'] - m.loc[x, 'enrich_old']:+.2f}" for x in done) + ".",
                "Enrichment minus baseline, fixed items: " + "; ".join(
                    f"{ARCH_LABEL[x]} {m.loc[x, 'enrich_fixed'] - m.loc[x, 'baseline']:+.2f}" for x in done) + ".",
                "Each small dot is one seed; where they sit tightly the interval is narrow."])
    p_traj(doc, d, archs, "subj",
           "Over training, by architecture",
           "Subject-drop items. Lines are the mean over seeds and bands the 95% interval across seeds, at the checkpoints "
           "every run shares. The x axis is training tokens on a log scale.",
           ["Solid orange is enrichment on the fixed items; dashed orange is the same runs on the old items.",
            "Checkpoints sit at the same token counts for every architecture, so panels can be compared at matched data."])
    p_traj(doc, d, archs, "overall",
           "The same, over all item groups except extraction and object drop",
           "The project's overall measure: subject drop, subject drop without agreement, expletive, conjunction, embedded "
           "and control items pooled.",
           ["The fix changed 256 of 576 null items, so the overall measure moves less than the subject-drop one.",
            "Layout and colours as on the previous page."])
    p_groups(doc, es, archs,
             "End state by item group",
             "Share preferring the overt form at the end of training, for each item group (rows) and training condition "
             "(columns). Blue prefers the pronoun, orange the null version.",
             ["Compare the last two columns of a panel to see which item groups the fix moved.",
              "The fix moves only the two subject-drop rows; the other rows are identical, because their null items "
              "were not changed.",
              "Control items come out low under every condition: there the version without the pronoun wins."])
    doc.close()
    print(f"wrote {a.out} ({doc.n} pages)")
    if a.export:
        a.export.mkdir(parents=True, exist_ok=True)
        for value in ("subj", "overall"):
            t = trajectory(d, value)
            t["architecture"], t["condition"] = t.arch.map(ARCH_LABEL), t.series.map(LABEL)
            t.rename(columns={"tokens": "x_tokens_seen", "n": "n_seeds"})[
                ["arch", "architecture", "series", "condition", "checkpoint_step", "x_tokens_seen", "mean", "lower", "upper",
                 "n_seeds"]].to_csv(a.export / f"part1_trajectories_{value}.csv", index=False)
        e = es.melt(id_vars=["arch", "series", "seed"], value_vars=["subj", "overall"] + GROUPS, var_name="items",
                    value_name="pref")
        s = e.groupby(["arch", "series", "items"]).pref.agg(mean="mean", half=ci, n_seeds="count").reset_index()
        s["lower"], s["upper"] = s["mean"] - s.half, s["mean"] + s.half
        s["architecture"], s["condition"] = s.arch.map(ARCH_LABEL), s.series.map(LABEL)
        s.drop(columns="half").to_csv(a.export / "part1_endstate.csv", index=False)
        e.to_csv(a.export / "part1_endstate_by_seed.csv", index=False)
        (a.export / "README_part1.txt").write_text(
            "Grammar-intervention wave (no pronoun removal), condition-matched eval.\n"
            "Runs: full-length seeds at the winning hyperparameter setting (h0). Intervals: 95% t-interval across seeds.\n"
            "series enrich_fixed = enrichment on the fixed items (benchmark ..._matched_v2); enrich_old = same runs on the\n"
            "old items (superseded, do not present as a result). Other series are on ..._matched_v1.\n"
            "items: subj = subject_drop + subject_drop_no_agreement; overall = all groups except extraction and object_drop.\n"
            f"x_tokens_seen is training tokens; one epoch is {EPOCH_TOKENS:.0f} tokens; chance level 0.5.\n"
            "End state = mean of a run's last three checkpoints.\n")
        print(f"exported series to {a.export}")


if __name__ == "__main__":
    main()
