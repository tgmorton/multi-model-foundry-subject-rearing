#!/usr/bin/env python3
"""Seed-to-seed check on the Study 2 end state (GPT-2 large, matched evaluation).

First seeds: full trajectories on null_subj_v2_condition_matched_{v1,v2}; end state = mean of the last three
checkpoints. Second seeds: the same three checkpoint positions, scored on ..._{v1,v2}_endstate.
Enrichment cells use the v2 (fixed) items for both seeds.

Writes s2_seed_check_cells.csv (one row per cell and item pooling: seed 1, seed 2, difference) and
s2_seed_check_order_effect.csv (most minus least recoverable first, per condition and percent, each seed),
and prints the summary.

    python analysis/eval_v2/seed_check_endstate.py --data <main>/data/eval_results --out <deck_exports dir>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN = re.compile(r"^gpt2_large-en-pdrop2_(?P<arm>robbianti|robbi|rand|all)(?P<k>\d+)_(?P<iv>base|impcase|lemverb|rmexpl|enrichvm)-h0-s(?P<seed>\d+)$")
ARM = {"rand": "random removal", "robbi": "most recoverable first", "robbianti": "least recoverable first", "all": "all removed"}
IV = {"base": "baseline", "impcase": "impoverished case", "lemverb": "lemmatized verbs", "rmexpl": "expletive sentences removed",
      "enrichvm": "enriched verbal morphology"}
SUBJ = ["subject_drop", "subject_drop_no_agreement"]
OVERALL = SUBJ + ["expletive", "conjunction", "embedded_drop", "control"]
POOLS = {"subject drop (2 groups, 288 pairs)": SUBJ, "overall (6 groups, 480 pairs)": OVERALL}


def endstate(f: Path) -> pd.Series | None:
    """Per item: mean preference over the run's last three scored checkpoints."""
    p = pd.read_parquet(f, columns=["checkpoint_step", "category", "condition", "item_id", "prefers_overt_meanlp",
                                    "log_prob_diff_overt_minus_null"])
    dead = p.groupby("checkpoint_step").log_prob_diff_overt_minus_null.apply(lambda x: (x == 0).all())
    p = p[~p.checkpoint_step.isin(dead[dead].index)]
    steps = sorted(p.checkpoint_step.unique())[-3:]
    if len(steps) < 3:
        return None
    q = p[p.checkpoint_step.isin(steps)]
    return q.groupby(["category", "condition", "item_id"]).prefers_overt_meanlp.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cells: dict = {}
    for v in ("v1", "v2"):
        for bench, which in ((f"null_subj_v2_condition_matched_{v}", 1), (f"null_subj_v2_condition_matched_{v}_endstate", 2)):
            for f in sorted((a.data / bench / "pairs").glob("cell_id=gpt2_large-en-pdrop2_*.parquet")):
                m = RUN.match(f.stem.split("=", 1)[1])
                if not m or ((m["iv"] == "enrichvm") != (v == "v2")):
                    continue
                e = endstate(f)
                if e is not None:
                    cells.setdefault((m["iv"], m["arm"], int(m["k"])), {})[which] = e
    rows = []
    for (iv, arm, k), d in sorted(cells.items()):
        if 1 not in d or 2 not in d:
            continue
        for pool, cats in POOLS.items():
            s1 = d[1][d[1].index.get_level_values("category").isin(cats)]
            s2 = d[2].reindex(s1.index)
            diff = (s2 - s1).to_numpy()
            rng = np.random.default_rng(0)
            b = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(1)
            rows.append(dict(figure="s2_seed_check_cells", architecture="GPT-2 large", condition=IV[iv], arm=ARM[arm], percent=k,
                             item_group=pool, seed1=s1.mean(), seed2=s2.mean(), seed2_minus_seed1=diff.mean(),
                             lower=np.percentile(b, 2.5), upper=np.percentile(b, 97.5), n_items=len(diff),
                             item_correlation=float(np.corrcoef(s1, s2)[0, 1]),
                             interval_kind="difference of two seeds, paired by item; bootstrap over items (95%)"))
    c = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    c.to_csv(a.out / "s2_seed_check_cells.csv", index=False)

    # order effect per condition and percent, each seed
    eff = []
    for pool in POOLS:
        g = c[c.item_group == pool].pivot_table(index=["condition", "percent"], columns="arm", values=["seed1", "seed2"])
        for (cond, k), r in g.iterrows():
            for seed in ("seed1", "seed2"):
                try:
                    m_, l_, r_ = r[(seed, ARM["robbi"])], r[(seed, ARM["robbianti"])], r[(seed, ARM["rand"])]
                except KeyError:
                    continue
                eff.append(dict(figure="s2_seed_check_order_effect", condition=cond, percent=k, item_group=pool, seed=seed,
                                most_minus_least=m_ - l_, most_minus_random=m_ - r_, least_minus_random=l_ - r_))
    e = pd.DataFrame(eff).dropna(subset=["most_minus_least"])
    e.to_csv(a.out / "s2_seed_check_order_effect.csv", index=False)

    pd.set_option("display.width", 220)
    s = c[c.item_group.str.startswith("subject drop")]
    print(f"cells with both seeds: {len(s)} of 140")
    print(f"subject-drop end state: correlation of seed 1 and seed 2 across cells {np.corrcoef(s.seed1, s.seed2)[0, 1]:.3f}; "
          f"mean |difference| {s.seed2_minus_seed1.abs().mean():.3f}; mean difference {s.seed2_minus_seed1.mean():+.3f}; "
          f"largest |difference| {s.seed2_minus_seed1.abs().max():.2f}")
    print(f"cells whose difference interval excludes zero: {int(((s.lower > 0) | (s.upper < 0)).sum())} of {len(s)}")
    print(s.groupby("arm").agg(cells=("seed1", "size"), seed1=("seed1", "mean"), seed2=("seed2", "mean"),
                               mean_abs_diff=("seed2_minus_seed1", lambda x: x.abs().mean())).round(3).to_string())
    w = e[e.item_group.str.startswith("subject drop")].pivot_table(index=["condition", "percent"], columns="seed", values="most_minus_least").dropna()
    print(f"\nmost minus least recoverable first, {len(w)} (condition, percent) cells with both seeds:")
    print(f"  seed 1 mean {w.seed1.mean():+.3f}, seed 2 mean {w.seed2.mean():+.3f}; correlation {np.corrcoef(w.seed1, w.seed2)[0, 1]:.3f}; "
          f"positive in seed 1: {int((w.seed1 > 0).sum())}, in seed 2: {int((w.seed2 > 0).sum())} of {len(w)}")
    print(w.groupby(level="condition").mean().round(3).to_string())
    print(w.groupby(level="percent").mean().round(3).to_string())


if __name__ == "__main__":
    main()
