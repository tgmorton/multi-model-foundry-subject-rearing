#!/usr/bin/env python3
"""Deck export: Study 1 item-group trajectories by training condition, and GPT-2 sizes pooled.

  s1_item_group_by_condition_trajectories.csv
      every item group (the eight; control split into subject and object control; conjunction split into
      same-subject and topic-shift) x five training conditions x each architecture, every checkpoint,
      plus rows with architecture = "GPT-2 (small, medium, large pooled)".

Also appends the pooled-GPT-2 rows to s1_baseline_trajectories.csv and s1_condition_trajectories.csv
(run this after deck_exports.py, which rewrites those two files without them).

Same conventions as deck_exports.py: full-length seeds at the winning hyperparameter setting, the early-only
seeds added up to 256M tokens, the untrained model at tokens_seen = 0, enrichment from the fixed items.

Pooled rows: all seeds of the three sizes are treated as one sample (mean and 95% t-interval across seeds).
Checkpoints are aligned by tokens_seen: a pooled row exists at every token count that all three sizes'
full-length runs share (within 1%); each seed contributes the checkpoint it has within 1% of that count.

    python analysis/eval_v2/deck_exports_item_groups.py --data <main>/data/eval_results --out <deck_exports dir> \\
        [--skip mamba_370m:enrich_verbal_morphology]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_exports as X  # noqa: E402

POOLED = ["gpt2_small", "gpt2_medium", "gpt2_large"]
POOLED_LABEL = "GPT-2 (small, medium, large pooled)"
# label -> (categories, condition names within the category or None for all)
ITEM_GROUPS = {**{lab: ([cat], None) for cat, lab in X.GROUPS.items()},
               "control: subject control": (["control"], ["subj_control"]),
               "control: object control": (["control"], ["obj_control"]),
               "conjunction: same subject": (["conjunction"], ["conj_no_shift"]),
               "conjunction: topic shift": (["conjunction"], ["conj_shift"])}
TOL = 0.01


def pick(p: pd.DataFrame, cats, conds) -> pd.DataFrame:
    q = p[p.category.isin(cats)]
    return q if conds is None else q[q.condition.isin(conds)]


def by_tokens(p: pd.DataFrame) -> pd.Series:
    return p.groupby("tokens_seen").pref.mean().sort_index()


def shared_grid(runs: dict, cond: str) -> np.ndarray:
    """Token counts present (within TOL) in a full-length run of each pooled size."""
    grids = []
    for arch in POOLED:
        g = [np.sort(p.tokens_seen.unique()) for (a, c, _), p in runs.items() if a == arch and c == cond]
        if not g:
            return np.zeros(0)
        grids.append(max(g, key=len))
    out = []
    for t in grids[0]:
        ok = all(np.min(np.abs(g - t)) <= TOL * max(t, 1.0) for g in grids[1:])
        if ok:
            out.append(t)
    return np.asarray(out)


def pooled(runs: dict, early: dict, cond: str, cats, conds, min_runs: int = 6) -> pd.DataFrame:
    grid = shared_grid(runs, cond)
    if not len(grid):
        return pd.DataFrame()
    cols, kind, size = {}, {}, {}
    for source, label in ((runs, "full"), (early, "early")):
        for (arch, c, seed), p in source.items():
            if arch not in POOLED or c != cond:
                continue
            s = by_tokens(pick(p, cats, conds))
            if s.empty:
                continue
            t = s.index.to_numpy(float)
            v = np.full(len(grid), np.nan)
            for k, g in enumerate(grid):
                j = int(np.argmin(np.abs(t - g)))
                if abs(t[j] - g) <= TOL * max(g, 1.0):
                    v[k] = s.iloc[j]
            cols[(arch, seed)] = v
            kind[(arch, seed)], size[(arch, seed)] = label, arch
    if not cols:
        return pd.DataFrame()
    M = pd.DataFrame(cols, index=grid)
    n = M.notna().sum(axis=1)
    M, n = M[n >= min_runs], n[n >= min_runs]
    mean = M.mean(axis=1)
    half = np.array([X.tval(int(k)) for k in n]) * M.std(axis=1, ddof=1).to_numpy() / np.sqrt(n.to_numpy())
    nf = M[[k for k in M.columns if kind[k] == "full"]].notna().sum(axis=1)
    per = {a: M[[k for k in M.columns if size[k] == a]].notna().sum(axis=1) for a in POOLED}

    def describe(i):
        k, f = int(n.iloc[i]), int(nf.iloc[i])
        sizes = ", ".join(f"{int(per[a].iloc[i])} {X.ARCHS[a].split()[-1]}" for a in POOLED)
        tail = f"{f} full-length" + (f" + {k - f} that stop after two epochs or earlier" if k > f else "")
        return f"across {k} seeds pooled over three GPT-2 sizes ({sizes}; 95% t-interval): {tail}"

    return pd.DataFrame({"checkpoint_step": -1, "tokens_seen": M.index.to_numpy(), "mean": mean.to_numpy(),
                         "lower": mean.to_numpy() - half, "upper": mean.to_numpy() + half, "n_runs": n.to_numpy(),
                         "interval_kind": [describe(i) for i in range(len(M))]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip", nargs="*", default=[], help="arch:condition cells to leave out (e.g. an unverified re-score)")
    ap.add_argument("--only-pooled-sizes", action="store_true", help="per-architecture rows for the three GPT-2 sizes only")
    a = ap.parse_args()
    skip = {tuple(s.split(":")) for s in a.skip}
    runs, _, early = X.study1_runs(a.data)
    runs = {k: p for k, p in runs.items() if (k[0], k[1]) not in skip}
    early = {k: p for k, p in early.items() if (k[0], k[1]) not in skip}

    def tag(df, **kw):
        for k, v in kw.items():
            df[k] = v
        return df[X.COLS]

    rows = []
    archs = {k: v for k, v in X.ARCHS.items() if not a.only_pooled_sizes or k in POOLED}
    for cond, clabel in X.CONDS.items():
        for glabel, (cats, conds) in ITEM_GROUPS.items():
            for arch, alabel in archs.items():
                sub = {k: pick(p, cats, conds) for k, p in runs.items() if k[0] == arch and k[1] == cond}
                sub_e = {k: pick(p, cats, conds) for k, p in early.items() if k[0] == arch and k[1] == cond}
                t = X.across_seeds(sub, arch, cond, cats, early=sub_e, from_step=0)
                if not t.empty:
                    rows.append(tag(t, figure="s1_item_group_by_condition_trajectories", architecture=alabel, condition=clabel,
                                    arm="none", percent=0, item_group=glabel))
            t = pooled(runs, early, cond, cats, conds)
            if not t.empty:
                rows.append(tag(t, figure="s1_item_group_by_condition_trajectories", architecture=POOLED_LABEL, condition=clabel,
                                arm="none", percent=0, item_group=glabel))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(a.out / "s1_item_group_by_condition_trajectories.csv", index=False)
    print("s1_item_group_by_condition_trajectories.csv", len(out))

    # pooled rows for the two existing Study 1 files
    for fname, conds in (("s1_baseline_trajectories.csv", ["baseline"]), ("s1_condition_trajectories.csv", list(X.CONDS))):
        f = a.out / fname
        if not f.exists():
            continue
        old = pd.read_csv(f)
        old = old[old.architecture != POOLED_LABEL]
        add = []
        for cond in conds:
            for pool, groups in X.POOLS.items():
                t = pooled(runs, early, cond, groups, None)
                if not t.empty:
                    add.append(tag(t, figure=fname[:-4], architecture=POOLED_LABEL, condition=X.CONDS[cond], arm="none",
                                   percent=0, item_group=pool))
        pd.concat([old] + add, ignore_index=True).to_csv(f, index=False)
        print(fname, "pooled rows added:", sum(len(x) for x in add))
    (a.out / "README_item_groups.txt").write_text(f"""s1_item_group_by_condition_trajectories.csv (deck_exports_item_groups.py)

Study 1, matched evaluation. Preference for the overt form by item group, training condition and architecture
at every checkpoint. Same conventions as the other s1_* files: full-length seeds at the winning setting, the
early-only seeds added up to 256M tokens, untrained model at tokens_seen = 0, enrichment from the fixed items,
mean and 95% t-interval across seeds, n_runs and interval_kind per row.

item_group: the eight groups, plus
  control: subject control / control: object control          24 pairs each
  conjunction: same subject / conjunction: topic shift        24 pairs each
    same subject: the second clause's subject is the first clause's subject; English allows leaving it out.
    topic shift:  the second clause's subject is a different referent; leaving it out changes the meaning.

architecture = "{POOLED_LABEL}": all seeds of the three sizes as one sample (mean and 95% t-interval across
seeds; up to 18 full-length + 18 early-only). Rows exist at the token counts that the full-length runs of all
three sizes share (within 1%); each seed contributes the checkpoint it has within 1% of that count. GPT-2 large
has no full-length checkpoint between 65.5M and 131M or between 131M and 262M tokens, so the pooled rows
skip the dense checkpoints small and medium have there. checkpoint_step is -1 on pooled rows (steps differ by
size; use tokens_seen). The same pooled rows are appended to s1_baseline_trajectories.csv and
s1_condition_trajectories.csv.

Left out: {sorted(skip) if skip else "nothing"}.
""")
    g = out[(out.architecture == POOLED_LABEL) & (out.item_group == "conjunction")]
    print(g.groupby("condition").agg(checkpoints=("tokens_seen", "nunique"), n_min=("n_runs", "min"), n_max=("n_runs", "max")).to_string())


if __name__ == "__main__":
    main()
