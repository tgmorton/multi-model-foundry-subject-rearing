#!/usr/bin/env python3
"""Draw one eval set per recoverability decile from the scored test-set candidates.

An item enters decile d when the rater places its pronoun there with the item as its whole
input (score_items.py) and the same pronoun in its full corpus surroundings
(score_corpus_instances.py) is within `--corpus-slack` deciles of d. Cells are phenomenon x
person; every decile gets the same number of items per cell.

Within a cell the deciles are matched on source (conversation / subtitles / written) and on
whether the pronoun starts its corpus line: the composition is the one closest to the pooled
composition that every decile can supply, so no decile set differs from another in those
respects unless a cell runs short. A cell that cannot be filled from the matched strata is
topped up from whatever the decile has, and what is still missing is reported as the
shortfall to cover with synthetic variants.

Outputs in --out-dir:
  availability.csv     usable items per cell and decile (and the matched capacity)
  sets.parquet         the drawn items with all labels and both scores
  shortfall.csv        cells x deciles that could not be filled
  stimuli/<category>.csv  the eval runner's schema (overt and null rows), plus decile columns

    python analysis/recoverability/eval_stimuli/sample_sets.py --dir <test_10M dir> --out-dir <dir>/sets/robbi_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENRE_GROUP = {"childes": "conversation", "bnc_spoken": "conversation", "switchboard": "conversation",
               "open_subtitles": "subtitles", "gutenberg": "written", "simple_wiki": "written"}
# cell name -> (category in the eval schema, filter on the candidate table, items per decile)
CELLS = {
    "subj_1sg": ("subject_drop", lambda d: (d.ptype == "matrix") & (d.person == "1sg"), 24),
    "subj_1pl": ("subject_drop", lambda d: (d.ptype == "matrix") & (d.person == "1pl"), 24),
    "subj_2": ("subject_drop", lambda d: (d.ptype == "matrix") & (d.person == "2"), 24),
    "subj_3sg": ("subject_drop", lambda d: (d.ptype == "matrix") & (d.person == "3sg"), 24),
    "subj_3pl": ("subject_drop", lambda d: (d.ptype == "matrix") & (d.person == "3pl"), 24),
    "emb_that": ("embedded_drop", lambda d: (d.ptype == "embedded") & (d.kind == "that"), 24),
    "emb_bare": ("embedded_drop", lambda d: (d.ptype == "embedded") & (d.kind == "bare"), 24),
    "conj": ("conjunction", lambda d: (d.ptype == "conjunction") & d.kind.isin(["and", "but"]), 48),
    "adv": ("adverbial", lambda d: d.ptype == "adverbial", 48),
    "expl_it": ("expletive", lambda d: d.ptype == "expletive_it", 24),
}


def matched_quota(avail: pd.DataFrame, n: int) -> pd.Series:
    """Items per stratum that every decile can supply, as close to the pooled mix as possible.

    avail: decile x stratum counts. Returns stratum -> count, summing to min(n, capacity).
    """
    cap = avail.min(axis=0).astype(int)
    pooled = avail.sum(axis=0) / max(avail.values.sum(), 1)
    q = pd.Series(0, index=avail.columns)
    left = min(n, int(cap.sum()))
    while left > 0:
        room = cap - q
        open_ = room[room > 0].index
        # give the next item to the stratum furthest below its pooled share
        deficit = (pooled[open_] * n - q[open_])
        s = deficit.idxmax()
        q[s] += 1
        left -= 1
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True, help="directory with candidates and both score tables")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", default="robbi")
    ap.add_argument("--corpus-slack", type=int, default=1)
    ap.add_argument("--max-item-tokens", type=int, default=110, help="rater tokens of context + target")
    ap.add_argument("--target-words", type=int, nargs=2, default=[4, 20])
    ap.add_argument("--min-gap-lines", type=int, default=6, help="two drawn items are never this close in a file")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every cell size")
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    c = pd.read_parquet(a.dir / "items_candidates.parquet")
    corp = pd.read_parquet(a.dir / f"{a.label}__all.parquet")[["stem", "line_idx", "char_start", "surp", "decile"]]
    corp = corp.rename(columns={"surp": "surp_corpus", "decile": "decile_corpus"})
    item = pd.concat([pd.read_parquet(p) for p in sorted(a.dir.glob(f"items_{a.label}__*.parquet"))])
    item = item.rename(columns={"surp": "surp_item", "decile": "decile_item", "n_tokens": "rater_tokens"})
    d = c.merge(corp, on=["stem", "line_idx", "char_start"], how="left").merge(
        item[["cand_id", "surp_item", "decile_item", "rater_tokens", "top1"]], on="cand_id", how="left")
    d["genre_group"] = d.stem.map(GENRE_GROUP)
    d["stratum"] = d.genre_group + np.where(d.line_initial, "|line-initial", "|line-internal")
    n0 = len(d)
    d = d[(d.decile_item >= 0) & d.decile_corpus.notna()]
    agree = pd.crosstab(d.decile_corpus.astype(int), d.decile_item.astype(int))
    agree.to_csv(a.out_dir / "corpus_vs_item_decile.csv")
    usable = d[(d.rater_tokens <= a.max_item_tokens) & d.target_words.between(*a.target_words)
               & ((d.decile_corpus - d.decile_item).abs() <= a.corpus_slack)].copy()
    usable["decile"] = usable.decile_item.astype(int)
    usable["exact"] = usable.decile_corpus == usable.decile_item
    print(f"candidates {n0:,}; scored {len(d):,}; exact decile agreement {np.mean(d.decile_corpus == d.decile_item):.3f}; "
          f"within {a.corpus_slack}: {np.mean((d.decile_corpus - d.decile_item).abs() <= a.corpus_slack):.3f}; usable {len(usable):,}")

    rng = np.random.default_rng(a.seed)
    usable["rand"] = rng.random(len(usable))
    drawn, avail_rows, short_rows = [], [], []
    taken: dict = {}                                   # stem -> sorted list of taken line indices

    def free(stem, line):
        return all(abs(line - t) >= a.min_gap_lines for t in taken.get(stem, ()))

    for cell, (category, filt, size) in CELLS.items():
        n = int(round(size * a.scale))
        u = usable[filt(usable)]
        avail = u.groupby(["decile", "stratum"]).size().unstack(fill_value=0).reindex(range(10), fill_value=0)
        quota = matched_quota(avail, n)
        for dec in range(10):
            avail_rows.append({"cell": cell, "category": category, "decile": dec, "usable": int(avail.loc[dec].sum()),
                               "matched_capacity": int(quota.sum()), "wanted": n})
            pool = u[u.decile == dec]
            med = u.target_words.median()
            pool = pool.assign(len_dist=(pool.target_words - med).abs()).sort_values(
                ["exact", "len_dist", "rand"], ascending=[False, True, True])
            got = []
            for stratum, k in quota.items():           # matched part
                for r in pool[pool.stratum == stratum].itertuples():
                    if k == 0:
                        break
                    if free(r.stem, r.line_idx):
                        got.append((r.cand_id, True))
                        taken.setdefault(r.stem, []).append(r.line_idx)
                        k -= 1
            have = {g[0] for g in got}
            for r in pool.itertuples():                # top-up outside the matched composition
                if len(got) >= n:
                    break
                if r.cand_id not in have and free(r.stem, r.line_idx):
                    got.append((r.cand_id, False))
                    taken.setdefault(r.stem, []).append(r.line_idx)
            if len(got) < n:
                short_rows.append({"cell": cell, "category": category, "decile": dec, "wanted": n, "found": len(got),
                                   "missing": n - len(got)})
            sel = u.set_index("cand_id").loc[[g[0] for g in got]].reset_index()
            sel["cell"], sel["category"], sel["matched"] = cell, category, [g[1] for g in got]
            drawn.append(sel)
    sets = pd.concat(drawn, ignore_index=True)
    sets.insert(0, "item_id", np.arange(1, len(sets) + 1))
    sets.drop(columns=["rand", "norm_target"]).to_parquet(a.out_dir / "sets.parquet")
    pd.DataFrame(avail_rows).to_csv(a.out_dir / "availability.csv", index=False)
    pd.DataFrame(short_rows, columns=["cell", "category", "decile", "wanted", "found", "missing"]).to_csv(
        a.out_dir / "shortfall.csv", index=False)

    stim = a.out_dir / "stimuli"
    stim.mkdir(exist_ok=True)
    for category, g in sets.groupby("category"):
        rows = []
        for r in g.itertuples():
            common = {"item_id": r.item_id, "category": category, "condition": f"{r.cell}_d{r.decile}",
                      "context": r.context, "language": "en", "names": "", "generator": f"test_10M:{r.stem}",
                      "cell": r.cell, "decile": r.decile, "cand_id": r.cand_id}
            rows.append({**common, "pronoun_status": 1, "target": r.target_overt, "hotspot_token": r.hotspot_token,
                         "hotspot_position": r.hotspot_overt})
            rows.append({**common, "pronoun_status": 0, "target": r.target_null, "hotspot_token": r.hotspot_token,
                         "hotspot_position": r.hotspot_null})
        cols = ["item_id", "category", "condition", "pronoun_status", "context", "target", "hotspot_token",
                "hotspot_position", "language", "names", "generator", "cell", "decile", "cand_id"]
        pd.DataFrame(rows)[cols].to_csv(stim / f"{category}.csv", index=False)
    meta = {"label": a.label, "n_items": len(sets), "corpus_slack": a.corpus_slack, "seed": a.seed,
            "cells": {k: int(round(v[2] * a.scale)) for k, v in CELLS.items()},
            "exact_share": float(sets.exact.mean()), "matched_share": float(sets.matched.mean())}
    (a.out_dir / "sets.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    print(pd.DataFrame(avail_rows).pivot(index="cell", columns="decile", values="usable").to_string())
    print("drawn per cell x decile")
    print(sets.groupby(["cell", "decile"]).size().unstack(fill_value=0).to_string())
    if short_rows:
        print("SHORTFALL"); print(pd.DataFrame(short_rows).to_string(index=False))


if __name__ == "__main__":
    main()
