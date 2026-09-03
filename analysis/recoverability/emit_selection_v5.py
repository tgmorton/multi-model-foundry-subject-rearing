"""
Harmonized selection v5 — three rater arms, one population (2026-08-31).

Emits THREE frozen selections from a single harmonized population so the
gpt2_small rater-comparison cohorts share their random arms exactly:

  gpt2m  rank = -logprob_sum from external_gpt2_medium  (full-stream causal)
  bert   rank = -logprob_sum from external_bert_wwm_r1  (masked 250:1)
  comp   rank = z(gpt2m) + z(bert_fwd)                  (speaker + listener;
         bert_fwd = masked forward-only 0:250; z-constants frozen from the
         train population and reused verbatim for the pool)

Population: in_lexicon & aligned & ALL THREE ranking inputs present, on
both corpora. One rand permutation (seed 42 train / 43 pool) is shared by
every arm, so rand cells and the rater-independent k=100 cells are the
same corpora across raters — 50 of the 185 matrix cells are shared.

Layout (per label, compose-compatible with the v4 consumer):
  <out>/selection_v5/<label>/SELECTION_MANIFEST.json
  <out>/selection_v5/<label>/train_90M/<genre>.parquet
  <out>/selection_v5/<label>/pull_10M/<genre>.parquet
plus <out>/selection_v5/V5_FAMILY.json describing the family.

Tables carry (line_idx, token_i, form, rank_value, info_decile,
rand_decile); the graded ablation consumes only ids + deciles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

GENRES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles",
          "simple_wiki", "switchboard"]
EXTERNALS = ["gpt2_medium", "bert_wwm_r1", "bert_fwd"]
RANKINGS = {  # label -> inputs
    "gpt2m": ("gpt2_medium",),
    "bert": ("bert_wwm_r1",),
    "comp": ("gpt2_medium", "bert_fwd"),
}
SELECTION_VERSION = 5


def load_corpus(root: Path, corpus: str, genres) -> pd.DataFrame:
    frames = []
    for g in genres:
        base = None
        for name in EXTERNALS:
            p = root / corpus / f"external_{name}" / "instances" / f"{g}.parquet"
            cols = ["line_idx", "token_i", f"{name}__logprob_sum"]
            if base is None:
                cols += ["form", "in_lexicon"]
            d = pq.read_table(p, columns=cols).to_pandas()
            d[f"surp__{name}"] = -d[f"{name}__logprob_sum"]
            d = d.drop(columns=[f"{name}__logprob_sum"])
            base = d if base is None else base.merge(
                d, on=["line_idx", "token_i"], how="inner",
                validate="one_to_one")
        base["genre"] = g
        frames.append(base)
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df[df.in_lexicon].copy()
    for name in EXTERNALS:
        df = df[df[f"surp__{name}"].notna()]
    print(f"{corpus}: {len(df):,} population "
          f"(from {n0:,} joined; in_lexicon + full coverage)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/recoverability"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/recoverability/analysis"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--genres", nargs="+", default=GENRES)
    args = ap.parse_args()

    train = load_corpus(args.root, "train_90M", args.genres)
    pool = load_corpus(args.root, "pull_10M", args.genres)

    # frozen z-constants from the TRAIN population; reused for the pool
    zc = {name: (float(train[f"surp__{name}"].mean()),
                 float(train[f"surp__{name}"].std()))
          for name in EXTERNALS}

    def rank_value(df: pd.DataFrame, inputs) -> pd.Series:
        if len(inputs) == 1:
            return df[f"surp__{inputs[0]}"]
        return sum((df[f"surp__{n}"] - zc[n][0]) / zc[n][1] for n in inputs)

    # ONE shared rand assignment per corpus (identical across labels)
    rng = np.random.RandomState(args.seed)
    train = train.reset_index(drop=True)
    train["rand_decile"] = (rng.permutation(len(train)) * 10
                            // len(train)).astype(np.int8)
    pool_rng = np.random.RandomState(args.seed + 1)
    pool = pool.reset_index(drop=True)
    pool["rand_decile"] = (pool_rng.permutation(len(pool)) * 10
                           // max(len(pool), 1)).astype(np.int8)

    out_root = args.out / "selection_v5"
    family = {"selection_version": SELECTION_VERSION,
              "labels": {}, "population_n": int(len(train)),
              "pool_population_n": int(len(pool)),
              "population": "in_lexicon & aligned & all three externals "
                            "present (harmonized intersection)",
              "externals": EXTERNALS,
              "z_constants_train": {k: {"mean": m, "std": s}
                                    for k, (m, s) in zc.items()},
              "rand_seed_train": args.seed, "rand_seed_pool": args.seed + 1,
              "shared_rand_note": "rand_decile is identical in every label's "
                                  "tables (one permutation per corpus), so "
                                  "rand cells and k=100 cells are shared "
                                  "corpora across rater arms",
              "per_genre_n": train.genre.value_counts().to_dict()}

    for label, inputs in RANKINGS.items():
        tr = train.copy()
        tr["rank_value"] = rank_value(tr, inputs)
        tr["info_decile"] = pd.qcut(tr.rank_value, 10, labels=False
                                    ).astype(np.int8)
        thresholds = [float(tr[tr.info_decile <= d].rank_value.max())
                      for d in range(9)]
        po = pool.copy()
        po["rank_value"] = rank_value(po, inputs)
        po["info_decile"] = np.searchsorted(
            np.array(thresholds), po.rank_value.values,
            side="left").astype(np.int8)

        ldir = out_root / label
        for corpus, df in (("train_90M", tr), ("pull_10M", po)):
            cdir = ldir / corpus
            cdir.mkdir(parents=True, exist_ok=True)
            for g, gdf in df.groupby("genre"):
                pq.write_table(pa.Table.from_pandas(
                    gdf[["line_idx", "token_i", "form", "rank_value",
                         "info_decile", "rand_decile"]],
                    preserve_index=False), cdir / f"{g}.parquet")
        man = {
            "selection_version": SELECTION_VERSION,
            "ranking_label": label,
            "ranking": " + ".join(f"z(surp__{n})" for n in inputs)
                       if len(inputs) > 1 else f"surp__{inputs[0]}",
            "ranking_inputs": list(inputs),
            "population_n": int(len(tr)),
            "cumulative_semantics": "condition K%% removes instances with "
                                    "decile < K/10 (either arm)",
            "decile_upper_thresholds": thresholds,
            "z_constants": ({n: {"mean": zc[n][0], "std": zc[n][1]}
                             for n in inputs} if len(inputs) > 1 else None),
            "random_seed": args.seed,
            "genres": sorted(tr.genre.unique().tolist()),
            "pool": {"corpus": "pull_10M", "n": int(len(po)),
                     "info": "train-derived absolute thresholds "
                             "(searchsorted on rank_value)",
                     "rand_seed": args.seed + 1,
                     "info_decile_frac": [
                         float((po.info_decile <= d).mean())
                         for d in range(10)]},
        }
        with open(ldir / "SELECTION_MANIFEST.json", "w") as f:
            json.dump(man, f, indent=2)
        family["labels"][label] = {
            "ranking_inputs": list(inputs),
            "decile_upper_thresholds": thresholds,
            "pool_info_decile_frac": man["pool"]["info_decile_frac"],
        }
        print(f"label {label}: emitted "
              f"({len(tr):,} train, {len(po):,} pool)")

    with open(out_root / "V5_FAMILY.json", "w") as f:
        json.dump(family, f, indent=2)
    print(f"wrote {out_root}/V5_FAMILY.json "
          f"(population {len(train):,}; 3 labels; shared rand)")


if __name__ == "__main__":
    main()
