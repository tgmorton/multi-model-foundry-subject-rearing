"""Add the 'robbi' label (bidirectional roberta-large ±250, listener
construal) to the frozen selection-v5 family — 2026-09-19.

The population and the shared rand_decile assignment are taken VERBATIM
from the existing v5 tables (label 'gpt2m' as the carrier), so the rand
and all100 corpora already composed/training serve this arm unchanged;
only 45 info cells are new.

Instances in the v5 population that the roberta pass failed to cover
(aligner edge cases) are ranked LEAST recoverable (decile 9 — removed
only at k=100, where everything is removed anyway); the count is
recorded in the manifest.

Saturation note (recorded, known): at ±250 roberta is saturated
(~53% of instances < 0.1 nats), so ordering within the most-recoverable
deciles is measurement noise; the arm is interpreted at decile
granularity as the listener-construal contrast.
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
LABEL = "robbi"
SOURCE = "roberta_bi"


def load_scores(root: Path, corpus: str, genres) -> pd.DataFrame:
    frames = []
    for g in genres:
        p = (root / corpus / f"external_{SOURCE}" / "instances"
             / f"{g}.parquet")
        d = pq.read_table(p, columns=["line_idx", "token_i",
                                      f"{SOURCE}__logprob_sum"]).to_pandas()
        d["surp"] = -d[f"{SOURCE}__logprob_sum"]
        d["genre"] = g
        frames.append(d[["genre", "line_idx", "token_i", "surp"]])
    return pd.concat(frames, ignore_index=True)


def load_family_tables(sel_root: Path, corpus: str, genres) -> pd.DataFrame:
    frames = []
    for g in genres:
        d = pq.read_table(sel_root / "gpt2m" / corpus / f"{g}.parquet"
                          ).to_pandas()
        d["genre"] = g
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/recoverability"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/recoverability/analysis"))
    ap.add_argument("--genres", nargs="+", default=GENRES)
    args = ap.parse_args()
    sel_root = args.out / "selection_v5"

    fam = json.loads((sel_root / "V5_FAMILY.json").read_text())
    assert fam["selection_version"] == 5

    counts = {}
    thresholds = None
    for corpus in ("train_90M", "pull_10M"):
        pop = load_family_tables(sel_root, corpus, args.genres)[
            ["genre", "line_idx", "token_i", "form", "rand_decile"]]
        sc = load_scores(args.root, corpus, args.genres)
        df = pop.merge(sc, on=["genre", "line_idx", "token_i"], how="left")
        missing = int(df.surp.isna().sum())
        df["rank_value"] = df.surp.fillna(np.inf)
        if corpus == "train_90M":
            df["info_decile"] = pd.qcut(
                df.rank_value.rank(method="first"), 10,
                labels=False).astype(np.int8)
            thresholds = [float(df[df.info_decile <= d].rank_value.max())
                          for d in range(9)]
        else:
            df["info_decile"] = np.searchsorted(
                np.array(thresholds), df.rank_value.values,
                side="left").astype(np.int8)
        counts[corpus] = {"n": int(len(df)), "missing_to_decile9": missing}
        cdir = sel_root / LABEL / corpus
        cdir.mkdir(parents=True, exist_ok=True)
        for g, gdf in df.groupby("genre"):
            pq.write_table(pa.Table.from_pandas(
                gdf[["line_idx", "token_i", "form", "rank_value",
                     "info_decile", "rand_decile"]].replace(
                     {np.inf: np.finfo("f8").max}),
                preserve_index=False), cdir / f"{g}.parquet")
        print(f"{corpus}: n={len(df):,} missing={missing:,}")

    man = {
        "selection_version": 5,
        "ranking_label": LABEL,
        "ranking": f"surp__{SOURCE} (bidirectional roberta-large ±250; "
                   "listener construal; decile 0 = most recoverable)",
        "ranking_inputs": [SOURCE],
        "population_n": counts["train_90M"]["n"],
        "population_note": "frozen v5 population + shared rand verbatim; "
                           "roberta-uncovered instances ranked decile 9",
        "missing_coverage": counts,
        "saturation_note": "~53% of instances < 0.1 nats at ±250 — "
                           "within-decile order among saturated instances "
                           "is noise; interpret at decile granularity",
        "cumulative_semantics": "condition K%% removes instances with "
                                "decile < K/10 (either arm)",
        "decile_upper_thresholds": thresholds,
        "random_seed": fam["rand_seed_train"],
        "genres": sorted(args.genres),
        "pool": {"corpus": "pull_10M", **counts["pull_10M"],
                 "info": "train-derived absolute thresholds",
                 "rand_seed": fam["rand_seed_pool"]},
    }
    with open(sel_root / LABEL / "SELECTION_MANIFEST.json", "w") as f:
        json.dump(man, f, indent=2)
    fam.setdefault("labels", {})[LABEL] = {
        "ranking_inputs": [SOURCE], "added": "2026-09-19",
        "decile_upper_thresholds": thresholds,
        "missing_coverage": counts}
    with open(sel_root / "V5_FAMILY.json", "w") as f:
        json.dump(fam, f, indent=2)
    print(f"label '{LABEL}' emitted into the v5 family")


if __name__ == "__main__":
    main()
