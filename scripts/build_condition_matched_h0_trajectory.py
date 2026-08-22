#!/usr/bin/env python3
"""Build per-cell trajectories from condition-matched pair evaluations.

The output has one row per cell, category, and tokens-seen coordinate.  Item
preferences are averaged within the cell first so downstream summaries give
each seed equal weight.  Checkpoints sharing a tokens-seen coordinate are
collapsed within the cell before any across-seed aggregation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CELL_RE = re.compile(
    r"^cell_id=(?P<arch>.+)-en-(?P<condition>.+)-h(?P<hp>\d+)-s(?P<seed>\d+)\.parquet$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--architecture", default="gpt2_small")
    parser.add_argument(
        "--hp-rank", type=int, action="append",
        help="HP rank to include; repeat for several. Omit to include all ranks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pair_root = args.results_root / "pairs"
    checkpoint_root = args.results_root / "checkpoints"
    rows: list[pd.DataFrame] = []

    for pair_path in sorted(pair_root.glob(f"cell_id={args.architecture}-en-*.parquet")):
        match = CELL_RE.match(pair_path.name)
        if not match:
            continue
        hp_rank = int(match.group("hp"))
        if args.hp_rank is not None and hp_rank not in args.hp_rank:
            continue
        checkpoint_path = checkpoint_root / pair_path.name
        if not checkpoint_path.exists():
            raise SystemExit(f"missing checkpoint sidecar: {checkpoint_path}")

        pairs = pd.read_parquet(
            pair_path,
            columns=[
                "cell_id", "architecture", "intervention", "checkpoint_step",
                "category", "prefers_overt_meanlp",
            ],
        ).dropna(subset=["prefers_overt_meanlp"])
        checkpoints = pd.read_parquet(
            checkpoint_path,
            columns=[
                "checkpoint_step", "tokens_seen", "epoch", "hp_rank", "seed",
                "benchmark", "scoring_version", "stimuli_manifest_sha256",
            ],
        )
        if checkpoints.empty or pairs.empty:
            raise SystemExit(f"empty matched result: {pair_path.name}")
        if set(checkpoints["hp_rank"].astype(int)) != {hp_rank}:
            raise SystemExit(f"unexpected HP rank in {checkpoint_path.name}")

        merged = pairs.merge(
            checkpoints[
                ["checkpoint_step", "tokens_seen", "epoch", "hp_rank", "seed", "benchmark",
                 "scoring_version", "stimuli_manifest_sha256"]
            ].drop_duplicates("checkpoint_step"),
            on="checkpoint_step",
            how="inner",
            validate="many_to_one",
        )
        cell = (
            merged.groupby(
                [
                    "cell_id", "architecture", "intervention", "hp_rank", "seed", "category",
                    "tokens_seen", "benchmark", "scoring_version",
                    "stimuli_manifest_sha256",
                ],
                as_index=False,
            )["prefers_overt_meanlp"]
            .mean()
            .rename(columns={"prefers_overt_meanlp": "preference"})
        )
        rows.append(cell)

    if not rows:
        raise SystemExit("no matching cells found")
    output = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    cells = output[["cell_id", "intervention", "hp_rank", "seed"]].drop_duplicates()
    counts = cells.groupby(["intervention", "hp_rank"])["cell_id"].nunique().sort_index()
    print(f"wrote {len(output):,} cell-category-token rows to {args.output}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
