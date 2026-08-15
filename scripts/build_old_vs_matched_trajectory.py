#!/usr/bin/env python3
"""Build cell-level old-versus-matched trajectory data for R plotting."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


PAIR_KEYS = ["cell_id", "checkpoint_step", "category", "condition", "item_id"]


def posthoc_slor(per_token_path: Path, unigram_path: Path) -> pd.DataFrame:
    """Recover pair-level SLOR from persisted token scores without a rescore."""
    from evaluation.unigram import UnigramTable

    unigram = UnigramTable.load(unigram_path)
    frame = pd.read_parquet(
        per_token_path,
        columns=PAIR_KEYS + [
            "pronoun_status", "per_token_ids", "per_token_log_prob"
        ],
    )
    log_probs = unigram.log_probs

    def score(row: pd.Series) -> float:
        token_ids = np.asarray(row.per_token_ids, dtype=np.int64)
        if token_ids.size == 0:
            return np.nan
        model_sum = float(np.asarray(row.per_token_log_prob, dtype=float).sum())
        unigram_sum = float(log_probs[token_ids].sum())
        return (model_sum - unigram_sum) / token_ids.size

    frame["slor"] = frame.apply(score, axis=1)
    pivoted = frame.pivot(
        index=PAIR_KEYS, columns="pronoun_status", values="slor"
    )
    if 0 not in pivoted or 1 not in pivoted:
        raise RuntimeError(f"incomplete pronoun pairs in {per_token_path}")
    output = pivoted.reset_index()
    output["slor_diff_overt_minus_null"] = output[1] - output[0]
    return output[PAIR_KEYS + ["slor_diff_overt_minus_null"]]


def parse_run_id(run_id: str) -> tuple[str, str, int, int]:
    match = re.fullmatch(r"(.+)-en-(.+)-h(\d+)-s(\d+)", run_id)
    if match is None:
        raise ValueError(f"cannot parse run ID: {run_id}")
    return match.group(1), match.group(2), int(match.group(3)), int(match.group(4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--hp-rank", action="append", type=int, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--unigram-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected_hp = set(args.hp_rank)
    candidates: dict[str, list[tuple[str, Path, Path, Path, int, int]]] = {}
    for matched_path in (args.matched_root / "pairs").glob(
        f"cell_id={args.architecture}-en-*.parquet"
    ):
        run_id = matched_path.stem.removeprefix("cell_id=")
        architecture, intervention, hp_rank, seed = parse_run_id(run_id)
        if architecture != args.architecture or hp_rank not in selected_hp:
            continue
        old_path = args.old_root / "pairs" / matched_path.name
        checkpoint_path = args.matched_root / "checkpoints" / matched_path.name
        if old_path.is_file() and checkpoint_path.is_file():
            candidates.setdefault(intervention, []).append(
                (
                    run_id,
                    old_path,
                    matched_path,
                    checkpoint_path,
                    hp_rank,
                    seed,
                )
            )

    complete_conditions = {}
    for intervention, cells in candidates.items():
        counts = pd.Series([cell[4] for cell in cells]).value_counts().to_dict()
        seeds_by_hp = {
            hp: {cell[5] for cell in cells if cell[4] == hp} for hp in selected_hp
        }
        if (
            set(counts) == selected_hp
            and len(set(counts.values())) == 1
            and len({tuple(sorted(values)) for values in seeds_by_hp.values()}) == 1
        ):
            complete_conditions[intervention] = sorted(cells)

    if not complete_conditions:
        raise SystemExit("no intervention has a complete selected-HP overlap")

    output_frames = []
    for intervention, cells in sorted(complete_conditions.items()):
        for run_id, old_path, matched_path, checkpoint_path, hp_rank, seed in cells:
            columns = PAIR_KEYS + [
                "prefers_overt_meanlp", "slor_diff_overt_minus_null"
            ]
            old = pd.read_parquet(old_path, columns=columns)
            matched = pd.read_parquet(matched_path, columns=columns)
            if old["slor_diff_overt_minus_null"].isna().any():
                unigram_path = (
                    args.unigram_root
                    / f"en_{intervention}__shared_unigram.pkl"
                )
                if not unigram_path.is_file():
                    raise FileNotFoundError(
                        f"missing unigram table for {intervention}: {unigram_path}"
                    )
                recovered = posthoc_slor(
                    args.old_root / "per_token" / old_path.name,
                    unigram_path,
                )
                old = old.drop(columns="slor_diff_overt_minus_null").merge(
                    recovered, on=PAIR_KEYS, how="left", validate="one_to_one"
                )
            if (
                old["slor_diff_overt_minus_null"].isna().any()
                or matched["slor_diff_overt_minus_null"].isna().any()
            ):
                raise RuntimeError(f"missing SLOR after recovery: {run_id}")
            if old.duplicated(PAIR_KEYS).any() or matched.duplicated(PAIR_KEYS).any():
                raise RuntimeError(f"duplicate pair keys: {run_id}")
            joined = old.merge(
                matched,
                on=PAIR_KEYS,
                how="inner",
                validate="one_to_one",
                suffixes=("_old", "_matched"),
            )
            checkpoints = pd.read_parquet(
                checkpoint_path, columns=["checkpoint_step", "tokens_seen"]
            ).drop_duplicates()
            joined = joined.merge(
                checkpoints,
                on="checkpoint_step",
                how="inner",
                validate="many_to_one",
            )
            for version, preference_column, slor_column in (
                (
                    "Old generic",
                    "prefers_overt_meanlp_old",
                    "slor_diff_overt_minus_null_old",
                ),
                (
                    "Condition-matched",
                    "prefers_overt_meanlp_matched",
                    "slor_diff_overt_minus_null_matched",
                ),
            ):
                summary = (
                    joined.groupby(
                        ["checkpoint_step", "tokens_seen", "category"],
                        as_index=False,
                    )[[preference_column, slor_column]]
                    .mean()
                    .rename(
                        columns={
                            preference_column: "preference",
                            slor_column: "slor_diff",
                        }
                    )
                )
                summary.insert(0, "cell_id", run_id)
                summary.insert(1, "architecture", args.architecture)
                summary.insert(2, "intervention", intervention)
                summary.insert(3, "hp_rank", hp_rank)
                summary.insert(4, "seed", seed)
                summary.insert(5, "eval_version", version)
                output_frames.append(summary)

    output = pd.concat(output_frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        f"wrote {args.output}: rows={len(output)} "
        f"conditions={sorted(complete_conditions)} "
        f"cells={output['cell_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
