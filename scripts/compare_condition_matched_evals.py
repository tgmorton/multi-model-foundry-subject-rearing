#!/usr/bin/env python3
"""Compare legacy generic-stimulus and condition-matched eval results.

Only rows with the same ``(cell, checkpoint, category, condition, item)`` key
are compared.  The report keeps H0 separate because those checkpoints were
replaced during the corrected early-run wave while legacy eval files remained.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["cell_id", "checkpoint_step", "category", "condition", "item_id"]
VALUE_COLUMNS = [
    "architecture",
    "intervention",
    "log_prob_diff_overt_minus_null",
    "prefers_overt_meanlp",
]


def fresh_stats() -> dict[str, Any]:
    return {
        "files": set(),
        "n": 0,
        "old_rows": 0,
        "new_rows": 0,
        "mismatch_files": 0,
        "sum_delta": 0.0,
        "sum_abs_delta": 0.0,
        "sum_squared_delta": 0.0,
        "old_preferred": 0,
        "new_preferred": 0,
        "flips": 0,
        "sum_old": 0.0,
        "sum_new": 0.0,
        "sum_old_squared": 0.0,
        "sum_new_squared": 0.0,
        "sum_cross": 0.0,
        "absolute_deltas": [],
        "category_preference_deltas": [],
        "overall_preference_deltas": [],
    }


def add_rows(
    stats: dict[str, Any],
    run_id: str,
    old_rows: int,
    new_rows: int,
    coverage_mismatch: bool,
    joined: pd.DataFrame,
) -> None:
    old_margin = joined["old_margin"].to_numpy(float)
    new_margin = joined["new_margin"].to_numpy(float)
    delta = new_margin - old_margin
    old_pref = joined["old_pref"].fillna(False).to_numpy(bool)
    new_pref = joined["new_pref"].fillna(False).to_numpy(bool)

    stats["files"].add(run_id)
    stats["n"] += len(joined)
    stats["old_rows"] += old_rows
    stats["new_rows"] += new_rows
    stats["mismatch_files"] += int(coverage_mismatch)
    stats["sum_delta"] += float(delta.sum())
    stats["sum_abs_delta"] += float(np.abs(delta).sum())
    stats["sum_squared_delta"] += float(np.square(delta).sum())
    stats["old_preferred"] += int(old_pref.sum())
    stats["new_preferred"] += int(new_pref.sum())
    stats["flips"] += int((old_pref != new_pref).sum())
    stats["sum_old"] += float(old_margin.sum())
    stats["sum_new"] += float(new_margin.sum())
    stats["sum_old_squared"] += float(np.square(old_margin).sum())
    stats["sum_new_squared"] += float(np.square(new_margin).sum())
    stats["sum_cross"] += float((old_margin * new_margin).sum())
    stats["absolute_deltas"].append(np.abs(delta).astype("float32"))

    aggregate = pd.DataFrame(
        {
            "checkpoint_step": joined["checkpoint_step"].to_numpy(),
            "category": joined["category"].astype(str).to_numpy(),
            "old": old_pref.astype(float),
            "new": new_pref.astype(float),
        }
    )
    by_category = aggregate.groupby(
        ["checkpoint_step", "category"], sort=False
    )[["old", "new"]].mean()
    by_checkpoint = aggregate.groupby("checkpoint_step", sort=False)[
        ["old", "new"]
    ].mean()
    stats["category_preference_deltas"].append(
        np.abs(by_category["new"] - by_category["old"])
        .to_numpy(float)
        .astype("float32")
    )
    stats["overall_preference_deltas"].append(
        np.abs(by_checkpoint["new"] - by_checkpoint["old"])
        .to_numpy(float)
        .astype("float32")
    )


def summarize(stats: dict[str, Any]) -> dict[str, Any]:
    n = int(stats["n"])
    absolute = np.concatenate(stats["absolute_deltas"])
    category = np.concatenate(stats["category_preference_deltas"])
    overall = np.concatenate(stats["overall_preference_deltas"])
    correlation_denominator = math.sqrt(
        max(
            0.0,
            (n * stats["sum_old_squared"] - stats["sum_old"] ** 2)
            * (n * stats["sum_new_squared"] - stats["sum_new"] ** 2),
        )
    )
    return {
        "run_files": len(stats["files"]),
        "pair_checkpoint_rows": n,
        "old_rows": int(stats["old_rows"]),
        "new_rows": int(stats["new_rows"]),
        "coverage_mismatch_files": int(stats["mismatch_files"]),
        "mean_margin_delta": stats["sum_delta"] / n,
        "mean_absolute_margin_delta": stats["sum_abs_delta"] / n,
        "median_absolute_margin_delta": float(np.quantile(absolute, 0.5)),
        "p95_absolute_margin_delta": float(np.quantile(absolute, 0.95)),
        "rmse_margin_delta": math.sqrt(stats["sum_squared_delta"] / n),
        "preference_flip_rate": stats["flips"] / n,
        "old_preference_rate": stats["old_preferred"] / n,
        "new_preference_rate": stats["new_preferred"] / n,
        "preference_rate_delta": (
            stats["new_preferred"] - stats["old_preferred"]
        ) / n,
        "margin_correlation": (
            (n * stats["sum_cross"] - stats["sum_old"] * stats["sum_new"])
            / correlation_denominator
            if correlation_denominator
            else None
        ),
        "mean_absolute_category_checkpoint_preference_delta": float(
            category.mean()
        ),
        "p95_absolute_category_checkpoint_preference_delta": float(
            np.quantile(category, 0.95)
        ),
        "mean_absolute_overall_checkpoint_preference_delta": float(
            overall.mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--audited-run-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audited = set(json.loads(args.audited_run_ids.read_text()))
    old_pairs = args.old_root / "pairs"
    matched_pairs = args.matched_root / "pairs"
    candidates = []
    for matched_path in matched_pairs.glob("cell_id=*.parquet"):
        run_id = matched_path.stem.removeprefix("cell_id=")
        old_path = old_pairs / matched_path.name
        if run_id in audited and old_path.is_file():
            candidates.append((run_id, old_path, matched_path))

    groups: dict[str, dict[str, Any]] = defaultdict(fresh_stats)
    category_groups: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0, 0])
    )
    for run_id, old_path, matched_path in sorted(candidates):
        hp_match = re.search(r"-h(\d+)-s", run_id)
        if hp_match is None:
            raise RuntimeError(f"cannot parse HP rank from {run_id}")
        hp_rank = int(hp_match.group(1))
        old = pd.read_parquet(old_path, columns=KEYS + VALUE_COLUMNS)
        new = pd.read_parquet(matched_path, columns=KEYS + VALUE_COLUMNS)
        if old.duplicated(KEYS).any() or new.duplicated(KEYS).any():
            raise RuntimeError(f"duplicate comparison keys in {run_id}")
        intervention = str(new["intervention"].iloc[0])
        architecture = str(new["architecture"].iloc[0])
        same_coverage = len(old) == len(new) and old[KEYS].equals(new[KEYS])
        joined = old[
            KEYS
            + ["log_prob_diff_overt_minus_null", "prefers_overt_meanlp"]
        ].merge(
            new[
                KEYS
                + ["log_prob_diff_overt_minus_null", "prefers_overt_meanlp"]
            ],
            on=KEYS,
            how="inner",
            validate="one_to_one",
            suffixes=("_old", "_new"),
        ).rename(
            columns={
                "log_prob_diff_overt_minus_null_old": "old_margin",
                "log_prob_diff_overt_minus_null_new": "new_margin",
                "prefers_overt_meanlp_old": "old_pref",
                "prefers_overt_meanlp_new": "new_pref",
            }
        )
        if joined.empty:
            continue
        cohort = "h1_h4" if hp_rank > 0 else "h0"
        names = [
            "all",
            f"cohort:{cohort}",
            f"intervention:{intervention}",
            f"cohort:{cohort}|intervention:{intervention}",
            f"cohort:{cohort}|intervention:{intervention}|arch:{architecture}",
        ]
        for name in names:
            add_rows(
                groups[name], run_id, len(old), len(new), not same_coverage, joined
            )

        old_pref = joined["old_pref"].fillna(False).to_numpy(bool)
        new_pref = joined["new_pref"].fillna(False).to_numpy(bool)
        for category, indices in joined.groupby("category").groups.items():
            selection = np.asarray(list(indices), dtype=int)
            before = old_pref[selection]
            after = new_pref[selection]
            values = category_groups[f"cohort:{cohort}|intervention:{intervention}"][
                str(category)
            ]
            values[0] += len(selection)
            values[1] += int((before != after).sum())
            values[2] += int(before.sum())
            values[3] += int(after.sum())

    report = {
        "format_version": "condition-matched-divergence.v1",
        "old_root": str(args.old_root),
        "matched_root": str(args.matched_root),
        "audited_run_ids_path": str(args.audited_run_ids),
        "audited_old_new_overlap_files": len(candidates),
        "groups": {name: summarize(value) for name, value in sorted(groups.items())},
        "category_groups": {},
    }
    for name, categories in sorted(category_groups.items()):
        report["category_groups"][name] = {}
        for category, (n, flips, old_preferred, new_preferred) in sorted(
            categories.items()
        ):
            report["category_groups"][name][category] = {
                "pair_checkpoint_rows": n,
                "preference_flip_rate": flips / n,
                "preference_rate_delta": (new_preferred - old_preferred) / n,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"compared {len(candidates)} audited run files; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
