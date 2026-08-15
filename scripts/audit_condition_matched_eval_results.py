#!/usr/bin/env python3
"""Prove local condition-matched Parquets cover an expected inventory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TABLES = ("items", "pairs", "per_token", "checkpoints")


def parse_arch_hp(values: list[str]) -> set[tuple[str, int]]:
    parsed = set()
    for value in values:
        arch, hp = value.rsplit(":", 1)
        parsed.add((arch, int(hp.removeprefix("h"))))
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--run-ids", type=Path,
                    help="Optional JSON list limiting the audited tranche.")
    ap.add_argument("--exclude-arch-hp", action="append", default=[],
                    metavar="ARCH:HP")
    ap.add_argument("--only-arch-hp", action="append", default=[],
                    metavar="ARCH:HP")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stimuli-manifest", type=Path, required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--scoring-version", required=True)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    inventory = json.loads(args.inventory.read_text())
    inventory_sha = hashlib.sha256(args.inventory.read_bytes()).hexdigest()
    manifest = json.loads(args.stimuli_manifest.read_text())
    manifest_sha = hashlib.sha256(args.stimuli_manifest.read_bytes()).hexdigest()
    wanted = set(json.loads(args.run_ids.read_text())) if args.run_ids else None
    excluded = parse_arch_hp(args.exclude_arch_hp)
    only = parse_arch_hp(args.only_arch_hp)
    runs = [r for r in inventory["runs"]
            if (wanted is None or r["run_id"] in wanted)
            and (r["architecture"], int(r["hp_rank"])) not in excluded
            and (not only or (r["architecture"], int(r["hp_rank"])) in only)]
    if not runs:
        raise SystemExit("selection contains no runs")
    errors = []
    verified = []
    for run in runs:
        rid = run["run_id"]
        expected_steps = {int(c["step"]) for c in run["checkpoints"]}
        tables = {}
        for table in TABLES:
            path = args.results_root / table / f"cell_id={rid}.parquet"
            if not path.is_file():
                errors.append({"run_id": rid, "error": f"missing_{table}"})
                continue
            try:
                tables[table] = pd.read_parquet(path)
            except Exception as exc:
                errors.append({"run_id": rid,
                               "error": f"unreadable_{table}: {exc}"})
        if len(tables) != len(TABLES):
            continue
        step_sets = {table: set(map(int, df["checkpoint_step"].unique()))
                     for table, df in tables.items()}
        for table, steps in step_sets.items():
            if steps != expected_steps:
                errors.append({
                    "run_id": rid, "error": f"step_mismatch_{table}",
                    "missing": sorted(expected_steps - steps),
                    "unexpected": sorted(steps - expected_steps),
                })
        n_checkpoints = len(expected_steps)
        expected_rows = {
            "items": 1152 * n_checkpoints,
            "pairs": 576 * n_checkpoints,
            "per_token": 1152 * n_checkpoints,
            "checkpoints": n_checkpoints,
        }
        for table, expected in expected_rows.items():
            if len(tables[table]) != expected:
                errors.append({"run_id": rid,
                               "error": f"row_count_{table}",
                               "expected": expected,
                               "actual": len(tables[table])})
            per_step = tables[table].groupby("checkpoint_step").size().to_dict()
            expected_per_step = expected // max(n_checkpoints, 1)
            if any(int(v) != expected_per_step for v in per_step.values()):
                errors.append({"run_id": rid,
                               "error": f"per_step_row_count_{table}",
                               "expected": expected_per_step,
                               "actual": {str(k): int(v) for k, v in per_step.items()}})
        for table, df in tables.items():
            if "cell_id" not in df or set(df["cell_id"].astype(str)) != {rid}:
                errors.append({"run_id": rid,
                               "error": f"cell_id_mismatch_{table}"})
        for table in ("items", "pairs"):
            interventions = set(tables[table]["intervention"].dropna().astype(str))
            if interventions != {run["condition"]}:
                errors.append({"run_id": rid,
                               "error": f"intervention_mismatch_{table}",
                               "values": sorted(interventions)})
        numeric = ["target_sum_log_prob", "target_mean_log_prob",
                   "hotspot_log_prob"]
        for col in numeric:
            if col in tables["items"] and not np.isfinite(
                    tables["items"][col].astype(float)).all():
                errors.append({"run_id": rid, "error": f"nonfinite_items_{col}"})
        if ("checkpoint_content_id" not in tables["checkpoints"] or
                tables["checkpoints"]["checkpoint_content_id"].isna().any()):
            errors.append({"run_id": rid,
                           "error": "missing_checkpoint_content_id"})
        side = tables["checkpoints"]
        provenance_expected = {
            "benchmark": args.benchmark,
            "scoring_version": args.scoring_version,
            "stimuli_manifest_sha256": manifest_sha,
        }
        for column, expected_value in provenance_expected.items():
            values = set(side[column].dropna().astype(str)) if column in side else set()
            if values != {expected_value}:
                errors.append({"run_id": rid,
                               "error": f"provenance_mismatch_{column}",
                               "values": sorted(values),
                               "expected": expected_value})
        if ("stimuli_content_id" not in side or
                side["stimuli_content_id"].isna().any() or
                side["stimuli_content_id"].astype(str).nunique() != 1):
            errors.append({"run_id": rid, "error": "invalid_stimuli_content_id"})
        expected_condition_hashes = json.dumps(
            manifest["conditions"][run["condition"]]["output_sha256"],
            sort_keys=True)
        hashes = (set(side["stimuli_condition_output_sha256"].dropna().astype(str))
                  if "stimuli_condition_output_sha256" in side else set())
        if hashes != {expected_condition_hashes}:
            errors.append({"run_id": rid,
                           "error": "stimuli_condition_hash_mismatch"})
        if not any(e["run_id"] == rid for e in errors):
            verified.append(rid)

    expected_ids = {r["run_id"] for r in runs}
    run_lookup = {r["run_id"]: r for r in runs}

    def coverage(run_ids):
        counts = Counter(
            (run_lookup[rid]["architecture"], run_lookup[rid]["condition"])
            for rid in run_ids
            if rid in run_lookup
        )
        return {
            arch: {
                condition: counts[(arch, condition)]
                for condition in sorted({r["condition"] for r in runs})
                if counts[(arch, condition)]
            }
            for arch in sorted({r["architecture"] for r in runs})
        }

    actual = set()
    actual_table_counts = {}
    for table in TABLES:
        paths = list((args.results_root / table).glob("cell_id=*.parquet"))
        actual_table_counts[table] = len(paths)
        for path in paths:
            actual.add(path.stem[len("cell_id="):])
    report = {
        "format_version": "condition-matched-results-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "scoring_version": args.scoring_version,
        "inventory_sha256": inventory_sha,
        "stimuli_manifest_sha256": manifest_sha,
        "expected_runs": len(runs), "verified_runs": len(verified),
        "expected_checkpoints": sum(len(r["checkpoints"]) for r in runs),
        "expected_coverage": coverage(expected_ids),
        "verified_coverage": coverage(verified),
        "actual_table_file_counts": actual_table_counts,
        "verified_run_ids": verified,
        "missing_result_run_ids": sorted(expected_ids - actual),
        "unexpected_result_run_ids": sorted(actual - expected_ids),
        "errors": errors,
        "complete": (len(verified) == len(runs) and not errors and
                     not (expected_ids - actual)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(f"verified={len(verified)}/{len(runs)} errors={len(errors)} "
          f"complete={report['complete']}")
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
