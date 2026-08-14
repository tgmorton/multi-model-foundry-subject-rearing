#!/usr/bin/env python3
"""Prove local condition-matched Parquets cover an expected inventory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TABLES = ("items", "pairs", "per_token", "checkpoints")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--run-ids", type=Path,
                    help="Optional JSON list limiting the audited tranche.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    inventory = json.loads(args.inventory.read_text())
    wanted = set(json.loads(args.run_ids.read_text())) if args.run_ids else None
    runs = [r for r in inventory["runs"]
            if wanted is None or r["run_id"] in wanted]
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
        if not any(e["run_id"] == rid for e in errors):
            verified.append(rid)

    actual = set()
    for table in TABLES:
        for path in (args.results_root / table).glob("cell_id=*.parquet"):
            actual.add(path.stem[len("cell_id="):])
    expected_ids = {r["run_id"] for r in runs}
    report = {
        "format_version": "condition-matched-results-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runs": len(runs), "verified_runs": len(verified),
        "expected_checkpoints": sum(len(r["checkpoints"]) for r in runs),
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
