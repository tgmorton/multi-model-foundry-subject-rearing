#!/usr/bin/env python3
"""Prove exact, immutable checkpoint -1 condition-matched coverage.

The initialization benchmark is intentionally separate from trained
checkpoints.  This verifier checks every selected cell, the zero-token
semantics, stimulus/inventory provenance, deterministic state sharing, and
the per-seed provenance record for the audited inventory tranche.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TABLES = ("items", "pairs", "per_token", "checkpoints")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_arch_hp(values: list[str]) -> set[tuple[str, int]]:
    parsed = set()
    for value in values:
        arch, hp = value.rsplit(":", 1)
        parsed.add((arch, int(hp.removeprefix("h"))))
    return parsed


def frame_digest(df, *, drop: tuple[str, ...] = ("cell_id",)) -> str:
    """Stable comparison digest, including ndarray/list-valued columns."""
    import numpy as np
    import pandas as pd

    out = df.drop(columns=[c for c in drop if c in df], errors="ignore").copy()
    order = [c for c in ("category", "condition", "item_id",
                         "pronoun_status", "checkpoint_step") if c in out]
    if order:
        out = out.sort_values(order, kind="stable").reset_index(drop=True)
    out = out.reindex(sorted(out.columns), axis=1)

    def normalize(value):
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist(), separators=(",", ":"),
                              allow_nan=True)
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              allow_nan=True)
        return value

    for column in out.select_dtypes(include=["object", "str"]).columns:
        out[column] = out[column].map(normalize)
    hashes = pd.util.hash_pandas_object(out, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--stimuli-manifest", type=Path, required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-ids", type=Path,
                    help="Optional JSON list limiting the audited tranche.")
    ap.add_argument("--exclude-arch-hp", action="append", default=[],
                    metavar="ARCH:HP")
    ap.add_argument("--only-arch-hp", action="append", default=[],
                    metavar="ARCH:HP")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    inventory = json.loads(args.inventory.read_text())
    if inventory.get("format_version") != "condition-matched-eval-inventory.v1":
        raise SystemExit("unexpected inventory format")
    if inventory.get("rejected"):
        raise SystemExit("inventory contains rejected checkpoint paths")
    inventory_sha = sha256_file(args.inventory)
    manifest = json.loads(args.stimuli_manifest.read_text())
    manifest_sha = sha256_file(args.stimuli_manifest)
    if manifest.get("vetted") is not True:
        raise SystemExit("stimulus manifest is not gold-vetted")

    wanted = set(json.loads(args.run_ids.read_text())) if args.run_ids else None
    excluded = parse_arch_hp(args.exclude_arch_hp)
    only = parse_arch_hp(args.only_arch_hp)
    runs = []
    for run in inventory["runs"]:
        lane = (run["architecture"], int(run["hp_rank"]))
        if wanted is not None and run["run_id"] not in wanted:
            continue
        if lane in excluded:
            continue
        if only and lane not in only:
            continue
        runs.append(run)
    if not runs:
        raise SystemExit("selection contains no runs")

    errors: list[dict] = []
    verified: list[str] = []
    group_states: dict[tuple[str, int], set[str]] = {}
    score_digests: dict[tuple[str, int, str, str], set[str]] = {}
    expected_groups: dict[tuple[str, int], set[str]] = {}
    for run in runs:
        rid = run["run_id"]
        group = (run["architecture"], int(run["seed"]))
        expected_groups.setdefault(group, set()).add(rid)
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

        expected_rows = {"items": 1152, "pairs": 576,
                         "per_token": 1152, "checkpoints": 1}
        for table, expected in expected_rows.items():
            frame = tables[table]
            if len(frame) != expected:
                errors.append({"run_id": rid,
                               "error": f"row_count_{table}",
                               "expected": expected, "actual": len(frame)})
            steps = set(map(int, frame["checkpoint_step"].unique()))
            if steps != {-1}:
                errors.append({"run_id": rid,
                               "error": f"checkpoint_step_{table}",
                               "values": sorted(steps)})
            if "cell_id" not in frame or set(frame["cell_id"].astype(str)) != {rid}:
                errors.append({"run_id": rid,
                               "error": f"cell_id_mismatch_{table}"})

        for table in ("items", "pairs"):
            values = set(tables[table]["intervention"].dropna().astype(str))
            if values != {run["condition"]}:
                errors.append({"run_id": rid,
                               "error": f"intervention_mismatch_{table}",
                               "values": sorted(values)})
        for column in ("target_sum_log_prob", "target_mean_log_prob",
                       "hotspot_log_prob"):
            values = tables["items"][column].dropna().astype(float)
            if not np.isfinite(values).all():
                errors.append({"run_id": rid,
                               "error": f"nonfinite_items_{column}"})

        side = tables["checkpoints"]
        expected_side = {
            "architecture": run["architecture"],
            "intervention": run["condition"],
            "hp_rank": int(run["hp_rank"]),
            "seed": int(run["seed"]),
            "checkpoint_step": -1,
            "tokens_seen": 0,
            "epoch": 0,
            "benchmark": args.benchmark,
            "inventory_sha256": inventory_sha,
            "stimuli_manifest_sha256": manifest_sha,
        }
        for column, expected in expected_side.items():
            values = set(side[column].dropna().tolist()) if column in side else set()
            if values != {expected}:
                errors.append({"run_id": rid,
                               "error": f"sidecar_mismatch_{column}",
                               "expected": expected,
                               "values": sorted(map(str, values))})
        state_values = (set(side["model_state_sha256"].dropna().astype(str))
                        if "model_state_sha256" in side else set())
        if len(state_values) != 1 or len(next(iter(state_values), "")) != 64:
            errors.append({"run_id": rid, "error": "invalid_model_state_sha256"})
        else:
            group_states.setdefault(group, set()).update(state_values)
        expected_hashes = json.dumps(
            manifest["conditions"][run["condition"]]["output_sha256"],
            sort_keys=True)
        hashes = (set(side["stimuli_condition_output_sha256"].dropna().astype(str))
                  if "stimuli_condition_output_sha256" in side else set())
        if hashes != {expected_hashes}:
            errors.append({"run_id": rid,
                           "error": "stimuli_condition_hash_mismatch"})
        stimuli_ids = (set(side["stimuli_content_id"].dropna().astype(str))
                       if "stimuli_content_id" in side else set())
        if len(stimuli_ids) != 1:
            errors.append({"run_id": rid, "error": "invalid_stimuli_content_id"})

        for table in ("items", "pairs", "per_token"):
            key = (run["architecture"], int(run["seed"]),
                   run["condition"], table)
            score_digests.setdefault(key, set()).add(frame_digest(tables[table]))
        if not any(e.get("run_id") == rid for e in errors):
            verified.append(rid)

    for (arch, seed), states in group_states.items():
        if len(states) != 1:
            errors.append({"group": f"{arch}:s{seed}",
                           "error": "state_hash_varies_across_cells",
                           "values": sorted(states)})
    for key, digests in score_digests.items():
        if len(digests) != 1:
            arch, seed, condition, table = key
            errors.append({"group": f"{arch}:s{seed}:{condition}",
                           "error": f"hp_outputs_differ_{table}",
                           "digests": sorted(digests)})

    records_root = args.results_root / "initialization_records"
    verified_records = []
    for (arch, seed), expected_cells in expected_groups.items():
        record = records_root / arch / f"seed-{seed}" / f"inventory-{inventory_sha}.json"
        if not record.is_file():
            errors.append({"group": f"{arch}:s{seed}",
                           "error": "missing_initialization_record"})
            continue
        try:
            payload = json.loads(record.read_text())
        except Exception as exc:
            errors.append({"group": f"{arch}:s{seed}",
                           "error": f"unreadable_initialization_record: {exc}"})
            continue
        record_expected = {
            "architecture": arch, "seed": seed, "benchmark": args.benchmark,
            "inventory_sha256": inventory_sha,
            "stimuli_manifest_sha256": manifest_sha,
        }
        for field, expected in record_expected.items():
            if payload.get(field) != expected:
                errors.append({"group": f"{arch}:s{seed}",
                               "error": f"record_mismatch_{field}"})
        if set(payload.get("cell_ids", [])) != expected_cells:
            errors.append({"group": f"{arch}:s{seed}",
                           "error": "record_cell_coverage_mismatch"})
        states = group_states.get((arch, seed), set())
        if len(states) != 1 or payload.get("model_state_sha256") not in states:
            errors.append({"group": f"{arch}:s{seed}",
                           "error": "record_state_hash_mismatch"})
        verified_records.append(f"{arch}:s{seed}")

    expected_ids = {run["run_id"] for run in runs}
    actual_by_table = {}
    for table in TABLES:
        actual_by_table[table] = {
            path.stem.removeprefix("cell_id=")
            for path in (args.results_root / table).glob("cell_id=*.parquet")
        }
    report = {
        "format_version": "condition-matched-init-results-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_sha256": inventory_sha,
        "stimuli_manifest_sha256": manifest_sha,
        "expected_runs": len(runs),
        "verified_runs": len(verified),
        "expected_initialization_groups": len(expected_groups),
        "verified_initialization_records": len(verified_records),
        "missing_by_table": {
            table: sorted(expected_ids - actual)
            for table, actual in actual_by_table.items()
        },
        # Extra outputs from a separate immutable tranche are allowed in the
        # shared benchmark folder and remain auditable by their own inventory.
        "unexpected_by_table": {
            table: sorted(actual - expected_ids)
            for table, actual in actual_by_table.items()
        },
        "errors": errors,
    }
    report["complete"] = (
        len(verified) == len(runs)
        and len(verified_records) == len(expected_groups)
        and not errors
        and all(not values for values in report["missing_by_table"].values())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(f"verified={len(verified)}/{len(runs)} "
          f"records={len(verified_records)}/{len(expected_groups)} "
          f"errors={len(errors)} complete={report['complete']}")
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
