#!/usr/bin/env python3
"""Merge complete per-architecture checkpoint inventory shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ARCHES = ("gpt2_small", "gpt2_medium", "gpt2_large", "bert_large",
          "lstm", "mamba_370m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    runs = []
    rejected = []
    models_root = None
    shard_files = {}
    for arch in ARCHES:
        path = args.input_dir / f"checkpoint_inventory_{arch}.json"
        if not path.is_file():
            raise SystemExit(f"missing inventory shard: {path}")
        payload = json.loads(path.read_text())
        if (payload.get("format_version") != "condition-matched-eval-inventory.v1" or
                payload.get("architecture_shard") != arch or
                payload.get("complete") is not True):
            raise SystemExit(f"incomplete or incompatible shard: {path}")
        if models_root is None:
            models_root = payload["models_root"]
        elif payload["models_root"] != models_root:
            raise SystemExit("shards disagree on models_root")
        shard_files[arch] = str(path)
        runs.extend(payload["runs"])
        rejected.extend(payload["rejected"])

    run_ids = [r["run_id"] for r in runs]
    if len(run_ids) != len(set(run_ids)):
        raise SystemExit("duplicate run IDs across inventory shards")
    payload = {
        "format_version": "condition-matched-eval-inventory.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models_root": models_root,
        "complete": True,
        "source_shards": shard_files,
        "run_count": len(runs),
        "checkpoint_count": sum(r["checkpoint_count"] for r in runs),
        "run_counts_by_architecture": dict(sorted(Counter(
            r["architecture"] for r in runs).items())),
        "run_counts_by_condition": dict(sorted(Counter(
            r["condition"] for r in runs).items())),
        "checkpoint_counts_by_architecture": {
            arch: sum(r["checkpoint_count"] for r in runs
                      if r["architecture"] == arch)
            for arch in ARCHES
        },
        "runs": sorted(runs, key=lambda r: r["run_id"]),
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(f"runs={payload['run_count']} checkpoints={payload['checkpoint_count']} "
          f"rejected={len(rejected)}", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
