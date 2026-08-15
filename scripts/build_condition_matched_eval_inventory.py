#!/usr/bin/env python3
"""Inventory every readable English Foundry scientific checkpoint on a PVC."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ARCHES = (
    "gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm",
    "mamba_370m",
)
CONDITIONS = (
    "baseline", "remove_expletive_sentences", "impoverish_case",
    "lemmatize_verbs", "enrich_verbal_morphology",
)
RUN_RE = re.compile(
    r"^(gpt2_small|gpt2_medium|gpt2_large|bert_large|lstm|mamba_370m)"
    r"-en-(baseline|remove_expletive_sentences|impoverish_case|"
    r"lemmatize_verbs|enrich_verbal_morphology)-h(\d+)-s(\d+)$"
)


def build_payload(models_root: Path, runs: list, rejected: list,
                  processed_run_ids: set[str], architecture: str | None,
                  complete: bool) -> dict:
    return {
        "format_version": "condition-matched-eval-inventory.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models_root": str(models_root),
        "architecture_shard": architecture,
        "complete": complete,
        "processed_run_ids": sorted(processed_run_ids),
        "run_count": len(runs),
        "checkpoint_count": sum(r["checkpoint_count"] for r in runs),
        "run_counts_by_architecture": dict(sorted(Counter(
            r["architecture"] for r in runs).items())),
        "run_counts_by_condition": dict(sorted(Counter(
            r["condition"] for r in runs).items())),
        "checkpoint_counts_by_architecture": dict(sorted(Counter({
            arch: sum(r["checkpoint_count"] for r in runs
                      if r["architecture"] == arch)
            for arch in ARCHES
            if architecture is None or arch == architecture
        }).items())),
        "runs": sorted(runs, key=lambda r: r["run_id"]),
        "rejected": rejected,
    }


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=Path,
                    default=Path("/mnt/data/models/production"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--include-seed-999", action="store_true")
    ap.add_argument("--architecture", choices=ARCHES)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()
    if not args.models_root.is_dir():
        raise SystemExit(f"missing models root: {args.models_root}")

    runs = []
    rejected = []
    processed_run_ids: set[str] = set()
    if args.resume and args.output.is_file():
        prior = json.loads(args.output.read_text())
        if (prior.get("format_version") != "condition-matched-eval-inventory.v1" or
                prior.get("models_root") != str(args.models_root) or
                prior.get("architecture_shard") != args.architecture):
            raise SystemExit(f"incompatible resume inventory: {args.output}")
        runs = list(prior.get("runs", []))
        rejected = list(prior.get("rejected", []))
        processed_run_ids = set(prior.get("processed_run_ids", []))
        print(f"resuming {args.output}: processed={len(processed_run_ids)}",
              flush=True)
    print(f"inventory start root={args.models_root} architecture={args.architecture or 'all'}",
          flush=True)
    n_new = 0
    for run_dir in sorted(args.models_root.iterdir()):
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        arch, condition, hp_rank, seed = match.groups()
        if args.architecture and arch != args.architecture:
            continue
        if run_dir.name in processed_run_ids:
            continue
        if not run_dir.is_dir():
            continue
        if seed == "999" and not args.include_seed_999:
            continue
        checkpoints = []
        checkpoint_dirs = []
        for checkpoint_dir in run_dir.glob("checkpoint-*"):
            try:
                step = int(checkpoint_dir.name.split("-")[-1])
            except ValueError:
                rejected.append({"path": str(checkpoint_dir),
                                 "reason": "invalid_step"})
                continue
            checkpoint_dirs.append((step, checkpoint_dir))
        for step, checkpoint_dir in sorted(checkpoint_dirs):
            weight = next((checkpoint_dir / name for name in
                           ("model.safetensors", "pytorch_model.bin")
                           if (checkpoint_dir / name).is_file()), None)
            metadata = checkpoint_dir / "metadata.json"
            if weight is None:
                rejected.append({"path": str(checkpoint_dir),
                                 "reason": "missing_weights"})
                continue
            checkpoints.append({
                "step": step,
                "path": str(checkpoint_dir),
                "weight_file": weight.name,
                "weight_size_bytes": weight.stat().st_size,
                # Inventory is a metadata walk, not a content-read pass.  The
                # evaluator hashes each weight exactly once during prefetch and
                # records that content ID in its checkpoint sidecar.
                "metadata_present": metadata.is_file(),
            })
        if checkpoints:
            runs.append({
                "run_id": run_dir.name, "architecture": arch,
                "language": "en", "condition": condition,
                "hp_rank": int(hp_rank), "seed": int(seed),
                "checkpoint_count": len(checkpoints),
                "checkpoints": checkpoints,
            })
        else:
            rejected.append({"path": str(run_dir), "reason": "no_readable_checkpoints"})
        processed_run_ids.add(run_dir.name)
        n_new += 1
        if args.progress_every > 0 and n_new % args.progress_every == 0:
            payload = build_payload(args.models_root, runs, rejected,
                                    processed_run_ids, args.architecture, False)
            write_payload(args.output, payload)
            print(f"progress architecture={args.architecture or 'all'} "
                  f"processed={len(processed_run_ids)} runs={len(runs)} "
                  f"checkpoints={payload['checkpoint_count']} "
                  f"rejected={len(rejected)}", flush=True)

    payload = build_payload(args.models_root, runs, rejected,
                            processed_run_ids, args.architecture, True)
    write_payload(args.output, payload)
    print(f"runs={payload['run_count']} checkpoints={payload['checkpoint_count']} "
          f"rejected={len(rejected)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
