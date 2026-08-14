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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=Path,
                    default=Path("/mnt/data/models/production"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--include-seed-999", action="store_true")
    args = ap.parse_args()
    if not args.models_root.is_dir():
        raise SystemExit(f"missing models root: {args.models_root}")

    runs = []
    rejected = []
    for run_dir in sorted(args.models_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        arch, condition, hp_rank, seed = match.groups()
        if seed == "999" and not args.include_seed_999:
            continue
        checkpoints = []
        for checkpoint_dir in sorted(run_dir.glob("checkpoint-*"),
                                     key=lambda p: int(p.name.split("-")[-1])):
            try:
                step = int(checkpoint_dir.name.split("-")[-1])
            except ValueError:
                rejected.append({"path": str(checkpoint_dir),
                                 "reason": "invalid_step"})
                continue
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

    payload = {
        "format_version": "condition-matched-eval-inventory.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models_root": str(args.models_root),
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
        }).items())),
        "runs": runs,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(f"runs={payload['run_count']} checkpoints={payload['checkpoint_count']} "
          f"rejected={len(rejected)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
