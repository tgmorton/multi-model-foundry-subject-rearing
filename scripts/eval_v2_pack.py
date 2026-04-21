#!/usr/bin/env python3
"""GPU packing experiment: N parallel eval_v2_bench workers on one GPU.

Spawns N subprocesses, each running scripts/eval_v2_bench.py against the
same checkpoint + stimuli but with a unique cell_id + output subdir +
scratch subdir. All N share one physical GPU.

Measures total wall-time and aggregates per-worker bench_summary.json
from each subprocess so we can compute throughput (evals/sec) as a
function of N.

Run:

    python scripts/eval_v2_pack.py \\
        --n_workers 4 \\
        --checkpoint_root /mnt/data/models/exp0_baseline_90M_smoke_resume \\
        --tokenizer_dir /mnt/data/tokenizers/exp0_baseline_90M_smoke \\
        --stimuli_dir evaluation/stimuli/null-subj-v2/staging \\
        --output_root /mnt/data/eval_v2/pack_n4 \\
        --scratch_dir /scratch/pack_n4 \\
        --summary_path /mnt/data/eval_v2/pack_n4/packing_summary.json

Each worker:
- Independent CUDA context (no MPS required).
- Own cell_id (`pack_w{i}`) so cache markers don't collide.
- Own output subdir (`output_root/worker_{i}`) so partition layouts
  don't merge.
- Own scratch subdir (`scratch_dir/w{i}`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="GPU packing experiment")
    p.add_argument("--n_workers", type=int, required=True)
    p.add_argument("--checkpoint_root", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--stimuli_dir", required=True)
    p.add_argument("--output_root", required=True,
                   help="Shared root on durable fs; workers write to subdirs.")
    p.add_argument("--scratch_dir", required=True,
                   help="Pod-local ephemeral root; workers get subdirs.")
    p.add_argument("--summary_path", required=True,
                   help="Where to write the consolidated packing summary JSON.")
    p.add_argument("--scoring_version", default="pack-v1",
                   help="Varied per N so repeated trials don't hit cache.")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    output_root = Path(args.output_root)
    scratch_dir = Path(args.scratch_dir)
    summary_path = Path(args.summary_path)

    output_root.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: log GPU props before we spawn anything.
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            print(f"[PACK] GPU: {dev_name} total_MB={total_mb:.1f}")
    except Exception as e:
        print(f"[PACK] torch probe failed: {e}")

    # Spawn N workers in parallel.
    procs = []
    t0 = time.perf_counter()
    for i in range(args.n_workers):
        worker_out = output_root / f"worker_{i}"
        worker_scratch = scratch_dir / f"w{i}"
        cmd = [
            sys.executable, "-u", "scripts/eval_v2_bench.py",
            "--checkpoint_root", args.checkpoint_root,
            "--tokenizer_dir", args.tokenizer_dir,
            "--stimuli_dir", args.stimuli_dir,
            "--output_root", str(worker_out),
            "--scratch_dir", str(worker_scratch),
            "--cell_id", f"pack_w{i}",
            "--architecture", "gpt2",
            "--intervention", "baseline",
            "--rep", "0",
            "--batch_size", str(args.batch_size),
            "--device", "auto",
            "--scoring_version", args.scoring_version,
        ]
        # Redirect each worker's stdout/stderr to a file so they don't
        # interleave and we can inspect individually.
        log_path = output_root / f"worker_{i}.log"
        logf = open(log_path, "wb")
        p_proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=os.environ.copy(),
        )
        procs.append((p_proc, logf, worker_out, i))
        print(f"[PACK] spawned worker {i} pid={p_proc.pid}")

    # Wait for all.
    rcs = []
    for proc, logf, _, i in procs:
        rc = proc.wait()
        logf.close()
        rcs.append((i, rc))
        print(f"[PACK] worker {i} exited rc={rc}")
    total = time.perf_counter() - t0

    # Aggregate per-worker bench_summary.json
    worker_summaries = []
    for _, _, worker_out, i in procs:
        sp = worker_out / "bench_summary.json"
        if sp.exists():
            worker_summaries.append({"worker": i, **json.loads(sp.read_text())})
        else:
            worker_summaries.append({"worker": i, "error": "no summary"})

    # Throughput summary across workers.
    # Each worker processes n_checkpoints × n_stimuli_rows items. Total
    # is n_workers × that. Divide by total wall-time for evals/sec.
    total_items = 0
    total_ckpts = 0
    for w in worker_summaries:
        if "error" in w:
            continue
        total_items += w.get("n_stimuli_rows", 0) * w.get("n_checkpoints", 0)
        total_ckpts += w.get("n_checkpoints", 0)

    out = {
        "n_workers": args.n_workers,
        "total_wall_secs": total,
        "total_items_processed": total_items,
        "total_checkpoints_processed": total_ckpts,
        "items_per_sec": total_items / total if total > 0 else 0,
        "ckpts_per_sec": total_ckpts / total if total > 0 else 0,
        "worker_return_codes": rcs,
        "worker_summaries": worker_summaries,
    }
    # Also capture final GPU info.
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["gpu_total_MB"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            )
    except Exception:
        pass

    summary_path.write_text(json.dumps(out, indent=2))
    print(f"[PACK] n_workers={args.n_workers} total_secs={total:.2f} "
          f"ckpts/sec={out['ckpts_per_sec']:.3f}")
    print(f"[PACK] wrote {summary_path}")

    # Non-zero if any worker failed.
    if any(rc != 0 for _, rc in rcs):
        sys.exit(1)


if __name__ == "__main__":
    main()
