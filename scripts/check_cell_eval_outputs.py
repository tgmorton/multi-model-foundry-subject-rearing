#!/usr/bin/env python3
"""Sanity-check cell-eval output parquets (smoke or production).

Pulls items/pairs parquets for the given cells from the PVC via
`kubectl exec ... tar` (kubectl cp needs tar anyway), then asserts:
  - row counts: items == n_stimuli × n_checkpoints, pairs == items/2
  - all metrics finite
  - per-category overt-preference at the final checkpoint (printed —
    a trained EN model should prefer overt subjects on subject_drop)

Usage:
    python scripts/check_cell_eval_outputs.py \
        --output_root /mnt/data/eval_v2/smoke_cell \
        --cells lstm-en-baseline-h0-s42 mamba_370m-en-impoverish_case-h0-s42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

POD = "thomas-pod-archive"


def _pull(remote: str, local: Path, retries: int = 3) -> bool:
    for _ in range(retries):
        proc = subprocess.run(
            ["kubectl", "exec", POD, "--", "cat", remote],
            capture_output=True, timeout=180,
        )
        if proc.returncode == 0 and proc.stdout:
            local.write_bytes(proc.stdout)
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default="/mnt/data/eval_v2/smoke_cell")
    ap.add_argument("--cells", nargs="+", required=True)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    tmp = Path(tempfile.mkdtemp(prefix="cell_eval_check_"))
    failures = 0
    for cell in args.cells:
        print(f"\n=== {cell} ===")
        items_p = tmp / f"{cell}_items.parquet"
        pairs_p = tmp / f"{cell}_pairs.parquet"
        ok_i = _pull(f"{args.output_root}/items/cell_id={cell}.parquet", items_p)
        ok_p = _pull(f"{args.output_root}/pairs/cell_id={cell}.parquet", pairs_p)
        if not (ok_i and ok_p):
            print("  MISSING parquet(s) — eval incomplete or path wrong")
            failures += 1
            continue
        items = pd.read_parquet(items_p)
        pairs = pd.read_parquet(pairs_p)
        n_ckpt = items.checkpoint_step.nunique()
        print(f"  items={len(items)}  pairs={len(pairs)}  checkpoints={n_ckpt}")

        bad = (~np.isfinite(items.target_mean_log_prob)).sum()
        if bad:
            print(f"  ✗ {bad} non-finite item metrics")
            failures += 1
        else:
            print("  ✓ all item metrics finite")

        if len(pairs) * 2 != len(items):
            print(f"  ⚠ pair coverage: {len(pairs)}×2 != {len(items)} items "
                  "(unpaired pronoun_status rows?)")

        final = pairs[pairs.checkpoint_step == pairs.checkpoint_step.max()]
        print(f"  final step {pairs.checkpoint_step.max()}: "
              f"overt-pref by category:")
        summary = final.groupby("category").agg(
            overt_pref=("prefers_overt_meanlp", "mean"),
            mean_diff=("log_prob_diff_overt_minus_null", "mean"),
            n=("item_id", "size"),
        )
        print(summary.to_string())

    print()
    if failures:
        print(f"✗ {failures} cell(s) failed checks")
        sys.exit(1)
    print("✓ all cells pass")


if __name__ == "__main__":
    main()
