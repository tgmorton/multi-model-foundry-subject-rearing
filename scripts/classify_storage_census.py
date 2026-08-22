#!/usr/bin/env python3
"""Classify the Phase-1 storage census into keep / strip / delete buckets.

READ-ONLY inputs (census jsonl, registry snapshot, eval keep-list CSVs);
outputs are the audit tables + Phase-2/3 manifests for review — nothing
here deletes anything (D6: deletions only happen in later phases after
Thomas reviews the manifests).

Buckets (models/*):
  keep_eval_registered   run_id in the condition_matched_v1 keep-list ->
                         weights stay; resume states strippable (Phase 3)
  hold_pending_eval      production runs not yet in the keep-list but
                         plausibly awaiting a delta-tranche eval (h1 lane
                         or recent COMPLETE) -> untouched this round
  keep_infrastructure    raters, initialization states, detsmoke smokes
  phase2_sweeps          models/sweeps trial checkpoints (winners long
                         extracted to data/sweep_winners/) -> delete after
                         review
  phase2_failed_orphan   registry FAILED, stale-RUNNING (>14d no update,
                         dir mtime old), invalidated/, or no registry
                         record at all -> delete after review
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import pandas as pd

AUDIT = Path("data/storage_audit")
KEEPLIST_GLOB = ("analysis/eval_v2/figures/foundry_trajectories/"
                 "condition_matched_v1/*.csv")
STALE_DAYS = 14


def load_keep_list() -> set:
    cells = set()
    for f in glob.glob(KEEPLIST_GLOB):
        df = pd.read_csv(f, usecols=lambda c: c == "cell_id")
        if "cell_id" in df.columns:
            cells.update(df.cell_id.unique())
    return cells


def main() -> None:
    with open(AUDIT / "runs.jsonl") as f:
        runs = pd.DataFrame(json.loads(line) for line in f if line.strip())
    reg = pd.read_parquet(AUDIT / "registry_by_run.parquet")
    reg = reg.drop_duplicates("run_id", keep="last").set_index("run_id")
    keep = load_keep_list()
    print(f"census runs: {len(runs)}, registry: {len(reg)}, keep-list: {len(keep)}")

    now = time.time()

    def classify(r):
        root = r["path"].split("/")[1] if r["path"].startswith("models/") else r["path"].split("/")[0]
        rid = r["run_id"]
        in_reg = rid in reg.index
        status = reg.loc[rid, "status"] if in_reg else None
        if root == "sweeps":
            return "phase2_sweeps"
        if root in ("raters", "initialization") or rid.startswith("detsmoke"):
            return "keep_infrastructure"
        if root == "invalidated":
            return "phase2_failed_orphan"
        if root == "archive":
            return "keep_eval_registered"
        # production
        if rid in keep:
            return "keep_eval_registered"
        stale = (now - r["latest_mtime"]) > STALE_DAYS * 86400
        if status == "FAILED":
            return "phase2_failed_orphan"
        if status == "COMPLETE" and not stale:
            return "hold_pending_eval"
        if status == "RUNNING" and not stale:
            return "hold_pending_eval"  # possibly live
        if not in_reg:
            return "phase2_failed_orphan"
        return "phase2_failed_orphan" if stale else "hold_pending_eval"

    runs["bucket"] = runs.apply(classify, axis=1)
    runs["gb_total"] = runs.bytes_total / 1e9
    runs["gb_resume"] = runs.bytes_resume / 1e9

    summary = (runs.groupby("bucket")
               .agg(n_runs=("run_id", "size"),
                    tb_total=("bytes_total", lambda s: s.sum() / 1e12),
                    tb_weights=("bytes_weights", lambda s: s.sum() / 1e12),
                    tb_resume=("bytes_resume", lambda s: s.sum() / 1e12),
                    n_ckpts=("n_ckpt", "sum"),
                    n_resumable=("n_resumable", "sum"))
               .round(2))
    print(summary.to_string())

    runs.to_parquet(AUDIT / "classification.parquet")
    for bucket, fname in (("phase2_sweeps", "phase2_sweeps_manifest.csv"),
                          ("phase2_failed_orphan", "phase2_failed_orphan_manifest.csv")):
        sub = runs[runs.bucket == bucket][
            ["path", "run_id", "n_ckpt", "n_resumable", "gb_total", "latest_mtime"]
        ].sort_values("gb_total", ascending=False)
        sub.to_csv(AUDIT / fname, index=False)
        print(f"wrote {fname}: {len(sub)} dirs, {sub.gb_total.sum()/1000:.2f} TB")
    strip = runs[(runs.bucket == "keep_eval_registered") & (runs.n_resumable > 0)][
        ["path", "run_id", "n_resumable", "resumable_steps", "gb_resume"]
    ].sort_values("gb_resume", ascending=False)
    strip.to_csv(AUDIT / "phase3_resume_strip_manifest.csv", index=False)
    print(f"wrote phase3_resume_strip_manifest.csv: {len(strip)} runs, "
          f"{strip.gb_resume.sum()/1000:.2f} TB reclaimable")
    summary.to_csv(AUDIT / "bucket_summary.csv")


if __name__ == "__main__":
    main()
