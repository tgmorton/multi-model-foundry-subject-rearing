#!/usr/bin/env python3
"""Phase-1 READ-ONLY storage census of the subject-drop-archive PVC.

Walks the volume and emits a complete inventory — no deletions, no
modifications of existing data (D6 phase 1). Output goes to
/mnt/data/storage_audit/census_<UTC-date>/:

  runs.jsonl    one row per model run dir under models/* and archive/*:
                checkpoint count, resumable (training_state.pt) count,
                step ranges, bytes split by weights/resume/other
  dirs.json     recursive sizes for every top-level dir (2 levels deep)
  SUMMARY.json  totals + df

Classification against the eval keep-list and the S3 registry happens
offline (analysis side) — this job only measures.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/mnt/data"))
MODEL_ROOTS = ["models/production", "models/sweeps", "models/raters",
               "models/initialization", "models/invalidated", "archive"]
_CKPT_RE = re.compile(r"(?:checkpoint|ckpt|epoch)[-_](\d+)$")
_WEIGHT_FILES = {"model.safetensors", "pytorch_model.bin"}
_RESUME_FILES = {"training_state.pt"}


def dir_size(path: Path) -> int:
    total = 0
    stack = [path]
    while stack:
        p = stack.pop()
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def census_run(run_dir: Path) -> dict:
    row = {
        "path": str(run_dir.relative_to(DATA_ROOT)),
        "run_id": run_dir.name,
        "n_ckpt": 0, "n_resumable": 0,
        "steps_min": None, "steps_max": None,
        "resumable_steps": [],
        "bytes_weights": 0, "bytes_resume": 0, "bytes_other": 0,
        "latest_mtime": 0.0,
    }
    steps = []
    try:
        entries = list(os.scandir(run_dir))
    except OSError:
        return row
    for e in entries:
        try:
            if not e.is_dir(follow_symlinks=False):
                row["bytes_other"] += e.stat().st_size
                row["latest_mtime"] = max(row["latest_mtime"], e.stat().st_mtime)
                continue
            m = _CKPT_RE.search(e.name)
            if not m:
                row["bytes_other"] += dir_size(Path(e.path))
                continue
            step = int(m.group(1))
            steps.append(step)
            row["n_ckpt"] += 1
            resumable = False
            with os.scandir(e.path) as ck:
                for f in ck:
                    try:
                        st = f.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    row["latest_mtime"] = max(row["latest_mtime"], st.st_mtime)
                    if f.name in _WEIGHT_FILES:
                        row["bytes_weights"] += st.st_size
                    elif f.name in _RESUME_FILES:
                        row["bytes_resume"] += st.st_size
                        resumable = True
                    else:
                        row["bytes_other"] += st.st_size
            if resumable:
                row["n_resumable"] += 1
                row["resumable_steps"].append(step)
        except OSError:
            pass
    if steps:
        row["steps_min"], row["steps_max"] = min(steps), max(steps)
    row["resumable_steps"] = sorted(row["resumable_steps"])[:64]
    row["bytes_total"] = (row["bytes_weights"] + row["bytes_resume"]
                          + row["bytes_other"])
    return row


def main() -> None:
    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = DATA_ROOT / "storage_audit" / f"census_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- model run census (parallel over run dirs) ---
    run_dirs = []
    for root in MODEL_ROOTS:
        rp = DATA_ROOT / root
        if not rp.is_dir():
            continue
        for e in os.scandir(rp):
            if e.is_dir(follow_symlinks=False):
                run_dirs.append(Path(e.path))
    print(f"model run dirs: {len(run_dirs)}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, row in enumerate(ex.map(census_run, run_dirs)):
            rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(run_dirs)} runs", flush=True)
    with open(out_dir / "runs.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # --- non-model directory sizes (2 levels) ---
    dirs = {}
    skip = {p.split("/")[0] for p in MODEL_ROOTS} | {"storage_audit"}
    top = [e for e in os.scandir(DATA_ROOT) if e.is_dir(follow_symlinks=False)]
    for e in top:
        name = e.name
        if name in skip:
            continue
        sub = {}
        try:
            children = [c for c in os.scandir(e.path)
                        if c.is_dir(follow_symlinks=False)]
        except OSError:
            children = []
        if len(children) <= 64:
            with ThreadPoolExecutor(max_workers=16) as ex:
                for c, sz in zip(children,
                                 ex.map(lambda c: dir_size(Path(c.path)),
                                        children)):
                    sub[c.name] = sz
            loose = dir_size(Path(e.path)) - sum(sub.values())
            dirs[name] = {"total": sum(sub.values()) + max(loose, 0),
                          "children": sub}
        else:
            dirs[name] = {"total": dir_size(Path(e.path)), "children": None}
        print(f"  sized {name}: {dirs[name]['total']/1e9:.1f} GB", flush=True)
    # models/ + archive: totals from the run census
    for root in MODEL_ROOTS:
        sub_rows = [r for r in rows if r["path"].startswith(root)]
        dirs[root] = {"total": sum(r["bytes_total"] for r in sub_rows),
                      "n_runs": len(sub_rows)}
    with open(out_dir / "dirs.json", "w") as f:
        json.dump(dirs, f, indent=2)

    import shutil
    u = shutil.disk_usage(DATA_ROOT)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "df": {"total": u.total, "used": u.used, "free": u.free},
        "n_model_runs": len(rows),
        "model_bytes": sum(r["bytes_total"] for r in rows),
        "model_bytes_weights": sum(r["bytes_weights"] for r in rows),
        "model_bytes_resume": sum(r["bytes_resume"] for r in rows),
        "n_resumable_ckpts": sum(r["n_resumable"] for r in rows),
        "n_ckpts": sum(r["n_ckpt"] for r in rows),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    # best-effort S3 mirror of the (small) census artifacts
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = os.environ.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts")
        for p in out_dir.iterdir():
            s3.upload_file(str(p), bucket, f"storage_audit/census_{stamp}/{p.name}")
        print("census mirrored to S3", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] S3 mirror failed: {e}", flush=True)
    print("CENSUS OK", flush=True)


if __name__ == "__main__":
    main()
