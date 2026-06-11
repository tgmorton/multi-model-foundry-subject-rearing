#!/usr/bin/env python3
"""Sync eval-result parquets from the PVC to S3 (idempotent).

Walks ``<root>/{items,pairs,per_token,checkpoints}/*.parquet`` and uploads
anything missing from ``s3://<bucket>/eval_results/<benchmark>/...`` or
whose size differs (eval re-runs overwrite in place). Run as a small
non-GPU pod after (or during) an eval wave; safe to re-run any time.

Usage (pod-side, env supplies AWS_* + REGISTRY_BUCKET):
    python scripts/sync_eval_results_s3.py \
        --root /mnt/data/eval_v2/null_subj_v2 --benchmark null_subj_v2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

TABLES = ["items", "pairs", "per_token", "checkpoints"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/eval_v2/null_subj_v2")
    ap.add_argument("--benchmark", default="null_subj_v2")
    ap.add_argument("--bucket", default=os.environ.get("REGISTRY_BUCKET",
                                                       "thomas-subject-drop-artifacts"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import boto3
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))

    # Existing remote sizes in one listing pass.
    remote: dict[str, int] = {}
    prefix = f"eval_results/{args.benchmark}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            remote[obj["Key"]] = obj["Size"]

    n_up = n_skip = 0
    for table in TABLES:
        tdir = Path(args.root) / table
        if not tdir.is_dir():
            continue
        for f in sorted(tdir.glob("*.parquet")):
            key = f"{prefix}{table}/{f.name}"
            size = f.stat().st_size
            if remote.get(key) == size:
                n_skip += 1
                continue
            print(f"upload {f} → s3://{args.bucket}/{key} ({size/1e6:.1f} MB)")
            if not args.dry_run:
                s3.upload_file(str(f), args.bucket, key)
            n_up += 1
    print(f"done: {n_up} uploaded, {n_skip} already current")


if __name__ == "__main__":
    main()
