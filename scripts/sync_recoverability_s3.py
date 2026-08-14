#!/usr/bin/env python3
"""Upload the recoverability scoring outputs (+ fold assignments) to S3.

Walks --root (default /mnt/data/recoverability) and uploads every file to
s3://$REGISTRY_BUCKET/recoverability/<relative path>, skipping keys that
already exist with the same size (idempotent re-runs). boto3 handles
multipart correctly (do NOT use `aws s3 cp` — NRP multipart bug >80MB).
"""

import os
import sys
from pathlib import Path

import boto3


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/recoverability")
    bucket = os.environ.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts")
    s3 = boto3.client("s3")

    existing = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="recoverability/"):
        for obj in page.get("Contents", []):
            existing[obj["Key"]] = obj["Size"]

    n_up = n_skip = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = f"recoverability/{path.relative_to(root)}"
        size = path.stat().st_size
        if existing.get(key) == size:
            n_skip += 1
            continue
        s3.upload_file(str(path), bucket, key)
        n_up += 1
        print(f"  up {key} ({size/1e6:.1f} MB)", flush=True)
    print(f"SYNC OK: {n_up} uploaded, {n_skip} already current", flush=True)


if __name__ == "__main__":
    main()
