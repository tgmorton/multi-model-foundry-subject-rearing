#!/usr/bin/env python3
"""Pull eval-result parquets from S3 to a local directory (laptop-side).

Mirrors ``s3://<bucket>/eval_results/<benchmark>/<table>/`` into
``<dest>/<table>/`` (size-checked, so re-pulls only fetch what changed),
then prints a ready-to-paste duckdb/pandas snippet.

Usage:
    AWS_PROFILE=nrp python scripts/pull_eval_results.py            # pairs+items+checkpoints
    AWS_PROFILE=nrp python scripts/pull_eval_results.py --tables pairs
    AWS_PROFILE=nrp python scripts/pull_eval_results.py --tables per_token  # big
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

DEFAULT_TABLES = ["items", "pairs", "checkpoints"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="null_subj_v2")
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--dest", default="data/eval_results")
    ap.add_argument("--tables", nargs="+", default=DEFAULT_TABLES,
                    choices=["items", "pairs", "per_token", "checkpoints",
                             "initialization_records"])
    ap.add_argument("--endpoint",
                    default=os.environ.get("AWS_ENDPOINT_URL",
                                           "https://s3-west.nrp-nautilus.io"))
    ap.add_argument("--require-sha256", action="store_true",
                    help="Require and verify the object's sha256 metadata.")
    args = ap.parse_args()

    import boto3
    s3 = boto3.client("s3", endpoint_url=args.endpoint)

    dest = Path(args.dest) / args.benchmark
    n_dl = n_skip = 0
    paginator = s3.get_paginator("list_objects_v2")
    for table in args.tables:
        prefix = f"eval_results/{args.benchmark}/{table}/"
        (dest / table).mkdir(parents=True, exist_ok=True)
        for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].rstrip("/") == prefix.rstrip("/"):
                    continue
                relative = Path(obj["Key"]).relative_to(prefix)
                local = dest / table / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                head = s3.head_object(Bucket=args.bucket, Key=obj["Key"])
                expected_sha = (head.get("Metadata") or {}).get("sha256")
                if args.require_sha256 and not expected_sha:
                    raise RuntimeError(f"missing sha256 metadata: s3://{args.bucket}/{obj['Key']}")
                if (local.exists() and local.stat().st_size == obj["Size"] and
                        (not expected_sha or sha256_file(local) == expected_sha)):
                    n_skip += 1
                    continue
                print(f"pull {obj['Key']} ({obj['Size']/1e6:.1f} MB)")
                tmp = local.with_suffix(local.suffix + ".tmp")
                s3.download_file(args.bucket, obj["Key"], str(tmp))
                if expected_sha and sha256_file(tmp) != expected_sha:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"sha256 mismatch after download: {obj['Key']}")
                tmp.replace(local)
                n_dl += 1
    print(f"done: {n_dl} pulled, {n_skip} current → {dest}/")
    print(f"""
# All runs, item level, all checkpoints:
import duckdb
con = duckdb.connect()
items = con.read_parquet("{dest}/items/*.parquet")
ckpts = con.read_parquet("{dest}/checkpoints/*.parquet")
con.sql(\"\"\"
  SELECT i.*, c.tokens_seen
  FROM items i JOIN ckpts c
    ON i.cell_id = c.cell_id AND i.checkpoint_step = c.checkpoint_step
\"\"\")""")


if __name__ == "__main__":
    main()
