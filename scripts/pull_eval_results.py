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
import os
from pathlib import Path

DEFAULT_TABLES = ["items", "pairs", "checkpoints"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="null_subj_v2")
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--dest", default="data/eval_results")
    ap.add_argument("--tables", nargs="+", default=DEFAULT_TABLES,
                    choices=["items", "pairs", "per_token", "checkpoints"])
    ap.add_argument("--endpoint",
                    default=os.environ.get("AWS_ENDPOINT_URL",
                                           "https://s3-west.nrp-nautilus.io"))
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
                local = dest / table / Path(obj["Key"]).name
                if local.exists() and local.stat().st_size == obj["Size"]:
                    n_skip += 1
                    continue
                print(f"pull {obj['Key']} ({obj['Size']/1e6:.1f} MB)")
                s3.download_file(args.bucket, obj["Key"], str(local))
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
