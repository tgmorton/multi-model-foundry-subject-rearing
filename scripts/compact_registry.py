#!/usr/bin/env python3
"""
Rebuild s3://<bucket>/run_registry/registry.parquet from the per-run
JSONs under s3://<bucket>/run_registry/by_run/.

The per-run JSONs are the authoritative source of truth (concurrent
writers, atomic per-key PutObject). This script materializes them into a
single Parquet file that analyses can load in one call.

Runs either on-demand (CLI) or from a K8s CronJob on a ~hourly cadence.
Cheap: ~17K small objects at full scale, seconds to minutes.

Usage:
    python scripts/compact_registry.py [--bucket BUCKET]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from typing import Any, Dict, List

import boto3
from botocore.config import Config
import pyarrow as pa
import pyarrow.parquet as pq

# Import from the installed package so we reuse the same constants/client
# configuration the write path uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_foundry.registry import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    BY_RUN_PREFIX,
    iter_all_records,
)

logger = logging.getLogger("compact_registry")


REGISTRY_KEY = "run_registry/registry.parquet"


def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce list/missing fields to pyarrow-friendly shapes.

    Keeps every key the record carries so the Parquet schema is a
    superset of the JSON schema; missing fields come through as nulls.
    """
    # Ensure list-typed fields are lists (not None) so pyarrow infers a
    # list type and downstream consumers don't have to special-case.
    for k in ("checkpoint_paths", "eval_parquet_paths"):
        if record.get(k) is None:
            record[k] = []
    return record


def compact(bucket: str) -> int:
    """Pull every run record, write them to the single registry parquet.
    Returns the number of records compacted."""
    records: List[Dict[str, Any]] = []
    # Side-step iter_all_records' env-driven bucket to allow override.
    os.environ["REGISTRY_BUCKET"] = bucket
    for rec in iter_all_records():
        records.append(_normalize(rec))
    logger.info("collected %d records from s3://%s/%s/",
                len(records), bucket, BY_RUN_PREFIX)

    if not records:
        logger.info("nothing to compact; skipping write")
        return 0

    table = pa.Table.from_pylist(records)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT),
        config=Config(signature_version="s3v4",
                      s3={"addressing_style": "path"}),
    )
    client.put_object(
        Bucket=bucket,
        Key=REGISTRY_KEY,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )
    logger.info("wrote s3://%s/%s  (%d rows, %.1f KiB)",
                bucket, REGISTRY_KEY, len(records), buf.tell() / 1024)
    return len(records)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=os.environ.get("REGISTRY_BUCKET", DEFAULT_BUCKET))
    args = ap.parse_args()
    n = compact(args.bucket)
    print(f"compacted {n} records to s3://{args.bucket}/{REGISTRY_KEY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
