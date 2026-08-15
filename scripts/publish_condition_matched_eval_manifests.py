#!/usr/bin/env python3
"""Publish immutable matched-evaluation inventory/provenance manifests to S3."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_once(s3, bucket: str, path: Path, key: str,
                metadata: dict[str, str]) -> str:
    from botocore.exceptions import ClientError

    digest = sha256_file(path)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        prior = (head.get("Metadata") or {}).get("sha256")
        if prior == digest:
            return "cached"
        raise RuntimeError(
            f"S3 collision at s3://{bucket}/{key}: existing sha256={prior!r}, "
            f"local sha256={digest}")
    md = {k: str(v) for k, v in metadata.items()}
    md["sha256"] = digest
    s3.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": md})
    return "uploaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="null_subj_v2_condition_matched_v1")
    ap.add_argument("--bucket", default=os.environ.get(
        "REGISTRY_BUCKET", "thomas-subject-drop-artifacts"))
    ap.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL"))
    ap.add_argument("--code-ref", default=os.environ.get("GIT_REF", "unknown"))
    ap.add_argument("--file", action="append", type=Path, required=True)
    args = ap.parse_args()

    import boto3
    s3 = boto3.client("s3", endpoint_url=args.endpoint,
                      region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-1"))
    prefix = f"eval_results/{args.benchmark}/manifests"
    metadata = {"benchmark": args.benchmark, "code_ref": args.code_ref,
                "schema": "condition-matched-eval-manifest-v1"}
    for path in args.file:
        if not path.is_file():
            raise SystemExit(f"missing manifest file: {path}")
        key = f"{prefix}/{path.name}"
        status = upload_once(s3, args.bucket, path, key, metadata)
        print(f"{status} s3://{args.bucket}/{key} sha256={sha256_file(path)}",
              flush=True)


if __name__ == "__main__":
    main()
