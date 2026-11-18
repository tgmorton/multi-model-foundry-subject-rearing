#!/usr/bin/env python3
"""Publish cell results that were kept on a volume to S3.

When the storage cluster behind S3 refuses writes, cells are run with
`launch_cell_evals.py --out-pvc <claim>` and `--output_root /mnt/out/eval_v2/<benchmark>
--results-to-pvc --no-registry`: the parquets and the done-markers stay on that volume and nothing
is uploaded. This copies them to S3 afterwards, under the same keys and with the same object
metadata as eval_v2_cell.py's own upload, so pull_eval_results.py and the resume check see no
difference. An object already there with the same sha256 is left alone; a different one is an error.
The registry's eval status is not written (the cells ran with --no-registry).

    python scripts/publish_eval_results_from_pvc.py --root /mnt/out/eval_v2/<benchmark>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

TABLES = ("items", "pairs", "per_token", "checkpoints")
META = ("benchmark", "scoring_version", "run_id", "cell_id", "stimuli_manifest_sha256", "stimuli_content_id")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_once(s3, bucket: str, path: Path, key: str, metadata: dict) -> str:
    from botocore.exceptions import ClientError
    digest = sha256(path)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code", "")) not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        prior = (head.get("Metadata") or {}).get("sha256")
        if prior == digest:
            return "cached"
        raise RuntimeError(f"S3 collision at s3://{bucket}/{key}: existing sha256={prior!r}, local sha256={digest}")
    md = {k: str(v) for k, v in metadata.items() if v is not None}
    md["sha256"] = digest
    s3.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": md})
    return "uploaded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="the --output_root the cells were run with")
    ap.add_argument("--bucket", default=os.environ.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import boto3
    import pandas as pd
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
    cells = sorted(p.name for p in (a.root / "checkpoints").glob("cell_id=*.parquet"))
    if not cells:
        sys.exit(f"no cells under {a.root}/checkpoints")
    done = failed = 0
    for name in cells:
        files = {t: a.root / t / name for t in TABLES}
        missing = [t for t, f in files.items() if not f.is_file()]
        if missing:
            print(f"SKIP {name}: missing {missing}")
            failed += 1
            continue
        row = pd.read_parquet(files["checkpoints"]).iloc[0]
        md = {k: row[k] for k in META}
        prefix = f"eval_results/{md['benchmark']}"
        if a.dry_run:
            print(f"would publish {name} → s3://{a.bucket}/{prefix}/")
            continue
        try:
            status = [upload_once(s3, a.bucket, f, f"{prefix}/{t}/{name}", md) for t, f in files.items()]
        except Exception as exc:  # one bad cell must not stop the rest
            print(f"FAILED {name}: {str(exc)[:160]}")
            failed += 1
            continue
        print(f"{name}: {', '.join(status)}")
        done += 1
    print(f"{done} cell(s) published, {failed} not published, of {len(cells)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
