#!/usr/bin/env python3
"""CPU-only fan-out for condition-matched initialization scores.

A GPU scorer publishes one real representative cell per condition. This stage
copies those scientifically identical checkpoint -1 scores to the other
HP/cell identities sharing `(architecture, seed, condition)`, updates their
sidecars, and performs the repeated Parquet/S3 work without reserving a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path


TABLES = ("items", "pairs", "per_token", "checkpoints")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cells(path: Path, arch: str, seed: int,
               exclude_hp: set[int], only_hp: set[int]) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("format_version") != "condition-matched-eval-inventory.v1":
        raise SystemExit("unexpected inventory format")
    if payload.get("rejected"):
        raise SystemExit("inventory contains rejected checkpoint paths")
    cells = []
    for raw in payload["runs"]:
        if raw["architecture"] != arch or int(raw["seed"]) != seed:
            continue
        hp = int(raw["hp_rank"])
        if hp in exclude_hp or (only_hp and hp not in only_hp):
            continue
        cell = dict(raw)
        cell["cell_id"] = raw["run_id"]
        cell["intervention"] = raw["condition"]
        cells.append(cell)
    if not cells:
        raise SystemExit(f"no selected cells for {arch} seed={seed}")
    return sorted(cells, key=lambda x: str(x["cell_id"]))


def representatives(cells: list[dict]) -> dict[str, dict]:
    out = {}
    for cell in sorted(cells, key=lambda x: (
            str(x["intervention"]), int(x["hp_rank"]), str(x["cell_id"]))):
        out.setdefault(str(cell["intervention"]), cell)
    return out


def s3_read_verified(s3, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    expected = (obj.get("Metadata") or {}).get("sha256")
    if not expected:
        raise RuntimeError(f"missing sha256 metadata: s3://{bucket}/{key}")
    if sha256_bytes(data) != expected:
        raise RuntimeError(f"sha256 mismatch: s3://{bucket}/{key}")
    return data


def upload_once(s3, bucket: str, path: Path, key: str,
                metadata: dict[str, str]) -> str:
    digest = sha256_file(path)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        code = (getattr(exc, "response", {}) or {}).get("Error", {}).get("Code")
        if code not in ("404", "NoSuchKey", "NotFound"):
            raise
    else:
        prior = (head.get("Metadata") or {}).get("sha256")
        if prior == digest:
            return "cached"
        raise RuntimeError(f"S3 collision at s3://{bucket}/{key}: sha256 differs")
    md = dict(metadata)
    md["sha256"] = digest
    s3.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": md})
    return "uploaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--stimuli-manifest", type=Path, required=True)
    ap.add_argument("--expected-stimuli-manifest-sha256")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--source-root", type=Path,
                    help="Local representative/full results (test/offline mode).")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--exclude-hp-rank", action="append", type=int, default=[])
    ap.add_argument("--only-hp-rank", action="append", type=int, default=[])
    ap.add_argument("--bucket", default=os.environ.get(
        "REGISTRY_BUCKET", "thomas-subject-drop-artifacts"))
    ap.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL"))
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    import boto3
    import pandas as pd

    inventory_sha = sha256_file(args.cells)
    manifest_sha = sha256_file(args.stimuli_manifest)
    if (args.expected_stimuli_manifest_sha256 and
            manifest_sha != args.expected_stimuli_manifest_sha256):
        raise SystemExit("stimulus manifest SHA mismatch")
    manifest = json.loads(args.stimuli_manifest.read_text())
    if manifest.get("vetted") is not True:
        raise SystemExit("stimulus manifest is not gold-vetted")
    cells = load_cells(args.cells, args.arch, args.seed,
                       set(args.exclude_hp_rank), set(args.only_hp_rank))
    reps = representatives(cells)
    expected_conditions = {str(c["intervention"]) for c in cells}
    if set(reps) != expected_conditions:
        raise SystemExit("representative condition coverage mismatch")

    s3 = None
    if not args.no_upload or args.source_root is None:
        s3 = boto3.client("s3", endpoint_url=args.endpoint,
                          region_name=os.environ.get(
                              "AWS_DEFAULT_REGION", "us-west-1"))

    source_frames = {}
    source_bytes = {}
    for condition, rep in reps.items():
        rid = rep["cell_id"]
        source_frames[condition] = {}
        source_bytes[condition] = {}
        for table in TABLES:
            filename = f"cell_id={rid}.parquet"
            local = args.source_root / table / filename if args.source_root else None
            if local is not None and local.is_file():
                data = local.read_bytes()
            else:
                if s3 is None:
                    raise SystemExit(f"missing local representative: {local}")
                key = f"eval_results/{args.benchmark}/{table}/{filename}"
                data = s3_read_verified(s3, args.bucket, key)
            source_bytes[condition][table] = data
            source_frames[condition][table] = pd.read_parquet(io.BytesIO(data))

        side = source_frames[condition]["checkpoints"]
        expected_rows = {"items": 1152, "pairs": 576,
                         "per_token": 1152, "checkpoints": 1}
        for table, expected_count in expected_rows.items():
            frame = source_frames[condition][table]
            if len(frame) != expected_count:
                raise SystemExit(
                    f"representative {rid} {table} rows: "
                    f"{len(frame)} != {expected_count}")
            if set(map(int, frame["checkpoint_step"].unique())) != {-1}:
                raise SystemExit(
                    f"representative {rid} {table} is not checkpoint -1")
            if set(frame["cell_id"].astype(str)) != {rid}:
                raise SystemExit(
                    f"representative {rid} {table} cell_id mismatch")
        expected = {
            "cell_id": rid, "architecture": args.arch,
            "intervention": condition, "seed": args.seed,
            "checkpoint_step": -1, "tokens_seen": 0,
            "benchmark": args.benchmark,
            "inventory_sha256": inventory_sha,
            "stimuli_manifest_sha256": manifest_sha,
        }
        for field, value in expected.items():
            actual = set(side[field].dropna().tolist()) if field in side else set()
            if actual != {value}:
                raise SystemExit(
                    f"representative {rid} sidecar mismatch for {field}: {actual}")
        expected_hashes = json.dumps(
            manifest["conditions"][condition]["output_sha256"], sort_keys=True)
        actual_hashes = set(
            side["stimuli_condition_output_sha256"].dropna().astype(str))
        if actual_hashes != {expected_hashes}:
            raise SystemExit(
                f"representative {rid} condition stimulus hash mismatch")

    args.output_root.mkdir(parents=True, exist_ok=True)
    common_md = {"benchmark": args.benchmark,
                 "inventory_sha256": inventory_sha,
                 "stimuli_manifest_sha256": manifest_sha,
                 "fanout": "cpu-v1"}
    state_hashes = set()
    stimuli_ids = {}
    for cell in cells:
        condition = str(cell["intervention"])
        source = reps[condition]
        source_id = source["cell_id"]
        target_id = cell["cell_id"]
        for table in TABLES:
            out = (args.output_root / table /
                   f"cell_id={target_id}.parquet")
            out.parent.mkdir(parents=True, exist_ok=True)
            if target_id == source_id:
                out.write_bytes(source_bytes[condition][table])
            else:
                frame = source_frames[condition][table].copy(deep=True)
                frame["cell_id"] = target_id
                if table == "checkpoints":
                    frame["hp_rank"] = int(cell["hp_rank"])
                    frame["seed"] = int(cell["seed"])
                    frame["architecture"] = args.arch
                    frame["intervention"] = condition
                frame.to_parquet(out, index=False)
            if not args.no_upload:
                key = (f"eval_results/{args.benchmark}/{table}/"
                       f"cell_id={target_id}.parquet")
                upload_once(s3, args.bucket, out, key,
                            {**common_md, "run_id": target_id})
        side = source_frames[condition]["checkpoints"]
        state_hashes.update(side["model_state_sha256"].dropna().astype(str))
        stimuli_ids[condition] = str(side["stimuli_content_id"].iloc[0])

    if len(state_hashes) != 1:
        raise SystemExit(f"state hash mismatch across conditions: {state_hashes}")
    state_hash = next(iter(state_hashes))
    canonical = {}
    if s3 is not None:
        key = f"initialization/{args.arch}/seed-{args.seed}/checkpoint_-1/metadata.json"
        canonical = json.loads(
            s3.get_object(Bucket=args.bucket, Key=key)["Body"].read())
        if canonical.get("model_state_sha256") != state_hash:
            raise SystemExit("canonical initialization state hash mismatch")
    record = dict(canonical)
    record.update({
        "architecture": args.arch, "seed": args.seed,
        "benchmark": args.benchmark, "inventory_sha256": inventory_sha,
        "stimuli_manifest_sha256": manifest_sha,
        "model_state_sha256": state_hash,
        "cell_ids": [c["cell_id"] for c in cells], "n_cells": len(cells),
        "stimuli_content_ids": stimuli_ids,
        "fanout_stage": "cpu-v1",
        "s3_eval_prefix": f"s3://{args.bucket}/eval_results/{args.benchmark}/",
    })
    record_name = f"inventory-{inventory_sha}.json"
    record_path = (args.output_root / "initialization_records" / args.arch /
                   f"seed-{args.seed}" / record_name)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if not args.no_upload:
        key = (f"eval_results/{args.benchmark}/initialization_records/"
               f"{args.arch}/seed-{args.seed}/{record_name}")
        upload_once(s3, args.bucket, record_path, key, common_md)
    print(f"fanout complete: arch={args.arch} seed={args.seed} "
          f"conditions={len(reps)} cells={len(cells)}")


if __name__ == "__main__":
    main()
