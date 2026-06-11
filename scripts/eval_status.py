#!/usr/bin/env python3
"""Eval-fleet status — join the S3 run registry's eval fields with live
eval-pod state, fleet_status.py-style.

Grid: (arch × condition) for one slot (default h0-s42, the all-cells
wave). Each cell shows the run's null_subj_v2 eval state, plus whether
its result parquets are pullable from S3.

Legend:  ✓ eval COMPLETE   ☁ + parquets on S3   ◐ eval pod live
         ✗ eval FAILED     · not started

Beyond the grid, any other run_ids with eval records (e.g. s137
validation siblings) are listed separately, with the registry's
final-checkpoint metric summary.

Usage:
    AWS_PROFILE=nrp python3 scripts/eval_status.py
    AWS_PROFILE=nrp python3 scripts/eval_status.py --slot h0-s137
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

ARCHS = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large",
         "lstm", "mamba_370m"]
CONDITIONS = ["baseline", "remove_expletive_sentences", "impoverish_case",
              "lemmatize_verbs", "enrich_verbal_morphology"]
COND_SHORT = {"baseline": "base", "remove_expletive_sentences": "rmexp",
              "impoverish_case": "impov", "lemmatize_verbs": "lemma",
              "enrich_verbal_morphology": "enrich"}
LANG = "en"
BENCHMARK = "null_subj_v2"
BUCKET = os.environ.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts")


def _s3_client():
    import boto3
    endpoint = os.environ.get("AWS_ENDPOINT_URL",
                              "https://s3-west.nrp-nautilus.io")
    return boto3.client("s3", endpoint_url=endpoint)


def _fetch_record(s3, arch: str, cond: str, run_id: str):
    key = f"run_registry/by_run/{arch}/{LANG}/{cond}/{run_id}.json"
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return run_id, json.loads(body)
    except Exception:
        return run_id, None


def _s3_result_cells(s3) -> set[str]:
    """run_ids whose pairs parquet is on S3 (laptop-pullable)."""
    cells: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"eval_results/{BENCHMARK}/pairs/"
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name.startswith("cell_id=") and name.endswith(".parquet"):
                cells.add(name[len("cell_id="):-len(".parquet")])
    return cells


def _live_eval_pods() -> dict[str, str]:
    """run_id → pod phase for pods of stage=eval-cell jobs."""
    try:
        jobs = json.loads(subprocess.run(
            ["kubectl", "get", "jobs", "-l", "owner=thomas,stage=eval-cell",
             "-o", "json"], capture_output=True, text=True, timeout=60,
        ).stdout or '{"items": []}')
        pods = json.loads(subprocess.run(
            ["kubectl", "get", "pods", "-l", "owner=thomas,stage=eval-cell",
             "-o", "json"], capture_output=True, text=True, timeout=60,
        ).stdout or '{"items": []}')
    except Exception:
        return {}

    run_ids_by_job: dict[str, list[str]] = {}
    for j in jobs.get("items", []):
        env = j["spec"]["template"]["spec"]["containers"][0].get("env", [])
        for e in env:
            if e.get("name") == "RUN_IDS_JSON":
                run_ids_by_job[j["metadata"]["name"]] = json.loads(e["value"])

    live: dict[str, str] = {}
    for p in pods.get("items", []):
        phase = p["status"].get("phase", "?")
        if phase in ("Succeeded", "Failed"):
            continue
        job = p["metadata"].get("labels", {}).get("job-name")
        idx = p["metadata"].get("annotations", {}).get(
            "batch.kubernetes.io/job-completion-index")
        ids = run_ids_by_job.get(job)
        if ids is not None and idx is not None and int(idx) < len(ids):
            live[ids[int(idx)]] = phase
    return live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="h0-s42")
    args = ap.parse_args()

    s3 = _s3_client()
    grid_ids = {(a, c): f"{a}-{LANG}-{c}-{args.slot}"
                for a in ARCHS for c in CONDITIONS}

    with ThreadPoolExecutor(max_workers=16) as ex:
        recs = dict(ex.map(
            lambda ac: _fetch_record(s3, ac[0], ac[1], grid_ids[ac]),
            grid_ids))
    on_s3 = _s3_result_cells(s3)
    live = _live_eval_pods()

    n_done = n_live = n_failed = 0
    print(f"=== eval status ({BENCHMARK}, slot {args.slot}) ===\n")
    print(f"{'cell':<28}" + "".join(f"{COND_SHORT[c]:>8}" for c in CONDITIONS))
    for arch in ARCHS:
        row = [f"{arch:<28}"]
        for cond in CONDITIONS:
            rid = grid_ids[(arch, cond)]
            rec = recs.get(rid)
            bench = ((rec or {}).get("eval_benchmarks") or {}).get(BENCHMARK, {})
            status = bench.get("status")
            if status == "COMPLETE":
                mark = "✓☁" if rid in on_s3 else "✓"
                n_done += 1
            elif status == "FAILED":
                mark, n_failed = "✗", n_failed + 1
            elif rid in live:
                mark, n_live = "◐", n_live + 1
            elif status == "RUNNING":
                mark = "◐?"   # registry says running but no pod found
            else:
                mark = "·"
            row.append(f"{mark:>8}")
        print("".join(row))
    print(f"\n{n_done} complete · {n_live} live · {n_failed} failed · "
          f"{30 - n_done - n_live - n_failed} pending   "
          f"(✓☁ = parquets pullable from S3)")

    # Off-grid runs with eval state (validation siblings etc.)
    extras = sorted((set(live) | on_s3) - set(grid_ids.values()))
    if extras:
        print("\nother evaluated runs:")
        for rid in extras:
            tag = "☁" if rid in on_s3 else live.get(rid, "?")
            print(f"  {rid:<48} {tag}")

    # Metric peek for completed grid cells.
    done = [(rid, recs[rid]) for rid in grid_ids.values()
            if recs.get(rid) and ((recs[rid].get("eval_benchmarks") or {})
                                  .get(BENCHMARK, {}).get("status")) == "COMPLETE"]
    if done:
        print("\nfinal-checkpoint summary (pooled overt-pref / mean Δlogprob / ckpts):")
        for rid, rec in sorted(done):
            ms = (rec["eval_benchmarks"][BENCHMARK].get("metric_summary")
                  or {})
            print(f"  {rid:<48} {ms.get('final_overt_pref', float('nan')):.3f}"
                  f"  {ms.get('final_mean_logprob_diff', float('nan')):+.3f}"
                  f"  {ms.get('n_checkpoints', '?')}")


if __name__ == "__main__":
    main()
