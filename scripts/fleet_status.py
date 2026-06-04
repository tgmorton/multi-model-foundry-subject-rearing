#!/usr/bin/env python3
"""Fleet status — join live kubectl pod state with the S3 run registry.

One-shot dashboard for the English-wave recovery: a 30-cell × 10-slot
grid (every (arch × condition) cell against its h0-first slot order),
plus a live-pod detail table with epoch/step progress from the registry
heartbeat.

The registry still holds COMPLETE records from the PRE-FIX truncated
wave (runs that trained 30 epochs but never checkpointed their back
half). Those do NOT count as done — only records updated after the
corrected schedule landed (--since, default 2026-06-03) are credited.

Usage:
    AWS_PROFILE=nrp python3 scripts/fleet_status.py            # grid + live pods
    AWS_PROFILE=nrp python3 scripts/fleet_status.py --all      # + full 300-row list
    AWS_PROFILE=nrp python3 scripts/fleet_status.py --since 2026-06-03T00:00:00Z

Legend:  ✓ done   ◐ pod live   ! pod not-running (Init/Pending/failed)
         ✗ FAILED in registry, no pod   · queued
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ARCHS = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm", "mamba_370m"]
CONDITIONS = ["baseline", "remove_expletive_sentences", "impoverish_case",
              "lemmatize_verbs", "enrich_verbal_morphology"]
COND_SHORT = {"baseline": "base", "remove_expletive_sentences": "rmexp",
              "impoverish_case": "impov", "lemmatize_verbs": "lemma",
              "enrich_verbal_morphology": "enrich"}
SEEDS = [42, 137]
LANG = "en"
EPOCHS = 30
# Slot order matches the dispatch (h0-first, wide before deep).
SLOTS = [(h, s) for h in range(5) for s in range(2)]

BUCKET = os.environ.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts")
DEFAULT_SINCE = "2026-06-03T00:00:00Z"  # corrected-schedule code landed


def _run_id(arch: str, cond: str, hp: int, seed_idx: int) -> str:
    return f"{arch}-{LANG}-{cond}-h{hp}-s{SEEDS[seed_idx]}"


def _s3_client():
    import boto3
    from botocore.exceptions import NoCredentialsError

    endpoint = os.environ.get("AWS_ENDPOINT_URL", "https://s3-west.nrp-nautilus.io")
    client = boto3.client("s3", endpoint_url=endpoint)
    try:  # cheap credential probe
        client.list_objects_v2(Bucket=BUCKET, MaxKeys=1)
        return client
    except NoCredentialsError:
        # Fall back to the documented local profile.
        import boto3.session
        return boto3.session.Session(profile_name="nrp").client(
            "s3", endpoint_url=endpoint)


def fetch_registry() -> dict[str, dict]:
    """run_id -> registry record, for every run under by_run/."""
    s3 = _s3_client()
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="run_registry/by_run/"):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".json")]

    wanted = {_run_id(a, c, h, si) for a in ARCHS for c in CONDITIONS for h, si in SLOTS}

    def get(key):
        rid = key.rsplit("/", 1)[-1][:-len(".json")]
        if rid not in wanted:
            return None
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return rid, json.loads(body)

    records = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        for res in ex.map(get, keys):
            if res:
                records[res[0]] = res[1]
    return records


def fetch_pods() -> dict[str, dict]:
    """run_id -> {phase, pod, reason} for every owner=thomas training pod."""
    out = subprocess.run(
        ["kubectl", "get", "pods", "-l", "owner=thomas", "-o", "json"],
        capture_output=True, text=True)
    if out.returncode != 0:
        print(f"[warn] kubectl failed: {out.stderr.strip()}", file=sys.stderr)
        return {}
    pods = {}
    for pod in json.loads(out.stdout).get("items", []):
        env = {e["name"]: e.get("value") for c in pod["spec"]["containers"]
               for e in c.get("env", []) if "value" in e}
        if "ARCH" not in env or "INTERVENTION" not in env:
            continue  # not a training pod (e.g. data-access)
        idx = pod["metadata"].get("annotations", {}).get(
            "batch.kubernetes.io/job-completion-index")
        if idx is None:
            idx = pod["metadata"].get("labels", {}).get(
                "batch.kubernetes.io/job-completion-index")
        if idx is None:
            continue
        slot_map = json.loads(env.get("SLOT_MAP_JSON", "null"))
        if slot_map:
            hp, seed_idx = slot_map[int(idx)]
        else:  # legacy full-grid divmod(idx, 2)
            hp, seed_idx = divmod(int(idx), 2)
        rid = _run_id(env["ARCH"], env["INTERVENTION"], hp, seed_idx)
        phase = pod["status"].get("phase", "?")
        reason = ""
        for cs in pod["status"].get("containerStatuses", []):
            term = cs.get("state", {}).get("terminated")
            wait = cs.get("state", {}).get("waiting")
            if term:
                reason = term.get("reason", "")
            elif wait:
                reason = wait.get("reason", "")
        # Prefer the newest pod per run (retries leave terminated pods around).
        prev = pods.get(rid)
        if prev is None or pod["metadata"]["creationTimestamp"] > prev["created"]:
            pods[rid] = {"phase": phase, "reason": reason,
                         "pod": pod["metadata"]["name"],
                         "created": pod["metadata"]["creationTimestamp"]}
    return pods


def classify(rid: str, rec: dict | None, pod: dict | None, since: str) -> str:
    """One of: done, live, podbad, failed, queued."""
    if rec and rec.get("status") == "COMPLETE" and (rec.get("updated_at") or "") >= since:
        return "done"
    if pod:
        return "live" if pod["phase"] == "Running" else "podbad"
    if rec and rec.get("status") == "FAILED":
        return "failed"
    return "queued"


SYMBOL = {"done": "✓", "live": "◐", "podbad": "!", "failed": "✗", "queued": "·"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help="COMPLETE records older than this are stale pre-fix "
                         f"runs, not credited (default {DEFAULT_SINCE})")
    ap.add_argument("--all", action="store_true",
                    help="also print one line per run (300 rows)")
    args = ap.parse_args()

    records = fetch_registry()
    pods = fetch_pods()

    states = {}
    for a in ARCHS:
        for c in CONDITIONS:
            for h, si in SLOTS:
                rid = _run_id(a, c, h, si)
                states[rid] = classify(rid, records.get(rid), pods.get(rid), args.since)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    n = lambda st: sum(1 for v in states.values() if v == st)
    print(f"=== fleet status @ {now} ===")
    print(f"runs: {n('done')} done · {n('live')} live · {n('podbad')} pod-not-running "
          f"· {n('failed')} failed · {n('queued')} queued   (of {len(states)})")

    # Wide (h0) layer — the analysis gate.
    h0 = [_run_id(a, c, 0, si) for a in ARCHS for c in CONDITIONS for si in (0, 1)]
    print(f"wide layer (h0): {sum(states[r] == 'done' for r in h0)}/{len(h0)} done, "
          f"{sum(states[r] == 'live' for r in h0)} live")
    print()

    # Grid: 30 cells × 10 slots in dispatch order.
    hdr = " ".join(f"h{h}s{SEEDS[si]:<3}" for h, si in SLOTS)
    print(f"{'cell':<22} {hdr}")
    for a in ARCHS:
        for c in CONDITIONS:
            row = "     ".join(SYMBOL[states[_run_id(a, c, h, si)]] for h, si in SLOTS)
            print(f"{a + '/' + COND_SHORT[c]:<22} {row}")
    print(f"\nlegend: ✓ done  ◐ pod live  ! pod not-running  ✗ failed  · queued")

    # Live-pod detail with heartbeat progress.
    live = [(rid, pods[rid]) for rid in sorted(pods) if rid in states]
    if live:
        print(f"\n{'run':<48} {'pod state':<18} {'epoch':>7} {'steps':>9} {'ckpts':>6} {'att':>4}")
        for rid, pod in live:
            rec = records.get(rid) or {}
            ep = rec.get("epochs_completed")
            ep = f"{ep}/{EPOCHS}" if ep is not None else "-"
            state = pod["phase"] + (f"/{pod['reason']}" if pod["reason"] else "")
            print(f"{rid:<48} {state:<18} {ep:>7} "
                  f"{rec.get('steps_completed') if rec.get('steps_completed') is not None else '-':>9} "
                  f"{rec.get('checkpoint_count') if rec.get('checkpoint_count') is not None else '-':>6} "
                  f"{rec.get('attempt_count') if rec.get('attempt_count') is not None else '-':>4}")

    if args.all:
        print()
        for a in ARCHS:
            for c in CONDITIONS:
                for h, si in SLOTS:
                    rid = _run_id(a, c, h, si)
                    rec = records.get(rid) or {}
                    print(f"{SYMBOL[states[rid]]} {rid:<48} "
                          f"reg={rec.get('status', '-'):<10} "
                          f"updated={rec.get('updated_at', '-')}")


if __name__ == "__main__":
    main()
