#!/usr/bin/env python3
"""One-shot CLI: print per-sweep status across all 12 thomas-sweep-* Jobs.

With the consolidated 1-trial-per-pod Job design (parallelism=3,
completions=30, backoffLimit=100), the Job's `succeeded` counter is
exactly the count of real trials done. Pod state shows what's currently
running, failed, or queued.

Usage:
    python scripts/sweep_status.py
    python scripts/sweep_status.py --watch   # refresh every 30s

Doesn't query wandb (avoids needing wandb auth on the laptop). For BO
metric data, browse to the wandb sweep URL printed at the top of each
trial's pod log, or use the wandb dashboard directly.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict


def kubectl_json(*args):
    """Run kubectl with -o json and parse. Returns {} on failure."""
    try:
        out = subprocess.check_output(
            ["kubectl", *args, "-o", "json"],
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return json.loads(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError):
        return {}


def gather():
    jobs = kubectl_json("get", "jobs", "-l", "owner=thomas").get("items", [])
    pods = kubectl_json("get", "pods", "-l", "owner=thomas").get("items", [])
    if not jobs:
        # Fallback for jobs without the owner label (current generation)
        jobs = [
            j for j in kubectl_json("get", "jobs").get("items", [])
            if j["metadata"]["name"].startswith("thomas-sweep-")
        ]
        pods = [
            p for p in kubectl_json("get", "pods").get("items", [])
            if p["metadata"]["name"].startswith("thomas-sweep-")
        ]

    # Index pods by Job name
    pods_by_job = defaultdict(list)
    for p in pods:
        owner_refs = p["metadata"].get("ownerReferences", []) or []
        for o in owner_refs:
            if o.get("kind") == "Job":
                pods_by_job[o["name"]].append(p)
                break
    return jobs, pods_by_job


def status_row(job, pods):
    name = job["metadata"]["name"].replace("thomas-sweep-", "")
    spec = job.get("spec", {})
    status = job.get("status", {})

    completions = spec.get("completions", "?")
    parallelism = spec.get("parallelism", "?")
    succeeded = status.get("succeeded", 0)
    failed = status.get("failed", 0)
    active = status.get("active", 0)

    # Decompose pod states for visibility into transient failures
    by_phase = defaultdict(int)
    short_completed = 0  # pods that completed in <10 min — likely false-completes
    for p in pods:
        phase = p["status"].get("phase", "Unknown")
        by_phase[phase] += 1
        if phase == "Succeeded":
            try:
                start = p["status"].get("startTime")
                end_t = None
                for cond in p["status"].get("conditions", []):
                    if cond.get("type") == "Ready" and cond.get("status") == "False":
                        end_t = cond.get("lastTransitionTime")
                if start and end_t:
                    from datetime import datetime
                    s = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    e = datetime.fromisoformat(end_t.replace("Z", "+00:00"))
                    if (e - s).total_seconds() < 600:
                        short_completed += 1
            except Exception:
                pass

    # Conclude job state
    job_phase = "RUNNING"
    for cond in status.get("conditions", []) or []:
        if cond.get("type") == "Complete" and cond.get("status") == "True":
            job_phase = "DONE"
        elif cond.get("type") == "Failed" and cond.get("status") == "True":
            job_phase = "FAILED"

    flag = ""
    if short_completed > 0:
        flag = f" ⚠ {short_completed}-short"

    return (
        f"{name:<22}  "
        f"{succeeded:>3}/{completions:<3}  "
        f"active={active:<2} fail={failed:<3}  "
        f"pods={dict(by_phase)}  "
        f"[{job_phase}]"
        f"{flag}"
    )


def render():
    jobs, pods_by_job = gather()
    if not jobs:
        print("(no thomas-sweep-* Jobs found)")
        return
    print(f"{'SWEEP':<22}  TRIAL    POD STATE                                STATE")
    print("-" * 100)
    for j in sorted(jobs, key=lambda x: x["metadata"]["name"]):
        ps = pods_by_job.get(j["metadata"]["name"], [])
        print(status_row(j, ps))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true",
                    help="refresh every 30s until interrupted")
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")  # clear screen
                print(f"sweep_status @ {time.strftime('%H:%M:%S')}")
                render()
                time.sleep(30)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        render()


if __name__ == "__main__":
    main()
