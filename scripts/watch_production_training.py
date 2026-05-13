#!/usr/bin/env python3
"""Watch the 20 production training Jobs and re-launch any that fail entirely.

Within a Job, ``backoffLimit=100`` absorbs transient pod failures. But if
a Job's backoff is exhausted (every retry failed for 100 attempts) the
Job goes to status=Failed and stops scheduling pods. This watcher polls
for that state and re-applies the YAML so the Job restarts from scratch.

Also reports the per-cell completion progress so you can see how far the
matrix is from done.

Usage:
    python scripts/watch_production_training.py --lang en              # one pass
    python scripts/watch_production_training.py --lang en --loop 600   # every 10m
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from launch_production_training import (  # noqa: E402
    ARCH_SETTINGS, INTERVENTIONS, _job_yaml, _apply_yaml,
)


def _list_jobs(lang: str) -> list[dict]:
    """All Job objects with stage=train-prod and lang=<lang>."""
    proc = subprocess.run(
        ["kubectl", "get", "jobs", "-n", "lemn-lab",
         "-l", f"stage=train-prod,lang={lang}",
         "-o", "json"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return []
    return json.loads(proc.stdout).get("items", [])


def _job_state(job: dict) -> str:
    """Map a Job's status block to one of: complete, failed, running."""
    conds = job.get("status", {}).get("conditions", []) or []
    for c in conds:
        if c.get("type") == "Complete" and c.get("status") == "True":
            return "complete"
        if c.get("type") == "Failed"   and c.get("status") == "True":
            return "failed"
    return "running"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "es"], required=True)
    ap.add_argument("--loop", type=int, default=0,
                    help="Seconds between polls. 0 = single pass.")
    ap.add_argument("--no-relaunch", action="store_true",
                    help="Report only; do not re-apply failed Jobs.")
    args = ap.parse_args()

    while True:
        jobs = _list_jobs(args.lang)
        if not jobs:
            print(f"No prod-training Jobs found for lang={args.lang}.")
            return

        by_name = {j["metadata"]["name"]: j for j in jobs}
        complete = []
        failed = []
        running = []
        total_succeeded = 0
        total_completions = 0

        for name, j in sorted(by_name.items()):
            state = _job_state(j)
            succ = int(j.get("status", {}).get("succeeded") or 0)
            comp = int(j.get("spec", {}).get("completions") or 10)
            total_succeeded += succ
            total_completions += comp
            line = f"  [{state:>8}] {name:60s} {succ:>3}/{comp}"
            if state == "complete":
                complete.append(name); print(line)
            elif state == "failed":
                failed.append(name); print(line)
            else:
                running.append(name); print(line)

        print(f"\nSummary: {len(complete)} complete, {len(running)} running, "
              f"{len(failed)} failed | runs: {total_succeeded}/{total_completions}")

        # Relaunch failed Jobs by deleting + re-applying.
        if failed and not args.no_relaunch:
            for name in failed:
                # Extract (arch, intervention) from labels — the launcher
                # labels every Job with arch + intervention.
                meta = by_name[name].get("metadata", {})
                labels = meta.get("labels", {})
                arch = labels.get("arch", "").replace("-", "_")
                # Some archs have underscores in their proper names (gpt2_small)
                # but k8s labels use hyphens; normalize back.
                if arch == "gpt2_small":
                    arch = "gpt2_small"
                elif arch == "gpt2_medium":
                    arch = "gpt2_medium"
                elif arch == "gpt2_large":
                    arch = "gpt2_large"
                elif arch == "bert_large":
                    arch = "bert_large"
                elif arch == "mamba_370m":
                    arch = "mamba_370m"
                # lstm stays "lstm"
                intervention = labels.get("intervention", "").replace("-", "_")
                if arch not in ARCH_SETTINGS:
                    print(f"  WARN cannot parse arch from labels of {name}; skipping relaunch")
                    continue

                phys, ram, cpu = ARCH_SETTINGS[arch]
                print(f"\n--- relaunching {name} ({arch} × {intervention}) ---")
                subprocess.run(
                    ["kubectl", "delete", "job", name, "-n", "lemn-lab",
                     "--ignore-not-found"],
                    check=False,
                )
                # Brief pause so the GC ages out before we re-apply.
                time.sleep(2)
                yml = _job_yaml(arch, args.lang, intervention, phys, ram, cpu)
                _apply_yaml(yml)

        if args.loop <= 0:
            return
        print(f"\nSleeping {args.loop}s ...\n")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
