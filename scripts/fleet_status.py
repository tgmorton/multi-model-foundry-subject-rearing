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
        limits = pod["spec"]["containers"][0].get("resources", {}).get("limits", {})
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
                         "created": pod["metadata"]["creationTimestamp"],
                         "cpu_limit_m": _parse_cpu(limits.get("cpu")),
                         "mem_limit_mi": _parse_mem(limits.get("memory"))}
    return pods


def _parse_cpu(v: str | None) -> int | None:
    if not v:
        return None
    return int(v[:-1]) if v.endswith("m") else int(float(v) * 1000)


def _parse_mem(v: str | None) -> int | None:
    if not v:
        return None
    units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024 * 1024,
             "K": 1 / 1024, "M": 1, "G": 1024}
    for suffix, mult in units.items():
        if v.endswith(suffix):
            return int(float(v[:-len(suffix)]) * mult)
    return int(int(v) / (1024 * 1024))  # plain bytes


def fetch_top() -> dict[str, tuple[int, int]]:
    """pod name -> (cpu_millicores, mem_Mi) from metrics-server."""
    out = subprocess.run(
        ["kubectl", "top", "pods", "-l", "owner=thomas", "--no-headers"],
        capture_output=True, text=True)
    usage = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            usage[parts[0]] = (_parse_cpu(parts[1]) or 0,
                               _parse_mem(parts[2]) or 0)
    return usage


def fetch_gpu(pod_names: list[str]) -> dict[str, int]:
    """pod name -> GPU SM utilization %, via nvidia-smi exec'd in-pod."""
    def query(pod):
        out = subprocess.run(
            ["kubectl", "exec", pod, "--", "nvidia-smi",
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        try:
            return pod, int(out.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None

    results = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(lambda p: _swallow(query, p), pod_names):
            if r:
                results[r[0]] = r[1]
    return results


def _swallow(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


_USE_COLOR = sys.stdout.isatty()


def util_bar(frac: float | None, width: int = 5, invert: bool = False) -> str:
    """Small colored bar. Default scale (cpu/mem vs limit): green <60%,
    yellow <85%, red ≥85% — high is dangerous. ``invert=True`` (GPU):
    green ≥85%, red <60% — LOW utilization is the problem (idle GPU,
    NRP utilization webhook)."""
    if frac is None:
        return " " * (width + 5)
    frac = min(1.0, frac)
    filled = round(frac * width)
    bar = "▰" * filled + "▱" * (width - filled)
    pct = f"{frac * 100:3.0f}%"
    if _USE_COLOR:
        level = 0 if frac < 0.60 else 1 if frac < 0.85 else 2
        if invert:
            level = 2 - level
        color = ("\033[32m", "\033[33m", "\033[31m")[level]
        return f"{color}{bar}\033[0m {pct}"
    return f"{bar} {pct}"


def classify(rid: str, rec: dict | None, pod: dict | None, since: str) -> str:
    """One of: done, live, podbad, failed, queued.

    SUPERSEDED (pre-fix truncated wave, awaiting its v2 resume) counts as
    queued. The --since gate on COMPLETE is kept as belt-and-suspenders for
    any stale COMPLETE record the migration missed.
    """
    if rec and rec.get("status") == "COMPLETE" and (rec.get("updated_at") or "") >= since:
        return "done"
    if pod:
        return "live" if pod["phase"] == "Running" else "podbad"
    if rec and rec.get("status") == "FAILED":
        return "failed"
    return "queued"


def _hb_is_fresh(rec: dict, pod: dict) -> bool:
    """True iff last_heartbeat_at is a REAL heartbeat from this pod's
    attempt. register_run_start seeds last_heartbeat_at without touching
    current_step/current_loss, so for the first ~5 min of an attempt the
    record can pair a fresh timestamp with the PREVIOUS attempt's step.
    Require the heartbeat to postdate both the pod and the attempt's
    started_at by a margin that clears the run-start seed."""
    hb = rec.get("last_heartbeat_at") or ""
    if not hb or hb < pod["created"]:
        return False
    started = rec.get("started_at") or ""
    if started:
        try:
            from datetime import timedelta
            s = datetime.fromisoformat(started.replace("Z", "+00:00"))
            h = datetime.fromisoformat(hb.replace("Z", "+00:00"))
            return h >= s + timedelta(seconds=90)
        except ValueError:
            pass
    return True


SYMBOL = {"done": "✓", "live": "◐", "podbad": "!", "failed": "✗", "queued": "·"}


def scrape_totals(pods_needing: dict[str, str]) -> dict[str, int]:
    """run_id -> total steps, scraped from pod logs.

    The trainer prints ``Current global_step: X, target: N`` at startup
    (loop.py), and the target is the full-run total even in resume mode.
    Third fallback for runs whose registry record carries neither
    train_steps (heartbeat) nor steps_completed (prior attempt).
    The line sits in the first few KB of output, so --limit-bytes keeps
    the scrape cheap."""
    import re

    def scrape(item):
        rid, pod = item
        # errors="replace": --limit-bytes can cut a multibyte UTF-8 char
        # (tqdm's bar glyphs) mid-sequence.
        out = subprocess.run(
            ["kubectl", "logs", pod, "--limit-bytes=200000"],
            capture_output=True, text=True, errors="replace")
        m = re.search(r"target: (\d+)", out.stdout)
        return (rid, int(m.group(1))) if m else None

    with ThreadPoolExecutor(max_workers=16) as ex:
        return dict(r for r in ex.map(scrape, pods_needing.items()) if r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help="COMPLETE records older than this are stale pre-fix "
                         f"runs, not credited (default {DEFAULT_SINCE})")
    ap.add_argument("--all", action="store_true",
                    help="also print one line per run (300 rows)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the in-pod nvidia-smi GPU sampling (the "
                         "slowest part; cpu/mem come from kubectl top)")
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

    # Live-pod detail with heartbeat progress. NOTE the field semantics:
    # epochs_completed / steps_completed are END-OF-RUN fields — for a
    # resumed run they hold the PREVIOUS attempt's final values until the
    # run re-completes. Live progress is current_step / current_loss from
    # the 5-min heartbeat, and only counts if the heartbeat is newer than
    # this pod (otherwise it's the prior attempt's heartbeat).
    live = [(rid, pods[rid]) for rid in sorted(pods) if rid in states]
    if live:
        # Registry-blind totals: scrape the trainer's startup "target: N"
        # line from pod logs (runs whose record has neither train_steps
        # nor a prior attempt's steps_completed).
        needing = {rid: pod["pod"] for rid, pod in live
                   if pod["phase"] == "Running"
                   and not ((records.get(rid) or {}).get("train_steps")
                            or (records.get(rid) or {}).get("steps_completed"))}
        scraped = scrape_totals(needing) if needing else {}
        top = fetch_top()
        running_pods = [pod["pod"] for _, pod in live if pod["phase"] == "Running"]
        gpu = {} if args.fast else fetch_gpu(running_pods)
        print(f"\n{'run':<48} {'state':<12} {'step':>8} {'loss':>7} "
              f"{'epoch':>6}  {'progress':<22} {'gpu':<11} {'cpu':<11} "
              f"{'mem':<11} {'hb':>6} {'att':>4}")
        for rid, pod in live:
            rec = records.get(rid) or {}
            fresh = _hb_is_fresh(rec, pod)
            step = rec.get("current_step") if fresh else None
            loss = rec.get("current_loss") if fresh else None
            if fresh:
                hb_at = rec["last_heartbeat_at"]
                age_s = (datetime.now(timezone.utc)
                         - datetime.fromisoformat(hb_at.replace("Z", "+00:00"))
                         ).total_seconds()
                hb = f"{int(age_s // 60)}m"
            else:
                hb = "await"  # no heartbeat from THIS pod yet (~5 min cadence)

            # Per-run step budget. Preference order:
            #   1. train_steps — written by the heartbeat (attempt-invariant,
            #      so no freshness gate needed).
            #   2. steps_completed — a resumed run's PRIOR attempt ran the
            #      same 30 epochs, so its end step ≈ this run's total.
            total = (rec.get("train_steps") or rec.get("steps_completed")
                     or scraped.get(rid))
            epoch = rec.get("current_epoch") if fresh else None
            if epoch is None and step is not None and total:
                # Uniform steps/epoch; derive when the heartbeat predates
                # the current_epoch field.
                epoch = min(EPOCHS, int(step / (total / EPOCHS)) + 1)
            ep = f"{epoch}/{EPOCHS}" if epoch is not None else "-"
            if step is not None and total:
                frac = min(1.0, step / total)
                bar = "█" * round(frac * 16) + "░" * (16 - round(frac * 16))
                prog = f"{bar} {frac * 100:3.0f}%"
            else:
                prog = ""
            state = pod["phase"] + (f"/{pod['reason']}" if pod["reason"] else "")
            cpu_m, mem_mi = top.get(pod["pod"], (None, None))
            cpu_bar = util_bar(cpu_m / pod["cpu_limit_m"]
                               if cpu_m is not None and pod["cpu_limit_m"] else None)
            mem_bar = util_bar(mem_mi / pod["mem_limit_mi"]
                               if mem_mi is not None and pod["mem_limit_mi"] else None)
            gpu_pct = gpu.get(pod["pod"])
            gpu_bar = util_bar(gpu_pct / 100 if gpu_pct is not None else None,
                               invert=True)
            print(f"{rid:<48} {state[:12]:<12} "
                  f"{step if step is not None else '-':>8} "
                  f"{f'{loss:.2f}' if loss is not None else '-':>7} "
                  f"{ep:>6}  {prog:<22} {gpu_bar}  {cpu_bar}  {mem_bar}  {hb:>6} "
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
