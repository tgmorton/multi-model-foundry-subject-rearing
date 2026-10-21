#!/usr/bin/env python3
"""Summarize the opportunistic-tier probes (scripts/opp_probe.py).

    python scripts/opp_probe_report.py --tag 1020

Per probe (GPU key x priority class), one row per pod attempt: how long it
waited to be scheduled, how long it ran, how it ended (preempted / running /
deadline), the GPU and throughput; then the finished seed replicates with
their final held-out loss and accuracy. Sources: S3 heartbeats (they survive
preemption) and kubectl pod records (creation, start, DisruptionTarget).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import boto3

NS = "null-subjects"


def ts(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def hms(sec: float | None) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bucket", default="thomas-subject-drop-artifacts")
    ap.add_argument("--profile", default="nrp")
    ap.add_argument("--endpoint", default="https://s3-west.nrp-nautilus.io")
    a = ap.parse_args(argv)
    s3 = boto3.Session(profile_name=a.profile).client("s3", endpoint_url=a.endpoint)
    prefix = f"opp_probe/{a.tag}/"
    beats, reps = defaultdict(dict), defaultdict(list)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            parts = o["Key"][len(prefix):].split("/")
            body = json.loads(s3.get_object(Bucket=a.bucket, Key=o["Key"])["Body"].read())
            if len(parts) == 3 and parts[1] == "pods":
                beats[parts[0]][body["pod"]] = body
            elif len(parts) == 3 and parts[1] == "replicates":
                reps[parts[0]].append(body)

    out = subprocess.run(["kubectl", "get", "pods", "-n", NS, "-l", "stage=opp-probe", "-o", "json"],
                         capture_output=True, text=True)
    pods = json.loads(out.stdout)["items"] if out.returncode == 0 else []
    k8s = defaultdict(dict)
    for p in pods:
        lab = p["metadata"]["labels"]
        pid = f"{lab.get('gpu')}-{lab.get('tier')}"
        st_ = p["status"]
        dis = next((c for c in st_.get("conditions", []) if c["type"] == "DisruptionTarget"), None)
        ctr = (st_.get("containerStatuses") or [{}])[0].get("state", {})
        term = ctr.get("terminated") or {}
        k8s[pid][p["metadata"]["name"]] = {
            "created": ts(p["metadata"]["creationTimestamp"]),
            "started": ts((ctr.get("running") or {}).get("startedAt") or term.get("startedAt")),
            "finished": ts(term.get("finishedAt")), "phase": st_.get("phase"),
            "node": p["spec"].get("nodeName"),
            "disruption": f"{dis['reason']}" if dis and dis.get("status") == "True" else None,
        }

    now = time.time()
    probes = sorted(set(beats) | set(k8s))
    for pid in probes:
        print(f"\n=== {pid}")
        names = sorted(set(beats[pid]) | set(k8s[pid]),
                       key=lambda n: (k8s[pid].get(n, {}).get("created") or beats[pid].get(n, {}).get("pod_started") or 0))
        print(f"  {'pod':40s} {'waited':>7s} {'ran':>7s}  {'ended':30s} {'GPU':24s} {'tok/s':>7s} step")
        for n in names:
            k, b = k8s[pid].get(n, {}), beats[pid].get(n, {})
            started = k.get("started") or b.get("pod_started")
            waited = (started - k["created"]) if started and k.get("created") else (
                now - k["created"] if k.get("created") and not started else None)
            last = k.get("finished") or b.get("heartbeat")
            alive = k.get("phase") in ("Running", "Pending") and not k.get("finished")
            ran = ((now if alive else last) - started) if started and last else None
            if k.get("phase") == "Pending" and not started:
                ended = "PENDING (not scheduled yet)"
            elif alive:
                ended = f"running ({b.get('status', '?')})"
            elif k.get("disruption"):
                ended = f"PREEMPTED ({k['disruption']})"
            elif b.get("status") == "deadline":
                ended = "deadline (clean exit)"
            elif not k:
                ended = "gone (pod deleted; last heartbeat)"
            else:
                ended = f"{k.get('phase')}"
            tps = b.get("tokens_per_s")
            print(f"  {n[-40:]:40s} {hms(waited):>7s} {hms(ran):>7s}  {ended[:30]:30s} "
                  f"{str(b.get('gpu') or '-')[:24]:24s} {(f'{tps:,.0f}' if tps else '-'):>7s} {b.get('step', '-')}")
        rs = sorted(reps[pid], key=lambda r: r["seed"])
        if rs:
            finals = [r["evals"][-1] for r in rs if r.get("evals")]
            print(f"  replicates done: {len(rs)} (seeds {', '.join(str(r['seed']) for r in rs)}); "
                  f"mean {st.mean(r['seconds'] for r in rs) / 3600:.2f} h each")
            for r, f in zip(rs, finals):
                print(f"    seed {r['seed']}: step {f['step']} held-out loss {f['held_out_loss']:.3f} "
                      f"acc {f['accuracy']:.3f} on {r.get('gpu')}")
    by_seed = {}
    for v in reps.values():          # the same seed on two probes is the same replicate
        for r in v:
            if r.get("evals"):
                by_seed.setdefault(r["seed"], r["evals"][-1]["held_out_loss"])
    allf = list(by_seed.values())
    if len(allf) >= 2:
        print(f"\nseed noise of final held-out loss over {len(allf)} distinct seeds: "
              f"mean {st.mean(allf):.4f}, sd {st.stdev(allf):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
