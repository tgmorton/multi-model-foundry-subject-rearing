#!/usr/bin/env python3
"""Fleet watcher: prints one ALERT line per NEW problem, silence otherwise.

Each check targets a failure mode that actually bit us (Sept 2026):
  dead_job        Job hit its backoffLimit (iv-conditioned burnout 9/28,
                  82-job skip-sentinel burnout 9/18)
  backoff         active Job has used >=50% / >=80% of backoffLimit —
                  caught BEFORE it dies
  bad_node        a node not yet in BAD_NODES accumulating failed pods
                  (gpu-18 flytrap 9/18, hcc-chase-shor 9/28)
  image_pull      ImagePullBackOff / ErrImagePull (credential outage 9/18)
  hung_pod        Running >30 min at ~0 CPU (silent eval hangs 9/18)
  api_down        kubectl failing repeatedly

Dedupe state persists in /tmp/fleet_watch_seen.json, so re-arming the
watcher does not re-announce problems already reported.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

SEEN = Path("/tmp/fleet_watch_seen.json")
LOGSTATE: dict = {}   # pod -> last log line, across passes
REPO = Path(__file__).resolve().parent.parent


def kjson(*args):
    r = subprocess.run(["kubectl", *args, "-o", "json"], capture_output=True,
                       text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200])
    return json.loads(r.stdout)


def known_bad_nodes() -> set:
    t = (REPO / "scripts" / "wave2_launcher.py").read_text()
    blk = t[t.index("BAD_NODES = ["):]
    blk = blk[:blk.index("]")]
    return set(re.findall(r'"([^"]+)"', blk))


def age_min(ts: str) -> float:
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - t).total_seconds() / 60


def check(seen: set, do_top: bool) -> list:
    out = []
    def alert(key, msg):
        if key not in seen:
            seen.add(key); out.append(msg)

    jobs = kjson("get", "jobs", "-l", "owner=thomas")["items"]
    for j in jobs:
        name = j["metadata"]["name"]
        conds = {c["type"]: c["status"] for c in j["status"].get("conditions", [])}
        failed = j["status"].get("failed", 0) or 0
        limit = j["spec"].get("backoffLimit", 6)
        if conds.get("Failed") == "True":
            alert(f"dead:{name}", f"ALERT dead_job {name} hit backoffLimit "
                                  f"({failed}/{limit}) — needs recreate")
        elif conds.get("Complete") != "True" and not j["spec"].get("suspend"):
            frac = failed / max(limit, 1)
            for lvl in (0.8, 0.5):
                if frac >= lvl:
                    alert(f"backoff{lvl}:{name}",
                          f"ALERT backoff {name} at {failed}/{limit} "
                          f"({frac:.0%}) — raise backoffLimit or find cause")
                    break

    pods = kjson("get", "pods", "-l", "owner=thomas")["items"]
    bad = known_bad_nodes()
    node_fail = {}
    for p in pods:
        st, node = p["status"], p["spec"].get("nodeName")
        if st.get("phase") == "Failed" and node:
            node_fail[node] = node_fail.get(node, 0) + 1
        for cs in st.get("containerStatuses") or []:
            w = (cs.get("state") or {}).get("waiting") or {}
            if w.get("reason") in ("ImagePullBackOff", "ErrImagePull"):
                alert(f"pull:{p['metadata']['name']}",
                      f"ALERT image_pull {p['metadata']['name']} "
                      f"({w.get('reason')}) — check registry credential")
    for node, n in node_fail.items():
        if n >= 3 and node not in bad:
            alert(f"node:{node}", f"ALERT bad_node {node} has {n} failed pods "
                                  f"of ours and is NOT excluded — add to BAD_NODES")

    if do_top:
        r = subprocess.run(["kubectl", "top", "pods", "-l", "owner=thomas",
                            "--no-headers"], capture_output=True, text=True,
                           timeout=120)
        cpu = {}
        for ln in r.stdout.splitlines():
            f = ln.split()
            if len(f) >= 2 and f[1].endswith("m"):
                cpu[f[0]] = int(f[1][:-1])
        for p in pods:
            nm = p["metadata"]["name"]; st = p["status"]
            if st.get("phase") != "Running" or nm not in cpu:
                continue
            started = st.get("startTime")
            if not (started and age_min(started) > 30 and cpu[nm] < 20):
                continue
            # Low CPU alone is ambiguous: an I/O-bound pod (CephFS reads)
            # idles legitimately. Hung = low CPU AND no new log output since
            # the previous pass. (Sampled fp16 dry-run 9/28 was a false
            # positive on CPU alone — it was logging steady progress.)
            lr = subprocess.run(["kubectl", "logs", nm, "--tail=1"],
                                capture_output=True, text=True, timeout=60)
            last = lr.stdout.strip()[-200:]
            prev = LOGSTATE.get(nm)
            LOGSTATE[nm] = last
            if prev is not None and prev == last:
                alert(f"hung:{nm}", f"ALERT hung_pod {nm} Running "
                                    f"{age_min(started):.0f}min at {cpu[nm]}m CPU with NO new "
                                    f"log output since last pass — stalled")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--reset", action="store_true", help="forget reported alerts")
    a = ap.parse_args()
    seen = set() if a.reset or not SEEN.exists() else set(json.loads(SEEN.read_text()))
    fails, loop = 0, 0
    while True:
        try:
            for msg in check(seen, do_top=True):
                print(msg, flush=True)
            SEEN.write_text(json.dumps(sorted(seen)))
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            if fails == 3:
                print(f"ALERT api_down kubectl failing 3x in a row: {e}", flush=True)
        if a.once:
            break
        loop += 1
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
