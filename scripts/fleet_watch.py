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
    sys.path.insert(0, str(REPO / "scripts"))
    from bad_nodes import load
    return set(load())


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
            # Stuck in a failure loop: nothing running, work left, and it keeps
            # failing. Budget % alone missed this (iv-conditioned sat at 9/40,
            # 0 active, every retry rejected by one node — 2026-09-28).
            if (j["status"].get("active") or 0) == 0 and failed >= 3:
                alert(f"stalled:{name}:{failed // 3}",
                      f"ALERT stalled_job {name} has 0 active pods and {failed} "
                      f"failures — stuck in a retry loop, check the node")
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
    pull_fail = []
    unsched = []
    for p in pods:
        st, node = p["status"], p["spec"].get("nodeName")
        # Never scheduled: Pending with no node. Jobs count these as "active",
        # so the stalled-job check can't see them (iv fill pods sat 93 min
        # unschedulable on a contended pool, 2026-09-29).
        created = p["metadata"].get("creationTimestamp")
        if st.get("phase") == "Pending" and not node and created \
                and age_min(created) > 30:
            unsched.append((p["metadata"]["name"], age_min(created)))
        # OOM kills: one line per pod, so the OOM rate of a wave is visible.
        for cs in st.get("containerStatuses") or []:
            for state in (cs.get("state") or {}, cs.get("lastState") or {}):
                t = state.get("terminated") or {}
                if t.get("reason") == "OOMKilled":
                    alert(f"oom:{p['metadata']['name']}",
                          f"ALERT oom {p['metadata']['name']} OOMKilled on {node} "
                          f"(limit {p['spec']['containers'][0]['resources'].get('limits', {}).get('memory')})")
        if st.get("phase") == "Failed" and node:
            node_fail.setdefault(node, []).append(
                (p["metadata"].get("labels") or {}).get("job-name", "?"))
        for cs in st.get("containerStatuses") or []:
            w = (cs.get("state") or {}).get("waiting") or {}
            if w.get("reason") in ("ImagePullBackOff", "ErrImagePull"):
                pull_fail.append((p["metadata"]["name"], node))
    # A wave bigger than the GPU pool leaves many pods Pending by design; one
    # aggregate line (re-alerting as the count moves by 25) instead of one per
    # pod. A handful of stuck pods still alert individually — that's the
    # narrow-pool case this check was written for (2026-09-29).
    if len(unsched) > 3:
        oldest = max(a for _, a in unsched)
        alert(f"unsched-many:{len(unsched) // 25}",
              f"ALERT unschedulable {len(unsched)} pods Pending >30min with no node "
              f"(oldest {oldest:.0f}min) — GPU pool full; widen it or wait")
    else:
        for name, a in unsched:
            alert(f"unsched:{name}", f"ALERT unschedulable {name} Pending {a:.0f}min "
                                     f"with no node — pool too narrow or full")
    # One failed pull is almost always a cold node timing out on the
    # 5-10 GB image and retrying. A credential/registry outage shows as
    # MANY pods at once (45 on 2026-09-18) — alert only on that.
    if len(pull_fail) >= 3:
        nodes = sorted({n for _, n in pull_fail if n})
        alert(f"pull:systemic:{len(pull_fail) // 5}",
              f"ALERT image_pull {len(pull_fail)} pods failing to pull across "
              f"{len(nodes)} node(s) — likely credential/registry outage")
    for node, jobs_on in node_fail.items():
        n = len(jobs_on)
        # A job that fails everywhere (its own bug, e.g. CUDA OOM from too
        # big a batch) isn't a bad node: require failures from >= 2 distinct
        # jobs (false positive on ry-gpu-01, 2026-10-05).
        if n < 3 or len(set(jobs_on)) < 2:
            continue
        if node not in bad:
            alert(f"node:{node}", f"ALERT bad_node {node} has {n} failed pods "
                                  f"of ours and is NOT excluded — add to configs/bad_nodes.txt")
        else:
            # Listed, yet still receiving pods: the job predates the exclusion
            # (pod templates are immutable). Editing the list won't help.
            alert(f"listed:{node}", f"ALERT bad_node {node} is excluded but still "
                                    f"got {n} failed pods — a job predates the list; recreate it")

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
