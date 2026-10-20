#!/usr/bin/env python3
"""NRP scheduling and availability, from our account's point of view.

Our user cannot list pods cluster-wide, so per-node GPU allocation comes from
NRP's public Prometheus (kube-state-metrics + DCGM, no auth), and node labels,
taints and allocatable come from ``kubectl get nodes``.

    python scripts/nrp_avail.py gpus [--all] [--json]
        Free GPUs per product right now, open nodes vs. nodes behind hard
        taints we must not tolerate, plus cluster-wide pending GPU demand.
    python scripts/nrp_avail.py fit FILE.yaml | NS/job/NAME | NS/pod/NAME
        Which nodes a pod spec can land on (affinity, taints, bad nodes, GPU
        key) and how many have room now; what blocks the rest.
    python scripts/nrp_avail.py ours
        Our pods in lemn-lab + null-subjects: phase, node, GPU utilization
        (DCGM, 15-min mean), CPU/RAM vs request, and NRP's utilization bands
        (GPU > 40%, CPU 20-200%, RAM 20-150%; <=1 CPU and <=2 GB exempt;
        more than 4 violators blocks new pods).
    python scripts/nrp_avail.py quota
        ResourceQuotas in both namespaces.

Prometheus numbers lag the cluster by up to a minute or so.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

PROM = "https://prometheus.nrp-nautilus.io/api/v1/query"
NAMESPACES = ("lemn-lab", "null-subjects")
REPO = Path(__file__).resolve().parent.parent
# Compute capability >= 8.0 (FA2-eligible) by product-name prefix.
AMPERE_PLUS = ("NVIDIA-A", "NVIDIA-L", "NVIDIA-H", "NVIDIA-GH", "NVIDIA-GeForce-RTX-30",
               "NVIDIA-GeForce-RTX-40", "NVIDIA-GeForce-RTX-50", "NVIDIA-RTX-A", "NVIDIA-RTX-6000-Ada",
               "NVIDIA-RTX-PRO", "Ada-Generation")
# Taint keys whose plain NoSchedule toleration is legitimate for any GPU pod.
BENIGN_TAINT_KEYS = {"nvidia.com/gpu"}


# ---------------------------------------------------------------- helpers ---

def kubectl_json(*args: str) -> dict:
    out = subprocess.run(["kubectl", *args, "-o", "json"], capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise SystemExit(f"kubectl {' '.join(args)} failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def prom(query: str) -> list:
    url = PROM + "?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url, timeout=60) as r:
        body = json.load(r)
    if body.get("status") != "success":
        raise SystemExit(f"prometheus error for {query!r}: {body}")
    return body["data"]["result"]


def prom_key(resource: str) -> str:
    """k8s resource name -> kube-state-metrics label ('nvidia.com/mig-small' -> 'nvidia_com_mig_small')."""
    return re.sub(r"[./-]", "_", resource)


def qty(v) -> float:
    """k8s quantity -> float (cpu in cores, memory in bytes)."""
    if v is None:
        return 0.0
    s = str(v)
    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
             "m": 1e-3}
    for u in sorted(units, key=len, reverse=True):
        if s.endswith(u):
            return float(s[: -len(u)]) * units[u]
    return float(s)


def gpu_keys(resources: dict) -> dict:
    return {k: int(qty(v)) for k, v in resources.items()
            if k.startswith("nvidia.com/") and k != "nvidia.com/gpu.present" and qty(v) > 0}


def bad_nodes() -> set:
    p = REPO / "configs" / "bad_nodes.txt"
    if not p.exists():
        return set()
    return {ln.split("#")[0].strip() for ln in p.read_text().splitlines() if ln.split("#")[0].strip()}


# ------------------------------------------------- scheduling predicates ----

def _expr_ok(labels: dict, e: dict) -> bool:
    key, op, vals = e["key"], e["operator"], e.get("values") or []
    have = key in labels
    if op == "In":
        return have and labels[key] in vals
    if op == "NotIn":
        return not have or labels[key] not in vals
    if op == "Exists":
        return have
    if op == "DoesNotExist":
        return not have
    if op in ("Gt", "Lt"):
        try:
            x, y = float(labels[key]), float(vals[0])
        except (KeyError, ValueError, IndexError):
            return False
        return x > y if op == "Gt" else x < y
    return False


def affinity_ok(labels: dict, pod_spec: dict, ignore_keys: frozenset = frozenset()) -> bool:
    """Required node affinity (OR over terms, AND within a term) plus nodeSelector.
    Expressions on ``ignore_keys`` are skipped (fit --widen ignores the GPU product)."""
    for k, v in (pod_spec.get("nodeSelector") or {}).items():
        if k not in ignore_keys and labels.get(k) != v:
            return False
    req = (((pod_spec.get("affinity") or {}).get("nodeAffinity") or {})
           .get("requiredDuringSchedulingIgnoredDuringExecution") or {})
    terms = req.get("nodeSelectorTerms") or []
    if not terms:
        return True
    return any(all(_expr_ok(labels, e) for e in (t.get("matchExpressions") or [])
                   if e["key"] not in ignore_keys) for t in terms)


def _tolerates(taint: dict, tol: dict) -> bool:
    if tol.get("effect") and tol["effect"] != taint.get("effect"):
        return False
    if not tol.get("key"):
        return tol.get("operator") == "Exists"
    if tol["key"] != taint.get("key"):
        return False
    if tol.get("operator", "Equal") == "Exists":
        return True
    return tol.get("value", "") == taint.get("value", "")


def untolerated_taints(taints: list, tolerations: list) -> list:
    hard = [t for t in taints or [] if t.get("effect") in ("NoSchedule", "NoExecute")]
    return [t for t in hard if not any(_tolerates(t, tol) for tol in tolerations or [])]


def hard_taint_class(taints: list) -> str:
    """'open', or the first hard taint key we would need (and mostly must not) tolerate."""
    for t in taints or []:
        if t.get("effect") in ("NoSchedule", "NoExecute") and t["key"] not in BENIGN_TAINT_KEYS:
            return f"{t['key']}={t.get('value', '')}"
    return "open"


# ------------------------------------------------------------ cluster ----

def load_cluster() -> dict:
    """Nodes with labels/taints/allocatable and Prometheus allocation per node."""
    nodes = kubectl_json("get", "nodes")["items"]
    phase_join = ('* on(namespace, pod) group_left() max by (namespace, pod) '
                  '(kube_pod_status_phase{phase=~"Running|Pending"} == 1)')
    alloc = defaultdict(dict)   # node -> prom resource -> requested
    for r in prom('sum by (node, resource) (kube_pod_container_resource_requests'
                  '{resource=~"nvidia_com_.*|cpu|memory", node!=""} ' + phase_join + ')'):
        alloc[r["metric"].get("node", "")][r["metric"]["resource"]] = float(r["value"][1])
    pending = {}
    for r in prom('sum by (resource) (kube_pod_container_resource_requests{resource=~"nvidia_com_.*", node=""} '
                  '* on(namespace, pod) group_left() max by (namespace, pod) '
                  '(kube_pod_status_phase{phase="Pending"} == 1))'):
        pending[r["metric"]["resource"]] = float(r["value"][1])
    out = []
    for n in nodes:
        md, st, sp = n["metadata"], n["status"], n["spec"]
        a = alloc.get(md["name"], {})
        g = gpu_keys(st.get("allocatable", {}))
        out.append({
            "name": md["name"], "labels": md.get("labels", {}), "taints": sp.get("taints") or [],
            "unschedulable": bool(sp.get("unschedulable")),
            "product": md.get("labels", {}).get("nvidia.com/gpu.product"),
            "vram_gb": round(qty(md.get("labels", {}).get("nvidia.com/gpu.memory", 0)) / 1024, 1),
            "gpu_alloc": g,
            "gpu_used": {k: a.get(prom_key(k), 0.0) for k in g},
            "cpu_alloc": qty(st.get("allocatable", {}).get("cpu")), "cpu_used": a.get("cpu", 0.0),
            "mem_alloc": qty(st.get("allocatable", {}).get("memory")), "mem_used": a.get("memory", 0.0),
        })
    return {"nodes": out, "pending": pending}


# ---------------------------------------------------------------- gpus ----

def cmd_gpus(a) -> int:
    c = load_cluster()
    bad = bad_nodes()
    rows = defaultdict(lambda: {"open_nodes": 0, "open_total": 0, "open_free": 0, "vram_gb": 0.0,
                                "tainted_total": 0, "tainted_free": 0, "keys": set(), "taints": set()})
    for n in c["nodes"]:
        if not n["product"] or not n["gpu_alloc"]:
            continue
        cls = hard_taint_class(n["taints"])
        total = sum(n["gpu_alloc"].values())
        free = sum(max(0, n["gpu_alloc"][k] - int(n["gpu_used"][k])) for k in n["gpu_alloc"])
        r = rows[n["product"]]
        r["keys"].update(n["gpu_alloc"])
        r["vram_gb"] = max(r["vram_gb"], n["vram_gb"])
        if n["unschedulable"] or n["name"] in bad:
            continue
        if cls == "open":
            r["open_nodes"] += 1
            r["open_total"] += total
            r["open_free"] += free
        else:
            r["tainted_total"] += total
            r["tainted_free"] += free
            r["taints"].add(cls.split("=")[0])
    if a.json:
        print(json.dumps({p: {**v, "keys": sorted(v["keys"]), "taints": sorted(v["taints"])}
                          for p, v in rows.items()}, indent=1))
        return 0
    items = sorted(rows.items(), key=lambda kv: -kv[1]["open_free"])
    print(f"{'GPU product':38s} {'VRAM':>5s} {'FA2':3s} {'open free/total':>15s} {'nodes':>5s}  "
          f"{'tainted free/total':>18s}  request key(s)")
    for p, r in items:
        if not a.all and r["open_total"] == 0:
            continue
        fa2 = "yes" if p.startswith(AMPERE_PLUS) or "Ada-Generation" in p else "no"
        print(f"{p[:38]:38s} {r['vram_gb']:>4.0f}G {fa2:3s} {r['open_free']:>7d}/{r['open_total']:<7d} {r['open_nodes']:>5d}  "
              f"{r['tainted_free']:>8d}/{r['tainted_total']:<9d}  {','.join(sorted(r['keys']))}")
    tot_free = sum(r["open_free"] for r in rows.values())
    print(f"\nopen free GPUs (schedulable, untainted, not in bad_nodes.txt): {tot_free}")
    if c["pending"]:
        print("cluster-wide GPUs requested by unscheduled pods: "
              + ", ".join(f"{k.replace('nvidia_com_', '')}={int(v)}" for k, v in
                          sorted(c["pending"].items(), key=lambda kv: -kv[1])))
    print("(tainted = behind a reservation/hardware/issue/system taint; only tolerate values admins assigned us)")
    return 0


# ----------------------------------------------------------------- fit ----

def _load_pod_spec(target: str) -> tuple[str, dict]:
    if Path(target).is_file():
        import yaml
        docs = [d for d in yaml.safe_load_all(Path(target).read_text()) if d]
        d = docs[0]
        spec = d["spec"]["template"]["spec"] if d.get("kind") in ("Job",) else d.get("spec", {})
        return f"{d.get('kind')}/{d['metadata']['name']}", spec
    ns, kind, name = target.split("/")
    obj = kubectl_json("get", kind, name, "-n", ns)
    return f"{ns}/{kind}/{name}", obj["spec"]["template"]["spec"] if kind == "job" else obj["spec"]


def _pod_requests(spec: dict) -> dict:
    req = defaultdict(float)
    for ctr in spec.get("containers", []):
        for k, v in ((ctr.get("resources") or {}).get("requests") or {}).items():
            req[k] += qty(v)
    return dict(req)


def cmd_fit(a) -> int:
    label, spec = _load_pod_spec(a.target)
    req = _pod_requests(spec)
    gpu_req = {k: int(v) for k, v in req.items() if k.startswith("nvidia.com/")}
    c = load_cluster()
    bad = bad_nodes()
    tolerations = spec.get("tolerations") or []
    reasons = defaultdict(int)
    fits, eligible = [], []
    for n in c["nodes"]:
        if n["unschedulable"]:
            reasons["cordoned"] += 1
            continue
        if not affinity_ok(n["labels"], spec):
            reasons["affinity/selector"] += 1
            continue
        if n["name"] in bad:
            reasons["bad_nodes.txt"] += 1
            continue
        ut = untolerated_taints(n["taints"], tolerations)
        if ut:
            reasons[f"taint {ut[0]['key']}"] += 1
            continue
        if any(n["gpu_alloc"].get(k, 0) < v for k, v in gpu_req.items()):
            reasons["no such GPU key"] += 1
            continue
        eligible.append(n)
        short = []
        for k, v in gpu_req.items():
            if n["gpu_alloc"][k] - n["gpu_used"].get(k, 0) < v:
                short.append("GPU")
        if n["cpu_alloc"] - n["cpu_used"] < req.get("cpu", 0):
            short.append("cpu")
        if n["mem_alloc"] - n["mem_used"] < req.get("memory", 0):
            short.append("memory")
        if short:
            reasons["full: " + "+".join(short)] += 1
        else:
            fits.append(n)
    print(f"{label}: requests {', '.join(f'{k}={v:g}' if 'memory' not in k else f'{k}={v/2**30:.1f}Gi' for k, v in req.items())}")
    print(f"eligible nodes {len(eligible)} ({sum(sum(n['gpu_alloc'].values()) for n in eligible)} GPUs); "
          f"with room right now: {len(fits)}")
    by_prod = defaultdict(lambda: [0, 0])
    for n in eligible:
        by_prod[n["product"]][0] += 1
    for n in fits:
        by_prod[n["product"]][1] += 1
    for p, (e, f) in sorted(by_prod.items(), key=lambda kv: -kv[1][1]):
        print(f"  {str(p):38s} eligible {e:3d}  room now {f:3d}")
    print("blocked: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    if fits:
        print("nodes with room: " + ", ".join(n["name"] for n in fits[:10]) + (" ..." if len(fits) > 10 else ""))
    if a.widen:
        _widen(c, spec, req, gpu_req, bad, tolerations, eligible, a.min_vram)
    return 0


def _widen(c, spec, req, gpu_req, bad, tolerations, eligible, min_vram):
    """Where the pod could go with any GPU product/request key, all other constraints kept."""
    n_gpu = sum(gpu_req.values()) or 1
    if min_vram is None:
        cur = [n["vram_gb"] for n in eligible if n["vram_gb"]]
        min_vram = min(cur) if cur else 0.0
    rows = defaultdict(lambda: {"room": 0, "nodes": 0, "free": 0, "keys": set(), "vram": 0.0})
    for n in c["nodes"]:
        if (not n["gpu_alloc"] or n["unschedulable"] or n["name"] in bad
                or n["vram_gb"] < min_vram
                or not affinity_ok(n["labels"], spec, frozenset({"nvidia.com/gpu.product"}))
                or untolerated_taints(n["taints"], tolerations)):
            continue
        free = {k: n["gpu_alloc"][k] - int(n["gpu_used"].get(k, 0)) for k in n["gpu_alloc"]}
        cpu_ok = n["cpu_alloc"] - n["cpu_used"] >= req.get("cpu", 0)
        mem_ok = n["mem_alloc"] - n["mem_used"] >= req.get("memory", 0)
        r = rows[n["product"]]
        r["nodes"] += 1
        r["keys"].update(n["gpu_alloc"])
        r["vram"] = n["vram_gb"]
        r["free"] += sum(max(0, v) for v in free.values())
        if cpu_ok and mem_ok and any(v >= n_gpu for v in free.values()):
            r["room"] += 1
    cur = {n["product"] for n in eligible}
    print(f"\nwiden (any GPU product with >= {min_vram:g} GB, same taints/bad nodes/cpu/mem):")
    print(f"  {'GPU product':38s} {'VRAM':>5s} {'FA2':3s} {'free GPUs':>9s} {'nodes w/ room':>13s}  key")
    for p, r in sorted(rows.items(), key=lambda kv: (-kv[1]["room"], -kv[1]["free"])):
        if not r["free"]:
            continue
        fa2 = "yes" if p.startswith(AMPERE_PLUS) or "Ada-Generation" in p else "no"
        mark = "" if p in cur else "  <- not in current pool"
        print(f"  {p[:38]:38s} {r['vram']:>4.0f}G {fa2:3s} {r['free']:>9d} {r['room']:>13d}  "
              f"{','.join(sorted(r['keys']))}{mark}")


# ---------------------------------------------------------------- ours ----

def cmd_ours(a) -> int:
    util = {}
    for r in prom('avg by (namespace, pod) (avg_over_time(DCGM_FI_DEV_GPU_UTIL'
                  '{namespace=~"' + "|".join(NAMESPACES) + '", pod=~"thomas-.*"}[15m]))'):
        util[(r["metric"]["namespace"], r["metric"]["pod"])] = float(r["value"][1])
    rows, violators = [], 0
    for ns in NAMESPACES:
        pods = kubectl_json("get", "pods", "-n", ns, "-l", "owner=thomas")["items"]
        top = {}
        t = subprocess.run(["kubectl", "top", "pod", "-n", ns, "--no-headers"], capture_output=True, text=True)
        for ln in t.stdout.splitlines():
            parts = ln.split()
            if len(parts) >= 3:
                top[parts[0]] = (qty(parts[1]), qty(parts[2]))
        for p in pods:
            ph = p["status"].get("phase")
            if ph in ("Succeeded", "Failed"):
                continue
            name = p["metadata"]["name"]
            req = _pod_requests(p["spec"])
            gpus = sum(v for k, v in req.items() if k.startswith("nvidia.com/"))
            cpu_r, mem_r = req.get("cpu", 0), req.get("memory", 0)
            cpu_u, mem_u = top.get(name, (None, None))
            gu = util.get((ns, name))
            exempt = cpu_r <= 1 and mem_r <= 2e9
            flags = []
            if ph == "Running" and not exempt:
                if gpus and gu is not None and gu < 40:
                    flags.append(f"GPU {gu:.0f}%<40")
                if cpu_u is not None and cpu_r and not 0.2 <= cpu_u / cpu_r <= 2.0:
                    flags.append(f"CPU {100 * cpu_u / cpu_r:.0f}%")
                if mem_u is not None and mem_r and not 0.2 <= mem_u / mem_r <= 1.5:
                    flags.append(f"RAM {100 * mem_u / mem_r:.0f}%")
            violators += bool(flags)
            rows.append((ns, name, ph, p["spec"].get("nodeName", "-"),
                         f"{gu:.0f}%" if gu is not None else "-",
                         f"{cpu_u:.2f}/{cpu_r:g}" if cpu_u is not None else f"-/{cpu_r:g}",
                         f"{mem_u / 2**30:.1f}/{mem_r / 2**30:.1f}Gi" if mem_u is not None else f"-/{mem_r / 2**30:.1f}Gi",
                         "exempt" if exempt else (" ".join(flags) or "ok")))
    print(f"{'ns':13s} {'pod':44s} {'phase':9s} {'node':34s} {'GPU':>4s} {'CPU use/req':>12s} {'RAM use/req':>13s}  band")
    for r in rows:
        print(f"{r[0]:13s} {r[1][:44]:44s} {r[2]:9s} {r[3][:34]:34s} {r[4]:>4s} {r[5]:>12s} {r[6]:>13s}  {r[7]}")
    print(f"\nNRP band violators: {violators} (more than 4 blocks new pods; the rolling window is ~1 h)")
    return 0


def cmd_quota(a) -> int:
    for ns in NAMESPACES:
        for q in kubectl_json("get", "resourcequota", "-n", ns)["items"]:
            used, hard = q["status"].get("used", {}), q["status"].get("hard", {})
            scope = [f"{e['scopeName']} {e['operator']} {','.join(e.get('values') or [])}".strip()
                     for e in ((q["spec"].get("scopeSelector") or {}).get("matchExpressions") or [])]
            print(f"{ns:13s} {q['metadata']['name']:22s} " +
                  ", ".join(f"{k}: {used.get(k, '0')}/{v}" for k, v in hard.items()) +
                  (f"   [only pods with {'; '.join(scope)}]" if scope else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gpus"); g.add_argument("--all", action="store_true"); g.add_argument("--json", action="store_true")
    f = sub.add_parser("fit"); f.add_argument("target", help="manifest path, or NS/job/NAME or NS/pod/NAME")
    f.add_argument("--widen", action="store_true", help="also list free GPUs on other products")
    f.add_argument("--min-vram", type=float, default=None,
                   help="GB floor for --widen (default: smallest card in the current pool)")
    sub.add_parser("ours"); sub.add_parser("quota")
    a = ap.parse_args(argv)
    return {"gpus": cmd_gpus, "fit": cmd_fit, "ours": cmd_ours, "quota": cmd_quota}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
