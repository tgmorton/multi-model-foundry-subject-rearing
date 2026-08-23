#!/usr/bin/env python3
"""Wave-2 launcher: render + dispatch (cell × arch) training Jobs.

One Indexed Job per (cell, arch), 10 completions = 5 HP ranks × 2
replicates; seeds derived in-pod (blake2b — see wave2_agent.py). Dispatch
is guarded by a PVC free-space watermark. rand-100 cells alias the
shared info-100 corpus (CORPUS_CELL) while keeping their own cell label.

Usage:
  python scripts/wave2_launcher.py --cells pdrop_info10_base pdrop_rand10_base \
      --archs lstm gpt2_small [--dry-run] [--apply]
  python scripts/wave2_launcher.py --cells @cells.txt --archs lstm --apply

Nothing dispatches without --apply (default renders to k8s/wave2/).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE = ("gitlab-registry.nrp-nautilus.io/thmorton/"
         "multi-model-foundry-subject-rearing:latest")
REPO_URL = "https://github.com/tgmorton/multi-model-foundry-subject-rearing.git"
GPU_POOL = ["NVIDIA-GeForce-RTX-3090", "NVIDIA-A10", "NVIDIA-L4",
            "NVIDIA-GeForce-RTX-4090"]
BAD_NODES = ["uicnrp-fiona2.evl.uic.edu"]
# (phys_batch, pod_ram, pod_cpu) — production-derived; no mamba in wave 2.
ARCH_SETTINGS = {
    "gpt2_small": (16, "4Gi", "2"),
    "gpt2_medium": (16, "4Gi", "2"),
    "gpt2_large": (4, "5Gi", "2"),
    "bert_large": (4, "5Gi", "2"),
    "lstm": (16, "4Gi", "2"),
}
_RAND100 = re.compile(r"^pdrop_rand100_(\w+)$")

POD_SCRIPT = """set -euo pipefail
echo "=== WAVE2 TRAIN — cell=$CELL arch=$ARCH idx=$JOB_COMPLETION_INDEX ==="
python3 -c "
import sys, torch
try:
    if not torch.cuda.is_available():
        print('FATAL: no CUDA', file=sys.stderr); sys.exit(2)
    torch.cuda.synchronize()
    x = torch.zeros(1024, device='cuda'); x.sum().item()
    print(f'  GPU healthy: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'FATAL: GPU unhealthy: {e}', file=sys.stderr); sys.exit(2)
"
cd /opt/repo
rm -rf /opt/repo/data/raw /opt/repo/data/manipulations /opt/repo/data/tokenized /opt/repo/data/chunked /opt/repo/tokenizers /opt/repo/models
mkdir -p /mnt/data/models/wave2
ln -sfn /mnt/data/raw           /opt/repo/data/raw
ln -sfn /mnt/data/manipulations /opt/repo/data/manipulations
ln -sfn /mnt/data/tokenized     /opt/repo/data/tokenized
ln -sfn /mnt/data/chunked       /opt/repo/data/chunked
ln -sfn /mnt/data/tokenizers    /opt/repo/tokenizers
ln -sfn /mnt/data/models        /opt/repo/models
rm -f /tmp/run_succeeded
python3 scripts/wave2_agent.py || true
if [ ! -f /tmp/run_succeeded ]; then
  echo "FAIL run sentinel absent"
  exit 1
fi
echo "RUN OK: $(cat /tmp/run_succeeded)"
"""


def render_job(cell: str, arch: str, wave_id: str, epochs: int,
               parallelism: int) -> dict:
    phys, ram, cpu = ARCH_SETTINGS[arch]
    m = _RAND100.match(cell)
    corpus_cell = f"pdrop_info100_{m.group(1)}" if m else cell
    short = f"{arch.replace('_', '')}-{cell.replace('pdrop_', '').replace('_', '-')}"
    env = [
        {"name": "PYTHONPATH", "value": "/opt/repo"},
        {"name": "PYTHONHASHSEED", "value": "0"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "WAVE_ID", "value": wave_id},
        {"name": "CELL", "value": cell},
        {"name": "CORPUS_CELL", "value": corpus_cell},
        {"name": "ARCH", "value": arch},
        {"name": "PHYS_BATCH", "value": str(phys)},
        {"name": "WAVE_EPOCHS", "value": str(epochs)},
        {"name": "NODE_NAME",
         "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "WANDB_API_KEY", "valueFrom": {"secretKeyRef": {
            "name": "wandb-secret-thomas", "key": "WANDB_API_KEY"}}},
        {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {
            "name": "s3-secret-thomas", "key": "AWS_ACCESS_KEY_ID"}}},
        {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {
            "name": "s3-secret-thomas", "key": "AWS_SECRET_ACCESS_KEY"}}},
        {"name": "AWS_ENDPOINT_URL",
         "value": "http://rook-ceph-rgw-nautiluss3.rook"},
        {"name": "AWS_DEFAULT_REGION", "value": "us-west-1"},
        {"name": "REGISTRY_BUCKET", "value": "thomas-subject-drop-artifacts"},
    ]
    labels = {"owner": "thomas", "study": "subject-drop", "lang": "en",
              "stage": "wave2-train", "arch": arch.replace("_", "-")}
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": f"thomas-w2-{short}"[:63], "labels": labels},
        "spec": {
            "backoffLimit": 10,
            "completionMode": "Indexed",
            "completions": 10,
            "parallelism": parallelism,
            "ttlSecondsAfterFinished": 604800,
            "podFailurePolicy": {"rules": [{
                "action": "Ignore",
                "onExitCodes": {"containerName": "trainer", "operator": "In",
                                "values": [2, 128]}}]},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "priorityClassName": "armada-default",
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": "gitlab-registry-cred-thomas"}],
                    "initContainers": [{
                        "name": "clone-repo", "image": "alpine/git",
                        "args": ["clone", "--single-branch", "--depth=1",
                                 "--branch=main", REPO_URL, "/opt/repo"],
                        "resources": {
                            "requests": {"memory": "1Gi", "cpu": "200m"},
                            "limits": {"memory": "1Gi", "cpu": "200m"}},
                        "volumeMounts": [{"name": "repo",
                                          "mountPath": "/opt/repo"}]}],
                    "containers": [{
                        "name": "trainer", "image": IMAGE,
                        "imagePullPolicy": "Always",
                        "command": ["/bin/bash", "-c"],
                        "args": [POD_SCRIPT],
                        "env": env,
                        "resources": {
                            "requests": {"memory": ram, "cpu": cpu,
                                         "nvidia.com/gpu": 1},
                            "limits": {"memory": ram, "cpu": cpu,
                                       "nvidia.com/gpu": 1}},
                        "volumeMounts": [
                            {"name": "repo", "mountPath": "/opt/repo"},
                            {"name": "data", "mountPath": "/mnt/data"}]}],
                    "volumes": [
                        {"name": "repo", "emptyDir": {}},
                        {"name": "data", "persistentVolumeClaim": {
                            "claimName": "subject-drop-archive"}}],
                    "tolerations": [{"key": "nvidia.com/gpu",
                                     "operator": "Exists",
                                     "effect": "PreferNoSchedule"}],
                    "affinity": {"nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [{"matchExpressions": [
                                {"key": "nvidia.com/gpu.product",
                                 "operator": "In", "values": GPU_POOL},
                                {"key": "kubernetes.io/hostname",
                                 "operator": "NotIn", "values": BAD_NODES},
                            ]}]}}},
                }}}}


def pvc_free_tb() -> float:
    """df probe via a throwaway busybox pod; returns free TB."""
    out = subprocess.run(
        ["kubectl", "run", "thomas-w2-dfprobe", "--restart=Never",
         "--image=busybox", "--rm", "-i", "--quiet",
         "--overrides", '{"spec":{"containers":[{"name":"c","image":"busybox",'
         '"stdin":true,"command":["sh","-c","df /mnt/data | tail -1"],'
         '"resources":{"requests":{"memory":"64Mi","cpu":"100m"},'
         '"limits":{"memory":"64Mi","cpu":"100m"}},'
         '"volumeMounts":[{"name":"d","mountPath":"/mnt/data"}]}],'
         '"volumes":[{"name":"d","persistentVolumeClaim":'
         '{"claimName":"subject-drop-archive"}}]}}'],
        capture_output=True, text=True, timeout=300)
    fields = out.stdout.split()
    return int(fields[-3]) / 1e9 if len(fields) >= 4 else -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="cell slugs, or @file with one slug per line")
    ap.add_argument("--archs", nargs="+", default=["lstm", "gpt2_small"],
                    choices=list(ARCH_SETTINGS))
    ap.add_argument("--wave-id", default="wave2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "k8s" / "wave2")
    ap.add_argument("--watermark-free-tb", type=float, default=8.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = []
    for c in args.cells:
        if c.startswith("@"):
            cells += [ln.strip() for ln in open(c[1:]) if ln.strip()]
        else:
            cells.append(c)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cell in cells:
        for arch in args.archs:
            job = render_job(cell, arch, args.wave_id, args.epochs,
                             args.parallelism)
            path = args.out_dir / f"{job['metadata']['name']}.yaml"
            path.write_text(yaml.safe_dump(job, sort_keys=False))
            jobs.append((job["metadata"]["name"], path))
    print(f"rendered {len(jobs)} jobs ({len(cells)} cells × "
          f"{len(args.archs)} archs = {len(jobs) * 10} runs) -> {args.out_dir}")
    if args.dry_run or not args.apply:
        print("dry run / no --apply: nothing dispatched")
        return

    free = pvc_free_tb()
    print(f"PVC free: {free:.1f} TB (watermark {args.watermark_free_tb})")
    if free < args.watermark_free_tb:
        sys.exit("FATAL: below free-space watermark — not dispatching")
    for name, path in jobs:
        rc = subprocess.call(["kubectl", "apply", "-f", str(path)])
        if rc != 0:
            sys.exit(f"FATAL: apply failed for {name}")
    print(f"dispatched {len(jobs)} jobs")


if __name__ == "__main__":
    main()
