#!/usr/bin/env python3
"""Render one opportunistic-tier probe Job (scripts/opp_probe.py) per
(GPU key, priority class) into null-subjects.

    python scripts/render_opp_probe.py --tag 1020 --hours 24 \\
        --keys a100 h100 h200 --classes opportunistic opportunistic2 > /tmp/probes.yaml

Preemptions don't burn retries: podFailurePolicy ignores the DisruptionTarget
condition (and the GPU-probe exit codes 2/128), and the Job controller
replaces the pod, which is the restart NRP's Priority Classes doc recommends.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bad_nodes import load as load_bad_nodes  # noqa: E402

IMAGE = "gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest"
REPO_URL = "https://github.com/tgmorton/multi-model-foundry-subject-rearing.git"
SCRIPT = """set -euo pipefail
python3 -c "
import sys, torch
try:
    if not torch.cuda.is_available():
        print('FATAL: no CUDA', file=sys.stderr); sys.exit(2)
    torch.zeros(1024, device='cuda').sum().item()
    print(f'  GPU healthy: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'FATAL: GPU unhealthy: {e}', file=sys.stderr); sys.exit(2)
"
cd /opt/repo
rm -rf data/raw data/tokenized data/chunked tokenizers models
mkdir -p data /mnt/data/models
ln -sfn /mnt/data/raw data/raw; ln -sfn /mnt/data/tokenized data/tokenized
ln -sfn /mnt/data/chunked data/chunked; ln -sfn /mnt/data/tokenizers tokenizers
ln -sfn /mnt/data/models models
pip install -q "wandb==0.26.0" 2>&1 | tail -2 || true
# One single-GPU probe process per card (the trainer is single-GPU and NRP
# forbids an idle card). They share the pod, so a preemption kills both.
N=${GPUS_PER_POD:-1}
if [ "$N" -le 1 ]; then exec python scripts/opp_probe.py; fi
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i POD_NAME="${POD_NAME}-g$i" SEED_BASE=$((SEED_BASE + 50 * i)) \
    python scripts/opp_probe.py > /tmp/probe_g$i.log 2>&1 &
done
tail -n +1 -F /tmp/probe_g*.log &
wait $(jobs -p | head -n "$N")
"""


def job(tag: str, key: str, cls: str, deadline: float, branch: str, seed_base: int,
        ngpu: int = 1) -> dict:
    kname = key if ngpu == 1 else f"{key}x{ngpu}"
    pid = f"{kname}-{cls}"
    labels = {"owner": "thomas", "study": "subject-drop", "stage": "opp-probe", "gpu": kname, "tier": cls}
    env = [
        {"name": "PYTHONPATH", "value": "/opt/repo"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "PROBE_TAG", "value": tag}, {"name": "PROBE_ID", "value": pid},
        {"name": "GPU_KEY", "value": f"nvidia.com/{key}"}, {"name": "PRIORITY", "value": cls},
        {"name": "DEADLINE_EPOCH", "value": str(int(deadline))},
        # Distinct seed range per probe: two probes on the same seeds train
        # identical replicates (the 2026-10-20 H100 pair did).
        {"name": "SEED_BASE", "value": str(seed_base)},
        {"name": "GPUS_PER_POD", "value": str(ngpu)},
        {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
        {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": "s3-secret-thomas", "key": "AWS_ACCESS_KEY_ID"}}},
        {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": "s3-secret-thomas", "key": "AWS_SECRET_ACCESS_KEY"}}},
        {"name": "AWS_ENDPOINT_URL", "value": "http://rook-ceph-rgw-nautiluss3.rook"},
        {"name": "AWS_DEFAULT_REGION", "value": "us-west-1"},
        {"name": "REGISTRY_BUCKET", "value": "thomas-subject-drop-artifacts"},
    ]
    res = {"cpu": str(2 * ngpu), "memory": f"{6 * ngpu}Gi", f"nvidia.com/{key}": ngpu}
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": f"thomas-opp-probe-{pid}", "namespace": "null-subjects", "labels": labels},
        "spec": {
            "backoffLimit": 20,
            "activeDeadlineSeconds": int(deadline - time.time()) + 3 * 3600,
            "ttlSecondsAfterFinished": 7 * 86400,
            "podFailurePolicy": {"rules": [
                {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]},
                {"action": "Ignore", "onExitCodes": {"containerName": "probe", "operator": "In", "values": [2, 128]}},
            ]},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "priorityClassName": cls,
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": "gitlab-registry-cred-thomas"}],
                    "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [{"matchExpressions": [
                            {"key": "kubernetes.io/hostname", "operator": "NotIn",
                             "values": load_bad_nodes() or ["none"]}]}]}}},
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "PreferNoSchedule"}],
                    "initContainers": [{
                        "name": "clone-repo", "image": "alpine/git",
                        "args": ["clone", "--single-branch", "--depth=1", f"--branch={branch}", REPO_URL, "/opt/repo"],
                        "resources": {"requests": {"cpu": "200m", "memory": "1Gi"}, "limits": {"cpu": "200m", "memory": "1Gi"}},
                        "volumeMounts": [{"name": "repo", "mountPath": "/opt/repo"}]}],
                    "containers": [{
                        "name": "probe", "image": IMAGE, "imagePullPolicy": "Always",
                        "command": ["/bin/bash", "-c"], "args": [SCRIPT], "env": env,
                        "resources": {"requests": res, "limits": res},
                        "volumeMounts": [{"name": "repo", "mountPath": "/opt/repo"},
                                         {"name": "data", "mountPath": "/mnt/data"}]}],
                    "volumes": [{"name": "repo", "emptyDir": {}},
                                {"name": "data", "persistentVolumeClaim": {"claimName": "thomas-rater-corpus"}}],
                },
            },
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--keys", nargs="+", default=["a100", "h100", "h200"])
    ap.add_argument("--classes", nargs="+", default=["opportunistic", "opportunistic2"],
                    choices=["opportunistic", "opportunistic2"])
    ap.add_argument("--branch", default="feat/bert-recipe-v2")
    ap.add_argument("--gpus-per-pod", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0, help="added to every probe's seed base")
    a = ap.parse_args(argv)
    deadline = time.time() + a.hours * 3600
    pairs = [(k, c) for k in a.keys for c in a.classes]
    docs = [job(a.tag, k, c, deadline, a.branch, 5000 + a.seed_offset + 100 * i, a.gpus_per_pod)
            for i, (k, c) in enumerate(pairs)]
    sys.stdout.write(yaml.safe_dump_all(docs, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
