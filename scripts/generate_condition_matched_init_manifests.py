#!/usr/bin/env python3
"""Render checkpoint -1 matched evaluations from a checkpoint inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
IMAGE = (
    "gitlab-registry.nrp-nautilus.io/thmorton/"
    "multi-model-foundry-subject-rearing@"
    "sha256:037b88f45101490ba412890bf431cfefc8c93cba80c7731041649d59aba9a259"
)
ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large",
          "lstm", "mamba_370m"]
SHORT = {"gpt2_small": "gpt2s", "gpt2_medium": "gpt2m",
         "gpt2_large": "gpt2l", "bert_large": "bertl", "lstm": "lstm",
         "mamba_370m": "mamba370m"}
RAM = {"gpt2_small": "10Gi", "gpt2_medium": "12Gi",
       "gpt2_large": "16Gi", "bert_large": "12Gi", "lstm": "14Gi",
       "mamba_370m": "12Gi"}
PARALLELISM = {a: 2 for a in ARCHES}
GPU_POOL = ["NVIDIA-GeForce-RTX-3090", "NVIDIA-A10", "NVIDIA-L4",
            "NVIDIA-GeForce-RTX-4090"]
BAD_NODES = ["gpu-14.nrp.mghpcc.org", "gpu-17.nrp.mghpcc.org",
             "nautilus-it-gpu01.fullerton.edu",
             "nautilus-it-gpu03.fullerton.edu",
             "rci-tide-gpu-03.sdsu.edu", "ry-gpu-10.sdsc.optiputer.net",
             "hcc-nrp-shor-c6017.unl.edu"]
BENCHMARK = "null_subj_v2_condition_matched_init_v1"


def git_ref() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(arch: str, seeds: list, ref: str, inventory_path: str,
           exclude_hp1: bool, manifest_sha256: str) -> dict:
    hp_arg = " --exclude-hp-rank 1" if exclude_hp1 else ""
    command = f'''set -euo pipefail
SEED=$(python3 -c 'import json,os; print(json.loads(os.environ["SEEDS_JSON"])[int(os.environ["JOB_COMPLETION_INDEX"])])')
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits -l 15 &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
python3 -c '
import sys, torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA")
    torch.cuda.synchronize()
    torch.zeros(1024, device="cuda").sum().item()
    print(f"GPU healthy: {{torch.cuda.get_device_name(0)}}")
except Exception as exc:
    print(f"FATAL: GPU unhealthy: {{exc}}", file=sys.stderr)
    sys.exit(2)
'
cd /opt/repo
python3 -u scripts/eval_v2_initialization.py \
  --arch {arch} --seed "$SEED" \
  --cells {inventory_path} --repo /opt/repo --data-root /mnt/data \
  --device cuda --benchmark {BENCHMARK} \
  --matched-stimuli-root /opt/repo/evaluation/stimuli/null-subj-v2-matched-v1 \
  --expected-stimuli-manifest-sha256 {manifest_sha256}{hp_arg}
'''
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": f"thomas-fdy-matched-init-{SHORT[arch]}-v1",
                     "namespace": "lemn-lab",
                     "labels": {"owner": "thomas", "study": "subject-drop",
                                "stage": "matched-eval-init", "arch": arch}},
        "spec": {
            "completionMode": "Indexed", "completions": len(seeds),
            "parallelism": min(PARALLELISM[arch], len(seeds)),
            "backoffLimitPerIndex": 100, "maxFailedIndexes": 0,
            "podReplacementPolicy": "Failed",
            "podFailurePolicy": {"rules": [{"action": "Ignore", "onExitCodes": {
                "containerName": "evaluator", "operator": "In", "values": [2]}}]},
            "activeDeadlineSeconds": 172800, "ttlSecondsAfterFinished": 172800,
            "template": {"metadata": {"labels": {
                "owner": "thomas", "study": "subject-drop",
                "stage": "matched-eval-init", "arch": arch}},
                "spec": {
                    "priorityClassName": "armada-default",
                    "imagePullSecrets": [{"name": "gitlab-registry-cred-thomas"}],
                    "affinity": {"nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [{"matchExpressions": [
                                {"key": "nvidia.com/gpu.product", "operator": "In",
                                 "values": GPU_POOL},
                                {"key": "kubernetes.io/hostname", "operator": "NotIn",
                                 "values": BAD_NODES},
                            ]}]}}},
                    "initContainers": [{
                        "name": "clone-repo", "image": "alpine/git",
                        "command": ["/bin/sh", "-c"],
                        "args": ["set -eu\ncd /opt/repo\ngit init .\ngit remote add origin https://github.com/tgmorton/multi-model-foundry-subject-rearing.git\ngit fetch --depth 1 origin \"$GIT_REF\"\ngit checkout --detach FETCH_HEAD\n"],
                        "env": [{"name": "GIT_REF", "value": ref}],
                        "resources": {"requests": {"cpu": "200m", "memory": "1Gi"},
                                      "limits": {"cpu": "200m", "memory": "1Gi"}},
                        "volumeMounts": [{"name": "repo", "mountPath": "/opt/repo"}],
                    }],
                    "containers": [{
                        "name": "evaluator", "image": IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-c"], "args": [command],
                        "env": [
                            {"name": "PYTHONPATH", "value": "/opt/repo"},
                            {"name": "PYTHONHASHSEED", "value": "0"},
                            {"name": "GIT_REF", "value": ref},
                            {"name": "IMAGE_DIGEST", "value": IMAGE.split("@", 1)[1]},
                            {"name": "SEEDS_JSON", "value": json.dumps(seeds)},
                            {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {
                                "name": "s3-secret-thomas", "key": "AWS_ACCESS_KEY_ID"}}},
                            {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {
                                "name": "s3-secret-thomas", "key": "AWS_SECRET_ACCESS_KEY"}}},
                            {"name": "AWS_ENDPOINT_URL", "value":
                                "http://rook-ceph-rgw-nautiluss3.rook"},
                            {"name": "AWS_DEFAULT_REGION", "value": "us-west-1"},
                            {"name": "REGISTRY_BUCKET", "value":
                                "thomas-subject-drop-artifacts"},
                        ],
                        "resources": {"requests": {"cpu": "4", "memory": RAM[arch],
                            "nvidia.com/gpu": 1, "ephemeral-storage": "20Gi"},
                            "limits": {"cpu": "4", "memory": RAM[arch],
                            "nvidia.com/gpu": 1, "ephemeral-storage": "40Gi"}},
                        "volumeMounts": [{"name": "repo", "mountPath": "/opt/repo"},
                                         {"name": "data", "mountPath": "/mnt/data"}],
                    }],
                    "volumes": [{"name": "repo", "emptyDir": {}},
                                {"name": "data", "persistentVolumeClaim": {
                                    "claimName": "subject-drop-archive"}}],
                    "restartPolicy": "Never", "automountServiceAccountToken": False,
                    "enableServiceLinks": False,
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists",
                                     "effect": "NoSchedule"}],
                }},
        },
    }
    return job


def dump(path: Path, docs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n".join(yaml.safe_dump(x, sort_keys=False) for x in docs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "k8s/condition_matched_eval/init")
    ap.add_argument("--exclude-hp1-arch", action="append", default=[])
    ap.add_argument("--cluster-inventory-path", default=(
        "/mnt/data/eval_v2/null_subj_v2_condition_matched_v1/"
        "checkpoint_inventory.json"))
    args = ap.parse_args()
    payload = json.loads(args.inventory.read_text())
    if payload.get("format_version") != "condition-matched-eval-inventory.v1":
        raise SystemExit("unexpected inventory format")
    if payload.get("rejected"):
        raise SystemExit(
            f"inventory has {len(payload['rejected'])} rejected paths; adjudicate first")
    excluded = set(args.exclude_hp1_arch)
    unknown = excluded.difference(ARCHES)
    if unknown:
        raise SystemExit(f"unknown --exclude-hp1-arch values: {sorted(unknown)}")
    manifest_path = ROOT / "evaluation/stimuli/null-subj-v2-matched-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("vetted") is not True:
        raise SystemExit("matched-stimulus manifest is not gold-vetted")
    manifest_sha = sha256_file(manifest_path)
    seeds = {arch: set() for arch in ARCHES}
    for run in payload["runs"]:
        if run["architecture"] in excluded and int(run["hp_rank"]) == 1:
            continue
        seeds[run["architecture"]].add(int(run["seed"]))
    ref = git_ref()
    jobs = []
    summary = {"git_ref": ref, "benchmark": BENCHMARK,
               "inventory_sha256": sha256_file(args.inventory),
               "stimuli_manifest_sha256": manifest_sha,
               "architectures": {}}
    for arch in ARCHES:
        selected = sorted(seeds[arch])
        if not selected:
            raise SystemExit(f"no seeds for {arch}")
        job = render(arch, selected, ref, args.cluster_inventory_path,
                     exclude_hp1=arch in excluded,
                     manifest_sha256=manifest_sha)
        jobs.append(job)
        dump(args.output_dir / f"job-init-{SHORT[arch]}-v1.yaml", [job])
        summary["architectures"][arch] = {
            "seeds": selected, "gpu_pods": len(selected),
            "parallelism": min(PARALLELISM[arch], len(selected)),
            "excluded_hp1": arch in excluded,
        }
    dump(args.output_dir / "job-init-all-v1.yaml", jobs)
    (args.output_dir / "fleet_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
