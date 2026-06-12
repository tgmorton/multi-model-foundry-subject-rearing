#!/usr/bin/env python3
"""Launch per-run null-subject eval pods (scripts/eval_v2_cell.py) as one
Indexed K8s Job.

One pod per run_id: index i of the Job evaluates RUN_IDS_JSON[i] over all
of its checkpoints. Output parquets land on the PVC under
``/mnt/data/eval_v2/null_subj_v2`` (D11 markers make re-launches free), and
each pod updates the S3 run registry's eval fields.

Eval pods are short (minutes-to-an-hour, I/O-bound on checkpoint reads), so
they run on the 24 GB open pool — every arch fits comfortably in fp32.

Usage:
    # one representative run per EN cell (h0-s42), the EOD all-cells pass
    python scripts/launch_cell_evals.py --lang en --slot h0-s42

    # explicit run list (smoke):
    python scripts/launch_cell_evals.py --run-ids gpt2_small-en-baseline-h0-s42 \
        --name-suffix smoke --extra-args "--max_checkpoints 1 --no-registry \
        --output_root /mnt/data/eval_v2/smoke_cell"

    # dry-run prints YAML
    python scripts/launch_cell_evals.py --lang en --slot h0-s42 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/tgmorton/multi-model-foundry-subject-rearing.git"
DEFAULT_IMAGE = (
    "gitlab-registry.nrp-nautilus.io/thmorton/"
    "multi-model-foundry-subject-rearing:latest"
)

ARCHS = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large",
         "lstm", "mamba_370m"]
CONDITIONS = ["baseline", "remove_expletive_sentences", "impoverish_case",
              "lemmatize_verbs", "enrich_verbal_morphology"]

# Mirrors launch_production_training.py (keep in sync).
GPU_POOL_24GB = [
    "NVIDIA-GeForce-RTX-3090",
    "NVIDIA-A10",
    "NVIDIA-L4",
    "NVIDIA-GeForce-RTX-4090",
]

# Eval-only pools. Eval runs eager-attention fp32 (no FA2), so pre-Ampere
# cards the trainer can't touch are fair game (~370 idle GPUs, plain
# nvidia.com/gpu key). mamba_ssm kernels need sm_70+ → keep mamba cells
# off the Pascal pool.
GPU_POOLS = {
    "default": GPU_POOL_24GB,
    "volta": ["Tesla-V100-SXM2-32GB", "Tesla-V100-SXM2-16GB",
              "Tesla-V100-PCIE-16GB"],
    "turing": ["NVIDIA-GeForce-RTX-2080-Ti", "Tesla-T4", "NVIDIA-TITAN-RTX"],
    "pascal": ["NVIDIA-GeForce-GTX-1080-Ti", "NVIDIA-TITAN-Xp"],
}
BAD_NODES = [
    "rci-tide-gpu-03.sdsu.edu",
    "ry-gpu-10.sdsc.optiputer.net",
    "nautilus-it-gpu03.fullerton.edu",
]


def _resolve_git_ref() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                         text=True, capture_output=True, check=True)
    return out.stdout.strip()


JOB_TEMPLATE = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  labels:
    owner: thomas
    study: subject-drop
    stage: eval-cell
    lang: {lang}
spec:
  backoffLimit: 30
  completionMode: Indexed
  completions: {completions}
  parallelism: {parallelism}
  activeDeadlineSeconds: 172800
  ttlSecondsAfterFinished: 604800
  podFailurePolicy:
    rules:
    - action: Ignore
      onExitCodes:
        containerName: evaluator
        operator: In
        values: [2]
  template:
    metadata:
      labels:
        owner: thomas
        study: subject-drop
        stage: eval-cell
        lang: {lang}
    spec:
      priorityClassName: armada-default
      imagePullSecrets:
      - name: gitlab-registry-cred-thomas
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
{gpu_values}
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
{bad_nodes}
      initContainers:
      - name: clone-repo
        image: alpine/git
        command: ["/bin/sh", "-c"]
        args:
        - |
          set -eu
          cd /opt/repo
          git init .
          git remote add origin {repo_url}
          git fetch --depth 1 origin "$GIT_REF"
          git checkout --detach FETCH_HEAD
        env:
        - {{name: GIT_REF, value: "{git_ref}"}}
        resources:
          requests: {{memory: 1Gi, cpu: "200m"}}
          limits:   {{memory: 1Gi, cpu: "200m"}}
        volumeMounts:
        - {{name: repo, mountPath: /opt/repo}}
      containers:
      - name: evaluator
        image: {image}
        imagePullPolicy: Always
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          RUN_ID=$(python3 -c "import json,os; print(json.loads(os.environ['RUN_IDS_JSON'])[int(os.environ['JOB_COMPLETION_INDEX'])])")
          echo "=========================================="
          echo "  CELL EVAL — $RUN_ID (idx=$JOB_COMPLETION_INDEX)"
          echo "=========================================="

          # GPU health probe (exit 2 → podFailurePolicy Ignore → reschedule).
          python3 -c "
          import sys, torch
          try:
              if not torch.cuda.is_available():
                  print('FATAL: no CUDA', file=sys.stderr); sys.exit(2)
              torch.cuda.synchronize()
              x = torch.zeros(1024, device='cuda'); x.sum().item()
              print(f'  GPU healthy: {{torch.cuda.get_device_name(0)}}')
          except Exception as e:
              print(f'FATAL: GPU unhealthy: {{e}}', file=sys.stderr); sys.exit(2)
          "

          cd /opt/repo
          # PACK>1: this pod owns a slice of RUN_IDS_JSON and runs PACK
          # cells concurrently on its one GPU — overlapping one cell's
          # checkpoint I/O and post-eval upload with another's forwards
          # keeps GPU/CPU/mem duty cycles high (NRP utilization webhook).
          if [ "${{PACK:-1}}" -gt 1 ]; then
            python3 -c 'import json, os; ids = json.loads(os.environ["RUN_IDS_JSON"]); i = int(os.environ["JOB_COMPLETION_INDEX"]); p = int(os.environ["PACK"]); [print(x) for x in ids[i * p:(i + 1) * p]]' > /tmp/my_run_ids
            fail=0
            while read -r rid; do
              [ -z "$rid" ] && continue
              ( python3 scripts/eval_v2_cell.py --run_id "$rid" \\
                  --batch_size {batch_size} --scratch_dir /tmp/eval_scratch \\
                  {extra_args} 2>&1 | sed "s/^/[$rid] /" ) &
            done < /tmp/my_run_ids
            for p in $(jobs -p); do wait "$p" || fail=1; done
            exit $fail
          else
            python3 scripts/eval_v2_cell.py \\
              --run_id "$RUN_ID" \\
              --batch_size {batch_size} \\
              --scratch_dir /tmp/eval_scratch {extra_args}
          fi
        env:
        - {{name: PYTHONPATH, value: "/opt/repo"}}
        - {{name: PYTHONHASHSEED, value: "0"}}
        - {{name: GIT_REF, value: "{git_ref}"}}
        - {{name: PACK, value: "{pack}"}}
        - {{name: RUN_IDS_JSON, value: '{run_ids_json}'}}
        - name: NODE_NAME
          valueFrom: {{fieldRef: {{fieldPath: spec.nodeName}}}}
        - name: AWS_ACCESS_KEY_ID
          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_ACCESS_KEY_ID}}}}
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_SECRET_ACCESS_KEY}}}}
        - {{name: AWS_ENDPOINT_URL,   value: "http://rook-ceph-rgw-nautiluss3.rook"}}
        - {{name: AWS_DEFAULT_REGION, value: "us-west-1"}}
        - {{name: REGISTRY_BUCKET,    value: "thomas-subject-drop-artifacts"}}
        resources:
          requests: {{memory: {pod_ram}, cpu: "{pod_cpu}", nvidia.com/gpu: 1}}
          limits:   {{memory: {pod_ram}, cpu: "{pod_cpu}", nvidia.com/gpu: 1}}
        volumeMounts:
        - {{name: repo, mountPath: /opt/repo}}
        - {{name: data, mountPath: /mnt/data}}
      volumes:
      - {{name: repo, emptyDir: {{}}}}
      - name: data
        persistentVolumeClaim:
          claimName: subject-drop-archive
      restartPolicy: Never
      tolerations:
      - {{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "es"], default="en")
    ap.add_argument("--slot", default="h0-s42",
                    help="HP/seed slot evaluated for every (arch × condition) "
                         "cell (default h0-s42).")
    ap.add_argument("--run-ids", nargs="+",
                    help="Explicit run_id list (overrides --slot grid).")
    ap.add_argument("--name-suffix", default="v1")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--parallelism", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pod-ram", default="4Gi",
                    help="Pod memory request==limit. Observed steady-state "
                         "~1GB (bert/gpt2) with mamba's state-dict load "
                         "peaking ~2.2GB — 4Gi is the 2x-headroom number. "
                         "The first wave's 10Gi requests sat at ~10% use and "
                         "fed the NRP utilization webhook (2026-06-11).")
    ap.add_argument("--gpu-pool", choices=sorted(GPU_POOLS), default="default",
                    help="GPU product pool. volta/turing/pascal target the "
                         "pre-Ampere cards eval can use but training can't "
                         "(eager fp32 — no FA2 needed). mamba needs sm_70+ "
                         "(volta/turing ok, pascal NOT).")
    ap.add_argument("--pack", type=int, default=1,
                    help="Cells run concurrently per pod/GPU. >1 overlaps "
                         "checkpoint I/O + result upload of one cell with "
                         "another's forwards (GPU duty cycle ×pack). Memory "
                         "and CPU requests scale with pack unless overridden.")
    ap.add_argument("--extra-args", default="",
                    help="Extra flags appended to eval_v2_cell.py.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.run_ids:
        run_ids = args.run_ids
    else:
        run_ids = [f"{arch}-{args.lang}-{cond}-{args.slot}"
                   for arch in ARCHS for cond in CONDITIONS]

    if args.gpu_pool == "pascal" and any("mamba" in r for r in run_ids):
        sys.exit("mamba cells need sm_70+ kernels — use volta/turing, "
                 "not pascal")

    pack = max(1, args.pack)
    n_pods = -(-len(run_ids) // pack)  # ceil
    pod_ram = args.pod_ram
    if pack > 1 and args.pod_ram == ap.get_default("pod_ram"):
        pod_ram = f"{2 + 2 * pack}Gi"   # ~2GB/cell + headroom

    name = f"thomas-eval-cell-{args.lang}-{args.name_suffix}"
    yaml_text = JOB_TEMPLATE.format(
        name=name,
        lang=args.lang,
        completions=n_pods,
        parallelism=min(args.parallelism, n_pods),
        gpu_values="\n".join(f"                - {g}"
                             for g in GPU_POOLS[args.gpu_pool]),
        bad_nodes="\n".join(f"                - {n}" for n in BAD_NODES),
        repo_url=REPO_URL,
        git_ref=_resolve_git_ref(),
        image=args.image,
        run_ids_json=json.dumps(run_ids),
        batch_size=args.batch_size,
        pod_ram=pod_ram,
        pod_cpu=str(pack),
        pack=pack,
        extra_args=args.extra_args,
    )

    if args.dry_run:
        print(yaml_text)
        return

    proc = subprocess.run(
        ["kubectl", "apply", "-n", "lemn-lab", "-f", "-"],
        input=yaml_text, text=True, capture_output=True,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)
    print(f"launched {name}: {len(run_ids)} run(s), parallelism "
          f"{min(args.parallelism, len(run_ids))}")


if __name__ == "__main__":
    main()
