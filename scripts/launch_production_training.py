#!/usr/bin/env python3
"""Launch production training Jobs across (arch × intervention) for one language.

Generates a K8s Job per (arch, intervention) cell, with the architecture's
physical batch size, GPU-pool affinity, and pod resources baked in. Each
Job has completions=10 (5 HP ranks × 2 seeds) and parallelism=2 — so
across the 20 Jobs we burn ~40 GPUs at peak when everything is in flight.

Idempotent: re-applying succeeds without duplication. Failed Jobs are
re-launched by ``watch_production_training.py``; per-pod failures absorb
into ``backoffLimit=100``.

Usage:
    python scripts/launch_production_training.py --lang en
    python scripts/launch_production_training.py --lang en --arch gpt2_medium
    python scripts/launch_production_training.py --lang en --dry-run

The intervention list excludes baseline (handled separately during the
HP sweep). Pass ``--include-baseline`` to launch baseline cells too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Per-arch settings — derived from sweep VRAM telemetry (2026-05-13).
# See memory/reference_production_batch_sizing.md for the audit table.
# ---------------------------------------------------------------------------
ARCH_SETTINGS = {
    # arch_id:    (phys_batch, pod_ram, pod_cpu)
    # pod_ram is BOTH request and limit (NRP requires request==limit for GPU
    # pods). Sized LEAN to MEASURED steady-state use (~2.9Gi gpt2/mamba,
    # ~3.8Gi bert peak; observed via kubectl top 2026-06-02) so NRP's
    # utilization webhook stays satisfied (~70% mem util). The earlier reason
    # for refusing resume entirely was the resume-load RAM transient (~7Gi to
    # torch.load training_state), which would force request==limit so high
    # that steady util drops below the webhook floor. That transient is now
    # mitigated: load_checkpoint uses torch.load(..., mmap=True) so the
    # optimizer tensors stay file-backed instead of spiking RAM. Combined with
    # the explicit resume_state_steps the agents now emit (a small set:
    # ep7 waypoint + midpoint + per-epoch back-half anchors), resume is cheap
    # enough to leave ON at the lean pod_ram. priorityClassName=armada-default
    # is still NON-preemptible, so mid-run pod death remains rare. See
    # memory/feedback_resource_sizing + feedback_mamba_node_cuda_fault_not_kernel.
    "gpt2_small":  (16, "4Gi", "2"),
    "gpt2_medium": (16, "4Gi", "2"),
    "gpt2_large":  ( 4, "4Gi", "2"),
    "bert_large":  ( 4, "5Gi", "2"),   # uses ~3.8Gi peak
    "lstm":        (16, "4Gi", "2"),
    "mamba_370m":  ( 4, "4Gi", "2"),
}

# 24 GB GPU pool only — no L40/L40S (those are 48 GB).
GPU_POOL_24GB = [
    "NVIDIA-GeForce-RTX-3090",
    "NVIDIA-A10",
    "NVIDIA-L4",
    "NVIDIA-GeForce-RTX-4090",
]

# Bad-node blocklist (rendered into nodeAffinity hostname NotIn).
BAD_NODES = [
    "rci-tide-gpu-03.sdsu.edu",
    "ry-gpu-10.sdsc.optiputer.net",
    "nautilus-it-gpu03.fullerton.edu",  # broken CUDA driver — soaked 252 exit-2 "FATAL: no CUDA" fast-fails (2026-05-28)
]

# Same 2 seeds across every (arch, lang, intervention) cell — seed becomes
# a controlled variable for the ablation contrast.
SEEDS = [42, 137]

# Ablations only — baseline is already trained via the HP sweep.
INTERVENTIONS = {
    "en": [
        "remove_expletive_sentences",
        "impoverish_case",
        "lemmatize_verbs",
        "enrich_verbal_morphology",
    ],
    "es": [
        "remove_expletive_sentences",
        "impoverish_case",
        "lemmatize_verbs",
    ],
}


# Mamba kernel works on every 24 GB FA2 card per the 2026-05-28 probe
# (3090 sm_86, A10 sm_86, RTX-4090 sm_89 all PASS; the one L4 failure was a
# node-level torch._C._cuda_init() fault, not an sm_89 kernel-build gap). So
# mamba is NOT pinned to a sub-pool — it keeps the full GPU_POOL_24GB. If a
# future probe shows a real sm_89 break, set MAMBA_POOL to the Ampere subset
# ["NVIDIA-GeForce-RTX-3090","NVIDIA-A10"] and select it for arch=="mamba_370m".
MAMBA_POOL = GPU_POOL_24GB


def _job_yaml(arch: str, lang: str, intervention: str,
              phys_batch: int, pod_ram: str, pod_cpu: str,
              slots: list | None = None,
              parallelism: int = 2,
              active_deadline_seconds: int = 2592000,
              save_resume_last_n: int = 3,
              resume: bool = False,
              job_suffix: str = "") -> str:
    """Return the K8s Job YAML for a single (arch × intervention) cell.

    If ``slots`` is given (an ordered list of [hp_rank, seed_idx] pairs), the
    Job is sized to ``completions=len(slots)`` and passes SLOT_MAP_JSON so the
    agent runs exactly those slots — used by the relaunch to recompute only
    missing/partial runs. If ``slots`` is None the legacy full-grid 10-pod
    Job is emitted.
    """
    name = f"thomas-train-prod-{arch.replace('_', '-')}-{lang}-{intervention.replace('_', '-')}"
    if job_suffix:
        name = f"{name}-{job_suffix}"
    # K8s names cap at 63 chars.
    if len(name) > 63:
        # Use a short hash to keep it unique but ≤63.
        import hashlib
        h = hashlib.sha1(name.encode()).hexdigest()[:6]
        name = name[:56] + "-" + h

    pool = MAMBA_POOL if arch == "mamba_370m" else GPU_POOL_24GB
    gpu_values = "\n".join(f"                - {g}" for g in pool)
    bad_nodes = "\n".join(f"                - {n}" for n in BAD_NODES)
    seeds_json = json.dumps(SEEDS)

    if slots is not None:
        completions = len(slots)
        slot_map_env = (
            f'\n        - {{name: SLOT_MAP_JSON, value: {json.dumps(json.dumps(slots))}}}'
        )
    else:
        completions = 10
        slot_map_env = ""

    # RESUME=1 makes production_agent resume each run IN PLACE from its newest
    # full-state checkpoint and emit a BACK-HALF-ONLY schedule (> resume_step).
    # Required for the recovery of the 207 truncated runs; fresh launches omit
    # it so production_agent computes the full schedule.
    resume_env = '\n        - {name: RESUME, value: "1"}' if resume else ""

    return f"""---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  labels:
    owner: thomas
    study: subject-drop
    stage: train-prod
    arch: {arch.replace('_', '-')}
    lang: {lang}
    intervention: {intervention.replace('_', '-')}
spec:
  backoffLimit: 100
  completionMode: Indexed
  completions: {completions}
  parallelism: {min(parallelism, completions)}
  activeDeadlineSeconds: {active_deadline_seconds}
  ttlSecondsAfterFinished: 604800
  podFailurePolicy:
    rules:
    # The GPU-health / kernel-import probe fast-fails with exit code 2 on a
    # bad node (e.g. the L4 torch._C._cuda_init() fault). Don't burn the
    # backoff budget on infrastructure faults — ignore that pod, let the
    # index reschedule onto a healthy node.
    - action: Ignore
      onExitCodes:
        containerName: trainer
        operator: In
        values: [2]
  template:
    metadata:
      labels:
        owner: thomas
        study: subject-drop
        stage: train-prod
        arch: {arch.replace('_', '-')}
        lang: {lang}
        intervention: {intervention.replace('_', '-')}
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
        args:
        - clone
        - --single-branch
        - --depth=1
        - --branch=main
        - https://github.com/tgmorton/multi-model-foundry-subject-rearing.git
        - /opt/repo
        resources:
          requests: {{memory: 1Gi, cpu: "200m"}}
          limits:   {{memory: 1Gi, cpu: "200m"}}
        volumeMounts:
        - {{name: repo, mountPath: /opt/repo}}
      containers:
      - name: trainer
        image: gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest
        imagePullPolicy: Always
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          echo "=========================================="
          echo "  PROD TRAIN — {arch} × {lang} × {intervention}"
          echo "  pod idx=$JOB_COMPLETION_INDEX  (slot resolved by production_agent: SLOT_MAP_JSON if set, else divmod(idx,2))"
          echo "=========================================="
          cat /opt/repo/.git/HEAD 2>/dev/null || true

          # GPU health probe — same 5-sec probe the sweep used.
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
          rm -rf /opt/repo/data/raw /opt/repo/data/manipulations /opt/repo/data/tokenized /opt/repo/data/chunked /opt/repo/tokenizers /opt/repo/models
          mkdir -p /mnt/data/tokenized /mnt/data/chunked /mnt/data/tokenizers /mnt/data/models/production
          ln -sfn /mnt/data/raw           /opt/repo/data/raw
          ln -sfn /mnt/data/manipulations /opt/repo/data/manipulations
          ln -sfn /mnt/data/tokenized     /opt/repo/data/tokenized
          ln -sfn /mnt/data/chunked       /opt/repo/data/chunked
          ln -sfn /mnt/data/tokenizers    /opt/repo/tokenizers
          ln -sfn /mnt/data/models        /opt/repo/models
          for d in /opt/repo/data/raw /opt/repo/data/tokenized /opt/repo/data/chunked /opt/repo/tokenizers /opt/repo/models; do
            [ -L "$d" ] || {{ echo "FAIL $d is not a symlink"; exit 1; }}
          done

          rm -f /tmp/run_succeeded
          python3 scripts/production_agent.py || true
          if [ ! -f /tmp/run_succeeded ]; then
            echo "FAIL run sentinel absent — training did not reach a clean completion"
            exit 1
          fi
          echo "RUN OK: $(cat /tmp/run_succeeded)"
        env:
        - {{name: PYTHONPATH, value: "/opt/repo"}}
        - {{name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}}
        - {{name: ARCH, value: "{arch}"}}
        - {{name: LANG, value: "{lang}"}}
        - {{name: INTERVENTION, value: "{intervention}"}}
        - {{name: PHYS_BATCH, value: "{phys_batch}"}}
        - {{name: SEEDS_JSON, value: '{seeds_json}'}}{slot_map_env}
        - {{name: SAVE_RESUME_LAST_N, value: "{save_resume_last_n}"}}{resume_env}
        - {{name: WANDB_PROJECT_PROD, value: "subject-drop-production"}}
        - name: WANDB_API_KEY
          valueFrom: {{secretKeyRef: {{name: wandb-secret-thomas, key: WANDB_API_KEY}}}}
        - name: AWS_ACCESS_KEY_ID
          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_ACCESS_KEY_ID}}}}
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_SECRET_ACCESS_KEY}}}}
        - {{name: AWS_ENDPOINT_URL,   value: "http://rook-ceph-rgw-nautiluss3.rook"}}
        - {{name: AWS_DEFAULT_REGION, value: "us-west-1"}}
        - {{name: REGISTRY_BUCKET,    value: "thomas-subject-drop-artifacts"}}
        - {{name: DOCKER_IMAGE,       value: "gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest"}}
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


def _apply_yaml(yaml_text: str) -> bool:
    """`kubectl apply -f -` with the YAML on stdin. Returns True on success."""
    proc = subprocess.run(
        ["kubectl", "apply", "-n", "lemn-lab", "-f", "-"],
        input=yaml_text, text=True, capture_output=True,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "es"], required=True)
    ap.add_argument("--arch", choices=list(ARCH_SETTINGS.keys()),
                    help="Only launch for this arch (default: all archs)")
    ap.add_argument("--intervention",
                    help="Only launch for this intervention (default: all)")
    ap.add_argument("--include-baseline", action="store_true",
                    help="Also launch baseline cells (default: false — sweep "
                         "already trained baseline)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print YAML to stdout instead of applying")
    ap.add_argument("--slots-file",
                    help="JSON map {'<arch>|<intervention>': [[hp_rank, "
                         "seed_idx], ...]} of only the slots that still need "
                         "training. Cells absent from the map are skipped; "
                         "the per-cell Job is sized to len(slots) and passed "
                         "SLOT_MAP_JSON so done seeds are never recomputed.")
    ap.add_argument("--parallelism", type=int, default=2,
                    help="Max concurrent pods per cell (capped at completions)")
    ap.add_argument("--active-deadline-seconds", type=int, default=2592000,
                    help="Job activeDeadlineSeconds (default 30d, was 14d)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume each run IN PLACE from its newest full-state "
                         "checkpoint (sets RESUME=1 → production_agent emits a "
                         "back-half-only schedule > resume_step). Use for the "
                         "recovery of truncated runs; omit for fresh launches.")
    ap.add_argument("--seeds",
                    help="JSON list overriding the default seeds [42, 137] "
                         "(e.g. '[999]' for a throwaway validation run). "
                         "SLOT_MAP_JSON seed_idx indexes into this list.")
    ap.add_argument("--job-suffix", default="",
                    help="Append '-<suffix>' to each Job name so two launches "
                         "of the SAME (arch, intervention) cell don't collide "
                         "(e.g. --job-suffix resume for recoverable slots vs "
                         "--job-suffix fresh for the unrecoverable/missing "
                         "slots of the same cell).")
    ap.add_argument("--save-resume-last-n", type=int, default=3,
                    help="LEGACY FALLBACK ONLY. production_agent now emits an "
                         "explicit resume_state_steps set ({ep7 waypoint, "
                         "midpoint, all back-half per-epoch anchors}) which the "
                         "training loop prioritizes over this value — so runs "
                         "DO write training_state.pt on those designated anchors "
                         "regardless of this flag. This last-N suffix is only "
                         "consulted if resume_state_steps were ever unset. Kept "
                         "for backward compat; default 3.")
    args = ap.parse_args()

    if args.seeds:
        global SEEDS
        SEEDS = json.loads(args.seeds)

    slots_map = {}
    if args.slots_file:
        slots_map = json.loads(Path(args.slots_file).read_text())

    interventions = list(INTERVENTIONS[args.lang])
    if args.include_baseline:
        interventions = ["baseline"] + interventions
    if args.intervention:
        if args.intervention not in interventions:
            sys.exit(f"intervention {args.intervention!r} not in valid set "
                     f"{interventions}")
        interventions = [args.intervention]

    archs = [args.arch] if args.arch else list(ARCH_SETTINGS.keys())

    print(f"=== Launching production training ===")
    print(f"  lang={args.lang}")
    print(f"  archs={archs}")
    print(f"  interventions={interventions}")
    print(f"  total cells={len(archs) * len(interventions)} "
          f"× 10 pods/cell = {len(archs) * len(interventions) * 10} runs")
    # Per-cell parallelism is capped at completions (10), so peak concurrent
    # GPUs per cell is min(parallelism, 10). (Cells run concurrently too, but
    # this reports the per-cell ceiling the prior code conflated.)
    print(f"  peak concurrent per cell: {min(args.parallelism, 10)} GPUs")
    print()

    successes, failures = 0, 0
    for arch in archs:
        phys, ram, cpu = ARCH_SETTINGS[arch]
        for intervention in interventions:
            slots = None
            if args.slots_file:
                key = f"{arch}|{intervention}"
                slots = slots_map.get(key)
                if not slots:
                    continue  # cell fully done — skip
            yml = _job_yaml(
                arch, args.lang, intervention, phys, ram, cpu,
                slots=slots, parallelism=args.parallelism,
                active_deadline_seconds=args.active_deadline_seconds,
                save_resume_last_n=args.save_resume_last_n,
                resume=args.resume,
                job_suffix=args.job_suffix,
            )
            if args.dry_run:
                n = len(slots) if slots is not None else 10
                print(f"--- {arch} × {intervention}  (completions={n}) ---")
                print(yml)
                continue
            ok = _apply_yaml(yml)
            successes += int(ok)
            failures += int(not ok)

    if not args.dry_run:
        print(f"\nApplied {successes} Jobs; {failures} failed.")
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
