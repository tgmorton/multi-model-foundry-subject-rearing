#!/usr/bin/env python3
"""Render the predeclared English Stage-1 early-seed Kubernetes Jobs.

This is a manifest generator only.  It never calls kubectl.

The design is:

* six architectures x five existing English conditions = 30 cells;
* the same 12 seeds in every cell;
* seeds 42 and 137 are legacy full-run anchors and are not relaunched;
* seed 314159 is the countable corrected-schedule sentinel;
* one Indexed Job per cell, with ``parallelism: 1``;
* only the selected h0 configuration is trained, for two epochs.

Render:

    python scripts/generate_stage1_early_manifests.py

Outputs:

    k8s/stage1/sentinels/job-stage1-early-sentinel-<arch>-en.yaml
    k8s/stage1/job-stage1-early-wave-en.yaml
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "k8s" / "stage1"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_production_training.py"

# Corrected implementation: stop after two epochs while calculating warmup
# and linear decay against the canonical 30-epoch horizon.
TRAINING_GIT_REF = "c5325534fd4e7021d7691a50f771279150c485ca"
TRAINING_IMAGE = (
    "gitlab-registry.nrp-nautilus.io/thmorton/"
    "multi-model-foundry-subject-rearing"
    "@sha256:037b88f45101490ba412890bf431cfefc8c93cba80c7731041649d59aba9a259"
)

LANG = "en"
PROD_EPOCHS = 2
SCHEDULER_EPOCHS = 30
HP_RANK = 0
RUN_REVISION = "lr30r1"
ACTIVE_DEADLINE_SECONDS = 604800  # seven-day finite guard for serial cells
STAGE1_EXCLUDED_NODES = {
    # Three independent GPT-2-small wave Pods were cgroup OOMKilled before
    # update 1 here while sibling Pods using the same tuple ran elsewhere.
    "nautilus-it-gpu01.fullerton.edu",
    # Repeated CUDA driver initialization failures in the corrected GPT-2
    # small tranche (2026-07-28).
    "gpu-14.nrp.mghpcc.org",
    "gpu-17.nrp.mghpcc.org",
}
# Cluster placement/driver failures must not cause a scientifically valid seed
# to disappear. This is an emergency ceiling rather than an expected retry
# count; the finite Job deadline and external monitor remain the primary guards.
STAGE1_RETRIES_PER_INDEX = 100

# Informal preregistration: identical seed IDs in every cell.
STAGE1_SEEDS = [
    42,
    137,
    314159,
    1568568120,
    1415936399,
    1640623142,
    1595274352,
    2022192329,
    1891911437,
    1881877998,
    302485963,
    2078344582,
]

LEGACY_ANCHOR_INDICES = {0, 1}
SENTINEL_SEED_INDEX = 4  # non-selected early-only seed 1415936399
FULL_RUN_SEEDS = {314159, 1568568120, 1640623142, 302485963}

ARCHITECTURES = [
    "gpt2_small",
    "gpt2_medium",
    "gpt2_large",
    "bert_large",
    "lstm",
    "mamba_370m",
]
CONDITIONS = [
    "baseline",
    "remove_expletive_sentences",
    "impoverish_case",
    "lemmatize_verbs",
    "enrich_verbal_morphology",
]

# The corrected rollout starts with exactly one representative GPT-2-small
# baseline sentinel. Once it passes, the remaining 299 early-only slots run
# at one Pod/GPU per cell.
COMPLETED_SENTINEL_ARCHES = {"gpt2_small"}
SENTINEL_ARCHES_TO_RUN = ["gpt2_small"]


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "launch_production_training", LAUNCHER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_id(arch: str, condition: str, seed: int) -> str:
    return f"{arch}-{LANG}-{condition}-h{HP_RANK}-s{seed}"


def _job_name(arch: str, condition: str, phase: str) -> str:
    arch_short = {
        "gpt2_small": "gpt2s",
        "gpt2_medium": "gpt2m",
        "gpt2_large": "gpt2l",
        "bert_large": "bertl",
        "lstm": "lstm",
        "mamba_370m": "mamba370m",
    }[arch]
    cond_short = {
        "baseline": "base",
        "remove_expletive_sentences": "rmexpl",
        "impoverish_case": "impcase",
        "lemmatize_verbs": "lemmaverb",
        "enrich_verbal_morphology": "enrichmorph",
    }[condition]
    name = f"thomas-fdy-s1e-{phase}-{arch_short}-{cond_short}"
    name += f"-{RUN_REVISION}"
    if len(name) > 63:
        raise AssertionError(f"Job name exceeds 63 characters: {name}")
    return name


def _slots_for_wave(arch: str, condition: str) -> list[list[int]]:
    slots: list[list[int]] = []
    for seed_index in range(len(STAGE1_SEEDS)):
        if seed_index in LEGACY_ANCHOR_INDICES:
            continue
        if STAGE1_SEEDS[seed_index] in FULL_RUN_SEEDS:
            continue
        if (
            condition == "baseline"
            and arch in COMPLETED_SENTINEL_ARCHES
            and seed_index == SENTINEL_SEED_INDEX
        ):
            continue
        slots.append([HP_RANK, seed_index])
    return slots


def _slots_for_full_runs() -> list[list[int]]:
    return [
        [HP_RANK, seed_index]
        for seed_index, seed in enumerate(STAGE1_SEEDS)
        if seed in FULL_RUN_SEEDS
    ]


def _labels(arch: str, condition: str, phase: str) -> dict[str, str]:
    return {
        "owner": "thomas",
        "study": "subject-drop",
        "stage": f"train-early-{phase}",
        "arch": arch.replace("_", "-"),
        "lang": LANG,
        "intervention": condition.replace("_", "-"),
    }


def _inject_collision_guard(
    job: dict[str, Any],
    arch: str,
    condition: str,
) -> None:
    """Fail before production_agent if the assigned output directory exists."""
    container = job["spec"]["template"]["spec"]["containers"][0]
    script = container["args"][0]
    needle = "rm -f /tmp/run_succeeded"
    guard = f"""\
# Resolve this Indexed Job's exact predeclared run ID and refuse to write into
# any existing checkpoint directory.  Completed/partial artifacts require a
# read-only provenance review and an explicit slot-map revision.
RUN_ID=$(python3 - <<'PY_SLOT'
import json, os
seeds = json.loads(os.environ["SEEDS_JSON"])
slots = json.loads(os.environ["SLOT_MAP_JSON"])
_, seed_idx = slots[int(os.environ["JOB_COMPLETION_INDEX"])]
seed = int(seeds[seed_idx])
print("{arch}-{LANG}-{condition}-h{HP_RANK}-s" + str(seed))
PY_SLOT
)
if [ -e "/mnt/data/models/production/$RUN_ID" ]; then
  RETRY_COUNT="${{JOB_INDEX_FAILURE_COUNT:-0}}"
  if [ "$RETRY_COUNT" -gt 0 ]; then
    RETRY_ARCHIVE="/mnt/data/models/production/_failed_attempts/${{RUN_ID}}__index-${{JOB_COMPLETION_INDEX}}__retry-${{RETRY_COUNT}}__$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p /mnt/data/models/production/_failed_attempts
    test ! -e "$RETRY_ARCHIVE"
    mv "/mnt/data/models/production/$RUN_ID" "$RETRY_ARCHIVE"
    echo "Archived failed index attempt to $RETRY_ARCHIVE"
  else
    echo "FATAL: collision: /mnt/data/models/production/$RUN_ID already exists"
    exit 3
  fi
fi
"""
    if needle not in script:
        raise AssertionError("Launcher script no longer contains collision insertion point")
    container["args"][0] = script.replace(needle, guard + needle, 1)


def _make_job(
    launcher,
    arch: str,
    condition: str,
    slots: list[list[int]],
    phase: str,
    parallelism: int = 1,
    run_epochs: int = PROD_EPOCHS,
) -> dict[str, Any]:
    phys_batch, pod_ram, pod_cpu = launcher.ARCH_SETTINGS[arch]
    original_seeds = copy.deepcopy(launcher.SEEDS)
    try:
        launcher.SEEDS = list(STAGE1_SEEDS)
        rendered = launcher._job_yaml(
            arch,
            LANG,
            condition,
            phys_batch,
            pod_ram,
            pod_cpu,
            slots=slots,
            parallelism=parallelism,
            active_deadline_seconds=ACTIVE_DEADLINE_SECONDS,
            save_resume_last_n=3,
            resume=False,
            job_suffix=f"s1e-{phase}",
            git_ref=TRAINING_GIT_REF,
            image=TRAINING_IMAGE,
            prod_epochs=run_epochs,
            scheduler_epochs=SCHEDULER_EPOCHS,
        )
    finally:
        launcher.SEEDS = original_seeds

    job = yaml.safe_load(rendered)
    job["metadata"]["name"] = _job_name(arch, condition, phase)
    if phase == "wave" and parallelism == 2:
        job["metadata"]["name"] += "-nf1"
    if arch == "bert_large" and condition == "baseline" and phase == "sentinel":
        job["metadata"]["name"] += "-r1"
    job["metadata"]["labels"] = _labels(arch, condition, phase)
    job["metadata"]["annotations"] = {
        "foundry.thesis/purpose": "stage1-early-seed-lr30-rerun",
        "foundry.thesis/design": "12-seeds-2-legacy-6-early-4-full",
        "foundry.thesis/hp-rank": str(HP_RANK),
        "foundry.thesis/epochs": str(run_epochs),
        "foundry.thesis/scheduler-epochs": str(SCHEDULER_EPOCHS),
        "foundry.thesis/revision": RUN_REVISION,
        "foundry.thesis/supersedes": "stage1-two-epoch-lr-schedule",
        "foundry.thesis/code-sha": TRAINING_GIT_REF,
        "foundry.thesis/image-digest": TRAINING_IMAGE.rsplit("@", 1)[1],
        "foundry.thesis/seed-set": ",".join(map(str, STAGE1_SEEDS)),
        "foundry.thesis/slot-map": json.dumps(slots, separators=(",", ":")),
    }
    if arch == "bert_large" and condition == "baseline" and phase == "sentinel":
        job["metadata"]["annotations"]["foundry.thesis/retry-of"] = (
            "thomas-fdy-s1e-sentinel-bertl-base"
        )
        job["metadata"]["annotations"]["foundry.thesis/retry-reason"] = (
            "container-memory-oom-at-5Gi"
        )

    spec = job["spec"]
    spec["parallelism"] = parallelism
    spec["completions"] = len(slots)
    if parallelism == 2:
        spec.pop("backoffLimit", None)
        spec["backoffLimitPerIndex"] = STAGE1_RETRIES_PER_INDEX
        spec["maxFailedIndexes"] = 0
        spec["podReplacementPolicy"] = "Failed"
    else:
        spec["backoffLimit"] = 0
    spec["activeDeadlineSeconds"] = ACTIVE_DEADLINE_SECONDS
    spec.pop("podFailurePolicy", None)
    spec.pop("ttlSecondsAfterFinished", None)

    pod_template = spec["template"]
    pod_template["metadata"]["labels"] = _labels(arch, condition, phase)
    pod = pod_template["spec"]
    pod["automountServiceAccountToken"] = False
    pod["enableServiceLinks"] = False
    expressions = pod["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    hostname_exclusion = next(
        expression
        for expression in expressions
        if expression["key"] == "kubernetes.io/hostname"
        and expression["operator"] == "NotIn"
    )
    hostname_exclusion["values"] = sorted(
        set(hostname_exclusion["values"]) | STAGE1_EXCLUDED_NODES
    )

    trainer = pod["containers"][0]
    trainer["imagePullPolicy"] = "IfNotPresent"
    if parallelism == 2:
        trainer["env"].append(
            {
                "name": "JOB_INDEX_FAILURE_COUNT",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": (
                            "metadata.annotations"
                            "['batch.kubernetes.io/job-index-failure-count']"
                        )
                    }
                },
            }
        )
    if arch == "bert_large":
        # The first production-shaped sentinel was cgroup OOM-killed at the
        # legacy 5 GiB host-memory allocation. Keep the 24 GB GPU tuple and
        # test the smallest sensible next RAM tier before any BERT wave.
        trainer["resources"]["requests"]["memory"] = "8Gi"
        trainer["resources"]["limits"]["memory"] = "8Gi"

    _inject_collision_guard(job, arch, condition)
    return job


def _header(kind: str, jobs: list[dict[str, Any]]) -> str:
    completions = sum(int(j["spec"]["completions"]) for j in jobs)
    return (
        "# GENERATED by scripts/generate_stage1_early_manifests.py; do not edit.\n"
        f"# phase: {kind}\n"
        f"# jobs: {len(jobs)}\n"
        f"# trajectories: {completions}\n"
        "# kubectl context/namespace are intentionally not embedded in Kubernetes\n"
        "# objects. Any future submission must explicitly use:\n"
        "#   kubectl --context=nautilus ... -n lemn-lab\n"
        "# No objects have been submitted by this generator.\n"
    )


class _ManifestDumper(yaml.SafeDumper):
    """Emit shell/Python command bodies as readable literal blocks."""


def _represent_string(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_ManifestDumper.add_representer(str, _represent_string)


def _write_documents(path: Path, kind: str, jobs: list[dict[str, Any]]) -> None:
    body = yaml.dump_all(
        jobs,
        Dumper=_ManifestDumper,
        sort_keys=False,
        explicit_start=True,
        allow_unicode=True,
    )
    path.write_text(_header(kind, jobs) + body)


def main() -> None:
    if len(STAGE1_SEEDS) != 12 or len(set(STAGE1_SEEDS)) != 12:
        raise AssertionError("Stage-1 seed set must contain 12 unique seeds")
    if HP_RANK != 0:
        raise AssertionError("Stage-1 is predeclared for selected h0 only")

    launcher = _load_launcher()
    if set(ARCHITECTURES) != set(launcher.ARCH_SETTINGS):
        raise AssertionError("Architecture list drifted from production launcher")
    if CONDITIONS[1:] != list(launcher.INTERVENTIONS[LANG]):
        raise AssertionError("English condition list drifted from production launcher")

    sentinel_jobs = [
        _make_job(
            launcher,
            arch,
            "baseline",
            [[HP_RANK, SENTINEL_SEED_INDEX]],
            "sentinel",
        )
        for arch in SENTINEL_ARCHES_TO_RUN
    ]

    early_jobs = [
        _make_job(
            launcher,
            arch,
            condition,
            _slots_for_wave(arch, condition),
            "early",
            parallelism=2,
        )
        for arch in ARCHITECTURES
        for condition in CONDITIONS
    ]
    full_jobs = [
        _make_job(
            launcher,
            arch,
            condition,
            _slots_for_full_runs(),
            "full",
            parallelism=2,
            run_epochs=SCHEDULER_EPOCHS,
        )
        for arch in ARCHITECTURES
        for condition in CONDITIONS
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sentinel_dir = OUT_DIR / "sentinels"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    for job in sentinel_jobs:
        arch = job["metadata"]["labels"]["arch"]
        _write_documents(
            sentinel_dir / f"job-stage1-early-sentinel-{arch}-en.yaml",
            f"architecture-sentinel-{arch}",
            [job],
        )
    _write_documents(
        OUT_DIR / "job-stage1-early-wave-en.yaml",
        "parallelism-2-early-only-wave",
        early_jobs,
    )
    _write_documents(
        OUT_DIR / "job-stage1-full-selected-wave-en.yaml",
        "parallelism-2-selected-full-wave-after-early",
        full_jobs,
    )

    print(
        json.dumps(
            {
                "sentinel_jobs": len(sentinel_jobs),
                "sentinel_trajectories": sum(
                    j["spec"]["completions"] for j in sentinel_jobs
                ),
                "early_jobs": len(early_jobs),
                "early_trajectories": sum(
                    j["spec"]["completions"] for j in early_jobs
                ),
                "full_jobs": len(full_jobs),
                "full_trajectories": sum(
                    j["spec"]["completions"] for j in full_jobs
                ),
                "max_parallel_per_cell": 2,
                "max_parallel_per_stage": 60,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
