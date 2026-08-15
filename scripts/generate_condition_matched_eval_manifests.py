#!/usr/bin/env python3
"""Render the complete condition-matched eval fleet from a PVC inventory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parent.parent
PINNED_IMAGE = (
    "gitlab-registry.nrp-nautilus.io/thmorton/"
    "multi-model-foundry-subject-rearing@"
    "sha256:037b88f45101490ba412890bf431cfefc8c93cba80c7731041649d59aba9a259"
)
ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large",
          "lstm", "mamba_370m"]
PACK = {"gpt2_small": 4, "gpt2_medium": 3, "gpt2_large": 2,
        "bert_large": 3, "lstm": 6, "mamba_370m": 2}
RAM = {"gpt2_small": "10Gi", "gpt2_medium": "12Gi",
       "gpt2_large": "16Gi", "bert_large": "12Gi", "lstm": "14Gi",
       "mamba_370m": "12Gi"}
PARALLELISM = {a: 4 for a in ARCHES}
SHORT = {"gpt2_small": "gpt2s", "gpt2_medium": "gpt2m",
         "gpt2_large": "gpt2l", "bert_large": "bertl", "lstm": "lstm",
         "mamba_370m": "mamba370m"}
BAD_NODES = ["gpu-14.nrp.mghpcc.org", "gpu-17.nrp.mghpcc.org",
             "nautilus-it-gpu01.fullerton.edu",
             "nautilus-it-gpu03.fullerton.edu",
             "rci-tide-gpu-03.sdsu.edu", "ry-gpu-10.sdsc.optiputer.net",
             "hcc-nrp-shor-c6017.unl.edu"]
BENCHMARK = "null_subj_v2_condition_matched_v1"
SCORING = "null-subj-v2-condition-matched-v1"


def load_launcher():
    path = ROOT / "scripts/launch_cell_evals.py"
    spec = importlib.util.spec_from_file_location("eval_launcher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def render_job(launcher, arch: str, run_ids: list, git_ref: str,
               manifest_sha256: str, name_suffix: str) -> dict:
    pack = PACK[arch]
    completions = math.ceil(len(run_ids) / pack)
    extra = (
        "--matched_stimuli_root "
        "/opt/repo/evaluation/stimuli/null-subj-v2-matched-v1 "
        f"--benchmark {BENCHMARK} "
        f"--output_root /mnt/data/eval_v2/{BENCHMARK} "
        f"--expected_stimuli_manifest_sha256 {manifest_sha256} "
        f"--scoring_version {SCORING} --slor"
    )
    text = launcher.JOB_TEMPLATE.format(
        name=f"thomas-fdy-matched-{SHORT[arch]}-{name_suffix}", lang="en",
        completions=completions,
        parallelism=min(PARALLELISM[arch], completions),
        gpu_values="\n".join(
            f"                - {gpu}" for gpu in launcher.GPU_POOL_24GB),
        bad_nodes="\n".join(f"                - {node}" for node in BAD_NODES),
        repo_url=launcher.REPO_URL, git_ref=git_ref, image=PINNED_IMAGE,
        run_ids_json=json.dumps(run_ids), batch_size=64,
        pod_ram=RAM[arch], pod_cpu=str(pack), pack=pack, extra_args=extra,
    )
    job = yaml.safe_load(text)
    job["metadata"]["namespace"] = "lemn-lab"
    job["metadata"]["labels"].update({
        "arch": arch, "phase": "production", "eval-regime": "matched-v1",
    })
    job["spec"].pop("backoffLimit", None)
    job["spec"].update({
        "backoffLimitPerIndex": 100, "maxFailedIndexes": 0,
        "podReplacementPolicy": "Failed",
    })
    pod = job["spec"]["template"]["spec"]
    pod["automountServiceAccountToken"] = False
    pod["enableServiceLinks"] = False
    container = pod["containers"][0]
    container["imagePullPolicy"] = "IfNotPresent"
    marker = "cd /opt/repo\n"
    telemetry = (marker +
        "          nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory "
        "--format=csv,noheader,nounits -l 15 &\n"
        "          TELEMETRY_PID=$!\n"
        "          trap 'kill $TELEMETRY_PID 2>/dev/null || true' EXIT\n")
    container["args"][0] = container["args"][0].replace(marker, telemetry, 1)
    if "nvidia-smi --query-gpu" not in container["args"][0]:
        raise RuntimeError(f"telemetry injection failed for {arch}")
    return job


def dump(path: Path, docs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n".join(
        yaml.safe_dump(doc, sort_keys=False) for doc in docs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "k8s/condition_matched_eval")
    ap.add_argument(
        "--exclude-run-ids", type=Path,
        help="Optional JSON list of currently mutable/in-progress run IDs.",
    )
    ap.add_argument(
        "--exclude-arch-hp", action="append", default=[], metavar="ARCH:HP",
        help="Exclude a mutable architecture/HP-rank lane from this tranche.",
    )
    ap.add_argument(
        "--only-arch-hp", action="append", default=[], metavar="ARCH:HP",
        help="Render only the selected architecture/HP-rank lanes.",
    )
    ap.add_argument(
        "--name-suffix", default="v1",
        help="Unique Kubernetes Job/file suffix (default: v1).",
    )
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", args.name_suffix):
        raise SystemExit("--name-suffix must be a lowercase DNS label fragment")
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("format_version") != "condition-matched-eval-inventory.v1":
        raise SystemExit("unexpected inventory format")
    if inventory.get("rejected"):
        raise SystemExit(
            f"inventory has {len(inventory['rejected'])} rejected paths; adjudicate first")
    manifest_path = ROOT / "evaluation/stimuli/null-subj-v2-matched-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("vetted") is not True:
        raise SystemExit("matched-stimulus manifest is not gold-vetted")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    excluded = set()
    if args.exclude_run_ids:
        excluded = set(json.loads(args.exclude_run_ids.read_text()))
    excluded_arch_hp = set()
    for value in args.exclude_arch_hp:
        arch, hp = value.rsplit(":", 1)
        if arch not in ARCHES:
            raise SystemExit(f"unknown architecture in --exclude-arch-hp: {arch}")
        excluded_arch_hp.add((arch, int(hp)))
    only_arch_hp = set()
    for value in args.only_arch_hp:
        arch, hp = value.rsplit(":", 1)
        if arch not in ARCHES:
            raise SystemExit(f"unknown architecture in --only-arch-hp: {arch}")
        only_arch_hp.add((arch, int(hp)))
    overlap = excluded_arch_hp & only_arch_hp
    if overlap:
        raise SystemExit(
            "lanes cannot be both included and excluded: "
            + ", ".join(f"{arch}:h{hp}" for arch, hp in sorted(overlap))
        )
    runs_by_arch = {arch: [] for arch in ARCHES}
    selected_checkpoint_count = 0
    for run in inventory["runs"]:
        if (run["run_id"] not in excluded and
                (run["architecture"], int(run["hp_rank"])) not in excluded_arch_hp and
                (not only_arch_hp or
                 (run["architecture"], int(run["hp_rank"])) in only_arch_hp)):
            runs_by_arch[run["architecture"]].append(run["run_id"])
            selected_checkpoint_count += int(run["checkpoint_count"])
    for ids in runs_by_arch.values():
        ids.sort()
    selected_arches = [arch for arch in ARCHES if runs_by_arch[arch]]
    if not selected_arches:
        raise SystemExit("selection contains no runs")
    if not only_arch_hp and len(selected_arches) != len(ARCHES):
        raise SystemExit("inventory is missing at least one architecture")

    launcher = load_launcher()
    git_ref = launcher._resolve_git_ref()
    jobs = []
    summary = {"git_ref": git_ref, "benchmark": BENCHMARK,
               "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
               "stimuli_manifest_sha256": manifest_sha256,
               "inventory_runs": inventory["run_count"],
               "runs": sum(len(ids) for ids in runs_by_arch.values()),
               "inventory_checkpoints": inventory["checkpoint_count"],
               "checkpoints": selected_checkpoint_count,
               "excluded_mutable_runs": sorted(excluded),
               "excluded_arch_hp_lanes": sorted(
                   f"{arch}:h{hp}" for arch, hp in excluded_arch_hp),
               "included_arch_hp_lanes": sorted(
                   f"{arch}:h{hp}" for arch, hp in only_arch_hp),
               "name_suffix": args.name_suffix,
               "architectures": {}}
    for arch in selected_arches:
        job = render_job(launcher, arch, runs_by_arch[arch], git_ref,
                         manifest_sha256, args.name_suffix)
        jobs.append(job)
        dump(args.output_dir /
             f"job-eval-{SHORT[arch]}-{args.name_suffix}.yaml", [job])
        summary["architectures"][arch] = {
            "runs": len(runs_by_arch[arch]),
            "indexed_gpu_pods": math.ceil(len(runs_by_arch[arch]) / PACK[arch]),
            "pack": PACK[arch], "parallelism": PARALLELISM[arch],
        }
    dump(args.output_dir / f"job-eval-all-{args.name_suffix}.yaml", jobs)
    (args.output_dir / "fleet_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
