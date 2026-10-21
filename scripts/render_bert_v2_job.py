#!/usr/bin/env python3
"""Render k8s/bert_v2/job-sweep-bert-v2.yaml for one BERT-v2 sweep (D15).

    python scripts/render_bert_v2_job.py --name pilot-en \\
        --sweep-id thmorton-uc-san-diego/subject-drop-bert-v2/<id> \\
        --base-config configs/sweeps/baselines/bert_large_preln_en_race.yaml \\
        --condition baseline --gpus 24g --completions 1 > /tmp/j.yaml

GPU pools (nvidia.com/gpu, FA2-era Ampere/Ada; null-subjects has no A100
quota): 24g = 3090/A10/L4/4090 + 48 GB L40/L40S, 48g = L40/L40S only (the
rater race's 128 x 128 micro-batch needs it).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bad_nodes import load as load_bad_nodes  # noqa: E402  configs/bad_nodes.txt

TEMPLATE = REPO / "k8s" / "bert_v2" / "job-sweep-bert-v2.yaml"
POOLS = {
    "24g": ["NVIDIA-GeForce-RTX-3090", "NVIDIA-A10", "NVIDIA-L4", "NVIDIA-GeForce-RTX-4090",
            "NVIDIA-L40", "NVIDIA-L40S"],
    "48g": ["NVIDIA-L40", "NVIDIA-L40S"],
}


def render(name: str, sweep_id: str, base_config: str, condition: str, corpus_dir: str,
           gpus: str, parallelism: int, completions: int,
           gpu_key: str = "gpu", priority: str = "armada-default") -> dict:
    """gpu_key "gpu" uses the --gpus product pool. Any other key (a100, h100,
    h200, rtxa6000, rtx8000) requests nvidia.com/<key> with no product
    affinity; a100/h100/h200 need priority opportunistic or opportunistic2
    (NRP: quota-free there, preemptible by owners)."""
    if not (REPO / base_config).is_file():
        raise SystemExit(f"base config not found: {base_config}")
    if sweep_id.count("/") != 2:
        raise SystemExit(f"--sweep-id must be <entity>/<project>/<id>, got {sweep_id!r}")
    job = yaml.safe_load(TEMPLATE.read_text())
    job["metadata"]["name"] = f"thomas-bert-v2-{name}"
    spec = job["spec"]
    spec["parallelism"], spec["completions"] = parallelism, completions
    pod = spec["template"]["spec"]
    terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    exprs = terms["nodeSelectorTerms"][0]["matchExpressions"]
    for m in exprs:
        if m["key"] == "nvidia.com/gpu.product":
            m["values"] = POOLS[gpus]
        elif m["key"] == "kubernetes.io/hostname":
            m["values"] = load_bad_nodes() or ["none"]
    if gpu_key != "gpu":
        terms["nodeSelectorTerms"][0]["matchExpressions"] = [
            m for m in exprs if m["key"] != "nvidia.com/gpu.product"]
        res = pod["containers"][0]["resources"]
        for side in ("requests", "limits"):
            res[side].pop("nvidia.com/gpu", None)
            res[side][f"nvidia.com/{gpu_key}"] = 1
        if gpu_key in ("a100", "h100", "h200") and not priority.startswith("opportunistic"):
            raise SystemExit(f"nvidia.com/{gpu_key} has quota 0 here; use --priority opportunistic2")
    pod["priorityClassName"] = priority
    if priority.startswith("opportunistic"):
        # Preemptions must not burn retries (NRP Priority Classes doc: run as a
        # Job so the controller restarts it).
        spec["podFailurePolicy"]["rules"].insert(
            0, {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]})
    env = {e["name"]: e for e in pod["containers"][0]["env"]}
    env["CONDITION"]["value"] = condition
    env["CORPUS_DIR"]["value"] = corpus_dir
    env["BASE_CONFIG"]["value"] = base_config
    env["SWEEP_ID"]["value"] = sweep_id
    rendered = [job["metadata"]["name"], *(e.get("value", "") for e in env.values())]
    for m in terms["nodeSelectorTerms"][0]["matchExpressions"]:
        rendered += m["values"]
    if any("REPLACE" in str(v) for v in rendered):
        raise SystemExit("template placeholder left unrendered")
    return job


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--sweep-id", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--corpus-dir", default="")
    ap.add_argument("--gpus", choices=sorted(POOLS), default="24g")
    ap.add_argument("--parallelism", type=int, default=1)
    ap.add_argument("--completions", type=int, default=1)
    ap.add_argument("--gpu-key", default="gpu",
                    help="gpu (product pool) or a100/h100/h200/rtxa6000/rtx8000")
    ap.add_argument("--priority", default="armada-default",
                    choices=["armada-default", "opportunistic", "opportunistic2"])
    a = ap.parse_args(argv)
    job = render(a.name, a.sweep_id, a.base_config, a.condition, a.corpus_dir, a.gpus,
                 a.parallelism, a.completions, a.gpu_key, a.priority)
    sys.stdout.write(yaml.safe_dump(job, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
