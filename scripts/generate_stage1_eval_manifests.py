#!/usr/bin/env python3
"""Render the gated Stage-1 English early-seed evaluation fleet.

This generator is intentionally launch-free.  It writes one read-only
integrity audit, six architecture-specific packing smokes, and six production
Indexed Jobs.  Production manifests must not be submitted until their matching
packing smoke has passed its utilization and output-integrity gate.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "k8s" / "stage1" / "eval"
PINNED_IMAGE = (
    "gitlab-registry.nrp-nautilus.io/thmorton/"
    "multi-model-foundry-subject-rearing@"
    "sha256:037b88f45101490ba412890bf431cfefc8c93cba80c7731041649d59aba9a259"
)
SEEDS = [42, 137, 314159, 1568568120, 1415936399, 1640623142,
         1595274352, 2022192329, 1891911437, 1881877998, 302485963,
         2078344582]
ARCHES = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large",
          "lstm", "mamba_370m"]
CONDS = ["baseline", "remove_expletive_sentences", "impoverish_case",
         "lemmatize_verbs", "enrich_verbal_morphology"]
PACK = {"gpt2_small": 4, "gpt2_medium": 3, "gpt2_large": 2,
        "bert_large": 3, "lstm": 6, "mamba_370m": 2}
RAM = {"gpt2_small": "10Gi", "gpt2_medium": "12Gi",
       "gpt2_large": "16Gi", "bert_large": "12Gi", "lstm": "14Gi",
       "mamba_370m": "12Gi"}
PAR = {"gpt2_small": 4, "gpt2_medium": 4, "gpt2_large": 4,
       "bert_large": 4, "lstm": 4, "mamba_370m": 4}
SHORT = {"gpt2_small": "gpt2s", "gpt2_medium": "gpt2m",
         "gpt2_large": "gpt2l", "bert_large": "bertl", "lstm": "lstm",
         "mamba_370m": "mamba370m"}
BAD_NODES = ["gpu-14.nrp.mghpcc.org", "gpu-17.nrp.mghpcc.org",
             "nautilus-it-gpu01.fullerton.edu",
             "nautilus-it-gpu03.fullerton.edu",
             "rci-tide-gpu-03.sdsu.edu", "ry-gpu-10.sdsc.optiputer.net",
             "hcc-nrp-shor-c6017.unl.edu"]


def load_launcher():
    p = ROOT / "scripts" / "launch_cell_evals.py"
    spec = importlib.util.spec_from_file_location("eval_launcher", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def run_ids(arch: str) -> list[str]:
    return [f"{arch}-en-{cond}-h0-s{seed}" for cond in CONDS for seed in SEEDS]


def render_job(launcher, arch: str, smoke: bool) -> dict:
    pack = PACK[arch]
    ids = run_ids(arch)
    selected = ids[:pack] if smoke else ids
    n = 1 if smoke else math.ceil(len(ids) / pack)
    smoke_version = "v2"
    suffix = (f"s1e-pack-smoke-{SHORT[arch]}-{smoke_version}"
              if smoke else f"s1e-{SHORT[arch]}-en-v1")
    output = f"/mnt/data/eval_v2/stage1_pack_smoke/{SHORT[arch]}"
    extra = (f"--no-registry --results-to-pvc --output_root {output} "
             f"--scoring_version stage1-pack-smoke-{SHORT[arch]}-{smoke_version}"
             if smoke else "--scoring_version null-subj-v2-r1")
    text = launcher.JOB_TEMPLATE.format(
        name=f"thomas-eval-{suffix}", lang="en", completions=n,
        parallelism=1 if smoke else min(PAR[arch], n),
        gpu_values="\n".join(f"                - {g}" for g in launcher.GPU_POOL_24GB),
        bad_nodes="\n".join(f"                - {x}" for x in BAD_NODES),
        repo_url=launcher.REPO_URL, git_ref=launcher._resolve_git_ref(),
        image=PINNED_IMAGE, run_ids_json=json.dumps(selected), batch_size=64,
        pod_ram=RAM[arch], pod_cpu=str(pack), pack=pack, extra_args=extra,
    )
    job = yaml.safe_load(text)
    job["metadata"]["namespace"] = "lemn-lab"
    job["metadata"]["labels"].update({"arch": arch,
        "phase": "pack-smoke" if smoke else "production"})
    job["spec"].pop("backoffLimit", None)
    job["spec"].update({"backoffLimitPerIndex": 2 if smoke else 100,
                         "maxFailedIndexes": 0,
                         "podReplacementPolicy": "Failed"})
    pod = job["spec"]["template"]["spec"]
    pod["automountServiceAccountToken"] = False
    pod["enableServiceLinks"] = False
    c = pod["containers"][0]
    c["imagePullPolicy"] = "IfNotPresent"
    # Disable canonical S3 output for smoke. Production keeps S3 canonical.
    if smoke:
        for e in c["env"]:
            if e.get("name") == "REGISTRY_BUCKET":
                e["value"] = ""
                e.pop("valueFrom", None)
    # Continuous bounded telemetry: nvidia-smi exits when the evaluator shell
    # exits; the trap prevents an orphan sampler.
    marker = "cd /opt/repo\n"
    telemetry = (marker +
        "          nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory "
        "--format=csv,noheader,nounits -l 15 &\n"
        "          TELEMETRY_PID=$!\n"
        "          trap 'kill $TELEMETRY_PID 2>/dev/null || true' EXIT\n")
    c["args"][0] = c["args"][0].replace(marker, telemetry, 1)
    if "nvidia-smi --query-gpu" not in c["args"][0]:
        raise RuntimeError(f"failed to inject telemetry for {arch}")
    return job


def audit_job() -> dict:
    targets = [(a, c, f"{a}-en-{c}-h0-s{s}")
               for a in ARCHES for c in CONDS for s in SEEDS]
    code = f'''import json, os\nfrom pathlib import Path\nimport boto3\nfrom botocore.exceptions import ClientError\ntargets=json.loads({json.dumps(json.dumps(targets))})\ns3=boto3.client("s3",endpoint_url=os.environ["AWS_ENDPOINT_URL"],region_name=os.environ["AWS_DEFAULT_REGION"])\nbucket=os.environ["REGISTRY_BUCKET"]\nmissing=[]; bad=[]; eval_done=0\nfor arch,cond,rid in targets:\n p=Path("/mnt/data/models/production")/rid\n if not p.is_dir(): missing.append(("PVC",rid)); continue\n ck=sorted(p.glob("checkpoint-*"))\n if not ck: bad.append(("NO_CHECKPOINTS",rid)); continue\n key=f"run_registry/by_run/{{arch}}/en/{{cond}}/{{rid}}.json"\n try: rec=json.loads(s3.get_object(Bucket=bucket,Key=key)["Body"].read())\n except ClientError: missing.append(("REGISTRY",rid)); continue\n if rec.get("status") != "COMPLETE": bad.append(("REGISTRY_STATUS="+str(rec.get("status")),rid))\n pair=f"eval_results/null_subj_v2/pairs/cell_id={{rid}}.parquet"\n try: s3.head_object(Bucket=bucket,Key=pair); eval_done+=1\n except ClientError as e:\n  if e.response["Error"]["Code"] not in ("404","NoSuchKey"): raise\nprint(f"TRAINING_TARGETS={{len(targets)}} EVAL_ALREADY_COMPLETE={{eval_done}} EVAL_PENDING={{len(targets)-eval_done}}")\nfor x in missing+bad: print("AUDIT_ERROR",*x)\nif missing or bad: raise SystemExit(2)\nprint("STAGE1_EVAL_AUDIT_OK=360")'''
    return yaml.safe_load(f'''apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: thomas-fdy-s1e-eval-audit-en-v1\n  namespace: lemn-lab\n  labels: {{owner: thomas, study: subject-drop, stage: eval-audit}}\nspec:\n  backoffLimit: 0\n  activeDeadlineSeconds: 1800\n  ttlSecondsAfterFinished: 86400\n  template:\n    metadata:\n      labels: {{owner: thomas, study: subject-drop, stage: eval-audit}}\n    spec:\n      automountServiceAccountToken: false\n      enableServiceLinks: false\n      restartPolicy: Never\n      containers:\n      - name: audit\n        image: {PINNED_IMAGE}\n        command: ["python3", "-c"]\n        args: [{json.dumps(code)}]\n        env:\n        - name: AWS_ACCESS_KEY_ID\n          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_ACCESS_KEY_ID}}}}\n        - name: AWS_SECRET_ACCESS_KEY\n          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_SECRET_ACCESS_KEY}}}}\n        - {{name: AWS_ENDPOINT_URL, value: "http://rook-ceph-rgw-nautiluss3.rook"}}\n        - {{name: AWS_DEFAULT_REGION, value: "us-west-1"}}\n        - {{name: REGISTRY_BUCKET, value: "thomas-subject-drop-artifacts"}}\n        resources:\n          requests: {{cpu: 500m, memory: 2Gi}}\n          limits: {{cpu: 500m, memory: 2Gi}}\n        volumeMounts:\n        - {{name: data, mountPath: /mnt/data, readOnly: true}}\n      volumes:\n      - name: data\n        persistentVolumeClaim: {{claimName: subject-drop-archive, readOnly: true}}\n''')


def dump(path: Path, docs: list[dict]):
    path.write_text("---\n".join(yaml.safe_dump(x, sort_keys=False) for x in docs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    launcher = load_launcher()
    dump(OUT / "job-stage1-eval-audit-en.yaml", [audit_job()])
    smokes = [render_job(launcher, a, True) for a in ARCHES]
    dump(OUT / "job-stage1-eval-pack-smokes-en.yaml", smokes)
    for arch, job in zip(ARCHES, smokes):
        dump(OUT / f"job-stage1-eval-pack-smoke-{SHORT[arch]}-en.yaml", [job])
    productions = [render_job(launcher, a, False) for a in ARCHES]
    dump(OUT / "job-stage1-eval-production-en.yaml", productions)
    for arch, job in zip(ARCHES, productions):
        dump(OUT / f"job-stage1-eval-production-{SHORT[arch]}-en.yaml", [job])
    print(json.dumps({"audit_jobs": 1, "smoke_jobs": 6,
                      "production_jobs": 6,
                      "production_gpu_pods": sum(math.ceil(60/PACK[a]) for a in ARCHES),
                      "production_peak_gpus": sum(PAR.values())}, indent=2))


if __name__ == "__main__":
    main()
