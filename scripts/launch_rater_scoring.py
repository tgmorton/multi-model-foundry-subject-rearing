#!/usr/bin/env python3
"""Re-rate the corpus with one of our own raters: Job YAML for scoring every subject pronoun.

Each rater scores its own condition's version of the text (the intervention applied to the
baseline-identified pronouns, `score_pronoun_recoverability.py --intervention`), with [SEP]
after every line, the layout the rater was trained on, and 250 pieces of context each side.

The rater's final checkpoint is read from S3 (put there by
k8s/bert_v2/job-upload-rater-final.yaml); the annotation cache and raw text come from the
lemn-lab volume. Scores go to the volume and to

    s3://thomas-subject-drop-artifacts/recoverability/rater_scores/rater_<short>/<corpus>/

Prints the YAML (three Jobs by memory class); pipe it to `kubectl apply -f -`.

    python scripts/launch_rater_scoring.py --rater enrichvm | kubectl apply -f -
    python scripts/launch_rater_scoring.py --rater enrichvm --corpora test_10M --suffix t1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bad_nodes  # noqa: E402

# short name -> (rater corpus name, --intervention value)
RATERS = {
    "base": ("base", "baseline"),
    "enrichvm": ("enrich_verbal_morphology", "enrich_verbal_morphology"),
    "impcase": ("impoverish_case_en", "impoverish_case"),
    "lemverb": ("lemmatize_verbs", "lemmatize_verbs"),
    "rmexpl": ("remove_expletive_sentences_en", "remove_expletive_sentences"),
}
# (corpus, genre) -> (number of shards, memory class). Shards sized from the 2026-08 BERT-large
# pass (60-140 instances/s): each is about 2.5 h of scoring or less.
PLAN = {
    ("train_90M", "childes"): (3, "lg"), ("train_90M", "open_subtitles"): (4, "lg"),
    ("train_90M", "gutenberg"): (3, "md"), ("train_90M", "bnc_spoken"): (2, "md"),
    ("train_90M", "simple_wiki"): (1, "md"), ("train_90M", "switchboard"): (1, "md"),
    **{("pull_10M", g): (1, "sm") for g in
       ("childes", "open_subtitles", "gutenberg", "bnc_spoken", "simple_wiki", "switchboard")},
    **{("test_10M", g): (1, "sm") for g in
       ("childes", "open_subtitles", "gutenberg", "bnc_spoken", "simple_wiki", "switchboard")},
}
MEMORY = {"lg": "24Gi", "md": "12Gi", "sm": "8Gi"}
GPUS = ["NVIDIA-GeForce-RTX-3090", "NVIDIA-A10", "NVIDIA-L4", "NVIDIA-GeForce-RTX-4090", "NVIDIA-L40", "NVIDIA-L40S",
        "NVIDIA-RTX-A5000", "NVIDIA-RTX-A4000"]

SCRIPT = r'''set -euo pipefail
TASKS=(__TASKS__)
read -r CORPUS GENRE SHARD <<< "${TASKS[$JOB_COMPLETION_INDEX]}"
echo "=== rater_$SHORT re-rating: $CORPUS/$GENRE shard $SHARD under $IV on ${NODE_NAME:-?} ==="
python3 -c "
import sys, torch
try:
    assert torch.cuda.is_available(); torch.zeros(1).cuda(); print('  GPU healthy:', torch.cuda.get_device_name(0))
except Exception as e:
    print('FATAL: no CUDA:', e); sys.exit(2)
"
export PYTHONPATH="/opt/repo:${PYTHONPATH:-}"
cd /opt/repo
python - <<PYDL
import boto3, pathlib
P = "raters/rater-bert_large_preln-en-$RATER_CORPUS-full-v1/checkpoint-23000/"
d = pathlib.Path("/tmp/rater"); d.mkdir(exist_ok=True)
s3 = boto3.client("s3")
for o in s3.list_objects_v2(Bucket="thomas-subject-drop-artifacts", Prefix=P)["Contents"]:
    s3.download_file("thomas-subject-drop-artifacts", o["Key"], str(d / o["Key"][len(P):]))
print("rater files:", sorted(p.name for p in d.iterdir()))
PYDL
python scripts/score_pronoun_recoverability.py \
  --corpus "$CORPUS" --file "$GENRE" \
  --hf-mlm "rater_$SHORT=/tmp/rater" \
  --mlm-ctx 250 --mlm-batch 24 --line-sep sep \
  --intervention "$IV" --shard "$SHARD" --amp
# Self-check and upload. A shard with NaN scores exits 1 so the index retries on another pod.
python - <<PYUP
import boto3, pathlib, sys, pyarrow.parquet as pq
d = pathlib.Path("/mnt/data/recoverability/$CORPUS/external_rater_$SHORT/intervention/$IV")
i, n = "$SHARD".split("/")
stem = "$GENRE.s%sof%s" % (i, n)
t = pq.read_table(d / (stem + ".parquet"), columns=["logprob_sum"]).to_pandas()
nan = t.logprob_sum.isna().mean()
print(f"NaN frac {nan:.4f} over {len(t)} (node $NODE_NAME)")
if nan > 0.001:
    sys.exit("FAIL: NaN scores, not uploading; the index will retry")
s3 = boto3.client("s3")
for p in list(d.glob(stem + ".*")) + list(d.glob("$GENRE.line_removed.parquet")):
    s3.upload_file(str(p), "thomas-subject-drop-artifacts", f"recoverability/rater_scores/rater_$SHORT/$CORPUS/{p.name}")
    print("uploaded", p.name)
PYUP
echo "SHARD $CORPUS/$GENRE $SHARD OK"
'''


def job(short: str, mem_class: str, tasks: list[str], suffix: str, parallelism: int, branch: str) -> str:
    rater_corpus, iv = RATERS[short]
    name = f"thomas-rerate-{short}-{mem_class}-{suffix}"
    script = SCRIPT.replace("__TASKS__", " ".join(f'"{t}"' for t in tasks))
    ind = " " * 10
    body = "\n".join(ind + ln if ln else "" for ln in script.splitlines())
    hosts = "\n".join(f"                - {h}" for h in bad_nodes.load())
    gpus = "\n".join(f"                - {g}" for g in GPUS)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: lemn-lab
  labels: &lab {{owner: thomas, study: subject-drop, lang: en, stage: score}}
spec:
  completionMode: Indexed
  completions: {len(tasks)}
  parallelism: {min(parallelism, len(tasks))}
  backoffLimit: 40
  ttlSecondsAfterFinished: 604800
  podFailurePolicy:
    rules:
    - action: Ignore
      onExitCodes: {{containerName: scorer, operator: In, values: [2, 128]}}
    - action: Ignore
      onPodConditions:
      - {{type: DisruptionTarget}}
  template:
    metadata:
      labels: *lab
    spec:
      priorityClassName: armada-default
      imagePullSecrets:
      - name: gitlab-registry-cred-thomas
      restartPolicy: Never
      tolerations:
      - {{key: nvidia.com/gpu, operator: Exists, effect: PreferNoSchedule}}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
{gpus}
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
{hosts}
      initContainers:
      - name: clone-repo
        image: alpine/git
        args: [clone, --single-branch, --depth=1, --branch={branch},
               "https://github.com/tgmorton/multi-model-foundry-subject-rearing.git", /opt/repo]
        resources:
          requests: {{memory: 1Gi, cpu: "200m"}}
          limits: {{memory: 1Gi, cpu: "200m"}}
        volumeMounts: [{{name: repo, mountPath: /opt/repo}}]
      containers:
      - name: scorer
        image: gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest
        imagePullPolicy: Always
        command: ["/bin/bash", "-c"]
        args:
        - |
{body}
        env:
        - {{name: SHORT, value: "{short}"}}
        - {{name: RATER_CORPUS, value: "{rater_corpus}"}}
        - {{name: IV, value: "{iv}"}}
        - {{name: PYTHONPATH, value: /opt/repo}}
        - {{name: HF_HOME, value: /tmp/hf}}
        - {{name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}}
        - name: NODE_NAME
          valueFrom: {{fieldRef: {{fieldPath: spec.nodeName}}}}
        - name: AWS_ACCESS_KEY_ID
          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_ACCESS_KEY_ID}}}}
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom: {{secretKeyRef: {{name: s3-secret-thomas, key: AWS_SECRET_ACCESS_KEY}}}}
        - {{name: AWS_ENDPOINT_URL, value: "http://rook-ceph-rgw-nautiluss3.rook"}}
        - {{name: AWS_DEFAULT_REGION, value: us-west-1}}
        resources:
          requests: {{memory: {MEMORY[mem_class]}, cpu: "2", nvidia.com/gpu: 1}}
          limits: {{memory: {MEMORY[mem_class]}, cpu: "2", nvidia.com/gpu: 1}}
        volumeMounts:
        - {{name: repo, mountPath: /opt/repo}}
        - {{name: data, mountPath: /mnt/data}}
        - {{name: shm, mountPath: /dev/shm}}
      volumes:
      - {{name: repo, emptyDir: {{}}}}
      - name: data
        persistentVolumeClaim: {{claimName: subject-drop-archive}}
      - name: shm
        emptyDir: {{medium: Memory, sizeLimit: 2Gi}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rater", required=True, choices=sorted(RATERS))
    ap.add_argument("--corpora", nargs="+", default=["train_90M", "pull_10M"], choices=["train_90M", "pull_10M", "test_10M"])
    ap.add_argument("--genres", nargs="+", default=None, help="restrict to these genres")
    ap.add_argument("--suffix", default="v1", help="job-name suffix")
    ap.add_argument("--parallelism", type=int, default=8, help="per Job")
    ap.add_argument("--branch", default="eval/bert-v2")
    a = ap.parse_args()
    by_class: dict[str, list[str]] = {}
    for (corpus, genre), (n, mc) in PLAN.items():
        if corpus in a.corpora and (a.genres is None or genre in a.genres):
            by_class.setdefault(mc, []).extend(f"{corpus} {genre} {i}/{n}" for i in range(n))
    docs = [job(a.rater, mc, tasks, a.suffix, a.parallelism, a.branch) for mc, tasks in sorted(by_class.items())]
    print("---\n".join(docs), end="")
    print(f"# {sum(len(t) for t in by_class.values())} pods in {len(docs)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
