# Kubernetes Job Workflow

This project runs heavy training, preprocessing, annotation, and analysis
workloads on the [National Research Platform (NRP)](https://nationalresearchplatform.org/)
Nautilus cluster. All manifests live under `k8s/` (~30 files). This page
explains how the workflow fits together, groups the manifests by purpose,
and shows common operations.

For registry / image-push setup, see
[`NRP_REGISTRY_SETUP.md`](NRP_REGISTRY_SETUP.md). For training checkpoint
policy, see [`checkpoint_scheduling.md`](checkpoint_scheduling.md).

## 1. Overview

### The NRP cluster

NRP Nautilus is a shared Kubernetes cluster with GPU nodes (T4 / A10 / A100).
This project runs in the `lemn-lab` namespace. Almost every job in `k8s/`
targets this cluster.

### General workflow

1. **Build / pick an image.** Most jobs use stock public images
   (`python:3.10-slim`, `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`,
   `nvidia/cuda:11.8.0-runtime-ubuntu22.04`). Only the sweep image is
   custom — see `k8s/Dockerfile.sweep`.
2. **Clone the repo at runtime.** Most manifests use an `alpine/git` init
   container that shallow-clones `multi-model-foundry-subject-rearing` into
   `/opt/repo` on an `emptyDir` volume. A few newer jobs (notably
   `job-europarl-sweep.yaml`) skip this and expect the repo to already be
   on the data PVC at `/mnt/data/repo`.
3. **Install deps at startup.** The main container typically runs
   `pip install --no-cache-dir spacy pydantic tqdm pyyaml pyarrow polars`
   and, for Python NLP jobs, `python -m spacy download <model>`.
4. **Submit the job.** `kubectl apply -f k8s/job-foo.yaml`.
5. **Monitor.** `kubectl logs -f job/foo` and `kubectl get pods`.
6. **Collect results** from the shared PVC via a scratch data-access pod
   (see [Retrieving output](#retrieving-output)).

### Where data lives

All persistent data is on ReadWriteMany PVCs, mounted at `/mnt/data` in
every job. The three active PVCs are:

| PVC | Manifest | Used by |
|---|---|---|
| `corpus-analysis-data` | `k8s/pvc.yaml` | Corpus analysis, annotation, tree detector, reports |
| `corpus-annotate-90m-data` | `k8s/pvc-annotate-90m.yaml` | Layered annotation output for `train_90M` |
| `pronoun-sweep-data` | `k8s/pvc-sweep.yaml` | Pronoun recovery sweeps, focal tests |
| `europarl-sweep-data` | `k8s/pvc-europarl-sweep.yaml` | Europarl sweep + TTQ data gen |

Within `/mnt/data` the layout is roughly:

```
/mnt/data/
  repo/                          # optional — used by europarl-sweep
  raw/                           # input corpora
  output/<split>/                # analyzer + annotator outputs
    annotated_corpus/{base,layers}
  pronoun_recovery/
    europarl_aligned/            # parallel alignment features
    tree_detector/               # detector training data + models
    models/europarl_sweep/       # sweep runs
```

## 2. Job categories

### Training / sweeps (GPU)

Pronoun-recovery and probe training, including hyperparameter sweeps.
Sweeps use indexed jobs so each pod picks up one cell of the grid.

| Manifest | Purpose |
|---|---|
| `k8s/job-europarl-sweep.yaml` | 27-way indexed sweep over Europarl-aligned features. Custom PyTorch image, 1 GPU per pod. Recently simplified (see [Anatomy](#3-anatomy-of-a-job)). |
| `k8s/job-sweep-r1.yaml` | 15-agent pronoun recovery sweep via W&B (`wandb-secret-thomas` secret), shares a 15-run grid. |
| `k8s/job-sweep-weight-alpha.yaml` | 5-agent variant sweeping the weight/alpha axis of the same grid (3 runs per agent). |
| `k8s/job-focal-test.yaml` | Single-pod focal-loss diagnostic run on `pronoun-sweep-data`. |

### Preprocessing / corpus build

Jobs that prepare parallel or alignment data consumed by later sweeps.

| Manifest | Purpose |
|---|---|
| `k8s/job-europarl-data-gen.yaml` | Builds Europarl-aligned feature tables for the pronoun recovery tree detector. Non-GPU. |
| `k8s/job-ttq-data-gen.yaml` | Generates TTQ (Tatoeba Translation Quality) aligned pairs. Uses CUDA image for GPU-accelerated alignment (see `docs/ttq_corpus_report.md`). |

### Corpus analysis (CPU)

Run the `analysis.corpus_descriptives.run` pipeline in analyzer mode over
a CHILDES split. Input on `corpus-analysis-data`.

| Manifest | Purpose |
|---|---|
| `k8s/job-train-90m.yaml` | Analyzer mode, `train_90M`. 4 CPU / 4Gi, `en_core_web_lg`. |
| `k8s/job-test-10m.yaml` | Analyzer mode, `test_10M`. |
| `k8s/job-pull-10m.yaml` | Analyzer mode, `pull_10M`. |
| `k8s/job-analysis-train-90m.yaml`, `k8s/job-analysis-test-10m.yaml` | Secondary analysis pass (downstream of layered annotations). |

Note: `job-train-90m.yaml` and `job-analysis-train-90m.yaml` both declare
`metadata.name: corpus-analysis-train-90m`. Only one can run at a time;
delete the previous Job before submitting the other.

### Layered annotation (English)

Annotator mode writes per-sentence layered Parquet into
`/mnt/data/output/<split>/annotated_corpus/`. See
`docs/LAYERED_ANNOTATION_ARCHITECTURE.md`.

| Manifest | Purpose |
|---|---|
| `k8s/job-annotate-train-90m.yaml` | `train_90M` layered annotation. Uses the `nvidia/cuda` image for `en_core_web_trf` on GPU; writes to the separate `corpus-annotate-90m-data` PVC. |
| `k8s/job-annotate-test-10m.yaml` | `test_10M` layered annotation. |

### Italian annotation pipeline

Recent addition — see [Section 5](#5-italian-annotation-pipeline) for
details.

| Manifest | Purpose |
|---|---|
| `k8s/job-annotate-smoke-test-it.yaml` | Tiny smoke test for the Italian annotators. |
| `k8s/job-annotate-test-10m-it.yaml` | `it_test_10M` annotation. |
| `k8s/job-annotate-train-90m-it.yaml` | `it_train_90M` annotation — indexed across 8 source files. |
| `k8s/job-annotate-spgc-it.yaml` | Standalone run for just the SPGC file with higher memory and smaller spaCy batch (long web documents). |

### Pronoun recovery / tree detector

Train and apply the Italian null-subject tree detector on annotated
corpora (see `docs/pronoun_recovery.md`).

| Manifest | Purpose |
|---|---|
| `k8s/job-tree-detect-it-train90m.yaml` | Indexed (`completions: 8, parallelism: 8`), one pod per source file (`childes`, `clta`, `corpus_isacco`, `europarl`, `leipzig_web`, `paccss`, `qcri`, `spgc`). Applies gradient-boosted tree detector at batch 256. |

### Analysis / reporting

| Manifest | Purpose |
|---|---|
| `k8s/job-null-subject-report-90m.yaml` | Runs `generate_null_subject_report.py` across layered annotations on `corpus-annotate-90m-data` and writes outputs back to `corpus-analysis-data`. |

### Utility pods

Long-running scratch pods for `kubectl cp` data transfer and interactive
debugging. Delete them when done.

| Manifest | Purpose |
|---|---|
| `k8s/pod-data-access.yaml` | Sleep pod bound to `corpus-analysis-data`. |
| `k8s/pod-sweep-data-access.yaml` | Sleep pod bound to `pronoun-sweep-data`. |

## 3. Anatomy of a job

Walkthrough of `k8s/job-europarl-sweep.yaml` — the most recently tweaked
manifest and a good reference for GPU indexed jobs.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: europarl-sweep
  namespace: lemn-lab
spec:
  completionMode: Indexed
  completions: 27
  parallelism: 27
  backoffLimit: 54
  backoffLimitPerIndex: 3
```

- **`kind: Job`** — one-shot batch workload (vs `Deployment`). The
  controller creates pods until `completions` successes are observed.
- **Indexed mode** — `completionMode: Indexed` assigns each pod a unique
  `JOB_COMPLETION_INDEX` env var (0..26). The runner reads it to pick its
  slice of the sweep grid. See `scripts/run_europarl_sweep.py`.
- **`parallelism: 27`** runs all 27 cells at once. Reduce it to throttle
  GPU usage (e.g. when quota is tight).
- **`backoffLimitPerIndex: 3`** — per-cell retry cap; a single bad cell
  can't consume the global `backoffLimit`.

```yaml
      containers:
      - name: sweep-runner
        image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e
          pip install --no-cache-dir \
            -r /mnt/data/repo/requirements.txt \
            accelerate>=0.26.0 python-dotenv \
            scikit-learn>=1.3.0 seqeval>=1.2.2 sentencepiece
          cd /mnt/data/repo
          python -u scripts/run_europarl_sweep.py
```

- **Image** — stock PyTorch runtime. Project deps are pip-installed at
  startup on top of it (saves rebuilding a custom image for every change).
- **No git-pull init container.** Unlike most other jobs in this repo,
  this manifest expects `/mnt/data/repo` to already exist on the PVC (kept
  in sync manually via a data-access pod). This was a deliberate
  simplification — see "Recent changes" below.

```yaml
        env:
        - name: WANDB_MODE
          value: "disabled"
        - name: PYTHONPATH
          value: "/mnt/data/repo"
        - name: SWEEP_DATA_DIR
          value: "/mnt/data/pronoun_recovery/europarl_aligned"
        - name: SWEEP_OUTPUT_DIR
          value: "/mnt/data/pronoun_recovery/models/europarl_sweep"
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "expandable_segments:True"
```

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** reduces CUDA
  fragmentation for variable-batch workloads; set it on every GPU job
  running variable-length NLP batches.
- **`WANDB_MODE=disabled`** — sweep runs locally and writes directly to
  `SWEEP_OUTPUT_DIR`. Other sweeps (`job-sweep-r1.yaml`,
  `job-sweep-weight-alpha.yaml`) re-enable W&B via `WANDB_API_KEY` pulled
  from the `wandb-secret-thomas` secret.

```yaml
        resources:
          requests:
            memory: 4Gi
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: 4Gi
            cpu: "2"
            nvidia.com/gpu: 1
```

- **Equal request/limit** is required for GPU pods on NRP (quota is
  enforced on requests, and NRP's 1.2x limit/request ratio policy applies
  to CPU/memory).
- **`nvidia.com/gpu: 1`** — one GPU per pod. With `parallelism: 27` this
  is 27 GPUs in flight; you will get scheduled as quota permits.

```yaml
        volumeMounts:
        - name: data
          mountPath: /mnt/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: europarl-sweep-data
      restartPolicy: Never
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

- **`restartPolicy: Never`** — on failure the Job controller creates a
  fresh pod rather than restarting the container; required for indexed
  jobs.
- **GPU toleration** lets the pod land on tainted GPU nodes.

### Recent changes (commit `4d0b18e`)

- **Dropped memory request 8Gi → 4Gi.** The sweep's working set fits in
  4 GB once intermediates are freed during data loading (earlier commits
  `296eb5c` and `6e11186` tuned this further).
- **Removed the `alpine/git` init container** that used to clone the
  repo into an `emptyDir`. The sweep now reads the repo directly from
  `/mnt/data/repo`, which the user keeps current from a data-access pod.

## 4. Common operations

### Submit a job

```bash
kubectl apply -f k8s/job-europarl-sweep.yaml
```

### Check status

```bash
kubectl get jobs
kubectl get pods -l job-name=europarl-sweep
kubectl describe job/europarl-sweep
```

For indexed jobs, pod names are suffixed with the completion index,
e.g. `europarl-sweep-0-xxxxx`.

### Tail logs

```bash
# Whole job (streams from the first matching pod)
kubectl logs -f job/europarl-sweep

# A specific indexed pod
kubectl logs -f europarl-sweep-0-xxxxx
```

### Cancel / clean up

```bash
kubectl delete job europarl-sweep
```

If a manifest has `ttlSecondsAfterFinished` set (e.g.
`job-null-subject-report-90m.yaml`, `job-analysis-train-90m.yaml`) the
Job is reaped automatically after 24 h.

### Debug a stuck pod

```bash
kubectl describe pod europarl-sweep-0-xxxxx      # events, scheduling, OOMKill
kubectl logs europarl-sweep-0-xxxxx --previous   # last container's logs
kubectl exec -it europarl-sweep-0-xxxxx -- bash  # shell in, if running
```

### Adjust parallelism / resources

For a quick throttle without editing the manifest:

```bash
kubectl scale job europarl-sweep --parallelism=4   # NOTE: not supported on all versions
```

Most of the time, edit the YAML and re-apply — but note you must
`kubectl delete job` first because `spec.template` is immutable on
existing Jobs.

### Retrieving output

```bash
kubectl apply -f k8s/pod-data-access.yaml
kubectl wait --for=condition=Ready pod/data-access --timeout=60s
kubectl cp data-access:/mnt/data/output/ ./analysis/output/corpus_descriptives/data/
kubectl delete pod data-access
```

## 5. Italian annotation pipeline

Recent addition (2026-02). The Italian annotators produce layered
sentence-level Parquet for the pronoun recovery tree detector.

### `k8s/job-annotate-spgc-it.yaml`

Standalone run for the SPGC Italian sub-corpus (web-scale long documents).
Key characteristics:

- **Image**: `python:3.10-slim`; installs `spacy pydantic tqdm pyyaml
  pyarrow polars` and downloads `it_core_news_lg`.
- **Resources**: 16 Gi memory, 7 CPU. Higher than the other Italian
  annotators because SPGC documents are long enough to blow through the
  default spaCy batch budget. Memory was tuned based on a smoke test
  (commit `aadcdf7` dropped another pod to 8Gi after the smoke test).
- **Config tweaks at startup**:
  ```bash
  sed -i 's/spacy_batch_size: 256/spacy_batch_size: 64/' \
    configs/analysis/corpus/corpus_analysis_it_train90m.yaml
  sed -i 's/chunk_size: 5000/chunk_size: 1000/' \
    configs/analysis/corpus/corpus_analysis_it_train90m.yaml
  ```
  These run-time overrides avoid having to maintain a separate config
  file just for SPGC.
- **Entry point**:
  ```bash
  python -m analysis.corpus_descriptives.run \
    --config configs/analysis/corpus/corpus_analysis_it_train90m.yaml \
    --annotate --layered --file spgc
  ```
- **Output** lands on `corpus-analysis-data` under
  `/mnt/data/output/it_train_90M/annotated_corpus/`, where
  `job-tree-detect-it-train90m.yaml` picks it up.

The bulk Italian annotation happens in
`k8s/job-annotate-train-90m-it.yaml` (indexed, 8 files). SPGC was split
off because it is slower and memory-hungrier than the other sources.

Related code: Italian-specific annotators live under
`analysis/pronoun_recovery/` and `analysis/corpus_descriptives/`. OOM
fixes for large files (lazy spaCy doc iteration) landed in commit
`aa88ff7`.

## 6. Sweep dashboard

`review/sweep_dashboard.html` is a static HTML viewer for sweep results.
Drop the sweep output JSONs next to it (or point it at a directory under
`review/data/`) and open in a browser — no server needed. Useful as a
lightweight alternative to W&B for the Europarl sweep, which runs with
`WANDB_MODE=disabled`.

## 7. Troubleshooting

### OOMKilled (exit code 137)

`kubectl describe pod <pod>` will show `Reason: OOMKilled`. Options:

- Bump `resources.requests.memory` and `limits.memory` in the manifest.
  NRP enforces a ~1.2x limit/request ratio; set them equal for GPU pods
  (see commit `eff42c1`).
- Reduce batch / chunk sizes. The `sed` overrides in
  `job-annotate-spgc-it.yaml` (`spacy_batch_size: 256 → 64`,
  `chunk_size: 5000 → 1000`) are a template.
- For annotation jobs on large files, make sure you're on commit
  `aa88ff7` or later, which iterates spaCy docs lazily instead of
  materialising the whole corpus in memory.
- The Europarl sweep itself went 12Gi → 8Gi → 4Gi as intermediates were
  freed during data loading (commits `296eb5c`, `4d0b18e`).

### GPU unavailable / Pending forever

- `kubectl describe pod` will show `Insufficient nvidia.com/gpu` or a
  taint mismatch. Check namespace GPU quota:
  ```bash
  kubectl get resourcequota -n lemn-lab
  ```
- Confirm the toleration block is present:
  ```yaml
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  ```
- For indexed sweeps, lower `parallelism` to fit within quota.

### ImagePullBackOff / ErrImagePull

- Double-check the image tag exists on the registry. Public images
  (`python:3.10-slim`, `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`)
  occasionally go stale; pick a newer tag.
- For the custom sweep image built from `k8s/Dockerfile.sweep`, follow
  [`NRP_REGISTRY_SETUP.md`](NRP_REGISTRY_SETUP.md) to push a new tag and
  update the manifest.
- `kubectl get events --sort-by=.lastTimestamp | tail` gives the real
  pull error (403 = auth, manifest unknown = bad tag).

### PVC stuck / ReadWriteOnce conflicts

`corpus-analysis-data` was originally ReadWriteOnce, which means only one
pod can mount it at a time. If a new job sits in `ContainerCreating`,
check for a lingering data-access pod from a previous session and
`kubectl delete pod` it.
