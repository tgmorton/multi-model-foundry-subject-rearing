# Project CLAUDE.md

This file is loaded at the start of every Claude session in this repo.
It's the minimum priors a new agent needs to not re-derive things we
already decided. Detail lives in `docs/` and in project memory.

## What the project is

Controlled-rearing study of subject-drop in language models. Training
a matrix of (architecture × language × condition × seed) models on
90M-word corpora (English BabyLM + Spanish BebeLM), with 7 ablations
per (arch × lang) plus a baseline. The goal is to compare generative
linguistics vs information-theoretic accounts of subject-drop.

## Where we are

- **Optimization stack** (phase 0.2–0.6, 1.1–1.3) landed and verified:
  content-addressed data cache, fused AdamW, tensor-loss (one sync per
  optimizer step, not per micro-batch), save-policy split (last N
  checkpoints keep `training_state.pt`, rest are analysis-only),
  timing instrumentation, Inductor backend.
- **Custom GitLab image** (details in "Container image" section below)
  ships torch + requirements + flash-attn + mamba-ssm + causal-conv1d
  pre-installed. Pod startup went from ~300 s (pip install) to 5 s.
- **Registry substrate** in place: bucket, K8s secret, `registry.py`
  module, per-run JSON + materialized `registry.parquet` plan, write
  API wired into `cli.py::run` + training-loop heartbeat.
- **Smokes pass** for all 5 architectures (GPT-2 medium/large, BERT
  large, LSTM, Mamba 370M). First full production baseline finished
  (GPT-2 medium, English, 10 epochs, loss 3.15).

## Study parameters (locked)

| Parameter | Value |
|---|---|
| Architectures | n-gram 1–5, GPT-2 small/medium/large, BERT large, LSTM, Mamba 370M |
| Languages | English + Spanish (Italian is dormant) |
| Conditions | baseline + 7 ablations per (arch, lang) |
| Seeds per cell | **30** |
| Epochs (production) | **30** |
| Checkpoints per run | **80**, must include anchors `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512`, then appropriate spacing |
| HP sweep horizon | **3 epochs** per trial |
| HP sweep metric | Both training loss (A) and held-out perplexity (B) — decide between after first round |
| HP sweep per language | **Separate sweeps** per language (no English→Spanish transfer) |

## Storage architecture

See `memory/reference_storage_diagram.md` for the canonical ASCII
picture. Summary:

- **CephFS PVC `corpus-analysis-data`** (200 GiB, RWX) — hot operational
  storage. Holds raw corpora, tokenizers, tokenized+chunked data,
  in-flight checkpoints.
- **CephFS PVC `subject-drop-archive`** (40 TiB, RWX) — cold
  reference-rep storage. Holds analysis-only checkpoints for the
  reference replicate of each cell after post-eval pruning.
- **Ceph RGW S3 bucket `thomas-subject-drop-artifacts`** — portable
  science outputs that travel with the paper. Registry, eval parquets,
  WandB exports, env snapshots.
- **Nautilus retention is 6 months of inactivity, not 6 months
  absolute.** Storage ceases to be a binding constraint during the
  active study. Post-publication migration is an institutional
  archiving question (Zenodo / OSF / UCSD service).

## Cluster / GPU allocation (everything needed to land a pod on the right hardware)

**Context**: `nautilus` · **Namespace**: `lemn-lab` (verify with
`kubectl config current-context` / `kubectl config view --minify -o
jsonpath='{..namespace}'`).

**Scale check** (measured 2026-04 via `kubectl get nodes`): ~1,500 GPUs
across 34 products on NRP. ~970 openly schedulable to us; another ~340
behind reservation/hardware tolerations. Cluster-wide pod listing is
blocked for our user, so for *live* utilization use the NRP dashboard
at `https://nrp.ai/viz/resources` (not `kubectl`).

**Re-derive the inventory at any time** (handy when planning a sweep):

```bash
kubectl get nodes -o json | jq -r '.items[] | [
    .metadata.name,
    (.metadata.labels["nvidia.com/gpu.product"] // "?"),
    ([.status.capacity | to_entries[]
      | select(.key|startswith("nvidia.com/"))
      | select(.key!="nvidia.com/gpu.present")
      | "\(.key)=\(.value)"] | join(","))
  ] | @tsv' | column -t
```

### Workload → GPU-tier cheat sheet

| Workload | Target pool | Open GPUs | Why |
|---|---|---:|---|
| **Production training** (baseline + HP sweeps) | **A100-SXM4-80GB** via reservation (`nvidia.com/a100`) | 83 | 3-4× RTX 3090 throughput, 80 GB headroom |
| Training without reservation | **48 GB Ampere/Ada**: A6000 + A40 + L40 + L40S | ~82 | FA2 ✅, 2× batch of 3090, no paperwork |
| Arch smoke / annotation / short fine-tunes | **24 GB Ampere/Ada**: RTX 3090 + A10 + L4 + RTX 4090 | ~400 | Biggest FA2-eligible pool; same VRAM |
| Tiny inference, eval-v2, light sweeps | **A100 MIG slice** (`nvidia.com/mig-small`) | 28 open | Hardware-isolated A100 slice, 10 GB, FA2 ✅ |
| Very small models / sweeps with reduced mem | **RTX A4000** (16 GB) | 32 | FA2 ✅, small footprint |

**Do NOT target** RTX 2080 Ti, GTX 1080 Ti, V100 (any variant), T4,
Titan Xp, Titan RTX, Quadro RTX 6000/8000, or A2 **with the current
trainer**. They're Turing / Volta / Pascal — FA2 will fail.
[model_foundry/trainer.py:129](model_foundry/trainer.py:129) hard-codes
`attn_implementation="flash_attention_2"`. The config plumbing for
`"eager"` fallback exists
([model_foundry/tests/unit/test_architectures.py:137](model_foundry/tests/unit/test_architectures.py:137))
but isn't exposed; plumb it through if you want to unlock the ~400 extra
pre-Ampere GPUs.

### Affinity / toleration patterns (copy-paste into a Job spec)

**(A) Open tier — most jobs.** Soft-taint toleration + 24 GB FA2 pool:

```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: PreferNoSchedule        # every GPU node has this soft taint
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: nvidia.com/gpu.product
              operator: In
              values:
                - NVIDIA-GeForce-RTX-3090
                - NVIDIA-A10
                - NVIDIA-L4
                - NVIDIA-GeForce-RTX-4090
resources:
  requests: { nvidia.com/gpu: 1 }
  limits:   { nvidia.com/gpu: 1 }
```

Swap the `values` list for `[NVIDIA-A40, NVIDIA-L40, NVIDIA-L40S]` if
you want the 48 GB FA2 pool (L40/L40S use `nvidia.com/gpu`; A40 needs
`nvidia.com/a40` instead — split into two job variants).

**(B) A100 (reservation-gated, special-resource request key).** Fill in
the reservation value admins assigned our group. Tolerating values you
aren't authorized for is against NRP policy
([docs/nrp-docs/running/special.md:17](docs/nrp-docs/running/special.md:17)):

```yaml
resources:
  requests: { nvidia.com/a100: 1 }   # NOT nvidia.com/gpu
  limits:   { nvidia.com/a100: 1 }

tolerations:
  - { key: nvidia.com/gpu, operator: Exists, effect: PreferNoSchedule }
  - key: nautilus.io/reservation
    operator: Equal
    value: "<OUR-GROUP>"             # confirm with admin
    effect: NoSchedule

affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: nautilus.io/reservation
              operator: In
              values: ["<OUR-GROUP>"]
            - key: nvidia.com/gpu.product
              operator: In
              values: ["NVIDIA-A100-SXM4-80GB"]   # pin to fastest tier
```

**(C) Other special-resource GPUs.** Request key differs from
`nvidia.com/gpu`:

| GPU | Request key | Extra needed? |
|---|---|---|
| A40 (45 GB) | `nvidia.com/a40` | Open |
| RTX A6000 (48 GB) | `nvidia.com/rtxa6000` | 37 open; more behind `erl-ucsd`/`sdccd` |
| MIG 1g.10gb | `nvidia.com/mig-small` | 28 open; more behind `sdsuinstruction` |
| Quadro RTX 8000 | `nvidia.com/rtx8000` | Open (but no FA2) |
| H200 (140 GB) | `nvidia.com/h200` | Reservation `csu-h200` |
| GH200 (96 GB) | `nvidia.com/gh200` | arm64 — needs matching image + `kubernetes.io/arch: arm64` selector |

**Never tolerate** `nautilus.io/issue=*` (broken nodes) or
`nautilus.io/system=*` (infrastructure — explicitly forbidden by
policy). `nautilus.io/hardware=<value>` is allowed only for values
admin-authorized for our group.

### FA2 eligibility (the rule)

Compute capability ≥ 8.0 (Ampere / Ada / Hopper). Everything in the
cheat-sheet table above qualifies. Hopper (H200/GH200) additionally
supports FA3. CUDA version is not a blocker — NRP nodes are on CUDA
13.x drivers, backward-compatible with our `cuda12.1`-based image.

### Existing job templates to copy from

- [k8s/job-train-baseline-90m-full.yaml](k8s/job-train-baseline-90m-full.yaml) — production training, strict RTX 3090 (candidate for broadening to 48 GB A-tier)
- [k8s/job-smoke-bert-large.yaml](k8s/job-smoke-bert-large.yaml) — arch smoke with flexible RTX-3090+A10 affinity
- [k8s/job-annotate-train-90m.yaml](k8s/job-annotate-train-90m.yaml) — 6-way indexed annotation with GPU
- [k8s/job-europarl-sweep.yaml](k8s/job-europarl-sweep.yaml) — 27-way indexed W&B sweep on tiny memory (4 Gi) — reference for MIG candidates
- [k8s/job-ttq-data-gen.yaml](k8s/job-ttq-data-gen.yaml) — the only job currently pinned to A10 (probably stale; safe to loosen)

### Other PVCs in the namespace (beyond the two canonical ones)

In addition to `corpus-analysis-data` and `subject-drop-archive`
documented under Storage, `lemn-lab` holds ~36 PVCs (~66 TiB) shared
with labmates. Ours include:

- `europarl-sweep-data` (500 Gi, RWX) — parallel-data sweeps + TTQ gen
- `pronoun-sweep-data` (50 Gi, RWX) — pronoun-recovery sweep runs
- `corpus-annotate-90m-data` (50 Gi, RWX) — English 90M annotation output
- `thomas-grace-model-cache` (250 Gi), `-llama-70b-cache` (400 Gi),
  `-deepseek-r1-70b-cache` (400 Gi), `-results` (50 Gi) — HF model/eval
  caches for Grace-connected work

List them all with `kubectl get pvc`. Don't use labmates' PVCs (anything
starting with `hongao-`, `yupeiwang-`, `asr-`, `adaptbpe-`, `tda-`,
`crac26-`, `uid-pass-act-`, `nk-`, `pengd`, `climb-`, `pvc-markjos`,
`whisper-*`) without asking.

## Container image (everything needed to launch a pod)

| Field | Value |
|---|---|
| Image | `gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest` |
| Registry | NRP-hosted GitLab registry, mirror of this repo |
| Pull secret (K8s) | `gitlab-registry-cred-thomas` (type `dockerconfigjson`) |
| Build trigger | pushes to `main` via `.gitlab-ci.yml` + `Dockerfile` (both in repo root) |
| Pre-installed | torch 2.5.1+cu121, flash-attn 2.7.4, mamba-ssm 2.3.0, causal-conv1d 1.6.0, transformers 4.41.2, datasets 2.19.2, sentencepiece 0.2.0, wandb 0.17.0, boto3 1.42.93, pyarrow 24.0.0 |
| Not in the image | the repo itself (cloned fresh at init), `/mnt/data` (PVC), tokenizer artefacts (PVC) |

**Every K8s Job template that runs training or eval needs this block**:

```yaml
spec:
  template:
    spec:
      imagePullSecrets:
      - name: gitlab-registry-cred-thomas
      containers:
      - name: <...>
        image: gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest
        imagePullPolicy: Always   # pick up newer :latest pushes
```

**Rebuild only when**: adding a Python dep that's going to stay, bumping
torch / FA / mamba / transformers majors, or changing CUDA. Python code
changes under `model_foundry/` / `preprocessing/` / `analysis/` /
`evaluation/` do NOT need a rebuild — the clone-at-init pattern handles
them.

**Ad-hoc deps without a rebuild**: for a package you need *now* and
aren't sure you want to bake in permanently, `pip install` at pod init
(after the clone step). Pattern:

```yaml
command: ["bash", "-lc"]
args:
  - |
    set -euo pipefail
    # Clone / pull latest code
    cd /workspace && git pull --ff-only
    # Load anything missing (idempotent — skips if already in image)
    pip install --no-deps -q some_new_package==x.y.z || true
    # Run the actual work
    python -m preprocessing.annotate --input /mnt/data/raw/es/train_90M/ ...
```

The `|| true` + `-q` keeps startup non-fatal if the package is already
in the image (`pip install` is idempotent but noisy) or if PyPI blips.
When the dep proves durable, fold it into `requirements.txt` and the
next `main` push triggers a rebuild that bakes it in permanently.

**Known friction**: cold image pull on a fresh node is ~10 min (image is
~5-10 GB compressed). Subsequent launches on the same node are fast.
If `gitlab-registry-cred-thomas` expires, all training stalls in
`ImagePullBackOff` — rotate with `kubectl create secret docker-registry
gitlab-registry-cred-thomas --docker-server=gitlab-registry.nrp-nautilus.io ...`.

See `memory/reference_gitlab_ci.md` for the full operational detail.

## S3 details (everything needed to talk to the bucket)

| Field | Value |
|---|---|
| Bucket | `s3://thomas-subject-drop-artifacts/` |
| Pool | West (default) |
| Inside-cluster endpoint (pods) | `http://rook-ceph-rgw-nautiluss3.rook` — high bandwidth, multi-OSD |
| Outside-cluster endpoint (laptops) | `https://s3-west.nrp-nautilus.io` — via load balancer |
| K8s secret | `s3-secret-thomas` (keys `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) |
| Local profile | `~/.aws/credentials` section `[nrp]` |

**Every training / eval pod needs this env block** (already added to
the 7 active training templates under `k8s/`):

```yaml
env:
- name: AWS_ACCESS_KEY_ID
  valueFrom: {secretKeyRef: {name: s3-secret-thomas, key: AWS_ACCESS_KEY_ID}}
- name: AWS_SECRET_ACCESS_KEY
  valueFrom: {secretKeyRef: {name: s3-secret-thomas, key: AWS_SECRET_ACCESS_KEY}}
- {name: AWS_ENDPOINT_URL,   value: "http://rook-ceph-rgw-nautiluss3.rook"}
- {name: AWS_DEFAULT_REGION, value: "us-west-1"}
- {name: REGISTRY_BUCKET,    value: "thomas-subject-drop-artifacts"}
```

**Tool of choice**: `boto3` for programmatic writes (registry JSONs,
small eval parquets), `pyarrow.fs.S3FileSystem` for native parquet I/O,
`rclone` for bulk / multi-GB transfers. **Do NOT use `aws s3 cp`** — the
NRP docs flag a multipart upload bug on files >80 MB; boto3 handles
multipart correctly.

Bucket layout (source of truth: `docs/RUN_REGISTRY.md` + `docs/S3_INTEGRATION.md`):

```
s3://thomas-subject-drop-artifacts/
├── run_registry/
│   ├── by_run/<arch>/<lang>/<condition>/<run_id>.json   ← authoritative
│   └── registry.parquet                                 ← hourly-compacted view
├── eval_results/{blimp,perplexity,null_subj,...}/<run_id>.parquet
├── training_curves/<run_id>.parquet
└── env_snapshots/<run_id>.txt
```

## WandB (live dashboard; registry is the archive)

WandB is the live tracking UI. The registry (S3) is the authoritative
paper artifact. Runs cross-reference via `wandb_run_id` (already in
every registry record).

| Field | Value |
|---|---|
| K8s secret | `wandb-secret-thomas` (key `WANDB_API_KEY`) |
| Entity | user's default (not explicitly set) |
| Production project | `just-drop-the-subject` (existing baseline is here) — new configs may use `subject-drop-production`; either works, routed by `config.logging.wandb_project` |
| HP sweep project | `subject-drop-sweeps` (configurable via `config.logging.wandb_project_sweeps`) |
| Smoke runs | `use_wandb: false` — no WandB noise from smokes |

**Every training pod calls `model_foundry.wandb_init.init_wandb(config, identity)`** — NOT `wandb.init(...)` directly — so naming, grouping, tagging are consistent across cli.py / sweep agent / future launchers:

- `name` = deterministic `run_id` (matches registry key; one click to cross-reference)
- `group` = `<arch>-<lang>-<condition>` (WandB's "group by group" view collapses 30 seeds into mean±std)
- `job_type` = `train` or `sweep`
- `tags` = flat `arch=…`, `lang=…`, `condition=…`, `seed=…`, `run_kind=…`

Training YAMLs need this env entry alongside the S3 block:

```yaml
- name: WANDB_API_KEY
  valueFrom: {secretKeyRef: {name: wandb-secret-thomas, key: WANDB_API_KEY}}
```

**Eval runner pattern** (future work): re-open the training run's WandB id to log final eval scores on the same WandB page as the training curves. Get the id from the registry: `registry.get_record(...).wandb_run_id`. Call `wandb.init(id=<...>, resume="must")`.

**Training-curve archival** (future CronJob): nightly export of WandB history for completed runs → `s3://.../training_curves/<run_id>.parquet`. WandB API will still work in 5 years; the export is the paper's frozen citation record.

## Code conventions that matter

- **Tokenizer = one-shot operator setup.** Not a per-training-pod step.
  Sentencepiece unigram has stochastic tie-breaking; re-training would
  shift the vocabulary and invalidate within-(arch, lang) ablation
  comparisons. Train once per (arch, lang) via
  `k8s/job-train-tokenizer.yaml`.
- **Content-addressed cache keys** (`model_foundry/cache_keys.py`) are
  hashed from (corpus_path, tokenizer bytes, max_sequence_length,
  dataset_manipulation). Paths: `data/tokenized/<hash>/` and
  `data/chunked/<hash>/`. Different experiments with same ingredients
  share cache.
- **Registry is organizational, not auto-scheduling.** User invokes
  launchers (`scripts/launch_training.py`, `scripts/launch_evals.py`)
  for group operations. Only two CronJobs: compactor + reaper.
  Everything else is user-driven or automatic-within-a-launched-pod.
- **Checkpoint save policy**: last N (default 3) scheduled checkpoints
  save full `training_state.pt`, earlier ones save weights-only. Set
  via `training.save_resume_state_last_n` in the config.
- **AMP** (`use_amp: true`) is on for every transformer/BERT/Mamba
  config — needed so FA2 sees fp16 activations via autocast.
- **Registry writes are non-fatal**. S3 outage never kills a 20-hour
  training job — the `_safe_*` wrappers log and move on. Stale records
  get reconciled on next run.

## Known bugs / WIP

- **Tokenizer CLI-path idempotency bug**: `python -m model_foundry.cli
  train-tokenizer <config>` re-trains even when the cache is present.
  Calling the factory directly from a REPL hits the cache correctly.
  Currently has `[debug]` prints committed in
  `model_foundry/tokenizer/tokenizer_factory.py`; next run reveals the
  faulty predicate. Workaround: only invoke via
  `k8s/job-train-tokenizer.yaml` once per (arch, lang).
- **Not yet built**: launcher scripts, eval runner (other agent),
  pruners (1.4 / 1.5), reference-rep selector (1.6), reaper CronJob,
  compactor CronJob, HP sweep infra (decisions locked, code not
  written).

## Key docs to read (in order of relevance for most tasks)

1. `docs/RUN_REGISTRY.md` — full v1 registry schema + state machine +
   writer contract + launcher shape + read patterns.
2. `docs/S3_INTEGRATION.md` — endpoint, secret, layout, API, failure
   modes, K8s env-var block.
3. `memory/reference_storage_diagram.md` — canonical ASCII data-flow
   diagram.
4. `memory/reference_storage_plan.md` — detailed sizing, retention,
   pruning policy.
5. `memory/reference_hp_sweep_plan.md` — HP sweep methodology and
   per-arch optimization priorities.
6. `memory/reference_implementation_roadmap.md` — phased plan P0 → P3,
   agent handoff map.

## Common operator commands

```bash
# Train a tokenizer (one-shot per arch/lang)
kubectl apply -f k8s/job-train-tokenizer.yaml
# Edit TOKENIZER_CONFIG env if needed for a different config file

# Launch a training run (smoke pattern; production launcher TBD)
kubectl apply -f k8s/job-smoke-gpt2-medium.yaml

# Read the registry locally
aws --profile nrp s3 cp \
  s3://thomas-subject-drop-artifacts/run_registry/registry.parquet - | \
  python -c "import sys, pyarrow.parquet as pq; \
             t = pq.read_table(sys.stdin.buffer); \
             print(t.to_pandas().head())"

# Force-compact the registry (run on-demand before an analysis)
python scripts/compact_registry.py --bucket thomas-subject-drop-artifacts
```
