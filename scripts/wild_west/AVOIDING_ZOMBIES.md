Got it. Here’s a **practical, opinionated guide** for running training jobs directly on a shared GPU server **without** a scheduler (SLURM), focused on preventing the exact mess you just had: zombie processes, ghost GPU memory, runaway jobs, and resource contention.

Use this as the “rules of the road” for anyone shipping run scripts on this machine.

---

# Principles (non-negotiable MUSTs)

## GPU selection & isolation

* **MUST** set and *honor* `CUDA_VISIBLE_DEVICES` for every run (host and inside containers).
* **MUST NOT** let containers wipe env accidentally: pass `CUDA_VISIBLE_DEVICES` (and friends) into Apptainer/Singularity even with `--cleanenv`.
* **MUST** provide a **lock** for each GPU you intend to use before launching a job, and **MUST** release it on exit (success or failure).

## Process lifecycle & cleanup

* **MUST** launch jobs in their **own session/process group** (e.g., `setsid`) and, on exit/signals, **MUST** kill the **entire PGID** (not just a PID).
* **MUST** trap `INT`, `TERM`, and `EXIT` in the launcher and forward them to the whole group.
* **MUST NOT** rely on `jobs -p` or shell job control; foreground containers won’t show up there.
* **MUST** use a tiny init (e.g., `tini`/`dumb-init`) *inside* the container for proper child reaping when available.

## Safety rails

* **MUST** use `timeout` around each long-running phase with a gentle → hard kill escalation.
* **MUST** fail fast when required files/paths/GPUs aren’t available.
* **MUST** cap resource usage where possible:

  * `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=512`
  * `NCCL_ASYNC_ERROR_HANDLING=1`
  * `TORCH_CUDA_MEMORY_FRACTION=0.95`
  * `OMP_NUM_THREADS`, `MKL_NUM_THREADS` to a sensible lower value (e.g., 4–8) to avoid CPU pileups.
* **MUST** keep datasets read-only and write outputs/checkpoints/logs to per-run directories.

## Observability & hygiene

* **MUST** log: job id, user, host, GPUs, PID/PGID, config, git commit, start/stop timestamps.
* **MUST** write a “run manifest” JSON/YAML next to logs with all hyperparams + environment + container tag.
* **MUST** set W\&B mode **deliberately**: `WANDB_MODE=offline` by default; only use `online` if you have bandwidth and creds; always call `wandb.finish()` in code on SIGTERM.
* **MUST** avoid “install-on-every-run” behavior inside containers (no `pip install`, `spacy download`, etc., in hot path).

## Etiquette on a shared box

* **MUST** default to low-impact scheduling: `nice` and `ionice` (unless latency sensitive).
* **MUST** check GPU idleness before taking a lock; **MUST NOT** squat on a GPU with <10 GB free unless you know what you’re doing.
* **MUST** coordinate/reset GPUs only when others aren’t running on them.

---

# Runbook: before/during/after a job

## Preflight (before you launch)

1. **Config & files present**: fail if missing.
2. **GPU locks**: acquire (`/tmp/gpu_locks/gpu_${id}.lock`) with user, time, host, PID, config, phase; fail if taken.
3. **GPU state**: inspect `nvidia-smi --query-compute-apps` for **\[Not Found]** ghosts on target GPUs; choose others or contact admin for reset.
4. **Container image** exists; refuse to run otherwise.
5. **Log dir** created; write a run manifest.

## During the run

* **Own session**: start container with `setsid`, record child PID & PGID.
* **Signal handling**: trap and kill **`-PGID`** on INT/TERM/EXIT; `wait` the child.
* **Timeouts**: wrap phases with `timeout --signal=TERM --kill-after=30s <duration>`.
* **Health prints**: every N minutes print GPU memory/SM util & sample throughput.

## Postflight (after exit)

* **Release locks** (even on error).
* **Summarize**: write exit code, duration, final GPU usage snapshot.
* **Auto-triage**: if exit non-zero, include the last 200 log lines in a `.fail` note.

---

# Hardened launcher: minimal template

Use this as the **baseline every script must start from**. (Edit paths/commands as needed.)

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
GPUS="${GPUS:-2}"                          # comma-separated
SIF="$PROJECT_DIR/singularity/training.sif"
CMD="python -m model_foundry.cli run configs/$CONFIG.yaml"
LOCK_DIR="/tmp/gpu_locks"
LOG_DIR="$PROJECT_DIR/logs/$(date +%Y%m%d_%H%M%S)_$USER"
TIMEOUT="24h"

mkdir -p "$LOCK_DIR" "$LOG_DIR"

log() { printf "[%(%F %T)T] %s\n" -1 "$*" | tee -a "$LOG_DIR/run.log"; }

# ---- lock GPUs ----
IFS=',' read -ra GPU_ARR <<< "$GPUS"
for g in "${GPU_ARR[@]}"; do
  f="$LOCK_DIR/gpu_${g}.lock"
  [[ -e $f ]] && { log "GPU $g locked"; exit 1; }
done
for g in "${GPU_ARR[@]}"; do
  f="$LOCK_DIR/gpu_${g}.lock"
  {
    echo "user=$(whoami)"; echo "host=$(hostname)"; echo "time=$(date -Is)";
    echo "pid=$$"; echo "gpus=$GPUS"; echo "cmd=$CMD";
  } > "$f"
done

cleanup() {
  trap - INT TERM EXIT
  [[ -n "${PGID:-}" ]] && { kill -TERM -"$PGID" 2>/dev/null || true; sleep 2; kill -KILL -"$PGID" 2>/dev/null || true; }
  for g in "${GPU_ARR[@]}"; do rm -f "$LOCK_DIR/gpu_${g}.lock"; done
}
trap cleanup INT TERM EXIT

# ---- env to pass into container ----
ENV_FLAGS=(
  --env "CUDA_VISIBLE_DEVICES=${GPUS}"
  --env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
  --env "TORCH_CUDA_MEMORY_FRACTION=${TORCH_CUDA_MEMORY_FRACTION:-0.95}"
  --env "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
  --env "NCCL_ASYNC_ERROR_HANDLING=1"
  --env "WANDB_MODE=${WANDB_MODE:-offline}"
)

# ---- pick container runtime ----
if command -v apptainer >/dev/null 2>&1; then R=apptainer; else R=singularity; fi

# ---- launch ----
log "GPUs=$GPUS SIF=$SIF"
set +e
setsid $R exec --nv --pid --contain --cleanenv \
  "${ENV_FLAGS[@]}" \
  --bind "${PROJECT_DIR}":/workspace \
  "$SIF" bash -lc "set -euo pipefail; cd /workspace; exec $CMD" &
CHILD=$!
PGID=$(ps -o pgid= "$CHILD" | tr -d ' ')
log "child=$CHILD pgid=$PGID"

timeout --preserve-status --signal=TERM --kill-after=30s "$TIMEOUT" \
  bash -c 'wait '"$CHILD"
STATUS=$?
set -e

log "exit_code=$STATUS"
exit "$STATUS"
```

**Why this works**

* `setsid` + trap killing `-PGID` ensures **all** container children die (NCCL, dataloaders, wandb, helpers).
* Env is explicitly passed into the container despite `--cleanenv`.
* Locks prevent accidental GPU contention and give you a breadcrumb trail (who/when/what).

---

# Coding guidelines for training scripts (inside Python)

* **Handle SIGTERM**: gracefully stop training, close dataloaders, `torch.distributed.destroy_process_group()`, `wandb.finish()`, and save a final checkpoint.
* **Don’t** spawn untracked subprocesses; if you do, ensure they die when the parent dies (set them to new process group or use `prctl`/`setpgrp` with care).
* **Use deterministic seeds** where required; log them.
* **Write checkpoints atomically** (temp file then `rename`) to avoid corrupt artifacts on kill.
* **Bounded workers**: set `num_workers` conservatively; use `persistent_workers=False` unless you have a reason.
* **Validate CUDA device index 0 exists** under the *effective* `CUDA_VISIBLE_DEVICES` you see at runtime.

---

# “What to page the admin with” (be kind)

If you need a GPU reset, send:

* GPU index, current `nvidia-smi` **Processes** table snippet
* Who else is on that GPU (users & PIDs)
* Confirmation that it’s idle (or who’s okay with a reset)
* The exact command you’d like run: `sudo nvidia-smi --gpu-reset -i <id>`
