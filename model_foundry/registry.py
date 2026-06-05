"""
Run registry for the 5,720-run production matrix (roadmap 0.8).

The registry is the authoritative, portable index of every training run in
the study. It lives on Nautilus Ceph S3 so it survives cluster churn, is
readable without a running WandB instance, and travels with the paper's
data-availability statement.

## Why we need it

With ~5,720 (arch × lang × condition × seed) cells we can't just "look at
CephFS to see what's done." Questions we need answered cheaply:

- Which cells still need training?
- Which have eval parquets written? Which checkpoints are those keyed to?
- Which seed is the reference replicate for each cell?
- What git commit / config hash / cache key produced each result?

WandB technically knows most of this but (a) rate-limits list queries,
(b) won't be a dependable API in 5 years when a reader reproduces the
paper, and (c) can't drive the in-training pruner (1.4) or post-eval
pruner (1.5), which need the portable source of truth.

## Storage layout

Authoritative per-run JSON, one key per run (atomic S3 PutObject):

    s3://thomas-subject-drop-artifacts/
      run_registry/
        by_run/
          <arch>/<lang>/<condition>/<run_id>.json   ← one file per run
        registry.parquet                             ← materialized view,
                                                       rebuilt by a compactor
                                                       (scripts/compact_registry.py)

"Every writer only touches its own `run_id` key" is the concurrency
story. Different runs never collide. Multiple writers for the SAME run
(training writer + eval writer) merge: each one reads the current JSON,
updates the fields it owns, and writes it back. This is safe as long as
the two writers touch disjoint fields — which they do, by design.

## Fields the registry stores

See ``SCHEMA_VERSION`` and the write-API functions below. In summary:
identity (run_id / arch / lang / condition / seed), reproducibility
(config_hash / git_commit / cache_key / tokenizer_dir), provenance
(wandb_run_id / node_name / gpu_product), lifecycle
(status / started_at / finished_at / last_heartbeat_at / attempt_count),
outputs (final_loss / total_tokens_processed / checkpoint_count /
checkpoint_paths / train_steps), eval (eval_status /
eval_parquet_paths / eval_finished_at), and analysis
(is_reference_rep / reference_rep_rationale).

## When each function is called

- ``register_run_start`` — by the training entrypoint right after config
  load, before the first optimizer step.
- ``heartbeat`` — from the training loop on a 5-minute timer, so a reaper
  can detect runs whose pods crashed without finishing.
- ``register_run_end`` — from the training entrypoint on completion
  (success or failure).
- ``write_env_snapshot`` — once, from ``register_run_start``, to capture
  pip freeze + git commit + nvidia-smi into
  ``env_snapshots/<run_id>.txt``.
- ``register_eval_end`` — by the eval runner after parquets land.
- ``mark_reference_rep`` — by the reference-rep selection script (1.6)
  after all reps in a cell finish eval.

## Concurrency / failure semantics

All mutators use a get-modify-put cycle. Individual S3 operations are
atomic per-key; two writers updating the same run_id concurrently may
race, but in practice training and eval never run concurrently on the
same run_id, and repeated updates from the training writer serialize
naturally (heartbeat → heartbeat → run_end).

If S3 is temporarily unreachable, boto3 retries 5× with exponential
backoff (configured in ``_client``). Beyond that we surface the error up
the stack and let the caller decide whether to fail the run or stash a
local sidecar and drain later. The training loop currently treats a
final-registry-write failure as non-fatal (logs loudly, keeps the
checkpoint it just saved); the compactor notices unfinished runs by
their stale ``last_heartbeat_at`` and warns.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional

try:
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
except ImportError:  # pragma: no cover — allow import without boto3 installed
    boto3 = None  # type: ignore
    Config = None  # type: ignore
    ClientError = Exception  # type: ignore


SCHEMA_VERSION = 1
DEFAULT_BUCKET = "thomas-subject-drop-artifacts"
DEFAULT_ENDPOINT = "http://rook-ceph-rgw-nautiluss3.rook"
BY_RUN_PREFIX = "run_registry/by_run"
ENV_SNAPSHOT_PREFIX = "env_snapshots"

logger = logging.getLogger(__name__)


def build_run_id(arch: str, lang: str, condition: str, seed: int) -> str:
    """Deterministic run identifier. Re-runs with the same tuple overwrite
    the same registry record (attempt_count is incremented in-record)."""
    return f"{arch}-{lang}-{condition}-s{seed}"


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_CLIENT = None


def _client():
    """Cached boto3 client. Credentials + endpoint from env."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if boto3 is None:
        raise RuntimeError(
            "boto3 is not installed. Add boto3 to requirements.txt."
        )
    _CLIENT = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )
    return _CLIENT


def _bucket() -> str:
    return os.environ.get("REGISTRY_BUCKET", DEFAULT_BUCKET)


def _run_key(record: Dict[str, Any]) -> str:
    return (
        f"{BY_RUN_PREFIX}/{record['arch']}/{record['lang']}/"
        f"{record['condition']}/{record['run_id']}.json"
    )


def _get_record(arch: str, lang: str, condition: str, run_id: str) -> Optional[Dict[str, Any]]:
    key = f"{BY_RUN_PREFIX}/{arch}/{lang}/{condition}/{run_id}.json"
    try:
        resp = _client().get_object(Bucket=_bucket(), Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read().decode("utf-8"))


def _put_record(record: Dict[str, Any]) -> None:
    key = _run_key(record)
    _client().put_object(
        Bucket=_bucket(),
        Key=key,
        Body=json.dumps(record, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )


def _merge_record(arch: str, lang: str, condition: str, run_id: str,
                  updates: Dict[str, Any]) -> Dict[str, Any]:
    """Read-modify-write merge. Preserves keys the current writer doesn't
    set (e.g. training writer doesn't touch eval fields, and vice versa).
    Always bumps ``updated_at``.

    Identifying fields (arch / lang / condition / run_id) are seeded into
    the merged record even when no prior record exists. Without this
    seeding, ``heartbeat()`` (whose ``updates`` only contains
    ``last_heartbeat_at``) writes a record missing ``arch`` and the next
    ``_run_key()`` call raises ``KeyError: 'arch'``. The KeyError was
    swallowed by ``_safe_merge`` but every 5 min spammed:
        WARNING registry heartbeat(<run_id>) failed: 'arch'
    Seeding here is idempotent for the normal path (where ``current``
    already has the keys) and corrects the heartbeat-without-prior-record
    case."""
    current = _get_record(arch, lang, condition, run_id) or {}
    identity = {"arch": arch, "lang": lang, "condition": condition, "run_id": run_id}
    merged = {**identity, **current, **updates}
    merged["updated_at"] = _utcnow()
    _put_record(merged)
    return merged


# ---------- Non-fatal wrappers ----------
#
# Most registry calls should NOT bring down a long training job over a
# transient S3 issue. These wrappers log the failure and return either
# ``None`` (for reads) or ``{}`` (for writes) so the caller can keep
# going. ``register_run_start`` is the one place where a hard failure
# might be appropriate, but we make even that non-fatal — better to keep
# the run going and reconcile the record manually than to lose 20 hours
# of GPU work to a network blip.

def _safe_get(arch: str, lang: str, condition: str,
              run_id: str) -> Optional[Dict[str, Any]]:
    try:
        return _get_record(arch, lang, condition, run_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("registry _safe_get(%s) failed: %s", run_id, e)
        return None


def _safe_merge(arch: str, lang: str, condition: str, run_id: str,
                updates: Dict[str, Any], op: str = "merge") -> Dict[str, Any]:
    try:
        return _merge_record(arch, lang, condition, run_id, updates)
    except Exception as e:  # noqa: BLE001
        logger.warning("registry %s(%s) failed: %s", op, run_id, e)
        return {}


# ---------- Environment snapshot ----------

def _collect_env_snapshot(git_commit: Optional[str] = None) -> str:
    """Plain-text, ~1 KB. Captured once per run so a reader can answer
    'what exactly was installed in the container that produced this run?'"""
    parts = [
        f"timestamp: {_utcnow()}",
        f"hostname:  {socket.gethostname()}",
    ]
    if git_commit:
        parts.append(f"git_commit: {git_commit}")

    def _run(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                            timeout=20).decode().strip()
        except Exception as e:  # noqa: BLE001
            return f"(failed: {e})"

    parts.append("--- pip freeze ---")
    parts.append(_run(["pip", "freeze"]))
    parts.append("--- nvidia-smi ---")
    parts.append(_run(["nvidia-smi", "-L"]))
    return "\n".join(parts) + "\n"


def write_env_snapshot(run_id: str, git_commit: Optional[str] = None) -> None:
    """Upload ``env_snapshots/<run_id>.txt``. Cheap and high-value for
    reproducibility audits. Silently skipped if S3 isn't reachable — we
    don't want to kill training over a sidecar artefact."""
    try:
        body = _collect_env_snapshot(git_commit=git_commit).encode("utf-8")
        _client().put_object(
            Bucket=_bucket(),
            Key=f"{ENV_SNAPSHOT_PREFIX}/{run_id}.txt",
            Body=body,
            ContentType="text/plain",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("env_snapshot upload failed for %s: %s", run_id, e)


# ---------- Write API ----------

def register_run_queued(
    *,
    run_id: str,
    arch: str,
    lang: str,
    condition: str,
    seed: int,
    run_kind: str = "production",
    hyperparameters: Optional[Dict[str, Any]] = None,
    train_steps: Optional[int] = None,
    docker_image: Optional[str] = None,
) -> Dict[str, Any]:
    """Called by the launcher (scripts/launch_training.py) right before
    `kubectl apply` so the registry knows a launch was attempted even if
    the pod never starts. Sets ``status=QUEUED`` without bumping
    ``attempt_count`` (that happens at register_run_start)."""
    updates = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "arch": arch,
        "lang": lang,
        "condition": condition,
        "seed": seed,
        "run_kind": run_kind,
        "status": "QUEUED",
        "hyperparameters": hyperparameters,
        "train_steps": train_steps,
        "docker_image": docker_image,
    }
    # Drop None values so we don't clobber fields from a previous record
    # (e.g. eval_status from an earlier completed run of the same cell).
    updates = {k: v for k, v in updates.items() if v is not None}
    return _safe_merge(arch, lang, condition, run_id, updates, op="register_run_queued")


def register_run_start(
    *,
    run_id: str,
    arch: str,
    lang: str,
    condition: str,
    seed: int,
    config_hash: str,
    run_kind: str = "production",
    git_commit: Optional[str] = None,
    cache_key: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    docker_image: Optional[str] = None,
    dataset_manipulation_hash: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    wandb_run_id: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_sweep_id: Optional[str] = None,
    node_name: Optional[str] = None,
    gpu_product: Optional[str] = None,
    k8s_pod_name: Optional[str] = None,
    train_steps: Optional[int] = None,
    write_env: bool = True,
) -> Dict[str, Any]:
    """Create or update the record, marking status=RUNNING.

    Idempotent: rerunning a cell (e.g. after preemption) overwrites the
    record but increments ``attempt_count`` so we can tell re-runs apart
    in postmortems.

    Non-fatal on S3 failure — logs and returns ``{}`` so a registry
    outage doesn't kill a 20-hour training job. The caller's retry on
    the next run will catch the record up.
    """
    existing = _safe_get(arch, lang, condition, run_id) or {}
    attempt = int(existing.get("attempt_count", 0)) + 1

    updates = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "arch": arch,
        "lang": lang,
        "condition": condition,
        "seed": seed,
        "run_kind": run_kind,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "cache_key": cache_key,
        "tokenizer_dir": tokenizer_dir,
        "docker_image": docker_image,
        "dataset_manipulation_hash": dataset_manipulation_hash,
        "hyperparameters": hyperparameters,
        "wandb_run_id": wandb_run_id,
        "wandb_project": wandb_project,
        "wandb_sweep_id": wandb_sweep_id,
        "node_name": node_name or os.environ.get("NODE_NAME"),
        "gpu_product": gpu_product or os.environ.get("GPU_PRODUCT"),
        "k8s_pod_name": k8s_pod_name or os.environ.get("HOSTNAME"),
        "train_steps": train_steps,
        "status": "RUNNING",
        "attempt_count": attempt,
        "started_at": _utcnow(),
        "last_heartbeat_at": _utcnow(),
    }
    # Don't clobber eval-side fields written by prior writers.
    # ``_safe_merge`` already preserves keys not present in ``updates``,
    # but be explicit about not passing None for them.
    updates = {k: v for k, v in updates.items() if v is not None}
    merged = _safe_merge(arch, lang, condition, run_id, updates, op="register_run_start")
    if write_env:
        write_env_snapshot(run_id, git_commit=git_commit)
    return merged


def heartbeat(arch: str, lang: str, condition: str, run_id: str,
              current_step: Optional[int] = None,
              current_loss: Optional[float] = None,
              current_epoch: Optional[int] = None,
              train_steps: Optional[int] = None) -> None:
    """Light-touch update to prove the run is still alive. Called on a
    ~5-minute cadence from the training loop. Non-fatal — a missed
    heartbeat is just a stale ``last_heartbeat_at``.

    ``current_epoch`` / ``train_steps`` let dashboards render epoch and
    progress bars without re-deriving the per-run step budget (which
    depends on the chunked-cache row count only pods can see)."""
    updates = {"last_heartbeat_at": _utcnow()}
    if current_step is not None:
        updates["current_step"] = current_step
    if current_loss is not None:
        updates["current_loss"] = float(current_loss)
    if current_epoch is not None:
        updates["current_epoch"] = int(current_epoch)
    if train_steps is not None:
        updates["train_steps"] = int(train_steps)
    _safe_merge(arch, lang, condition, run_id, updates, op="heartbeat")


def register_run_end(
    *,
    run_id: str,
    arch: str,
    lang: str,
    condition: str,
    status: str,
    final_loss: Optional[float] = None,
    total_tokens_processed: Optional[int] = None,
    checkpoint_paths: Optional[List[str]] = None,
    checkpoint_count: Optional[int] = None,
    resume_state_steps: Optional[List[int]] = None,
    steps_completed: Optional[int] = None,
    epochs_completed: Optional[int] = None,
    tokens_per_sec_avg: Optional[float] = None,
    data_fraction_avg: Optional[float] = None,
    oom_count: Optional[int] = None,
    n_params: Optional[int] = None,
    failure_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark training complete or failed. ``status`` must be one of
    COMPLETE / FAILED / PREEMPTED. Non-fatal on S3 errors.

    ``n_params`` (G4 provenance) is the model's total parameter count,
    captured at run end so the registry pins the materialized model size.
    """
    assert status in ("COMPLETE", "FAILED", "PREEMPTED"), status
    updates = {
        "status": status,
        "finished_at": _utcnow(),
        "final_loss": final_loss,
        "total_tokens_processed": total_tokens_processed,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_count": (
            checkpoint_count
            if checkpoint_count is not None
            else (len(checkpoint_paths) if checkpoint_paths else None)
        ),
        "resume_state_steps": resume_state_steps,
        "steps_completed": steps_completed,
        "epochs_completed": epochs_completed,
        "tokens_per_sec_avg": tokens_per_sec_avg,
        "data_fraction_avg": data_fraction_avg,
        "oom_count": oom_count,
        "n_params": n_params,
        "failure_reason": failure_reason,
    }
    # Drop None so we don't clobber eval-side fields previously written.
    updates = {k: v for k, v in updates.items() if v is not None}
    return _safe_merge(arch, lang, condition, run_id, updates, op="register_run_end")


def register_eval_start(
    *,
    run_id: str,
    arch: str,
    lang: str,
    condition: str,
    benchmark: str,
) -> Dict[str, Any]:
    """Called by the eval runner at the start of one benchmark.
    Updates ``eval_status=RUNNING`` (aggregate) and
    ``eval_benchmarks[benchmark]={status: RUNNING}``."""
    existing = _safe_get(arch, lang, condition, run_id) or {}
    benchmarks = dict(existing.get("eval_benchmarks") or {})
    benchmarks[benchmark] = {**benchmarks.get(benchmark, {}),
                              "status": "RUNNING",
                              "started_at": _utcnow()}
    return _safe_merge(arch, lang, condition, run_id, {
        "eval_status": "RUNNING",
        "eval_started_at": existing.get("eval_started_at") or _utcnow(),
        "eval_benchmarks": benchmarks,
    }, op="register_eval_start")


def register_eval_benchmark_done(
    *,
    run_id: str,
    arch: str,
    lang: str,
    condition: str,
    benchmark: str,
    status: str,
    parquet_path: Optional[str] = None,
    metric_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Called per-benchmark when its eval finishes.
    ``status`` one of COMPLETE / PARTIAL / FAILED."""
    assert status in ("COMPLETE", "PARTIAL", "FAILED"), status
    existing = _safe_get(arch, lang, condition, run_id) or {}
    benchmarks = dict(existing.get("eval_benchmarks") or {})
    benchmarks[benchmark] = {
        **benchmarks.get(benchmark, {}),
        "status": status,
        "parquet_path": parquet_path,
        "finished_at": _utcnow(),
        "metric_summary": metric_summary,
    }
    # Recompute aggregate eval_status: COMPLETE only if every benchmark is
    # COMPLETE; FAILED if any is FAILED; otherwise PARTIAL.
    statuses = {b.get("status") for b in benchmarks.values()}
    if "FAILED" in statuses:
        agg = "FAILED"
    elif statuses == {"COMPLETE"}:
        agg = "COMPLETE"
    else:
        agg = "PARTIAL"
    return _safe_merge(arch, lang, condition, run_id, {
        "eval_benchmarks": benchmarks,
        "eval_status": agg,
    }, op="register_eval_benchmark_done")


def register_eval_end(
    *,
    run_id: str,
    arch: str,
    lang: str,
    condition: str,
    eval_status: str,
    eval_parquet_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Called by the eval runner once all benchmarks have settled.
    ``eval_status`` one of COMPLETE / PARTIAL / FAILED. The
    per-benchmark detail goes through ``register_eval_benchmark_done``;
    this call just stamps the aggregate end-time.

    ``eval_parquet_paths`` is optional and kept for back-compat with
    callers that aggregate the parquet list themselves; new callers
    should let ``eval_benchmarks[*].parquet_path`` be the source of
    truth and skip this argument."""
    assert eval_status in ("COMPLETE", "PARTIAL", "FAILED"), eval_status
    updates = {
        "eval_status": eval_status,
        "eval_finished_at": _utcnow(),
    }
    if eval_parquet_paths is not None:
        updates["eval_parquet_paths"] = eval_parquet_paths
    return _safe_merge(arch, lang, condition, run_id, updates, op="register_eval_end")


def mark_reference_rep(
    *,
    run_id: str,
    arch: str,
    lang: str,
    condition: str,
    rationale: str,
    metric: Optional[str] = None,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    """Mark this seed as the reference replicate for its cell. Called by
    the selection script (1.6) after all reps in the cell have eval'd.
    ``metric`` is machine-readable (e.g. ``"blimp.metric_summary.accuracy_final_ckpt"``)
    and ``score`` is the actual value on that metric."""
    return _safe_merge(arch, lang, condition, run_id, {
        "is_reference_rep": True,
        "reference_rep_rationale": rationale,
        "reference_rep_metric": metric,
        "reference_rep_score": score,
        "reference_rep_selected_at": _utcnow(),
    }, op="mark_reference_rep")


# ---------- Public read helpers for launchers + analyses ----------

def get_record(arch: str, lang: str, condition: str,
               run_id: str) -> Optional[Dict[str, Any]]:
    """Return the current registry record for a run, or ``None`` if no
    record exists / S3 is unreachable. Used by launchers to check
    existing state before deciding whether to launch."""
    return _safe_get(arch, lang, condition, run_id)


# ---------- Convenience: load-all, for the compactor ----------

def iter_all_records():
    """Yield every run record currently in the registry. Used by the
    compactor to rebuild ``registry.parquet``."""
    client = _client()
    bucket = _bucket()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{BY_RUN_PREFIX}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            try:
                yield json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed registry record %s: %s", key, e)
