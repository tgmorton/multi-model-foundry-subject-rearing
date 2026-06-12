"""D11: Content-addressed cache / skip-if-exists guard.

Wraps the per-model runner with a "before each checkpoint evaluation,
check if the output already exists on disk; if so, skip" policy. Output
paths are deterministic functions of
  (cell_id, checkpoint_id, stimuli_id, unigram_id, scoring_version).

`checkpoint_id` is a 12-char content hash of the checkpoint's
`pytorch_model.bin`; recomputing it is cheap (one stat + one sha256
pass) compared to loading + running the model.

Example:

    from evaluation.cache import CachedRunner
    cached = CachedRunner(
        cell=cell,
        output_root=Path("/mnt/data/eval_v2"),
        scoring_version="v1",
    )
    cached.run_once()   # first call: evaluates and writes
    cached.run_once()   # second call: detects existing outputs, no forward passes
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evaluation.output_v2 import write_cell_results
from evaluation.runners.per_model_runner import (
    CellSpec,
    CheckpointItemResult,
    CheckpointPairResult,
    PerModelRunner,
    find_checkpoints_sorted,
)

logger = logging.getLogger(__name__)


# --- Content-hash helpers ---------------------------------------------------

def hash_file(path: Path, nbytes: int = 12) -> str:
    """Return first `nbytes` hex chars of sha256(file contents)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:nbytes]


def checkpoint_id(checkpoint_path: Path) -> str:
    """Content hash of a checkpoint directory's weights file.

    Accepts either `model.safetensors` (newer HF) or `pytorch_model.bin`.
    """
    ckpt = Path(checkpoint_path)
    for fname in ("model.safetensors", "pytorch_model.bin"):
        p = ckpt / fname
        if p.exists():
            return hash_file(p)
    raise FileNotFoundError(
        f"No weights file in {ckpt} (expected model.safetensors or pytorch_model.bin)"
    )


# --- Cache key --------------------------------------------------------------

@dataclass(frozen=True)
class CacheKey:
    """All inputs that determine a cell's checkpoint output."""
    cell_id: str
    checkpoint_id: str
    stimuli_id: str
    unigram_id: Optional[str]
    scoring_version: str

    def as_marker_filename(self) -> str:
        """Canonical marker filename for this key."""
        uni = self.unigram_id or "none"
        return (
            f"{self.cell_id}__ckpt={self.checkpoint_id}__stim={self.stimuli_id}"
            f"__uni={uni}__scoring={self.scoring_version}.done"
        )


def is_cached(output_root: Path, key: CacheKey) -> bool:
    """True iff the `.done` marker exists for this (cell, checkpoint)."""
    marker_dir = Path(output_root) / ".cache"
    return (marker_dir / key.as_marker_filename()).exists()


def mark_cached(output_root: Path, key: CacheKey) -> Path:
    """Drop a `.done` marker for this (cell, checkpoint). Atomic rename."""
    marker_dir = Path(output_root) / ".cache"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / key.as_marker_filename()
    tmp = marker.with_suffix(".tmp")
    tmp.write_text("ok\n")
    tmp.replace(marker)
    return marker


# --- Cached runner ----------------------------------------------------------

class _CheckpointPrefetcher:
    """Stage upcoming checkpoints' weight files onto local disk while the
    GPU scores the current one.

    Two wins on CephFS: the evaluator loads weights from node-local
    ephemeral disk (instant) instead of blocking on a ~66 MB/s single
    stream, and `streams` concurrent copiers raise aggregate CephFS read
    throughput (per-stream cap ≫ aggregate cap). Staging failures fall
    back to direct reads — the prefetcher can only make things faster,
    never break a cell.
    """

    def __init__(self, items: List[Tuple[int, Path]], stage_root: Path,
                 depth: int = 3, streams: int = 2):
        self._items = items
        self._stage_root = Path(stage_root)
        self._depth = max(1, depth)
        self._pool = ThreadPoolExecutor(max_workers=max(1, streams),
                                        thread_name_prefix="ckpt-prefetch")
        self._futs: Dict[int, Future] = {}
        self._next_submit = 0

    def _stage(self, step: int, path: Path) -> Optional[Path]:
        try:
            dst = self._stage_root / f"checkpoint-{step}"
            dst.mkdir(parents=True, exist_ok=True)
            for name in CachedRunner._WEIGHT_FILES:
                src = Path(path) / name
                if src.exists():
                    shutil.copy2(src, dst / name)
                    return dst
            return None
        except Exception as exc:
            logger.warning("prefetch of checkpoint-%d failed (%s) — "
                           "falling back to direct read", step, exc)
            return None

    def __iter__(self):
        """Yield (step, original_path, staged_dir_or_None) in order."""
        for idx in range(len(self._items)):
            while (self._next_submit < len(self._items)
                   and self._next_submit - idx < self._depth):
                i = self._next_submit
                s, p = self._items[i]
                self._futs[i] = self._pool.submit(self._stage, s, p)
                self._next_submit += 1
            step, path = self._items[idx]
            staged = self._futs.pop(idx).result()
            yield step, path, staged
            if staged is not None:
                shutil.rmtree(staged, ignore_errors=True)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        shutil.rmtree(self._stage_root, ignore_errors=True)


class CachedRunner:
    """Runs a cell's checkpoints with skip-if-cached semantics.

    On each checkpoint:
      1. Hash `pytorch_model.bin` to get checkpoint_id.
      2. Build CacheKey from (cell_id, checkpoint_id, stimuli_id, unigram_id, scoring_version).
      3. If the marker exists, skip (no model loading, no forward pass).
      4. Otherwise: evaluate the checkpoint, write parquet, drop the marker.

    Re-running over a fully-cached cell performs zero forward passes.
    """

    def __init__(
        self,
        cell: CellSpec,
        output_root: Path,
        scoring_version: str = "v1",
        scratch_dir: Optional[Path] = None,
        prefetch_dir: Optional[Path] = None,
        prefetch_depth: int = 3,
        prefetch_streams: int = 2,
    ):
        """
        Args:
            cell: Cell spec (one architecture × intervention × rep).
            output_root: Final destination for partitioned parquet +
                `.cache/` markers. Usually a shared filesystem (CephFS).
            scoring_version: Version tag for the cache key.
            scratch_dir: Optional pod-local directory where parquet files
                are written before being bulk-copied to `output_root`.
                On CephFS, per-file open/fsync round-trips dominate
                wall-time (measured: 576s vs 0.5s on ephemeral). Setting
                this to an emptyDir on the node drops partitioned-write
                cost by ~30×. Cache markers (`.done` files) always land
                on `output_root` so idempotency survives pod teardown.
        """
        self.cell = cell
        self.output_root = Path(output_root)
        self.scoring_version = scoring_version
        self.scratch_dir = Path(scratch_dir) if scratch_dir else None
        # Pod-local staging dir for checkpoint reads (see
        # _CheckpointPrefetcher). None disables prefetch (direct reads).
        self.prefetch_dir = Path(prefetch_dir) if prefetch_dir else None
        self.prefetch_depth = prefetch_depth
        self.prefetch_streams = prefetch_streams
        self._stimuli_id = cell.stimuli.stimuli_id
        self._unigram_id = (
            cell.unigram.tokenizer_id + "::" + cell.unigram.corpus_id
            if cell.unigram is not None
            else None
        )
        # Lazily-built inner runner; instantiate only if we find uncached work.
        self._inner: Optional[PerModelRunner] = None
        # Forward-pass accounting for tests.
        self.n_forward_passes = 0

    def _make_key(self, ckpt_id: str) -> CacheKey:
        return CacheKey(
            cell_id=self.cell.cell_id,
            checkpoint_id=ckpt_id,
            stimuli_id=self._stimuli_id,
            unigram_id=self._unigram_id,
            scoring_version=self.scoring_version,
        )

    _TABLES = ("items", "pairs", "per_token")

    _WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

    def _table_path(self, root: Path, table: str) -> Path:
        return Path(root) / table / f"cell_id={self.cell.cell_id}.parquet"

    def _persisted_steps(self) -> set:
        """checkpoint_steps with rows in the persisted items parquet."""
        p = self._table_path(self.output_root, "items")
        if not p.exists():
            return set()
        try:
            import pandas as pd
            return set(pd.read_parquet(p, columns=["checkpoint_step"])
                       .checkpoint_step.unique())
        except Exception as exc:  # unreadable parquet → trust nothing
            logger.warning("[%s] could not read persisted items parquet "
                           "(%s) — treating all markers as stale",
                           self.cell.cell_id, exc)
            return set()

    def _carry_rows(self, cached_steps: List[int]) -> dict:
        """Previously-persisted rows for validated-cached checkpoints."""
        import pandas as pd
        carry = {}
        for table in self._TABLES:
            p = self._table_path(self.output_root, table)
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
                keep = df[df.checkpoint_step.isin(cached_steps)]
                if len(keep):
                    carry[table] = keep
            except Exception as exc:
                logger.warning("[%s] carry read failed for %s: %s",
                               self.cell.cell_id, table, exc)
        return carry

    def _merge_carry(self, carry: dict, write_root: Path) -> None:
        """Concat carried rows into the freshly-written cell parquets."""
        if not carry:
            return
        import pandas as pd
        for table, keep in carry.items():
            dst = self._table_path(write_root, table)
            if dst.exists():
                merged = pd.concat([keep, pd.read_parquet(dst)],
                                   ignore_index=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                merged = keep
            merged = merged.sort_values("checkpoint_step", kind="stable")
            merged.to_parquet(dst, index=False)
            logger.info("[%s] merged %d carried rows into %s",
                        self.cell.cell_id, len(keep), dst.name)

    def run_once(self) -> dict:
        """Evaluate any uncached checkpoints; return a summary dict."""
        ckpts = find_checkpoints_sorted(self.cell.checkpoint_root)
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints under {self.cell.checkpoint_root}"
            )

        cached: List[int] = []
        processed: List[int] = []
        all_items: List[CheckpointItemResult] = []
        all_pairs: List[CheckpointPairResult] = []

        # Steps whose rows actually exist in the persisted cell parquet.
        # A marker is only trusted when its rows are present — a run that
        # died between parquet write and marker drop (or vice versa, as in
        # the 2026-06-11 Ceph-freeze incident) must self-heal, not leave a
        # permanent hole.
        existing_steps = self._persisted_steps()

        # With a prefetch_dir, each checkpoint's weights cross CephFS
        # exactly once (background copy to local disk); the content hash
        # for the cache key AND the state-dict load both hit the local
        # copy. Without it, hashing alone re-reads every checkpoint from
        # the shared filesystem (the historical 2× read cost).
        prefetcher = None
        if self.prefetch_dir:
            prefetcher = _CheckpointPrefetcher(
                ckpts, self.prefetch_dir,
                depth=self.prefetch_depth, streams=self.prefetch_streams,
            )
            stream = iter(prefetcher)
        else:
            stream = ((s, p, None) for s, p in ckpts)

        try:
            for step, path, staged in stream:
                local = staged or path
                cid = checkpoint_id(local)
                key = self._make_key(cid)
                if is_cached(self.output_root, key):
                    if step in existing_steps:
                        logger.info("[%s] checkpoint-%d cached (skip)",
                                    self.cell.cell_id, step)
                        cached.append(step)
                        continue
                    logger.warning(
                        "[%s] checkpoint-%d has a marker but no persisted "
                        "rows — stale marker, re-evaluating",
                        self.cell.cell_id, step)

                # Lazy init — only build the model if we actually have work.
                if self._inner is None:
                    self._inner = PerModelRunner(self.cell)
                self._inner.load_checkpoint(local)
                # Provenance records the real checkpoint path, not the
                # staging copy.
                items, pairs = self._inner.evaluate_checkpoint(step, path)
                self.n_forward_passes += 1
                all_items.extend(items)
                all_pairs.extend(pairs)
                processed.append(step)
        finally:
            if prefetcher is not None:
                prefetcher.close()

        # Write the freshly-computed results in one pass (partitioned parquet).
        # If scratch_dir is set, write to pod-local ephemeral first and
        # bulk-copy to output_root at the end — ~30× faster on CephFS.
        write_summary = None
        if all_items or all_pairs:
            # Carry forward rows of validated-cached checkpoints BEFORE the
            # write — write_cell_results overwrites the per-cell file, and a
            # resumed run must not drop previously-persisted checkpoints.
            carry = self._carry_rows(cached) if cached else {}
            write_root = self.scratch_dir or self.output_root
            if self.scratch_dir:
                # Cell-scoped subdir so parallel cells on the same pod
                # don't stomp each other before the final copy.
                write_root = self.scratch_dir / self.cell.cell_id
                write_root.mkdir(parents=True, exist_ok=True)
            write_summary = write_cell_results(
                output_root=write_root,
                cell_id=self.cell.cell_id,
                item_results=all_items,
                pair_results=all_pairs,
            )
            self._merge_carry(carry, write_root)
            if self.scratch_dir:
                logger.info(
                    "[%s] copying staged parquet tree %s → %s",
                    self.cell.cell_id, write_root, self.output_root,
                )
                # dirs_exist_ok=True lets multiple cells merge into the
                # same output_root tree without collision (cell_id is in
                # every file name).
                shutil.copytree(write_root, self.output_root, dirs_exist_ok=True)
                # Best-effort cleanup; pod teardown will finish if this fails.
                shutil.rmtree(write_root, ignore_errors=True)
            # Drop markers for the just-processed checkpoints. Always on
            # output_root so they land on the durable filesystem.
            for step, path in ckpts:
                if step in processed:
                    mark_cached(self.output_root, self._make_key(checkpoint_id(path)))

        return {
            "cell_id": self.cell.cell_id,
            "n_checkpoints_total": len(ckpts),
            "n_cached": len(cached),
            "n_processed": len(processed),
            "cached_steps": cached,
            "processed_steps": processed,
            "write_summary": write_summary,
            "n_forward_passes": self.n_forward_passes,
        }
