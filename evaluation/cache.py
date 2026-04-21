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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    """Content hash of a checkpoint directory's `pytorch_model.bin`."""
    bin_path = Path(checkpoint_path) / "pytorch_model.bin"
    return hash_file(bin_path)


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
    ):
        self.cell = cell
        self.output_root = Path(output_root)
        self.scoring_version = scoring_version
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

        for step, path in ckpts:
            cid = checkpoint_id(path)
            key = self._make_key(cid)
            if is_cached(self.output_root, key):
                logger.info("[%s] checkpoint-%d cached (skip)",
                            self.cell.cell_id, step)
                cached.append(step)
                continue

            # Lazy init — only build the model if we actually have work.
            if self._inner is None:
                self._inner = PerModelRunner(self.cell)
            self._inner.load_checkpoint(path)
            items, pairs = self._inner.evaluate_checkpoint(step, path)
            self.n_forward_passes += 1
            all_items.extend(items)
            all_pairs.extend(pairs)
            processed.append(step)

        # Write the freshly-computed results in one pass (partitioned parquet).
        write_summary = None
        if all_items or all_pairs:
            write_summary = write_cell_results(
                output_root=self.output_root,
                cell_id=self.cell.cell_id,
                item_results=all_items,
                pair_results=all_pairs,
            )
            # Drop markers for the just-processed checkpoints.
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
