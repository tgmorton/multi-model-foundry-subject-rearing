"""D9: Per-model runner — load architecture once, swap state_dict per checkpoint.

Replaces the per-checkpoint `AutoModelForCausalLM.from_pretrained()`
rebuild pattern in the v1 runners. For a single
`(architecture, intervention, rep)` cell, the runner:

1. Builds the model architecture once (via a user-supplied factory).
2. Loads a pre-tokenized `StimulusCache` (D6) once.
3. Loads a `UnigramTable` (D3) once for SLOR scoring.
4. For each checkpoint in `find_checkpoints_sorted(root)`:
     a. `model.load_state_dict(torch.load(ckpt/pytorch_model.bin))`
     b. Run `batched_surprisal` on all stimuli
     c. Compute per-pair scores (MeanLP, SLOR, preference, hotspot)
     d. Append results to an in-memory list (caller decides persistence —
        D10 wires up parquet output).

The expensive setup (Python imports, CUDA init, tokenizer load, stimulus
tensors, unigram baseline, model class instantiation) happens once,
amortizing across 40 checkpoints.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from evaluation.core.batched_surprisal import batched_surprisal
from evaluation.stimuli_cache import StimulusCache
from evaluation.unigram import UnigramTable

logger = logging.getLogger(__name__)


# --- Checkpoint discovery ---------------------------------------------------

_CKPT_PATTERNS = [
    re.compile(r"checkpoint-(\d+)$"),
    re.compile(r"epoch_(\d+)$"),
    re.compile(r"ckpt-(\d+)$"),
]


def find_checkpoints_sorted(checkpoint_root: Path) -> List[Tuple[int, Path]]:
    """Return `[(step, path), ...]` sorted ascending by step number.

    Mirrors the behavior of `evaluation/core/model_loader.py:171-208`.
    """
    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    results: List[Tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for pat in _CKPT_PATTERNS:
            m = pat.match(child.name)
            if m:
                results.append((int(m.group(1)), child))
                break
    results.sort(key=lambda t: t[0])
    return results


# --- Per-item result --------------------------------------------------------

@dataclass
class CheckpointItemResult:
    """One (checkpoint, item, pronoun_status) scored row."""
    # Provenance
    cell_id: str
    architecture: str
    intervention: str
    rep: int
    checkpoint_step: int
    checkpoint_path: str
    # Stimulus identity
    item_id: int
    category: str
    condition: str
    pronoun_status: int
    language: str
    # Metrics
    target_sum_log_prob: float
    target_mean_log_prob: float
    target_n_tokens: int
    target_unigram_sum_log_prob: Optional[float]
    slor: Optional[float]
    hotspot_log_prob: Optional[float]
    # Per-token (for MORCELA)
    per_token_log_prob: List[float] = field(default_factory=list)
    per_token_ids: List[int] = field(default_factory=list)


@dataclass
class CheckpointPairResult:
    """One (checkpoint, item, pair) derived row (overt-vs-null)."""
    cell_id: str
    architecture: str
    intervention: str
    rep: int
    checkpoint_step: int
    item_id: int
    category: str
    condition: str
    language: str
    # Overt
    overt_mean_log_prob: float
    overt_slor: Optional[float]
    overt_hotspot_log_prob: Optional[float]
    # Null
    null_mean_log_prob: float
    null_slor: Optional[float]
    null_hotspot_log_prob: Optional[float]
    # Derived
    prefers_overt_meanlp: bool
    prefers_overt_slor: Optional[bool]
    log_prob_diff_overt_minus_null: float
    slor_diff_overt_minus_null: Optional[float]
    hotspot_log_prob_diff_overt_minus_null: Optional[float]


# --- Cell spec --------------------------------------------------------------

@dataclass
class CellSpec:
    """One `(architecture, intervention, rep)` evaluation cell."""
    cell_id: str
    architecture: str
    intervention: str
    rep: int
    checkpoint_root: Path
    # Factory callable with signature `() -> torch.nn.Module`. Produces a
    # fresh model architecture with randomly-initialized weights; the
    # runner then calls `load_state_dict` to swap in each checkpoint.
    model_factory: Callable[[], torch.nn.Module]
    # Pre-tokenized stimuli (D6) and unigram baseline (D3). The stimuli
    # and unigram tokenizer ids must match.
    stimuli: StimulusCache
    unigram: Optional[UnigramTable] = None
    device: str = "cpu"
    batch_size: int = 16
    # Scoring function with the same signature as `batched_surprisal`.
    # None → causal left-to-right scoring. Bidirectional MLMs (bert_large)
    # pass a PLL scorer from `evaluation.core.pll_surprisal` instead.
    scorer: Optional[Callable] = None
    # Padding id used when stacking stimuli into a batch. Must be the
    # tokenizer's real pad id (positions are masked out of attention, but
    # the embedding lookup still has to be in-vocab).
    pad_id: int = 0


# --- Runner -----------------------------------------------------------------

class PerModelRunner:
    """Load a model architecture once; evaluate all its checkpoints."""

    def __init__(self, cell: CellSpec):
        self.cell = cell
        self._model: Optional[torch.nn.Module] = None

    def build_model(self) -> torch.nn.Module:
        """Instantiate the model architecture (once per runner)."""
        if self._model is None:
            logger.info("Building model architecture for cell %s",
                        self.cell.cell_id)
            m = self.cell.model_factory()
            m = m.to(self.cell.device)
            m.eval()
            self._model = m
        return self._model

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Swap in a checkpoint's weights via `load_state_dict`.

        Accepts either `model.safetensors` (newer HF default) or
        `pytorch_model.bin`.

        Uses `strict=False` to tolerate tied weights that HF's
        `save_pretrained` omits from the on-disk state, but only with an
        explicit allowlist from the model's own `_tied_weights_keys`
        attribute. Every HF `PreTrainedModel` subclass declares its
        tying pattern there, so this generalizes across architectures
        without any name-based heuristic. After loading we call
        `tie_weights()` to re-establish sharing.
        """
        model = self.build_model()
        ckpt = Path(checkpoint_path)
        safetensors_file = ckpt / "model.safetensors"
        bin_file = ckpt / "pytorch_model.bin"
        if safetensors_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_file), device=str(self.cell.device))
        elif bin_file.exists():
            state_dict = torch.load(
                bin_file, map_location=self.cell.device, weights_only=False
            )
        else:
            raise FileNotFoundError(
                f"No weights file in {ckpt} "
                f"(expected model.safetensors or pytorch_model.bin)"
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys loading checkpoint: {unexpected}"
            )
        # HF declares _tied_weights_keys relative to the task head (e.g.
        # BertForMaskedLM lists "predictions.decoder.bias" for the missing
        # key "cls.predictions.decoder.bias"), so match by suffix too.
        allowed_missing = list(getattr(model, "_tied_weights_keys", None) or [])
        unexpected_missing = [
            k for k in missing
            if not any(k == t or k.endswith(t) for t in allowed_missing)
        ]
        if unexpected_missing:
            raise RuntimeError(
                f"Unexpected missing keys loading checkpoint: {unexpected_missing}"
            )
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        model.eval()

    # --- Checkpoint-level evaluation ---------------------------------------

    def evaluate_checkpoint(
        self,
        checkpoint_step: int,
        checkpoint_path: Path,
    ) -> Tuple[List[CheckpointItemResult], List[CheckpointPairResult]]:
        """Run a checkpoint over all cached stimuli. Returns per-item and
        per-pair result lists.
        """
        model = self.build_model()
        stimuli = self.cell.stimuli.rows
        if not stimuli:
            return [], []

        full_ids_list = [e.full_token_ids for e in stimuli]
        context_lens = [e.context_len for e in stimuli]
        hotspot_indices = [e.hotspot_token_index for e in stimuli]

        # Run in batches
        all_results = []
        score_fn = self.cell.scorer or batched_surprisal
        bsz = max(1, self.cell.batch_size)
        for start in range(0, len(stimuli), bsz):
            end = min(start + bsz, len(stimuli))
            batch_out = score_fn(
                model,
                full_ids_list[start:end],
                context_lens[start:end],
                hotspot_indices[start:end],
                pad_id=self.cell.pad_id,
                device=self.cell.device,
            )
            all_results.extend(batch_out)

        # Project into per-item results
        item_results: List[CheckpointItemResult] = []
        for e, out in zip(stimuli, all_results):
            target_n = len(out.target_log_probs)
            target_sum = float(np.sum(out.target_log_probs)) if target_n else 0.0
            target_mean = (target_sum / target_n) if target_n else 0.0

            unigram_sum: Optional[float] = None
            slor: Optional[float] = None
            if self.cell.unigram is not None and target_n > 0:
                unigram_sum = float(
                    self.cell.unigram.sum_log_prob(out.target_token_ids)
                )
                slor = (target_sum - unigram_sum) / target_n

            item_results.append(CheckpointItemResult(
                cell_id=self.cell.cell_id,
                architecture=self.cell.architecture,
                intervention=self.cell.intervention,
                rep=self.cell.rep,
                checkpoint_step=checkpoint_step,
                checkpoint_path=str(checkpoint_path),
                item_id=e.item_id,
                category=e.category,
                condition=e.condition,
                pronoun_status=e.pronoun_status,
                language=e.language,
                target_sum_log_prob=target_sum,
                target_mean_log_prob=target_mean,
                target_n_tokens=target_n,
                target_unigram_sum_log_prob=unigram_sum,
                slor=slor,
                hotspot_log_prob=out.hotspot_log_prob,
                per_token_log_prob=list(out.target_log_probs),
                per_token_ids=list(out.target_token_ids),
            ))

        # Join into per-pair results (overt + null sharing item_id+condition)
        pair_results = _build_pair_results(item_results)
        return item_results, pair_results

    # --- Full-cell run -----------------------------------------------------

    def run(self) -> Tuple[List[CheckpointItemResult], List[CheckpointPairResult]]:
        """Iterate through all checkpoints; return aggregated results."""
        ckpts = find_checkpoints_sorted(self.cell.checkpoint_root)
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints found under {self.cell.checkpoint_root}"
            )

        all_items: List[CheckpointItemResult] = []
        all_pairs: List[CheckpointPairResult] = []

        for step, path in ckpts:
            logger.info("[%s] checkpoint-%d", self.cell.cell_id, step)
            self.load_checkpoint(path)
            items, pairs = self.evaluate_checkpoint(step, path)
            all_items.extend(items)
            all_pairs.extend(pairs)

        return all_items, all_pairs


# --- Pair-building helper ---------------------------------------------------

def _build_pair_results(
    item_results: List[CheckpointItemResult],
) -> List[CheckpointPairResult]:
    """Group per-item results into (overt, null) pairs and compute diffs."""
    by_key: Dict[Tuple[int, str, str, str, int], Dict[int, CheckpointItemResult]] = {}
    for r in item_results:
        key = (r.item_id, r.category, r.condition, r.language, r.checkpoint_step)
        by_key.setdefault(key, {})[r.pronoun_status] = r

    pairs: List[CheckpointPairResult] = []
    for (item_id, category, condition, language, checkpoint_step), rows in by_key.items():
        if 1 not in rows or 0 not in rows:
            continue  # incomplete pair — skip
        overt = rows[1]
        null = rows[0]

        slor_diff: Optional[float] = None
        prefers_overt_slor: Optional[bool] = None
        if overt.slor is not None and null.slor is not None:
            slor_diff = overt.slor - null.slor
            prefers_overt_slor = overt.slor > null.slor

        hs_diff: Optional[float] = None
        if overt.hotspot_log_prob is not None and null.hotspot_log_prob is not None:
            hs_diff = overt.hotspot_log_prob - null.hotspot_log_prob

        pairs.append(CheckpointPairResult(
            cell_id=overt.cell_id,
            architecture=overt.architecture,
            intervention=overt.intervention,
            rep=overt.rep,
            checkpoint_step=checkpoint_step,
            item_id=item_id,
            category=category,
            condition=condition,
            language=overt.language,
            overt_mean_log_prob=overt.target_mean_log_prob,
            overt_slor=overt.slor,
            overt_hotspot_log_prob=overt.hotspot_log_prob,
            null_mean_log_prob=null.target_mean_log_prob,
            null_slor=null.slor,
            null_hotspot_log_prob=null.hotspot_log_prob,
            prefers_overt_meanlp=(
                overt.target_mean_log_prob > null.target_mean_log_prob
            ),
            prefers_overt_slor=prefers_overt_slor,
            log_prob_diff_overt_minus_null=(
                overt.target_mean_log_prob - null.target_mean_log_prob
            ),
            slor_diff_overt_minus_null=slor_diff,
            hotspot_log_prob_diff_overt_minus_null=hs_diff,
        ))

    return pairs
