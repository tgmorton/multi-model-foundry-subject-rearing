"""Shared checkpoint-schedule helper.

Single source of truth for the production / sweep / continuation
checkpoint schedule, so all three agents (``scripts/production_agent.py``,
``scripts/sweep_agent_lm.py``, ``scripts/continue_from_winner.py``) emit
byte-identical anchor lists for the same (arch, hp) inputs.

Locked design (from the PI):

- **Schedule shape is identical for fresh AND resumed runs.**
- **EARLY region** — a step-anchored dense log head
  ``[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]`` followed by log-spaced
  anchors from 512 up to the midpoint (~epoch 13.6 of ``epochs``). The
  early head stays in *step* space (NOT token-converted) so the dense
  early-training detail lands at the same optimizer steps for every run.
  ~80 anchors total fall in the early region.
- **BACK HALF** — exactly one anchor per epoch,
  ``e * opt_steps_per_epoch`` for ``e in 14..epochs`` inclusive. These are
  naturally token-matched because they sit on epoch boundaries.
- **ENDPOINT** — NOT a computed anchor. The training loop guarantees a
  final checkpoint via an explicit guard after the epoch loop terminates
  (see ``model_foundry/training/loop.py``). We never append a
  ``train_steps`` anchor because it is structurally unreachable (the
  dataloader iterates physical batches while ``train_steps`` is computed
  from the effective batch, and the save-check fires before the
  ``global_step`` increment).

restartable (``training_state.pt``) placement:
``resume_state_steps = {ep7 waypoint} ∪ {midpoint ~epoch 13.6} ∪ {all
back-half per-epoch anchors}``; every other anchor is weights-only.

``opt_steps_per_epoch`` is LOOP-ACCURATE:
``floor(ceil(num_chunks / phys_batch) / grad_accum)``. This matches the
physical-batch dataloader in ``model_foundry/data.py`` (``batch_size`` =
physical batch) divided down by gradient accumulation — NOT
``ceil(num_chunks / effective_batch)``, which over-counts and produces
anchors the loop never reaches.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# The fixed, step-anchored dense log head. Kept in optimizer-step space
# (NOT token-converted) per the locked design.
EARLY_LOG_HEAD: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# First epoch (1-indexed) of the back half: one anchor per epoch from
# here through ``epochs``.
BACK_HALF_FIRST_EPOCH = 14

# Fractional-epoch position of the midpoint anchor (between the early
# region and the back half). round(MIDPOINT_EPOCH_FRACTION * opt_steps_per_epoch).
MIDPOINT_EPOCH_FRACTION = 13.6

# Approximate early-region anchor budget (the dense head + log-spaced fill
# up to the midpoint). Preserves the prior ~80-anchor early density.
EARLY_ANCHOR_TARGET = 80

# Epoch (1-indexed) whose boundary anchor doubles as the early resume
# waypoint, so a run preempted in the first half can resume from ~here.
EARLY_RESUME_WAYPOINT_EPOCH = 7


def compute_opt_steps_per_epoch(
    num_chunks: int, phys_batch: int, grad_accum: int
) -> int:
    """Loop-accurate optimizer steps per epoch.

    Mirrors the training loop: the dataloader yields
    ``ceil(num_chunks / phys_batch)`` physical batches per epoch, and the
    optimizer steps once every ``grad_accum`` of them →
    ``floor(physical_batches / grad_accum)``.
    """
    phys_batch = max(1, int(phys_batch))
    grad_accum = max(1, int(grad_accum))
    physical_batches = math.ceil(num_chunks / phys_batch)
    return max(1, physical_batches // grad_accum)


def _nearest(anchors: List[int], target: int) -> int:
    """Return the anchor closest to ``target`` (ties → smaller)."""
    return min(anchors, key=lambda a: (abs(a - target), a))


def compute_checkpoint_schedule(
    num_chunks: int,
    phys_batch: int,
    grad_accum: int,
    epochs: int,
    seq_len: Optional[int] = None,
    *,
    back_half_only: bool = False,
    resume_step: int = 0,
) -> Tuple[List[int], List[int]]:
    """Build the (schedule, resume_state_steps) pair.

    Args:
        num_chunks: Number of chunked training examples (rows in the
            content-addressed chunked cache). Read via
            ``datasets.load_from_disk(chunked_dir).num_rows`` — NOT a
            hardcoded estimate.
        phys_batch: Physical (per-step) batch size — what the dataloader
            actually iterates.
        grad_accum: Gradient-accumulation steps.
        epochs: Total production epochs (30 for production, the sweep
            horizon for sweeps).
        seq_len: Sequence length. Accepted for signature completeness /
            future token-space work; not needed for the step-anchored
            schedule.
        back_half_only: If True, emit ONLY anchors strictly greater than
            ``resume_step`` (used by resume mode — the early head and any
            already-passed back-half anchors are dropped). The returned
            resume_state_steps are likewise filtered to ``> resume_step``.
        resume_step: The optimizer step already reached (resume mode).

    Returns:
        ``(schedule, resume_state_steps)`` where ``schedule`` is the sorted
        list of optimizer steps to checkpoint at, and
        ``resume_state_steps`` is the subset that should also persist full
        ``training_state.pt`` (the rest are weights-only).
    """
    _ = seq_len  # reserved; step-anchored schedule doesn't need it
    opt_steps_per_epoch = compute_opt_steps_per_epoch(
        num_chunks, phys_batch, grad_accum
    )
    total_steps = opt_steps_per_epoch * int(epochs)

    midpoint_step = int(round(MIDPOINT_EPOCH_FRACTION * opt_steps_per_epoch))

    # --- EARLY region: fixed log head, then log-spaced up to midpoint ---
    early: List[int] = [a for a in EARLY_LOG_HEAD if a <= total_steps]
    head_max = early[-1] if early else 0

    # Log-spaced fill from the head's last anchor up to the midpoint, to
    # bring the early region up to ~EARLY_ANCHOR_TARGET anchors.
    if midpoint_step > head_max:
        slots_left = max(0, EARLY_ANCHOR_TARGET - len(early))
        if slots_left > 0:
            log_lo = math.log(float(head_max if head_max > 0 else 1))
            log_hi = math.log(float(midpoint_step))
            for i in range(1, slots_left + 1):
                frac = i / slots_left
                step = int(round(math.exp(log_lo + (log_hi - log_lo) * frac)))
                if step > head_max:
                    early.append(step)

    # Ensure the midpoint anchor itself is present (resume waypoint).
    if midpoint_step > head_max and midpoint_step <= total_steps:
        early.append(midpoint_step)

    # --- BACK HALF: one anchor per epoch, naturally token-matched ---
    back_half: List[int] = []
    for e in range(BACK_HALF_FIRST_EPOCH, int(epochs) + 1):
        step = e * opt_steps_per_epoch
        if step <= total_steps:
            back_half.append(step)

    # Full schedule, capped to reachable steps. The early log-spaced fill
    # runs up to ``midpoint_step`` (~epoch 13.6); for a short horizon
    # (``epochs`` < ~14, i.e. an HP sweep) the midpoint sits PAST the run's
    # end, so the fill would place anchors the loop never reaches. Dropping
    # anything ``> total_steps`` keeps the schedule honest at any horizon.
    # For production (epochs=30) nothing is > total_steps, so this is a
    # no-op there.
    full_schedule = sorted(s for s in (set(early) | set(back_half)) if s <= total_steps)

    # --- resume_state_steps: {ep7 waypoint} ∪ {midpoint} ∪ {back half} ---
    early_sorted = sorted(set(early))
    ep7_target = round(EARLY_RESUME_WAYPOINT_EPOCH * opt_steps_per_epoch)
    ep7_waypoint = _nearest(early_sorted, ep7_target) if early_sorted else 0

    resume_state_steps_set = set(back_half)
    if early_sorted:
        resume_state_steps_set.add(ep7_waypoint)
    if midpoint_step > head_max and midpoint_step <= total_steps:
        resume_state_steps_set.add(midpoint_step)

    # Robustness at any horizon: (a) never designate an unreachable step as
    # a resume point, and (b) ALWAYS make the final reachable anchor carry
    # full resume state, so a run can be continued from its end. Production
    # already includes the epoch-30 anchor here; this is what lets a
    # short-horizon sweep (whose back-half set is empty) still hand a
    # resumable final checkpoint to continue_from_winner.
    resume_state_steps_set = {s for s in resume_state_steps_set if s <= total_steps}
    if full_schedule:
        resume_state_steps_set.add(full_schedule[-1])

    if back_half_only:
        # Resume mode: keep only anchors the resumed run will actually
        # reach (strictly past where it left off). The dense early head is
        # behind us; only remaining back-half (and any midpoint past the
        # resume point) survive.
        schedule = [s for s in full_schedule if s > resume_step]
        resume_state_steps = sorted(
            s for s in resume_state_steps_set if s > resume_step
        )
    else:
        schedule = full_schedule
        resume_state_steps = sorted(resume_state_steps_set)

    return schedule, resume_state_steps


def read_num_chunks(chunked_dir: str) -> int:
    """Read the real chunk count from the content-addressed chunked cache.

    Uses the same ``datasets.load_from_disk`` call that
    ``model_foundry/data.py`` and ``scripts/generate_checkpoint_schedule.py``
    make, then returns ``.num_rows``. Raises if the cache is absent — the
    caller must have ensured it exists (production pods fail-fast on a
    missing cache before reaching here).
    """
    from datasets import load_from_disk

    ds = load_from_disk(chunked_dir)
    return int(ds.num_rows)
