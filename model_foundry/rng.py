"""Dedicated, purpose-salted RNG stream derivation.

Replicability redesign (2026-06-04 audit): the pipeline previously drew
every random decision — data order, MLM masking, model init — from the
single global torch RNG, entangling data order with arch-specific init
draw counts and making restart/resume trajectories unreproducible.

Every random consumer now derives its own stream from the run's base
seed via a STABLE hash (blake2b — explicitly NOT Python's builtin
``hash()``, which is salted per-process by PYTHONHASHSEED):

- weight init      → global torch RNG, ``set_seed(base_seed)`` re-called
                     immediately before model construction (condition-
                     invariant by construction — the G3 paired design)
- data order       → ``derive_seed(base, "data_order", epoch)`` →
                     dedicated DataLoader generator, reseeded per epoch
- MLM masking      → ``derive_seed(base, "mlm_mask", epoch, batch)`` →
                     per-batch generator inside the collator
- dropout          → CUDA philox stream, seeded by ``set_seed`` and
                     saved/restored in ``training_state.pt`` (unchanged)

Data order is deliberately NOT arch-salted: at equal seed, archs that
share a chunked cache see the same permutation — a designed property
(matched data order where caches coincide) rather than the previous
accidental entanglement with init draw counts.
"""

from __future__ import annotations

import hashlib

# torch.Generator.manual_seed accepts a signed int64; keep the high bit
# clear so the derived value is always valid.
_MASK63 = (1 << 63) - 1


def derive_seed(base_seed: int, purpose: str, epoch: int = 0, batch: int = 0) -> int:
    """Derive a stable stream seed from (base_seed, purpose, epoch, batch).

    Stable across processes, pods, and Python versions — no builtin
    ``hash()``, no PYTHONHASHSEED dependence.
    """
    payload = f"{int(base_seed)}|{purpose}|{int(epoch)}|{int(batch)}".encode()
    h = hashlib.blake2b(payload, digest_size=8)
    return int.from_bytes(h.digest(), "little") & _MASK63
