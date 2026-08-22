"""Deterministic seed derivation for the wave launcher (lang-manifold port).

Seeds are treated as unique samples from a population (design decision
2026-08-22): never shared across HPs, corpora, or architectures. Rather
than drawing from a stateful RNG at launch time (unrecoverable if the
launcher is re-run), every run's seed is a pure blake2b function of the
wave seed and the run's identity — any process can recompute the full
seed assignment, and the launcher still records each seed explicitly in
the registry and job spec.

Never use Python's builtin hash() for this: it is salted per process
(PYTHONHASHSEED) and not stable across machines.
"""

from hashlib import blake2b


def derive_seed(*parts, bits: int = 31) -> int:
    """Derive a deterministic seed from identity parts.

    >>> derive_seed("wave2", "gpt2_small", "pronoun_drop_info_10", "h0", 0)
    ... # stable across processes and machines

    Returns a non-negative int < 2**bits (default fits int32, which every
    seed consumer in the stack accepts).
    """
    key = ":".join(str(p) for p in parts).encode("utf-8")
    digest = blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (1 << bits)
