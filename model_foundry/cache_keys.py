"""
Content-addressed cache keys for tokenized / chunked datasets (0.2).

The training pipeline previously keyed tokenized+chunked data on
``config.experiment_name``. Two experiments that share an identical corpus,
tokenizer, sequence length, and manipulation pipeline would re-tokenize and
re-chunk from scratch — wasteful once we're running a 5,720-run matrix
where most (arch × seed × condition) cells share tokenized inputs.

This module exposes a small deterministic hash over the ingredients that
actually determine tokenized/chunked output:

  * absolute training-corpus path (directory containing .train files)
  * SHA-256 over the actual corpus CONTENT (``corpus_content_hash``) — the
    top-level ``*.train`` files, not just the path string (see below)
  * SHA-256 of the tokenizer artefact (``tokenizer.model`` for
    SentencePiece, falling back to ``tokenizer.json``)
  * ``max_sequence_length``
  * canonical JSON of the ``dataset_manipulation`` pipeline (empty list for
    the baseline)

Callers write to ``data/tokenized/<key>/`` and ``data/chunked/<key>/``, and
should fall back to the legacy ``data/tokenized/<experiment_name>/`` path
on read so runs that predate this change (e.g. the in-flight baseline)
continue to work.

**Content-addressing (2026-06-05).** Previously the key hashed only the
corpus *path* string, so a corpus that was regenerated in place (same
directory, different bytes) produced the SAME key and silently reused a
stale cache. The key now also folds in ``corpus_content_hash`` — a
SHA-256 over the top-level ``*.train`` files' bytes. This INTENTIONALLY
invalidates every pre-2026-06-05 cache key (the clean-rebuild epoch
following the 2026-06-04 corpus-contamination incident) and guarantees a
regenerated-in-place corpus produces a different key. Content discovery
is TOP-LEVEL ONLY (sorted filenames); it never recurses into intermediate
build subdirs (``_train/``, ``_pool/``, ``pool_remainder/``).
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Optional


def _hash_file_bytes(path: str, chunk_size: int = 1 << 20) -> str:
    """Stream SHA-256 of a file; safe for arbitrarily large tokenizer files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _corpus_content_hash(corpus_path: str) -> str:
    """SHA-256 over the CONTENT of a corpus dir's top-level ``*.train`` files.

    For each top-level ``*.train`` file (sorted by basename — never
    recursing into intermediate build subdirs like ``_train/``, ``_pool/``,
    ``pool_remainder/``), we fold the basename and the file's own streamed
    SHA-256 into an outer hash. This makes the cache key sensitive to the
    actual corpus bytes, so a corpus regenerated in place (same path,
    different content) yields a different key instead of silently reusing a
    stale cache.

    If ``corpus_path`` is a single file rather than a directory, that one
    file is hashed. Returns the hex digest of the outer hash.
    """
    if os.path.isfile(corpus_path):
        train_files = [corpus_path]
    else:
        # TOP-LEVEL ONLY, sorted by basename — same anti-contamination rule
        # as the tokenization pipeline; never use a recursive '**' glob.
        train_files = sorted(
            glob.glob(os.path.join(corpus_path, "*.train")),
            key=os.path.basename,
        )

    outer = hashlib.sha256()
    for fp in train_files:
        basename = os.path.basename(fp)
        file_hash = _hash_file_bytes(fp)
        # Length-prefix-free but unambiguous: fixed-format (name, hash) pair
        # with explicit separators, joined into the outer digest.
        outer.update(basename.encode("utf-8"))
        outer.update(b"\x00")
        outer.update(file_hash.encode("utf-8"))
        outer.update(b"\x00")
    return outer.hexdigest()


def _resolve_tokenizer_artefact(tokenizer_dir: str) -> tuple[str, str]:
    """Return (artefact_path, artefact_kind) for the tokenizer directory."""
    sp = os.path.join(tokenizer_dir, "tokenizer.model")
    hf = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.exists(sp):
        return sp, "tokenizer.model"
    if os.path.exists(hf):
        return hf, "tokenizer.json"
    raise FileNotFoundError(
        f"No tokenizer artefact in {tokenizer_dir} "
        f"(expected tokenizer.model or tokenizer.json)"
    )


def compute_cache_key(
    corpus_path: str,
    tokenizer_dir: str,
    max_sequence_length: int,
    manipulation: Optional[List[Dict[str, Any]]] = None,
    *,
    length: int = 12,
) -> str:
    """
    Deterministic, short cache key for tokenized/chunked data.

    Args:
        corpus_path: Path to the training corpus directory (containing
            .train files). Converted to an absolute path before hashing,
            and its top-level ``*.train`` content is hashed into the key.
        tokenizer_dir: Directory containing the trained tokenizer artefact
            (``tokenizer.model`` or ``tokenizer.json``).
        max_sequence_length: Chunk length; affects the chunked output, so
            it participates in the key.
        manipulation: The ``dataset_manipulation`` pipeline from the
            experiment config (a list of dicts) or ``None``/empty for the
            baseline.
        length: Hex characters to keep (default 12 — ~48 bits, collision
            probability is negligible at this scale).

    Returns:
        A hex-digest string of ``length`` characters.

    Note:
        ``corpus_content_hash`` folds the actual corpus bytes into the key.
        This intentionally invalidates every pre-2026-06-05 cache key and
        makes a regenerated-in-place corpus produce a different key.
    """
    artefact, kind = _resolve_tokenizer_artefact(tokenizer_dir)
    tok_hash = _hash_file_bytes(artefact)

    payload = {
        "corpus_path": os.path.abspath(corpus_path),
        "corpus_content_hash": _corpus_content_hash(corpus_path),
        "tokenizer_hash": tok_hash,
        "tokenizer_source": kind,
        "max_sequence_length": int(max_sequence_length),
        "manipulation": manipulation if manipulation is not None else [],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def cache_meta(
    corpus_path: str,
    tokenizer_dir: str,
    max_sequence_length: int,
    manipulation: Optional[List[Dict[str, Any]]] = None,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the sidecar metadata dict written alongside a cached dataset.

    Dumped to ``meta.json`` inside the cache directory so a human can tell
    which inputs produced this cache without having to recompute the hash.
    """
    artefact, kind = _resolve_tokenizer_artefact(tokenizer_dir)
    meta: Dict[str, Any] = {
        "corpus_path": os.path.abspath(corpus_path),
        "corpus_content_hash": _corpus_content_hash(corpus_path),
        "tokenizer_dir": os.path.abspath(tokenizer_dir),
        "tokenizer_artefact": kind,
        "tokenizer_hash": _hash_file_bytes(artefact),
        "max_sequence_length": int(max_sequence_length),
        "manipulation": manipulation if manipulation is not None else [],
        "created_at": datetime.datetime.now().isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


def resolve_cached_path(
    base_dir: str,
    cache_type: str,
    cache_key: str,
    experiment_name: Optional[str] = None,
) -> tuple[str, str, bool]:
    """
    Resolve a cached dataset directory, falling back to the legacy
    ``experiment_name``-keyed path for backward compatibility.

    Args:
        base_dir: Project root.
        cache_type: "tokenized" or "chunked".
        cache_key: Content-addressed hash from ``compute_cache_key``.
        experiment_name: Legacy cache key. Used only as a read-only fallback
            when the hashed path is absent.

    Returns:
        (path, kind, is_fallback) — ``kind`` is the string "hashed" or
        "legacy"; ``is_fallback`` is True when we resolved to the legacy
        path because the hashed path was absent. New writes should always
        use the hashed path regardless.
    """
    hashed_path = os.path.join(base_dir, "data", cache_type, cache_key)
    if os.path.exists(hashed_path):
        return hashed_path, "hashed", False

    if experiment_name:
        legacy_path = os.path.join(base_dir, "data", cache_type, experiment_name)
        if os.path.exists(legacy_path):
            return legacy_path, "legacy", True

    return hashed_path, "hashed", False
