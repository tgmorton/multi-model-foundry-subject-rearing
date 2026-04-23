"""
Compose a final ablated corpus from two independently-ablated directories.

Third step of the three-step ablation workflow:

    1. annotate (corpus + pool)  →  DocBin caches
    2. ablate   (corpus + pool)  →  two ablated directories (no backfill)
    3. compose  (this script)    →  final corpus: ablated train + samples
                                      drawn from ablated pool to hit target
                                      token counts per file.

No spaCy in this step. Reads pre-ablated text, samples from the pool,
concatenates, writes. Determinism controlled by ``--seed``.

Usage::

    python scripts/compose_corpus.py \\
        --ablated-train data/processed/exp_remove_expl_es_train/ \\
        --ablated-pool  data/processed/exp_remove_expl_es_pool/ \\
        --target-corpus data/spanish/train_90M/ \\
        --output        data/processed/exp_remove_expl_es/ \\
        --seed 42

``--target-corpus`` is the original (pre-ablation) source directory. Its
per-file token counts define the target each composed output file must
hit. Without it, ``--target-tokens-per-file`` can be given as a JSON dict.

The output directory gets::

    output/
        {genre}.train              # ablated train + pool-backfill samples
        COMPOSE_MANIFEST.json      # provenance: which lines from which source
        pool_remainder/            # unused ablated-pool lines (for audit)
            {genre}.train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Whitespace tokenizer matching preprocessing/utils.py::count_tokens.
# Re-implemented here to keep compose_corpus.py import-light (no spaCy,
# no torch, no pydantic). See ``_count_tokens`` below.
_TOKEN_RE = re.compile(r"\S+")


def _count_tokens(text: str) -> int:
    """Count whitespace-separated tokens. Matches ``preprocessing.utils.count_tokens``."""
    return len(_TOKEN_RE.findall(text))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_paired_files(train_dir: Path, pool_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Return ``(stem, train_path, pool_path)`` for every .train file present
    in BOTH directories. Errors loudly if either side is missing a stem so
    the caller has a chance to regenerate rather than silently dropping genres.
    """
    train_files = {p.stem: p for p in train_dir.glob("*.train")}
    pool_files = {p.stem: p for p in pool_dir.glob("*.train")}

    missing_in_pool = sorted(train_files.keys() - pool_files.keys())
    missing_in_train = sorted(pool_files.keys() - train_files.keys())
    if missing_in_pool:
        raise FileNotFoundError(
            f"Train dir has files without a matching pool file: "
            f"{missing_in_pool}. Ablate the pool before composing."
        )
    if missing_in_train:
        # Extra pool files are tolerable but worth a warning.
        print(
            f"[warn] Pool dir has files without matching train: "
            f"{missing_in_train} (will be ignored)",
            file=sys.stderr,
        )

    return sorted([
        (stem, train_files[stem], pool_files[stem])
        for stem in train_files.keys() & pool_files.keys()
    ])


def _load_target_token_counts(
    target_corpus: Optional[Path],
    explicit: Optional[Dict[str, int]],
    stems: List[str],
) -> Dict[str, int]:
    """Resolve per-file target token counts.

    If ``target_corpus`` is given, read its .train files and count tokens.
    Else use ``explicit`` (must have a key for every stem).
    """
    if target_corpus is not None:
        result = {}
        for stem in stems:
            path = target_corpus / f"{stem}.train"
            if not path.exists():
                raise FileNotFoundError(
                    f"--target-corpus missing file for stem '{stem}': {path}"
                )
            with open(path, "r", encoding="utf-8") as f:
                result[stem] = _count_tokens(f.read())
        return result

    if explicit is not None:
        missing = [s for s in stems if s not in explicit]
        if missing:
            raise ValueError(
                f"--target-tokens-per-file JSON missing keys: {missing}"
            )
        return {s: int(explicit[s]) for s in stems}

    raise ValueError(
        "Provide either --target-corpus (preferred) or "
        "--target-tokens-per-file (JSON dict of stem → count)."
    )


def _compose_file(
    stem: str,
    train_path: Path,
    pool_path: Path,
    target_tokens: int,
    output_path: Path,
    rng: random.Random,
) -> Dict[str, int]:
    """Build the final file: concatenate ablated train + enough pool samples
    to hit the target token count. Returns stats for the manifest.

    Sampling is without replacement at the line level — each pool line is
    used at most once. If the pool is too small to close the gap, we raise
    so the caller knows to either loosen the target or regenerate the pool.
    """
    # Read both sides once. File sizes are bounded (pool is ~10% of train);
    # comfortably fits in memory for BabyLM/BebeLM source files.
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    train_tokens = _count_tokens(train_text)
    shortfall = target_tokens - train_tokens

    lines_drawn_from_pool = 0
    pool_tokens_added = 0
    pool_lines_used_indices: List[int] = []

    composed_text = train_text
    # If the train is already at or above target (rare — only if ablation
    # added tokens; not typical), no backfill needed.
    if shortfall > 0:
        with open(pool_path, "r", encoding="utf-8") as f:
            pool_lines = f.readlines()

        if not pool_lines:
            raise ValueError(
                f"[{stem}] pool is empty but {shortfall} tokens of "
                f"backfill needed. Re-ablate the pool."
            )

        # Shuffle indices once; draw from the front until we hit target.
        order = list(range(len(pool_lines)))
        rng.shuffle(order)

        for idx in order:
            if pool_tokens_added >= shortfall:
                break
            line = pool_lines[idx]
            # Treat empty/whitespace pool lines as zero-contribution —
            # they're either blank or artifacts; they add no tokens and
            # shouldn't count as "draws". Skip outright so we don't waste
            # the sample allowance on them.
            line_tokens = _count_tokens(line)
            if line_tokens == 0:
                continue
            composed_text += line
            pool_tokens_added += line_tokens
            lines_drawn_from_pool += 1
            pool_lines_used_indices.append(idx)

        if pool_tokens_added < shortfall:
            raise ValueError(
                f"[{stem}] ablated pool exhausted at {pool_tokens_added} "
                f"tokens; still need {shortfall - pool_tokens_added} more. "
                f"Pool had {len(pool_lines)} lines; ablation may have removed "
                f"too many. Options: (1) use a larger pool, (2) accept "
                f"under-target output."
            )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(composed_text)

    final_tokens = _count_tokens(composed_text)
    return {
        "stem": stem,
        "train_tokens": train_tokens,
        "target_tokens": target_tokens,
        "shortfall": shortfall,
        "pool_lines_drawn": lines_drawn_from_pool,
        "pool_tokens_added": pool_tokens_added,
        "final_tokens": final_tokens,
        "pool_lines_available": (
            sum(1 for _ in open(pool_path, "r", encoding="utf-8"))
        ),
        "pool_line_indices_used": pool_lines_used_indices,
        "output_checksum": _sha256(output_path),
    }


def _write_pool_remainder(
    pool_path: Path,
    used_indices: List[int],
    remainder_dir: Path,
) -> int:
    """Write the unused portion of the ablated pool to a remainder file.

    Preserves audit trail: every pool line is accounted for (used here
    vs left unused).
    """
    with open(pool_path, "r", encoding="utf-8") as f:
        pool_lines = f.readlines()

    used = set(used_indices)
    remainder = [line for i, line in enumerate(pool_lines) if i not in used]

    remainder_dir.mkdir(parents=True, exist_ok=True)
    remainder_path = remainder_dir / pool_path.name
    with open(remainder_path, "w", encoding="utf-8") as f:
        f.writelines(remainder)
    return len(remainder)


def compose(
    ablated_train: Path,
    ablated_pool: Path,
    output: Path,
    target_corpus: Optional[Path] = None,
    target_tokens_per_file: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Dict:
    """Run the compose step. Returns the manifest dict written to disk."""
    start = time.time()
    output.mkdir(parents=True, exist_ok=True)
    remainder_dir = output / "pool_remainder"

    paired = _find_paired_files(ablated_train, ablated_pool)
    if not paired:
        raise ValueError(
            f"No paired .train files between {ablated_train} and {ablated_pool}"
        )

    stems = [stem for stem, _, _ in paired]
    target_tokens = _load_target_token_counts(
        target_corpus=target_corpus,
        explicit=target_tokens_per_file,
        stems=stems,
    )

    # Fresh per-file RNG derivation so file A's sampling doesn't perturb
    # file B's — any file can be recomposed independently given the seed
    # + stem. Same pattern as ``EuroparlAlignmentGenerator`` uses.
    rng_master = random.Random(seed)

    file_stats = []
    total_train_tokens = 0
    total_pool_tokens_added = 0
    total_pool_lines_drawn = 0
    total_pool_remainder_lines = 0

    for stem, train_path, pool_path in paired:
        # Per-file seed: master seed XOR hash of stem — deterministic + isolated.
        per_file_seed = rng_master.getrandbits(64) ^ int(
            hashlib.sha256(stem.encode()).hexdigest()[:16], 16
        )
        file_rng = random.Random(per_file_seed)

        output_path = output / f"{stem}.train"
        stats = _compose_file(
            stem=stem,
            train_path=train_path,
            pool_path=pool_path,
            target_tokens=target_tokens[stem],
            output_path=output_path,
            rng=file_rng,
        )
        total_train_tokens += stats["train_tokens"]
        total_pool_tokens_added += stats["pool_tokens_added"]
        total_pool_lines_drawn += stats["pool_lines_drawn"]

        # Write pool remainder for audit
        remainder_count = _write_pool_remainder(
            pool_path=pool_path,
            used_indices=stats["pool_line_indices_used"],
            remainder_dir=remainder_dir,
        )
        total_pool_remainder_lines += remainder_count
        stats["pool_remainder_lines"] = remainder_count

        # Drop the full index list from the manifest — keep it but under
        # a nested key so the top-level file_stats list stays readable.
        stats_for_manifest = dict(stats)
        file_stats.append(stats_for_manifest)

        print(
            f"  [{stem}] train={stats['train_tokens']:,} tokens + "
            f"pool draws={stats['pool_lines_drawn']:,} lines "
            f"({stats['pool_tokens_added']:,} tokens) "
            f"→ final={stats['final_tokens']:,} / target={target_tokens[stem]:,}"
        )

    elapsed = time.time() - start

    # Compute input checksums for reproducibility
    input_checksums = {}
    for stem, train_path, pool_path in paired:
        input_checksums[f"train/{stem}.train"] = _sha256(train_path)
        input_checksums[f"pool/{stem}.train"] = _sha256(pool_path)

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "ablated_train_dir": str(ablated_train),
        "ablated_pool_dir": str(ablated_pool),
        "target_corpus": str(target_corpus) if target_corpus else None,
        "output_dir": str(output),
        "totals": {
            "files": len(paired),
            "train_tokens": total_train_tokens,
            "pool_tokens_added": total_pool_tokens_added,
            "pool_lines_drawn": total_pool_lines_drawn,
            "pool_remainder_lines": total_pool_remainder_lines,
            "processing_time_seconds": elapsed,
        },
        "per_file": file_stats,
        "input_checksums": input_checksums,
    }

    manifest_path = output / "COMPOSE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/compose_corpus.py",
        description=(
            "Compose a final ablated corpus from an ablated train "
            "directory and an ablated pool directory."
        ),
    )
    p.add_argument(
        "--ablated-train", type=Path, required=True,
        help="Directory of ablated train .train files",
    )
    p.add_argument(
        "--ablated-pool", type=Path, required=True,
        help="Directory of ablated pool .train files",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for composed corpus",
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-corpus", type=Path,
        help="Original (pre-ablation) corpus directory; per-file token counts "
             "are measured from its .train files to define the compose target.",
    )
    target.add_argument(
        "--target-tokens-per-file", type=str,
        help='Explicit per-file token targets as a JSON dict '
             '\'{"stem": int, ...}\'. Alternative to --target-corpus.',
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    explicit = None
    if args.target_tokens_per_file:
        explicit = json.loads(args.target_tokens_per_file)

    manifest = compose(
        ablated_train=args.ablated_train,
        ablated_pool=args.ablated_pool,
        output=args.output,
        target_corpus=args.target_corpus,
        target_tokens_per_file=explicit,
        seed=args.seed,
    )
    t = manifest["totals"]
    print(
        f"\nComposed {t['files']} files in "
        f"{t['processing_time_seconds']:.1f}s:"
    )
    print(
        f"  train tokens: {t['train_tokens']:,}\n"
        f"  + pool tokens: {t['pool_tokens_added']:,} "
        f"({t['pool_lines_drawn']:,} lines)\n"
        f"  pool lines remaining: {t['pool_remainder_lines']:,}"
    )
    print(f"\nManifest: {args.output / 'COMPOSE_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
