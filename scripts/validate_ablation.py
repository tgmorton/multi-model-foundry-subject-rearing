#!/usr/bin/env python3
"""
Validate an ablation by producing a stratified i.i.d. sample for hand review.

Three sampling modes:

1. ``--mode token`` (default) — token-replacement ablations. Line N in
   the original matches line N in the ablated file; diff side-by-side.
2. ``--mode line-removal`` — line-removal ablations where the ablated
   file is shorter. Sample from the original; label each as
   ``<KEPT>`` / ``<REMOVED>`` based on set membership in the ablated file.
3. ``--mode three-step`` — the compose workflow (ablate train + pool
   separately, then compose). Reads ``COMPOSE_MANIFEST.json`` from
   ``--composed``; for each genre samples both from the train side
   (to audit the ablation itself) AND from the pool-backfill lines
   (to audit what's being appended). Requires ``--original``,
   ``--composed``, and manifest metadata from the compose step.

All modes write a TSV with columns suitable for manual annotation:

    genre | source | line_num | original | ablated | correct? | notes

The ``correct?`` and ``notes`` columns are left empty for the reviewer to fill in.

Usage
-----
Token-replacement::

    python scripts/validate_ablation.py --mode token \\
        --original data/raw/train_90M/ \\
        --ablated  data/processed/exp_impoverish_case_es/ \\
        --n-per-genre 20 \\
        --output validation/impoverish_case_es.tsv

Line-removal::

    python scripts/validate_ablation.py --mode line-removal \\
        --original data/raw/train_90M/ \\
        --ablated  data/processed/exp_remove_expletives_en/ \\
        --n-per-genre 20 \\
        --output validation/remove_expl.tsv

Three-step (compose workflow)::

    python scripts/validate_ablation.py --mode three-step \\
        --original  data/spanish/train_90M/ \\
        --composed  data/processed/exp_remove_expl_es/ \\
        --ablated-pool data/processed/exp_remove_expl_es_pool/ \\
        --n-per-genre 30 \\
        --output validation/remove_expl_es.tsv

The ``source`` column distinguishes:
- ``train-kept``     — line survived the ablation in train
- ``train-removed``  — line was removed from train
- ``train-modified`` — (token mode) line differs between original and ablated
- ``pool-backfill``  — line was drawn from ablated pool to hit target size

The script also prints a summary of per-genre ablation statistics (lines
kept/removed, character-level deltas) to stdout.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


def find_matching_files(original_dir: Path, ablated_dir: Path):
    """
    Yield (genre_name, original_path, ablated_path) for each file present
    in both directories.  Matches by filename.
    """
    orig_files = {p.name: p for p in sorted(original_dir.iterdir()) if p.is_file()}
    abl_files = {p.name: p for p in sorted(ablated_dir.iterdir()) if p.is_file()}

    for name in sorted(orig_files.keys() & abl_files.keys()):
        # Skip non-data files (manifests, logs, etc.)
        if name.startswith(".") or name.endswith(".json"):
            continue
        genre = Path(name).stem
        yield genre, orig_files[name], abl_files[name]


def sample_lines_token(original_path: Path, ablated_path: Path, n: int, rng: random.Random):
    """
    Sample up to *n* aligned (original, ablated) line pairs at random indices.

    For token-replacement ablations where line counts are preserved, the indices
    map 1-to-1 between original and ablated files.
    """
    with open(original_path, "r", encoding="utf-8") as f:
        orig_lines = f.readlines()
    with open(ablated_path, "r", encoding="utf-8") as f:
        abl_lines = f.readlines()

    # Determine how many lines we can sample
    max_idx = len(orig_lines) - 1
    if max_idx < 0:
        return []

    sample_size = min(n, max_idx + 1)
    indices = sorted(rng.sample(range(max_idx + 1), sample_size))

    pairs = []
    for idx in indices:
        orig = orig_lines[idx].rstrip("\n")
        abl = abl_lines[idx].rstrip("\n") if idx < len(abl_lines) else "<LINE_REMOVED>"
        pairs.append((idx + 1, orig, abl))  # 1-indexed for human readability

    return pairs


def sample_lines_line_removal(original_path: Path, ablated_path: Path, n: int, rng: random.Random):
    """
    Sample up to *n* lines from the original file and label each as
    ``<KEPT>`` or ``<REMOVED>`` based on whether the line appears in the
    ablated file.

    For line-removal ablations, original line N does NOT correspond to ablated
    line N (removed lines shift all subsequent indices, and replacement pool
    lines may be appended).  Instead, we check set membership.

    Note: duplicate lines in the original may produce false-positive KEPT
    labels.  For a small i.i.d. sample this is acceptable.
    """
    with open(original_path, "r", encoding="utf-8") as f:
        orig_lines = f.readlines()
    with open(ablated_path, "r", encoding="utf-8") as f:
        abl_lines_set = set(line.rstrip("\n") for line in f.readlines())

    max_idx = len(orig_lines) - 1
    if max_idx < 0:
        return []

    sample_size = min(n, max_idx + 1)
    indices = sorted(rng.sample(range(max_idx + 1), sample_size))

    pairs = []
    for idx in indices:
        orig = orig_lines[idx].rstrip("\n")
        label = "<KEPT>" if orig in abl_lines_set else "<REMOVED>"
        pairs.append((idx + 1, orig, label))

    return pairs


def sample_three_step(
    original_path: Path,
    composed_path: Path,
    ablated_pool_path: Path,
    manifest_entry: dict,
    n: int,
    rng: random.Random,
):
    """Stratified sample for the three-step (ablate + compose) workflow.

    Splits the N samples between two populations:
    - Train-side: sample from the original raw corpus. For each sampled
      line, check if it survived in the composed output (set membership,
      same as line-removal mode).
    - Pool-side: sample from the list of pool line indices that were
      actually drawn (``pool_line_indices_used`` in the compose manifest).
      Show the ablated pool line verbatim.

    Returns tuples of ``(source, line_num, original_text, final_text)``.
    """
    with open(original_path, "r", encoding="utf-8") as f:
        orig_lines = f.readlines()
    with open(composed_path, "r", encoding="utf-8") as f:
        composed_set = set(line.rstrip("\n") for line in f.readlines())

    # 60/40 split favoring train-side review (more lines, more interesting)
    n_train = max(1, int(n * 0.6))
    n_pool = n - n_train

    pairs = []

    # Train-side samples
    if orig_lines:
        sample_size = min(n_train, len(orig_lines))
        indices = sorted(rng.sample(range(len(orig_lines)), sample_size))
        for idx in indices:
            orig = orig_lines[idx].rstrip("\n")
            label = "train-kept" if orig in composed_set else "train-removed"
            # For removed lines, the "ablated" column is a clear marker.
            ablated = orig if label == "train-kept" else "<REMOVED>"
            pairs.append((label, idx + 1, orig, ablated))

    # Pool-side samples: draw from the used indices list.
    used_indices = manifest_entry.get("pool_line_indices_used", [])
    if used_indices and n_pool > 0:
        with open(ablated_pool_path, "r", encoding="utf-8") as f:
            pool_lines = f.readlines()
        sample_size = min(n_pool, len(used_indices))
        picked = rng.sample(used_indices, sample_size)
        for idx in sorted(picked):
            ablated = pool_lines[idx].rstrip("\n") if idx < len(pool_lines) else "<POOL_LINE_MISSING>"
            # No single "original" for pool-backfill rows — the original
            # *pool* line is one layer back (the raw pool we annotated).
            # Since reviewers care about whether the ablated pool line
            # reads as acceptable Spanish, show the ablated line in both
            # the "original" and "ablated" columns of the TSV — the source
            # label makes the context obvious.
            pairs.append(("pool-backfill", idx + 1, "<pool sample>", ablated))

    return pairs


def compute_stats(original_path: Path, ablated_path: Path):
    """Return a dict of basic statistics comparing original vs ablated."""
    with open(original_path, "r", encoding="utf-8") as f:
        orig_lines = f.readlines()
    with open(ablated_path, "r", encoding="utf-8") as f:
        abl_lines = f.readlines()

    orig_chars = sum(len(l) for l in orig_lines)
    abl_chars = sum(len(l) for l in abl_lines)

    return {
        "orig_lines": len(orig_lines),
        "abl_lines": len(abl_lines),
        "lines_delta": len(abl_lines) - len(orig_lines),
        "orig_chars": orig_chars,
        "abl_chars": abl_chars,
        "char_delta_pct": ((abl_chars - orig_chars) / orig_chars * 100) if orig_chars else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Produce a stratified i.i.d. sample for hand review of an ablation."
    )
    parser.add_argument(
        "--original", required=True, type=Path,
        help="Directory with original (pre-ablation) corpus files.",
    )
    parser.add_argument(
        "--ablated", type=Path,
        help="Directory with ablated corpus files (modes: token, line-removal).",
    )
    parser.add_argument(
        "--composed", type=Path,
        help="Directory with composed corpus files — mode=three-step only.",
    )
    parser.add_argument(
        "--ablated-pool", type=Path,
        help="Directory with ablated pool files — mode=three-step only.",
    )
    parser.add_argument(
        "--n-per-genre", type=int, default=20,
        help="Number of lines to sample per genre file (default: 20).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--mode",
        choices=["token", "line-removal", "three-step"],
        default="token",
        help=(
            "Sampling mode. 'token' (default) compares original and ablated "
            "lines at the same index. 'line-removal' samples from the "
            "original and labels each line as <KEPT> or <REMOVED>. "
            "'three-step' reads COMPOSE_MANIFEST.json and samples from "
            "both the train side and the pool-backfill lines, adding a "
            "`source` column to the TSV."
        ),
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output TSV file path.",
    )
    args = parser.parse_args()

    # Validate mode-specific arguments
    if not args.original.is_dir():
        print(f"Error: original directory not found: {args.original}", file=sys.stderr)
        sys.exit(1)
    if args.mode in ("token", "line-removal"):
        if args.ablated is None:
            print(f"Error: mode={args.mode} requires --ablated", file=sys.stderr)
            sys.exit(1)
        if not args.ablated.is_dir():
            print(f"Error: ablated directory not found: {args.ablated}", file=sys.stderr)
            sys.exit(1)
    if args.mode == "three-step":
        if args.composed is None or args.ablated_pool is None:
            print(
                "Error: mode=three-step requires --composed and --ablated-pool",
                file=sys.stderr,
            )
            sys.exit(1)
        manifest_path = args.composed / "COMPOSE_MANIFEST.json"
        if not manifest_path.exists():
            print(
                f"Error: no COMPOSE_MANIFEST.json at {manifest_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    all_stats = []

    # ----------------------------------------------------------------------
    # Mode: three-step (compose workflow)
    # ----------------------------------------------------------------------
    if args.mode == "three-step":
        manifest = json.loads((args.composed / "COMPOSE_MANIFEST.json").read_text())
        per_file = {entry["stem"]: entry for entry in manifest["per_file"]}

        for genre in sorted(per_file.keys()):
            orig_path = args.original / f"{genre}.train"
            composed_path = args.composed / f"{genre}.train"
            pool_path = args.ablated_pool / f"{genre}.train"
            if not (orig_path.exists() and composed_path.exists() and pool_path.exists()):
                print(
                    f"[warn] {genre}: missing file in original/composed/pool; skipping",
                    file=sys.stderr,
                )
                continue

            pairs = sample_three_step(
                original_path=orig_path,
                composed_path=composed_path,
                ablated_pool_path=pool_path,
                manifest_entry=per_file[genre],
                n=args.n_per_genre,
                rng=rng,
            )
            for source, line_num, orig, abl in pairs:
                all_samples.append((genre, source, line_num, orig, abl))

            stats = compute_stats(orig_path, composed_path)
            stats["genre"] = genre
            all_stats.append(stats)

    # ----------------------------------------------------------------------
    # Modes: token or line-removal (pre-existing)
    # ----------------------------------------------------------------------
    else:
        matches = list(find_matching_files(args.original, args.ablated))
        if not matches:
            print(
                "Error: no matching files found between the two directories.",
                file=sys.stderr,
            )
            sys.exit(1)

        sample_fn = (
            sample_lines_line_removal if args.mode == "line-removal"
            else sample_lines_token
        )
        for genre, orig_path, abl_path in matches:
            pairs = sample_fn(orig_path, abl_path, args.n_per_genre, rng)
            for line_num, orig, abl in pairs:
                # Older modes don't carry a `source` label; mark as
                # the inferred status so the TSV schema is consistent.
                if args.mode == "token":
                    source = "train-modified" if orig != abl else "train-kept"
                else:
                    source = (
                        "train-removed" if abl == "<REMOVED>" else "train-kept"
                    )
                all_samples.append((genre, source, line_num, orig, abl))

            stats = compute_stats(orig_path, abl_path)
            stats["genre"] = genre
            all_stats.append(stats)

    # Write TSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "genre", "source", "line_num", "original", "ablated",
            "correct?", "notes",
        ])
        for genre, source, line_num, orig, abl in all_samples:
            writer.writerow([genre, source, line_num, orig, abl, "", ""])

    print(f"Wrote {len(all_samples)} sample rows to {args.output}\n")

    # Print summary
    print("=== Per-Genre Ablation Statistics ===\n")
    print(f"{'Genre':<20} {'Orig Lines':>12} {'Abl Lines':>12} {'Δ Lines':>10} {'Δ Chars %':>10}")
    print("-" * 68)
    for s in all_stats:
        print(
            f"{s['genre']:<20} {s['orig_lines']:>12,} {s['abl_lines']:>12,} "
            f"{s['lines_delta']:>10,} {s['char_delta_pct']:>9.2f}%"
        )


if __name__ == "__main__":
    main()
