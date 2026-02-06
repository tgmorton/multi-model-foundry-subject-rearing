#!/usr/bin/env python3
"""
Validate an ablation by producing a stratified i.i.d. sample for hand review.

For each genre file present in both the original and ablated directories, this
script randomly samples N line indices (seeded) and writes an aligned TSV with
columns suitable for manual annotation:

    genre | line_num | original | ablated | correct? | notes

The ``correct?`` and ``notes`` columns are left empty for the reviewer to fill in.

Usage
-----
::

    python scripts/validate_ablation.py \\
        --original data/raw/train_90M/ \\
        --ablated  data/processed/exp_remove_expletive_sentences_en/ \\
        --n-per-genre 20 \\
        --seed 42 \\
        --output validation/remove_expletive_sentences_en_sample.tsv

The script also prints a summary of per-genre ablation statistics (lines
kept/removed, character-level deltas) to stdout.
"""

import argparse
import csv
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


def sample_lines(original_path: Path, ablated_path: Path, n: int, rng: random.Random):
    """
    Sample up to *n* aligned (original, ablated) line pairs at random indices.

    If the ablated file has fewer lines (because whole-line ablation removed
    some), we sample from the *original* file indices and pair with the ablated
    line at the same index — which may be empty / different due to the ablation
    pipeline's line-level processing.

    For token-replacement ablations where line counts are preserved, the indices
    map 1-to-1.
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
        "--ablated", required=True, type=Path,
        help="Directory with ablated corpus files.",
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
        "--output", required=True, type=Path,
        help="Output TSV file path.",
    )
    args = parser.parse_args()

    # Validate directories
    if not args.original.is_dir():
        print(f"Error: original directory not found: {args.original}", file=sys.stderr)
        sys.exit(1)
    if not args.ablated.is_dir():
        print(f"Error: ablated directory not found: {args.ablated}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    matches = list(find_matching_files(args.original, args.ablated))
    if not matches:
        print("Error: no matching files found between the two directories.", file=sys.stderr)
        sys.exit(1)

    # Collect samples and stats
    all_samples = []
    all_stats = []

    for genre, orig_path, abl_path in matches:
        pairs = sample_lines(orig_path, abl_path, args.n_per_genre, rng)
        for line_num, orig, abl in pairs:
            all_samples.append((genre, line_num, orig, abl))

        stats = compute_stats(orig_path, abl_path)
        stats["genre"] = genre
        all_stats.append(stats)

    # Write TSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["genre", "line_num", "original", "ablated", "correct?", "notes"])
        for genre, line_num, orig, abl in all_samples:
            writer.writerow([genre, line_num, orig, abl, "", ""])

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
