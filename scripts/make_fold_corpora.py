"""
Split a corpus directory into two document-level folds (A/B) for training
cross-fold rater models.

Purpose (recoverability scoring, 2026-08): every subject-pronoun instance in
train_90M must be scored by a model that never saw it in training. Rater A
trains on fold A and scores fold B, and vice versa. The split must therefore
be at DOCUMENT granularity — a line-level split would leak near-duplicate
discourse context (the rest of the same conversation/book) into the rater's
training fold and defeat the held-out design.

Mechanics, per *.train file in --input:
- If the file contains document boundary markers ("= = = <id> = = =" lines,
  as in childes/gutenberg/simple_wiki), documents alternate A/B in file
  order; the marker line travels with its document.
- Otherwise (no markers), contiguous blocks of --block-lines lines alternate
  A/B — approximate documents for conversation-stream genres.

Deterministic: no RNG anywhere; re-runs are byte-identical. Writes
FOLD_MANIFEST.json into each output dir with per-file line/word counts.

Usage:
  python scripts/make_fold_corpora.py \
      --input data/raw/en/train_90M \
      --output-a data/raw/en/train_90M_fold_a \
      --output-b data/raw/en/train_90M_fold_b
"""

import argparse
import json
from pathlib import Path

MIN_MARKERS_FOR_DOC_MODE = 10


def is_boundary_marker(line: str) -> bool:
    s = line.strip()
    return len(s) > 10 and s.startswith("= = =") and s.endswith("= = =")


def count_markers(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if is_boundary_marker(line):
                n += 1
    return n


def split_file(path: Path, out_a: Path, out_b: Path, block_lines: int) -> dict:
    n_markers = count_markers(path)
    doc_mode = n_markers >= MIN_MARKERS_FOR_DOC_MODE

    stats = {
        "file": path.name,
        "mode": "document" if doc_mode else f"block:{block_lines}",
        "n_markers": n_markers,
        "units": 0,
        "a": {"lines": 0, "words": 0},
        "b": {"lines": 0, "words": 0},
    }

    unit_idx = -1  # first unit becomes 0 -> fold A
    lines_in_block = 0
    with open(path, encoding="utf-8", errors="replace") as f, \
         open(out_a / path.name, "w", encoding="utf-8") as fa, \
         open(out_b / path.name, "w", encoding="utf-8") as fb:
        for line in f:
            if doc_mode:
                if is_boundary_marker(line) or unit_idx < 0:
                    unit_idx += 1
            else:
                if unit_idx < 0 or lines_in_block >= block_lines:
                    unit_idx += 1
                    lines_in_block = 0
                lines_in_block += 1

            key = "a" if unit_idx % 2 == 0 else "b"
            (fa if key == "a" else fb).write(line)
            stats[key]["lines"] += 1
            stats[key]["words"] += len(line.split())

    stats["units"] = unit_idx + 1
    return stats


def emit_assignments(input_dir: Path, out_dir: Path, block_lines: int) -> None:
    """Write line_idx -> fold parquets by re-walking the split decision
    (deterministic; must run against the SAME files split_file consumed).
    Used to stitch cross-fold rater scores: an instance's clean score comes
    from the rater trained on the OTHER fold."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.train")):
        doc_mode = count_markers(path) >= MIN_MARKERS_FOR_DOC_MODE
        folds = []
        unit_idx = -1
        lines_in_block = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if doc_mode:
                    if is_boundary_marker(line) or unit_idx < 0:
                        unit_idx += 1
                else:
                    if unit_idx < 0 or lines_in_block >= block_lines:
                        unit_idx += 1
                        lines_in_block = 0
                    lines_in_block += 1
                folds.append("a" if unit_idx % 2 == 0 else "b")
        pq.write_table(
            pa.table({"line_idx": list(range(len(folds))), "fold": folds}),
            out_dir / f"{path.stem}.parquet",
        )
        print(f"{path.name}: {len(folds):,} lines "
              f"(a={folds.count('a'):,} b={folds.count('b'):,})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-a", type=Path)
    ap.add_argument("--output-b", type=Path)
    ap.add_argument("--block-lines", type=int, default=200)
    ap.add_argument("--assignment-out", type=Path,
                    help="Emit line_idx->fold parquets instead of writing "
                         "fold corpora (same decision walk).")
    args = ap.parse_args()

    if args.assignment_out:
        emit_assignments(args.input, args.assignment_out, args.block_lines)
        return
    if not (args.output_a and args.output_b):
        ap.error("--output-a/--output-b required unless --assignment-out")

    files = sorted(args.input.glob("*.train"))
    if not files:
        raise SystemExit(f"No *.train files in {args.input}")
    args.output_a.mkdir(parents=True, exist_ok=True)
    args.output_b.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for path in files:
        st = split_file(path, args.output_a, args.output_b, args.block_lines)
        all_stats.append(st)
        print(f"{st['file']}: mode={st['mode']} units={st['units']:,} "
              f"A={st['a']['words']:,}w B={st['b']['words']:,}w", flush=True)

    manifest = {
        "source": str(args.input),
        "block_lines": args.block_lines,
        "files": all_stats,
        "totals": {
            k: {
                "lines": sum(s[k]["lines"] for s in all_stats),
                "words": sum(s[k]["words"] for s in all_stats),
            }
            for k in ("a", "b")
        },
    }
    for out in (args.output_a, args.output_b):
        with open(out / "FOLD_MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["totals"], indent=2))


if __name__ == "__main__":
    main()
