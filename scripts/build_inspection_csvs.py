"""Build per-ablation inspection CSVs.

For each <lang>_<slug>.jsonl in
``data/validation_samples/2026-04-24/large_random_samples/``, write the
first ``--n`` rows to a CSV at
``data/validation_samples/2026-04-24/inspection/<lang>_<slug>.csv``
with columns: ``row_id, genre, line_num, original, ablated, verdict, notes``.

Annotators fill the ``verdict`` column with ``c`` (correct), ``i``
(incorrect), or ``b`` (borderline). The ``notes`` column is free-form.

The full 900-row JSONL stays as the deposit pool; the inspection CSV is
the operational subset.
"""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path,
                   default=Path("data/validation_samples/2026-04-24/large_random_samples"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("data/validation_samples/2026-04-24/inspection"))
    p.add_argument("--n", type=int, default=100,
                   help="Rows per ablation (default 100)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.in_dir.glob("*.jsonl"))
    if not sources:
        print(f"no JSONLs at {args.in_dir}")
        return 1

    for jf in sources:
        rows = []
        with open(jf) as f:
            for i, line in enumerate(f):
                if i >= args.n:
                    break
                r = json.loads(line)
                rows.append(r)
        out = args.out_dir / f"{jf.stem}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "genre", "line_num", "original", "ablated",
                        "verdict", "notes"])
            for i, r in enumerate(rows):
                w.writerow([i, r.get("genre", ""), r.get("line_num", ""),
                            r.get("original", ""), r.get("ablated", ""),
                            "", ""])
        print(f"  wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
