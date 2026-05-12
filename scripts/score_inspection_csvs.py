"""Score filled-in inspection CSVs and emit the verification report tables.

Reads ``data/validation_samples/2026-04-24/inspection/*.csv`` after
annotators have filled the ``verdict`` column with ``c``/``i``/``b``,
and emits per-ablation correctness rate + Wilson 95% CI plus a
markdown table block ready to paste into §3 of the verification
report.

Usage:
    python scripts/score_inspection_csvs.py
    python scripts/score_inspection_csvs.py --markdown > tables.md
"""

from __future__ import annotations
import argparse, csv, math
from collections import Counter
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def score(csv_path: Path) -> dict:
    counts = Counter()
    n_total = 0
    n_judged = 0
    examples_incorrect = []
    examples_borderline = []
    # utf-8-sig strips the BOM that we write when generating the CSVs
    # (so that Excel correctly decodes Spanish accents). Without -sig,
    # the first column name would be "﻿row_id" instead of "row_id".
    with open(csv_path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            n_total += 1
            v = (row.get("verdict") or "").strip().lower()
            if not v:
                continue
            n_judged += 1
            if v.startswith("c"):
                counts["correct"] += 1
            elif v.startswith("i"):
                counts["incorrect"] += 1
                examples_incorrect.append(row)
            elif v.startswith("b"):
                counts["borderline"] += 1
                examples_borderline.append(row)
            else:
                counts[f"unknown:{v}"] += 1
    return {
        "path": csv_path,
        "n_total": n_total,
        "n_judged": n_judged,
        "counts": counts,
        "examples_incorrect": examples_incorrect[:10],
        "examples_borderline": examples_borderline[:10],
    }


def render_markdown(s: dict) -> str:
    name = s["path"].stem
    n = s["n_judged"]
    c = s["counts"]
    lines = [f"### {name} — N={n} judged", ""]
    if n == 0:
        lines += ["_Not yet annotated._", ""]
        return "\n".join(lines)
    lines += [
        "| Verdict | Count | Rate | 95% Wilson CI |",
        "|---|---|---|---|",
    ]
    for v in ("correct", "incorrect", "borderline"):
        k = c.get(v, 0)
        lo, hi = wilson_ci(k, n)
        lines.append(
            f"| {v.title()} | {k} | {100*k/n:.1f}% | "
            f"[{100*lo:.1f}%, {100*hi:.1f}%] |"
        )
    lines.append("")
    if c.get("incorrect", 0):
        lines.append("**Sample of incorrect cases:**")
        lines.append("")
        for ex in s["examples_incorrect"][:5]:
            orig = ex.get("original", "")[:120]
            abl = ex.get("ablated", "")[:120]
            note = ex.get("notes", "")
            lines.append(f"- `{orig}` → `{abl}`" + (f"  *({note})*" if note else ""))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path,
                   default=Path("data/validation_samples/2026-04-24/inspection"))
    p.add_argument("--markdown", action="store_true",
                   help="Emit markdown tables for the report instead of summary text")
    args = p.parse_args(argv)

    csvs = sorted(args.in_dir.glob("*.csv"))
    if not csvs:
        print(f"no CSVs at {args.in_dir}")
        return 1

    scores = [score(c) for c in csvs]
    if args.markdown:
        for s in scores:
            print(render_markdown(s))
            print()
    else:
        for s in scores:
            n = s["n_judged"]
            c = s["counts"]
            tag = s["path"].stem
            if n == 0:
                print(f"  {tag:50s}  not yet annotated")
                continue
            corr = c.get("correct", 0)
            lo, hi = wilson_ci(corr, n)
            print(f"  {tag:50s}  N={n:3d}  correct {100*corr/n:5.1f}% "
                  f"[{100*lo:.1f}%, {100*hi:.1f}%]  "
                  f"i={c.get('incorrect',0)} b={c.get('borderline',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
