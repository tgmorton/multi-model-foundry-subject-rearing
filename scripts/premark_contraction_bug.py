"""Pre-mark rows in the EN lemmatize_verbs and enrich_verbal_morphology
coding CSVs that exhibit the contraction-glue bug.

The bug glues pronouns/demonstratives/wh-words/possessives directly to a
lemmatized auxiliary (be/have/will/do), optionally with an enrich suffix
or ``n't`` also glued on. Real English never has these as single tokens,
so a tight regex catches them with very few false positives.

For each matched row:
    verdict        = "i"
    category_hit   = "bug:contraction-glue"
    notes          = "auto-marked (known contraction-glue bug)"

The annotator can still review or override; auto-marking just removes the
mechanical work of marking every contraction artefact manually. Run after
``build_inspection_csvs.py`` (or the project's first-250 sampler).

Usage:
    python scripts/premark_contraction_bug.py
    python scripts/premark_contraction_bug.py --dry-run
"""

from __future__ import annotations
import argparse, csv, re
from pathlib import Path

# Catch "<pronoun-like word><lemmatized aux/modal>[optional enrich suffix][optional n't]"
# glued into a single token. Word-boundary anchored at both ends.
GLUE_RE = re.compile(
    r"\b("
    # Subject pronouns + demonstratives + wh-words + locatives
    r"I|you|he|she|it|we|they"
    r"|that|this|these|those|what|where|when|who|how|there|here"
    # Possessives (would appear if upstream impoverish_case or similar glued)
    r"|my|our|your|his|her|its|their"
    r")"
    r"(be|have|will|do)"                       # lemmatized aux/modal
    r"(at|o|as|amus|atis|ant)?"                # optional Latin-style enrich suffix
    r"(n't)?"                                  # optional negation glued
    r"\b",
    re.IGNORECASE,
)

# Extra patterns: cases where the lemmatized aux ITSELF is glued to "n't" or
# to an enrich suffix without a preceding pronoun. The bug also produces
# "ben't" (was/were + n't), "doont" (do + 1sg -o + n't), "doant" (do + 3pl -ant
# + n't), etc.
EXTRA_RE = re.compile(
    r"\bben't\b"                                     # "wasn't"/"weren't" -> "ben't"
    r"|\bdo(o|a)nt\b|\bdo(o|a)n't\b"                 # "don't"/"doesn't" with enrich suffix glue
    r"|\b(be|have|do|will)(at|o|as|amus|atis|ant)n't\b",  # generic: aux+suffix+n't
    re.IGNORECASE,
)


def find_bug(text: str) -> bool:
    return bool(GLUE_RE.search(text) or EXTRA_RE.search(text))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir", type=Path,
        default=Path("data/validation_samples/2026-04-24/coding_guide"),
        help="Directory containing the coding CSVs",
    )
    p.add_argument(
        "--slugs", nargs="*",
        default=["lemmatize_verbs", "enrich_verbal_morphology"],
        help="Which en_<slug>.csv files to pre-mark",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report counts; do not write files.",
    )
    args = p.parse_args(argv)

    fieldnames = ["row_id", "genre", "line_num", "original", "ablated",
                  "verdict", "category_hit", "notes"]

    for slug in args.slugs:
        path = args.dir / f"coding_sheet_en_{slug}.csv"
        if not path.exists():
            print(f"  skip: {path} not found")
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        n_marked = 0
        n_pre_marked = 0
        for r in rows:
            if r.get("verdict", "").strip():
                # Already has a verdict (manual review or prior auto-mark) — leave it
                n_pre_marked += 1
                continue
            if find_bug(r["ablated"]):
                r["verdict"] = "i"
                r["category_hit"] = "bug:contraction-glue"
                r["notes"] = "auto-marked (known contraction-glue bug)"
                n_marked += 1
        total = len(rows)
        print(f"  {slug:30s}  total={total}  new auto-marked={n_marked}  "
              f"already-had-verdict={n_pre_marked}  remaining-blank="
              f"{total - n_marked - n_pre_marked}")
        if not args.dry_run:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    main()
