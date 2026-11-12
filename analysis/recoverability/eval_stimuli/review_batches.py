#!/usr/bin/env python3
"""Hand drawn items to a reader for a type and quality check, and fold the verdicts back in.

The rater decides which decile an item is in. It cannot tell whether the pronoun really is
the subject of the stated clause type or whether the sentence is usable, so every drawn item
is also read. `export` writes the items not yet reviewed as JSON batches; `merge` collects
verdict files into reviewed.csv (one row per cand_id, kept across redraws) and rejected.txt,
which sample_sets.py takes as --exclude so a rejected item is replaced on the next draw.

    python review_batches.py export --sets <dir>/sets/robbi_v1 --out <review dir> --batch-by decile
    python review_batches.py merge  --sets <dir>/sets/robbi_v1 --verdicts <review dir>/verdicts_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TYPE_TEXT = {
    "subj_1sg": "main-clause subject (I)", "subj_1pl": "main-clause subject (we)", "subj_2": "main-clause subject (you)",
    "subj_3sg": "main-clause subject (he/she/it)", "subj_3pl": "main-clause subject (they)",
    "emb_that": "subject of a finite complement clause introduced by 'that'",
    "emb_bare": "subject of a finite complement clause that directly follows its verb (no 'that')",
    "conj": "subject of a clause joined by and/but to an earlier clause of the same sentence",
    "adv": "subject of a finite adverbial clause (if, when, because, as, ...)",
    "expl_it": "expletive 'it' (weather, 'it seems that ...', 'it is clear that ...'), not a referring 'it'",
}
PROBLEMS = ["none", "wrong_type", "garbled", "context_bad", "offensive"]


def bracketed(r) -> str:
    k = r.rater_char_start - (len(json.loads(r.rater_lines)[-1]) - len(r.target_overt))
    t = r.target_overt
    return t[:k] + "[" + t[k:k + len(r.form)] + "]" + t[k + len(r.form):]


def export(a) -> None:
    s = pd.read_parquet(a.sets / "sets.parquet")
    done = set()
    if (a.sets / "reviewed.csv").exists():
        done = set(pd.read_csv(a.sets / "reviewed.csv").cand_id)
    s = s[~s.cand_id.isin(done)]
    a.out.mkdir(parents=True, exist_ok=True)
    for f in a.out.glob("batch_*.json"):
        f.unlink()
    if a.batch_by == "decile":
        groups = [(f"d{d}", g) for d, g in s.groupby("decile")]
    else:
        n = max(1, -(-len(s) // a.batch_size))
        groups = [(f"{i:02d}", s.iloc[i::n]) for i in range(n)]
    for name, g in groups:
        items = [{"id": int(r.cand_id), "type": TYPE_TEXT[r.cell], "context": r.context.replace("\t", " "),
                  "target": bracketed(r), "without_pronoun": r.target_null} for r in g.itertuples()]
        (a.out / f"batch_{name}.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
        print(f"batch_{name}.json", len(items))
    print("to review:", len(s), "already reviewed:", len(done))


def merge(a) -> None:
    rows = []
    for f in a.verdicts:
        v = json.loads(Path(f).read_text())
        rows += v["verdicts"] if isinstance(v, dict) else v
    new = pd.DataFrame(rows).rename(columns={"id": "cand_id"})
    new = new[new.problem.isin(PROBLEMS)].drop_duplicates("cand_id", keep="last")
    # remember which cell and decile the item was drawn for (a rejected item leaves later draws)
    drawn = pd.read_parquet(a.sets / "sets.parquet")
    drawn = drawn[[c for c in ("cand_id", "cell", "decile", "tier") if c in drawn]]
    new = new.merge(drawn, on="cand_id", how="left")
    path = a.sets / "reviewed.csv"
    if path.exists():
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates("cand_id", keep="last")
    new.to_csv(path, index=False)
    bad = new[new.problem != "none"]
    (a.sets / "rejected.txt").write_text("\n".join(str(int(x)) for x in bad.cand_id) + "\n")
    print(f"reviewed {len(new)}; rejected {len(bad)} ({len(bad) / max(len(new), 1):.1%})")
    print(bad.problem.value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--sets", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--batch-by", choices=["decile", "size"], default="decile")
    e.add_argument("--batch-size", type=int, default=150)
    m = sub.add_parser("merge")
    m.add_argument("--sets", type=Path, required=True)
    m.add_argument("--verdicts", nargs="+", required=True)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else merge(a)


if __name__ == "__main__":
    main()
