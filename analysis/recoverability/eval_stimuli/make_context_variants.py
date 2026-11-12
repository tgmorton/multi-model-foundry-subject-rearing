#!/usr/bin/env python3
"""Synthetic variants of test-set items: the same target sentence with less of its real context.

Where a decile cannot be filled from items as they come, the first thing to vary is how much
of the preceding corpus text the item carries. The target and its pronoun are untouched and
every word of the context is still the text that preceded it in the corpus; only the starting
point moves forward, to a line start or a sentence boundary. Less context usually makes the
pronoun less recoverable, so one sentence can supply several deciles. Each variant is scored
by score_items.py like any other item (`--id-col var_id`).

`--pronoun-free` builds the pool for a pronoun-free version instead: per item the longest
stretch of real context, ending at the target, in which no pronoun occurs.

Output: the candidate columns the scorer and sampler need, plus
  var_id         new id (cand_id * 100 + k)
  parent_cand_id the item it was cut from
  context_words  words of context left

    python make_context_variants.py --items <dir>/items_candidates.parquet --ptype expletive_it \\
        --out <dir>/variants_expletive_it.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BOUNDARY = re.compile(r"[.!?…][\"'”’)\]]*\s+")
WORD = re.compile(r"[a-z]+")
# Same lists as scripts/build_nopron_context_stimuli.py (Thomas, 2026-10-21): personal,
# possessive and reflexive pronouns, and words that can also function as pronouns.
PRONOUNS = frozenset(
    "i me my mine myself you your yours yourself yourselves he him his himself "
    "she her hers herself it its itself we us our ours ourselves they them "
    "their theirs themselves".split())
AMBIGUOUS = frozenset(
    "one ones this that these those some any all each both either neither none "
    "who whom whose which what whoever whatever everyone everybody everything "
    "someone somebody something anyone anybody anything nobody nothing noone "
    "another other others such few many several much more most less least half "
    "enough plenty lot lots own else there".split())
BANNED = {"personal": PRONOUNS, "strict": PRONOUNS | AMBIGUOUS}


def pronoun_free(text: str, banned) -> bool:
    return not any(w in banned for w in WORD.findall(text.lower().replace("’", "'")))


def cuts(frags: list[str], target_len: int):
    """(fragment index, offset) of every place a shorter context may start."""
    out = []
    for j, f in enumerate(frags):
        limit = len(f) - target_len if j == len(frags) - 1 else len(f)   # never cut into the target
        if j > 0:
            out.append((j, 0))
        for m in BOUNDARY.finditer(f):
            if m.end() < limit:
                out.append((j, m.end()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ptype", nargs="*", default=None, help="restrict to these phenomenon types")
    ap.add_argument("--cand-ids", type=Path, default=None, help="restrict to cand_ids listed in this file")
    ap.add_argument("--min-context-words", type=int, default=4)
    ap.add_argument("--max-variants", type=int, default=5)
    ap.add_argument("--pronoun-free", choices=sorted(BANNED), default=None,
                    help="instead of a spread of lengths, keep per item the longest context (the whole one "
                         "included) that contains no pronoun of this list")
    ap.add_argument("--id-offset", type=int, default=0, help="var_id = cand_id * 100 + offset + k")
    a = ap.parse_args()

    d = pd.read_parquet(a.items)
    if a.ptype:
        d = d[d.ptype.isin(a.ptype)]
    if a.cand_ids:
        d = d[d.cand_id.isin({int(x) for x in a.cand_ids.read_text().split()})]
    rows = []
    for r in d.itertuples(index=False):
        frags = json.loads(r.rater_lines)
        tlen = len(r.target_overt)
        tstart = len(frags[-1]) - tlen
        opts = []
        for j, o in cuts(frags, tlen):
            new = [frags[j][o:]] + frags[j + 1:]
            ctx = new[:-1] + [new[-1][: len(new[-1]) - tlen]]
            context = " ".join(f.strip() for f in ctx if f.strip())
            n = len(context.split())
            if a.min_context_words <= n < r.context_words:
                opts.append((n, new, context, r.rater_char_start - (o if j == len(frags) - 1 else 0)))
        if a.pronoun_free:
            whole = (r.context_words, frags, r.context, r.rater_char_start)
            ok = [o for o in opts + [whole] if pronoun_free(o[2], BANNED[a.pronoun_free])]
            opts = [max(ok, key=lambda x: x[0])] if ok else []
        if not opts:
            continue
        # spread the kept variants over the range of context lengths
        opts.sort(key=lambda x: x[0])
        if len(opts) > a.max_variants:
            idx = sorted({round(i * (len(opts) - 1) / (a.max_variants - 1)) for i in range(a.max_variants)})
            opts = [opts[i] for i in idx]
        for k, (n, new, context, start) in enumerate(opts, 1):
            assert new[-1][start:start + len(r.form)].lower() == r.form, (r.cand_id, new[-1], start)
            rec = r._asdict()
            rec.update({"var_id": int(r.cand_id) * 100 + a.id_offset + k, "parent_cand_id": int(r.cand_id), "context": context,
                        "context_words": n, "rater_lines": json.dumps(new), "rater_char_start": int(start),
                        "context_lines": len(new) - 1})
            rows.append(rec)
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    print(f"{len(d):,} items -> {len(out):,} variants from {out.parent_cand_id.nunique():,} items")


if __name__ == "__main__":
    main()
