#!/usr/bin/env python3
"""Candidate eval items as a training condition's text, for a rater trained on that condition.

build_items.py stores each candidate as raw corpus line fragments (`rater_lines`, the last one cut
at the end of the target sentence) with the pronoun's offset in the last fragment
(`rater_char_start`). A rater trained on an intervention's corpus has to read the item the way that
corpus reads: this applies the intervention's token edits to the parsed test-set lines (the same
edit functions the training corpora and score_pronoun_recoverability.py --intervention use) and
cuts the same fragments out of the edited lines.

Needs the test_10M parse cache (DocBin, en_core_web_trf; k8s/job-annotate-test-10m-docbin.yaml).
Writes one row per candidate: cand_id, rater_lines (edited), rater_char_start (in the edited last
fragment), form_text (the pronoun as it stands in the edited text), n_edits (tokens changed inside
the item), rendered (False when a line has no parse or a fragment could not be located; such
items are left out of scoring).

    python analysis/recoverability/eval_stimuli/render_items_condition.py --annotated <docbin dir> \\
        --raw data/raw/test_10M --items <dir>/items_candidates.parquet \\
        --intervention enrich_verbal_morphology --out <dir>/items_candidates__enrich_verbal_morphology.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotated", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--ext", default="test")
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id-col", default="cand_id")
    a = ap.parse_args()

    import score_pronoun_recoverability as S
    from preprocessing.annotate import iter_annotated_file

    edit_fn, line_remover = S.resolve_intervention(a.intervention)
    items = pd.read_parquet(a.items)
    out_rows = []
    counts = {"items": len(items), "rendered": 0, "no_parse": 0, "fragment_not_found": 0, "pronoun_not_a_token": 0,
              "line_removed": 0}
    for stem, g in items.groupby("stem", sort=False):
        raw = [t.rstrip("\r") for t in (a.raw / f"{stem}.{a.ext}").read_text(encoding="utf-8").split("\n")]
        frags = {getattr(r, a.id_col): json.loads(r.rater_lines) for r in g.itertuples()}
        need = set()
        for r in g.itertuples():
            n = len(frags[getattr(r, a.id_col)])
            need.update(range(r.line_idx - n + 1, r.line_idx + 1))
        # per needed line: token starts, token ends, edited offsets, edited text
        info = {}
        for entry, doc in iter_annotated_file(a.annotated, stem):
            li = entry["line_idx"]
            if li not in need:
                continue
            text = entry.get("raw_text", "").rstrip("\n\r")
            if doc is None or doc.text != text:
                info[li] = None
                continue
            gone = bool(line_remover(doc)[1]) if line_remover is not None else False
            edits = (edit_fn(doc) or {}) if edit_fn is not None else {}
            etext, eoff = S.apply_edits(doc, edits)
            info[li] = (np.array([t.idx for t in doc]), np.array([t.idx + len(t.text) for t in doc]),
                        np.array([eoff[t.i] for t in doc]), [edits.get(t.i, t.text) for t in doc], etext, set(edits), gone)
        print(f"{stem}: {len(g):,} items on {len(need):,} lines ({sum(v is None for v in info.values())} without a parse)",
              flush=True)
        for r in g.itertuples():
            cid = getattr(r, a.id_col)
            fr = frags[cid]
            n = len(fr)
            row = {a.id_col: cid, "rater_lines": None, "rater_char_start": -1, "form_text": None, "n_edits": 0, "rendered": False}
            new, why, n_edits, pron_off, pron_text = [], None, 0, -1, None
            for j, frag in enumerate(fr):
                li = r.line_idx - n + 1 + j
                inf = info.get(li)
                if inf is None:
                    why = "no_parse"
                    break
                starts, ends, eoff, etoks, etext, edited, gone = inf
                if gone:
                    why = "line_removed"
                    break
                line = raw[li]
                if j == n - 1:
                    lo = int(r.char_start) - int(r.rater_char_start)       # exact: the pronoun pins it
                    if line[lo:lo + len(frag)] != frag:
                        lo = -1
                else:
                    lo = len(line) - len(frag) if line.endswith(frag) else line.find(frag)
                if lo < 0:
                    why = "fragment_not_found"
                    break
                hi = lo + len(frag)
                inside = np.flatnonzero((starts >= lo) & (ends <= hi))
                if len(inside) == 0:
                    new.append("")
                    continue
                f0, f1 = int(inside[0]), int(inside[-1])
                new.append(etext[int(eoff[f0]):int(eoff[f1]) + len(etoks[f1])])
                n_edits += sum(1 for k in range(f0, f1 + 1) if k in edited)
                if j == n - 1:
                    hit = np.flatnonzero(starts == int(r.char_start))
                    if len(hit) == 0 or not (f0 <= int(hit[0]) <= f1):
                        why = "pronoun_not_a_token"
                        break
                    p = int(hit[0])
                    pron_off, pron_text = int(eoff[p] - eoff[f0]), etoks[p]
            if why is None:
                row.update(rater_lines=json.dumps(new), rater_char_start=pron_off, form_text=pron_text, n_edits=n_edits, rendered=True)
                counts["rendered"] += 1
            else:
                counts[why] += 1
            out_rows.append(row)
    out = pd.DataFrame(out_rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    a.out.with_suffix(".meta.json").write_text(json.dumps({"intervention": a.intervention, "items": str(a.items), **counts}, indent=1))
    print(json.dumps(counts))
    ok = out[out.rendered]
    if len(ok):
        k = ok.iloc[len(ok) // 2]
        print("example:", json.loads(k.rater_lines)[-1][:160], "| pronoun:", k.form_text, "at", k.rater_char_start)


if __name__ == "__main__":
    main()
