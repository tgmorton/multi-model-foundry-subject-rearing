#!/usr/bin/env python3
"""Score every pronoun in an eval stimulus root with the corpus rater.

Places the pronouns of our own evaluation items on the recoverability scale that
defined the training deciles (selection v5, label `robbi`: roberta-large, +-250
tokens, all pieces of the pronoun masked, surprisal = -sum log P(true piece)).
The masking and scoring are `phase_b_mlm` from scripts/score_pronoun_recoverability.py,
called once per item, so the numbers are on the corpus scale by construction.

Each item is scored under four renderings, because the corpus was scored as
natural cased text with one utterance or paragraph per line, and our stimuli are
lowercased with spaced punctuation:

  render   as_is      the stimulus text exactly as the learners see it
           natural    truecased and detokenized ("We reviewed the new contract.")
  layout   two_lines  context and target tokenized as separate corpus lines and
                      concatenated with no separator (how utterance corpora were
                      scored: a line-initial pronoun has no leading-space piece)
           one_line   context + " " + target as a single line (running prose)

`natural` + `two_lines` is the closest match to how the corpus was scored.

Output: one row per (item, pronoun token, render, layout) with the surprisal, the
decile it falls in, the inventory probabilities at the slot, and surface features
named as in analysis/recoverability/survey (sent_initial, line_initial, after_and ...).

    python analysis/recoverability/eval_stimuli/score_eval_stimuli.py \\
        --stimuli evaluation/stimuli/null-subj-v2-matched-v2/baseline/en \\
        --out eval_pronouns_robbi_baseline.parquet
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# selection_v5/robbi/SELECTION_MANIFEST.json: upper surprisal bound (nats) of deciles 0..8.
ROBBI_THRESHOLDS = [0.0029813863802701235, 0.008927312679588795, 0.02135789766907692,
                    0.04446164146065712, 0.08588272333145142, 0.16255170106887817,
                    0.31855273246765137, 0.6843868494033813, 1.7035140991210938]
SUBJECT_FORMS = {"i", "you", "he", "she", "it", "we", "they"}
OTHER_FORMS = {"me", "him", "her", "us", "them"}
SENT_PUNCT = {".", "!", "?", "...", ";"}
CONJ = {"and", "but", "so", "or"}
WH = {"who", "what", "where", "when", "why", "how", "which", "whom", "whose"}
AUX_INV = {"do", "does", "did", "is", "are", "was", "were", "am", "have", "has", "had", "will",
           "would", "can", "could", "should", "shall", "may", "might", "must"}


def critical_index(overt: list[str], null: list[str]) -> int:
    """Index in `overt` of the one token the null version lacks."""
    assert len(overt) == len(null) + 1, (overt, null)
    i = 0
    while i < len(null) and overt[i] == null[i]:
        i += 1
    assert overt[:i] + overt[i + 1:] == null, (overt, null)
    return i


def render(tokens: list[str], names: set[str], natural: bool) -> tuple[str, list[int]]:
    """Text plus the character offset of each input token."""
    out, starts = "", []
    for k, t in enumerate(tokens):
        w = t
        if natural:
            if t == "i":
                w = "I"
            elif t in names:
                w = t.capitalize()
            if k == 0 or tokens[k - 1] in SENT_PUNCT:
                w = w[:1].upper() + w[1:]
            # "the artist ' s work": the apostrophe and the clitic attach leftwards
            glue = t in {".", ",", "?", "!", "'"} or (k > 0 and tokens[k - 1] == "'")
        else:
            glue = False
        if out and not glue:
            out += " "
        starts.append(len(out))
        out += w
    return out, starts


def features(tokens: list[str], i: int, in_target: bool, has_context: bool) -> dict:
    l1 = tokens[i - 1] if i > 0 else ""
    l2 = tokens[i - 2] if i > 1 else ""
    r1 = tokens[i + 1] if i + 1 < len(tokens) else ""
    inverted = l1 in AUX_INV
    return {
        "l1": l1, "r1": r1,
        "line_initial": i == 0,                     # first token of its sentence line
        "sent_initial": i == 0 or l1 in SENT_PUNCT,
        "item_initial": i == 0 and not (in_target and has_context),
        "after_and": l1 in CONJ,
        "after_wh": l1 in WH,
        "inverted": inverted,
        "wh_inverted": inverted and l2 in WH,
        "after_that": l1 == "that",
        "after_comma": l1 == ",",
        "line_ends_q": tokens[-1] == "?",
        "tokens_before": i,
        "line_len_tokens": len(tokens),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stimuli", type=Path, required=True, help="directory of category CSVs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-mlm", default="roberta-large")
    ap.add_argument("--label", default="robbi")
    ap.add_argument("--mlm-ctx", type=int, default=250)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pairs (smoke test)")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import score_pronoun_recoverability as S

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.hf_mlm)
    aligner = S.HFAligner(tok)
    model = AutoModelForMaskedLM.from_pretrained(a.hf_mlm, attn_implementation="eager").to(device).eval()
    inv_ids = [aligner.inventory_ids(f) for f in S.INVENTORY_FORMS]
    inv_group = np.repeat(np.arange(len(inv_ids)), [len(g) for g in inv_ids])
    print(f"model {a.hf_mlm} on {device}; inventory {dict(zip(S.INVENTORY_FORMS, inv_ids))}", flush=True)

    d = pd.concat([pd.read_csv(f) for f in sorted(a.stimuli.glob("*.csv"))], ignore_index=True)
    pairs = []
    for (cat, cond, item), g in d.groupby(["category", "condition", "item_id"], sort=False):
        o, n = g[g.pronoun_status == 1].iloc[0], g[g.pronoun_status == 0].iloc[0]
        pairs.append((cat, cond, int(item), o, n))
    if a.limit:
        pairs = pairs[:a.limit]

    rows, misaligned = [], 0
    for cat, cond, item, o, n in pairs:
        ctx = str(o.context).split() if isinstance(o.context, str) else []
        tgt, null = str(o.target).split(), str(n.target).split()
        crit = critical_index(tgt, null)
        names = {x for x in re.split(r"[;,| ]+", str(o.names)) if x and x != "nan"}
        sents = [("context", ctx), ("target", tgt)] if ctx else [("target", tgt)]
        todo = []   # (sentence name, token index, role)
        for sname, toks in sents:
            for i, t in enumerate(toks):
                is_crit = sname == "target" and i == crit
                if is_crit or t in SUBJECT_FORMS or t in OTHER_FORMS:
                    todo.append((sname, i, "critical" if is_crit else "other"))
        for natural in (False, True):
            rend = {s: render(toks, names, natural) for s, toks in sents}
            for layout in ("two_lines", "one_line"):
                if layout == "two_lines":
                    lines_txt = [rend[s][0] for s, _ in sents]
                    loc = {s: (k, 0) for k, (s, _) in enumerate(sents)}       # (line, char shift)
                else:
                    joined, shift, loc = "", 0, {}
                    for s, _ in sents:
                        if joined:
                            joined += " "
                        loc[s] = (0, len(joined))
                        joined += rend[s][0]
                    lines_txt = [joined]
                lines = [{"line_idx": k, "ids": aligner.encode_ids(t)} for k, t in enumerate(lines_txt)]
                insts, meta = [], []
                for sname, i, role in todo:
                    toks = dict(sents)[sname]
                    li, shift = loc[sname]
                    start = rend[sname][1][i] + shift
                    text = lines_txt[li]
                    word = text[start:start + len(toks[i])]
                    al = aligner.align_line(text, [S._Shim(start, word)])[0]
                    if al is None:
                        misaligned += 1
                        continue
                    insts.append({"line_idx": li, "piece_in_line": al[0], "n_pieces": al[1]})
                    meta.append((sname, i, role, word, al[1], al[2]))
                if not insts:
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    res, _ = S.phase_b_mlm(lines, insts, a.label, a.hf_mlm, tok, inv_ids, device, a.mlm_ctx,
                                           batch_size=64, use_amp=False, counters={}, model=model)
                r = res[a.label]
                for j, (sname, i, role, word, npieces, word_initial) in enumerate(meta):
                    toks = dict(sents)[sname]
                    surp = float(-r["logprob_sum_arr"][j])
                    inv = np.bincount(inv_group, weights=r["inv"][j], minlength=len(inv_ids))
                    rows.append({
                        "category": cat, "condition": cond, "item_id": item,
                        "render": "natural" if natural else "as_is", "layout": layout,
                        "sentence": sname, "token_i": i, "role": role, "form": toks[i], "scored_text": word,
                        "n_pieces": npieces, "word_initial_piece": bool(word_initial),
                        "surp": surp, "decile": int(np.searchsorted(ROBBI_THRESHOLDS, surp, side="left")),
                        "entropy": float(r["entropy"][j]),
                        **{f"p_{f}": float(v) for f, v in zip(S.INVENTORY_FORMS, inv)},
                        "top_form": S.INVENTORY_FORMS[int(np.argmax(inv))],
                        "context": " ".join(ctx), "target": " ".join(tgt),
                        "item_text": " | ".join(lines_txt),
                        **features(toks, i, sname == "target", bool(ctx)),
                    })
        if len(rows) % 500 < 8:
            print(f"  {cat}/{cond}/{item}: {len(rows)} rows", flush=True)

    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    meta = {"stimuli": str(a.stimuli), "hf_mlm": a.hf_mlm, "label": a.label, "mlm_ctx": a.mlm_ctx,
            "thresholds": ROBBI_THRESHOLDS, "pairs": len(pairs), "rows": len(out), "misaligned": misaligned}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    c = out[(out.role == "critical") & (out.render == "natural") & (out.layout == "two_lines")]
    print(json.dumps(meta)[:300])
    print("critical pronouns, natural two_lines, decile counts by category:")
    print(c.groupby("category").decile.value_counts().unstack(fill_value=0).to_string())
    print("SCORE OK", flush=True)


if __name__ == "__main__":
    main()
