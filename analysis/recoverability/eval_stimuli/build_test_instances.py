#!/usr/bin/env python3
"""Every subject pronoun in the held-out test set, with our annotations, addressed by raw line.

Source: the layered annotation of test_10M (data/annotated_corpus_test_10M: base sentences
plus the pronouns, clause_structure and expletives layers) and the raw test files. Each
annotated sentence is located in its raw line, so a pronoun gets the address the corpus
rater needs (file, line index, character offset) and the labels a stimulus needs
(phenomenon, person, position).

Output, one parquet per genre file plus all.parquet:
  stem, line_idx, char_start, tok_text, form                    address
  sentence_id, sent_char_start, sent_text, token_idx            where it sits in our annotation
  person, number, clause_type, verb_lemma, expletive_class      what it is
  contracted, inverted, sent_initial, line_initial, after_and,
  after_that, after_wh, l1, r1, n_tokens, n_subj_pronouns       surface properties
  genre, speaker, role                                          source
  in_train                                                      the sentence also occurs in train_90M

    python analysis/recoverability/eval_stimuli/build_test_instances.py \\
        --annotated data/annotated_corpus_test_10M --raw data/raw/test_10M \\
        --train-annotated data/annotated_corpus_train_90M --out data/recoverability/analysis/eval_stimuli/test_10M
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_FORMS = {"i", "you", "he", "she", "it", "we", "they"}
CLITICS = {"'m", "'re", "'s", "'ve", "'ll", "'d", "’m", "’re", "’s", "’ve", "’ll", "’d"}
SENT_PUNCT = {".", "!", "?", "...", ";", "…"}
OPENERS = {'"', "'", "“", "‘", "-", "--", "—", "(", "["}
CONJ = {"and", "but", "so", "or"}
WH = {"who", "what", "where", "when", "why", "how", "which", "whom", "whose"}
AUX_INV = {"do", "does", "did", "is", "are", "was", "were", "am", "have", "has", "had", "will", "would", "can",
           "could", "should", "shall", "may", "might", "must", "ai", "wo", "ca", "n't"}
_WS = re.compile(r"\s+")


def norm_sentence(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def token_offsets(text: str, tokens) -> list[int] | None:
    offs, cur = [], 0
    for t in tokens:
        p = text.find(t, cur)
        if p < 0:
            return None
        offs.append(p)
        cur = p + len(t)
    return offs


def align_sentences(lines: list[str], texts: list[str], lookahead: int = 4):
    """(line_idx, char_start) of each sentence, in order; (-1, -1) when it cannot be placed."""
    out = np.full((len(texts), 2), -1, dtype=np.int64)
    li, cur = 0, 0
    for k, t in enumerate(texts):
        t = t.strip()
        if not t:
            continue
        for step in range(lookahead + 1):
            j = li + step
            if j >= len(lines):
                break
            p = lines[j].find(t, cur if step == 0 else 0)
            if p >= 0:
                out[k] = (j, p)
                li, cur = j, p + len(t)
                break
    return out


def build_stem(stem, base, layers, lines, train_hashes):
    b = base[base.file_name == stem].sort_values("sent_idx").reset_index(drop=True)
    where = align_sentences(lines, b.text.tolist())
    pr = layers["pronouns"].set_index("sentence_id").pronouns
    cl = layers["clause_structure"].set_index("sentence_id").clauses
    ex = layers["expletives"].set_index("sentence_id").expletives
    stats = {"sentences": len(b), "sentences_aligned": int((where[:, 0] >= 0).sum()), "subject_tokens": 0,
             "offset_fail": 0, "kept": 0}
    rows = []
    for k, r in enumerate(b.itertuples(index=False)):
        prons = pr.get(r.sentence_id)
        if prons is None or len(prons) == 0:
            continue
        toks = [str(t) for t in r.tokens]
        subj = [p for p in prons if p.get("function") == "subject" and 0 <= int(p["token_idx"]) < len(toks)
                and toks[int(p["token_idx"])].lower() in SUBJECT_FORMS]
        if not subj:
            continue
        stats["subject_tokens"] += len(subj)
        if where[k, 0] < 0:
            continue
        offs = token_offsets(r.text.strip(), toks)
        if offs is None:
            stats["offset_fail"] += len(subj)
            continue
        line_idx, sent_start = int(where[k, 0]), int(where[k, 1])
        line = lines[line_idx]
        clauses = {int(c["subject_idx"]): c for c in (cl.get(r.sentence_id) if cl.get(r.sentence_id) is not None else [])
                   if c.get("subject_idx") is not None and not pd.isna(c.get("subject_idx"))}
        expl = {int(e["token_idx"]): e for e in (ex.get(r.sentence_id) if ex.get(r.sentence_id) is not None else [])}
        low = [t.lower() for t in toks]
        first_content = next((j for j, t in enumerate(low) if t not in OPENERS), 0)
        for p in subj:
            i = int(p["token_idx"])
            cs = sent_start + offs[i]
            if line[cs:cs + len(toks[i])] != toks[i]:
                stats["offset_fail"] += 1
                continue
            l1 = low[i - 1] if i > 0 else ""
            l2 = low[i - 2] if i > 1 else ""
            r1 = low[i + 1] if i + 1 < len(low) else ""
            glued = i + 1 < len(toks) and offs[i + 1] == offs[i] + len(toks[i])
            c = clauses.get(i, {})
            e = expl.get(i, {})
            inverted = l1 in AUX_INV
            rows.append({
                "stem": stem, "line_idx": line_idx, "char_start": cs, "tok_text": toks[i], "form": low[i],
                "sentence_id": r.sentence_id, "sent_char_start": sent_start, "sent_text": r.text.strip(), "token_idx": i,
                "person": p.get("person"), "number": p.get("number"),
                "clause_type": c.get("clause_type"), "verb_lemma": c.get("verb_lemma"), "clause_finite": c.get("is_finite"),
                "expletive_class": e.get("expletive_class"),
                "contracted": bool(glued and r1 in CLITICS),
                "inverted": inverted, "wh_inverted": bool(inverted and (l2 in WH or (l2 in AUX_INV and i > 2 and low[i - 3] in WH))),
                "sent_initial": i <= first_content or l1 in SENT_PUNCT,
                "line_initial": line[:cs].strip() == "" or bool(re.fullmatch(r"\S+:\t[\"'“‘\-\s]*", line[:cs])),
                "after_and": l1 in CONJ, "after_that": l1 == "that", "after_wh": l1 in WH,
                "ends_q": low[-1] == "?" if low else False,
                "l1": l1, "r1": r1, "n_tokens": len(toks), "n_subj_pronouns": len(subj),
                "genre": r.genre, "speaker": r.speaker, "role": r.role,
            })
    stats["kept"] = len(rows)
    d = pd.DataFrame(rows)
    if train_hashes is not None and len(d):
        # One vectorized lookup: is the (normalized) sentence also a training sentence?
        d["in_train"] = np.isin(pd.util.hash_array(d.sent_text.map(norm_sentence).to_numpy(dtype=object)), train_hashes)
    return d, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotated", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--train-annotated", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test_10M")
    ap.add_argument("--ext", default="test")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(a.annotated / f"base/{a.split}.parquet",
                           columns=["sentence_id", "genre", "speaker", "role", "file_name", "sent_idx", "text", "tokens"])
    layers = {n: pd.read_parquet(a.annotated / f"layers/{n}/{a.split}.parquet")
              for n in ("pronouns", "clause_structure", "expletives")}
    train_hashes = None
    if a.train_annotated:
        parts = sorted((a.train_annotated / "base").glob("*.parquet"))
        hs = []
        for p in parts:
            s = pd.read_parquet(p, columns=["text"]).text
            s = s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()     # == norm_sentence, vectorized
            hs.append(pd.util.hash_array(s.to_numpy(dtype=object)))
        train_hashes = np.unique(np.concatenate(hs))
        print(f"train sentences hashed: {len(train_hashes):,} unique", flush=True)

    all_stats, frames = {}, []
    for stem in sorted(base.file_name.unique()):
        lines = (a.raw / f"{stem}.{a.ext}").read_text(encoding="utf-8").split("\n")
        d, st = build_stem(stem, base, layers, lines, train_hashes)
        d.to_parquet(a.out / f"instances__{stem}.parquet")
        all_stats[stem] = st
        frames.append(d)
        print(stem, json.dumps(st), flush=True)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(a.out / "instances__all.parquet")
    (a.out / "instances.meta.json").write_text(json.dumps({"split": a.split, "stems": all_stats, "n": len(full)}, indent=2))
    print("total", len(full))


if __name__ == "__main__":
    main()
