#!/usr/bin/env python3
"""Turn annotated test-set pronouns into candidate eval items with their real context.

For every subject pronoun kept by the filters, the item is:
  target   the sentence that contains the pronoun (overt), and the same sentence with the
           pronoun deleted (null)
  context  the text that precedes the sentence in the corpus, taken back to a sentence
           boundary within a word budget; speaker tags stay, as the learners saw them
The phenomenon label comes from our annotations: the dependency of the pronoun's head
(matrix, embedded, conjunction, adverbial, relative, parataxis) and, for "it", the
expletive test of the remove_expletive_sentences intervention (weather, raising,
extraposition).

Filters (each counted in the meta file): subject forms only, not contracted, not glued to
following punctuation, not inverted, not a reporting tag after a quotation, target of 4..30
words, sentence not present in train_90M, no transcription codes or anonymization gaps,
optional exclusion of child speech, duplicates dropped, at least `--min-context-words` of
context. The context stops at the nearest line that carries transcription codes, markup or
garbled spacing, so it never contains them. The phenomenon label is checked against the
surface string (`type_check`): a conjunction item has its conjunction directly before the
pronoun, an adverbial item its subordinator, an embedded item its governing verb or "that".

Output columns include `rater_lines` (the item as raw corpus lines, last line cut at the end
of the target sentence) and `rater_char_start` (pronoun offset in the last line), which
score_items.py scores the corpus way.

    python analysis/recoverability/eval_stimuli/build_items.py --annotated data/annotated_corpus_test_10M \\
        --raw data/raw/test_10M --instances <dir>/instances__all.parquet --out <dir>/items_candidates.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_test_instances import SUBJECT_FORMS, align_sentences, norm_sentence  # noqa: E402

WEATHER_VERBS = {"rain", "snow", "hail", "drizzle", "thunder", "sleet", "pour", "storm", "freeze", "thaw"}
RAISING_VERBS = {"seem", "appear", "happen", "turn", "follow", "matter", "suffice", "transpire", "emerge"}
RAISING_ADJECTIVES = {"clear", "obvious", "likely", "unlikely", "possible", "impossible", "certain", "evident",
                      "apparent", "true", "false", "important", "necessary", "probable", "fortunate", "unfortunate",
                      "surprising", "known"}
CLAUSAL = {"ccomp", "xcomp", "csubj", "csubjpass", "csubj:pass", "advcl", "acl"}
PTYPE = {"ROOT": "matrix", "ccomp": "embedded", "conj": "conjunction", "advcl": "adverbial", "relcl": "relative",
         "parataxis": "parataxis"}
BAD_TARGET = re.compile(r"[_\[\]%&@<>|=*#^~]|xxx|yyy|www|\d{3,}|\S  +\S|[A-Za-z]{22,}")
# A context line is unusable when it carries codes or markup. "*MOT:" style speaker tags are
# the one use of "*" that is corpus text the learners saw.
BAD_CONTEXT = re.compile(r"[_\[\]%&@<>|=#^~]|xxx|yyy|www|\*(?![A-Z]{3}:)|\S  +\S|[A-Za-z]{22,}")
CONJ = {"and", "but", "or", "so"}
SUBORD = {"if", "when", "because", "as", "while", "since", "before", "after", "until", "till", "though", "although",
          "unless", "once", "whenever", "cause", "'cause", "cos", "'cos", "whereas", "where", "wherever"}
QUOTES = {'"', "'", "”", "’", "“", "‘", "``", "''"}
OPENERS = {'"', "'", "“", "‘", "``", "-", "--", "—", "(", "["}
BREAKS = QUOTES | {":", ";", "-", "--", "—", "(", ")"}
WH_COMP = {"what", "who", "where", "when", "why", "how", "which", "whether", "if", "whom", "whose"}
PERSON = {"i": "1sg", "we": "1pl", "you": "2", "he": "3sg", "she": "3sg", "it": "3sg", "they": "3pl"}


def expletive_it(i, lemmas, deps, heads, pos):
    """The English test of preprocessing/ablations/remove_expletive_sentences.py, on parse arrays."""
    h = int(heads[i])
    if not 0 <= h < len(lemmas):
        return None
    hl = str(lemmas[h]).lower()
    kids = [j for j in range(len(heads)) if int(heads[j]) == h and j != h]

    def clausal(j):
        ks = [k for k in range(len(heads)) if int(heads[k]) == j and k != j]
        return any(deps[k] in CLAUSAL or (deps[k] == "mark" and str(lemmas[k]).lower() == "that") for k in ks)

    if hl in WEATHER_VERBS:
        return "weather"
    if hl in RAISING_VERBS and clausal(h):
        return "raising"
    if hl == "be" or pos[h] == "AUX":
        for j in kids:
            if deps[j] in ("acomp", "attr", "oprd") and str(lemmas[j]).lower() in RAISING_ADJECTIVES:
                if clausal(j) or any(deps[k] in {"ccomp", "csubj", "csubjpass", "csubj:pass", "advcl"} for k in kids):
                    return "extraposition"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotated", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test_10M")
    ap.add_argument("--ext", default="test")
    ap.add_argument("--context-words", type=int, default=45)
    ap.add_argument("--min-context-words", type=int, default=4)
    ap.add_argument("--target-words", type=int, nargs=2, default=[4, 30])
    ap.add_argument("--keep-child-speech", action="store_true")
    a = ap.parse_args()

    inst = pd.read_parquet(a.instances)
    base = pd.read_parquet(a.annotated / f"base/{a.split}.parquet",
                           columns=["sentence_id", "file_name", "sent_idx", "text", "tokens", "lemmas", "pos_tags", "dep_rels",
                                    "dep_heads"])
    drop = {"n_in": len(inst)}
    out_frames = []
    for stem, g in inst.groupby("stem", sort=True):
        lines = (a.raw / f"{stem}.{a.ext}").read_text(encoding="utf-8").split("\n")
        b = base[base.file_name == stem].sort_values("sent_idx").reset_index(drop=True)
        where = align_sentences(lines, b.text.tolist())
        pos_of = {sid: k for k, sid in enumerate(b.sentence_id)}
        texts = [t.strip() for t in b.text]
        # is this sentence the first one located on its line?
        first_on_line = np.ones(len(b), dtype=bool)
        first_on_line[1:] = where[1:, 0] != where[:-1, 0]
        rows = []
        for r in g.itertuples(index=False):
            def skip(why):
                drop[why] = drop.get(why, 0) + 1
            if r.form not in SUBJECT_FORMS:
                skip("not_subject_form"); continue
            if r.contracted:
                skip("contracted"); continue
            if r.inverted:
                skip("inverted"); continue
            if bool(getattr(r, "in_train", False)):
                skip("in_train"); continue
            if r.role == "child" and not a.keep_child_speech:
                skip("child_speech"); continue
            k = pos_of[r.sentence_id]
            sent = texts[k]
            off = r.char_start - r.sent_char_start
            end = off + len(r.tok_text)
            if sent[off:end] != r.tok_text or end >= len(sent) or sent[end] != " ":
                skip("glued_right"); continue
            if off > 0 and sent[off - 1] not in " \"'“‘(-—":
                skip("glued_left"); continue
            n_words = len(sent.split())
            if not a.target_words[0] <= n_words <= a.target_words[1]:
                skip("target_length"); continue
            if BAD_TARGET.search(sent):
                skip("transcription_code"); continue
            brow = b.iloc[k]
            i = int(r.token_idx)
            h = int(brow.dep_heads[i])
            head_dep = brow.dep_rels[h] if 0 <= h < len(brow.dep_rels) else None
            expl = expletive_it(i, brow.lemmas, brow.dep_rels, brow.dep_heads, brow.pos_tags) if r.form == "it" else None
            ptype = "expletive_it" if expl else PTYPE.get(head_dep)
            if ptype is None:
                skip("untyped_clause"); continue
            low = [str(t).lower() for t in brow.tokens]
            l1 = low[i - 1] if i > 0 else ""
            if l1 in ("n’t", "nt") or (l1 in QUOTES and not r.sent_initial):
                skip("inverted_or_quote_tag"); continue
            kind = None
            if ptype == "matrix":
                kind = ("initial" if r.sent_initial else "after_conj" if l1 in CONJ else
                        "after_comma" if l1 == "," else "medial")
            elif ptype == "conjunction":
                if l1 not in CONJ:
                    skip("type_check"); continue
                kind = l1
            elif ptype == "adverbial":
                if l1 not in SUBORD:
                    skip("type_check"); continue
                kind = l1
            elif ptype == "embedded":
                gov = int(brow.dep_heads[h])
                if not (0 <= gov < i) or i - gov > 8 or r.sent_initial or any(t in BREAKS for t in low[gov + 1:i]):
                    skip("type_check"); continue
                kind = "that" if l1 == "that" else "bare" if gov == i - 1 else "wh" if l1 in WH_COMP else "other"
            elif ptype == "expletive_it":
                kind = expl
            # what precedes the pronoun inside its sentence (opening quotes and dashes aside)
            lead = [j for j in range(i) if low[j] not in OPENERS]
            lead_text = " ".join(low[j] for j in lead[-6:])
            lead_pos = " ".join(str(brow.pos_tags[j]) for j in lead[-6:])
            nxt = i + 1 if i + 1 < len(low) else None
            # context: walk back over preceding sentences while the word budget allows
            L, s = int(where[k, 0]), int(where[k, 1])
            j, words = k, 0
            while j > 0 and where[j - 1, 0] >= 0 and L - where[j - 1, 0] <= 12:
                w = len(texts[j - 1].split())
                if words + w > a.context_words:
                    break
                j -= 1
                words += w
            if words < a.min_context_words:
                skip("no_context"); continue
            l0 = int(where[j, 0])
            c0 = 0 if first_on_line[j] else int(where[j, 1])
            # the context never reaches back past a line with codes or markup
            for li in range(L - 1, l0 - 1, -1):
                if BAD_CONTEXT.search(lines[li][c0:] if li == l0 else lines[li]):
                    l0, c0 = li + 1, 0
                    break
            last_lo = c0 if L == l0 else 0
            if BAD_CONTEXT.search(lines[L][last_lo:s]):
                skip("context_junk"); continue
            frags = []
            for li in range(l0, L + 1):
                lo = c0 if li == l0 else 0
                hi = s + len(sent) if li == L else len(lines[li])
                frags.append(lines[li][lo:hi].rstrip("\r"))
            ctx_frags = frags[:-1] + [lines[L][last_lo:s]]
            context = " ".join(f.strip() for f in ctx_frags if f.strip())
            if len(context.split()) < a.min_context_words:
                skip("no_context"); continue
            null = sent[:off] + sent[end + 1:]
            wi = len(sent[:off].split())            # word index of the pronoun's word
            words_overt = sent.split()
            if wi + 1 >= len(words_overt):
                skip("no_hotspot"); continue
            rows.append({
                "stem": stem, "line_idx": int(r.line_idx), "char_start": int(r.char_start), "sentence_id": r.sentence_id,
                "form": r.form, "person": PERSON[r.form], "ptype": ptype, "kind": kind, "expletive": expl,
                "head_dep": head_dep,
                "verb_lemma": str(brow.lemmas[h]).lower() if 0 <= h < len(brow.lemmas) else None,
                "sent_initial": bool(r.sent_initial), "line_initial": bool(r.line_initial), "after_and": bool(r.after_and),
                "after_that": bool(r.after_that), "l1": r.l1, "r1": r.r1,
                "lead_n": len(lead), "lead_text": lead_text, "lead_pos": lead_pos,
                "r1_pos": str(brow.pos_tags[nxt]) if nxt is not None else None,
                "r1_lemma": str(brow.lemmas[nxt]).lower() if nxt is not None else None,
                "genre": r.genre, "role": r.role, "n_subj_pronouns": int(r.n_subj_pronouns),
                "context": context, "target_overt": sent, "target_null": null,
                "hotspot_token": words_overt[wi + 1], "hotspot_overt": wi + 1, "hotspot_null": wi,
                "context_words": len(context.split()), "target_words": n_words, "context_lines": L - l0,
                "rater_lines": json.dumps([f for f in frags]), "rater_char_start": int(r.char_start) - last_lo,
                "norm_target": norm_sentence(sent),
            })
        d = pd.DataFrame(rows)
        if len(d):
            before = len(d)
            d = d.drop_duplicates("norm_target").reset_index(drop=True)
            drop["duplicate_target"] = drop.get("duplicate_target", 0) + before - len(d)
        out_frames.append(d)
        print(stem, len(d), flush=True)
    out = pd.concat(out_frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates("norm_target").reset_index(drop=True)
    drop["duplicate_target"] = drop.get("duplicate_target", 0) + before - len(out)
    out.insert(0, "cand_id", np.arange(len(out)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    drop["kept"] = len(out)
    a.out.with_suffix(".meta.json").write_text(json.dumps({"filters": drop, "context_words": a.context_words,
                                                           "target_words": a.target_words}, indent=2))
    print(json.dumps(drop))
    print(out.groupby(["ptype", "person"]).size().unstack(fill_value=0).to_string())
    print(out.groupby(["ptype", "kind"]).size().to_string())


if __name__ == "__main__":
    main()
