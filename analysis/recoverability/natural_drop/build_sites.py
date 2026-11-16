#!/usr/bin/env python3
"""Sites for the natural-drop recoverability analysis (CHILDES, train_90M).

Null sites: every finite clause with no subject in the corpus null-subject annotation (clause_structure
layer, subject_status == "none"), read from the per-genre CHILDES files only. Each is mapped onto the raw
corpus line the rater scored, with the character offset where an overt subject would stand, and classed
by the form of its first verbal element:

  inflected     the form rules out an imperative (3sg, past, am/is/are/was, a modal)
  stative       base form of a verb that is rarely an imperative (want, need, like, know, ...)
  bare          base form that could be an imperative; kept apart

(childes_null_type.parquet from 2026-02 is not used: it also took clause rows from the monolithic
annotation file, whose CHILDES sentence ids collide with the per-genre ids but name other sentences.)

Overt sites: a random sample of the personal-pronoun subjects the rater ranked (robbi_joined.parquet),
for the same speakers, to be re-scored with the same code.

    python analysis/recoverability/natural_drop/build_sites.py --main <main checkout> --out <dir>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

BRACKET = re.compile(r"\[[^\]]*\]")
SPACE_PUNCT = re.compile(r"\s+([.?!,;:])")
WS = re.compile(r"\s+")
AUX_DEPS = {"aux", "auxpass", "aux:pass", "neg"}
# Adverbs that stand between subject and verb ("I just want"); others ("then", "now") precede the subject.
MID_ADV = {"just", "really", "never", "always", "also", "still", "even", "only", "probably", "actually", "already",
           "ever", "sure", "certainly", "definitely", "usually", "often", "almost", "hardly", "barely", "simply",
           "kinda", "sorta", "sometimes", "maybe"} - {"maybe"}
SIBLING_CODES = {"SIS", "BRO", "COU", "CHI2", "SI1", "SI2", "SI3", "BR1", "BR2", "FRI", "KID"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}
STATIVE = {"want", "need", "like", "love", "hate", "know", "think", "guess", "mean", "hope", "wish", "wanna", "gonna",
           "hafta", "gotta", "have", "wonder", "bet", "suppose", "reckon", "seem", "doubt"}


FORMULA = {"thank", "welcome", "bless", "excuse", "pardon", "beg"}
CLITICS = {"'s", "'m", "'re", "'d", "'ll", "'ve"}
NOMINAL_POS = {"PRON", "PROPN", "NOUN", "NUM", "DET", "ADJ"}


def verb_class(first: int, tokens: list[str], lemmas: list[str]) -> str:
    form, lemma = tokens[first].lower(), (lemmas[first] or "").lower()
    if not (form.isalpha() or form in CLITICS) or form in ("its", "xxx", "yyy", "www", "s", "m", "your"):
        return "unusable"
    if lemma in FORMULA:
        return "formula"
    if form == "got":
        return "inflected"
    if lemma in MODALS or form in ("ca", "wo", "sha"):
        return "inflected"
    if form in ("wanna", "gonna", "hafta", "gotta") or lemma in STATIVE and form == lemma:
        return "stative"
    return "bare" if form == lemma else "inflected"


def norm(s: str) -> str:
    s = BRACKET.sub(" ", s).replace("[", " ").replace("]", " ")
    return WS.sub(" ", SPACE_PUNCT.sub(r"\1", s)).strip()


def split_line(ln: str):
    """(speaker code or '', content start offset) of a raw CHILDES line."""
    if ln.startswith("*"):
        c = ln.find("\t")
        if c > 0:
            return ln[1:c].rstrip(":"), c + 1
    return "", 0


def align(base: pd.DataFrame, raw: list[str], window: int = 80, far: int = 20000) -> np.ndarray:
    """Raw line index of every annotated sentence (-1 if not found), by speaker and normalized text.

    Sentences whose speaker the pipeline did not parse carry the whole raw line as text and are matched
    on the whole line. A long sentence that is not found nearby is searched for further ahead, so a block
    of raw lines with no annotated counterpart does not derail everything after it.
    """
    info = [split_line(ln) for ln in raw]
    rnorm = [norm(ln[s:]) if code else ln for ln, (code, s) in zip(raw, info)]
    codes = [c for c, _ in info]
    txt, spk = base.text.to_numpy(), base.speaker.to_numpy()
    out = np.full(len(base), -1, dtype=np.int64)
    i, used = 0, 0          # i: current raw line; used: chars of its normalized content already consumed
    for k in range(len(base)):
        t = txt[k]
        nt = norm(t)
        if not nt:
            continue
        if not isinstance(spk[k], str):
            for j in range(i + (1 if used > 0 else 0), min(i + window, len(raw))):
                if not codes[j] and norm(raw[j]) == nt or (codes[j] and norm(raw[j]) == nt):
                    out[k], i, used = j, j, len(rnorm[j])
                    break
            continue
        # continuing the same raw line (several sentences on one line)
        if i < len(raw) and codes[i] == spk[k] and rnorm[i][used:].lstrip().startswith(nt) and used > 0:
            out[k] = i
            used = rnorm[i].index(nt, used) + len(nt)
            continue
        j0 = i + (1 if used > 0 else 0)
        for j in range(j0, min(j0 + window, len(raw))):
            if codes[j] != spk[k]:
                continue
            if rnorm[j] == nt or rnorm[j].startswith(nt):
                out[k], i, used = j, j, len(nt)
                break
        else:
            if len(nt) >= 20:
                for j in range(j0 + window, min(j0 + far, len(raw))):
                    if codes[j] == spk[k] and rnorm[j] == nt:
                        out[k], i, used = j, j, len(nt)
                        break
    return out


def token_offsets(line: str, start: int, tokens: list[str]) -> list[int] | None:
    """Character offset of each token in the raw line, searching in order and skipping [bracketed] spans."""
    spans = [(m.start(), m.end()) for m in BRACKET.finditer(line)]
    cur, out = start, []
    for t in tokens:
        p = line.find(t, cur)
        while p >= 0 and any(a <= p < b for a, b in spans):
            p = line.find(t, p + 1)
        if p < 0:
            return None
        out.append(p)
        cur = p + len(t)
    return out


def insertion_index(v: int, deps: list[str], heads: list[int], lemmas: list[str]) -> tuple[int, int]:
    """(token index a subject would precede, index of the first verbal element)."""
    kids = [c for c, h in enumerate(heads) if h == v and c != v]
    first = min([v] + [c for c in kids if deps[c].lower() in ("aux", "auxpass", "aux:pass") and c < v])
    ins = min([first] + [c for c in kids if deps[c].lower() == "neg" and c < v])
    while ins > 0 and heads[ins - 1] == v and deps[ins - 1].lower() == "advmod" and (lemmas[ins - 1] or "").lower() in MID_ADV:
        ins -= 1
    return ins, first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overt-per-role", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    corp = a.main / "data/output/train_90M"
    raw = (a.main / "data/raw/train_90M/childes.train").read_text(encoding="utf-8").split("\n")
    if raw[-1] == "":
        raw = raw[:-1]
    base_path = corp / "annotated_corpus/base/train_90M_childes.parquet"
    base = pd.read_parquet(base_path, columns=["sentence_id", "sent_idx", "speaker", "role", "text", "child_age_months"])
    line_of = align(base, raw)
    base["line_idx"] = line_of
    print(f"sentences {len(base):,}; aligned to a raw line {np.mean(line_of >= 0):.4f}")
    per_line = pd.Series(line_of[line_of >= 0]).value_counts()
    print("raw lines carrying 1 / 2 / 3+ sentences:", int((per_line == 1).sum()), int((per_line == 2).sum()), int((per_line > 2).sum()))

    # ---- null sites
    nt = pd.read_parquet(a.out / "childes_null_type_regenerated.parquet").drop(columns="role")
    nt = nt[nt.is_finite & (nt.subject_status == "none")]
    nt = nt.merge(base, on="sentence_id", how="left")
    print(f"finite clauses without a subject {len(nt):,}; on an aligned line {np.mean(nt.line_idx >= 0):.4f}")
    ids = pc.field("sentence_id").isin(nt.sentence_id.unique().tolist())
    par = ds.dataset(base_path).to_table(filter=ids, columns=["sentence_id", "tokens", "lemmas", "pos_tags", "dep_rels", "dep_heads"]).to_pandas()
    nt = nt.merge(par, on="sentence_id", how="left")
    rows, drop = [], {"unaligned": 0, "tokens_not_found": 0}
    for r in nt.itertuples(index=False):
        if r.line_idx < 0:
            drop["unaligned"] += 1
            continue
        ln = raw[r.line_idx]
        _, start = split_line(ln)
        toks = list(r.tokens)
        offs = token_offsets(ln, start, toks)
        if offs is None:
            drop["tokens_not_found"] += 1
            continue
        if int(r.verb_idx) >= len(toks):
            drop["verb_index_out_of_range"] = drop.get("verb_index_out_of_range", 0) + 1
            continue
        ins, first = insertion_index(int(r.verb_idx), list(r.dep_rels), [int(h) for h in r.dep_heads], list(r.lemmas))
        pos = offs[ins]
        new = ln[:pos] + "it " + ln[pos:]
        rows.append(dict(kind="null", sentence_id=r.sentence_id, line_idx=int(r.line_idx), speaker=r.speaker, role=r.role,
                         child_age_months=r.child_age_months, verb_idx=int(r.verb_idx), verb_lemma=r.verb_lemma,
                         clause_type=r.clause_type, heuristic_null_type=r.null_type,
                         verb_class=verb_class(first, toks, list(r.lemmas)), first_verbal=toks[first].lower(),
                         prev_pos=(r.pos_tags[ins - 1] if ins > 0 else ""),
                         nominal_before=bool(ins > 0 and r.pos_tags[ins - 1] in NOMINAL_POS),
                         glued=bool(pos > 0 and not ln[pos - 1].isspace()),
                         ins_token=ins, char_start=pos, line_text=ln, scored_line=new,
                         first_in_utterance=bool(norm(ln[start:pos]) == ""), n_tokens=len(toks),
                         before=" ".join(toks[max(0, ins - 3):ins]), after=" ".join(toks[ins:ins + 5])))
    null = pd.DataFrame(rows)
    print(f"null sites kept {len(null):,}; dropped {drop}")
    print(null.groupby(["role", "verb_class"]).size().unstack(fill_value=0).to_string())
    keep = null[~null.nominal_before & ~null.glued & null.verb_class.isin(["inflected", "stative", "bare"])]
    print("after dropping sites with a nominal right before them or a clitic verb glued to the word before "
          "(here's, where's), formulas and unusable tokens:")
    print(keep.groupby(["role", "verb_class"]).size().unstack(fill_value=0).to_string())
    print(null.groupby(["verb_class", "heuristic_null_type"]).size().unstack(fill_value=0).to_string())
    print("first in utterance:", null.groupby(["role", "verb_class"]).first_in_utterance.mean().round(3).unstack().to_string())

    # ---- overt sample from the rater's ranked personal-pronoun subjects
    rj = pd.read_parquet(a.main / "data/recoverability/analysis/survey/robbi_joined.parquet",
                         columns=["genre", "line_idx", "token_i", "form", "d_robbi", "s_robbi"])
    rj = rj[rj.genre == "childes"].drop(columns="genre")
    code = np.array([split_line(raw[j])[0] for j in rj.line_idx.to_numpy()])
    role_of = base.drop_duplicates("speaker").set_index("speaker").role.to_dict()
    rj["speaker"] = code
    rj["role"] = pd.Series(code).map(role_of).fillna("unknown").to_numpy()
    age = base[base.line_idx >= 0].drop_duplicates("line_idx").set_index("line_idx").child_age_months
    rj["child_age_months"] = rj.line_idx.map(age)
    print("ranked pronoun subjects in CHILDES by role:", rj.role.value_counts().to_dict())
    inst = pd.read_parquet(a.main / "data/recoverability/train_90M/instances/childes.parquet",
                           columns=["line_idx", "token_i", "piece_in_line", "n_pieces"])
    rng = np.random.default_rng(a.seed)
    samp = pd.concat([g.iloc[rng.permutation(len(g))[:a.overt_per_role]] for role, g in rj.groupby("role") if role in ("child", "adult")])
    samp = samp.merge(inst, on=["line_idx", "token_i"], how="left")
    samp["kind"] = "overt"
    samp["line_text"] = [raw[j] for j in samp.line_idx]
    a.out.mkdir(parents=True, exist_ok=True)
    null.to_parquet(a.out / "null_sites.parquet", index=False)
    samp.to_parquet(a.out / "overt_sites.parquet", index=False)
    pop = rj.groupby(["role", "d_robbi"]).size().rename("n").reset_index()
    pop.to_csv(a.out / "overt_population_deciles.csv", index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
