"""Step 1: population_features.parquet — one row per selected instance (6.46M).

Inputs
  <out>/robbi_joined.parquet                       (s0)
  <data>/recoverability/analysis/instance_measures/<genre>.parquet
  <data>/raw/train_90M/<genre>.train, childes_metadata.json

Columns (beyond keys, deciles, surprisals, and instance_measures' lemma / dep /
person / number / is_title / word_initial / head_i):
  tok_ok              spaCy tokenizer reproduces form at token_i (verification)
  char_start/char_end pronoun span in the raw line
  content_start       char offset where content starts (after speaker tag/dash)
  l2 l1 r1 r2         content tokens around the slot, normalized, <s>/</s> pad
  contracted, clitic  next token is a clitic ('m 're 's 've 'll 'd n't) with no
                      space before it (curly apostrophes normalized)
  head_text           the pronoun's parser head (normalized surface form)
  inverted            aux immediately before the pronoun (do you / isn't it /
                      what's he)
  agree_set_size      # of {i,you,he,she,it,we,they} compatible with the
                      right-adjacent finite verb/aux/clitic (table in
                      survey_common); NaN when r1 is not one, incl. inversion
  agree_source        aux_table | lexical | inverted | archaic_thou | none
  agree_set_size_l1   same table applied to the fronted aux when inverted
  line_initial        first content token (after speaker tag / dialogue dash)
  sent_initial        line_initial or l1 in . ! ? ... ;
  after_and           l1 in and/but/so/or
  after_wh            l1 is a wh-word;  wh_inverted: wh + aux + pronoun
  line_ends_q         line ends with '?' (ignoring closing quotes/brackets)
  inverted_question   inverted & line_ends_q
  tag_question        ', aux (n't) PRON ?' at line end
  r1_punct_eol        r1 is punctuation or end of line
  line_len_tokens     content tokens in the line
  tokens_before       content tokens before the pronoun in its line
  line_norm_hash      stable hash of the normalized content text
  line_dup_count      # lines in the genre with the same normalized text
  speaker             CHILDES tier code / Switchboard A|B (else null)
  turn_marker         line starts with a speaker tag or dialogue dash
  prev_line_end       class (alnum/punct/space/start) of the last char of the
                      previous non-empty line = what the robbi scorer's
                      no-separator stream glues directly before this line
  char0               pronoun is the very first character of the raw line
  speaker_change      speaker differs from the previous tagged line (CHILDES:
                      within doc, comment lines skipped; Switchboard: previous
                      line); null when undeterminable
  childes_is_child    speaker == CHI (CHILDES only)
  doc_id              header text (CHILDES meta key, PG id, wiki title);
                      '__prefix__' for lines before the first header; null for
                      genres without headers
  tokens_since_doc_start  content tokens in the doc before the pronoun
  age_months          CHILDES child age (childes_metadata.json, via header)
  cluster_id          doc_id, or <genre>|blk<line_idx//200> without headers
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_common as C  # noqa: E402

K = ["genre", "line_idx", "token_i"]
IM_COLS = ["line_idx", "token_i", "form", "lemma", "dep", "person", "number",
           "is_title", "word_initial", "head_i"]
CLOSERS = {'"', "'", ")", "]", "}", "”", "’", "»"}

_TOK = None


def _init_worker():
    global _TOK
    import spacy
    nlp = spacy.load("en_core_web_trf", exclude=[
        "transformer", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    _TOK = nlp.tokenizer


STR_FEATS = ["l2", "l1", "r1", "r2", "clitic", "head_text", "agree_source"]
BOOL_FEATS = ["tok_ok", "contracted", "inverted", "line_initial", "sent_initial",
              "after_and", "after_wh", "wh_inverted", "line_ends_q",
              "inverted_question", "tag_question", "r1_punct_eol"]
INT_FEATS = ["char_start", "char_end", "tokens_before", "line_len_tokens"]
FLT_FEATS = ["agree_set_size", "agree_set_size_l1"]


def _obj(s: pd.Series) -> np.ndarray:
    """object array with None (not NaN) for missing -> arrow string w/ nulls."""
    v = s.to_numpy(object)
    v[pd.isna(v)] = None
    return v


def _is_punct(s: str) -> bool:
    return not any(ch.isalnum() for ch in s)


def _aux_agree(a: str):
    if a in C.AGREE_TABLE:
        return float(len(C.AGREE_TABLE[a]))
    if a in ("need", "dare"):
        return 7.0
    return np.nan


def tokenize_chunk(task):
    """task = (a, lines, cstarts, count_all, inst_line, inst_tok, inst_form,
    inst_head). Returns per-line content-token counts and per-instance
    features (in input order)."""
    a, lines, cstarts, count_all, il, it, iform, ihead = task
    n_lines = len(lines)
    n_content = np.full(n_lines, -1, dtype=np.int32)
    need = set((il - a).tolist())
    by_line = {}
    for j in range(len(il)):
        by_line.setdefault(int(il[j]) - a, []).append(j)
    n = len(il)
    out = {f: [None] * n for f in STR_FEATS}
    out.update({f: np.zeros(n, dtype=bool) for f in BOOL_FEATS})
    out.update({f: np.full(n, -1, dtype=np.int32) for f in INT_FEATS})
    out.update({f: np.full(n, np.nan, dtype=np.float32) for f in FLT_FEATS})
    for li in range(n_lines):
        if not count_all and li not in need:
            continue
        line = lines[li]
        cs = cstarts[li]
        doc = _TOK(line)
        cidx = [t.i for t in doc if t.idx >= cs and not t.is_space]
        n_content[li] = len(cidx)
        if li not in need:
            continue
        ct = [C.norm_tok(doc[i].text) for i in cidx]
        pos = {ti: k for k, ti in enumerate(cidx)}
        # line ends with '?' (ignoring trailing closers)
        e = len(ct) - 1
        while e >= 0 and ct[e] in CLOSERS:
            e -= 1
        ends_q = e >= 0 and ct[e].endswith("?")
        L = len(ct)

        def g(k):
            if k < 0:
                return "<s>"
            if k >= L:
                return "</s>"
            return ct[k]

        for j in by_line[li]:
            ti = int(it[j])
            form = iform[j]
            if ti >= len(doc) or ti not in pos:
                out["tok_ok"][j] = False
                for f in STR_FEATS:
                    out[f][j] = ""
                continue
            tok = doc[ti]
            out["tok_ok"][j] = tok.text.lower() == form
            k = pos[ti]
            l3, l2, l1, r1, r2 = g(k - 3), g(k - 2), g(k - 1), g(k + 1), g(k + 2)
            out["l2"][j], out["l1"][j], out["r1"][j], out["r2"][j] = l2, l1, r1, r2
            out["char_start"][j] = tok.idx
            out["char_end"][j] = tok.idx + len(tok.text)
            out["tokens_before"][j] = k
            out["line_len_tokens"][j] = L
            nxt = doc[ti + 1] if ti + 1 < len(doc) else None
            contracted = (nxt is not None and tok.whitespace_ == ""
                          and C.norm_tok(nxt.text) in C.CLITICS)
            out["contracted"][j] = contracted
            out["clitic"][j] = C.norm_tok(nxt.text) if contracted else ""
            hi = int(ihead[j])
            out["head_text"][j] = (C.norm_tok(doc[hi].text)
                                   if 0 <= hi < len(doc) else "")
            inverted = (l1 in C.AUX_INV or (l1 == "n't" and l2 in C.AUX_INV)
                        or (l1 in C.CLITIC_AUX and l2 in C.WH))
            out["inverted"][j] = inverted
            # agreement from the right-adjacent token
            r1_doc_i = cidx[k + 1] if k + 1 < L else -1
            if inverted:
                ag, src = np.nan, "inverted"
            elif r1 in C.AGREE_TABLE:
                ag, src = float(len(C.AGREE_TABLE[r1])), "aux_table"
            elif r1 in C.ARCHAIC_THOU:
                ag, src = np.nan, "archaic_thou"
            elif (r1 not in C.NOT_FINITE and r1.isalpha() and hi == r1_doc_i
                  and l1 not in C.SMALL_CLAUSE_L1):
                ag, src = float(C.lexical_agree(r1)), "lexical"
            else:
                ag, src = np.nan, "none"
            out["agree_set_size"][j] = ag
            out["agree_source"][j] = src
            if inverted:
                out["agree_set_size_l1"][j] = _aux_agree(l2 if l1 == "n't" else l1)
            out["line_initial"][j] = k == 0
            out["sent_initial"][j] = k == 0 or l1 in C.SENT_PUNCT
            out["after_and"][j] = l1 in C.CONJ
            out["after_wh"][j] = l1 in C.WH
            out["wh_inverted"][j] = inverted and (
                l2 in C.WH or (l1 == "n't" and l3 in C.WH))
            out["line_ends_q"][j] = ends_q
            out["inverted_question"][j] = inverted and ends_q
            tail_ok = r1 == "?" and all(x in CLOSERS for x in ct[k + 2:])
            out["tag_question"][j] = tail_ok and (
                (l1 in C.AUX_INV and l2 == ",")
                or (l1 == "n't" and l2 in C.AUX_INV and l3 == ","))
            out["r1_punct_eol"][j] = r1 == "</s>" or _is_punct(r1)
    return a, n_content, out


def line_structure(genre, lines, meta):
    n = len(lines)
    is_header = np.zeros(n, dtype=bool)
    is_comment = np.zeros(n, dtype=bool)
    cstart = np.zeros(n, dtype=np.int32)
    speaker = [None] * n
    headers = []
    for i, line in enumerate(lines):
        h, ht, sp, cs, cm = C.parse_line(genre, line)
        is_header[i], cstart[i], speaker[i], is_comment[i] = h, cs, sp, cm
        if h:
            headers.append((i, ht))
    doc_num = (np.cumsum(is_header) - 1).astype(np.int32)       # -1 = prefix
    norm = [C.norm_line(line[cstart[i]:]) if not is_header[i] else ""
            for i, line in enumerate(lines)]
    h = C.stable_hash(norm)
    valid = np.array([bool(s) for s in norm]) & ~is_header
    uniq, inv, cnt = np.unique(h[valid], return_inverse=True, return_counts=True)
    dup = np.zeros(n, dtype=np.int32)
    dup[valid] = cnt[inv]
    # speaker change
    spk_change = np.full(n, np.nan, dtype=np.float32)
    if genre in C.SPEAKER_GENRES:
        prev = None
        for i in range(n):
            if is_header[i]:
                prev = None
                continue
            if speaker[i] is None:
                continue
            if prev is not None:
                spk_change[i] = float(speaker[i] != prev)
            prev = speaker[i]
    turn_marker = cstart > 0
    # Class of the character that precedes each line in the robbi scorer's
    # stream (lines concatenated with NO separator; empty lines contribute
    # nothing): the last char of the nearest previous non-empty line.
    idx = np.arange(n)
    ne = np.fromiter((bool(x) for x in lines), dtype=bool, count=n)
    last_ne = np.maximum.accumulate(np.where(ne, idx, -1))
    prev_ne = np.concatenate([[-1], last_ne[:-1]])
    pe_cls = np.empty(n, dtype=object)
    for i in range(n):
        j = prev_ne[i]
        if j < 0:
            pe_cls[i] = "start"
        else:
            c = lines[j][-1]
            pe_cls[i] = ("alnum" if c.isalnum() else
                         "space" if c.isspace() else "punct")
    doc_ids, ages = None, None
    if genre in C.DOC_GENRES:
        ids = []
        for _, ht in headers:
            ids.append(C.childes_meta_key(ht) if genre == "childes" else ht)
        doc_ids = ids
        if genre == "childes":
            ages = [meta.get(k, {}).get("age_months") for k in ids]
    return dict(is_header=is_header, is_comment=is_comment, cstart=cstart,
                speaker=speaker, doc_num=doc_num, headers=headers, norm_hash=h,
                dup=dup, spk_change=spk_change, turn_marker=turn_marker,
                prev_end=pe_cls,
                doc_ids=doc_ids, ages=ages)


def chunks_for(lines, inst_line, target_chars=1_500_000):
    lens = np.fromiter((len(x) + 1 for x in lines), dtype=np.int64, count=len(lines))
    cum = np.cumsum(lens)
    bounds = [0]
    nxt = target_chars
    while True:
        j = int(np.searchsorted(cum, nxt))
        if j >= len(lines) - 1:
            break
        bounds.append(j + 1)
        nxt = cum[j] + target_chars
    bounds.append(len(lines))
    starts = np.searchsorted(inst_line, bounds)
    return [(bounds[q], bounds[q + 1], int(starts[q]), int(starts[q + 1]))
            for q in range(len(bounds) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--joined", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--genres", nargs="+", default=C.GENRES)
    a = ap.parse_args()
    P = C.default_paths(a.data_root)
    out_dir = P["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = a.joined or (out_dir / "robbi_joined.parquet")
    T0 = time.time()
    meta_info = {"genres": {}}
    jf = pd.read_parquet(joined)
    meta = json.loads((P["raw"] / "childes_metadata.json").read_text())
    tables = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker) as pool:
        for genre in a.genres:
            t0 = time.time()
            df = (jf[jf.genre == genre].sort_values(["line_idx", "token_i"])
                  .reset_index(drop=True))
            im = pq.read_table(P["measures"] / f"{genre}.parquet",
                               columns=IM_COLS).to_pandas()
            im = im.rename(columns={"form": "im_form"})
            df = df.merge(im, on=["line_idx", "token_i"], how="left")
            n_join = int(df.dep.notna().sum())
            form_mismatch = int(((df.im_form != df.form) & df.im_form.notna()).sum())
            lines = C.read_lines(P["raw"], genre)
            S = line_structure(genre, lines, meta)
            t_struct = time.time() - t0
            il = df.line_idx.to_numpy(np.int64)
            it = df.token_i.to_numpy(np.int64)
            ih = df.head_i.fillna(-1).to_numpy(np.int64)
            iform = df.form.tolist()
            count_all = genre in C.DOC_GENRES
            tasks = []
            for (la, lb, ia, ib) in chunks_for(lines, il):
                if not count_all and ia == ib:
                    continue
                tasks.append((la, lines[la:lb], S["cstart"][la:lb].tolist(),
                              count_all, il[ia:ib], it[ia:ib], iform[ia:ib],
                              ih[ia:ib]))
            n_content = np.full(len(lines), -1, dtype=np.int32)
            feats = {f: [] for f in STR_FEATS + BOOL_FEATS + INT_FEATS + FLT_FEATS}
            for la, nc, o in pool.imap(tokenize_chunk, tasks, chunksize=1):
                n_content[la:la + len(nc)] = np.where(nc >= 0, nc,
                                                       n_content[la:la + len(nc)])
                for f in feats:
                    feats[f].append(o[f])
            del tasks
            for f in STR_FEATS:
                feats[f] = [x for part in feats[f] for x in part]
            for f in BOOL_FEATS + INT_FEATS + FLT_FEATS:
                feats[f] = np.concatenate(feats[f]) if feats[f] else np.zeros(0)
            assert len(feats["l1"]) == len(df)
            t_tok = time.time() - t0 - t_struct
            # document-level fields
            doc_num = S["doc_num"][il]
            if genre in C.DOC_GENRES:
                ncz = np.where(n_content < 0, 0, n_content).astype(np.int64)
                excl = np.cumsum(ncz) - ncz                 # tokens before line
                hdr_lines = np.array([h for h, _ in S["headers"]], dtype=np.int64)
                doc_start_line = np.where(doc_num >= 0,
                                          hdr_lines[np.maximum(doc_num, 0)] + 1, 0)
                tsds = (excl[il] - excl[doc_start_line]
                        + feats["tokens_before"]).astype(np.float64)
                ids = np.array(S["doc_ids"] + ["__prefix__"], dtype=object)
                doc_id = ids[doc_num]          # -1 -> last element = prefix
                cluster = np.array([f"{genre}|{d}" for d in doc_id], dtype=object)
                if genre == "childes":
                    ages = np.array([np.nan if x is None else float(x)
                                     for x in S["ages"]] + [np.nan])
                    age = ages[doc_num]
                else:
                    age = np.full(len(df), np.nan)
            else:
                tsds = np.full(len(df), np.nan)
                doc_id = np.array([None] * len(df), dtype=object)
                cluster = np.array([f"{genre}|blk{x // C.BLOCK_LINES:06d}"
                                    for x in il], dtype=object)
                age = np.full(len(df), np.nan)
            spk = [S["speaker"][x] for x in il]
            cols = {
                "genre": df.genre.to_numpy(object),
                "line_idx": il, "token_i": it, "form": df.form.to_numpy(object),
                "d_robbi": df.d_robbi.to_numpy(np.int8),
                "s_robbi": df.s_robbi.to_numpy(np.float64),
                "rand_decile": df.rand_decile.to_numpy(np.int8),
                "d_bert": df.d_bert.to_numpy(np.int8),
                "s_bert": df.s_bert.to_numpy(np.float64),
                "d_gpt2m": df.d_gpt2m.to_numpy(np.int8),
                "s_gpt2m": df.s_gpt2m.to_numpy(np.float64),
                "im_joined": df.dep.notna().to_numpy(bool),
                "lemma": _obj(df.lemma), "dep": _obj(df.dep),
                "person": df.person.to_numpy(np.float64),
                "number": _obj(df.number),
                "is_title": df.is_title.astype("float64").to_numpy(),
                "word_initial": df.word_initial.astype("float64").to_numpy(),
                "head_i": df.head_i.to_numpy(np.float64),
                "content_start": S["cstart"][il],
            }
            for f in STR_FEATS + BOOL_FEATS + INT_FEATS + FLT_FEATS:
                cols[f] = feats[f]
            cols.update({
                "line_norm_hash": S["norm_hash"][il],
                "line_dup_count": S["dup"][il],
                "speaker": np.array(spk, dtype=object),
                "turn_marker": S["turn_marker"][il],
                "prev_line_end": S["prev_end"][il],
                "char0": feats["char_start"] == 0,
                "speaker_change": S["spk_change"][il],
                "childes_is_child": (np.array([s == "CHI" for s in spk], dtype=float)
                                     if genre == "childes"
                                     else np.full(len(df), np.nan)),
                "doc_id": doc_id,
                "tokens_since_doc_start": tsds,
                "age_months": age,
                "cluster_id": cluster,
            })
            tables.append(pa.table(cols))
            info = {"n": int(len(df)), "im_joined": n_join,
                    "im_form_mismatch": form_mismatch,
                    "tok_ok": int(feats["tok_ok"].sum()),
                    "lines": len(lines), "docs": len(S["headers"]),
                    "seconds_structure": round(t_struct, 1),
                    "seconds_tokenize": round(t_tok, 1),
                    "seconds_total": round(time.time() - t0, 1)}
            meta_info["genres"][genre] = info
            print(genre, info, flush=True)
            del df, im, lines, S, feats
    tab = pa.concat_tables(tables, promote_options="default")
    pq.write_table(tab, out_dir / "population_features.parquet",
                   compression="zstd", row_group_size=1_000_000)
    meta_info["seconds_total"] = round(time.time() - T0, 1)
    meta_info["n_rows"] = tab.num_rows
    meta_info["workers"] = a.workers
    (out_dir / "population_features.meta.json").write_text(
        json.dumps(meta_info, indent=2))
    print("wrote", out_dir / "population_features.parquet", tab.num_rows,
          f"{meta_info['seconds_total']}s")


if __name__ == "__main__":
    main()
