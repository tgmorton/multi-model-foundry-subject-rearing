#!/usr/bin/env python3
"""Count "that" + verb strings in train_90M: as is, and the strings that pronoun removal creates.

Removal deletes personal-pronoun subjects only, so it never removes "that"; it creates a "that" + verb
string wherever a removed pronoun stood between "that" and a verb form. Verb forms: auxiliaries and modals
by list, other verbs by majority tag (VERB or AUX) in the annotated corpus, -ing forms excluded.

    .venv/bin/python analysis/eval_v2/adhoc/that_verb_counts.py <main checkout> <out dir>
"""
import collections
import json
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import spacy

MAIN, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
GENRES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
AUX = {"will", "'ll", "would", "'d", "can", "ca", "could", "shall", "should", "may", "might", "must", "is", "'s", "are", "'re", "was",
       "were", "am", "'m", "has", "have", "'ve", "had", "does", "do", "did", "wo", "ai"}
MODAL = {"will", "'ll", "would", "'d", "can", "ca", "could", "shall", "should", "may", "might", "must", "wo"}
tot, verb = collections.Counter(), collections.Counter()
for g in GENRES:
    t = pq.read_table(MAIN + f"data/output/train_90M/annotated_corpus/base/train_90M_{g}.parquet", columns=["tokens", "pos_tags"])
    df = pd.DataFrame({"w": pc.utf8_lower(pc.list_flatten(t["tokens"])).to_pandas(),
                       "v": pc.is_in(pc.list_flatten(t["pos_tags"]), value_set=pa.array(["VERB", "AUX"])).to_pandas()})
    a = df.groupby("w").v.agg(["sum", "size"])
    for w, s, n in zip(a.index, a["sum"], a["size"]):
        tot[w] += n
        verb[w] += s
FIN = {w for w, n in tot.items() if n >= 20 and verb[w] / n >= 0.6 and not w.endswith("ing") and w not in ("be", "been", "being", "to")} | AUX
rj = pd.read_parquet(MAIN + "data/recoverability/analysis/survey/robbi_joined.parquet",
                     columns=["genre", "line_idx", "token_i", "form", "d_robbi", "rand_decile"])
nlp = spacy.blank("en")
THAT = re.compile(r"\bthat\b", re.I)
base, rows = collections.Counter(), []
for g in GENRES:
    raw = open(MAIN + f"data/raw/train_90M/{g}.train", encoding="utf-8").read().split("\n")
    cand = [i for i, ln in enumerate(raw) if THAT.search(ln)]
    r = rj[(rj.genre == g) & rj.line_idx.isin(set(cand))]
    inst = {}
    for li, ti, f, d, rd in zip(r.line_idx, r.token_i, r.form, r.d_robbi, r.rand_decile):
        inst.setdefault(int(li), {})[int(ti)] = (f, int(d), int(rd))
    for li in cand:
        toks = [t.text.lower() for t in nlp.make_doc(raw[li])]
        here = inst.get(li, {})
        for i, w in enumerate(toks):
            if w != "that":
                continue
            base["that"] += 1
            if i + 1 < len(toks):
                nx = toks[i + 1]
                base["that+verb"] += nx in FIN
                base["that+aux"] += nx in AUX
                base["that+modal"] += nx in MODAL
                base["that+will"] += nx == "will"
                if (i + 1) in here and i + 2 < len(toks) and toks[i + 2] in FIN:
                    f, d, rd = here[i + 1]
                    rows.append((g, f.lower(), d, rd, toks[i + 2] in AUX, toks[i + 2] in MODAL, toks[i + 2] == "will"))
c = pd.DataFrame(rows, columns=["genre", "pron", "d_robbi", "rand_decile", "aux", "modal", "will"])
c.to_parquet(OUT + "that_pron_verb_instances.parquet")
json.dump({"base": dict(base), "n_tokens": sum(tot.values())}, open(OUT + "that_verb_base.json", "w"))
levels = {"unmodified": None, "all removed": np.ones(len(c), bool), "random 50%": (c.rand_decile < 5).to_numpy(),
          "most recoverable first 50%": (c.d_robbi < 5).to_numpy(), "least recoverable first 50%": (c.d_robbi >= 5).to_numpy()}
out = []
for name, m in levels.items():
    add = {"that+verb": 0, "that+aux": 0, "that+modal": 0, "that+will": 0} if m is None else {
        "that+verb": int(m.sum()), "that+aux": int((m & c.aux).sum()), "that+modal": int((m & c.modal).sum()), "that+will": int((m & c.will).sum())}
    out.append({"corpus": name, **{k: base[k] + add[k] for k in add}})
pd.set_option("display.width", 250)
print("personal-pronoun subjects:", len(rj), "| between 'that' and a verb form:", len(c))
print(pd.DataFrame(out).to_string(index=False))
print("decile of those pronouns (share):", c.d_robbi.value_counts(normalize=True).sort_index().round(3).to_dict())
