#!/usr/bin/env python3
"""Score natural-drop sites with the selection rater (roberta-large, +-250 pieces of the raw corpus stream).

The corpus ranking (`phase_b_mlm` in scripts/score_pronoun_recoverability.py) tokenizes each raw line on its
own, concatenates the lines, masks the pronoun's piece and reads the rater's distribution there. This does
the same for:

  null sites    a subject slot is opened where an overt subject would stand (the line is re-tokenized with a
                placeholder pronoun at that offset, and the placeholder's piece is masked);
  overt sites   the pronoun's own piece is masked, as in the ranking (the stored score is reproduced as a check).

Per site: probability of each personal pronoun at the slot (summed over case and leading-space variants),
of a few non-pronoun fillers, the full-vocabulary entropy and the top pieces.

    python analysis/recoverability/natural_drop/score_sites.py --main <main checkout> --dir <site dir> [--limit N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

FORMS = ["i", "you", "he", "she", "it", "we", "they", "one"]
EXTRA = ["that", "there", "this", "what", "who", "let", "do", "to", "and", "mommy", "daddy"]
CTX = 250


def variants(tok, form: str) -> list[int]:
    out = set()
    for v in {form, form.capitalize()}:
        for cand in (" " + v, v):
            ids = tok(cand, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
    return sorted(out)


def stream_of(raw: list[str], tok, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return z["stream"], z["offsets"]
    ids = []
    for b0 in range(0, len(raw), 100000):
        ids += tok(raw[b0:b0 + 100000], add_special_tokens=False)["input_ids"]
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in ids], out=offsets[1:])
    stream = np.fromiter((t for x in ids for t in x), dtype=np.int32, count=int(offsets[-1]))
    np.savez(cache, stream=stream, offsets=offsets)
    return stream, offsets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--hf-mlm", default="roberta-large")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score a random subset of each table (timing, checks)")
    ap.add_argument("--out", default="scores.parquet")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.hf_mlm, use_fast=True)
    raw = (a.main / "data/raw/train_90M/childes.train").read_text(encoding="utf-8").split("\n")
    if raw[-1] == "":
        raw = raw[:-1]
    t0 = time.time()
    stream, offsets = stream_of(raw, tok, a.dir / "childes_stream_roberta.npz")
    print(f"stream {len(stream):,} pieces over {len(offsets) - 1:,} lines ({time.time() - t0:.0f}s)", flush=True)
    ref = pd.read_parquet(a.main / "data/recoverability/train_90M/lines/childes.parquet", columns=["line_idx", "n_pieces"])
    mine = np.diff(offsets)[ref.line_idx.to_numpy()]
    print(f"lines whose piece count matches the ranking's: {np.mean(mine == ref.n_pieces.to_numpy()):.5f}", flush=True)

    null = pd.read_parquet(a.dir / "null_sites.parquet")
    null = null[~null.nominal_before & ~null.glued & null.verb_class.isin(["inflected", "stative", "bare"])
                & null.role.isin(["child", "adult"])]
    overt = pd.read_parquet(a.dir / "overt_sites.parquet")
    if a.limit:
        null = null.sample(min(a.limit, len(null)), random_state=0)
        overt = overt.sample(min(a.limit, len(overt)), random_state=0)

    # ---- build (left, right, truth) per site
    sites, meta = [], []
    enc = tok(null.scored_line.tolist(), add_special_tokens=False, return_offsets_mapping=True)
    bad = 0
    for k, r in enumerate(null.itertuples(index=False)):
        ids, offs, s = enc["input_ids"][k], enc["offset_mapping"][k], int(r.char_start)
        j = next((q for q, (x, _) in enumerate(offs) if x == s or (x == s - 1 and s > 0 and r.scored_line[s - 1] == " ")), None)
        if j is None or tok.decode([ids[j]]).strip() != "it":
            bad += 1
            continue
        L = int(r.line_idx)
        left = np.concatenate([stream[max(0, offsets[L] - CTX):offsets[L]], np.asarray(ids[:j], dtype=np.int32)])[-CTX:]
        right = np.concatenate([np.asarray(ids[j + 1:], dtype=np.int32), stream[offsets[L + 1]:offsets[L + 1] + CTX]])[:CTX]
        sites.append((left, right, -1))
        meta.append(("null", r.sentence_id, L, int(r.verb_idx), -1, j))
    print(f"null sites {len(null):,}; placeholder not aligned {bad}", flush=True)
    # overt pronouns: the instance tables index spaCy tokens of the raw line; find the pronoun's roberta piece
    import spacy
    nlp = spacy.blank("en")
    lines = overt.line_text.tolist()
    enc = tok(lines, add_special_tokens=False, return_offsets_mapping=True)
    skipped = {"token_mismatch": 0, "not_one_piece": 0}
    for k, r in enumerate(overt.itertuples(index=False)):
        doc, ti = nlp.make_doc(r.line_text), int(r.token_i)
        if ti >= len(doc) or doc[ti].text.lower() != r.form:
            skipped["token_mismatch"] += 1
            continue
        s, e = doc[ti].idx, doc[ti].idx + len(doc[ti].text)
        offs = enc["offset_mapping"][k]
        j = next((q for q, (x, _) in enumerate(offs) if x == s or (x == s - 1 and s > 0 and r.line_text[s - 1] == " ")), None)
        if j is None or offs[j][1] < e:
            skipped["not_one_piece"] += 1
            continue
        L = int(r.line_idx)
        p = int(offsets[L] + j)
        assert stream[p] == enc["input_ids"][k][j]
        sites.append((stream[max(0, p - CTX):p], stream[p + 1:p + 1 + CTX], int(stream[p])))
        meta.append(("overt", "", L, -1, ti, j))
    print(f"overt sites {len(overt):,}; skipped {skipped}", flush=True)
    print(f"sites to score {len(sites):,} on {device}", flush=True)

    inv = [variants(tok, f) for f in FORMS]
    ext = [variants(tok, f) for f in EXTRA]
    model = AutoModelForMaskedLM.from_pretrained(a.hf_mlm).to(device).eval()
    cls_id, sep_id, mask_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id, tok.pad_token_id
    n = len(sites)
    P = np.full((n, len(FORMS)), np.nan, dtype=np.float32)
    X = np.full((n, len(EXTRA)), np.nan, dtype=np.float32)
    ent = np.full(n, np.nan, dtype=np.float32)
    lp_true = np.full(n, np.nan, dtype=np.float32)
    top_id = np.full((n, 5), -1, dtype=np.int32)
    top_p = np.full((n, 5), np.nan, dtype=np.float32)
    order = sorted(range(n), key=lambda i: len(sites[i][0]) + len(sites[i][1]))
    t0 = time.time()
    with torch.inference_mode():
        for b0 in range(0, n, a.batch):
            idxs = order[b0:b0 + a.batch]
            seqs = [[cls_id] + sites[i][0].tolist() + [mask_id] + sites[i][1].tolist() + [sep_id] for i in idxs]
            slots = [len(sites[i][0]) + 1 for i in idxs]
            Lm = max(len(s) for s in seqs)
            batch = torch.full((len(seqs), Lm), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), Lm), dtype=torch.long)
            for r, s in enumerate(seqs):
                batch[r, :len(s)] = torch.tensor(s)
                attn[r, :len(s)] = 1
            logits = model(input_ids=batch.to(device), attention_mask=attn.to(device)).logits
            rows = torch.stack([logits[r, slots[r]] for r in range(len(seqs))]).float()
            lsm = torch.log_softmax(rows, dim=-1).cpu()
            pr = lsm.exp()
            for r, i in enumerate(idxs):
                P[i] = [float(pr[r, g].sum()) for g in inv]
                X[i] = [float(pr[r, g].sum()) for g in ext]
                ent[i] = float(-(pr[r] * lsm[r]).sum())
                tp = torch.topk(pr[r], 5)
                top_id[i], top_p[i] = tp.indices.numpy(), tp.values.numpy()
                if sites[i][2] >= 0:
                    lp_true[i] = float(lsm[r, sites[i][2]])
            if (b0 // a.batch) % 50 == 0:
                done = b0 + len(idxs)
                el = time.time() - t0
                print(f"  {done}/{n} ({el:.0f}s, {done / max(el, 1e-9):.1f}/s)", flush=True)
    out = pd.DataFrame(meta, columns=["kind", "sentence_id", "line_idx", "verb_idx", "token_i", "piece_in_line"])
    for c, f in enumerate(FORMS):
        out[f"p_{f}"] = P[:, c]
    for c, f in enumerate(EXTRA):
        out[f"x_{f}"] = X[:, c]
    out["entropy"], out["logprob_true"] = ent, lp_true
    out["top_pieces"] = [" | ".join(f"{tok.decode([int(t)]).strip() or '∅'}:{p:.2f}" for t, p in zip(top_id[i], top_p[i])) for i in range(n)]
    out["top1"] = [tok.decode([int(top_id[i, 0])]).strip() for i in range(n)]
    out["top1_p"] = top_p[:, 0]
    out.to_parquet(a.dir / a.out, index=False)
    print(f"wrote {a.dir / a.out} ({len(out):,} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
