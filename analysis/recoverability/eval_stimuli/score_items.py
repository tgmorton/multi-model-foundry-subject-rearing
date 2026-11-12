#!/usr/bin/env python3
"""Score the pronoun of each candidate item with the rater, the item being all the rater sees.

An item (build_items.py) carries `rater_lines`: its context and target as raw corpus lines,
the last one cut at the end of the target sentence. Each line is tokenized on its own and
the lines are concatenated with no separator, exactly as the training corpus was ranked
(`phase_b_mlm` in scripts/score_pronoun_recoverability.py); every piece of the pronoun is
masked and the surprisal is minus the summed log-probability of the true pieces. Because an
item is far shorter than the +-250-token window, the whole item is the context, so the
score says how recoverable the pronoun is from the stimulus alone. The frozen robbi decile
thresholds are applied to it.

`--check N` re-scores N items through `phase_b_mlm` itself and fails if the two disagree.

    python analysis/recoverability/eval_stimuli/score_items.py --items <dir>/items_candidates.parquet \\
        --out <dir>/items_robbi.parquet --check 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_corpus_instances import ROBBI_THRESHOLDS  # noqa: E402


def encode_items(items: pd.DataFrame, tok, aligner, S, lines_col="rater_lines", start_col="rater_char_start",
                 word_col="form_text"):
    """Per item: token ids (lines tokenized separately, concatenated), mask start, piece count."""
    all_lines, owner = [], []
    parsed = [json.loads(x) for x in items[lines_col]]
    for k, ls in enumerate(parsed):
        all_lines += ls
        owner += [k] * len(ls)
    enc = []
    for b0 in range(0, len(all_lines), 50000):
        enc += tok(all_lines[b0:b0 + 50000], add_special_tokens=False)["input_ids"]
    ids, pos, npieces, wi = [None] * len(items), np.full(len(items), -1), np.zeros(len(items), int), np.zeros(len(items), bool)
    cur = 0
    for k, ls in enumerate(parsed):
        mine = enc[cur:cur + len(ls)]
        cur += len(ls)
        al = aligner.align_line(ls[-1], [S._Shim(int(items[start_col].iat[k]), items[word_col].iat[k])])[0]
        ids[k] = np.concatenate([np.asarray(m, dtype=np.int64) for m in mine]) if mine else np.zeros(0, np.int64)
        if al is None:
            continue
        pos[k] = sum(len(m) for m in mine[:-1]) + al[0]
        npieces[k], wi[k] = al[1], al[2]
    return ids, pos, npieces, wi


def score(ids, pos, npieces, tok, model, device, batch_size=64, max_len=510, log=True, amp=False):
    import torch
    cls_id, sep_id, mask_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id, tok.pad_token_id or 0
    n = len(ids)
    surp = np.full(n, np.nan)
    first = np.full(n, np.nan)
    ent = np.full(n, np.nan)
    top = np.full(n, -1, dtype=np.int64)
    ok = [k for k in range(n) if pos[k] >= 0]
    order = sorted(ok, key=lambda k: len(ids[k]))
    t0 = time.time()
    with torch.inference_mode():
        for b0 in range(0, len(order), batch_size):
            idxs = order[b0:b0 + batch_size]
            seqs, slots, truths = [], [], []
            for k in idxs:
                x, p, m = ids[k].copy(), int(pos[k]), int(npieces[k])
                if len(x) > max_len:                      # keep the pronoun, trim the far left
                    cut = len(x) - max_len
                    x, p = x[cut:], p - cut
                truths.append(x[p:p + m].tolist())
                x[p:p + m] = mask_id
                seqs.append([cls_id] + x.tolist() + [sep_id])
                slots.append(p + 1)
            L = max(len(s) for s in seqs)
            batch = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, s in enumerate(seqs):
                batch[r, :len(s)] = torch.tensor(s)
                attn[r, :len(s)] = 1
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                logits = model(input_ids=batch.to(device), attention_mask=attn.to(device)).logits
            logits = logits.float()
            for r, k in enumerate(idxs):
                lsm = torch.log_softmax(logits[r, slots[r]:slots[r] + len(truths[r])], dim=-1)
                lp = [float(lsm[j, t]) for j, t in enumerate(truths[r])]
                surp[k], first[k] = -sum(lp), lp[0]
                ent[k] = float(-(lsm[0].exp() * lsm[0]).sum())
                top[k] = int(lsm[0].argmax())
            if log and (b0 // batch_size) % 100 == 0:
                print(f"  {b0}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    return surp, first, ent, top


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-mlm", default="roberta-large")
    ap.add_argument("--label", default="robbi")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n: score every n-th item starting at i")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on cuda, as the corpus ranking used")
    ap.add_argument("--check", type=int, default=0, help="verify N items against phase_b_mlm")
    ap.add_argument("--lines-col", default="rater_lines")
    ap.add_argument("--start-col", default="rater_char_start")
    ap.add_argument("--id-col", default="cand_id")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import score_pronoun_recoverability as S

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.hf_mlm, use_fast=True)
    aligner = S.HFAligner(tok)
    items = pd.read_parquet(a.items)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        items = items.iloc[i::n].reset_index(drop=True)
    if a.limit:
        items = items.sample(a.limit, random_state=0).reset_index(drop=True)
    if "form_text" not in items:
        last = items[a.lines_col].map(lambda x: json.loads(x)[-1])
        items["form_text"] = [l[s:s + len(f)] for l, s, f in zip(last, items[a.start_col], items.form)]
        bad = (items.form_text.str.lower() != items.form).sum()
        if bad:
            sys.exit(f"FAIL: {bad} items whose pronoun is not at rater_char_start")
    t0 = time.time()
    ids, pos, npieces, wi = encode_items(items, tok, aligner, S, a.lines_col, a.start_col)
    print(f"{len(items):,} items encoded in {time.time() - t0:.0f}s; unaligned {(pos < 0).sum()}; "
          f"median length {int(np.median([len(x) for x in ids]))} tokens", flush=True)
    model = AutoModelForMaskedLM.from_pretrained(a.hf_mlm, attn_implementation="eager").to(device).eval()
    surp, first, ent, top = score(ids, pos, npieces, tok, model, device, a.batch, amp=a.amp)

    if a.check:
        inv_ids = [aligner.inventory_ids(f) for f in S.INVENTORY_FORMS]
        rng = np.random.default_rng(0)
        worst = -1.0
        for k in rng.choice(np.flatnonzero(pos >= 0), size=min(a.check, int((pos >= 0).sum())), replace=False):
            ls = json.loads(items[a.lines_col].iat[k])
            enc = tok(ls, add_special_tokens=False)["input_ids"]
            lines = [{"line_idx": j, "ids": np.asarray(v, dtype=np.int32)} for j, v in enumerate(enc)]
            al = aligner.align_line(ls[-1], [S._Shim(int(items[a.start_col].iat[k]), items.form_text.iat[k])])[0]
            inst = [{"line_idx": len(ls) - 1, "piece_in_line": al[0], "n_pieces": al[1], "word_initial": al[2]}]
            res, _ = S.phase_b_mlm(lines, inst, a.label, a.hf_mlm, tok, inv_ids, device, 250, batch_size=1,
                                   use_amp=a.amp and device == "cuda", counters={}, model=model)
            ref = -float(res[a.label]["logprob_sum_arr"][0])
            # fp32 agrees to 1e-4. Under fp16 autocast a score moves by a few percent with the
            # batch it is padded into (the corpus ranking has the same jitter), so allow for it.
            rel, floor = (0.10, 0.005) if (a.amp and device == "cuda") else (0.005, 0.0005)
            worst = max(worst, abs(ref - surp[k]) - (rel * abs(ref) + floor))
        print(f"check against phase_b_mlm on {a.check} items: worst excess over tolerance {worst:.2e}", flush=True)
        if worst > 0:
            sys.exit("FAIL: item scoring disagrees with phase_b_mlm")

    out = items[[a.id_col]].copy()
    out["n_pieces"], out["word_initial_piece"], out["n_tokens"] = npieces, wi, [len(x) for x in ids]
    out["surp"], out["logprob_first"], out["entropy"] = surp, first, ent
    out["top1"] = [tok.convert_ids_to_tokens(int(t)) if t >= 0 else None for t in top]
    out["decile"] = np.where(np.isnan(surp), -1, np.searchsorted(ROBBI_THRESHOLDS, np.nan_to_num(surp), side="left")).astype(np.int8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    share = out[out.decile >= 0].decile.value_counts(normalize=True).sort_index().round(4)
    a.out.with_suffix(".meta.json").write_text(json.dumps({
        "items": str(a.items), "hf_mlm": a.hf_mlm, "label": a.label, "n": len(out), "unaligned": int((pos < 0).sum()),
        "thresholds": ROBBI_THRESHOLDS, "decile_share": share.to_dict()}, indent=2))
    print(share.to_string())
    print("SCORE OK", flush=True)


if __name__ == "__main__":
    main()
