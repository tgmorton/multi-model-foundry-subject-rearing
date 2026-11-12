#!/usr/bin/env python3
"""Score pronoun instances of a raw corpus file the way the training corpus was ranked.

Input is a raw corpus file and a table of instances addressed by (line_idx, char_start,
tok_text), e.g. from build_test_instances.py. Every line of the file is tokenized on its own
and the lines are concatenated with no separator; each instance is scored with all of its
pieces masked and +-`--mlm-ctx` tokens of that stream as context. This is `phase_b_mlm` from
scripts/score_pronoun_recoverability.py, so the surprisal is on the scale of selection v5
(`robbi` = roberta-large, +-250) and the frozen decile thresholds apply directly.

    python analysis/recoverability/eval_stimuli/score_corpus_instances.py \\
        --lines-file /mnt/data/raw/en/test_10M/switchboard.test \\
        --instances instances__switchboard.parquet --out robbi__switchboard.parquet --amp
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

# selection_v5/robbi/SELECTION_MANIFEST.json: upper surprisal bound (nats) of deciles 0..8.
ROBBI_THRESHOLDS = [0.0029813863802701235, 0.008927312679588795, 0.02135789766907692,
                    0.04446164146065712, 0.08588272333145142, 0.16255170106887817,
                    0.31855273246765137, 0.6843868494033813, 1.7035140991210938]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lines-file", type=Path, required=True)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-mlm", default="roberta-large")
    ap.add_argument("--label", default="robbi")
    ap.add_argument("--mlm-ctx", type=int, default=250)
    ap.add_argument("--mlm-batch", type=int, default=24)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N instances (smoke test)")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import score_pronoun_recoverability as S

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.hf_mlm, use_fast=True)
    aligner = S.HFAligner(tok)
    inv_ids = [aligner.inventory_ids(f) for f in S.INVENTORY_FORMS]
    inv_group = np.repeat(np.arange(len(inv_ids)), [len(g) for g in inv_ids])

    texts = [t.rstrip("\r") for t in a.lines_file.read_text(encoding="utf-8").split("\n")]
    inst = pd.read_parquet(a.instances).reset_index(drop=True)
    if a.limit:
        inst = inst.head(a.limit)
    t0 = time.time()
    ids = []
    for b0 in range(0, len(texts), 20000):
        ids += tok(texts[b0:b0 + 20000], add_special_tokens=False)["input_ids"]
    lines = [{"line_idx": k, "ids": np.asarray(v, dtype=np.int32)} for k, v in enumerate(ids)]
    print(f"{a.lines_file.name}: {len(texts):,} lines, {sum(len(v) for v in ids):,} tokens, "
          f"{len(inst):,} instances, tokenized in {time.time() - t0:.0f}s", flush=True)

    instances, keep, text_mismatch, unaligned = [], [], 0, 0
    for line_idx, g in inst.groupby("line_idx", sort=True):
        text = texts[int(line_idx)]
        cands, rows = [], []
        for row, cs, w in zip(g.index, g.char_start, g.tok_text):
            if text[int(cs):int(cs) + len(w)] != w:
                text_mismatch += 1
                continue
            cands.append(S._Shim(int(cs), w))
            rows.append(row)
        if not cands:
            continue
        for row, al in zip(rows, aligner.align_line(text, cands)):
            if al is None:
                unaligned += 1
                continue
            instances.append({"line_idx": int(line_idx), "piece_in_line": al[0], "n_pieces": al[1], "word_initial": al[2]})
            keep.append(row)
    print(f"aligned {len(instances):,}; text mismatch {text_mismatch}, unaligned {unaligned}", flush=True)
    if text_mismatch > 0.002 * max(len(inst), 1):
        sys.exit("FAIL: the lines file does not match the one the instances were built from")

    counters: dict = {}
    model = AutoModelForMaskedLM.from_pretrained(a.hf_mlm, attn_implementation="eager").to(device).eval()
    res, _ = S.phase_b_mlm(lines, instances, a.label, a.hf_mlm, tok, inv_ids, device, a.mlm_ctx,
                           batch_size=a.mlm_batch, use_amp=a.amp and device == "cuda", counters=counters, model=model)
    r = res[a.label]
    out = inst.loc[keep, ["stem", "line_idx", "char_start", "tok_text"]].reset_index(drop=True)
    out["n_pieces"] = [i["n_pieces"] for i in instances]
    out["word_initial_piece"] = [bool(i["word_initial"]) for i in instances]
    out["surp"] = -r["logprob_sum_arr"].astype(np.float64)
    out["logprob_first"] = r["logprob_first_arr"]
    out["entropy"] = r["entropy"]
    inv = np.stack([np.bincount(inv_group, weights=row, minlength=len(inv_ids)) for row in r["inv"]])
    for j, f in enumerate(S.INVENTORY_FORMS):
        out[f"p_{f}"] = inv[:, j].astype(np.float32)
    out["decile"] = np.searchsorted(ROBBI_THRESHOLDS, out.surp.values, side="left").astype(np.int8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    meta = {"lines_file": str(a.lines_file), "hf_mlm": a.hf_mlm, "label": a.label, "mlm_ctx": a.mlm_ctx,
            "instances_in": len(inst), "scored": len(out), "text_mismatch": text_mismatch, "unaligned": unaligned,
            "thresholds": ROBBI_THRESHOLDS, "counters": counters,
            "decile_share": out.decile.value_counts(normalize=True).sort_index().round(4).tolist()}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta)[:600])
    print("SCORE OK", flush=True)


if __name__ == "__main__":
    main()
