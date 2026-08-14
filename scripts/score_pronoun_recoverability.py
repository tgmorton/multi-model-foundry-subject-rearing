#!/usr/bin/env python3
"""Corpus-wide recoverability scoring pass (graded pronoun-drop study).

For one genre file of one corpus (train_90M or pull_10M), walk the
annotation DocBin/linemap cache, enumerate every overt subject-pronoun
instance (PRON + normalized dep in {nsubj, nsubj:pass} — same criterion the
graded ablation will consume, same (file, line_idx, token_i) addressing),
and bank per-instance sufficient statistics under one or more trained
causal LMs:

- log P of each of the pronoun's subword pieces (form surprisal = -sum)
- the model's probability for every pronoun-inventory first-piece at the
  instance's slot (phi-feature surprisal / slot entropy are post-hoc sums)
- full next-token distribution entropy at the slot
- per-token log P for EVERY token in the corpus stream (enables SLOR and
  any future normalization without a GPU)

Context policy: the corpus is scored as ONE continuous token stream per
file (lines concatenated with no separator, crossing document boundaries)
in sliding windows of --window at --stride. This exactly matches how the
training chunker built model inputs (model_foundry/data.py concatenates
lines with zero separator), so every scored position sees fully
in-distribution context of >= stride tokens (window 0 excepted).

Tokenization matches training byte-for-byte: raw SentencePieceProcessor
encode of each line's raw_text, no BOS/EOS (tokenize_dataset.py path).

Outputs under --out-dir/<corpus>/:
  instances/<file>.parquet    one row per subject-pronoun instance
  per_token/<file>.<scorer>.parquet   per-line log-prob vectors
  lines/<file>.parquet        line_idx -> stream offset, n_pieces, doc_idx
  manifest/<file>.json        config, inventory ids, counters, throughput
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_WORD_RE = re.compile(r"\S+")
WORD_MARKER = "▁"  # SentencePiece ▁

# Subject-pronoun lexicon: form -> (person, number). Case is irrelevant
# here (all instances are subject-position by selection).
PRONOUN_LEXICON: Dict[str, Tuple[Optional[int], Optional[str]]] = {
    "i": (1, "Sing"), "we": (1, "Plur"),
    "you": (2, None), "thou": (2, "Sing"), "ye": (2, "Plur"),
    "he": (3, "Sing"), "she": (3, "Sing"), "it": (3, "Sing"),
    "they": (3, "Plur"), "one": (3, "Sing"),
}
# Inventory over which slot probabilities are recorded (order is the
# column order of the stored probability vectors).
INVENTORY_FORMS = ["i", "you", "he", "she", "it", "we", "they", "one"]


# --------------------------------------------------------------------------
# Alignment: spaCy token -> subword piece span within a line
# --------------------------------------------------------------------------

def word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def piece_word_starts(pieces: List[str]) -> List[int]:
    return [i for i, p in enumerate(pieces) if p.startswith(WORD_MARKER)]


def align_token_to_pieces(
    tok_char_start: int,
    tok_text: str,
    text: str,
    pieces: List[str],
) -> Optional[Tuple[int, int]]:
    """Return (first_piece_idx, n_pieces) for a token that begins a
    whitespace word, or None when alignment is not possible (mid-word
    token, or marker structure doesn't cover the words).

    Only handles word-initial tokens — subject pronouns virtually always
    begin their whitespace word ("he's" -> "he"); callers count the rest.
    """
    spans = word_spans(text)
    widx = None
    for i, (s, e) in enumerate(spans):
        if s == tok_char_start:
            widx = i
            break
        if s > tok_char_start:
            return None  # token starts mid-word
    if widx is None:
        return None

    starts = piece_word_starts(pieces)
    if len(starts) != len(spans):
        # Metaspace normally yields exactly one ▁-piece per whitespace
        # word; NFKC edge cases can break that. Skip + count.
        return None

    first = starts[widx]
    end = starts[widx + 1] if widx + 1 < len(starts) else len(pieces)
    # Pieces covering just the pronoun: accumulate visible chars.
    need = len(tok_text)
    got = 0
    n = 0
    for p in pieces[first:end]:
        got += len(p.lstrip(WORD_MARKER)) if n == 0 else len(p)
        n += 1
        if got >= need:
            break
    return first, n


# --------------------------------------------------------------------------
# Phase A — enumerate lines + instances (CPU)
# --------------------------------------------------------------------------

def phase_a(annotated_dir: Path, file_stem: str, sp, limit_lines: Optional[int],
            counters: Dict[str, int]):
    from preprocessing.annotate import iter_annotated_file
    from preprocessing.dep_labels import normalize_dep

    lines = []       # dicts: line_idx, doc_idx, ids(list[int])
    instances = []   # dicts: line-level address + alignment + lexicon

    for entry, doc in iter_annotated_file(annotated_dir, file_stem):
        line_idx = entry["line_idx"]
        if limit_lines is not None and line_idx >= limit_lines:
            break
        # linemap raw_text keeps the trailing newline; both the spaCy parse
        # (annotate.py) and training tokenization (HF text loader) strip it.
        raw = entry.get("raw_text", "").rstrip("\n\r")
        ids = sp.encode(raw, out_type=int) if raw else []
        lines.append({"line_idx": line_idx,
                      "doc_idx": entry.get("doc_idx"),
                      "ids": ids})
        counters["lines"] += 1
        if doc is None:
            counters["passthrough_lines"] += 1
            continue
        if doc.text != raw:
            counters["doc_text_mismatch"] += 1
            continue

        cands = [t for t in doc
                 if t.pos_ == "PRON"
                 and normalize_dep(t.dep_) in ("nsubj", "nsubj:pass")]
        if not cands:
            continue
        pieces = sp.encode(raw, out_type=str)
        for t in cands:
            form = t.text.lower()
            counters["instances_seen"] += 1
            aligned = align_token_to_pieces(t.idx, t.text, raw, pieces)
            if aligned is None:
                counters["instances_unaligned"] += 1
                continue
            first, n_pieces = aligned
            person, number = PRONOUN_LEXICON.get(form, (None, None))
            instances.append({
                "line_idx": line_idx,
                "token_i": t.i,
                "form": form,
                "lemma": t.lemma_.lower(),
                "dep": normalize_dep(t.dep_),
                "person": person,
                "number": number,
                "in_lexicon": form in PRONOUN_LEXICON,
                "piece_in_line": first,
                "n_pieces": n_pieces,
                "head_i": t.head.i,
            })
            counters["instances_aligned"] += 1
    return lines, instances


# --------------------------------------------------------------------------
# Phase B — stream-window scoring (GPU)
# --------------------------------------------------------------------------

def load_causal_model(ckpt_dir: Path, device: str):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(str(ckpt_dir))
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    weights = ckpt_dir / "model.safetensors"
    if weights.exists():
        state = load_file(str(weights), device="cpu")
    else:
        import torch as _t
        state = _t.load(ckpt_dir / "pytorch_model.bin", map_location="cpu",
                        weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"{ckpt_dir}: unexpected keys {unexpected[:5]}")
    tied = set(getattr(model, "_tied_weights_keys", None) or [])
    bad = [k for k in missing if k not in tied]
    if bad:
        raise RuntimeError(f"{ckpt_dir}: missing non-tied keys {bad[:5]}")
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model.to(device).eval()


def resolve_final_checkpoint(root: Path) -> Path:
    from evaluation.runners.per_model_runner import find_checkpoints_sorted
    ckpts = find_checkpoints_sorted(root)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {root}")
    return ckpts[-1][1]


def scored_range(start: int, window: int, stride: int, n: int):
    """Global positions scored by the window at `start` = [lo, hi).
    Window 0 scores from position 1; later windows score only their back
    (window - stride) region, so every position is scored exactly once."""
    lo = 1 if start == 0 else start + stride
    hi = min(start + window, n)
    return lo, hi


def phase_b(lines, instances, scorers: Dict[str, Path], inv_ids: List[List[int]],
            device: str, window: int, stride: int, batch_windows: int,
            use_amp: bool, counters: Dict[str, int]):
    import torch

    ids_per_line = [np.asarray(l["ids"], dtype=np.int32) for l in lines]
    offsets = np.zeros(len(ids_per_line) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in ids_per_line], out=offsets[1:])
    stream = (np.concatenate(ids_per_line) if ids_per_line
              else np.zeros(0, dtype=np.int32))
    n = len(stream)
    counters["stream_tokens"] = int(n)

    # Instance global positions in the token stream.
    line_pos = {l["line_idx"]: i for i, l in enumerate(lines)}
    for inst in instances:
        li = line_pos[inst["line_idx"]]
        inst["pos"] = int(offsets[li]) + inst["piece_in_line"]

    flat_inv = [i for grp in inv_ids for i in grp]
    inv_tensor = torch.tensor(flat_inv, dtype=torch.long, device=device)

    results = {}
    stream_t = torch.from_numpy(stream.astype(np.int64))

    window_starts = [0]
    k = 1
    while k * stride + stride < n:
        window_starts.append(k * stride)
        k += 1

    for name, ckpt in scorers.items():
        t0 = time.time()
        model = load_causal_model(ckpt, device)
        logprobs = np.full(n, np.nan, dtype=np.float32)
        ent = np.full(len(instances), np.nan, dtype=np.float32)
        invp = np.full((len(instances), len(flat_inv)), np.nan, dtype=np.float32)
        inst_index = {(inst["pos"]): [] for inst in instances}
        for idx, inst in enumerate(instances):
            inst_index[inst["pos"]].append(idx)

        with torch.inference_mode():
            for b0 in range(0, len(window_starts), batch_windows):
                starts = window_starts[b0:b0 + batch_windows]
                full = [s for s in starts if s + window <= n]
                tail = [s for s in starts if s + window > n]
                batches = []
                if full:
                    batches.append((full, torch.stack(
                        [stream_t[s:s + window] for s in full]).to(device)))
                for s in tail:  # ragged tail windows run singly
                    batches.append(([s], stream_t[s:min(s + window, n)]
                                    .unsqueeze(0).to(device)))
                for b_starts, inp in batches:
                    ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                           if use_amp else torch.no_grad())
                    with ctx:
                        out = model(input_ids=inp)
                    logits = out.logits.float()
                    lsm = torch.log_softmax(logits, dim=-1)
                    for row, s in enumerate(b_starts):
                        wlen = inp.shape[1]
                        lo, hi = scored_range(s, window, stride, n)
                        lo = max(lo, s + 1)
                        hi = min(hi, s + wlen)
                        if hi <= lo:
                            continue
                        # logits[t] predicts token at s+t+1
                        pred = lsm[row, lo - s - 1:hi - s - 1, :]
                        tgt = stream_t[lo:hi].to(device)
                        logprobs[lo:hi] = (
                            pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                            .cpu().numpy())
                        # instance rows inside this scored span
                        for p in range(lo, hi):
                            if p not in inst_index:
                                continue
                            rowvec = pred[p - lo]
                            probs = rowvec.exp()
                            e = float(-(probs * rowvec).sum())
                            iv = probs[inv_tensor].cpu().numpy()
                            for idx in inst_index[p]:
                                ent[idx] = e
                                invp[idx] = iv
        dt = time.time() - t0
        counters[f"seconds_{name}"] = round(dt, 1)
        results[name] = {"logprobs": logprobs, "entropy": ent, "inv": invp}
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    return results, offsets


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["train_90M", "pull_10M"])
    ap.add_argument("--file", required=True, help="genre stem, e.g. switchboard")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/data"))
    ap.add_argument("--spacy-model", default="en_core_web_trf")
    ap.add_argument("--tokenizer-dir", type=Path, default=None)
    ap.add_argument("--scorer", action="append", required=True,
                    metavar="NAME=CKPT_ROOT",
                    help="repeatable; CKPT_ROOT holds checkpoint-*/ dirs")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=500)
    ap.add_argument("--batch-windows", type=int, default=8)
    ap.add_argument("--limit-lines", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    import sentencepiece as spm

    from preprocessing.annotate import compute_annotation_cache_key, DOCBIN_ATTRS

    corpus_dir = f"{args.data_root}/raw/en/{args.corpus}/"
    cache_key = compute_annotation_cache_key(
        corpus_dir, args.spacy_model, DOCBIN_ATTRS)
    annotated_dir = args.data_root / "annotated" / cache_key
    tok_dir = args.tokenizer_dir or (args.data_root / "tokenizers" / "en_shared_unigram")
    out_root = (args.out_dir or (args.data_root / "recoverability")) / args.corpus
    for sub in ("instances", "per_token", "lines", "manifest"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_dir / "tokenizer.model"))

    # Inventory ids: lowercase + capitalized first-piece variants per form.
    inv_ids: List[List[int]] = []
    inv_meta = {}
    unk = sp.unk_id()
    for form in INVENTORY_FORMS:
        grp = []
        for v in {form, form.capitalize()}:
            pid = sp.piece_to_id(WORD_MARKER + v)
            if pid != unk:
                grp.append(pid)
        inv_ids.append(grp)
        inv_meta[form] = grp
    counters: Dict[str, int] = {k: 0 for k in (
        "lines", "passthrough_lines", "doc_text_mismatch",
        "instances_seen", "instances_aligned", "instances_unaligned")}

    scorers = {}
    for spec in args.scorer:
        name, root = spec.split("=", 1)
        scorers[name] = resolve_final_checkpoint(Path(root))
        print(f"scorer {name}: {scorers[name]}", flush=True)

    print(f"=== phase A: {args.corpus}/{args.file} from {annotated_dir}", flush=True)
    t0 = time.time()
    lines, instances = phase_a(annotated_dir, args.file, sp,
                               args.limit_lines, counters)
    counters["seconds_phase_a"] = round(time.time() - t0, 1)
    print(f"  {counters}", flush=True)

    print("=== phase B: scoring", flush=True)
    results, offsets = phase_b(
        lines, instances, scorers, inv_ids, args.device,
        args.window, args.stride, args.batch_windows, args.amp, counters)

    # ---- write outputs ----
    fname = args.file
    inst_cols = {
        "line_idx": [i["line_idx"] for i in instances],
        "token_i": [i["token_i"] for i in instances],
        "pos": [i["pos"] for i in instances],
        "piece_in_line": [i["piece_in_line"] for i in instances],
        "n_pieces": [i["n_pieces"] for i in instances],
        "form": [i["form"] for i in instances],
        "lemma": [i["lemma"] for i in instances],
        "dep": [i["dep"] for i in instances],
        "person": [i["person"] for i in instances],
        "number": [i["number"] for i in instances],
        "in_lexicon": [i["in_lexicon"] for i in instances],
        "head_i": [i["head_i"] for i in instances],
    }
    for name, res in results.items():
        lp = res["logprobs"]
        sums, firsts = [], []
        for i in instances:
            p0, np_ = i["pos"], i["n_pieces"]
            seg = lp[p0:p0 + np_]
            firsts.append(float(lp[p0]) if not math.isnan(float(lp[p0])) else None)
            sums.append(float(np.nansum(seg)) if not np.all(np.isnan(seg)) else None)
        inst_cols[f"{name}__logprob_sum"] = sums
        inst_cols[f"{name}__logprob_first"] = firsts
        inst_cols[f"{name}__entropy"] = [
            None if math.isnan(float(x)) else float(x) for x in res["entropy"]]
        inst_cols[f"{name}__inv_probs"] = [
            None if np.all(np.isnan(row)) else row.astype(float).tolist()
            for row in res["inv"]]
    pq.write_table(pa.table(inst_cols),
                   out_root / "instances" / f"{fname}.parquet")

    line_tbl = pa.table({
        "line_idx": [l["line_idx"] for l in lines],
        "doc_idx": [l["doc_idx"] for l in lines],
        "stream_offset": offsets[:-1].tolist(),
        "n_pieces": [len(l["ids"]) for l in lines],
    })
    pq.write_table(line_tbl, out_root / "lines" / f"{fname}.parquet")

    for name, res in results.items():
        # ListArray over the flat float32 stream — no per-line python lists.
        list_arr = pa.ListArray.from_arrays(
            pa.array(offsets.astype(np.int64), type=pa.int64()),
            pa.array(res["logprobs"], type=pa.float32()),
        )
        pq.write_table(pa.table({
            "line_idx": pa.array([l["line_idx"] for l in lines]),
            "logprobs": list_arr,
        }), out_root / "per_token" / f"{fname}.{name}.parquet")

    manifest = {
        "corpus": args.corpus, "file": fname,
        "annotated_dir": str(annotated_dir),
        "tokenizer": str(tok_dir),
        "window": args.window, "stride": args.stride, "amp": args.amp,
        "scorers": {k: str(v) for k, v in scorers.items()},
        "inventory_ids": inv_meta,
        "inventory_order": INVENTORY_FORMS,
        "counters": counters,
    }
    with open(out_root / "manifest" / f"{fname}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(counters, indent=2))
    print("SCORING OK", flush=True)


if __name__ == "__main__":
    main()
