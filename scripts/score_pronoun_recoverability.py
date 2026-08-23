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
    """Return (first_piece_idx, n_pieces, word_initial) for a token, or
    None when alignment is not possible.

    Word-initial tokens ("he's" -> "he") map to their word's first piece.
    Mid-word tokens (quote-initial pronouns: '"He' -> pieces ['▁"', 'He'])
    are resolved by walking the word's pieces to the token's intra-word
    character offset; if the pronoun is fused into a piece with preceding
    characters, alignment is impossible at piece granularity -> None.
    """
    spans = word_spans(text)
    widx = None
    word_initial = False
    for i, (s, e) in enumerate(spans):
        if s <= tok_char_start < e:
            widx = i
            word_initial = (s == tok_char_start)
            break
        if s > tok_char_start:
            return None
    if widx is None:
        return None

    starts = piece_word_starts(pieces)
    if len(starts) != len(spans):
        # Metaspace normally yields exactly one ▁-piece per whitespace
        # word; normalizer-dropped chars can break that. Skip + count.
        return None

    wstart = starts[widx]
    wend = starts[widx + 1] if widx + 1 < len(starts) else len(pieces)

    if word_initial:
        first = wstart
    else:
        # Walk intra-word visible offsets to the token's start. Assumes
        # the normalizer preserved char lengths up to the token (true for
        # ASCII quotes/brackets; drifting cases fall out as unaligned).
        target = tok_char_start - spans[widx][0]
        off = 0
        first = None
        for j in range(wstart, wend):
            if off == target and j > wstart:
                first = j
                break
            off += (len(pieces[j].lstrip(WORD_MARKER)) if j == wstart
                    else len(pieces[j]))
        if first is None:
            return None  # fused with preceding chars

    # Pieces covering just the pronoun: accumulate visible chars.
    need = len(tok_text)
    got = 0
    n = 0
    for j in range(first, wend):
        p = pieces[j]
        got += len(p.lstrip(WORD_MARKER)) if j == wstart and n == 0 else len(p)
        n += 1
        if got >= need:
            break
    return first, n, word_initial


class SPAligner:
    """In-house SentencePiece path: byte-matches training tokenization."""

    def __init__(self, sp):
        self.sp = sp

    def encode_ids(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def align_line(self, text: str, cands):
        pieces = self.sp.encode(text, out_type=str)
        return [align_token_to_pieces(t.idx, t.text, text, pieces)
                for t in cands]

    def inventory_ids(self, form: str) -> List[int]:
        unk = self.sp.unk_id()
        out = set()
        for v in {form, form.capitalize()}:
            for piece in (WORD_MARKER + v, v):
                pid = self.sp.piece_to_id(piece)
                if pid != unk:
                    out.add(pid)
        return sorted(out)


class HFAligner:
    """External HF tokenizer path: char-offset alignment (BPE etc.)."""

    def __init__(self, tok):
        assert tok.is_fast, "external scorer needs a fast tokenizer"
        self.tok = tok

    def encode_ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def align_line(self, text: str, cands):
        enc = self.tok(text, add_special_tokens=False,
                       return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        out = []
        for t in cands:
            s, e = t.idx, t.idx + len(t.text)
            first = None
            for j, (a, b) in enumerate(offs):
                # GPT-2-style offsets start a Ġ-token at its leading space.
                if a == s or (a == s - 1 and s > 0 and text[s - 1] == " "):
                    first = j
                    break
                if a > s:
                    break
            if first is None:
                out.append(None)
                continue
            n = 0
            for j in range(first, len(offs)):
                n += 1
                if offs[j][1] >= e:
                    break
            word_initial = s == 0 or text[s - 1].isspace()
            out.append((first, n, word_initial))
        return out

    def inventory_ids(self, form: str) -> List[int]:
        out = set()
        for v in {form, form.capitalize()}:
            for cand in (" " + v, v):
                ids = self.tok(cand, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    out.add(ids[0])
        return sorted(out)


# --------------------------------------------------------------------------
# Phase A — enumerate lines + instances (CPU)
# --------------------------------------------------------------------------

def phase_a(annotated_dir: Path, file_stem: str, aligner, limit_lines: Optional[int],
            counters: Dict[str, int], sample=None):
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
        ids = aligner.encode_ids(raw) if raw else []
        # int32 array, not a python int list — the big shards (childes,
        # open_subtitles) OOM'd at 16Gi on list overhead alone.
        lines.append({"line_idx": line_idx,
                      "doc_idx": entry.get("doc_idx"),
                      "ids": np.asarray(ids, dtype=np.int32)})
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
        if sample is not None:
            cands = [t for t in cands if (line_idx, t.i) in sample]
            if not cands:
                continue
        alignments = aligner.align_line(raw, cands)
        # Matrix-verb (head) spans for the verb-frame planning window.
        head_alignments = aligner.align_line(raw, [t.head for t in cands])
        for t, aligned, h_aligned in zip(cands, alignments, head_alignments):
            form = t.text.lower()
            counters["instances_seen"] += 1
            if aligned is None:
                counters["instances_unaligned"] += 1
                continue
            first, n_pieces, word_initial = aligned
            if not word_initial:
                counters["instances_midword_recovered"] += 1
            person, number = PRONOUN_LEXICON.get(form, (None, None))
            instances.append({
                "line_idx": line_idx,
                "token_i": t.i,
                "piece_ids": [int(x) for x in ids[first:first + n_pieces]],
                "is_title": t.text[:1].isupper(),
                "word_initial": word_initial,
                "form": form,
                "lemma": t.lemma_.lower(),
                "dep": normalize_dep(t.dep_),
                "person": person,
                "number": number,
                "in_lexicon": form in PRONOUN_LEXICON,
                "piece_in_line": first,
                "n_pieces": n_pieces,
                "head_i": t.head.i,
                "head_piece_in_line": h_aligned[0] if h_aligned else None,
                "head_n_pieces": h_aligned[1] if h_aligned else None,
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


def phase_b_mlm(lines, instances, mlm_name: str, hf_id: str, tok,
                inv_ids: List[List[int]], device: str, ctx_halfwidth: int,
                batch_size: int, use_amp: bool, counters: Dict[str, int],
                ctx_right: Optional[int] = None, model=None):
    """Masked-slot scoring with an external MLM (BERT-family).

    For each instance: take +-ctx_halfwidth stream tokens around the slot,
    mask ALL of the pronoun's wordpieces (whole-word-masking style, matching
    the rater's pretraining), one forward per instance (batched). Banks:
      logprob_first  log P(true first piece | first mask position)
      logprob_sum    sum over mask positions of log P(true piece there)
                     (multi-mask conditional-independence approximation;
                     exact for the single-piece majority)
      entropy        full-vocab entropy at the first mask
      inv_probs      inventory-piece probabilities at the first mask
    No corpus-wide per-token PLL (130M masked forwards not worth it).
    """
    import torch
    from transformers import AutoModelForMaskedLM

    ids_per_line = [np.asarray(l["ids"], dtype=np.int32) for l in lines]
    offsets = np.zeros(len(ids_per_line) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in ids_per_line], out=offsets[1:])
    stream = (np.concatenate(ids_per_line) if ids_per_line
              else np.zeros(0, dtype=np.int32))
    n = len(stream)
    counters["stream_tokens"] = int(n)

    line_pos = {l["line_idx"]: i for i, l in enumerate(lines)}
    for inst in instances:
        li = line_pos[inst["line_idx"]]
        inst["pos"] = int(offsets[li]) + inst["piece_in_line"]

    flat_inv = [i for grp in inv_ids for i in grp]
    inv_tensor = torch.tensor(flat_inv, dtype=torch.long, device=device)
    cls_id, sep_id, mask_id = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id
    assert None not in (cls_id, sep_id, mask_id), "MLM special tokens missing"

    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(
            hf_id, attn_implementation="eager").to(device).eval()

    t0 = time.time()
    N = len(instances)
    logprob_first = np.full(N, np.nan, dtype=np.float32)
    logprob_sum = np.full(N, np.nan, dtype=np.float32)
    ent = np.full(N, np.nan, dtype=np.float32)
    invp = np.full((N, len(flat_inv)), np.nan, dtype=np.float32)

    order = sorted(range(N), key=lambda i: instances[i]["pos"])
    with torch.inference_mode():
        for b0 in range(0, N, batch_size):
            idxs = order[b0:b0 + batch_size]
            seqs, mask_slots, true_pieces = [], [], []
            for i in idxs:
                inst = instances[i]
                p, np_ = inst["pos"], inst["n_pieces"]
                # Fit within the encoder's 512-position budget: CLS + SEP +
                # pieces + context. A pathological many-piece instance with
                # the full +-ctx_halfwidth blew past 512 (observed 513 ->
                # RuntimeError in BERT position expansion, 2026-08-23).
                budget = 512 - 2 - np_
                if ctx_right in ("V", "VX"):
                    # Verb-frame planning window: forward context ends at
                    # the dependency head — through its last piece ("V")
                    # or just before its first ("VX"). Head-before-pronoun
                    # (inversion/questions) => no forward window.
                    hp = inst.get("head_piece_in_line")
                    if hp is None:
                        cr = 2  # head unaligned; next-word-ish fallback
                        counters["dyn_head_fallback"] = counters.get(
                            "dyn_head_fallback", 0) + 1
                    else:
                        head_start = p - inst["piece_in_line"] + hp
                        head_end = head_start + (inst["head_n_pieces"] or 1)
                        tgt = head_end if ctx_right == "V" else head_start
                        cr = max(0, min(tgt - (p + np_), budget))
                else:
                    cr = ctx_halfwidth if ctx_right is None else ctx_right
                if cr + ctx_halfwidth > budget:
                    scale = budget / max(cr + ctx_halfwidth, 1)
                    cl, cr = int(ctx_halfwidth * scale), int(cr * scale)
                else:
                    cl = ctx_halfwidth
                lo = max(0, p - cl)
                hi = min(n, p + np_ + cr)
                ids = stream[lo:hi].astype(np.int64).copy()
                m0 = p - lo
                truth = ids[m0:m0 + np_].tolist()
                ids[m0:m0 + np_] = mask_id
                seqs.append([cls_id] + ids.tolist() + [sep_id])
                mask_slots.append(m0 + 1)  # +1 for CLS
                true_pieces.append(truth)
            maxlen = max(len(s) for s in seqs)
            batch = torch.full((len(seqs), maxlen), tok.pad_token_id or 0,
                               dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, s in enumerate(seqs):
                batch[r, :len(s)] = torch.tensor(s)
                attn[r, :len(s)] = 1
            batch, attn = batch.to(device), attn.to(device)
            ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if use_amp else torch.no_grad())
            with ctx:
                out = model(input_ids=batch, attention_mask=attn)
            lsm = torch.log_softmax(out.logits.float(), dim=-1)
            for r, i in enumerate(idxs):
                m0 = mask_slots[r]
                truth = true_pieces[r]
                row0 = lsm[r, m0]
                logprob_first[i] = float(row0[truth[0]])
                logprob_sum[i] = float(sum(
                    lsm[r, m0 + j, t] for j, t in enumerate(truth)))
                probs0 = row0.exp()
                ent[i] = float(-(probs0 * row0).sum())
                invp[i] = probs0[inv_tensor].cpu().numpy()
            if (b0 // batch_size) % 200 == 0:
                print(f"  mlm {b0}/{N} ({time.time()-t0:.0f}s)", flush=True)
    counters[f"seconds_{mlm_name}"] = round(time.time() - t0, 1)
    return {mlm_name: {"logprob_first_arr": logprob_first,
                       "logprob_sum_arr": logprob_sum,
                       "entropy": ent, "inv": invp}}, offsets


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
        if isinstance(ckpt, tuple) and ckpt[0] == "hf":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                ckpt[1], attn_implementation="eager").to(device).eval()
        else:
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
    ap.add_argument("--scorer", action="append", default=None,
                    metavar="NAME=CKPT_ROOT",
                    help="repeatable; CKPT_ROOT holds checkpoint-*/ dirs")
    ap.add_argument("--hf-model", metavar="NAME=HF_ID", default=None,
                    help="score with an EXTERNAL pretrained HF model using "
                         "its own tokenizer (offset alignment); outputs go "
                         "to <corpus>/external_<NAME>/")
    ap.add_argument("--hf-mlm", metavar="NAME=HF_ID", default=None,
                    help="score with an EXTERNAL pretrained MLM (BERT-family):"
                         " masked-slot scoring, no per-token stream output;"
                         " outputs go to <corpus>/external_<NAME>/")
    ap.add_argument("--mlm-ctx", type=int, default=250,
                    help="stream tokens of context on EACH side of the slot")
    ap.add_argument("--mlm-ctx-right", type=str, default=None,
                    help="override RIGHT-side context (0 = left-only "
                         "prediction; default: same as --mlm-ctx). Added "
                         "2026-08-24: bidirectional masked recovery of "
                         "subject pronouns saturates (~57%% of instances "
                         "<0.1 nats), collapsing the ranking; left-only "
                         "converts cloze to prediction.")
    ap.add_argument("--mlm-batch", type=int, default=48)
    ap.add_argument("--mlm-ctx-grid", default=None,
                    help="MLM locality-experiment grid: comma-separated "
                         "L:R context configs (stream wordpieces), e.g. "
                         "'250:250,64:0,0:16'. One scoring pass per config "
                         "over the same instances; outputs under "
                         "external_<NAME>/grid/L{l}R{r}/.")
    ap.add_argument("--sample-file", type=Path, default=None,
                    help="parquet with (line_idx, token_i) — restrict "
                         "scoring to these instances (frozen sample)")
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

    if sum(bool(x) for x in (args.scorer, args.hf_model, args.hf_mlm)) != 1:
        ap.error("exactly one of --scorer / --hf-model / --hf-mlm")

    corpus_dir = f"{args.data_root}/raw/en/{args.corpus}/"
    cache_key = compute_annotation_cache_key(
        corpus_dir, args.spacy_model, DOCBIN_ATTRS)
    annotated_dir = args.data_root / "annotated" / cache_key

    hf_name = hf_id = None
    hf_tok = None
    if args.hf_model or args.hf_mlm:
        hf_name, hf_id = (args.hf_model or args.hf_mlm).split("=", 1)
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
        aligner = HFAligner(hf_tok)
        out_root = ((args.out_dir or (args.data_root / "recoverability"))
                    / args.corpus / f"external_{hf_name}")
    else:
        tok_dir = args.tokenizer_dir or (args.data_root / "tokenizers" / "en_shared_unigram")
        sp = spm.SentencePieceProcessor()
        sp.load(str(tok_dir / "tokenizer.model"))
        aligner = SPAligner(sp)
        out_root = (args.out_dir or (args.data_root / "recoverability")) / args.corpus
    for sub in ("instances", "per_token", "lines", "manifest"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # Inventory ids per form: marker + bare, lowercase + capitalized.
    inv_ids: List[List[int]] = []
    inv_meta = {}
    for form in INVENTORY_FORMS:
        grp = aligner.inventory_ids(form)
        inv_ids.append(grp)
        inv_meta[form] = grp
    counters: Dict[str, int] = {k: 0 for k in (
        "lines", "passthrough_lines", "doc_text_mismatch",
        "instances_seen", "instances_aligned", "instances_unaligned",
        "instances_midword_recovered")}

    scorers = {}
    if args.hf_model or args.hf_mlm:
        scorers[hf_name] = ("hf", hf_id)
        print(f"scorer {hf_name}: hf:{hf_id}"
              + (" (MLM masked-slot)" if args.hf_mlm else ""), flush=True)
    else:
        for spec in args.scorer:
            name, root = spec.split("=", 1)
            scorers[name] = resolve_final_checkpoint(Path(root))
            print(f"scorer {name}: {scorers[name]}", flush=True)

    sample = None
    if args.sample_file is not None:
        import pyarrow.parquet as _pq
        st = _pq.read_table(args.sample_file,
                            columns=["line_idx", "token_i"])
        sample = set(zip(st.column("line_idx").to_pylist(),
                         st.column("token_i").to_pylist()))
        print(f"sample filter: {len(sample):,} instances", flush=True)

    print(f"=== phase A: {args.corpus}/{args.file} from {annotated_dir}", flush=True)
    t0 = time.time()
    lines, instances = phase_a(annotated_dir, args.file, aligner,
                               args.limit_lines, counters, sample=sample)
    counters["seconds_phase_a"] = round(time.time() - t0, 1)
    print(f"  {counters}", flush=True)

    if args.hf_mlm and args.mlm_ctx_grid:
        # Locality experiment: score the SAME instances under every (L, R)
        # context config, one model load. Per-config outputs only.
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transformers import AutoModelForMaskedLM
        configs = []
        for spec in args.mlm_ctx_grid.split(","):
            l_s, r_s = spec.strip().split(":")
            # "V" = dynamic forward window through the dependency-head
            # verb's last piece (verb-frame planning hypothesis);
            # "VX" = up to but excluding the head's first piece.
            configs.append((int(l_s),
                            r_s if r_s in ("V", "VX") else int(r_s)))
        model = AutoModelForMaskedLM.from_pretrained(
            hf_id, attn_implementation="eager").to(args.device).eval()
        print(f"=== phase B (grid): {len(configs)} configs × "
              f"{len(instances):,} instances", flush=True)
        for l_ctx, r_ctx in configs:
            cfg_counters: Dict[str, int] = {}
            res, _ = phase_b_mlm(
                lines, instances, hf_name, hf_id, hf_tok, inv_ids,
                args.device, l_ctx, args.mlm_batch, args.amp, cfg_counters,
                ctx_right=r_ctx, model=model)
            r = res[hf_name]
            gdir = out_root / "grid" / f"L{l_ctx}R{r_ctx}"
            gdir.mkdir(parents=True, exist_ok=True)
            def _gap(i):
                hp = i.get("head_piece_in_line")
                if hp is None:
                    return None
                return hp - (i["piece_in_line"] + i["n_pieces"])

            tbl = {
                "line_idx": [i["line_idx"] for i in instances],
                "token_i": [i["token_i"] for i in instances],
                "form": [i["form"] for i in instances],
                "person": [i["person"] for i in instances],
                "head_gap": [_gap(i) for i in instances],
                "head_n_pieces": [i.get("head_n_pieces") for i in instances],
                "logprob_sum": [None if math.isnan(float(x)) else float(x)
                                for x in r["logprob_sum_arr"]],
                "logprob_first": [None if math.isnan(float(x)) else float(x)
                                  for x in r["logprob_first_arr"]],
                "entropy": [None if math.isnan(float(x)) else float(x)
                            for x in r["entropy"]],
            }
            pq.write_table(pa.table(tbl), gdir / f"{args.file}.parquet")
            with open(gdir / f"{args.file}.manifest.json", "w") as f:
                json.dump({"L": l_ctx, "R": r_ctx, "file": args.file,
                           "n": len(instances), "hf_id": hf_id,
                           "counters": cfg_counters}, f)
            print(f"  grid L{l_ctx}R{r_ctx}: done "
                  f"({cfg_counters.get(f'seconds_{hf_name}', '?')}s)",
                  flush=True)
        print("GRID SCORING OK", flush=True)
        return

    print("=== phase B: scoring", flush=True)
    if args.hf_mlm:
        cr = args.mlm_ctx_right
        if cr is not None and cr not in ("V", "VX"):
            cr = int(cr)
        results, offsets = phase_b_mlm(
            lines, instances, hf_name, hf_id, hf_tok, inv_ids, args.device,
            args.mlm_ctx, args.mlm_batch, args.amp, counters,
            ctx_right=cr)
    else:
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
        "piece_ids": [i["piece_ids"] for i in instances],
        "is_title": [i["is_title"] for i in instances],
        "word_initial": [i["word_initial"] for i in instances],
        "form": [i["form"] for i in instances],
        "lemma": [i["lemma"] for i in instances],
        "dep": [i["dep"] for i in instances],
        "person": [i["person"] for i in instances],
        "number": [i["number"] for i in instances],
        "in_lexicon": [i["in_lexicon"] for i in instances],
        "head_i": [i["head_i"] for i in instances],
    }
    for name, res in results.items():
        if "logprob_first_arr" in res:  # MLM mode: per-instance arrays
            inst_cols[f"{name}__logprob_sum"] = [
                None if math.isnan(float(x)) else float(x)
                for x in res["logprob_sum_arr"]]
            inst_cols[f"{name}__logprob_first"] = [
                None if math.isnan(float(x)) else float(x)
                for x in res["logprob_first_arr"]]
        else:
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
        if "logprobs" not in res:
            continue  # MLM mode banks no per-token stream
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
        "tokenizer": f"hf:{hf_id}" if (args.hf_model or args.hf_mlm) else str(tok_dir),
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
