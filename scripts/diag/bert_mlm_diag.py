#!/usr/bin/env python3
"""Diagnose the bert_large MLM plateau (2026-10-19).

Production bert_large finished 30 epochs at train loss 5.62 and the long
rater probe sat at ~6.1 for 20k steps: neither left the unigram-like
plateau. This separates a pipeline bug from an optimization problem:

  1. tokenizer + data sanity (special ids, [UNK] rate, decoded text)
  2. the final production checkpoint: masked loss/accuracy with our
     collator and HF's, prediction diversity vs a unigram baseline
  3. short from-scratch races on the same data: small/base/large BERT,
     our MaskedLMDataCollator vs HF DataCollatorForLanguageModeling

Run in a GPU pod with the PVC at /mnt/data and the repo at /opt/repo.
"""
from __future__ import annotations

import collections
import glob
import math
import os
import random
import sys
import time

import torch
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

sys.path.insert(0, "/opt/repo")
from model_foundry.data_collators import MaskedLMDataCollator  # noqa: E402
from model_foundry.training.tokenization import load_tokenizer  # noqa: E402

TOK_DIR = "/mnt/data/tokenizers/en_bert_wordpiece/"
SUBSET = "/mnt/data/rater_corpus/sweep_subset_v1/base/"
PROD_RUN = "/mnt/data/models/production/bert_large-en-baseline-h0-s137"
PROBE_ROOT = "/mnt/data/models/rater_sweeps/base"
SEQ = 512
DEV = "cuda"


def say(*a):
    print(*a, flush=True)


def read_text(path: str, max_bytes: int) -> list[str]:
    with open(path, "rb") as f:
        raw = f.read(max_bytes).decode("utf-8", "ignore")
    return [ln for ln in raw.splitlines() if ln.strip()]


def chunks_from(tok, lines: list[str], n_max: int) -> list[list[int]]:
    ids: list[int] = []
    for i in range(0, len(lines), 2000):
        for enc in tok(lines[i:i + 2000], add_special_tokens=False)["input_ids"]:
            ids.extend(enc)
        if len(ids) >= n_max * SEQ:
            break
    return [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ + 1, SEQ)][:n_max]


def part1(tok, train_lines):
    say("== PART 1: tokenizer + data")
    say(f"len(tok)={len(tok)} vocab_size={tok.vocab_size} class={type(tok).__name__}")
    for name in ("pad", "unk", "cls", "sep", "mask", "bos", "eos"):
        say(f"  {name}_token={getattr(tok, name + '_token', None)!r} id={getattr(tok, name + '_token_id', None)}")
    say(f"  id 0 -> {tok.convert_ids_to_tokens(0)!r}; ids 0-9 -> {tok.convert_ids_to_tokens(list(range(10)))}")
    sample = train_lines[:3]
    for s in sample:
        enc = tok(s, add_special_tokens=False)["input_ids"]
        say(f"  text: {s[:160]!r}")
        say(f"  toks: {tok.convert_ids_to_tokens(enc[:40])}")
    enc = [i for e in tok(train_lines[:5000], add_special_tokens=False)["input_ids"] for i in e]
    unk = sum(1 for i in enc if i == tok.unk_token_id) / max(1, len(enc))
    say(f"  [UNK] rate over {len(enc):,} tokens: {unk:.4f}; tokens/line {len(enc) / 5000:.1f}")
    c = collections.Counter(enc)
    say("  top tokens:", [(tok.convert_ids_to_tokens(i), round(n / len(enc), 3)) for i, n in c.most_common(12)])


def unigram_xent(train_chunks, held_chunks, vocab):
    c = torch.zeros(vocab, dtype=torch.float64)
    for ch in train_chunks:
        c += torch.bincount(torch.tensor(ch), minlength=vocab).double()
    p = (c + 0.1) / (c + 0.1).sum()
    held = torch.tensor([i for ch in held_chunks for i in ch])
    return float(-(p[held].log()).mean())


@torch.no_grad()
def masked_eval(model, collator, chunks, tok, seed=0, bs=8):
    torch.manual_seed(seed)
    model.eval()
    tot, n, correct = 0.0, 0, 0
    preds = collections.Counter()
    for i in range(0, len(chunks), bs):
        b = collator([{"input_ids": list(c)} for c in chunks[i:i + bs]])
        b = {k: v.to(DEV) for k, v in b.items() if k in ("input_ids", "attention_mask", "labels")}
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**b)
        m = b["labels"] != -100
        logits = out.logits.float()[m]
        tgt = b["labels"][m]
        tot += float(torch.nn.functional.cross_entropy(logits, tgt, reduction="sum"))
        n += int(m.sum())
        am = logits.argmax(-1)
        correct += int((am == tgt).sum())
        preds.update(am.tolist())
    model.train()
    top = [(tok.convert_ids_to_tokens(i), round(k / n, 3)) for i, k in preds.most_common(8)]
    return tot / n, correct / n, len(preds), top


@torch.no_grad()
def probe_sentences(model, tok):
    model.eval()
    for s in ("the cat sat on the [MASK] .", "thank you very [MASK] .",
              "she opened the [MASK] and walked in .", "[MASK] went to the store yesterday .",
              "i don't [MASK] what you mean ."):
        enc = tok(s, return_tensors="pt", add_special_tokens=True).to(DEV)
        logits = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits[0]
        pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
        top = logits[pos].float().softmax(-1).topk(5)
        say(f"    {s!r} -> {[(tok.convert_ids_to_tokens(int(i)), round(float(p), 3)) for p, i in zip(top.values, top.indices)]}")
    model.train()


def latest_ckpt(root: str) -> str | None:
    """Highest checkpoint-N under root; without step numbers, the newest dir."""
    best = None
    for cfg in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        d = os.path.dirname(cfg)
        tail = os.path.basename(d).rsplit("-", 1)[-1]
        key = (int(tail) if tail.isdigit() else -1, os.path.getmtime(cfg), d)
        if best is None or key > best:
            best = key
    return best[2] if best else None


def part2(tok, held, ours, hf):
    say("== PART 2: trained checkpoints")
    for label, root in (("production s137 final", PROD_RUN), ("rater probe-long final", PROBE_ROOT)):
        ck = latest_ckpt(root)
        if not ck:
            say(f"  {label}: no checkpoint under {root}")
            continue
        say(f"  {label}: {ck}")
        model = BertForMaskedLM.from_pretrained(ck).to(DEV)
        for cname, col in (("ours", ours), ("hf", hf)):
            loss, acc, nd, top = masked_eval(model, col, held, tok)
            say(f"    [{cname}] masked loss {loss:.3f} (ppl {math.exp(loss):.0f}) acc {acc:.3f} distinct argmax {nd} top {top}")
        probe_sentences(model, tok)
        del model
        torch.cuda.empty_cache()


def race(name, tok, col, train, held, layers, hidden, heads, ffn, lr, steps, bs):
    torch.manual_seed(0)
    random.seed(0)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=hidden, num_hidden_layers=layers,
                     num_attention_heads=heads, intermediate_size=ffn, max_position_embeddings=SEQ,
                     pad_token_id=tok.pad_token_id or 0, attn_implementation="sdpa")
    model = BertForMaskedLM(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    scaler = torch.cuda.amp.GradScaler()
    t0, recent, marks = time.time(), [], []
    for step in range(1, steps + 1):
        b = col([{"input_ids": list(train[i])} for i in random.sample(range(len(train)), bs)])
        b = {k: v.to(DEV) for k, v in b.items() if k in ("input_ids", "attention_mask", "labels")}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**b).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        recent.append(float(loss))
        if step % 250 == 0:
            marks.append(f"{step}:{sum(recent[-100:]) / len(recent[-100:]):.2f}")
    hl, acc, nd, _ = masked_eval(model, col, held, tok)
    say(f"  {name}: train {' '.join(marks)} | held-out {hl:.3f} acc {acc:.3f} distinct {nd} | {time.time() - t0:.0f}s")
    del model, opt
    torch.cuda.empty_cache()


def main():
    tok = load_tokenizer(TOK_DIR)
    shards = sorted(glob.glob(os.path.join(SUBSET, "*.train")))
    say("subset shards:", len(shards), shards[:2])
    train_lines = read_text(shards[0], 40_000_000)
    held_lines = read_text(os.path.join(SUBSET, "heldout.test"), 4_000_000)
    part1(tok, train_lines)
    train = chunks_from(tok, train_lines, 12000)
    held = chunks_from(tok, held_lines, 160)
    say(f"chunks: train {len(train)} held {len(held)} x {SEQ}")
    say(f"unigram cross-entropy on held-out tokens: {unigram_xent(train, held, len(tok)):.3f} nats (the plateau reference)")
    ours = MaskedLMDataCollator(tokenizer=tok, mlm_probability=0.15, pad_to_multiple_of=8)
    hf = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)
    b = ours([{"input_ids": list(c)} for c in train[:8]])
    m = b["labels"] != -100
    say(f"our collator: masked frac {float(m.float().mean()):.3f}; [MASK] frac of masked "
        f"{float((b['input_ids'][m] == tok.mask_token_id).float().mean()):.3f}; labels==orig at masked "
        f"{float((b['labels'][m] == torch.tensor(train[:8])[m]).float().mean()):.3f}")
    part2(tok, held, ours, hf)
    say("== PART 3: from-scratch races (same data, same seeds)")
    race("small 4L/256  lr5e-4 ours", tok, ours, train, held, 4, 256, 4, 1024, 5e-4, 2000, 32)
    race("small 4L/256  lr5e-4 hf  ", tok, hf, train, held, 4, 256, 4, 1024, 5e-4, 2000, 32)
    race("base 12L/768  lr1e-4 ours", tok, ours, train, held, 12, 768, 12, 3072, 1e-4, 2000, 32)
    race("base 12L/768  lr3e-4 ours", tok, ours, train, held, 12, 768, 12, 3072, 3e-4, 2000, 32)
    race("large 24L/1024 lr1e-4 ours", tok, ours, train, held, 24, 1024, 16, 4096, 1e-4, 1500, 16)
    say("DIAG DONE")


if __name__ == "__main__":
    main()
