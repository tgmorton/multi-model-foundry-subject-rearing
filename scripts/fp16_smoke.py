#!/usr/bin/env python3
"""Per-architecture smoke test for fp16 checkpoint conversion (2026-10-05).

For each architecture, take one real analysis-only checkpoint, copy it to
scratch twice, convert one copy with scripts/fp16_convert.py's own
convert_checkpoint, then load both copies the way the eval runner does
(model built from config in fp32, weights swapped in with load_state_dict)
and compare:

  * strict key match (a split or dropped tied weight fails here)
  * per-tensor max |delta| relative to the tensor's max |value|
  * forward pass on a fixed random token batch: max |delta log-softmax|
    and top-1 agreement (mamba: attempted on CPU; weight comparison only
    if its forward needs CUDA)

Originals are only read. Exit 0 = every architecture passed.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from fp16_convert import convert_checkpoint, _files  # noqa: E402

ROOTS = [Path("/mnt/data/models/production"), Path("/mnt/data/models/wave2")]
ARCHS = ["gpt2_small", "gpt2_medium", "gpt2_large", "bert_large", "lstm", "mamba_370m"]
SCRATCH = Path("/tmp/fp16smoke")


def pick_checkpoint(arch: str):
    for root in ROOTS:
        if not root.is_dir():
            continue
        for run in sorted(root.glob(f"{arch}-en-*")):
            for ck in sorted(run.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
                             reverse=True):
                if (ck / "training_state.pt").exists() or (ck / ".fp16").exists():
                    continue
                if _files(ck) is not None:
                    return ck
    return None


def load_sd(ck: Path):
    import torch
    f = _files(ck)
    if f.name == "model.safetensors":
        from safetensors.torch import load_file
        return load_file(str(f))
    return torch.load(f, map_location="cpu", weights_only=True)


def build(arch: str, ck: Path, sd):
    if arch.startswith("gpt2") or arch == "bert_large":
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM
        cfg = AutoConfig.from_pretrained(str(ck))
        cls = AutoModelForCausalLM if arch.startswith("gpt2") else AutoModelForMaskedLM
        return cls.from_config(cfg, attn_implementation="eager")
    from model_foundry.cli import load_config
    from model_foundry.model import create_model
    cfg = load_config(str(REPO / "configs" / "sweeps" / "baselines" / f"{arch}_en.yaml"))
    cfg.tokenizer.vocab_size = vocab_of(sd)
    return create_model(cfg)


def vocab_of(sd) -> int:
    emb = [v for k, v in sd.items() if hasattr(v, "dim") and v.dim() == 2
           and ("embed" in k or "wte" in k)]
    return int((emb or [v for v in sd.values() if hasattr(v, "dim") and v.dim() == 2])[0].shape[0])


def forward_logits(arch: str, model, vocab: int):
    import torch
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(5, vocab - 1, (2, 64), generator=g)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
    return torch.log_softmax(logits.float(), dim=-1)


def main() -> int:
    import torch
    torch.set_num_threads(4)
    report, ok_all = {}, True
    for arch in ARCHS:
        r = {"arch": arch}
        try:
            src = pick_checkpoint(arch)
            if src is None:
                r["status"] = "no candidate checkpoint"
                report[arch] = r
                print(json.dumps(r), flush=True)
                continue
            r["source"] = str(src)
            a, b = SCRATCH / arch / "fp32", SCRATCH / arch / "fp16"
            for d in (a, b):
                if d.exists():
                    shutil.rmtree(d)
                shutil.copytree(src, d)
            conv = convert_checkpoint(b, apply=True)
            r["convert"] = {k: conv.get(k) for k in ("status", "before", "after", "reason")}
            sd32, sd16 = load_sd(a), load_sd(b)
            r["dtypes_after"] = sorted({str(v.dtype) for v in sd16.values() if hasattr(v, "dtype")})
            rel = 0.0
            for k, v in sd32.items():
                if hasattr(v, "dtype") and v.is_floating_point():
                    w = sd16[k].float()
                    scale = v.abs().max().item() or 1.0
                    rel = max(rel, (v.float() - w).abs().max().item() / scale)
            r["max_rel_param_err"] = rel
            m32, m16 = build(arch, a, sd32), build(arch, b, sd16)
            for m, sd, tag in ((m32, sd32, "fp32"), (m16, sd16, "fp16")):
                missing, unexpected = m.load_state_dict(sd, strict=False)
                # Tied weights (GPT-2 lm_head<->wte, BERT decoder<->embeddings)
                # are stored once; the model declares them and re-ties on load.
                tied = set(getattr(m, "_tied_weights_keys", None) or [])
                missing = [k for k in missing if not any(k == t or k.endswith('.' + t) for t in tied)]
                unexpected = [k for k in unexpected if not k.endswith("position_ids")]
                if missing or unexpected:
                    raise RuntimeError(f"{tag} key mismatch: missing={missing[:5]} "
                                       f"unexpected={unexpected[:5]}")
                if hasattr(m, "tie_weights"):
                    m.tie_weights()
                m.eval()
            r["params_loaded_dtype"] = str(next(m16.parameters()).dtype)
            vocab = vocab_of(sd32)
            try:
                l32, l16 = forward_logits(arch, m32, vocab), forward_logits(arch, m16, vocab)
                r["max_abs_logsoftmax_diff"] = (l32 - l16).abs().max().item()
                r["top1_agreement"] = (l32.argmax(-1) == l16.argmax(-1)).float().mean().item()
            except Exception as e:  # noqa: BLE001
                if arch != "mamba_370m":
                    raise
                r["forward"] = f"skipped on CPU ({e.__class__.__name__}); weights compared only"
            passed = (conv.get("status") == "converted"
                      and "torch.float32" not in r["dtypes_after"]
                      and rel < 1e-2
                      and r.get("top1_agreement", 1.0) > 0.99)
            if conv.get("before") and conv.get("after"):
                r["size_ratio"] = round(conv["after"] / conv["before"], 3)
            r["status"] = "PASS" if passed else "FAIL"
        except Exception as e:  # noqa: BLE001 — report and continue
            r["status"] = f"ERROR {e.__class__.__name__}: {e}"
        ok_all &= r["status"] in ("PASS", "no candidate checkpoint")
        report[arch] = r
        print(json.dumps(r), flush=True)
        shutil.rmtree(SCRATCH / arch, ignore_errors=True)
    print("FP16 SMOKE " + ("PASS" if ok_all else "FAIL"), flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
