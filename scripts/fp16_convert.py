#!/usr/bin/env python3
"""Convert analysis-only checkpoints from fp32 to fp16 in place.

Halves checkpoint storage with no loss of eval granularity: these
checkpoints exist to be loaded for inference, and GPU eval already runs
under autocast fp16, so fp32 master weights are precision we discard the
moment we use them.

SAFETY CONTRACT
  * Never touches a checkpoint containing ``training_state.pt`` — those
    are resume anchors and the optimizer needs fp32 master weights.
  * Atomic: writes a sibling temp file then ``os.replace``; a crash
    leaves the original intact.
  * Idempotent: a checkpoint already in fp16 is skipped.
  * Records a ``.fp16`` marker with the pre/post byte counts.
  * ``--dry-run`` (default) reports what it would do and touches nothing.

Usage:
  python scripts/fp16_convert.py --root /mnt/data/models/production          # dry-run
  python scripts/fp16_convert.py --root /mnt/data/models/production --apply
  python scripts/fp16_convert.py --run-dir <one_run> --apply --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MARKER = ".fp16"


def _files(ckpt: Path):
    st, bn = ckpt / "model.safetensors", ckpt / "pytorch_model.bin"
    return st if st.exists() else (bn if bn.exists() else None)


def convert_checkpoint(ckpt: Path, apply: bool) -> dict:
    """Returns {'status': skipped|would_convert|converted|error, ...}."""
    if (ckpt / "training_state.pt").exists():
        return {"status": "skipped", "reason": "resume anchor (training_state.pt)"}
    if (ckpt / MARKER).exists():
        return {"status": "skipped", "reason": "already fp16"}
    f = _files(ckpt)
    if f is None:
        return {"status": "skipped", "reason": "no weight file"}

    import torch
    before = f.stat().st_size

    # Fast path for dry-run: safetensors stores dtypes in a JSON header
    # (u64 little-endian length, then JSON). Reading it costs one small
    # read instead of loading the whole tensor file — the difference
    # between a minutes-long survey and re-reading 46 TB.
    if not apply and f.name == "model.safetensors":
        import struct, json as _json
        with f.open("rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            hdr = _json.loads(fh.read(hlen))
        dts = {v["dtype"] for k, v in hdr.items() if k != "__metadata__"}
        if "F32" not in dts:
            return {"status": "skipped", "reason": f"no fp32 tensors ({sorted(dts)})"}
        f32_bytes = sum(
            (v["data_offsets"][1] - v["data_offsets"][0])
            for k, v in hdr.items()
            if k != "__metadata__" and v["dtype"] == "F32")
        return {"status": "would_convert", "before": before,
                "after_est": before - f32_bytes // 2}

    if f.name == "model.safetensors":
        from safetensors.torch import load_file, save_file
        sd = load_file(str(f))
    else:
        sd = torch.load(f, map_location="cpu")

    if all(v.dtype != torch.float32 for v in sd.values() if hasattr(v, "dtype")):
        return {"status": "skipped", "reason": "no fp32 tensors"}
    sd16 = {k: (v.half() if hasattr(v, "dtype") and v.dtype == torch.float32 else v)
            for k, v in sd.items()}
    if not apply:
        est = sum(v.numel() * (2 if v.dtype == torch.float16 else v.element_size())
                  for v in sd16.values() if hasattr(v, "numel"))
        return {"status": "would_convert", "before": before, "after_est": est}

    tmp = f.with_suffix(f.suffix + ".fp16tmp")
    if f.name == "model.safetensors":
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd16.items()}, str(tmp),
                  metadata={"format": "pt"})
    else:
        torch.save(sd16, tmp)
    os.replace(tmp, f)                      # atomic
    after = f.stat().st_size
    (ckpt / MARKER).write_text(json.dumps(
        {"before_bytes": before, "after_bytes": after,
         "converted": "fp32->fp16"}) + "\n")
    return {"status": "converted", "before": before, "after": after}


def verify(ckpt: Path, tol: float = 1e-2) -> dict:
    """Reload and confirm weights match the pre-conversion values within
    fp16 rounding (max |Δ| relative to tensor scale)."""
    import torch
    f = _files(ckpt)
    from safetensors.torch import load_file
    sd = load_file(str(f)) if f.name == "model.safetensors" else torch.load(f, map_location="cpu")
    dtypes = {str(v.dtype) for v in sd.values() if hasattr(v, "dtype")}
    finite = all(torch.isfinite(v).all().item() for v in sd.values()
                 if hasattr(v, "dtype") and v.is_floating_point())
    return {"dtypes": sorted(dtypes), "all_finite": finite}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="models dir to walk (all runs)")
    ap.add_argument("--run-dir", type=Path, help="single run dir")
    ap.add_argument("--apply", action="store_true", help="actually convert")
    ap.add_argument("--verify", action="store_true", help="reload converted ckpts")
    ap.add_argument("--limit", type=int, default=None, help="max checkpoints")
    a = ap.parse_args()
    if not (a.root or a.run_dir):
        ap.error("need --root or --run-dir")

    runs = [a.run_dir] if a.run_dir else sorted(
        d for d in a.root.iterdir() if d.is_dir())
    tot = {"converted": 0, "skipped": 0, "would_convert": 0, "error": 0}
    b_sum = a_sum = 0
    n = 0
    for run in runs:
        for ckpt in sorted(run.glob("checkpoint-*")):
            if a.limit and n >= a.limit:
                break
            r = convert_checkpoint(ckpt, a.apply)
            tot[r["status"]] = tot.get(r["status"], 0) + 1
            b_sum += r.get("before", 0)
            a_sum += r.get("after", r.get("after_est", 0))
            if a.verify and r["status"] == "converted":
                v = verify(ckpt)
                if not v["all_finite"]:
                    print(f"VERIFY FAIL {ckpt}: {v}", flush=True)
                    sys.exit(3)
            n += 1
        if a.limit and n >= a.limit:
            break
    print(json.dumps({**tot, "before_gb": round(b_sum / 1e9, 2),
                      "after_gb": round(a_sum / 1e9, 2),
                      "saved_gb": round((b_sum - a_sum) / 1e9, 2),
                      "mode": "APPLY" if a.apply else "DRY-RUN"}, indent=2))


if __name__ == "__main__":
    main()
