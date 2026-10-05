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
  python scripts/fp16_convert.py --run-list runs.txt --roots /mnt/data/models/production /mnt/data/models/wave2 \
      --shard 3 --num-shards 24 --apply --verify    # wave mode: COMPLETE runs only, sharded
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

    # .bin (pickle) has no cheap header. For the survey, assume fp32 — the
    # June wave was trained/saved fp32 — and estimate from size; apply mode
    # still loads and checks dtypes for real before writing anything.
    if not apply and f.name == "pytorch_model.bin":
        return {"status": "would_convert", "before": before,
                "after_est": before // 2, "est": "size-based (.bin)"}

    # Low-memory paths (2026-10-05): the full fp32 dict plus its fp16 copy
    # peaked at 3.4Gi and forced 5Gi requests that sat at ~37% use (NRP
    # utilization webhook). safetensors streams one tensor at a time; .bin is
    # memory-mapped, so only the fp16 copy is resident either way.
    if f.name == "model.safetensors":
        from safetensors import safe_open
        sd16, any32 = {}, False
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                t = fh.get_tensor(k)
                if t.dtype == torch.float32:
                    any32 = True
                    t = t.half()
                sd16[k] = t
        if not any32:
            return {"status": "skipped", "reason": "no fp32 tensors"}
    else:
        sd = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        if all(v.dtype != torch.float32 for v in sd.values() if hasattr(v, "dtype")):
            return {"status": "skipped", "reason": "no fp32 tensors"}
        # Preserve storage sharing. Tied weights (GPT-2 wte<->lm_head, BERT's
        # MLM decoder<->embeddings) are ONE tensor under two keys; torch.save
        # stores it once. Halving each key independently splits the tie and
        # stores the embedding twice (+38.6M params on gpt2 — caught by the
        # .bin test). (safetensors files never hold shared tensors.)
        cache, sd16 = {}, {}
        for k, v in sd.items():
            if hasattr(v, "dtype") and v.dtype == torch.float32:
                ident = (v.untyped_storage().data_ptr(), v.storage_offset(),
                         tuple(v.shape), tuple(v.stride()))
                if ident not in cache:
                    cache[ident] = v.half()
                sd16[k] = cache[ident]
            else:
                sd16[k] = v
    for k, v in sd16.items():
        if hasattr(v, "is_floating_point") and v.is_floating_point() \
                and not torch.isfinite(v).all():
            return {"status": "error", "reason": f"non-finite after fp16 cast: {k}"}
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
    dtypes, finite = set(), True
    if f.name == "model.safetensors":                   # stream, low memory
        from safetensors import safe_open
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                v = fh.get_tensor(k)
                dtypes.add(str(v.dtype))
                if v.is_floating_point():
                    finite &= bool(torch.isfinite(v).all())
    else:
        sd = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        for v in sd.values():
            if hasattr(v, "dtype"):
                dtypes.add(str(v.dtype))
                if v.is_floating_point():
                    finite &= bool(torch.isfinite(v).all())
    return {"dtypes": sorted(dtypes), "all_finite": finite}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="models dir to walk (all runs)")
    ap.add_argument("--run-dir", type=Path, help="single run dir")
    ap.add_argument("--run-list", type=Path,
                    help="file of run_ids (one per line) to convert — e.g. the "
                         "registry's COMPLETE runs, so in-flight runs are never "
                         "touched; resolved against --roots")
    ap.add_argument("--roots", type=Path, nargs="+",
                    default=[Path("/mnt/data/models/production"),
                             Path("/mnt/data/models/wave2")])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--apply", action="store_true", help="actually convert")
    ap.add_argument("--verify", action="store_true", help="reload converted ckpts")
    ap.add_argument("--limit", type=int, default=None, help="max checkpoints")
    ap.add_argument("--sample-per-arch", type=int, default=None,
                    help="survey mode: only the first N runs of each arch "
                         "(run-name prefix before '-en-') — a census of 70K "
                         "tiny CephFS stats stalls; a sample doesn't")
    ap.add_argument("--progress-every", type=int, default=200,
                    help="print progress every N checkpoints (so slow is "
                         "distinguishable from hung)")
    a = ap.parse_args()
    if not (a.root or a.run_dir or a.run_list):
        ap.error("need --root, --run-dir or --run-list")

    if a.run_list:
        ids = sorted({l.strip() for l in a.run_list.read_text().splitlines()
                      if l.strip() and not l.startswith("#")})
        runs, missing = [], 0
        for rid in ids:
            hit = next((r / rid for r in a.roots if (r / rid).is_dir()), None)
            if hit is None:
                missing += 1
            else:
                runs.append(hit)
        print(f"run list: {len(ids)} ids, {len(runs)} found, {missing} without a run dir",
              flush=True)
    else:
        runs = [a.run_dir] if a.run_dir else sorted(
            d for d in a.root.iterdir() if d.is_dir())
    if a.num_shards > 1:
        runs = runs[a.shard::a.num_shards]
        print(f"shard {a.shard}/{a.num_shards}: {len(runs)} runs", flush=True)
    if a.sample_per_arch:
        by_arch, picked = {}, []
        for d in runs:
            arch = d.name.split("-en-")[0]
            if by_arch.get(arch, 0) < a.sample_per_arch:
                by_arch[arch] = by_arch.get(arch, 0) + 1
                picked.append(d)
        runs = picked
        print(f"sampling {len(runs)} runs: {by_arch}", flush=True)
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
            if n % a.progress_every == 0:
                print(f"  progress: {n} checkpoints, {dict(tot)}", flush=True)
        if a.limit and n >= a.limit:
            break
    print(json.dumps({**tot, "before_gb": round(b_sum / 1e9, 2),
                      "after_gb": round(a_sum / 1e9, 2),
                      "saved_gb": round((b_sum - a_sum) / 1e9, 2),
                      "mode": "APPLY" if a.apply else "DRY-RUN"}, indent=2))


if __name__ == "__main__":
    main()
