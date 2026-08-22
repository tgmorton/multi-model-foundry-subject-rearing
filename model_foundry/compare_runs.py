"""Run-pair comparator (lang-manifold port): bitwise-or-epsilon verdicts.

Compares two run output dirs' endpoint checkpoints (and optionally their
matching intermediate steps): first content-hash equality of the weight
files (the bitwise claim), else per-tensor max abs/rel deviation against
tolerances. A tolerance-only pass is labeled ``epsilon`` — never silently
conflated with bitwise; a failure is ``stop_and_review``.

Used by the preemption/resume smoke and any wave sentinel. CLI:

    python -m model_foundry.compare_runs RUN_DIR_A RUN_DIR_B \
        [--step N] [--atol 1e-5] [--rtol 1e-5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def _endpoint(run_dir: Path, step=None) -> Path:
    ckpts = {int(m.group(1)): d for d in run_dir.iterdir()
             if d.is_dir() and (m := _CKPT_RE.search(d.name))}
    if not ckpts:
        sys.exit(f"no checkpoints under {run_dir}")
    key = step if step is not None else max(ckpts)
    if key not in ckpts:
        sys.exit(f"step {key} not present under {run_dir} (has {sorted(ckpts)})")
    return ckpts[key]


def _weights(ckpt: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (ckpt / name).exists():
            return ckpt / name
    sys.exit(f"no weights file in {ckpt}")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(ckpt_a: Path, ckpt_b: Path, atol: float, rtol: float) -> dict:
    wa, wb = _weights(ckpt_a), _weights(ckpt_b)
    ha, hb = _sha256(wa), _sha256(wb)
    out = {"ckpt_a": str(ckpt_a), "ckpt_b": str(ckpt_b),
           "sha_a": ha, "sha_b": hb, "bitwise": ha == hb,
           "atol": atol, "rtol": rtol}
    if out["bitwise"]:
        out.update(verdict="bitwise", max_abs=0.0, max_rel=0.0)
        return out

    import torch
    if wa.suffix == ".safetensors":
        from safetensors.torch import load_file
        sa, sb = load_file(str(wa)), load_file(str(wb))
    else:
        sa = torch.load(wa, map_location="cpu", weights_only=True)
        sb = torch.load(wb, map_location="cpu", weights_only=True)
    if set(sa) != set(sb):
        out.update(verdict="stop_and_review",
                   reason=f"key mismatch: {sorted(set(sa) ^ set(sb))[:5]}")
        return out
    max_abs = max_rel = 0.0
    worst = None
    for k in sa:
        a, b = sa[k].float(), sb[k].float()
        if a.shape != b.shape:
            out.update(verdict="stop_and_review", reason=f"shape mismatch {k}")
            return out
        d = (a - b).abs()
        ma = float(d.max()) if d.numel() else 0.0
        mr = float((d / a.abs().clamp_min(1e-12)).max()) if d.numel() else 0.0
        if ma > max_abs:
            max_abs, worst = ma, k
        max_rel = max(max_rel, mr)
    within = max_abs <= atol and max_rel <= rtol
    out.update(verdict="epsilon" if within else "stop_and_review",
               max_abs=max_abs, max_rel=max_rel, worst_tensor=worst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()
    res = compare(_endpoint(args.run_a, args.step),
                  _endpoint(args.run_b, args.step), args.atol, args.rtol)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["verdict"] in ("bitwise", "epsilon") else 1)


if __name__ == "__main__":
    main()
