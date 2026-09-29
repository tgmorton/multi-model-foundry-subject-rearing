"""Post-compose verification for one selection label's 45 info cells.

Compares matrix_verification/<LABEL>_actual.json (written on-cluster by
k8s/job-matrix-verify-label.yaml) against the removal counts implied by
the label's own selection tables (selection_v5/<LABEL>/ on S3). The
non-expletive cells must match exactly with zero pool exhaustion.
rmexpl cells are allow-short by design, so only their deficits are
reported.

Exit 0 = pass; 1 = mismatch/violation (the arm chain halts on nonzero).
Writes <LABEL>_expected.json / <LABEL>_actual.json locally.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import boto3
import pandas as pd

B = "thomas-subject-drop-artifacts"
G = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki",
     "switchboard"]
V = Path("data/recoverability/analysis/matrix_verification")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--n-cells", type=int, default=45)
    a = ap.parse_args()
    s = boto3.Session(profile_name="nrp").client(
        "s3", endpoint_url="https://s3-west.nrp-nautilus.io")

    exp = {}
    for corpus in ("train_90M", "pull_10M"):
        df = pd.concat([pd.read_parquet(io.BytesIO(s.get_object(
            Bucket=B, Key=f"recoverability/analysis/selection_v5/{a.label}/"
                          f"{corpus}/{g}.parquet")["Body"].read()),
            columns=["info_decile"]) for g in G], ignore_index=True)
        for k in range(10, 100, 10):
            exp[f"{corpus}:{a.label}:{k}"] = int((df.info_decile < k // 10).sum())
    act = json.loads(s.get_object(
        Bucket=B, Key=f"recoverability/analysis/matrix_verification/"
                      f"{a.label}_actual.json")["Body"].read())
    V.mkdir(parents=True, exist_ok=True)
    (V / f"{a.label}_expected.json").write_text(json.dumps(exp, indent=1))
    (V / f"{a.label}_actual.json").write_text(json.dumps(act, indent=1))

    ok = bad = 0
    viol, expl = [], []
    errs = [c for c in act["cells"] if "error" in c]
    for c in act["cells"]:
        if "error" in c:
            print(f"ERROR {c['slug']}: {c['error']}")
            continue
        k, short = re.match(rf"pdrop2_{a.label}(\d+)_(\w+)$", c["slug"]).groups()
        k = int(k)
        if short == "rmexpl":
            expl.append(c["short_after_pool"])
            continue
        e_tr, e_po = exp[f"train_90M:{a.label}:{k}"], exp[f"pull_10M:{a.label}:{k}"]
        m = c["train_removed"] == e_tr and c["pool_removed"] == e_po
        ok += m
        bad += not m
        if not m:
            print(f"MISMATCH {c['slug']}: train {c['train_removed']:,}/{e_tr:,} "
                  f"pool {c['pool_removed']:,}/{e_po:,}")
        if c["short_after_pool"] or c["exhausted"]:
            viol.append(c["slug"])
    rng = f"{min(expl):,}..{max(expl):,}" if expl else "-"
    print(f"VERIFY {a.label}: cells={len(act['cells'])}/{a.n_cells} "
          f"errors={len(errs)} exact={ok}/{ok + bad} violations={viol or 'NONE'} "
          f"rmexpl deficits {rng} ({len(expl)} cells) | "
          f"PVC free {act['df_free_tb']:.1f} TB")
    passed = (len(act["cells"]) == a.n_cells and not errs and bad == 0
              and not viol)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
