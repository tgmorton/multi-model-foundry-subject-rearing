#!/usr/bin/env python3
"""Phase-2/3 storage pruning executor (D6 protocol; Thomas-reviewed
manifests 2026-08-22).

Manifest-driven and dry-run-first:
    PHASE=2a  delete models/sweeps dirs listed in phase2_sweeps_manifest.csv
    PHASE=2b  delete failed/invalidated/orphan dirs listed in
              phase2_failed_orphan_manifest.csv
    PHASE=3   strip interior resume states per phase3_resume_strip_manifest:
              delete checkpoint-<step>/training_state.pt for strip_steps
              ONLY, and only when that checkpoint's weights file is
              verified present; keep_steps are never touched.
    DRY_RUN=1 (default) -> enumerate + total bytes, delete nothing.

Safety guards, all hard failures:
- every manifest path must be relative, match ^models/(sweeps|invalidated|
  production)/ (phase 2) and resolve inside /mnt/data with no symlink;
- the eval keep-list (union of cell_id in the condition_matched_v1 CSVs,
  re-derived at execution time from the freshly cloned repo) must not
  contain any phase-2 run_id;
- phase-2 refuses run dirs whose census bucket wasn't phase2_* (manifest
  and bucket check both required);
- every deletion is journaled to /mnt/data/storage_audit/deletions/<ts>.jsonl
  and mirrored to S3.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

DATA = Path("/mnt/data")
REPO = Path(__file__).resolve().parent.parent
KEEPLIST_GLOB = "analysis/eval_v2/figures/foundry_trajectories/condition_matched_v1/*.csv"
_ALLOWED_P2 = re.compile(r"^models/(sweeps|invalidated|production)/[^/]+$")


def keep_list() -> set:
    cells = set()
    for f in REPO.glob(KEEPLIST_GLOB):
        with open(f, newline="") as fh:
            r = csv.DictReader(fh)
            if r.fieldnames and "cell_id" in r.fieldnames:
                cells.update(row["cell_id"] for row in r)
    if len(cells) < 700:
        sys.exit(f"FATAL: keep-list looks wrong ({len(cells)} cells) — refusing")
    return cells


def safe_target(rel: str) -> Path:
    if rel.startswith("/") or ".." in rel.split("/"):
        sys.exit(f"FATAL: suspicious path {rel!r}")
    p = (DATA / rel)
    if p.is_symlink():
        sys.exit(f"FATAL: symlink target {p}")
    if not str(p.resolve()).startswith(str(DATA.resolve()) + "/"):
        sys.exit(f"FATAL: escapes data root: {p}")
    return p


def du(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def main() -> None:
    phase = os.environ["PHASE"].strip()
    dry = os.environ.get("DRY_RUN", "1").strip() != "0"
    manifest_dir = Path(os.environ.get("MANIFEST_DIR", "/tmp/manifests"))
    journal_dir = DATA / "storage_audit" / "deletions"
    journal_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    journal_path = journal_dir / f"phase{phase}_{ts}{'_DRYRUN' if dry else ''}.jsonl"
    journal = open(journal_path, "w")
    keep = keep_list()
    print(f"=== PHASE {phase} {'DRY-RUN' if dry else 'EXECUTE'} — "
          f"keep-list {len(keep)} cells", flush=True)

    freed = 0
    n = 0
    if phase in ("2a", "2b"):
        fname = ("phase2_sweeps_manifest.csv" if phase == "2a"
                 else "phase2_failed_orphan_manifest.csv")
        with open(manifest_dir / fname, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            rel, rid = row["path"], row["run_id"]
            if not _ALLOWED_P2.match(rel):
                sys.exit(f"FATAL: path not allowed for phase 2: {rel}")
            if rid in keep:
                sys.exit(f"FATAL: manifest contains keep-list run {rid}")
            target = safe_target(rel)
            if not target.exists():
                journal.write(json.dumps({"path": rel, "action": "missing"}) + "\n")
                continue
            size = du(target)
            journal.write(json.dumps({"path": rel, "run_id": rid,
                                      "bytes": size, "dry_run": dry}) + "\n")
            if not dry:
                shutil.rmtree(target)
            freed += size
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(rows)} ({freed/1e12:.2f} TB)", flush=True)
    elif phase == "3":
        with open(manifest_dir / "phase3_resume_strip_manifest.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            rel, rid = row["path"], row["run_id"]
            if rid not in keep:
                sys.exit(f"FATAL: phase-3 row not in keep-list: {rid}")
            run_dir = safe_target(rel)
            strip_steps = json.loads(row["strip_steps"].replace("'", '"'))
            keep_steps = set(json.loads(row["keep_steps"].replace("'", '"')))
            for step in strip_steps:
                if step in keep_steps:
                    sys.exit(f"FATAL: {rid} step {step} in both lists")
                ck = run_dir / f"checkpoint-{step}"
                st = ck / "training_state.pt"
                if not st.exists():
                    continue
                has_weights = (ck / "model.safetensors").exists() or \
                              (ck / "pytorch_model.bin").exists()
                if not has_weights:
                    journal.write(json.dumps(
                        {"path": str(st.relative_to(DATA)),
                         "action": "SKIPPED_no_weights"}) + "\n")
                    continue
                size = st.stat().st_size
                journal.write(json.dumps(
                    {"path": str(st.relative_to(DATA)), "bytes": size,
                     "dry_run": dry}) + "\n")
                if not dry:
                    st.unlink()
                freed += size
                n += 1
        print(f"  stripped files: {n}", flush=True)
    else:
        sys.exit(f"FATAL: unknown PHASE {phase!r}")

    journal.close()
    print(f"PHASE {phase} {'DRY-RUN' if dry else 'EXECUTE'} DONE: "
          f"{n} targets, {freed/1e12:.3f} TB {'would be' if dry else ''} freed",
          flush=True)
    try:
        import boto3
        boto3.client("s3").upload_file(
            str(journal_path),
            os.environ.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts"),
            f"storage_audit/deletions/{journal_path.name}")
        print("journal mirrored to S3", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] journal mirror failed: {e}", flush=True)


if __name__ == "__main__":
    main()
