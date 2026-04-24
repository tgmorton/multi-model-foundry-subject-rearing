"""
Plot model preference for overt subjects over training checkpoints.

Reads per-checkpoint JSONL files from the evaluation runner
(``{checkpoint_name}_null_subject_detailed.jsonl``), aggregates the
binary ``prefers_overt`` flag across trials, and emits one figure per
category bucket:

  figures/baseline_eval_2026-04-24/
    overt_required_preference_over_time.{png,pdf}   ← main figure
    control_preference_over_time.{png,pdf}
    extraction_preference_over_time.{png,pdf}
    conjunction_preference_over_time.{png,pdf}

Usage:
    python scripts/plot_baseline_overt_preference.py \\
        --results-dir /path/to/eval_results \\
        --out-dir figures/baseline_eval_2026-04-24

``--results-dir`` can be either a local directory of JSONLs or an S3
prefix (``s3://bucket/prefix/``). If S3, the script downloads the
JSONLs to a tempdir before plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# V1 item_group → grammatical category bucket.
# Overt-required = English grammar forbids subject drop in finite clauses,
# so the model should prefer the overt form. Everything else goes to alt
# figures because the expected preference depends on the construction.
OVERT_REQUIRED_GROUPS = {
    "1a_3rdSG": "subj_3sg",
    "1b_3rdPL": "subj_3pl",
    "2a_2ndSG": "subj_2sg",
    "2b_2ndPL": "subj_2pl",
    "3a_1stSg": "subj_1sg",
    "3b_1stPL": "subj_1pl",
    "5a_expletive_seems": "expl_seems",
    "5b_expletive_be": "expl_be",
}
CONTROL_GROUPS = {
    "4a_subject_control": "subj_control",
    "4b_object_control": "obj_control",
}
EXTRACTION_GROUPS = {
    "6_long_distance_binding": "long_dist_binding",
}
CONJUNCTION_GROUPS = {
    "7a_conjunction_no_topic_shift": "conj_no_shift",
    "7b_conjunction_topic_shift": "conj_topic_shift",
}

ALL_BUCKETS = {
    "overt_required": OVERT_REQUIRED_GROUPS,
    "control": CONTROL_GROUPS,
    "extraction": EXTRACTION_GROUPS,
    "conjunction": CONJUNCTION_GROUPS,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_CKPT_RE = re.compile(r"checkpoint-(\d+)_null_subject_detailed\.jsonl$")


def extract_step(fname: str) -> int:
    """Extract the training step from a ``checkpoint-N_null_subject_*.jsonl``
    filename. Returns -1 if no step found (keeps unsortable entries out of
    the main time-course plot)."""
    m = _CKPT_RE.search(fname)
    return int(m.group(1)) if m else -1


def _maybe_download_from_s3(s3_uri: str) -> Path:
    """If ``s3_uri`` points to an S3 prefix, mirror it to a local tempdir and
    return the local path. Requires boto3 + AWS env vars."""
    if not s3_uri.startswith("s3://"):
        return Path(s3_uri)

    import boto3  # local import so laptop runs without boto3 for local-only case

    tmp = Path(tempfile.mkdtemp(prefix="eval-"))
    rest = s3_uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue
            dst = tmp / Path(key).name
            s3.download_file(bucket, key, str(dst))
            print(f"  ↓ {key} ({obj['Size']/1024:.1f} KB)", file=sys.stderr)
    print(f"Mirrored {len(list(tmp.iterdir()))} files to {tmp}", file=sys.stderr)
    return tmp


def load_checkpoint_records(results_dir: Path) -> pd.DataFrame:
    """Load all ``{checkpoint_name}_null_subject_detailed.jsonl`` files in
    ``results_dir`` and return a long-format DataFrame with columns:
        step, checkpoint_name, item_id, item_group, prefers_overt,
        surprisal_difference, overt_mean_surprisal, null_mean_surprisal
    """
    files = sorted(results_dir.glob("*_null_subject_detailed.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No *_null_subject_detailed.jsonl files in {results_dir}"
        )
    rows: List[Dict] = []
    for f in files:
        step = extract_step(f.name)
        if step < 0:
            print(f"  skip {f.name} (no step)", file=sys.stderr)
            continue
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                rows.append(
                    {
                        "step": step,
                        "checkpoint_name": f.stem.replace(
                            "_null_subject_detailed", ""
                        ),
                        "item_id": r.get("item_id") or r.get("item"),
                        "item_group": r.get("item_group") or r.get("category"),
                        "prefers_overt": r.get("prefers_overt"),
                        "surprisal_difference": r.get("surprisal_difference"),
                        "overt_mean_surprisal": r.get("overt_mean_surprisal"),
                        "null_mean_surprisal": r.get("null_mean_surprisal"),
                    }
                )
    df = pd.DataFrame(rows)
    print(
        f"Loaded {len(df)} trial rows from {df['step'].nunique()} checkpoints",
        file=sys.stderr,
    )
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame, groups: Dict[str, str]) -> pd.DataFrame:
    """Restrict to rows whose ``item_group`` is in ``groups``, then aggregate
    per (step, item_group): proportion of trials where ``prefers_overt`` is
    True. Returns a DataFrame with columns ``step, item_group, condition,
    prefers_overt_rate, n``.
    """
    mask = df["item_group"].isin(groups)
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub["condition"] = sub["item_group"].map(groups)
    agg = (
        sub.groupby(["step", "item_group", "condition"], as_index=False)
           .agg(prefers_overt_rate=("prefers_overt", "mean"),
                n=("prefers_overt", "size"))
    )
    return agg


def plot_bucket(agg: pd.DataFrame, title: str, out_base: Path,
                y_ref: float = 0.5) -> None:
    """Time-course plot of prefers_overt_rate per condition. X axis is
    training step (log-scaled; step=0 placed at 0.5 for log display)."""
    if agg.empty:
        print(f"[skip] {title}: no rows", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    # Plot a line per condition
    for condition, g in agg.groupby("condition"):
        g_sorted = g.sort_values("step")
        steps = g_sorted["step"].replace(0, 0.5)  # log-safe
        ax.plot(
            steps, g_sorted["prefers_overt_rate"],
            marker="o", markersize=3, linewidth=1.2, label=condition,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y_ref, color="grey", linestyle="--", linewidth=0.8,
               label="chance (0.5)")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("P(prefers overt subject)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(out_base.with_suffix(ext))
        print(f"  wrote {out_base.with_suffix(ext)}", file=sys.stderr)
    plt.close(fig)


def plot_aggregate_overall(agg: pd.DataFrame, title: str, out_base: Path) -> None:
    """One-line aggregate plot: average prefers_overt_rate across all
    conditions in a bucket, per step."""
    if agg.empty:
        return
    overall = (
        agg.groupby("step", as_index=False)
           .agg(prefers_overt_rate=("prefers_overt_rate", "mean"))
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    steps = overall["step"].replace(0, 0.5)
    ax.plot(steps, overall["prefers_overt_rate"],
            marker="o", markersize=4, linewidth=1.6, color="tab:blue",
            label="aggregate")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8,
               label="chance (0.5)")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("P(prefers overt subject)  (aggregate)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(out_base.with_suffix(ext))
        print(f"  wrote {out_base.with_suffix(ext)}", file=sys.stderr)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True,
                   help="Local dir or s3://… prefix containing per-checkpoint "
                        "*_null_subject_detailed.jsonl")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output figure directory")
    args = p.parse_args(argv)

    results_dir = _maybe_download_from_s3(str(args.results_dir))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_checkpoint_records(results_dir)
    if df.empty:
        print("No data loaded. Exiting.", file=sys.stderr)
        return 1

    # Main figure: overt-required (subject_drop + expletive)
    agg_overt = _aggregate(df, OVERT_REQUIRED_GROUPS)
    plot_bucket(
        agg_overt,
        "Baseline English GPT-2 medium: P(overt) on overt-required trials",
        out_dir / "overt_required_per_condition",
    )
    plot_aggregate_overall(
        agg_overt,
        "Aggregate preference for overt subject (overt-required trials)",
        out_dir / "overt_required_aggregate",
    )

    # Alt figures
    for bucket_name in ("control", "extraction", "conjunction"):
        groups = ALL_BUCKETS[bucket_name]
        agg = _aggregate(df, groups)
        plot_bucket(
            agg,
            f"Baseline EN baseline: P(overt) on {bucket_name} trials",
            out_dir / f"{bucket_name}_per_condition",
        )

    # Also save the long-format aggregate table so the user can re-plot.
    agg_all = _aggregate(df, {**OVERT_REQUIRED_GROUPS, **CONTROL_GROUPS,
                              **EXTRACTION_GROUPS, **CONJUNCTION_GROUPS})
    agg_all.to_csv(out_dir / "aggregate_preferences.csv", index=False)
    print(f"Wrote aggregate table to {out_dir / 'aggregate_preferences.csv'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
