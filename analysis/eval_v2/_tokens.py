"""Shared token-axis helper for the wave2 plotting scripts.

Tokens per checkpoint_step is NOT a per-architecture constant: for gpt2_small
the pdrop2 runs use 128,000 (h0), 256,000 (h1) or 512,000 (h2/h3) tokens per
step depending on HP rank; gpt2_large is 512,000. The pairs parquets carry no
tokens_seen column, but the sibling checkpoint parquets do:

    s3://<bucket>/eval_results/null_subj_v2_condition_matched_v1/checkpoints/
        cell_id=<run_id>.parquet      (columns incl. cell_id, run_id,
                                       checkpoint_step, tokens_seen)

This module caches those (separately from the pairs cache), and attaches the
recorded per-checkpoint tokens_seen to any frame with (run id, checkpoint_step)
columns:

  * recorded tokens_seen is used as-is when present;
  * a checkpoint whose tokens_seen is missing (null, or the step is absent from
    the run's checkpoints parquet) is filled as
        step * median(tokens_seen / step over that run's other checkpoints);
  * a run with no usable checkpoints parquet at all falls back to the per-arch
    TOKENS_PER_STEP_FALLBACK constant.
Every fallback is counted and printed (TokenAxis.report()).

Usage:
    from _tokens import TokenAxis, sync_checkpoints, is_arch
    sync_checkpoints(bucket, profile, endpoint, arch)        # optional
    axis = TokenAxis(arch)
    df["tokens"] = axis.tokens(df.cell_id, df.checkpoint_step)
    axis.report()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CKPT_PREFIX = "eval_results/null_subj_v2_condition_matched_v1/checkpoints/"
# Separate from the pairs cache (~/.cache/subject-drop/pairs_cache).
CKPT_CACHE = Path.home() / ".cache" / "subject-drop" / "ckpt_cache"
# Last-resort per-arch constant, used only for runs with no checkpoints parquet
# (or one with no usable tokens_seen at all). NOT correct for every gpt2_small
# HP rank -- that is the point of this module.
TOKENS_PER_STEP_FALLBACK = {"gpt2_small": 256_000, "gpt2_large": 512_000}


def is_arch(name: str, arch: str) -> bool:
    """True if an S3 key / local filename belongs to `arch`. cell_ids look like
    '<arch>-<lang>-<condition>-h<hp>-s<seed>'; the trailing '-' keeps
    'gpt2_small' from matching a longer arch name."""
    base = name.rsplit("/", 1)[-1]
    if base.startswith("cell_id="):
        base = base[len("cell_id="):]
    return base.startswith(f"{arch}-")


def run_id_from_name(name: str) -> str:
    """'cell_id=<run_id>.parquet' (S3 basename / cache filename) -> run_id."""
    base = name.rsplit("/", 1)[-1]
    if base.startswith("cell_id="):
        base = base[len("cell_id="):]
    return base[:-len(".parquet")] if base.endswith(".parquet") else base


# A checkpoint save cut off mid-write leaves a valid safetensors header over
# zero-filled weights; the model then predicts uniformly (log p = -ln V for
# every token), so overt and null tie on every item and prefers_overt reads 0.
# Three wave2 gpt2_large checkpoints are like this on disk (2026-10-15).
DEGENERATE_COLS = ["overt_mean_log_prob", "null_mean_log_prob"]


def drop_degenerate(d: pd.DataFrame) -> pd.DataFrame:
    """Drop checkpoints whose every item scored identically (zeroed weights).
    Needs DEGENERATE_COLS in `d`; returns the frame without them."""
    keys = [c for c in ("cell_id", "checkpoint_step") if c in d.columns]
    spread = d.groupby(keys)[DEGENERATE_COLS].transform("std").max(axis=1)
    bad = spread.fillna(1.0) < 1e-6
    if bad.any():
        steps = sorted(d.loc[bad, "checkpoint_step"].unique().tolist())
        print(f"[degenerate] dropped {bad.sum()} rows at steps {steps}")
    return d.loc[~bad].drop(columns=DEGENERATE_COLS)


def sync_checkpoints(bucket, profile, endpoint, arch, prefix=CKPT_PREFIX,
                     cache=CKPT_CACHE):
    """Download the pdrop2 checkpoint parquets for `arch` into `cache`, in
    parallel with retries; skip files already cached at the right size (same
    pattern as the pairs-cache sync). Returns the local paths."""
    import boto3
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor
    s = boto3.Session(profile_name=profile).client(
        "s3", endpoint_url=endpoint,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=120, max_pool_connections=16))
    cache.mkdir(parents=True, exist_ok=True)
    objs = [o for page in s.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix)
            for o in page.get("Contents", [])
            if o["Key"].endswith(".parquet") and "pdrop2_" in o["Key"]
            and is_arch(o["Key"], arch)]
    if not objs:
        print(f"[tokens] WARNING: no pdrop2 checkpoint parquets for arch "
              f"{arch!r} under s3://{bucket}/{prefix}; every run will use the "
              f"per-arch fallback")
        return []

    def fetch(o):
        p = cache / o["Key"].rsplit("/", 1)[-1]
        if not (p.exists() and p.stat().st_size == o["Size"]):
            s.download_file(bucket, o["Key"], str(p))
        return p

    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(fetch, objs))
    print(f"[tokens] checkpoint parquets for {arch}: {len(paths)} synced to {cache}")
    return paths


class TokenAxis:
    """Per-checkpoint tokens for every run of one architecture."""

    def __init__(self, arch: str, cache: Path = CKPT_CACHE):
        if arch not in TOKENS_PER_STEP_FALLBACK:
            raise ValueError(f"no fallback tokens/step for arch {arch!r}")
        self.arch = arch
        self.fallback_tps = TOKENS_PER_STEP_FALLBACK[arch]
        self.has_file: set[str] = set()            # runs with a checkpoints parquet
        self.recorded: dict[str, pd.Series] = {}   # run -> tokens_seen by step (non-null)
        self.present: dict[str, np.ndarray] = {}   # run -> every step in its parquet
        self.median_tps: dict[str, float] = {}     # run -> median tokens_seen/step
        self.outlier_ckpts = 0
        self.n_files = 0
        self.n_runs = self.n_ckpts = self.n_recorded = 0
        self.n_fill_null = self.n_fill_absent = 0
        self.fill_runs: set[str] = set()
        self.nofile_runs: set[str] = set(); self.n_nofile = 0
        self.nousable_runs: set[str] = set(); self.n_nousable = 0

        cache = Path(cache)
        # never a bare glob over the shared cache dir: always filter by arch
        paths = ([p for p in sorted(cache.glob("*.parquet"))
                  if is_arch(p.name, arch) and "pdrop2_" in p.name]
                 if cache.is_dir() else [])
        for p in paths:
            d = pd.read_parquet(p, columns=None)
            run = run_id_from_name(p.name)
            for col in ("cell_id", "run_id"):   # join key: cell_id, else run_id
                if col in d.columns and d[col].notna().any():
                    run = str(d[col].dropna().iloc[0])
                    break
            self.n_files += 1
            self.has_file.add(run)
            self.present[run] = d["checkpoint_step"].to_numpy()
            ok = d.dropna(subset=["tokens_seen"]).drop_duplicates(
                "checkpoint_step", keep="last")
            self.recorded[run] = ok.set_index("checkpoint_step").tokens_seen.astype("int64")
            pos = ok[ok.checkpoint_step > 0]
            if len(pos):
                ratio = (pos.tokens_seen.astype("float64")
                         / pos.checkpoint_step.astype("float64"))
                med = float(np.median(ratio))
                self.median_tps[run] = med
                self.outlier_ckpts += int((np.abs(ratio - med) > 0.01 * med).sum())

    def tokens(self, run_ids, steps) -> np.ndarray:
        """int64 tokens for each (run_id, checkpoint_step) row of the inputs
        (any array-likes of equal length). Computed once per unique
        (run, step) pair, then broadcast back, so it is cheap on big frames."""
        codes, runs = pd.factorize(np.asarray(run_ids).astype(str))
        steps = np.asarray(steps).astype("int64")
        key = codes.astype("int64") * (1 << 40) + steps       # steps << 2**40
        ukey, inv = np.unique(key, return_inverse=True)
        urun, ustep = ukey >> 40, ukey & ((1 << 40) - 1)
        tok = np.zeros(len(ukey), dtype="int64")
        for ri, run in enumerate(map(str, runs)):
            idx = np.flatnonzero(urun == ri)
            if not len(idx):
                continue
            step = ustep[idx]
            self.n_runs += 1
            self.n_ckpts += len(idx)
            if run not in self.has_file:           # no checkpoints parquet at all
                tok[idx] = step * self.fallback_tps
                self.nofile_runs.add(run); self.n_nofile += len(idx)
                continue
            if run not in self.median_tps:         # file, but no usable tokens_seen
                tok[idx] = step * self.fallback_tps
                self.nousable_runs.add(run); self.n_nousable += len(idx)
                continue
            rec = self.recorded[run]
            got = rec.reindex(step).values         # NaN where null/absent
            miss = np.isnan(got)
            out = np.where(miss, 0, got).astype("int64")
            if miss.any():
                absent = ~np.isin(step, self.present[run])
                out[miss] = np.rint(step[miss] * self.median_tps[run]).astype("int64")
                self.fill_runs.add(run)
                self.n_fill_absent += int((miss & absent).sum())
                self.n_fill_null += int((miss & ~absent).sum())
            self.n_recorded += int((~miss).sum())
            tok[idx] = out
        return tok[inv]

    def report(self) -> None:
        if self.n_runs == 0:
            print(f"[tokens] {self.arch}: nothing joined "
                  f"({self.n_files} cached checkpoint parquets)")
            return
        print(f"[tokens] {self.arch}: {self.n_runs} runs / {self.n_ckpts} checkpoints "
              f"joined against {self.n_files} cached checkpoint parquets; "
              f"{self.n_recorded} used recorded tokens_seen")
        print(f"[tokens]   fallback A, step*median(run tokens/step): "
              f"{self.n_fill_null + self.n_fill_absent} checkpoints in "
              f"{len(self.fill_runs)} runs (null tokens_seen: {self.n_fill_null}; "
              f"step absent from checkpoints parquet: {self.n_fill_absent})")
        print(f"[tokens]   fallback B, per-arch {self.fallback_tps:,}/step: "
              f"{len(self.nofile_runs)} runs / {self.n_nofile} checkpoints with no "
              f"checkpoints parquet; {len(self.nousable_runs)} runs / "
              f"{self.n_nousable} checkpoints whose parquet has no usable tokens_seen")
        if self.outlier_ckpts:
            print(f"[tokens]   note: {self.outlier_ckpts} recorded checkpoints deviate "
                  f">1% from their run's median tokens/step (used as recorded)")
        for label, runs in (("fallback A runs", self.fill_runs),
                            ("fallback B (no parquet) runs", self.nofile_runs),
                            ("fallback B (no usable tokens) runs", self.nousable_runs)):
            if runs:
                print(f"[tokens]   {label}: {sorted(runs)[:8]}"
                      f"{' ...' if len(runs) > 8 else ''}")
