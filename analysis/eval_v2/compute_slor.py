#!/usr/bin/env python3
"""Post-hoc SLOR from stored per-token log-probs + per-condition unigrams.

SLOR = (Σ log P_model(t) − Σ log P_unigram(t)) / n_tokens, per target span.
Every term is already persisted: per_token parquets hold per_token_log_prob
and per_token_ids; the unigram tables (one per condition corpus, FIT-CLAMS
normalization) come from S3. So SLOR needs no GPU rescore — it's a
deterministic transform of existing eval output.

Emits slor_pairs/cell_id=*.parquet (pairs-shaped: prefers_overt_slor,
slor_diff_overt_minus_null) so the figure script can plot it as a metric.

Usage:
    AWS_PROFILE=nrp python analysis/eval_v2/compute_slor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "data/eval_results/null_subj_v2"
UNI_LOCAL = REPO / "data/unigrams"


def _load_unigrams(lang: str, conditions) -> dict:
    import boto3
    from evaluation.unigram import UnigramTable
    UNI_LOCAL.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", endpoint_url="https://s3-west.nrp-nautilus.io")
    tables = {}
    for cond in conditions:
        name = f"{lang}_{cond}__shared_unigram.pkl"
        local = UNI_LOCAL / name
        if not local.exists():
            s3.download_file("thomas-subject-drop-artifacts",
                             f"unigrams/{name}", str(local))
        tables[cond] = UnigramTable.load(local)
    return tables


def main() -> None:
    pt_dir = DATA / "per_token"
    out_dir = DATA / "slor_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pt_dir.glob("*.parquet"))
    # only shared_unigram archs (bert wordpiece unigram not built yet)
    files = [f for f in files if "bert_large" not in f.name]
    conds = ["baseline", "remove_expletive_sentences", "impoverish_case",
             "lemmatize_verbs", "enrich_verbal_morphology"]
    unigrams = _load_unigrams("en", conds)
    print(f"loaded {len(unigrams)} unigram tables; processing {len(files)} cells")

    n_done = 0
    for f in files:
        df = pd.read_parquet(f)
        cond = df.intervention.iloc[0]
        uni = unigrams.get(cond)
        if uni is None:
            continue
        lp = uni.log_probs  # np array indexed by token id
        # per-item SLOR
        def _slor(row):
            ids = np.asarray(row.per_token_ids, dtype=np.int64)
            if ids.size == 0:
                return np.nan
            model_sum = float(np.sum(row.per_token_log_prob))
            uni_sum = float(lp[ids].sum())
            return (model_sum - uni_sum) / ids.size
        df["slor"] = df.apply(_slor, axis=1)

        # pair overt (pronoun_status==1) vs null (0)
        key = ["checkpoint_step", "category", "condition", "item_id"]
        piv = df.pivot_table(index=key, columns="pronoun_status",
                             values="slor", aggfunc="first")
        piv = piv.dropna(subset=[0, 1]).reset_index()
        piv["slor_diff_overt_minus_null"] = piv[1] - piv[0]
        piv["prefers_overt_slor"] = piv["slor_diff_overt_minus_null"] > 0
        piv["cell_id"] = df.cell_id.iloc[0]
        piv["architecture"] = df.architecture.iloc[0]
        piv["intervention"] = cond
        piv.drop(columns=[0, 1]).to_parquet(
            out_dir / f"cell_id={df.cell_id.iloc[0]}.parquet", index=False)
        n_done += 1
    print(f"wrote {n_done} slor_pairs parquets → {out_dir}")


if __name__ == "__main__":
    main()
