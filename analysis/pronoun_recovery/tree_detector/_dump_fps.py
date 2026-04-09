"""Dump false positive analysis data for inspection by sub-agents."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

from analysis.pronoun_recovery.tree_detector.label_aligner import load_aligned_data
from analysis.pronoun_recovery.tree_detector.trainer import _binarise_labels, prepare_data
from analysis.pronoun_recovery.tree_detector.feature_extractor import FEATURE_NAMES

DATA_DIR = Path("data/pronoun_recovery/tree_detector/it")
OUTPUT_DIR = DATA_DIR / "fp_analysis"

# Columns excluded from tree (must match trainer.py)
_DETECTION_EXCLUDE_COLS = {"verb_person", "verb_number"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw features and labels
    X_raw = pd.read_parquet(DATA_DIR / "features.parquet")
    y = np.load(DATA_DIR / "labels.npy", allow_pickle=True)

    # Reproduce the exact same split as training
    y_bin = _binarise_labels(y)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y_bin,
    )

    X_test_raw = X_raw.iloc[test_idx].copy()
    y_test = y[test_idx]
    y_test_bin = y_bin[test_idx]

    # Load model and pipeline config
    dt_model = joblib.load(DATA_DIR / "dt_model.joblib")
    encoder = joblib.load(DATA_DIR / "feature_encoder.joblib")
    pipeline_config = joblib.load(DATA_DIR / "pipeline_config.joblib")

    tree_feature_names = pipeline_config["tree_feature_names"]

    # Encode for prediction
    from analysis.pronoun_recovery.tree_detector.trainer import _identify_categorical_columns
    cat_cols = _identify_categorical_columns(X_test_raw)
    X_test_enc = X_test_raw.copy()
    if cat_cols:
        X_test_enc[cat_cols] = encoder.transform(X_test_raw[cat_cols].astype(str))
    X_test_enc = X_test_enc.fillna(-1)

    # Predict using tree features only
    preds = dt_model.predict(X_test_enc[tree_feature_names])

    # Identify FPs: predicted=1, actual=0
    fp_mask = (preds == 1) & (y_test_bin == 0)
    # Also get TPs for comparison
    tp_mask = (preds == 1) & (y_test_bin == 1)
    # And FNs
    fn_mask = (preds == 0) & (y_test_bin == 1)

    # Build analysis DataFrames with raw (human-readable) features
    fp_df = X_test_raw[fp_mask].copy()
    fp_df["gold_label"] = y_test[fp_mask]
    fp_df["prediction"] = "FP"

    tp_df = X_test_raw[tp_mask].copy()
    tp_df["gold_label"] = y_test[tp_mask]
    tp_df["prediction"] = "TP"

    fn_df = X_test_raw[fn_mask].copy()
    fn_df["gold_label"] = y_test[fn_mask]
    fn_df["prediction"] = "FN"

    # Save full FP data
    fp_df.to_csv(OUTPUT_DIR / "all_fps.csv", index=False)
    tp_df.to_csv(OUTPUT_DIR / "all_tps.csv", index=False)
    fn_df.to_csv(OUTPUT_DIR / "all_fns.csv", index=False)

    # Summary stats
    summary = {
        "total_test": len(y_test),
        "total_positive": int(y_test_bin.sum()),
        "total_negative": int((y_test_bin == 0).sum()),
        "n_fp": int(fp_mask.sum()),
        "n_tp": int(tp_mask.sum()),
        "n_fn": int(fn_mask.sum()),
        "n_tn": int(((preds == 0) & (y_test_bin == 0)).sum()),
    }

    # FP breakdown by dep_label
    fp_by_dep = fp_df["dep_label"].value_counts().to_dict()
    summary["fp_by_dep_label"] = fp_by_dep

    # FP breakdown by key boolean features
    for feat in [
        "has_overt_nsubj", "has_reachable_subject", "is_aux_or_cop",
        "head_has_overt_nsubj", "has_expl", "has_csubj",
        "is_imperative", "is_fragment", "is_weather_verb",
        "is_impersonal_verb", "is_essere", "is_avere", "is_modal",
        "is_in_relcl", "is_xcomp", "is_bridge_complement",
        "has_negation", "is_sentence_initial_verb",
    ]:
        if feat in fp_df.columns:
            true_count = int(fp_df[feat].sum()) if fp_df[feat].dtype == bool else int((fp_df[feat] == True).sum())
            summary[f"fp_{feat}_true"] = true_count
            summary[f"fp_{feat}_pct"] = round(100 * true_count / len(fp_df), 1) if len(fp_df) > 0 else 0

    # Same for TPs for comparison
    tp_summary = {}
    for feat in [
        "has_overt_nsubj", "has_reachable_subject", "is_aux_or_cop",
        "head_has_overt_nsubj", "is_imperative", "is_fragment",
        "is_weather_verb", "is_impersonal_verb", "is_in_relcl",
        "is_xcomp", "is_essere", "is_avere",
    ]:
        if feat in tp_df.columns:
            true_count = int(tp_df[feat].sum()) if tp_df[feat].dtype == bool else int((tp_df[feat] == True).sum())
            tp_summary[f"tp_{feat}_true"] = true_count
            tp_summary[f"tp_{feat}_pct"] = round(100 * true_count / len(tp_df), 1) if len(tp_df) > 0 else 0

    summary["tp_comparison"] = tp_summary

    # Save by-dep-label splits for sub-agents
    for dep_label, group in fp_df.groupby("dep_label"):
        safe_name = dep_label.replace(":", "_")
        group.to_csv(OUTPUT_DIR / f"fps_dep_{safe_name}.csv", index=False)

    # Save by clause_type splits
    for clause_type, group in fp_df.groupby("clause_type"):
        safe_name = str(clause_type).replace(":", "_")
        group.to_csv(OUTPUT_DIR / f"fps_clause_{safe_name}.csv", index=False)

    # TP comparison by dep_label
    tp_by_dep = tp_df["dep_label"].value_counts().to_dict()
    summary["tp_by_dep_label"] = tp_by_dep

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"FPs: {fp_mask.sum()}, TPs: {tp_mask.sum()}, FNs: {fn_mask.sum()}")
    print(f"\nFP by dep_label:\n{json.dumps(fp_by_dep, indent=2)}")
    print(f"\nFiles saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
