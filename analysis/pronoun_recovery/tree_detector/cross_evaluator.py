"""
Cross-domain evaluation of tree detectors across multiple data sources.

Trains models on different domain combinations (Europarl-only, TTQ at
several sizes, combined) and evaluates each model on all test sets to
measure cross-domain generalisation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from .evaluator import evaluate_predictions
from .trainer import (
    _DETECTION_EXCLUDE_COLS,
    _binarise_labels,
    _identify_categorical_columns,
    _NumpyEncoder,
    export_feature_importance,
    export_tree_rules,
    train_decision_tree,
    train_gradient_boosted,
)

logger = logging.getLogger(__name__)


def _split_domain(
    X: pd.DataFrame,
    y: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified split on binarised labels.

    Returns raw (un-encoded) DataFrames + original string labels.
    """
    y_bin = _binarise_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y_bin,
    )
    logger.info(
        "Split: %d train, %d test (pos: %d/%d)",
        len(X_train), len(X_test),
        (y_train != "NONE").sum(), (y_test != "NONE").sum(),
    )
    return X_train, X_test, y_train, y_test


def _encode_features(
    X_train: pd.DataFrame,
    cat_cols: List[str],
    encoder: Optional[OrdinalEncoder] = None,
) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    """Fit OrdinalEncoder on train if encoder is None, else transform.

    Excludes verb_person/verb_number from the returned feature set.
    """
    exclude = set(_DETECTION_EXCLUDE_COLS)
    feature_cols = [c for c in X_train.columns if c not in exclude]

    X_out = X_train[feature_cols].copy()
    cat_cols_filtered = [c for c in cat_cols if c not in exclude]

    if encoder is None:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        if cat_cols_filtered:
            encoder.fit(X_out[cat_cols_filtered].astype(str))
        else:
            encoder.fit(pd.DataFrame())

    if cat_cols_filtered:
        X_out[cat_cols_filtered] = encoder.transform(
            X_out[cat_cols_filtered].astype(str)
        )

    X_out = X_out.fillna(-1)
    return X_out, encoder


def _encode_test_set(
    X_test: pd.DataFrame,
    cat_cols: List[str],
    encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """Encode a test set with an existing encoder."""
    exclude = set(_DETECTION_EXCLUDE_COLS)
    feature_cols = [c for c in X_test.columns if c not in exclude]
    cat_cols_filtered = [c for c in cat_cols if c not in exclude]

    X_out = X_test[feature_cols].copy()
    if cat_cols_filtered:
        X_out[cat_cols_filtered] = encoder.transform(
            X_out[cat_cols_filtered].astype(str)
        )
    X_out = X_out.fillna(-1)
    return X_out


def _train_and_evaluate(
    name: str,
    X_train_enc: pd.DataFrame,
    y_train: np.ndarray,
    y_train_bin: np.ndarray,
    test_sets: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    output_dir: Path,
    cv_folds: int = 5,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    gb_max_depth: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train DT + HGB on one training set and evaluate on all test sets.

    Args:
        name: Model configuration name (e.g. 'europarl', 'ttq_20k').
        X_train_enc: Encoded training features.
        y_train: Original string labels for training set.
        y_train_bin: Binary training labels.
        test_sets: Dict mapping test set name to
            (X_test_enc, y_test_bin, y_test_orig).
        output_dir: Base output directory for this model.
        cv_folds .. seed: Training hyperparameters.

    Returns:
        Dict with train_size, test_results per test set per model type.
    """
    model_dir = output_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(X_train_enc.columns)

    logger.info("Training %s (N=%d, pos=%d)...", name, len(X_train_enc), y_train_bin.sum())

    # Train decision tree
    dt_model = train_decision_tree(
        X_train_enc, y_train_bin,
        cv_folds=cv_folds,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )

    # Train gradient boosted
    hgb_model = train_gradient_boosted(
        X_train_enc, y_train_bin,
        cv_folds=cv_folds,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=gb_max_depth,
        seed=seed,
    )

    # Save models
    joblib.dump(dt_model, model_dir / "dt_model.joblib")
    joblib.dump(hgb_model, model_dir / "hgb_model.joblib")

    # Export tree rules and feature importance
    export_tree_rules(dt_model, feature_names, model_dir / "dt_rules.txt")
    export_feature_importance(
        dt_model, feature_names, model_dir / "dt_feature_importance.json"
    )
    export_feature_importance(
        hgb_model, feature_names, model_dir / "hgb_feature_importance.json"
    )

    # Evaluate on all test sets
    test_results = {}
    for ts_name, (X_test_enc, y_test_bin, y_test_orig) in test_sets.items():
        dt_preds = dt_model.predict(X_test_enc)
        hgb_preds = hgb_model.predict(X_test_enc)

        dt_metrics = evaluate_predictions(
            y_test_bin, dt_preds, y_test_orig,
            f"{name}/DT/{ts_name}",
        )
        hgb_metrics = evaluate_predictions(
            y_test_bin, hgb_preds, y_test_orig,
            f"{name}/HGB/{ts_name}",
        )

        test_results[ts_name] = {
            "dt": dt_metrics,
            "hgb": hgb_metrics,
        }

    # Label distribution in training set
    train_label_dist = pd.Series(y_train).value_counts().to_dict()

    return {
        "train_size": len(X_train_enc),
        "train_positive": int(y_train_bin.sum()),
        "train_label_distribution": train_label_dist,
        "test_results": test_results,
    }


def run_cross_evaluation(
    domains: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    cfg,
) -> Dict[str, Any]:
    """Main orchestrator for cross-domain evaluation.

    Args:
        domains: Dict mapping domain name to (X, y) tuples.
        cfg: TreeCrossEvalConfig instance.

    Returns:
        Full comparison report dict.
    """
    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Split each domain 80/20 ──────────────────────────────
    splits = {}
    for name, (X, y) in domains.items():
        logger.info("Splitting domain '%s' (%d rows)...", name, len(X))
        X_train, X_test, y_train, y_test = _split_domain(
            X, y, cfg.test_fraction, cfg.seed
        )
        splits[name] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    # Identify categorical columns from the first domain
    first_domain = next(iter(domains))
    cat_cols = _identify_categorical_columns(domains[first_domain][0])
    logger.info("Categorical columns: %s", cat_cols)

    # ── Step 2: Define test sets ─────────────────────────────────────
    # Each test set: (X_raw, y_orig)  — will be encoded per training config
    raw_test_sets = {}
    for name in splits:
        raw_test_sets[name] = (
            splits[name]["X_test"],
            splits[name]["y_test"],
        )

    # Combined test set (all domains concatenated)
    combined_X_test = pd.concat(
        [splits[n]["X_test"] for n in splits], ignore_index=True
    )
    combined_y_test = np.concatenate(
        [splits[n]["y_test"] for n in splits]
    )
    raw_test_sets["combined"] = (combined_X_test, combined_y_test)

    for ts_name, (X_ts, y_ts) in raw_test_sets.items():
        y_bin = _binarise_labels(y_ts)
        logger.info(
            "Test set '%s': %d rows (%d pos)",
            ts_name, len(X_ts), y_bin.sum(),
        )

    # ── Step 3: Define training configurations ───────────────────────
    # Each config: (name, X_train_raw, y_train)
    training_configs: List[Tuple[str, pd.DataFrame, np.ndarray]] = []

    # Europarl-only (if present)
    if "europarl" in splits:
        training_configs.append((
            "europarl",
            splits["europarl"]["X_train"],
            splits["europarl"]["y_train"],
        ))

    # TTQ at various sizes
    if "ttq" in splits:
        ttq_X_train = splits["ttq"]["X_train"]
        ttq_y_train = splits["ttq"]["y_train"]
        ttq_full_size = len(ttq_X_train)

        for size in cfg.ttq_sweep_sizes:
            if size == 0 or size >= ttq_full_size:
                # Use all available TTQ training data
                training_configs.append((
                    "ttq_all",
                    ttq_X_train,
                    ttq_y_train,
                ))
            else:
                # Subsample TTQ training data (stratified)
                ttq_y_bin = _binarise_labels(ttq_y_train)
                X_sub, _, y_sub, _ = train_test_split(
                    ttq_X_train, ttq_y_train,
                    train_size=size,
                    random_state=cfg.seed,
                    stratify=ttq_y_bin,
                )
                training_configs.append((
                    f"ttq_{size // 1000}k",
                    X_sub,
                    y_sub,
                ))

    # Combined: europarl + TTQ downsampled to match europarl size
    if "europarl" in splits and "ttq" in splits:
        ep_train_size = len(splits["europarl"]["X_train"])
        ttq_X_train = splits["ttq"]["X_train"]
        ttq_y_train = splits["ttq"]["y_train"]

        if len(ttq_X_train) > ep_train_size:
            ttq_y_bin = _binarise_labels(ttq_y_train)
            ttq_X_sub, _, ttq_y_sub, _ = train_test_split(
                ttq_X_train, ttq_y_train,
                train_size=ep_train_size,
                random_state=cfg.seed,
                stratify=ttq_y_bin,
            )
        else:
            ttq_X_sub = ttq_X_train
            ttq_y_sub = ttq_y_train

        combined_X_train = pd.concat(
            [splits["europarl"]["X_train"], ttq_X_sub], ignore_index=True
        )
        combined_y_train = np.concatenate(
            [splits["europarl"]["y_train"], ttq_y_sub]
        )
        training_configs.append((
            "combined",
            combined_X_train,
            combined_y_train,
        ))

    # ── Step 4: Train and evaluate each configuration ────────────────
    all_results = {}

    for train_name, X_train_raw, y_train in training_configs:
        logger.info("=" * 60)
        logger.info("Training configuration: %s", train_name)
        logger.info("=" * 60)

        y_train_bin = _binarise_labels(y_train)

        # Fit encoder on this training set
        X_train_enc, encoder = _encode_features(X_train_raw, cat_cols)

        # Save encoder
        model_dir = output_dir / "models" / train_name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, model_dir / "feature_encoder.joblib")

        # Encode all test sets with this encoder
        encoded_test_sets = {}
        for ts_name, (X_ts_raw, y_ts) in raw_test_sets.items():
            X_ts_enc = _encode_test_set(X_ts_raw, cat_cols, encoder)
            y_ts_bin = _binarise_labels(y_ts)
            encoded_test_sets[ts_name] = (X_ts_enc, y_ts_bin, y_ts)

        # Train and evaluate
        result = _train_and_evaluate(
            name=train_name,
            X_train_enc=X_train_enc,
            y_train=y_train,
            y_train_bin=y_train_bin,
            test_sets=encoded_test_sets,
            output_dir=output_dir,
            cv_folds=cfg.cv_folds,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            gb_max_depth=cfg.gb_max_depth,
            seed=cfg.seed,
        )

        all_results[train_name] = result

    # ── Step 5: Save reports ─────────────────────────────────────────
    report = {"models": all_results}

    report_path = output_dir / "cross_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=_NumpyEncoder)
    logger.info("Full report saved to %s", report_path)

    # Generate markdown summary
    summary = _generate_summary_markdown(all_results, raw_test_sets)
    summary_path = output_dir / "cross_eval_summary.md"
    summary_path.write_text(summary)
    logger.info("Summary saved to %s", summary_path)

    return report


def _generate_summary_markdown(
    results: Dict[str, Any],
    test_sets: Dict[str, Any],
) -> str:
    """Generate a human-readable markdown comparison table."""
    test_names = [n for n in test_sets]
    lines = []

    lines.append("# Cross-Domain Tree Detector Evaluation\n")

    # ── HGB summary table ────────────────────────────────────────────
    lines.append("## HGB Detection F1\n")
    header = "| Model | Train N |"
    separator = "|-------|---------|"
    for ts in test_names:
        header += f" {ts} F1 |"
        separator += "---------|"
    lines.append(header)
    lines.append(separator)

    for model_name, model_result in results.items():
        row = f"| {model_name} | {model_result['train_size']:,} |"
        for ts in test_names:
            ts_result = model_result["test_results"].get(ts, {})
            hgb = ts_result.get("hgb", {})
            f1 = hgb.get("detection_f1", 0)
            row += f" {f1:.3f} |"
        lines.append(row)

    lines.append("")

    # ── DT summary table ─────────────────────────────────────────────
    lines.append("## DT Detection F1\n")
    header = "| Model | Train N |"
    separator = "|-------|---------|"
    for ts in test_names:
        header += f" {ts} F1 |"
        separator += "---------|"
    lines.append(header)
    lines.append(separator)

    for model_name, model_result in results.items():
        row = f"| {model_name} | {model_result['train_size']:,} |"
        for ts in test_names:
            ts_result = model_result["test_results"].get(ts, {})
            dt = ts_result.get("dt", {})
            f1 = dt.get("detection_f1", 0)
            row += f" {f1:.3f} |"
        lines.append(row)

    lines.append("")

    # ── Per-label recall breakdown (HGB) ─────────────────────────────
    lines.append("## Per-Label Recall (HGB)\n")

    for model_name, model_result in results.items():
        lines.append(f"### {model_name} (N={model_result['train_size']:,})\n")

        # Collect all labels across test sets
        all_labels = set()
        for ts in test_names:
            ts_result = model_result["test_results"].get(ts, {})
            hgb = ts_result.get("hgb", {})
            per_label = hgb.get("per_label", {})
            all_labels.update(per_label.keys())

        if not all_labels:
            lines.append("No per-label data available.\n")
            continue

        sorted_labels = sorted(all_labels)
        header = "| Label |"
        separator = "|-------|"
        for ts in test_names:
            header += f" {ts} |"
            separator += "------|"
        header += " Support |"
        separator += "---------|"
        lines.append(header)
        lines.append(separator)

        for label in sorted_labels:
            row = f"| {label} |"
            support = 0
            for ts in test_names:
                ts_result = model_result["test_results"].get(ts, {})
                hgb = ts_result.get("hgb", {})
                per_label = hgb.get("per_label", {})
                label_data = per_label.get(label, {})
                recall = label_data.get("recall", 0)
                support = max(support, label_data.get("support", 0))
                row += f" {recall:.3f} |"
            row += f" {support} |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)
