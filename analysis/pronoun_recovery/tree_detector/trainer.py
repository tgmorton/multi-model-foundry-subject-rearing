"""
Train decision tree and gradient boosted classifiers for null subject detection.

Binary detection: is this finite verb a referential null-subject position?
Trains both DecisionTreeClassifier and HistGradientBoostingClassifier,
compares metrics, and exports models + reports.
"""

import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from analysis.pronoun_recovery.constants import LABEL_NONE

from .evaluator import evaluate_predictions
from .feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Columns excluded from the detection tree.
# verb_person/verb_number: detection should be person-agnostic; classification
#   comes from morphology (already 100% accurate).
_DETECTION_EXCLUDE_COLS = frozenset({
    "verb_person", "verb_number",
})

# Hard prefilter column: if this feature is True, predict NONE without the
# tree.  Set to None to disable prefiltering.
_PREFILTER_COL = None


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _binarise_labels(y: np.ndarray) -> np.ndarray:
    """Convert PRO.* labels to 1 (null subject) and NONE to 0."""
    return (y != LABEL_NONE).astype(int)


def _identify_categorical_columns(X: pd.DataFrame) -> list:
    """Find columns with object/bool dtype that need ordinal encoding."""
    cats = []
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype == bool:
            cats.append(col)
    return cats


def prepare_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, OrdinalEncoder]:
    """Encode categoricals and split into train/test.

    Returns:
        (X_train, X_test, y_train_bin, y_test_bin, encoder)
    """
    y_bin = _binarise_labels(y)

    # Encode categoricals with OrdinalEncoder
    cat_cols = _identify_categorical_columns(X)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_encoded = X.copy()
    if cat_cols:
        X_encoded[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
    else:
        encoder.fit(pd.DataFrame())  # fit on empty so it's serialisable

    # Replace None/NaN with -1 for tree compatibility
    X_encoded = X_encoded.fillna(-1)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_bin,
        test_size=test_fraction,
        random_state=seed,
        stratify=y_bin,
    )

    logger.info(
        "Split: %d train (%d pos), %d test (%d pos)",
        len(X_train), y_train.sum(),
        len(X_test), y_test.sum(),
    )

    return X_train, X_test, y_train, y_test, encoder


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = 5,
    max_depth: int = None,
    min_samples_leaf: int = 5,
    seed: int = 42,
) -> DecisionTreeClassifier:
    """Train a DecisionTreeClassifier with grid search over max_depth.

    Args:
        X_train: Encoded training features.
        y_train: Binary training labels.
        cv_folds: Number of stratified CV folds.
        max_depth: If set, use this depth; otherwise grid-search.
        min_samples_leaf: Minimum samples per leaf.
        seed: Random seed.

    Returns:
        Best fitted DecisionTreeClassifier.
    """
    if max_depth is not None:
        param_grid = {
            "max_depth": [max_depth],
            "min_samples_leaf": [min_samples_leaf],
        }
    else:
        param_grid = {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_leaf": [2, 5, 10, 20],
        }

    dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=seed,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        dt,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info(
        "DT best params: %s  CV F1: %.4f",
        grid.best_params_,
        grid.best_score_,
    )

    return grid.best_estimator_


def train_gradient_boosted(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = 5,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = None,
    seed: int = 42,
) -> HistGradientBoostingClassifier:
    """Train a HistGradientBoostingClassifier with grid search.

    Args:
        X_train: Encoded training features.
        y_train: Binary training labels.
        cv_folds: Number of stratified CV folds.
        n_estimators: Max number of boosting iterations.
        learning_rate: Step size shrinkage.
        max_depth: Max tree depth for boosting.
        seed: Random seed.

    Returns:
        Best fitted HistGradientBoostingClassifier.
    """
    param_grid = {
        "max_depth": [3, 5, 8] if max_depth is None else [max_depth],
        "learning_rate": [0.05, 0.1] if learning_rate == 0.1 else [learning_rate],
        "max_iter": [100, 200] if n_estimators == 200 else [n_estimators],
    }

    # Compute sample weight for class balancing
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    ratio = n_neg / n_pos if n_pos > 0 else 1.0
    sample_weight = np.where(y_train == 1, ratio, 1.0)

    hgb = HistGradientBoostingClassifier(
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        hgb,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train, sample_weight=sample_weight)

    logger.info(
        "HGB best params: %s  CV F1: %.4f",
        grid.best_params_,
        grid.best_score_,
    )

    return grid.best_estimator_


def export_tree_rules(
    dt_model: DecisionTreeClassifier,
    feature_names: list,
    output_path: Path,
) -> str:
    """Export decision tree rules as human-readable text."""
    rules = export_text(dt_model, feature_names=feature_names, max_depth=20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rules)
    logger.info("Tree rules exported to %s", output_path)
    return rules


def export_feature_importance(
    model,
    feature_names: list,
    output_path: Path,
) -> Dict[str, float]:
    """Export feature importance as JSON."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

    importance_dict = {
        name: float(imp)
        for name, imp in sorted(
            zip(feature_names, importances),
            key=lambda x: -x[1],
        )
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(importance_dict, f, indent=2)

    logger.info("Feature importance exported to %s", output_path)
    return importance_dict


def run_training(
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path,
    test_fraction: float = 0.2,
    cv_folds: int = 5,
    max_depth: int = None,
    min_samples_leaf: int = 5,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    gb_max_depth: int = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full training pipeline: prepare, train both models, evaluate, export.

    Two-stage detection pipeline:
    1. Hard prefilter: verbs with a reachable subject → NONE (not sent to tree).
    2. Decision tree on remaining verbs, without person/number features
       (detection should be person-agnostic).

    Args:
        X: Feature DataFrame.
        y: Label array (PRO.* or NONE).
        output_dir: Directory for all outputs.
        test_fraction: Fraction held out for testing.
        cv_folds: Cross-validation folds.
        max_depth: DT max depth (None = grid search).
        min_samples_leaf: DT min samples leaf.
        n_estimators: HGB max iterations.
        learning_rate: HGB learning rate.
        gb_max_depth: HGB max tree depth.
        seed: Random seed.

    Returns:
        Comparison report dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data (encode + split)
    X_train, X_test, y_train, y_test, encoder = prepare_data(
        X, y, test_fraction=test_fraction, seed=seed
    )

    all_feature_names = list(X_train.columns)

    # ── Prefilter ────────────────────────────────────────────────────
    # If _PREFILTER_COL is set, rows where it's True are auto-NONE.
    # OrdinalEncoder maps bool→str: "False"→0.0, "True"→1.0
    if _PREFILTER_COL and _PREFILTER_COL in X_train.columns:
        pf_train = X_train[_PREFILTER_COL] > 0.5
        pf_test = X_test[_PREFILTER_COL] > 0.5
        logger.info(
            "Prefilter (%s): %d/%d train, %d/%d test → auto NONE",
            _PREFILTER_COL,
            pf_train.sum(), len(X_train), pf_test.sum(), len(X_test),
        )
    else:
        pf_train = pd.Series(False, index=X_train.index)
        pf_test = pd.Series(False, index=X_test.index)
        logger.info("No prefilter applied.")

    # ── Tree feature columns (exclude person/number + prefilter col) ─
    exclude = set(_DETECTION_EXCLUDE_COLS)
    if _PREFILTER_COL:
        exclude.add(_PREFILTER_COL)
    tree_feature_names = [
        c for c in all_feature_names if c not in exclude
    ]
    logger.info(
        "Tree features: %d (excluded %s)",
        len(tree_feature_names),
        sorted(exclude),
    )

    # Training subsets: only rows NOT caught by prefilter
    X_train_tree = X_train.loc[~pf_train, tree_feature_names]
    y_train_tree = y_train[~pf_train.values]

    logger.info(
        "Tree training set: %d rows (%d pos)",
        len(X_train_tree), y_train_tree.sum(),
    )

    # Train decision tree
    logger.info("Training DecisionTreeClassifier...")
    dt_model = train_decision_tree(
        X_train_tree, y_train_tree,
        cv_folds=cv_folds,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )

    # Train gradient boosted
    logger.info("Training HistGradientBoostingClassifier...")
    hgb_model = train_gradient_boosted(
        X_train_tree, y_train_tree,
        cv_folds=cv_folds,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=gb_max_depth,
        seed=seed,
    )

    # ── Evaluate on full test set ────────────────────────────────────
    # Prefiltered rows → predict 0 (NONE), rest → tree prediction
    dt_preds = np.zeros(len(X_test), dtype=int)
    hgb_preds = np.zeros(len(X_test), dtype=int)
    tree_mask = ~pf_test

    if tree_mask.any():
        X_test_tree = X_test.loc[tree_mask, tree_feature_names]
        dt_preds[tree_mask.values] = dt_model.predict(X_test_tree)
        hgb_preds[tree_mask.values] = hgb_model.predict(X_test_tree)

    # Reconstruct original multi-class labels for feature_accuracy
    y_bin = _binarise_labels(y)
    _, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=test_fraction,
        random_state=seed,
        stratify=y_bin,
    )
    y_test_orig = y[test_indices]

    dt_metrics = evaluate_predictions(
        y_test, dt_preds, y_test_orig, "DecisionTree"
    )
    hgb_metrics = evaluate_predictions(
        y_test, hgb_preds, y_test_orig, "GradientBoosted"
    )

    # ── Export models + pipeline config ──────────────────────────────
    joblib.dump(dt_model, output_dir / "dt_model.joblib")
    joblib.dump(hgb_model, output_dir / "hgb_model.joblib")
    joblib.dump(encoder, output_dir / "feature_encoder.joblib")

    pipeline_config = {
        "prefilter_col": _PREFILTER_COL,
        "detection_exclude_cols": sorted(_DETECTION_EXCLUDE_COLS),
        "tree_feature_names": tree_feature_names,
    }
    joblib.dump(pipeline_config, output_dir / "pipeline_config.joblib")
    logger.info("Models + pipeline config saved to %s", output_dir)

    # Export tree rules and feature importance using tree column names
    export_tree_rules(dt_model, tree_feature_names, output_dir / "dt_rules.txt")

    dt_importance = export_feature_importance(
        dt_model, tree_feature_names, output_dir / "dt_feature_importance.json"
    )
    hgb_importance = export_feature_importance(
        hgb_model, tree_feature_names, output_dir / "hgb_feature_importance.json"
    )

    # Comparison report
    report = {
        "decision_tree": {
            "params": {
                "max_depth": dt_model.get_depth(),
                "n_leaves": dt_model.get_n_leaves(),
                "min_samples_leaf": dt_model.min_samples_leaf,
            },
            "metrics": dt_metrics,
            "top_features": dict(list(dt_importance.items())[:10]),
        },
        "gradient_boosted": {
            "params": {
                "n_estimators": getattr(hgb_model, "n_iter_", n_estimators),
                "learning_rate": hgb_model.learning_rate,
                "max_depth": hgb_model.max_depth,
            },
            "metrics": hgb_metrics,
            "top_features": dict(list(hgb_importance.items())[:10]),
        },
        "data": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positive_train": int(y_train.sum()),
            "n_positive_test": int(y_test.sum()),
            "n_tree_train": len(X_train_tree),
            "n_tree_test": int(tree_mask.sum()),
            "n_prefiltered_train": int(pf_train.sum()),
            "n_prefiltered_test": int(pf_test.sum()),
            "n_features": len(all_feature_names),
            "n_tree_features": len(tree_feature_names),
        },
    }

    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=_NumpyEncoder)
    logger.info("Comparison report saved to %s", report_path)

    # Print summary
    logger.info(
        "Decision Tree — detection_f1: %.4f | feature_accuracy: %.4f",
        dt_metrics.get("detection_f1", 0),
        dt_metrics.get("feature_accuracy", 0),
    )
    logger.info(
        "Gradient Boosted — detection_f1: %.4f | feature_accuracy: %.4f",
        hgb_metrics.get("detection_f1", 0),
        hgb_metrics.get("feature_accuracy", 0),
    )

    return report
