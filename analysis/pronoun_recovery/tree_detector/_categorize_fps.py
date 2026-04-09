"""Categorize ALL false positives into gold data gaps vs genuine model errors.

Produces a complete taxonomy of every FP with an assigned category.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

from analysis.pronoun_recovery.tree_detector.trainer import (
    _binarise_labels, _identify_categorical_columns, _DETECTION_EXCLUDE_COLS,
)
from analysis.pronoun_recovery.tree_detector.feature_extractor import FEATURE_NAMES

DATA_DIR = Path("data/pronoun_recovery/tree_detector/it")
OUTPUT_DIR = DATA_DIR / "fp_analysis"


def categorize_fp(row: pd.Series) -> str:
    """Assign a category to a single FP row (using raw feature values)."""
    dep = row["dep_label"]
    person = row["verb_person"]
    is_cop_imp = row.get("is_copular_impersonal", False)
    rel_subj = row.get("rel_pronoun_is_subject", False)
    is_aux_cop = row["is_aux_or_cop"]
    is_modal = row["is_modal"]
    is_essere = row["is_essere"]
    is_avere = row["is_avere"]
    has_reach = row["has_reachable_subject"]
    head_nsubj = row["head_has_overt_nsubj"]
    is_imp_verb = row["is_impersonal_verb"]
    is_weather = row["is_weather_verb"]
    is_necessity = row["is_necessity_verb"]
    verb_mood = row["verb_mood"]

    # ── Genuine FPs: constructions that are NOT null-subject ──

    # 1. Copular impersonal: "è importante che..."
    if is_cop_imp:
        return "genuine_copular_impersonal"

    # 2. Relative pronoun as subject: "la commissione che decide..."
    if rel_subj:
        return "genuine_rel_pronoun_subject"

    # 3. Impersonal/weather/necessity verbs already detected
    if is_weather:
        return "genuine_weather_verb"
    if is_imp_verb:
        return "genuine_impersonal_verb"
    if is_necessity:
        return "genuine_necessity_verb"

    # 4. 3rd-person cop NOT caught by is_copular_impersonal
    #    (maybe head is not ADJ/NOUN but still impersonal)
    if dep == "cop" and person == "3":
        return "genuine_3p_cop_other"

    # 5. Aux/cop with overt subject on head
    if is_aux_cop and head_nsubj:
        return "genuine_aux_head_has_subject"

    # 6. Verbs with reachable subject (overt subject somewhere in chain)
    if has_reach:
        if person == "3":
            return "genuine_3p_has_reachable_subject"
        else:
            return "uncertain_1p2p_has_reachable_subject"

    # ── Gold data gaps: likely genuine null-subject, gold missed it ──

    # 7. 1st/2nd person aux (compound tense, label not propagated)
    if dep in ("aux", "aux:pass") and person in ("1", "2"):
        if is_modal:
            return "gold_gap_1p2p_modal_aux"
        elif is_avere:
            return "gold_gap_1p2p_avere_aux"
        elif is_essere:
            return "gold_gap_1p2p_essere_aux"
        else:
            return "gold_gap_1p2p_other_aux"

    # 8. 1st/2nd person cop (copular null-subject)
    if dep == "cop" and person in ("1", "2"):
        return "gold_gap_1p2p_cop"

    # 9. 1st/2nd person ROOT (bridge verb, translation restructuring)
    if dep == "ROOT" and person in ("1", "2"):
        return "gold_gap_1p2p_root"

    # 10. 1st/2nd person conj (coordinated verb, label not propagated)
    if dep == "conj" and person in ("1", "2"):
        return "gold_gap_1p2p_conj"

    # 11. 1st/2nd person subordinate clause
    if dep in ("advcl", "ccomp") and person in ("1", "2"):
        return "gold_gap_1p2p_subordinate"

    # 12. 1st/2nd person in relative clause (object relative)
    if dep in ("acl:relcl", "acl") and person in ("1", "2"):
        return "gold_gap_1p2p_relcl"

    # ── 3rd person without clear signal — harder to classify ──

    # 13. 3rd person aux (no subject detected)
    if dep in ("aux", "aux:pass") and person == "3":
        if is_modal:
            return "uncertain_3p_modal_aux"
        elif is_avere:
            return "uncertain_3p_avere_aux"
        elif is_essere:
            return "uncertain_3p_essere_aux"
        else:
            return "uncertain_3p_other_aux"

    # 14. 3rd person ROOT (could be overt subject we miss, or gold gap)
    if dep == "ROOT" and person == "3":
        return "uncertain_3p_root"

    # 15. 3rd person subordinate clause
    if dep in ("advcl", "ccomp") and person == "3":
        return "uncertain_3p_subordinate"

    # 16. 3rd person conj
    if dep == "conj" and person == "3":
        return "uncertain_3p_conj"

    # 17. 3rd person in relative clause (object relative, no rel_pronoun_is_subject)
    if dep in ("acl:relcl", "acl") and person == "3":
        return "uncertain_3p_relcl"

    # 18. 1st/2nd person other dep labels
    if person in ("1", "2"):
        return "gold_gap_1p2p_other_dep"

    # 19. 3rd person other dep labels
    if person == "3":
        return "uncertain_3p_other_dep"

    # Fallback
    return "unclassified"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw features and labels
    X_raw = pd.read_parquet(DATA_DIR / "features.parquet")
    y = np.load(DATA_DIR / "labels.npy", allow_pickle=True)

    # Reproduce the exact same split
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
    cat_cols = _identify_categorical_columns(X_test_raw)
    X_test_enc = X_test_raw.copy()
    if cat_cols:
        X_test_enc[cat_cols] = encoder.transform(X_test_raw[cat_cols].astype(str))
    X_test_enc = X_test_enc.fillna(-1)

    # Predict
    preds = dt_model.predict(X_test_enc[tree_feature_names])

    # Identify FPs
    fp_mask = (preds == 1) & (y_test_bin == 0)
    fp_df = X_test_raw[fp_mask].copy()
    fp_df["gold_label"] = y_test[fp_mask]

    # Categorize each FP
    fp_df["category"] = fp_df.apply(categorize_fp, axis=1)

    # Build summary
    cats = fp_df["category"].value_counts()

    # Group into supercategories
    supercat = {}
    for cat, count in cats.items():
        if cat.startswith("genuine_"):
            supercat.setdefault("genuine_fp", 0)
            supercat["genuine_fp"] += count
        elif cat.startswith("gold_gap_"):
            supercat.setdefault("gold_data_gap", 0)
            supercat["gold_data_gap"] += count
        elif cat.startswith("uncertain_"):
            supercat.setdefault("uncertain", 0)
            supercat["uncertain"] += count
        else:
            supercat.setdefault("other", 0)
            supercat["other"] += count

    report = {
        "total_fps": int(fp_mask.sum()),
        "supercategory_counts": supercat,
        "supercategory_pcts": {
            k: round(100 * v / fp_mask.sum(), 1) for k, v in supercat.items()
        },
        "category_counts": cats.to_dict(),
        "category_pcts": {
            k: round(100 * v / fp_mask.sum(), 1) for k, v in cats.items()
        },
    }

    # Save
    fp_df.to_csv(OUTPUT_DIR / "fps_categorized.csv", index=False)
    with open(OUTPUT_DIR / "category_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print
    print(f"Total FPs: {fp_mask.sum()}\n")
    print("=== Supercategories ===")
    for k, v in sorted(supercat.items(), key=lambda x: -x[1]):
        pct = 100 * v / fp_mask.sum()
        print(f"  {k:25s} {v:5d}  ({pct:5.1f}%)")

    print(f"\n=== Detailed categories ===")
    for cat, count in cats.items():
        pct = 100 * count / fp_mask.sum()
        print(f"  {cat:45s} {count:5d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
