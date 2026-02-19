"""
Apply trained tree model to new Italian text.

TreeNullSubjectDetector loads a saved model + encoder and produces
predictions compatible with the existing neural model interface.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import spacy

from analysis.pronoun_recovery.constants import MORPH_TO_LABEL_SUFFIX

from .feature_extractor import FEATURE_NAMES, ItalianVerbFeatureExtractor

logger = logging.getLogger(__name__)


class TreeNullSubjectDetector:
    """Rule-based null subject detector using a trained decision tree.

    Two-stage architecture:
    1. Binary detection: tree model predicts PRO (1) vs NONE (0).
    2. Person/number: read directly from Italian verb morphology.
    """

    def __init__(
        self,
        model_dir: Path,
        classifier_type: str = "decision_tree",
        spacy_model: str = "it_core_news_lg",
        nlp: Optional[spacy.language.Language] = None,
    ):
        """Load model, encoder, and spaCy.

        Args:
            model_dir: Directory containing saved model files.
            classifier_type: Which model to use ("decision_tree" or
                "gradient_boosted").
            spacy_model: Name of Italian spaCy model.
            nlp: Pre-loaded spaCy model (avoids reloading).
        """
        self.model_dir = Path(model_dir)

        # Load model
        if classifier_type == "gradient_boosted":
            model_path = self.model_dir / "hgb_model.joblib"
        else:
            model_path = self.model_dir / "dt_model.joblib"

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(self.model_dir / "feature_encoder.joblib")

        # Load pipeline config (prefilter + column exclusion)
        config_path = self.model_dir / "pipeline_config.joblib"
        if config_path.exists():
            self.pipeline_config = joblib.load(config_path)
        else:
            # Backward compat: no prefilter, use all features
            self.pipeline_config = {
                "prefilter_col": None,
                "detection_exclude_cols": [],
                "tree_feature_names": FEATURE_NAMES,
            }

        logger.info("Loaded %s model from %s", classifier_type, model_path)

        # spaCy
        if nlp is not None:
            self.nlp = nlp
        else:
            self.nlp = spacy.load(spacy_model)

        # Feature extractor
        self.extractor = ItalianVerbFeatureExtractor(language="it")

    def predict_text(self, text: str) -> List[Dict[str, Any]]:
        """Predict null subjects in Italian text.

        Args:
            text: Italian text string.

        Returns:
            List of prediction dicts, one per detected null subject verb.
        """
        doc = self.nlp(text)
        return self.predict_doc(doc)

    def predict_doc(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Predict null subjects in a pre-parsed spaCy Doc.

        Args:
            doc: Parsed Italian spaCy Doc.

        Returns:
            List of prediction dicts with keys: verb_text, verb_char_offset,
            label, confidence, lexical_form, token_idx, verb_lemma.
        """
        # Extract features
        verb_rows = self.extractor.extract_doc(doc, genre="unknown")

        if not verb_rows:
            return []

        # Build feature DataFrame
        feature_dicts = [row.features for row in verb_rows]
        X = pd.DataFrame(feature_dicts, columns=FEATURE_NAMES)

        # Encode categoricals
        cat_cols = [
            col for col in X.columns
            if X[col].dtype == object or X[col].dtype == bool
        ]
        X_encoded = X.copy()
        if cat_cols and hasattr(self.encoder, "categories_") and len(self.encoder.categories_) > 0:
            X_encoded[cat_cols] = self.encoder.transform(X[cat_cols].astype(str))
        X_encoded = X_encoded.fillna(-1)

        # Apply prefilter: verbs with reachable subject → NONE (pred=0)
        prefilter_col = self.pipeline_config.get("prefilter_col")
        tree_cols = self.pipeline_config.get("tree_feature_names", list(X_encoded.columns))

        preds = np.zeros(len(X_encoded), dtype=int)
        probas = np.zeros(len(X_encoded))

        if prefilter_col and prefilter_col in X_encoded.columns:
            tree_mask = ~(X_encoded[prefilter_col] > 0.5)
        else:
            tree_mask = pd.Series(True, index=X_encoded.index)

        if tree_mask.any():
            X_tree = X_encoded.loc[tree_mask, tree_cols]
            preds[tree_mask.values] = self.model.predict(X_tree)
            if hasattr(self.model, "predict_proba"):
                probas[tree_mask.values] = self.model.predict_proba(X_tree)[:, 1]
            else:
                probas[tree_mask.values] = preds[tree_mask.values].astype(float)

        # Build output for detected null subjects
        results = []
        for row, pred, proba in zip(verb_rows, preds, probas):
            if pred != 1:
                continue

            # Stage 2: person/number from morphology
            if row.morph_label_suffix:
                label = f"PRO.{row.morph_label_suffix}"
            else:
                label = "PRO.3sg"  # fallback

            # Look up default lexical form
            from analysis.pronoun_recovery.constants import IT_DEFAULT_PRONOUN
            lexical_form = IT_DEFAULT_PRONOUN.get(label, "")

            results.append({
                "verb_text": row.verb_text,
                "verb_char_offset": row.char_offset,
                "token_idx": row.token_idx,
                "verb_lemma": row.verb_lemma,
                "label": label,
                "confidence": float(proba),
                "lexical_form": lexical_form,
            })

        return results
