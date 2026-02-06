"""
Annotation validation metrics for pronoun recovery.

Compares LLM-generated annotations against a manually curated gold standard,
computing precision, recall, F1, and feature (label) accuracy for
pronoun-insertion markers.

Each annotation item is a dict with:
    - 'id': unique identifier for the text sequence
    - 'markers': list of dicts, each containing:
        - 'label': str (e.g. 'PRO.3sg', 'IMP')
        - 'position': int (token index where insertion was predicted)
        - 'lexical_form': Optional[str] (surface pronoun, may be None)

Matching logic:
    A predicted marker matches a gold marker when they share the same
    document id and the predicted position is within +/- position_tolerance
    tokens of the gold position.  Among position-matched pairs, a marker
    counts as label-correct when the predicted label string is identical to
    the gold label.

Edge-case conventions:
    - Empty predictions AND empty gold  ->  P=1, R=1 (vacuously correct)
    - Empty predictions, non-empty gold ->  P=0, R=0
    - Non-empty predictions, empty gold ->  P=0 (all false positives), R=1
    - No position-matched markers       ->  feature_accuracy=0.0
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AnnotationMetrics:
    """Container for annotation quality metrics.

    Attributes:
        precision: Fraction of predicted insertions that matched a gold insertion.
        recall: Fraction of gold insertions recovered by predictions.
        f1: Harmonic mean of precision and recall.
        feature_accuracy: Fraction of position-matched insertions whose label
            also matched the gold label.
        total_predicted: Total number of predicted insertion markers.
        total_gold: Total number of gold insertion markers.
        correct_insertions: Number of predicted markers that matched a gold
            marker by position (within tolerance).
        correct_features: Number of position-matched markers whose label was
            also correct.
    """

    precision: float
    recall: float
    f1: float
    feature_accuracy: float
    total_predicted: int
    total_gold: int
    correct_insertions: int
    correct_features: int


def compute_annotation_metrics(
    predicted: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
    position_tolerance: int = 2,
) -> AnnotationMetrics:
    """Compare predicted annotations against a gold standard.

    Args:
        predicted: List of annotation dicts, each with 'id' and 'markers'.
        gold: List of gold-standard annotation dicts with the same schema.
        position_tolerance: Maximum absolute token-position difference for
            two markers to be considered a positional match.

    Returns:
        An :class:`AnnotationMetrics` instance summarising quality.
    """
    # Index gold annotations by id for O(1) lookup.
    gold_by_id: Dict[str, List[Dict]] = {}
    for item in gold:
        item_id = item["id"]
        markers = item.get("markers", [])
        if item_id in gold_by_id:
            logger.warning("Duplicate gold id '%s'; extending marker list.", item_id)
            gold_by_id[item_id].extend(markers)
        else:
            gold_by_id[item_id] = list(markers)

    total_predicted = 0
    total_gold = sum(len(m) for m in gold_by_id.values())
    total_position_matched = 0
    total_label_matched = 0

    matched_gold_ids_global: Dict[str, set] = {
        item_id: set() for item_id in gold_by_id
    }

    for item in predicted:
        item_id = item["id"]
        pred_markers = item.get("markers", [])
        total_predicted += len(pred_markers)

        gold_markers = gold_by_id.get(item_id)
        if gold_markers is None:
            # Predicted markers for an id absent in gold are all false positives.
            if pred_markers:
                logger.debug(
                    "Predicted %d markers for id '%s' which has no gold entry.",
                    len(pred_markers),
                    item_id,
                )
            continue

        position_matched, label_matched = _match_markers(
            pred_markers,
            gold_markers,
            tolerance=position_tolerance,
            already_matched=matched_gold_ids_global.get(item_id, set()),
        )
        total_position_matched += position_matched
        total_label_matched += label_matched

    # ── Compute aggregate metrics ────────────────────────────────────────
    if total_predicted == 0 and total_gold == 0:
        # Vacuously correct: nothing to predict, nothing predicted.
        precision = 1.0
        recall = 1.0
    elif total_predicted == 0:
        # No predictions but there are gold markers: total miss.
        precision = 0.0
        recall = 0.0
    elif total_gold == 0:
        # Predictions exist but no gold markers: all false positives.
        precision = 0.0
        recall = 1.0
    else:
        precision = total_position_matched / total_predicted
        recall = total_position_matched / total_gold

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    feature_accuracy = (
        total_label_matched / total_position_matched
        if total_position_matched > 0
        else 0.0
    )

    metrics = AnnotationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        feature_accuracy=feature_accuracy,
        total_predicted=total_predicted,
        total_gold=total_gold,
        correct_insertions=total_position_matched,
        correct_features=total_label_matched,
    )

    logger.info(
        "Annotation metrics: P=%.3f  R=%.3f  F1=%.3f  feat_acc=%.3f  "
        "(pred=%d  gold=%d  pos_match=%d  label_match=%d)",
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.feature_accuracy,
        metrics.total_predicted,
        metrics.total_gold,
        metrics.correct_insertions,
        metrics.correct_features,
    )

    return metrics


def _match_markers(
    pred_markers: List[Dict],
    gold_markers: List[Dict],
    tolerance: int,
    already_matched: Optional[set] = None,
) -> Tuple[int, int]:
    """Match predicted markers to gold markers within position tolerance.

    Uses greedy closest-first matching: for each predicted marker (sorted by
    position), find the nearest unmatched gold marker whose position is within
    ``tolerance`` tokens.  This avoids suboptimal pairings that can occur with
    naive iteration order.

    Args:
        pred_markers: List of predicted marker dicts with at least 'position'
            and 'label' keys.
        gold_markers: List of gold marker dicts with the same schema.
        tolerance: Maximum absolute difference in token position for a match.
        already_matched: Optional set of gold marker indices (into
            *gold_markers*) that have already been claimed by earlier calls
            for the same document.  Updated in-place.

    Returns:
        A tuple ``(num_position_matched, num_label_matched)``.
    """
    if already_matched is None:
        already_matched = set()

    if not pred_markers or not gold_markers:
        return 0, 0

    # Build candidate pairs: (absolute distance, pred_idx, gold_idx).
    candidate_pairs: List[Tuple[int, int, int]] = []
    for p_idx, pm in enumerate(pred_markers):
        pred_pos = pm["position"]
        for g_idx, gm in enumerate(gold_markers):
            distance = abs(pred_pos - gm["position"])
            if distance <= tolerance:
                candidate_pairs.append((distance, p_idx, g_idx))

    # Sort by distance (greedy closest-first).
    candidate_pairs.sort(key=lambda t: t[0])

    matched_pred: set = set()
    position_matched = 0
    label_matched = 0

    for distance, p_idx, g_idx in candidate_pairs:
        if p_idx in matched_pred or g_idx in already_matched:
            continue

        # Position match found.
        matched_pred.add(p_idx)
        already_matched.add(g_idx)
        position_matched += 1

        # Check label match.
        if pred_markers[p_idx].get("label") == gold_markers[g_idx].get("label"):
            label_matched += 1

    return position_matched, label_matched
