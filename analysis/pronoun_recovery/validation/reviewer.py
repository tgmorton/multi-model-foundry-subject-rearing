"""
Human review workflow for pronoun-recovery annotation validation.

Provides stratified sampling of LLM annotations for manual review,
JSONL-based review sheet I/O, and quality-gate checking against the
thresholds defined in :class:`~analysis.pronoun_recovery.config.ValidationConfig`.

Review strata
-------------
1. **random** -- uniform random sample (default 50 % of review budget).
2. **high_insertion** -- items with the most insertion markers, which carry
   the highest risk of cascading annotation error (default 25 %).
3. **flagged** -- items where the LLM annotator flagged potential issues,
   indicated by a truthy ``'flagged'`` field (default 25 %).

If a stratum cannot fill its quota (e.g. no items are flagged), the shortfall
is redistributed proportionally to the other strata.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:
    from .metrics import AnnotationMetrics

from ..config import ValidationConfig

logger = logging.getLogger(__name__)


class HumanReviewer:
    """Manages stratified sampling, serialisation, and gate-checking for
    human review of LLM-generated pronoun-recovery annotations.

    Args:
        config: A :class:`ValidationConfig` instance specifying sample sizes,
            stratum fractions, and quality-gate thresholds.
    """

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample_for_review(
        self, annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select a stratified sample of annotations for human review.

        Sampling proceeds in three passes:

        1. **high_insertion** stratum -- items sorted descending by number of
           markers; the top *n* are drawn (without randomisation so the
           riskiest items are always included).
        2. **flagged** stratum -- items whose ``'flagged'`` field is truthy,
           sampled randomly.
        3. **random** stratum -- uniform random draw from remaining items.

        If any stratum cannot fill its quota, the surplus budget is
        redistributed proportionally among the other strata that still have
        eligible items.  Duplicates across strata are prevented.

        Args:
            annotations: Full list of LLM annotation dicts.  Each must have
                at least ``'id'``, ``'text'``, ``'annotated_text'``, and
                ``'markers'`` keys.  Optional: ``'flagged'`` (bool).

        Returns:
            A list of review-item dicts, each containing:
                - ``id``: annotation identifier
                - ``text``: original (unannotated) text
                - ``annotated_text``: LLM-annotated text
                - ``markers``: list of marker dicts from the annotation
                - ``review_stratum``: ``'random'``, ``'high_insertion'``,
                  or ``'flagged'``
                - ``corrected``: ``None`` (placeholder for human reviewer)
        """
        if not annotations:
            logger.warning("No annotations provided for review sampling.")
            return []

        total_budget = min(self.config.review_sample_size, len(annotations))
        if total_budget == 0:
            return []

        # Compute initial stratum quotas, rounding to ensure they sum to budget.
        quota_high = int(round(total_budget * self.config.high_insertion_fraction))
        quota_flagged = int(round(total_budget * self.config.flagged_fraction))
        quota_random = total_budget - quota_high - quota_flagged
        # Guard against negative quota from rounding.
        if quota_random < 0:
            quota_random = 0
            # Re-balance high and flagged to fit within total_budget.
            overflow = quota_high + quota_flagged - total_budget
            reduction_high = min(overflow, quota_high)
            quota_high -= reduction_high
            overflow -= reduction_high
            quota_flagged -= overflow

        selected_ids: Set[str] = set()
        review_items: List[Dict[str, Any]] = []

        # -- Stratum 1: high insertion count -----------------------------------
        # Sort descending by marker count so the riskiest items come first.
        sorted_by_markers = sorted(
            annotations,
            key=lambda a: len(a.get("markers", [])),
            reverse=True,
        )
        high_insertion_items = self._draw_from_pool(
            sorted_by_markers,
            quota_high,
            selected_ids,
            stratum="high_insertion",
            shuffle=False,
        )
        review_items.extend(high_insertion_items)

        # -- Stratum 2: flagged ------------------------------------------------
        flagged_pool = [a for a in annotations if a.get("flagged")]
        flagged_items = self._draw_from_pool(
            flagged_pool,
            quota_flagged,
            selected_ids,
            stratum="flagged",
            shuffle=True,
        )
        review_items.extend(flagged_items)

        # -- Redistribution of shortfall ---------------------------------------
        shortfall = total_budget - len(review_items)
        if shortfall > 0:
            # Anything not yet selected is eligible for random stratum (or
            # redistribution from under-filled strata).
            quota_random = shortfall

        # -- Stratum 3: random -------------------------------------------------
        random_items = self._draw_from_pool(
            annotations,
            quota_random,
            selected_ids,
            stratum="random",
            shuffle=True,
        )
        review_items.extend(random_items)

        logger.info(
            "Sampled %d items for review: %d high_insertion, %d flagged, %d random.",
            len(review_items),
            len(high_insertion_items),
            len(flagged_items),
            len(random_items),
        )

        return review_items

    # ── I/O ──────────────────────────────────────────────────────────────

    def save_review_sheet(
        self, review_items: List[Dict], output_path: Path
    ) -> None:
        """Persist review items as a JSONL file for a human reviewer.

        Each line is a self-contained JSON object.  The reviewer is expected
        to fill in the ``'corrected'`` field and return the file.

        Args:
            review_items: Items produced by :meth:`sample_for_review`.
            output_path: Destination ``.jsonl`` path.  Parent directories are
                created automatically.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            for item in review_items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            "Saved %d review items to %s.", len(review_items), output_path
        )

    def load_completed_reviews(
        self, review_path: Path
    ) -> List[Dict[str, Any]]:
        """Load a completed review sheet where human reviewers have filled in
        the ``'corrected'`` fields.

        Args:
            review_path: Path to the completed ``.jsonl`` file.

        Returns:
            List of review-item dicts with populated ``'corrected'`` values.

        Raises:
            FileNotFoundError: If *review_path* does not exist.
            ValueError: If the file contains malformed JSON lines.
        """
        review_path = Path(review_path)
        items: List[Dict[str, Any]] = []
        malformed_lines = 0

        with open(review_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed_lines += 1
                    logger.error(
                        "Malformed JSON on line %d of %s: %s",
                        line_no,
                        review_path,
                        exc,
                    )
                    continue
                items.append(item)

        if malformed_lines:
            raise ValueError(
                f"Found {malformed_lines} malformed JSON line(s) in "
                f"{review_path}. See log for details."
            )

        completed = sum(
            1 for it in items if it.get("corrected") is not None
        )
        logger.info(
            "Loaded %d review items from %s (%d completed).",
            len(items),
            review_path,
            completed,
        )

        return items

    # ── Quality gates ────────────────────────────────────────────────────

    def check_gates(self, metrics: AnnotationMetrics) -> Dict[str, bool]:
        """Evaluate annotation quality metrics against configured thresholds.

        Args:
            metrics: An :class:`AnnotationMetrics` instance (typically
                computed by :func:`compute_annotation_metrics`).

        Returns:
            A dict with boolean results:
                - ``precision_pass``: ``metrics.precision >= config.min_precision``
                - ``recall_pass``: ``metrics.recall >= config.min_recall``
                - ``feature_accuracy_pass``: ``metrics.feature_accuracy >= config.min_feature_accuracy``
                - ``all_pass``: ``True`` iff all individual gates pass.
        """
        precision_pass = metrics.precision >= self.config.min_precision
        recall_pass = metrics.recall >= self.config.min_recall
        feature_accuracy_pass = (
            metrics.feature_accuracy >= self.config.min_feature_accuracy
        )
        all_pass = precision_pass and recall_pass and feature_accuracy_pass

        gate_results = {
            "precision_pass": precision_pass,
            "recall_pass": recall_pass,
            "feature_accuracy_pass": feature_accuracy_pass,
            "all_pass": all_pass,
        }

        if all_pass:
            logger.info("All quality gates passed: %s", gate_results)
        else:
            logger.warning(
                "Quality gate FAILURE: P=%.3f (min=%.2f, %s)  "
                "R=%.3f (min=%.2f, %s)  feat_acc=%.3f (min=%.2f, %s)",
                metrics.precision,
                self.config.min_precision,
                "PASS" if precision_pass else "FAIL",
                metrics.recall,
                self.config.min_recall,
                "PASS" if recall_pass else "FAIL",
                metrics.feature_accuracy,
                self.config.min_feature_accuracy,
                "PASS" if feature_accuracy_pass else "FAIL",
            )

        return gate_results

    # ── Internal helpers ─────────────────────────────────────────────────

    def _draw_from_pool(
        self,
        pool: List[Dict[str, Any]],
        quota: int,
        selected_ids: Set[str],
        stratum: str,
        shuffle: bool,
    ) -> List[Dict[str, Any]]:
        """Draw up to *quota* items from *pool*, skipping already-selected ids.

        Args:
            pool: Candidate annotation dicts.
            quota: Maximum number of items to draw.
            selected_ids: Set of ids already selected (updated in-place).
            stratum: Label for the ``review_stratum`` field.
            shuffle: If ``True``, shuffle eligible items before drawing.  If
                ``False``, items are drawn in their existing order (useful
                for the high-insertion stratum where order is meaningful).

        Returns:
            List of review-item dicts drawn from this stratum.
        """
        eligible = [a for a in pool if a["id"] not in selected_ids]

        if shuffle:
            # Copy to avoid mutating the caller's list order.
            eligible = list(eligible)
            random.shuffle(eligible)

        drawn: List[Dict[str, Any]] = []
        for annotation in eligible:
            if len(drawn) >= quota:
                break
            item_id = annotation["id"]
            if item_id in selected_ids:
                # Defensive: should already be filtered, but guard anyway.
                continue

            selected_ids.add(item_id)
            drawn.append(
                self._build_review_item(annotation, stratum=stratum)
            )

        if len(drawn) < quota:
            logger.debug(
                "Stratum '%s' shortfall: requested %d, got %d.",
                stratum,
                quota,
                len(drawn),
            )

        return drawn

    @staticmethod
    def _build_review_item(
        annotation: Dict[str, Any], stratum: str
    ) -> Dict[str, Any]:
        """Construct a review-item dict from an annotation and stratum label.

        Args:
            annotation: Source annotation dict.
            stratum: One of ``'random'``, ``'high_insertion'``, ``'flagged'``.

        Returns:
            A dict ready for serialisation and human review.
        """
        return {
            "id": annotation["id"],
            "text": annotation.get("text", ""),
            "annotated_text": annotation.get("annotated_text", ""),
            "markers": annotation.get("markers", []),
            "review_stratum": stratum,
            "corrected": None,
        }
