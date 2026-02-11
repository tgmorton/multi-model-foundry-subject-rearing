"""
Pilot validation: statistics and spot-check sampling.

Computes label distributions, confidence breakdowns, and skip-reason
summaries from a pilot run.  Samples random records for human review.
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .quality_filters import FilterStats

logger = logging.getLogger(__name__)


def compute_pilot_statistics(
    records: List[Dict[str, Any]],
    stats: FilterStats,
) -> Dict[str, Any]:
    """Compute summary statistics for a pilot run.

    Args:
        records: List of output checkpoint records (each has clean_text,
            markers, id, genre, source).
        stats: Accumulated filter statistics from the run.

    Returns:
        Dict with label_distribution, confidence_breakdown,
        skip_reasons, and overall counts.
    """
    label_counts: Counter = Counter()
    confidence_counts: Counter = Counter()
    markers_per_record: List[int] = []

    for rec in records:
        markers = rec.get("markers", [])
        markers_per_record.append(len(markers))
        for m in markers:
            label_counts[m.get("label", "UNKNOWN")] += 1
            confidence_counts[m.get("confidence", "unknown")] += 1

    total_markers = sum(markers_per_record)
    records_with_markers = sum(1 for n in markers_per_record if n > 0)

    return {
        "total_records": len(records),
        "records_with_markers": records_with_markers,
        "total_markers": total_markers,
        "avg_markers_per_record": (
            total_markers / records_with_markers if records_with_markers else 0
        ),
        "label_distribution": dict(label_counts.most_common()),
        "confidence_breakdown": dict(confidence_counts.most_common()),
        "filter_stats": stats.summary(),
    }


def sample_for_spot_check(
    records: List[Dict[str, Any]],
    n: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample random records for human spot-check review.

    Only samples records that have at least one marker.

    Args:
        records: All output records from the pilot.
        n: Number of samples to draw.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled records.
    """
    with_markers = [r for r in records if r.get("markers")]
    rng = random.Random(seed)
    n = min(n, len(with_markers))
    return rng.sample(with_markers, n)


def save_validation_report(
    pilot_stats: Dict[str, Any],
    spot_check_samples: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write pilot statistics and spot-check samples to disk.

    Args:
        pilot_stats: Output of compute_pilot_statistics().
        spot_check_samples: Output of sample_for_spot_check().
        output_dir: Directory to write reports to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "pilot_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(pilot_stats, f, indent=2, ensure_ascii=False)
    logger.info("Pilot statistics saved to %s", stats_path)

    spot_path = output_dir / "spot_check_samples.jsonl"
    with open(spot_path, "w", encoding="utf-8") as f:
        for rec in spot_check_samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Spot-check samples (%d) saved to %s", len(spot_check_samples), spot_path)
