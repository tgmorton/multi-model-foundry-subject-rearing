"""
Output formatters for corpus descriptive analysis.

Produces JSON (per-analyzer + combined) and CSV (flat, R-friendly) output.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def save_results(
    results: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path,
    split_name: str,
) -> None:
    """
    Save all analyzer results as JSON and CSV.

    Args:
        results: {analyzer_name: analyzer.get_results()} mapping
        metadata: Processing metadata dict
        output_dir: Output directory
        split_name: Split identifier (added as column in CSVs)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-analyzer JSON
    for name, data in results.items():
        json_path = output_dir / f"{name}.json"
        json_path.write_text(json.dumps(data, indent=2, default=str))

    # Combined results JSON
    combined_path = output_dir / "results.json"
    combined_path.write_text(json.dumps(results, indent=2, default=str))

    # Metadata JSON
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    # Per-analyzer CSV
    for name, data in results.items():
        csv_path = output_dir / f"{name}.csv"
        _write_analyzer_csv(csv_path, name, data, split_name)


def _write_analyzer_csv(
    path: Path,
    analyzer_name: str,
    data: Dict[str, Any],
    split_name: str,
) -> None:
    """Write a single analyzer's results as a flat CSV with genre column."""
    rows = _flatten_results(analyzer_name, data, split_name)
    if not rows:
        return

    # Collect all unique fieldnames across all rows (heterogeneous dicts)
    seen = set()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_results(
    analyzer_name: str,
    data: Dict[str, Any],
    split_name: str,
) -> List[Dict[str, Any]]:
    """
    Flatten nested analyzer results into flat rows suitable for CSV.

    Each row gets a 'split' and 'genre' column.
    """
    rows: List[Dict[str, Any]] = []

    # Overall
    overall = data.get("overall", {})
    rows.extend(_flatten_section(overall, "overall", split_name, analyzer_name))

    # By genre
    by_genre = data.get("by_genre", {})
    for genre, genre_data in by_genre.items():
        rows.extend(_flatten_section(genre_data, genre, split_name, analyzer_name))

    return rows


def _flatten_section(
    section: Dict[str, Any],
    genre: str,
    split_name: str,
    analyzer_name: str,
) -> List[Dict[str, Any]]:
    """Flatten one section (overall or a specific genre) into rows."""
    rows: List[Dict[str, Any]] = []

    for key, value in section.items():
        if isinstance(value, list):
            # List of dicts — each becomes a row
            for item in value:
                if isinstance(item, dict):
                    row = {"split": split_name, "genre": genre}
                    row.update(item)
                    rows.append(row)
        elif isinstance(value, dict):
            # Single dict — one row
            row = {"split": split_name, "genre": genre, "metric": key}
            row.update(value)
            rows.append(row)
        else:
            # Scalar — one row
            rows.append({
                "split": split_name,
                "genre": genre,
                "metric": key,
                "value": value,
            })

    return rows
