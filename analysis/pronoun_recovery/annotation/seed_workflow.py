"""
Seed annotation workflow: select diverse candidates and export/import
editable YAML files for manual annotation.

The annotator adds bracket-format markers to the ``annotated`` field::

    [PRO.3sg:lei] Parla italiano.
    [PRO.1pl] siamo a posto.

Tags use the closed label set (PRO.1sg, PRO.2sg, PRO.3sg, PRO.1pl,
PRO.2pl, PRO.3pl, IMP, CONJ) with an optional ``:lexical_form`` suffix.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .manual_format import parse_annotations, validate_annotation

logger = logging.getLogger(__name__)


def select_seed_candidates(
    candidates_path: Path,
    n: int = 15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select a genre-diverse subset of candidates for manual annotation.

    Guarantees at least one candidate per genre (if available), then
    fills remaining slots proportionally.

    Args:
        candidates_path: Path to candidates.jsonl.
        n: Number of seed candidates to select.
        seed: Random seed.

    Returns:
        List of candidate dicts.
    """
    with open(candidates_path, encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f]

    rng = random.Random(seed)

    # Group by genre
    by_genre: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_genre.setdefault(c["genre"], []).append(c)

    selected: List[Dict[str, Any]] = []

    # First pass: one from each genre
    for genre in sorted(by_genre):
        pick = rng.choice(by_genre[genre])
        selected.append(pick)
        by_genre[genre].remove(pick)

    # Second pass: fill remaining slots proportionally
    remaining_n = n - len(selected)
    if remaining_n > 0:
        pool = [c for cands in by_genre.values() for c in cands]
        if pool:
            extra = rng.sample(pool, min(remaining_n, len(pool)))
            selected.extend(extra)

    # Sort by genre then id for readability
    selected.sort(key=lambda c: (c["genre"], c["id"]))
    return selected


def export_seed_yaml(candidates: List[Dict[str, Any]], output_path: Path) -> None:
    """Write candidates to an editable YAML file for manual annotation.

    Each entry has ``id``, ``genre``, ``text`` (read-only), and an
    ``annotated`` field initialized as a copy of ``text`` for the
    annotator to edit.
    """
    entries = []
    for c in candidates:
        entries.append({
            "id": c["id"],
            "genre": c["genre"],
            "text": c["text"],
            "annotated": c["text"],  # annotator edits this
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Pronoun Recovery — Manual Seed Annotations\n")
        f.write("# \n")
        f.write("# For each entry, edit the 'annotated' field to insert\n")
        f.write("# pronoun markers where a subject pronoun has been dropped.\n")
        f.write("# \n")
        f.write("# Marker format:  [PRO.<label>:<form>]\n")
        f.write("# Labels: PRO.1sg, PRO.2sg, PRO.3sg, PRO.1pl, PRO.2pl, PRO.3pl, IMP, CONJ\n")
        f.write("# Examples:\n")
        f.write("#   [PRO.3sg:lei] Parla italiano.    (she speaks Italian)\n")
        f.write("#   [PRO.1pl] siamo a posto.         (we are all set)\n")
        f.write("#   [PRO.1sg:I] went to the store.   (English)\n")
        f.write("# \n")
        f.write("# Leave 'annotated' unchanged if no pronouns are dropped.\n")
        f.write("# Do NOT edit the 'id', 'genre', or 'text' fields.\n")
        f.write("#\n\n")

        yaml.dump(
            entries,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    logger.info("Wrote %d seed candidates to %s", len(entries), output_path)


def import_seed_yaml(
    yaml_path: Path,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """Read completed seed annotations from YAML and validate them.

    Returns a list of dicts with keys: ``id``, ``genre``, ``text``,
    ``annotated``, ``markers`` (parsed annotation markers).

    Raises ValueError if any annotations have validation errors.
    """
    with open(yaml_path, encoding="utf-8") as f:
        entries = yaml.safe_load(f)

    results: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    for entry in entries:
        eid = entry["id"]
        annotated = entry["annotated"]
        clean_text, markers = parse_annotations(annotated)

        # Validate each marker
        entry_errors = []
        for marker in markers:
            errs = validate_annotation(marker, language=language)
            entry_errors.extend(f"{eid}: {e}" for e in errs)

        if entry_errors:
            all_errors.extend(entry_errors)

        results.append({
            "id": eid,
            "genre": entry["genre"],
            "text": entry["text"],
            "annotated": annotated,
            "markers": [
                {
                    "label": m.label,
                    "lexical_form": m.lexical_form,
                    "position": m.position,
                }
                for m in markers
            ],
        })

    if all_errors:
        error_msg = "Validation errors in seed annotations:\n" + "\n".join(
            f"  - {e}" for e in all_errors
        )
        raise ValueError(error_msg)

    n_with_markers = sum(1 for r in results if r["markers"])
    logger.info(
        "Imported %d seed annotations (%d with markers, %d unchanged)",
        len(results),
        n_with_markers,
        len(results) - n_with_markers,
    )
    return results


def save_seed_jsonl(annotations: List[Dict[str, Any]], output_path: Path) -> None:
    """Save validated seed annotations as JSONL for the LLM annotator."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")
    logger.info("Saved %d seed annotations to %s", len(annotations), output_path)
