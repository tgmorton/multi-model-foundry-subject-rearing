"""JSONL export in markers format for the pronoun recovery pipeline."""

from typing import Any, Dict, List

from . import db


def _confidence_label(score: int) -> str:
    """Map 1-5 confidence to high/medium/low."""
    if score >= 4:
        return "high"
    elif score >= 2:
        return "medium"
    return "low"


def _word_gap_to_char_offset(text: str, gap_index: int) -> int:
    """Convert a word-gap index to a character offset in the text.

    Gap 0 = before word 0 -> offset 0
    Gap N = between word N-1 and word N -> offset of the start of word N
    """
    words = text.split()
    if gap_index <= 0:
        return 0
    if gap_index >= len(words):
        return len(text)

    offset = 0
    for i in range(gap_index):
        offset = text.index(words[i], offset) + len(words[i])
        # Skip whitespace to find start of next word
        while offset < len(text) and text[offset] == " ":
            offset += 1
    return offset


def _build_record(
    text: str,
    source: str,
    stimulus_id: int,
    has_null_subject: bool,
    pronouns: List[Dict],
    confidence: int,
    agreement: str,
    metadata: Any = None,
) -> Dict:
    markers = []
    if has_null_subject:
        for p in pronouns:
            char_offset = _word_gap_to_char_offset(text, p["position"])
            markers.append({
                "label": p["label"],
                "lexical_form": p["lexical_form"],
                "position": char_offset,
                "confidence": _confidence_label(p.get("confidence", confidence)),
            })

    record = {
        "clean_text": text,
        "markers": markers,
        "source": "human_annotation",
        "language": "es",
        "genre": source or "unknown",
        "stimulus_id": stimulus_id,
        "annotator_agreement": agreement,
    }
    return record


def export_gold(db_path=None) -> List[Dict]:
    """Export final gold JSONL: adjudicated + agreed items."""
    records = []

    with db.get_conn(db_path) as conn:
        # 1) Adjudicated stimuli
        adj_rows = conn.execute(
            """
            SELECT adj.*, s.text, s.source, s.metadata
            FROM adjudications adj
            JOIN stimuli s ON s.id = adj.stimulus_id
            ORDER BY s.ordering
            """
        ).fetchall()

        for adj in adj_rows:
            pronouns = conn.execute(
                "SELECT * FROM adjudication_pronouns WHERE adjudication_id = ? ORDER BY ordering",
                (adj["id"],),
            ).fetchall()
            records.append(_build_record(
                text=adj["text"],
                source=adj["source"],
                stimulus_id=adj["stimulus_id"],
                has_null_subject=bool(adj["has_null_subject"]),
                pronouns=[db.dict_row(p) for p in pronouns],
                confidence=adj["overall_confidence"],
                agreement="adjudicated",
            ))

        adjudicated_ids = {r["stimulus_id"] for r in records}

        # 2) Agreed stimuli (2 annotators agree on binary, not yet adjudicated)
        agreed_ids = set()
        stim_rows = conn.execute(
            """
            SELECT s.id, s.text, s.source, s.metadata
            FROM stimuli s
            WHERE s.id NOT IN ({})
            AND (SELECT COUNT(DISTINCT a.user_id)
                 FROM annotations a WHERE a.stimulus_id = s.id) >= 2
            ORDER BY s.ordering
            """.format(
                ",".join(str(i) for i in adjudicated_ids) if adjudicated_ids else "0"
            )
        ).fetchall()

        for stim in stim_rows:
            anns = conn.execute(
                """
                SELECT * FROM annotations
                WHERE stimulus_id = ? ORDER BY user_id LIMIT 2
                """,
                (stim["id"],),
            ).fetchall()

            if len(anns) < 2:
                continue

            a, b = anns[0], anns[1]
            if a["has_null_subject"] != b["has_null_subject"]:
                continue  # Disagreement, needs adjudication

            # Use first annotator's pronouns for agreed items
            pronouns = conn.execute(
                "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                (a["id"],),
            ).fetchall()

            records.append(_build_record(
                text=stim["text"],
                source=stim["source"],
                stimulus_id=stim["id"],
                has_null_subject=bool(a["has_null_subject"]),
                pronouns=[db.dict_row(p) for p in pronouns],
                confidence=a["overall_confidence"],
                agreement="full",
            ))
            agreed_ids.add(stim["id"])

        # 3) Single-annotator stimuli (1 completed annotation, not covered above)
        excluded_ids = adjudicated_ids | agreed_ids
        single_rows = conn.execute(
            """
            SELECT s.id, s.text, s.source, s.metadata
            FROM stimuli s
            JOIN annotations a ON a.stimulus_id = s.id
            WHERE a.has_null_subject IS NOT NULL
              AND s.id NOT IN ({})
            GROUP BY s.id
            HAVING COUNT(DISTINCT a.user_id) = 1
            ORDER BY s.ordering
            """.format(
                ",".join(str(i) for i in excluded_ids) if excluded_ids else "0"
            )
        ).fetchall()

        for stim in single_rows:
            ann = conn.execute(
                "SELECT * FROM annotations WHERE stimulus_id = ? "
                "AND has_null_subject IS NOT NULL LIMIT 1",
                (stim["id"],),
            ).fetchone()
            if not ann:
                continue

            pronouns = conn.execute(
                "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                (ann["id"],),
            ).fetchall()

            records.append(_build_record(
                text=stim["text"],
                source=stim["source"],
                stimulus_id=stim["id"],
                has_null_subject=bool(ann["has_null_subject"]),
                pronouns=[db.dict_row(p) for p in pronouns],
                confidence=ann["overall_confidence"],
                agreement="single",
            ))

    return records


def export_raw(db_path=None) -> List[Dict]:
    """Export all raw annotations."""
    records = []
    with db.get_conn(db_path) as conn:
        anns = conn.execute(
            """
            SELECT a.*, s.text, s.source, u.username
            FROM annotations a
            JOIN stimuli s ON s.id = a.stimulus_id
            JOIN users u ON u.id = a.user_id
            ORDER BY s.ordering, a.user_id
            """
        ).fetchall()

        for ann in anns:
            pronouns = conn.execute(
                "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                (ann["id"],),
            ).fetchall()
            record = {
                "stimulus_id": ann["stimulus_id"],
                "clean_text": ann["text"],
                "annotator": ann["username"],
                "has_null_subject": bool(ann["has_null_subject"]),
                "overall_confidence": ann["overall_confidence"],
                "starred": bool(ann["starred"]),
                "note": ann["note"],
                "pronouns": [
                    {
                        "position": p["position"],
                        "label": p["label"],
                        "lexical_form": p["lexical_form"],
                        "confidence": p["confidence"],
                    }
                    for p in pronouns
                ],
            }
            records.append(record)

    return records
