"""SQLite schema initialization and query helpers."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

DB_PATH = Path(__file__).parent / "data" / "annotation.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY,
    username     TEXT UNIQUE NOT NULL,
    token        TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT 'annotator',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stimuli (
    id             INTEGER PRIMARY KEY,
    text           TEXT NOT NULL,
    source         TEXT,
    context_before TEXT,
    context_after  TEXT,
    metadata       TEXT,
    ordering       INTEGER NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS annotations (
    id                 INTEGER PRIMARY KEY,
    stimulus_id        INTEGER NOT NULL REFERENCES stimuli(id),
    user_id            INTEGER NOT NULL REFERENCES users(id),
    has_null_subject   INTEGER,
    overall_confidence INTEGER NOT NULL DEFAULT 3,
    starred            INTEGER NOT NULL DEFAULT 0,
    note               TEXT DEFAULT '',
    context_expansions INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at         TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(stimulus_id, user_id)
);

CREATE TABLE IF NOT EXISTS placed_pronouns (
    id            INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    label         TEXT NOT NULL,
    lexical_form  TEXT NOT NULL,
    confidence    INTEGER NOT NULL DEFAULT 3,
    ordering      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS adjudications (
    id               INTEGER PRIMARY KEY,
    stimulus_id      INTEGER NOT NULL UNIQUE REFERENCES stimuli(id),
    adjudicator_id   INTEGER NOT NULL REFERENCES users(id),
    resolution       TEXT NOT NULL,
    accepted_user_id INTEGER REFERENCES users(id),
    has_null_subject INTEGER NOT NULL,
    overall_confidence INTEGER NOT NULL DEFAULT 3,
    note             TEXT DEFAULT '',
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS adjudication_pronouns (
    id              INTEGER PRIMARY KEY,
    adjudication_id INTEGER NOT NULL REFERENCES adjudications(id) ON DELETE CASCADE,
    position        INTEGER NOT NULL,
    label           TEXT NOT NULL,
    lexical_form    TEXT NOT NULL,
    confidence      INTEGER NOT NULL DEFAULT 3,
    ordering        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS annotation_history (
    id                 INTEGER PRIMARY KEY,
    stimulus_id        INTEGER NOT NULL REFERENCES stimuli(id),
    user_id            INTEGER NOT NULL REFERENCES users(id),
    has_null_subject   INTEGER,
    overall_confidence INTEGER NOT NULL DEFAULT 3,
    starred            INTEGER NOT NULL DEFAULT 0,
    note               TEXT DEFAULT '',
    context_expansions INTEGER NOT NULL DEFAULT 0,
    pronouns_json      TEXT NOT NULL DEFAULT '[]',
    saved_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables if they don't exist."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")


@contextmanager
def get_conn(db_path: Optional[Path] = None):
    """Yield a SQLite connection with row_factory=sqlite3.Row."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def dict_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)


def seed_users(
    users: List[Dict[str, str]], db_path: Optional[Path] = None
) -> None:
    """Insert users if they don't already exist.

    Each dict: {username, display_name, role}.
    Token is auto-generated.
    """
    with get_conn(db_path) as conn:
        for u in users:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (u["username"],)
            ).fetchone()
            if existing:
                continue
            token = str(uuid4())
            conn.execute(
                "INSERT INTO users (username, token, display_name, role) "
                "VALUES (?, ?, ?, ?)",
                (u["username"], token, u["display_name"], u.get("role", "annotator")),
            )


def get_user_by_token(token: str, db_path: Optional[Path] = None) -> Optional[Dict]:
    """Look up a user by bearer token."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE token = ?", (token,)
        ).fetchone()
        return dict_row(row) if row else None


def get_user_by_username(
    username: str, db_path: Optional[Path] = None
) -> Optional[Dict]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict_row(row) if row else None


def list_users(db_path: Optional[Path] = None) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [dict_row(r) for r in rows]


# ── Stimuli ──────────────────────────────────────────────────────────


def _normalize_context(val) -> Optional[str]:
    """Normalize context to a JSON array of sentences.

    Accepts: None, a string (single sentence), or a list of strings.
    Stores as JSON array string, nearest sentence first.
    """
    if val is None:
        return None
    if isinstance(val, list):
        return json.dumps(val, ensure_ascii=False) if val else None
    val = val.strip()
    return json.dumps([val], ensure_ascii=False) if val else None


def load_stimuli(
    items: List[Dict[str, Any]], db_path: Optional[Path] = None
) -> int:
    """Bulk-insert stimuli. Returns count inserted."""
    with get_conn(db_path) as conn:
        count = 0
        for i, item in enumerate(items):
            conn.execute(
                "INSERT INTO stimuli (text, source, context_before, context_after, "
                "metadata, ordering) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    item["text"],
                    item.get("source"),
                    _normalize_context(item.get("context_before")),
                    _normalize_context(item.get("context_after")),
                    json.dumps(item.get("metadata")) if item.get("metadata") else None,
                    item.get("ordering", i),
                ),
            )
            count += 1
        return count


def get_stimuli_page(
    user_id: int,
    page: int = 1,
    page_size: int = 50,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return paginated stimuli with annotation status and pronouns for user."""
    with get_conn(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM stimuli").fetchone()[0]
        offset = (page - 1) * page_size
        rows = conn.execute(
            """
            SELECT s.*, a.id AS annotation_id, a.has_null_subject, a.starred,
                   a.overall_confidence
            FROM stimuli s
            LEFT JOIN annotations a ON a.stimulus_id = s.id AND a.user_id = ?
            ORDER BY s.ordering
            LIMIT ? OFFSET ?
            """,
            (user_id, page_size, offset),
        ).fetchall()

        items = [dict_row(r) for r in rows]

        # Batch-fetch pronouns for all annotated items on this page
        ann_ids = [item["annotation_id"] for item in items
                   if item.get("annotation_id")]
        pronoun_map: Dict[int, List[Dict]] = {}
        if ann_ids:
            placeholders = ",".join("?" * len(ann_ids))
            pronoun_rows = conn.execute(
                f"SELECT annotation_id, lexical_form, label, position "
                f"FROM placed_pronouns "
                f"WHERE annotation_id IN ({placeholders}) "
                f"ORDER BY annotation_id, ordering",
                ann_ids,
            ).fetchall()
            for pr in pronoun_rows:
                aid = pr["annotation_id"]
                if aid not in pronoun_map:
                    pronoun_map[aid] = []
                pronoun_map[aid].append({
                    "lexical_form": pr["lexical_form"],
                    "label": pr["label"],
                    "position": pr["position"],
                })

        for item in items:
            aid = item.get("annotation_id")
            item["pronouns"] = pronoun_map.get(aid, []) if aid else []

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }


def get_stimulus(
    stimulus_id: int, user_id: int, db_path: Optional[Path] = None
) -> Optional[Dict]:
    """Get a stimulus with the current user's annotation and placed pronouns."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM stimuli WHERE id = ?", (stimulus_id,)
        ).fetchone()
        if not row:
            return None
        result = dict_row(row)

        ann = conn.execute(
            "SELECT * FROM annotations WHERE stimulus_id = ? AND user_id = ?",
            (stimulus_id, user_id),
        ).fetchone()
        if ann:
            result["annotation"] = dict_row(ann)
            pronouns = conn.execute(
                "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                (ann["id"],),
            ).fetchall()
            result["annotation"]["pronouns"] = [dict_row(p) for p in pronouns]
        else:
            result["annotation"] = None

        return result


def get_starred_stimuli(
    user_id: int, db_path: Optional[Path] = None
) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.text, s.ordering
            FROM stimuli s
            JOIN annotations a ON a.stimulus_id = s.id AND a.user_id = ?
            WHERE a.starred = 1
            ORDER BY s.ordering
            """,
            (user_id,),
        ).fetchall()
        return [dict_row(r) for r in rows]


# ── Annotations ──────────────────────────────────────────────────────


def upsert_annotation(
    stimulus_id: int,
    user_id: int,
    has_null_subject: Optional[bool],
    pronouns: List[Dict],
    overall_confidence: int = 3,
    starred: bool = False,
    note: str = "",
    context_expansions: int = 0,
    db_path: Optional[Path] = None,
) -> Dict:
    """Create or update an annotation with its placed pronouns."""
    has_null_int = int(has_null_subject) if has_null_subject is not None else None
    with get_conn(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM annotations WHERE stimulus_id = ? AND user_id = ?",
            (stimulus_id, user_id),
        ).fetchone()

        if existing:
            ann_id = existing["id"]
            conn.execute(
                """
                UPDATE annotations
                SET has_null_subject = ?, overall_confidence = ?, starred = ?,
                    note = ?, context_expansions = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (has_null_int, overall_confidence, int(starred), note,
                 context_expansions, ann_id),
            )
            conn.execute(
                "DELETE FROM placed_pronouns WHERE annotation_id = ?", (ann_id,)
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO annotations
                    (stimulus_id, user_id, has_null_subject, overall_confidence,
                     starred, note, context_expansions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (stimulus_id, user_id, has_null_int, overall_confidence,
                 int(starred), note, context_expansions),
            )
            ann_id = cursor.lastrowid

        # Record audit history
        pronouns_snapshot = json.dumps(
            [{"position": p["position"], "label": p["label"],
              "lexical_form": p["lexical_form"],
              "confidence": p.get("confidence", 3)}
             for p in pronouns],
            ensure_ascii=False,
        )
        conn.execute(
            """
            INSERT INTO annotation_history
                (stimulus_id, user_id, has_null_subject, overall_confidence,
                 starred, note, context_expansions, pronouns_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (stimulus_id, user_id, has_null_int, overall_confidence,
             int(starred), note, context_expansions, pronouns_snapshot),
        )

        for i, p in enumerate(pronouns):
            conn.execute(
                """
                INSERT INTO placed_pronouns
                    (annotation_id, position, label, lexical_form, confidence, ordering)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ann_id, p["position"], p["label"], p["lexical_form"],
                 p.get("confidence", 3), i),
            )

        # Fetch back
        row = conn.execute(
            "SELECT * FROM annotations WHERE id = ?", (ann_id,)
        ).fetchone()
        result = dict_row(row)
        prows = conn.execute(
            "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
            (ann_id,),
        ).fetchall()
        result["pronouns"] = [dict_row(r) for r in prows]
        return result


def get_annotation_progress(
    user_id: int, db_path: Optional[Path] = None
) -> Dict[str, int]:
    with get_conn(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM stimuli").fetchone()[0]
        completed = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE user_id = ? AND has_null_subject IS NOT NULL",
            (user_id,),
        ).fetchone()[0]
        starred = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE user_id = ? AND starred = 1",
            (user_id,),
        ).fetchone()[0]
        return {
            "total": total,
            "completed": completed,
            "starred": starred,
            "remaining": total - completed,
        }


def get_first_unannotated(
    user_id: int, db_path: Optional[Path] = None
) -> Optional[Dict]:
    """Return the first stimulus (by ordering) with no completed annotation."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT s.id, s.ordering,
                   (SELECT COUNT(*) FROM stimuli s2
                    WHERE s2.ordering < s.ordering) AS global_index
            FROM stimuli s
            LEFT JOIN annotations a ON a.stimulus_id = s.id AND a.user_id = ?
            WHERE a.id IS NULL OR a.has_null_subject IS NULL
            ORDER BY s.ordering
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "stimulus_id": row["id"],
            "ordering": row["ordering"],
            "global_index": row["global_index"],
        }


# ── Adjudication ────────────────────────────────────────────────────


def get_all_annotations_for_stimulus(
    stimulus_id: int, db_path: Optional[Path] = None
) -> List[Dict]:
    """Get all annotations (with pronouns) for a stimulus."""
    with get_conn(db_path) as conn:
        anns = conn.execute(
            """
            SELECT a.*, u.username, u.display_name
            FROM annotations a
            JOIN users u ON u.id = a.user_id
            WHERE a.stimulus_id = ?
            ORDER BY a.user_id
            """,
            (stimulus_id,),
        ).fetchall()
        results = []
        for ann in anns:
            d = dict_row(ann)
            prows = conn.execute(
                "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                (ann["id"],),
            ).fetchall()
            d["pronouns"] = [dict_row(r) for r in prows]
            results.append(d)
        return results


def upsert_adjudication(
    stimulus_id: int,
    adjudicator_id: int,
    resolution: str,
    accepted_user_id: Optional[int],
    has_null_subject: bool,
    pronouns: List[Dict],
    overall_confidence: int = 3,
    note: str = "",
    db_path: Optional[Path] = None,
) -> Dict:
    with get_conn(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM adjudications WHERE stimulus_id = ?", (stimulus_id,)
        ).fetchone()

        if existing:
            adj_id = existing["id"]
            conn.execute(
                """
                UPDATE adjudications
                SET adjudicator_id = ?, resolution = ?, accepted_user_id = ?,
                    has_null_subject = ?, overall_confidence = ?, note = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (adjudicator_id, resolution, accepted_user_id,
                 int(has_null_subject), overall_confidence, note, adj_id),
            )
            conn.execute(
                "DELETE FROM adjudication_pronouns WHERE adjudication_id = ?",
                (adj_id,),
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO adjudications
                    (stimulus_id, adjudicator_id, resolution, accepted_user_id,
                     has_null_subject, overall_confidence, note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (stimulus_id, adjudicator_id, resolution, accepted_user_id,
                 int(has_null_subject), overall_confidence, note),
            )
            adj_id = cursor.lastrowid

        for i, p in enumerate(pronouns):
            conn.execute(
                """
                INSERT INTO adjudication_pronouns
                    (adjudication_id, position, label, lexical_form, confidence, ordering)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (adj_id, p["position"], p["label"], p["lexical_form"],
                 p.get("confidence", 3), i),
            )

        row = conn.execute(
            "SELECT * FROM adjudications WHERE id = ?", (adj_id,)
        ).fetchone()
        result = dict_row(row)
        prows = conn.execute(
            "SELECT * FROM adjudication_pronouns WHERE adjudication_id = ? ORDER BY ordering",
            (adj_id,),
        ).fetchall()
        result["pronouns"] = [dict_row(r) for r in prows]
        return result


def get_adjudication_queue(db_path: Optional[Path] = None) -> List[Dict]:
    """Get stimuli that have 2+ annotations with disagreements and no adjudication."""
    with get_conn(db_path) as conn:
        # Get stimuli with 2+ annotations and no adjudication
        candidates = conn.execute(
            """
            SELECT s.id, s.text, s.ordering
            FROM stimuli s
            JOIN annotations a ON a.stimulus_id = s.id
            LEFT JOIN adjudications adj ON adj.stimulus_id = s.id
            WHERE adj.id IS NULL
            GROUP BY s.id
            HAVING COUNT(DISTINCT a.user_id) >= 2
            ORDER BY s.ordering
            """
        ).fetchall()

        # Filter to only those with disagreements
        results = []
        for row in candidates:
            anns = conn.execute(
                "SELECT has_null_subject FROM annotations WHERE stimulus_id = ? "
                "ORDER BY user_id LIMIT 2",
                (row["id"],),
            ).fetchall()
            if len(anns) >= 2 and anns[0]["has_null_subject"] != anns[1]["has_null_subject"]:
                results.append(dict_row(row))
                continue
            # Also check position/label disagreement when both say yes
            if len(anns) >= 2 and anns[0]["has_null_subject"] and anns[1]["has_null_subject"]:
                ann_full = conn.execute(
                    "SELECT id FROM annotations WHERE stimulus_id = ? ORDER BY user_id LIMIT 2",
                    (row["id"],),
                ).fetchall()
                a_pros = conn.execute(
                    "SELECT position, label FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                    (ann_full[0]["id"],),
                ).fetchall()
                b_pros = conn.execute(
                    "SELECT position, label FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                    (ann_full[1]["id"],),
                ).fetchall()
                a_set = {(p["position"], p["label"]) for p in a_pros}
                b_set = {(p["position"], p["label"]) for p in b_pros}
                if a_set != b_set:
                    results.append(dict_row(row))

        return results


def get_adjudication_progress(db_path: Optional[Path] = None) -> Dict[str, int]:
    with get_conn(db_path) as conn:
        # Count disagreements (matching queue logic)
        queue = get_adjudication_queue(db_path)
        adjudicated = conn.execute(
            "SELECT COUNT(*) FROM adjudications"
        ).fetchone()[0]
        total = len(queue) + adjudicated

        return {
            "total": total,
            "adjudicated": adjudicated,
            "remaining": len(queue),
        }
