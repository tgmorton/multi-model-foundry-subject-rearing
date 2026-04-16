"""Smoke tests for the annotation app.

Covers: DB schema, audit trail, single-annotator export, multi-pronoun
positions, API auth, and the critical endpoints the app depends on.
"""

import json

import pytest

from annotation import db
from annotation.export import export_gold, export_raw


# ── DB layer ─────────────────────────────────────────────────────────


def test_init_db_creates_all_tables(tmp_db):
    import sqlite3
    conn = sqlite3.connect(str(tmp_db))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"users", "stimuli", "annotations", "placed_pronouns",
            "adjudications", "adjudication_pronouns",
            "annotation_history"}.issubset(tables)


def test_seed_users_is_idempotent(tmp_db):
    db.seed_users([{"username": "a", "display_name": "A", "role": "admin"}], tmp_db)
    token1 = db.get_user_by_username("a", tmp_db)["token"]
    db.seed_users([{"username": "a", "display_name": "A", "role": "admin"}], tmp_db)
    token2 = db.get_user_by_username("a", tmp_db)["token"]
    assert token1 == token2, "re-seeding must not rotate tokens"


def test_upsert_annotation_writes_history(tmp_db, seeded, stimuli):
    alice = seeded["alice"]
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=True,
        pronouns=[{"position": 0, "label": "PRO.1sg",
                   "lexical_form": "yo", "confidence": 4}],
        overall_confidence=4,
        db_path=tmp_db,
    )
    # Update the same annotation
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=True,
        pronouns=[{"position": 0, "label": "PRO.3sg",
                   "lexical_form": "él", "confidence": 3}],
        overall_confidence=3,
        db_path=tmp_db,
    )

    with db.get_conn(tmp_db) as conn:
        rows = conn.execute(
            "SELECT * FROM annotation_history ORDER BY id"
        ).fetchall()
    assert len(rows) == 2, "each upsert must append a history row"
    snapshots = [json.loads(r["pronouns_json"]) for r in rows]
    assert snapshots[0][0]["label"] == "PRO.1sg"
    assert snapshots[1][0]["label"] == "PRO.3sg"


def test_upsert_annotation_updates_in_place(tmp_db, seeded, stimuli):
    alice = seeded["alice"]
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=True,
        pronouns=[{"position": 0, "label": "PRO.1sg",
                   "lexical_form": "yo", "confidence": 4}],
        db_path=tmp_db,
    )
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=False, pronouns=[],
        db_path=tmp_db,
    )
    with db.get_conn(tmp_db) as conn:
        anns = conn.execute(
            "SELECT * FROM annotations WHERE stimulus_id = ? AND user_id = ?",
            (1, alice["id"]),
        ).fetchall()
        pros = conn.execute("SELECT * FROM placed_pronouns").fetchall()
    assert len(anns) == 1, "annotations must upsert, not duplicate"
    assert anns[0]["has_null_subject"] == 0
    assert len(pros) == 0, "placed_pronouns must be cleared on change"


def test_get_first_unannotated(tmp_db, seeded, stimuli):
    alice = seeded["alice"]
    assert db.get_first_unannotated(alice["id"], tmp_db)["stimulus_id"] == 1
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=False, pronouns=[], db_path=tmp_db,
    )
    assert db.get_first_unannotated(alice["id"], tmp_db)["stimulus_id"] == 2
    # Finish all
    for sid in (2, 3):
        db.upsert_annotation(
            stimulus_id=sid, user_id=alice["id"],
            has_null_subject=False, pronouns=[], db_path=tmp_db,
        )
    assert db.get_first_unannotated(alice["id"], tmp_db) is None


def test_get_stimuli_page_includes_pronouns_inline(tmp_db, seeded, stimuli):
    """Regression test: pronouns must be embedded in the page response
    to avoid the N+1 query the old frontend had."""
    alice = seeded["alice"]
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=True,
        pronouns=[
            {"position": 0, "label": "PRO.1sg", "lexical_form": "yo",
             "confidence": 4},
            {"position": 0, "label": "PRO.3sg", "lexical_form": "él",
             "confidence": 4},
        ],
        db_path=tmp_db,
    )
    page = db.get_stimuli_page(alice["id"], 1, 50, tmp_db)
    item = next(i for i in page["items"] if i["id"] == 1)
    assert len(item["pronouns"]) == 2
    assert {p["lexical_form"] for p in item["pronouns"]} == {"yo", "él"}


# ── Export ───────────────────────────────────────────────────────────


def test_export_gold_single_annotator(tmp_db, seeded, stimuli):
    """The critical fix: single-annotator records must appear in gold."""
    alice = seeded["alice"]
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=True,
        pronouns=[{"position": 0, "label": "PRO.1pl",
                   "lexical_form": "nosotros/nosotras", "confidence": 4}],
        db_path=tmp_db,
    )
    db.upsert_annotation(
        stimulus_id=2, user_id=alice["id"],
        has_null_subject=False, pronouns=[], db_path=tmp_db,
    )

    records = export_gold(tmp_db)
    assert len(records) == 2
    for r in records:
        assert r["annotator_agreement"] == "single"
        assert r["language"] == "es"
    # Stimulus 3 was never annotated — should be excluded
    assert {r["stimulus_id"] for r in records} == {1, 2}


def test_export_gold_empty_when_no_annotations(tmp_db, seeded, stimuli):
    assert export_gold(tmp_db) == []


def test_export_gold_multi_pronoun_same_position(tmp_db, seeded, stimuli):
    alice = seeded["alice"]
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=True,
        pronouns=[
            {"position": 0, "label": "PRO.1pl",
             "lexical_form": "nosotros/nosotras", "confidence": 4},
            {"position": 0, "label": "PRO.3sg",
             "lexical_form": "él", "confidence": 4},
        ],
        db_path=tmp_db,
    )
    records = export_gold(tmp_db)
    markers = records[0]["markers"]
    assert len(markers) == 2
    assert {m["label"] for m in markers} == {"PRO.1pl", "PRO.3sg"}
    assert all(m["position"] == 0 for m in markers), \
        "both pronouns must share the same position"


def test_export_raw_includes_every_annotation(tmp_db, seeded, stimuli):
    alice = seeded["alice"]
    db.upsert_annotation(
        stimulus_id=1, user_id=alice["id"],
        has_null_subject=True,
        pronouns=[{"position": 0, "label": "PRO.1sg",
                   "lexical_form": "yo", "confidence": 4}],
        note="test note",
        db_path=tmp_db,
    )
    raw = export_raw(tmp_db)
    assert len(raw) == 1
    assert raw[0]["annotator"] == "alice"
    assert raw[0]["note"] == "test note"


# ── API layer (via TestClient) ───────────────────────────────────────


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_login_valid(client, seeded):
    r = client.post("/api/auth/login", json={
        "username": "alice",
        "token": seeded["alice"]["token"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["username"] == "alice"
    assert body["user"]["role"] == "annotator"


def test_login_invalid_token(client, seeded):
    r = client.post("/api/auth/login", json={
        "username": "alice",
        "token": "wrong-token",
    })
    assert r.status_code == 401


def test_save_annotation_round_trip(client, seeded, stimuli):
    headers = {"Authorization": f"Bearer {seeded['alice']['token']}"}
    r = client.put("/api/annotations/1", headers=headers, json={
        "has_null_subject": True,
        "pronouns": [
            {"position": 0, "label": "PRO.1sg",
             "lexical_form": "yo", "confidence": 4}
        ],
        "overall_confidence": 4,
    })
    assert r.status_code == 200
    # Re-fetch and verify
    r2 = client.get("/api/stimuli/1", headers=headers)
    assert r2.status_code == 200
    ann = r2.json()["annotation"]
    assert ann["has_null_subject"] == 1
    assert len(ann["pronouns"]) == 1
    assert ann["pronouns"][0]["lexical_form"] == "yo"


def test_non_admin_cannot_export(client, seeded, stimuli):
    headers = {"Authorization": f"Bearer {seeded['alice']['token']}"}
    r = client.get("/api/export/gold", headers=headers)
    assert r.status_code == 403


def test_admin_can_export_gold(client, seeded, stimuli):
    alice_hdr = {"Authorization": f"Bearer {seeded['alice']['token']}"}
    admin_hdr = {"Authorization": f"Bearer {seeded['admin']['token']}"}

    client.put("/api/annotations/1", headers=alice_hdr, json={
        "has_null_subject": True,
        "pronouns": [{"position": 0, "label": "PRO.1sg",
                      "lexical_form": "yo", "confidence": 4}],
    })

    r = client.get("/api/export/gold", headers=admin_hdr)
    assert r.status_code == 200
    records = r.json()
    assert len(records) == 1
    assert records[0]["annotator_agreement"] == "single"


def test_invalid_pronoun_label_rejected(client, seeded, stimuli):
    headers = {"Authorization": f"Bearer {seeded['alice']['token']}"}
    r = client.put("/api/annotations/1", headers=headers, json={
        "has_null_subject": True,
        "pronouns": [{"position": 0, "label": "PRO.bogus",
                      "lexical_form": "nope", "confidence": 3}],
    })
    assert r.status_code == 422


def test_next_unannotated_endpoint(client, seeded, stimuli):
    headers = {"Authorization": f"Bearer {seeded['alice']['token']}"}
    r = client.get("/api/annotations/next-unannotated", headers=headers)
    assert r.status_code == 200
    assert r.json()["stimulus_id"] == 1

    client.put("/api/annotations/1", headers=headers, json={
        "has_null_subject": False, "pronouns": [],
    })
    r2 = client.get("/api/annotations/next-unannotated", headers=headers)
    assert r2.json()["stimulus_id"] == 2
