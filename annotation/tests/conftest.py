"""Shared pytest fixtures for annotation app tests."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from annotation import db
from annotation.server import app


@pytest.fixture
def tmp_db(monkeypatch):
    """Provide a temporary SQLite database, patched in as the module's DB_PATH."""
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db(path)
    yield path
    import shutil
    shutil.rmtree(tmp_dir)


@pytest.fixture
def seeded(tmp_db):
    """Seed one admin and one annotator; return {admin, alice} with tokens."""
    db.seed_users([
        {"username": "admin", "display_name": "Admin", "role": "admin"},
        {"username": "alice", "display_name": "Alice", "role": "annotator"},
    ], tmp_db)
    return {
        "admin": db.get_user_by_username("admin", tmp_db),
        "alice": db.get_user_by_username("alice", tmp_db),
    }


@pytest.fixture
def stimuli(tmp_db):
    """Load 3 test stimuli and return the count."""
    count = db.load_stimuli([
        {"text": "Comimos paella ayer.", "source": "test", "ordering": 0},
        {"text": "Fuimos al parque.", "source": "test", "ordering": 1},
        {"text": "Es muy bonito.", "source": "test", "ordering": 2},
    ], tmp_db)
    return count


@pytest.fixture
def client(tmp_db):
    """FastAPI TestClient bound to the temp DB."""
    return TestClient(app)
