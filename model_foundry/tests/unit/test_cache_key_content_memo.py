"""The corpus content hash is memoized on the file listing (name, size, mtime)."""

import hashlib
import os

import pytest

from model_foundry import cache_keys as ck


def _reference_hash(corpus_dir):
    """The digest as computed before the memo existed."""
    outer = hashlib.sha256()
    for name in sorted(n for n in os.listdir(corpus_dir) if n.endswith(".train")):
        with open(os.path.join(corpus_dir, name), "rb") as f:
            inner = hashlib.sha256(f.read()).hexdigest()
        outer.update(name.encode() + b"\x00" + inner.encode() + b"\x00")
    return outer.hexdigest()


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "b.train").write_text("second file\n")
    (tmp_path / "a.train").write_text("first file\n")
    (tmp_path / "notes.txt").write_text("not corpus\n")
    ck._CONTENT_HASH_MEMO.clear()
    return tmp_path


def test_digest_is_unchanged_and_sidecar_written(corpus):
    assert ck._corpus_content_hash(str(corpus)) == _reference_hash(corpus)
    assert (corpus / ck._CONTENT_HASH_SIDECAR).exists()


def test_second_process_reads_the_sidecar_not_the_files(corpus, monkeypatch):
    first = ck._corpus_content_hash(str(corpus))
    ck._CONTENT_HASH_MEMO.clear()          # a new process
    monkeypatch.setattr(ck, "_hash_file_bytes", lambda *a, **k: pytest.fail("re-read the corpus"))
    assert ck._corpus_content_hash(str(corpus)) == first


def test_changed_file_is_rehashed(corpus):
    first = ck._corpus_content_hash(str(corpus))
    (corpus / "a.train").write_text("first file, edited\n")
    second = ck._corpus_content_hash(str(corpus))
    assert second != first and second == _reference_hash(corpus)


def test_corrupt_sidecar_is_ignored(corpus):
    first = ck._corpus_content_hash(str(corpus))
    ck._CONTENT_HASH_MEMO.clear()
    (corpus / ck._CONTENT_HASH_SIDECAR).write_text("{not json")
    assert ck._corpus_content_hash(str(corpus)) == first


def test_unwritable_dir_still_hashes(corpus, monkeypatch):
    def deny(*a, **k):
        raise OSError("read-only")
    monkeypatch.setattr(ck.os, "replace", deny)
    assert ck._corpus_content_hash(str(corpus)) == _reference_hash(corpus)


def test_sidecar_does_not_change_the_cache_key(corpus):
    tok = corpus / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    before = ck.compute_cache_key(str(corpus), str(tok), 128, [], packing="cls_sep")
    ck._CONTENT_HASH_MEMO.clear()
    assert ck.compute_cache_key(str(corpus), str(tok), 128, [], packing="cls_sep") == before
