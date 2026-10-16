"""Tests for scripts/rater_corpus/build_sweep_subset.py.

Run:  <venv>/bin/python -m pytest scripts/rater_corpus/tests/test_build_sweep_subset.py

A tiny fake option-B tree in a temp dir; nothing touches the cluster.
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_sweep_subset as bss  # noqa: E402

N_SHARDS = 6
LINES_PER_SHARD = 30


def _make_tree(root: Path, corpora=bss.CORPORA, shards=range(N_SHARDS), done=True):
    for corpus in corpora:
        for sid in shards:
            d = root / corpus / bss.shard_name(sid)
            d.mkdir(parents=True, exist_ok=True)
            lines = [f"{corpus} shard {sid} line {i} w1 w2" for i in range(LINES_PER_SHARD)]
            (d / (bss.shard_name(sid) + ".train")).write_text("\n".join(lines) + "\n")
            if corpus == bss.BASE_CORPUS:
                (d / "docs.jsonl.gz").write_bytes(b"not a corpus")
            elif done:
                (d / bss.DONE_MARKER).write_text("{}")
    return root


@pytest.fixture
def src(tmp_path):
    return _make_tree(tmp_path / "optionb_v1")


def _build(src, out, **kw):
    args = dict(corpora=bss.CORPORA, train_shards=[0, 1, 2], heldout_shard=5,
                heldout_lines=10)
    args.update(kw)
    return bss.build(src, out, **args)


class TestLayout:
    def test_flat_dirs_with_identical_shards(self, src, tmp_path):
        out = tmp_path / "subset"
        _build(src, out)
        for corpus in bss.CORPORA:
            d = out / corpus
            # Exactly what top-level corpus discovery will see.
            assert sorted(os.path.basename(p) for p in glob.glob(str(d / "*.train"))) == \
                ["shard_000.train", "shard_001.train", "shard_002.train"]
            assert [os.path.basename(p) for p in glob.glob(str(d / "*.test"))] == ["heldout.test"]
            for sid in (0, 1, 2):
                link = d / f"shard_{sid:03d}.train"
                assert link.is_symlink()
                assert os.readlink(link) == os.path.abspath(bss.shard_train_path(src, corpus, sid))
            # docs.jsonl.gz and other shard-dir contents are never linked.
            assert not list(d.glob("*.gz"))
            held = (d / "heldout.test").read_text().splitlines()
            assert len(held) == 10 and held[0] == f"{corpus} shard 5 line 0 w1 w2"
            assert not (d / "heldout.test").is_symlink()

    def test_manifest(self, src, tmp_path):
        out = tmp_path / "subset"
        _build(src, out)
        m = json.loads((out / "base" / bss.MANIFEST_NAME).read_text())
        assert m["corpus"] == "base"
        assert m["train_shards"] == [0, 1, 2] and m["heldout_shard"] == 5
        assert [e["name"] for e in m["train_files"]] == \
            ["shard_000.train", "shard_001.train", "shard_002.train"]
        size = os.path.getsize(bss.shard_train_path(src, "base", 0))
        assert m["train_files"][0]["bytes"] == size
        assert m["train_bytes_total"] == 3 * size
        assert m["heldout"]["mode"] == "prefix"
        assert m["heldout"]["lines"] == 10 and m["heldout"]["words"] == 10 * 7
        spec = json.loads((out / bss.SPEC_NAME).read_text())
        assert spec["train_shards"] == [0, 1, 2] and spec["heldout_shard"] == 5

    def test_heldout_lines_zero_symlinks_whole_shard(self, src, tmp_path):
        out = tmp_path / "subset"
        _build(src, out, corpora=["base"], heldout_lines=0)
        held = out / "base" / "heldout.test"
        assert held.is_symlink()
        assert len(held.read_text().splitlines()) == LINES_PER_SHARD

    def test_cache_key_hashes_link_targets(self, src, tmp_path):
        """model_foundry's corpus content hash follows the symlinks: the
        subset dir hashes like real copies of the same shards."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from model_foundry.cache_keys import _corpus_content_hash

        out = tmp_path / "subset"
        _build(src, out, corpora=["base"])
        copies = tmp_path / "copies"
        copies.mkdir()
        for sid in (0, 1, 2):
            shutil.copy(bss.shard_train_path(src, "base", sid), copies / f"shard_{sid:03d}.train")
        assert _corpus_content_hash(str(out / "base")) == _corpus_content_hash(str(copies))

    def test_no_temp_dirs_left(self, src, tmp_path):
        out = tmp_path / "subset"
        _build(src, out)
        assert sorted(p.name for p in out.iterdir()) == \
            sorted([*bss.CORPORA, bss.SPEC_NAME])


class TestFailures:
    def test_missing_shard_fails_before_writing(self, tmp_path):
        src = _make_tree(tmp_path / "src", shards=[0, 1, 5])   # no shard 2
        out = tmp_path / "subset"
        with pytest.raises(bss.SubsetError) as e:
            _build(src, out)
        msg = str(e.value)
        assert "shard_002" in msg and "nothing written" in msg
        # Every corpus missing it is listed.
        assert all(c in msg for c in bss.CORPORA)
        assert not out.exists()

    def test_unfinished_ablation_is_missing(self, tmp_path):
        src = _make_tree(tmp_path / "src", done=False)
        with pytest.raises(bss.SubsetError, match="ablation not finished"):
            _build(src, tmp_path / "subset", corpora=["base", "lemmatize_verbs"])
        # base alone has no marker requirement.
        _build(src, tmp_path / "subset", corpora=["base"])

    @pytest.mark.parametrize("train,held", [([0, 1, 1], 5), ([0, 1], 1), ([], 5), ([-1], 5)])
    def test_bad_selection(self, src, tmp_path, train, held):
        with pytest.raises(bss.SubsetError):
            _build(src, tmp_path / "subset", train_shards=train, heldout_shard=held)

    def test_unknown_corpus(self, src, tmp_path):
        with pytest.raises(bss.SubsetError, match="unknown corpora"):
            _build(src, tmp_path / "subset", corpora=["base", "nope"])

    def test_cli_exit_code(self, tmp_path, capsys):
        src = _make_tree(tmp_path / "src", shards=[0])
        rc = bss.main(["--src", str(src), "--out", str(tmp_path / "o"),
                       "--train-shards", "0", "1", "--heldout-shard", "5"])
        assert rc == 1
        assert "FATAL: missing shards" in capsys.readouterr().err


class TestIncremental:
    def test_later_corpora_reuse_the_pinned_selection(self, src, tmp_path):
        out = tmp_path / "subset"
        _build(src, out, corpora=["base"])
        before = (out / "base" / bss.MANIFEST_NAME).read_text()
        _build(src, out, corpora=["base", "lemmatize_verbs"])
        assert (out / "base" / bss.MANIFEST_NAME).read_text() == before   # kept as-is
        assert (out / "lemmatize_verbs" / "shard_000.train").is_symlink()

    def test_different_selection_against_same_out_refused(self, src, tmp_path):
        out = tmp_path / "subset"
        _build(src, out, corpora=["base"])
        with pytest.raises(bss.SubsetError, match="pins a different selection"):
            _build(src, out, corpora=["lemmatize_verbs"], train_shards=[0, 1, 3])

    def test_broken_existing_dir_refused(self, src, tmp_path):
        out = tmp_path / "subset"
        _build(src, out, corpora=["base"])
        (out / "base" / "shard_001.train").unlink()
        with pytest.raises(bss.SubsetError, match="intact symlink"):
            _build(src, out, corpora=["base"])
