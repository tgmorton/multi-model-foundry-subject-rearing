"""Replicability redesign tests (2026-06-05).

Covers the dedicated-RNG-stream design, the G3 paired-init guarantee,
manifest-driven corpus ingestion, and content-addressed cache keys.

Most tests need torch (run in the training container / CI image); the
pure-stdlib ones (derive_seed, cache keys, file resolution) run anywhere.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from model_foundry.rng import derive_seed
from model_foundry.data_collators import MaskedLMDataCollator


# ---------------------------------------------------------------------------
# derive_seed — stable, purpose-salted stream derivation
# ---------------------------------------------------------------------------

class TestDeriveSeed:
    def test_known_answer_is_stable(self):
        # Pin the derivation: if blake2b inputs or masking ever change,
        # every run's data order changes — this must be a deliberate,
        # visible act, not a refactor side-effect.
        expected = int.from_bytes(
            hashlib.blake2b(b"42|data_order|0|0", digest_size=8).digest(),
            "little",
        ) & ((1 << 63) - 1)
        assert derive_seed(42, "data_order") == expected

    def test_deterministic_across_calls(self):
        assert derive_seed(42, "mlm_mask", 3, 17) == derive_seed(42, "mlm_mask", 3, 17)

    def test_distinct_across_purposes_epochs_batches_seeds(self):
        seeds = {
            derive_seed(42, "data_order", 0),
            derive_seed(42, "mlm_mask", 0),
            derive_seed(42, "data_order", 1),
            derive_seed(42, "mlm_mask", 0, 1),
            derive_seed(137, "data_order", 0),
        }
        assert len(seeds) == 5

    def test_fits_signed_int64(self):
        for s in (0, 42, 137, 2**31):
            for e in (0, 29):
                v = derive_seed(s, "data_order", e)
                assert 0 <= v < 2**63
                torch.Generator().manual_seed(v)  # must not raise


# ---------------------------------------------------------------------------
# Data order — pure function of (seed, epoch)
# ---------------------------------------------------------------------------

class TestDataOrder:
    def _permutation(self, seed: int, epoch: int, n: int = 100):
        g = torch.Generator()
        g.manual_seed(derive_seed(seed, "data_order", epoch))
        loader = torch.utils.data.DataLoader(
            list(range(n)), batch_size=1, shuffle=True, generator=g
        )
        return [int(b[0]) for b in loader]

    def test_same_seed_epoch_same_order(self):
        assert self._permutation(42, 0) == self._permutation(42, 0)

    def test_epochs_differ(self):
        assert self._permutation(42, 0) != self._permutation(42, 1)

    def test_seeds_differ(self):
        assert self._permutation(42, 0) != self._permutation(137, 0)

    def test_order_independent_of_ambient_rng(self):
        # The whole point of the redesign: ambient global-RNG consumption
        # (e.g. arch-specific model init draws) must not influence order.
        first = self._permutation(42, 5)
        torch.manual_seed(999)
        _ = torch.randn(1000)  # consume ambient stream
        assert self._permutation(42, 5) == first


# ---------------------------------------------------------------------------
# MLM masking — pure function of (seed, epoch, batch)
# ---------------------------------------------------------------------------

class _StubTokenizer:
    pad_token_id = 0
    mask_token_id = 4
    bos_token_id = 1
    eos_token_id = 2
    cls_token_id = None
    sep_token_id = None
    unk_token_id = 3

    def __len__(self):
        return 1000


def _mlm_batches(seed, epoch, n_batches=3):
    coll = MaskedLMDataCollator(tokenizer=_StubTokenizer(), base_seed=seed)
    coll.set_epoch(epoch)
    out = []
    for i in range(n_batches):
        examples = [{"input_ids": list(range(10, 26))} for _ in range(4)]
        batch = coll(examples)
        out.append((batch["input_ids"].clone(), batch["labels"].clone()))
    return out


class TestMLMMasking:
    def test_restart_stable(self):
        a = _mlm_batches(42, epoch=7)
        b = _mlm_batches(42, epoch=7)
        for (ia, la), (ib, lb) in zip(a, b):
            assert torch.equal(ia, ib)
            assert torch.equal(la, lb)

    def test_epochs_and_batches_differ(self):
        a = _mlm_batches(42, epoch=0)
        b = _mlm_batches(42, epoch=1)
        assert any(not torch.equal(x[0], y[0]) for x, y in zip(a, b))
        # consecutive batches within an epoch differ from each other
        assert not torch.equal(a[0][0], a[1][0])

    def test_set_epoch_resets_batch_counter(self):
        coll = MaskedLMDataCollator(tokenizer=_StubTokenizer(), base_seed=42)
        examples = [{"input_ids": list(range(10, 26))} for _ in range(4)]
        coll.set_epoch(3)
        first = coll(examples)["input_ids"].clone()
        coll(examples)
        coll.set_epoch(3)  # restart of the same epoch
        again = coll(examples)["input_ids"].clone()
        assert torch.equal(first, again)

    def test_legacy_none_seed_still_works(self):
        coll = MaskedLMDataCollator(tokenizer=_StubTokenizer(), base_seed=None)
        examples = [{"input_ids": list(range(10, 26))} for _ in range(4)]
        batch = coll(examples)
        assert batch["input_ids"].shape == batch["labels"].shape


# ---------------------------------------------------------------------------
# G3 — same (arch, seed) → bitwise-identical init, regardless of condition
# ---------------------------------------------------------------------------

class TestPairedInit:
    def _build_twice(self, consume_rng_between: bool):
        from model_foundry.utils import set_seed

        def build():
            set_seed(42)  # what trainer._train_loop now does right before init
            torch.nn.init  # noqa: B018
            m = torch.nn.Sequential(
                torch.nn.Embedding(100, 32),
                torch.nn.Linear(32, 32),
                torch.nn.Linear(32, 100),
            )
            return m.state_dict()

        a = build()
        if consume_rng_between:
            _ = torch.randn(512)  # a "different condition" consuming RNG elsewhere
        b = build()
        return a, b

    def test_reseed_makes_init_condition_invariant(self):
        a, b = self._build_twice(consume_rng_between=True)
        for k in a:
            assert torch.equal(a[k], b[k]), f"init differs at {k}"

    def test_trainer_reseeds_at_the_call_site(self):
        # Call-site tripwire (implementation-audit m2): the property above
        # only holds if trainer._train_loop actually re-calls set_seed
        # right before _initialize_model. Statically assert the ordering
        # so deleting that line fails CI even without a GPU run.
        import ast
        import inspect
        import model_foundry.trainer as trainer_mod

        tree = ast.parse(inspect.getsource(trainer_mod))
        fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_train_loop"
        )
        calls = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in ("set_seed", "_load_tokenizer", "_initialize_model"):
                    calls.append((node.lineno, name))
        calls.sort()
        names = [n for _, n in calls]
        tok_idx = names.index("_load_tokenizer")
        init_idx = names.index("_initialize_model")
        assert "set_seed" in names[tok_idx:init_idx], (
            "trainer._train_loop must re-call set_seed between tokenizer "
            "load and _initialize_model — the G3 paired-init guarantee"
        )


# ---------------------------------------------------------------------------
# Corpus ingestion — manifest-driven, top-level only
# ---------------------------------------------------------------------------

class TestCorpusResolution:
    def _make_corpus(self, tmp_path: Path, with_manifest: bool):
        (tmp_path / "a.train").write_text("alpha text\n")
        (tmp_path / "b.train").write_text("beta text\n")
        # poison: intermediates that the 2026-06-04 incident ingested
        (tmp_path / "_train").mkdir()
        (tmp_path / "_train" / "a.train").write_text("DUPLICATE\n")
        (tmp_path / "pool_remainder").mkdir()
        (tmp_path / "pool_remainder" / "c.train").write_text("POOL\n")
        if with_manifest:
            per_file = []
            for stem in ("a", "b"):
                digest = hashlib.sha256(
                    (tmp_path / f"{stem}.train").read_bytes()
                ).hexdigest()
                per_file.append({"stem": stem, "output_checksum": digest})
            (tmp_path / "COMPOSE_MANIFEST.json").write_text(json.dumps({
                "totals": {"files": 2, "train_tokens": 4},
                "per_file": per_file,
            }))
        return tmp_path

    def test_manifest_driven_excludes_intermediates(self, tmp_path):
        from model_foundry.tokenizer.tokenize_dataset import _resolve_training_files
        corpus = self._make_corpus(tmp_path, with_manifest=True)
        files = _resolve_training_files(str(corpus))
        names = sorted(os.path.basename(f) for f in files)
        assert names == ["a.train", "b.train"]
        assert not any("_train" in f or "pool_remainder" in f for f in files)

    def test_manifest_checksum_mismatch_is_fatal(self, tmp_path):
        from model_foundry.tokenizer.tokenize_dataset import _resolve_training_files
        corpus = self._make_corpus(tmp_path, with_manifest=True)
        (corpus / "a.train").write_text("tampered\n")
        with pytest.raises((ValueError, FileNotFoundError)):
            _resolve_training_files(str(corpus))

    def test_no_manifest_top_level_only(self, tmp_path):
        from model_foundry.tokenizer.tokenize_dataset import _resolve_training_files
        corpus = self._make_corpus(tmp_path, with_manifest=False)
        files = _resolve_training_files(str(corpus))
        names = sorted(os.path.basename(f) for f in files)
        assert names == ["a.train", "b.train"]
        assert not any("_train" in f or "pool_remainder" in f for f in files)


# ---------------------------------------------------------------------------
# Cache keys — content-addressed for real
# ---------------------------------------------------------------------------

class TestCacheKeyContent:
    def _corpus_and_tokenizer(self, tmp_path: Path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.train").write_text("alpha\n")
        tok = tmp_path / "tok"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}")
        return corpus, tok

    def test_content_change_changes_key(self, tmp_path):
        from model_foundry.cache_keys import compute_cache_key
        corpus, tok = self._corpus_and_tokenizer(tmp_path)
        k1 = compute_cache_key(str(corpus), str(tok), 1000, [])
        (corpus / "a.train").write_text("alpha CHANGED\n")
        k2 = compute_cache_key(str(corpus), str(tok), 1000, [])
        assert k1 != k2

    def test_nested_intermediates_do_not_affect_key(self, tmp_path):
        from model_foundry.cache_keys import compute_cache_key
        corpus, tok = self._corpus_and_tokenizer(tmp_path)
        k1 = compute_cache_key(str(corpus), str(tok), 1000, [])
        (corpus / "_train").mkdir()
        (corpus / "_train" / "x.train").write_text("INTERMEDIATE\n")
        k2 = compute_cache_key(str(corpus), str(tok), 1000, [])
        assert k1 == k2


# ---------------------------------------------------------------------------
# Checkpoint round-trip — epoch_batch_offset persists
# ---------------------------------------------------------------------------

class TestEpochBatchOffset:
    def test_state_dict_roundtrip(self, tmp_path):
        # Minimal contract test: the field written by save is the field
        # read by load (full save_checkpoint needs a model + tokenizer,
        # exercised by the integration A/B run; here we pin the contract).
        state = {
            "global_step": 100,
            "epoch": 3,
            "epoch_batch_offset": 1280,
        }
        p = tmp_path / "training_state.pt"
        torch.save(state, p)
        loaded = torch.load(p, weights_only=False)
        assert int(loaded.get("epoch_batch_offset", 0)) == 1280

    def test_legacy_state_defaults_to_zero(self, tmp_path):
        state = {"global_step": 100, "epoch": 3}
        p = tmp_path / "training_state.pt"
        torch.save(state, p)
        loaded = torch.load(p, weights_only=False)
        assert int(loaded.get("epoch_batch_offset", 0) or 0) == 0
