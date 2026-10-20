"""Wiring tests for the rater mode of scripts/sweep_agent_lm.py.

Run:  <venv>/bin/python -m pytest tests/test_sweep_agent_rater.py

CPU only, no W&B / S3 / cluster: Trainer, wandb and the registry are
replaced with recorders. The pure decision logic is covered in
tests/test_rater_sweep_mode.py; this file checks that the agent uses it
(config dispatch, capped eval set, step hook, stop, final eval, sentinel)
and that the epoch path is untouched.
"""

import math
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model_foundry.config import ExperimentConfig  # noqa: E402
from scripts import rater_sweep as rs  # noqa: E402
from scripts import sweep_agent_lm as agent  # noqa: E402

RATER_BASE = "configs/sweeps/baselines/rater_bert_large_en.yaml"
BERT_BASE = "configs/sweeps/baselines/bert_large_en.yaml"
SUBSET = "/mnt/data/rater_corpus/sweep_subset_v1"

_AGENT_ENV = ("RATER_SWEEP", "CONDITION", "CORPUS_DIR", "REGISTRY_RUN_KIND",
              "REGISTRY_LANG", "REGISTRY_CONDITION", "REGISTRY_SEED", "REGISTRY_RUN_ID")


@pytest.fixture
def clean_env(monkeypatch):
    """_prepare_config writes REGISTRY_* into os.environ; register them with
    monkeypatch first so teardown restores them."""
    for k in _AGENT_ENV:
        monkeypatch.setenv(k, "")
    return monkeypatch


def _run_obj(**wc):
    return types.SimpleNamespace(config=wc, id="run1", sweep_id="sw1", name="orig")


# --------------------------------------------------------------------------- #
# _prepare_config dispatch
# --------------------------------------------------------------------------- #
class TestPrepareConfig:
    def test_rater_baseline_takes_rater_path(self, clean_env):
        clean_env.setenv("CONDITION", "lemmatize_verbs")
        clean_env.setenv("CORPUS_DIR", f"{SUBSET}/lemmatize_verbs/")
        clean_env.setattr(agent, "_read_cache_num_chunks", lambda c, b: 400_000)
        run = _run_obj(arch="bert_large", lang="en", base_config=RATER_BASE,
                       condition="lemmatize_verbs", learning_rate=3e-4,
                       effective_batch_size=256, dropout=0.05)

        cfg, identity, settings = agent._prepare_config(run)

        assert settings == rs.load_settings(yaml.safe_load((REPO / RATER_BASE).read_text()))
        assert identity["condition"] == "rater_lemmatize_verbs"
        assert identity["run_kind"] == "hp_sweep"
        assert identity["run_id"] == "sweep-sw1-run1"
        assert run.name == "rater_lemmatize_verbs_lr3.0e-4_eb256_d0.05"
        t = cfg["training"]
        assert t["learning_rate"] == 3e-4
        assert t["gradient_accumulation_steps"] == 256 // 32
        assert t["train_steps"] == t["scheduler_train_steps"] == 3000
        assert t["checkpoint_schedule"] == [] and t["resume_state_steps"] is None
        assert t["output_dir"] == "models/rater_sweeps/lemmatize_verbs/run1"
        assert cfg["data"]["training_corpus"] == f"{SUBSET}/lemmatize_verbs/"
        assert cfg["data"]["test_corpus"] == f"{SUBSET}/lemmatize_verbs/"
        ExperimentConfig(**cfg)

    def test_env_switch_without_block(self, clean_env):
        """RATER_SWEEP=1 turns rater mode on (default settings) for a
        baseline that has no rater_sweep block."""
        clean_env.setenv("RATER_SWEEP", "1")
        clean_env.setattr(agent, "_read_cache_num_chunks", lambda c, b: 1_000_000)
        run = _run_obj(arch="bert_large", lang="en", base_config=BERT_BASE)
        cfg, identity, settings = agent._prepare_config(run)
        assert settings == rs.RaterSweepSettings()
        assert identity["condition"] == "rater_baseline"
        assert cfg["training"]["train_steps"] == 3000

    def test_corpus_condition_mispairing_refused(self, clean_env):
        clean_env.setenv("CONDITION", "base")
        clean_env.setenv("CORPUS_DIR", f"{SUBSET}/impoverish_case_en")
        clean_env.setattr(agent, "_read_cache_num_chunks", lambda c, b: 1)
        with pytest.raises(RuntimeError, match="impoverish_case_en"):
            agent._prepare_config(_run_obj(arch="bert_large", lang="en",
                                           base_config=RATER_BASE))

    def test_epoch_path_unchanged_and_ignores_rater_env(self, clean_env):
        """An existing baseline stays on the epoch path even with CONDITION /
        CORPUS_DIR in the environment (they are rater-mode only)."""
        clean_env.setenv("CONDITION", "base")
        clean_env.setenv("CORPUS_DIR", f"{SUBSET}/base")
        clean_env.setattr(agent, "_compute_anchor_schedule",
                          lambda cfg, base_dir, epochs: ([0, 1, 2, 4], [4]))
        before = yaml.safe_load((REPO / BERT_BASE).read_text())
        run = _run_obj(arch="bert_large", lang="en", base_config=BERT_BASE)

        cfg, identity, settings = agent._prepare_config(run)

        assert settings is None
        assert identity["condition"] == "baseline"
        assert run.name == "orig"
        assert cfg["data"] == before["data"]
        t = cfg["training"]
        assert t["epochs"] == before["training"]["sweep_epochs"]
        assert t["train_steps"] is None
        assert t["checkpoint_schedule"] == [0, 1, 2, 4]
        assert t["resume_state_steps"] == [4]
        assert t["output_dir"] == "models/sweeps/run1"


# --------------------------------------------------------------------------- #
# held-out eval set
# --------------------------------------------------------------------------- #
def _tokenized_test_split(tmp_path, n_rows=100, row_len=7):
    from datasets import Dataset
    ds = Dataset.from_dict({"input_ids": [[i] * row_len for i in range(n_rows)]})
    ds.save_to_disk(str(tmp_path / "tok" / "test"))
    trainer = types.SimpleNamespace(
        data_processor=types.SimpleNamespace(tokenized_data_dir=str(tmp_path / "tok")))
    return trainer


def _cfg_ns(seq_len=10, batch_size=4, arch="bert", test_corpus="x"):
    return types.SimpleNamespace(
        data=types.SimpleNamespace(test_corpus=test_corpus, max_sequence_length=seq_len,
                                   batch_size=batch_size),
        model=types.SimpleNamespace(architecture=arch),
        training=types.SimpleNamespace(mlm_probability=0.15),
    )


class TestHeldOutChunks:
    def test_cap_takes_a_prefix(self, tmp_path):
        trainer = _tokenized_test_split(tmp_path)
        full = agent._load_held_out_chunks(trainer, _cfg_ns())
        capped = agent._load_held_out_chunks(trainer, _cfg_ns(), max_sequences=5)
        assert full.shape == (70, 10)            # 700 tokens packed into rows of 10
        assert capped.shape == (5, 10)
        assert torch.equal(capped, full[:5])

    def test_cap_larger_than_split(self, tmp_path):
        trainer = _tokenized_test_split(tmp_path)
        assert agent._load_held_out_chunks(trainer, _cfg_ns(), max_sequences=10_000).shape == (70, 10)

    def test_missing_test_corpus(self, tmp_path):
        trainer = _tokenized_test_split(tmp_path)
        assert agent._load_held_out_chunks(trainer, _cfg_ns(test_corpus=None)) is None
        assert math.isnan(agent._compute_held_out_perplexity(trainer, _cfg_ns(test_corpus=None)))


class _RecordingMLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, input_ids, labels):
        self.seen.append(input_ids.clone())
        return types.SimpleNamespace(loss=torch.tensor(2.0))


class TestCappedPerplexity:
    def test_same_masks_every_eval(self, tmp_path):
        trainer = _tokenized_test_split(tmp_path)
        trainer.model = _RecordingMLM()
        trainer.tokenizer = types.SimpleNamespace(mask_token_id=999)
        cfg = _cfg_ns()
        chunks = agent._load_held_out_chunks(trainer, cfg, max_sequences=8)

        p1 = agent._compute_held_out_perplexity(trainer, cfg, chunks=chunks)
        first = list(trainer.model.seen)
        trainer.model.seen.clear()
        p2 = agent._compute_held_out_perplexity(trainer, cfg, chunks=chunks)

        assert p1 == p2 == pytest.approx(math.exp(2.0))
        assert len(first) == 2                   # 8 rows / batch 4
        assert all(torch.equal(a, b) for a, b in zip(first, trainer.model.seen))
        assert any((x == 999).any() for x in first)


# --------------------------------------------------------------------------- #
# _run_rater_trial
# --------------------------------------------------------------------------- #
class _FakeTrainer:
    """Calls on_step_end for steps 1..train_steps like TrainingLoop.run."""
    last = None

    def __init__(self, config, base_dir):
        self.config = config
        self.training_loop = types.SimpleNamespace(global_step=0, final_loss=1.25)
        self.train_kwargs = None
        _FakeTrainer.last = self

    def train(self, on_epoch_end=None, on_step_end=None, endpoint_checkpoint="full"):
        self.train_kwargs = {"on_epoch_end": on_epoch_end,
                             "endpoint_checkpoint": endpoint_checkpoint}
        for step in range(1, self.config.training.train_steps + 1):
            self.training_loop.global_step = step
            if on_step_end is not None and on_step_end(step):
                break
        return self.training_loop.global_step


@pytest.fixture
def trial(monkeypatch, tmp_path):
    """Patch the agent's collaborators; return a runner + the recorders."""
    rec = {"logs": [], "registry_end": [], "register_end": []}
    ppl_series = []

    monkeypatch.setattr(agent, "Trainer", _FakeTrainer)
    monkeypatch.setattr(agent, "wandb", types.SimpleNamespace(
        log=lambda payload, step=None: rec["logs"].append((step, dict(payload)))))
    monkeypatch.setattr(agent, "_registry", types.SimpleNamespace(
        register_run_end=lambda **kw: rec["registry_end"].append(kw)))
    monkeypatch.setattr(agent, "_register_start", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_register_end",
                        lambda identity, run, trainer, loss, ppl:
                        rec["register_end"].append((loss, ppl)))
    monkeypatch.setattr(agent, "_load_held_out_chunks",
                        lambda trainer, config, max_sequences=None: torch.zeros(2, 4))
    monkeypatch.setattr(agent, "_compute_held_out_metrics",
                        lambda trainer, config, chunks=None: {
                            "ppl": ppl_series.pop(0), "accuracy": float("nan"),
                            "top_token_share": float("nan"), "top_token_id": -1})
    sentinel = tmp_path / "sentinel"
    monkeypatch.setenv("TRIAL_SENTINEL", str(sentinel))

    def run(ppls, **settings_kw):
        ppl_series[:] = list(ppls)
        settings = rs.RaterSweepSettings(**settings_kw)
        cfg = yaml.safe_load((REPO / RATER_BASE).read_text())
        rs.apply_rater_overrides(cfg, settings, wandb_run_id="run1", condition="base",
                                 corpus_dir=None, num_chunks=lambda c: 10_000_000)
        config = ExperimentConfig(**cfg)
        identity = {"run_id": "sweep-sw1-run1", "arch": "bert_large", "lang": "en",
                    "condition": "rater_base", "seed": 42, "run_kind": "hp_sweep"}
        agent._run_rater_trial(None, config, identity, settings, str(REPO))
        return _FakeTrainer.last

    rec["sentinel"] = sentinel
    return run, rec


SMALL = dict(sweep_train_steps=1000, eval_every_steps=100, patience_evals=2,
             min_steps_before_stop=300, min_rel_improvement=0.01)


class TestRunRaterTrial:
    def test_plateau_stop_finishes_normally(self, trial):
        run, rec = trial
        trainer = run([50, 40, 30, 30, 30, 30], **SMALL)

        assert trainer.train_kwargs == {"on_epoch_end": None, "endpoint_checkpoint": "weights"}
        assert trainer.training_loop.global_step == 500          # bad evals at 400, 500
        eval_logs = [(s, p) for s, p in rec["logs"] if s is not None]
        assert [s for s, _ in eval_logs] == [100, 200, 300, 400, 500]
        assert [p["proxy/min_held_out_perplexity"] for _, p in eval_logs] == [50, 40, 30, 30, 30]
        assert eval_logs[0][1]["proxy/tokens_seen"] == 100 * 32 * 8 * 512

        summary = rec["logs"][-1][1]
        assert summary["proxy/min_held_out_perplexity"] == 30
        assert summary["proxy/stopped_early"] is True
        assert summary["proxy/best_step"] == 300
        assert summary["proxy/steps_completed"] == 500
        assert summary["proxy/n_evals"] == 5       # stop landed on an eval: no extra eval
        assert rec["registry_end"] == []           # no FAILED record
        assert rec["register_end"] == [(1.25, 30)]
        text = rec["sentinel"].read_text()
        assert "stopped_early=True" in text and "min_ppl=30" in text

    def test_budget_end_off_cadence_gets_a_final_eval(self, trial):
        run, rec = trial
        ppls = [100 * 0.95 ** i for i in range(11)]   # 5% better every eval
        trainer = run(ppls, **{**SMALL, "sweep_train_steps": 1050})
        assert trainer.training_loop.global_step == 1050
        steps = [s for s, _ in rec["logs"] if s is not None]
        assert steps == [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1050]
        summary = rec["logs"][-1][1]
        assert summary["proxy/stopped_early"] is False
        assert summary["proxy/min_held_out_perplexity"] == pytest.approx(100 * 0.95 ** 10)
        assert summary["proxy/best_step"] == 1050

    def test_save_final_false_requests_no_endpoint(self, trial):
        run, _ = trial
        trainer = run([10.0] * 10, **{**SMALL, "save_final": False})
        assert trainer.train_kwargs["endpoint_checkpoint"] == "none"

    def test_all_nan_is_a_failed_trial(self, trial):
        run, rec = trial
        with pytest.raises(RuntimeError, match="non-finite"):
            run([float("nan")] * 10, **SMALL)
        assert rec["registry_end"] and rec["registry_end"][0]["status"] == "FAILED"
        assert rec["register_end"] == []
        assert not rec["sentinel"].exists()
