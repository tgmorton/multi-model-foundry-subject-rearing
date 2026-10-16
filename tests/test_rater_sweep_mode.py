"""Tests for scripts/rater_sweep.py (rater-sweep mode of sweep_agent_lm).

Run:  <venv>/bin/python -m pytest tests/test_rater_sweep_mode.py

Pure logic only: settings parsing, config overrides (config in -> config
out), the plateau early-stop decision, and the render-config CLI.
"""

import copy
import math
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model_foundry.config import ExperimentConfig  # noqa: E402
from scripts import rater_sweep as rs  # noqa: E402

RATER_BASE = REPO / "configs" / "sweeps" / "baselines" / "rater_bert_large_en.yaml"


def _base_cfg() -> dict:
    return yaml.safe_load(RATER_BASE.read_text())


# --------------------------------------------------------------------------- #
# switches + settings
# --------------------------------------------------------------------------- #
class TestEnablement:
    def test_rater_baseline_enables(self):
        assert rs.rater_mode_enabled(_base_cfg(), {})

    def test_env_enables_without_block(self):
        assert rs.rater_mode_enabled({}, {"RATER_SWEEP": "1"})
        assert rs.rater_mode_enabled({}, {"RATER_SWEEP": "true"})
        assert not rs.rater_mode_enabled({}, {"RATER_SWEEP": "0"})
        assert not rs.rater_mode_enabled({}, {})

    def test_block_without_enabled_true_is_off(self):
        assert not rs.rater_mode_enabled({"rater_sweep": {"sweep_train_steps": 5}}, {})
        assert not rs.rater_mode_enabled({"rater_sweep": {"enabled": "yes"}}, {})

    def test_no_existing_baseline_enables_rater_mode(self):
        """Inertness: every pre-existing sweep baseline stays on the epoch path."""
        others = [p for p in (REPO / "configs" / "sweeps" / "baselines").glob("*.yaml")
                  if not p.name.startswith("rater_")]
        assert others, "expected the existing sweep baselines"
        for p in others:
            assert not rs.rater_mode_enabled(yaml.safe_load(p.read_text()), {}), p.name


class TestSettings:
    def test_baseline_values(self):
        s = rs.load_settings(_base_cfg())
        assert s == rs.RaterSweepSettings(
            sweep_train_steps=3000, eval_every_steps=250, eval_max_sequences=2000,
            patience_evals=4, min_steps_before_stop=1000, min_rel_improvement=0.002,
            save_final=True,
        )

    def test_missing_block_gives_defaults(self):
        assert rs.load_settings({}) == rs.RaterSweepSettings()

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="patience_eval"):
            rs.load_settings({"rater_sweep": {"enabled": True, "patience_eval": 3}})

    @pytest.mark.parametrize("field,value", [
        ("sweep_train_steps", 0), ("eval_every_steps", -1), ("patience_evals", 1.5),
        ("eval_max_sequences", True), ("min_steps_before_stop", -1),
        ("min_rel_improvement", 1.0), ("save_final", "yes"),
    ])
    def test_invalid_values_rejected(self, field, value):
        with pytest.raises(ValueError):
            rs.RaterSweepSettings(**{field: value})


class TestConditionAndGroup:
    def test_env_condition_wins_and_defaults(self):
        assert rs.resolve_condition({"CONDITION": "lemmatize_verbs"}, {}) == "lemmatize_verbs"
        assert rs.resolve_condition({}, {"condition": "base"}) == "base"
        assert rs.resolve_condition({}, {}) == "baseline"

    def test_env_vs_sweep_constant_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="wrong sweep"):
            rs.resolve_condition({"CONDITION": "base"}, {"condition": "lemmatize_verbs"})
        assert rs.resolve_condition({"CONDITION": "base"}, {"condition": "base"}) == "base"

    def test_corpus_dir_leaf_must_match_known_corpus(self):
        rs.check_corpus_matches_condition("/x/sweep_subset_v1/base/", "base")
        rs.check_corpus_matches_condition("/x/my_custom_subset", "base")
        rs.check_corpus_matches_condition(None, "base")
        with pytest.raises(RuntimeError, match="lemmatize_verbs"):
            rs.check_corpus_matches_condition("/x/sweep_subset_v1/lemmatize_verbs/", "base")

    def test_group_only_in_rater_mode_with_condition(self):
        assert rs.group_from_env({"RATER_SWEEP": "1", "CONDITION": "base"}) == "rater-sweep-base"
        assert rs.group_from_env({"CONDITION": "base"}) is None
        assert rs.group_from_env({"RATER_SWEEP": "1"}) is None

    def test_registry_condition_prefixed(self):
        assert rs.registry_condition("base") == "rater_base"

    def test_endpoint_mode(self):
        assert rs.endpoint_checkpoint_mode(rs.RaterSweepSettings()) == "weights"
        assert rs.endpoint_checkpoint_mode(rs.RaterSweepSettings(save_final=False)) == "none"


# --------------------------------------------------------------------------- #
# config in -> config out
# --------------------------------------------------------------------------- #
class TestApplyRaterOverrides:
    CORPUS = "/mnt/data/rater_corpus/sweep_subset_v1/lemmatize_verbs/"

    def _apply(self, cfg, settings=None, num_chunks=50_000, corpus_dir=CORPUS):
        seen = {}

        def fake_num_chunks(c):
            # The cache key depends on the corpus, so the override must
            # already be in place when the chunk count is read.
            seen["corpus"] = c["data"]["training_corpus"]
            seen["experiment_name"] = c["experiment_name"]
            return num_chunks

        rs.apply_rater_overrides(
            cfg, settings or rs.load_settings(cfg), wandb_run_id="abc123",
            condition="lemmatize_verbs", corpus_dir=corpus_dir, num_chunks=fake_num_chunks,
        )
        return seen

    def test_config_in_config_out(self):
        before = _base_cfg()
        cfg = copy.deepcopy(before)
        seen = self._apply(cfg)

        assert seen == {"corpus": self.CORPUS, "experiment_name": "sweep-abc123"}
        assert "rater_sweep" not in cfg
        assert cfg["experiment_name"] == "sweep-abc123"
        for key in ("training_corpus", "source_corpus", "test_corpus"):
            assert cfg["data"][key] == self.CORPUS
        t = cfg["training"]
        assert t["output_dir"] == "models/rater_sweeps/lemmatize_verbs/abc123"
        assert t["train_steps"] == t["scheduler_train_steps"] == 3000
        assert t["warmup_steps"] is None
        # 50_000 chunks / phys 32 -> 1563 batches // accum 8 -> 195 steps/epoch
        assert t["epochs"] == math.ceil(3000 / 195) == 16
        assert t["checkpoint_schedule"] == []
        assert t["auto_generate_checkpoints"] is False
        assert t["resume_from_checkpoint"] is False
        assert t["resume_state_steps"] is None
        assert t["rolling_resume_every_steps"] is None

        # Nothing else moved.
        touched_training = {"output_dir", "train_steps", "scheduler_train_steps",
                            "warmup_steps", "epochs", "checkpoint_schedule",
                            "auto_generate_checkpoints", "resume_from_checkpoint",
                            "resume_state_steps", "rolling_resume_every_steps"}
        for k, v in before["training"].items():
            if k not in touched_training:
                assert t[k] == v, k
        for section in ("model", "tokenizer", "logging", "dataset_manipulation", "random_seed"):
            assert cfg[section] == before[section], section
        assert cfg["data"]["batch_size"] == before["data"]["batch_size"]
        assert cfg["data"]["max_sequence_length"] == 512

        # And the result is a valid ExperimentConfig.
        config = ExperimentConfig(**cfg)
        assert config.training.train_steps == 3000
        assert config.training.scheduler_train_steps == 3000

    def test_big_corpus_one_epoch(self):
        cfg = _base_cfg()
        self._apply(cfg, num_chunks=10_000_000)
        assert cfg["training"]["epochs"] == 1

    def test_no_corpus_dir_keeps_baseline_corpus(self):
        cfg = _base_cfg()
        before = copy.deepcopy(cfg["data"])
        seen = self._apply(cfg, corpus_dir=None)
        assert cfg["data"] == before
        assert seen["corpus"] == before["training_corpus"]

    def test_step_budget_follows_settings(self):
        cfg = _base_cfg()
        self._apply(cfg, settings=rs.RaterSweepSettings(sweep_train_steps=500))
        assert cfg["training"]["train_steps"] == cfg["training"]["scheduler_train_steps"] == 500


# --------------------------------------------------------------------------- #
# early stop
# --------------------------------------------------------------------------- #
def _run(ppls, settings, every=None):
    """Feed a perplexity series at steps every, 2*every, ...; return the
    final state and the step at which should_stop first fired (or None)."""
    every = every or settings.eval_every_steps
    state, stop_at = rs.EarlyStopState(), None
    for i, ppl in enumerate(ppls, start=1):
        state = rs.early_stop_update(state, i * every, ppl, settings)
        if state.should_stop and stop_at is None:
            stop_at = i * every
    return state, stop_at


S = rs.RaterSweepSettings(eval_every_steps=100, patience_evals=3,
                          min_steps_before_stop=500, min_rel_improvement=0.01)


class TestEarlyStop:
    def test_steady_improvement_never_stops(self):
        state, stop_at = _run([100 * 0.9 ** i for i in range(20)], S)
        assert stop_at is None
        assert state.bad_evals == 0 and state.n_evals == 20
        assert state.best_step == 2000

    def test_plateau_stops_after_patience(self):
        # Improves through step 600, flat afterwards: bad evals at 700, 800, 900.
        state, stop_at = _run([50, 40, 30, 25, 22, 20, 20, 20, 20, 20], S)
        assert stop_at == 900
        assert state.min_ppl == 20 and state.best_step == 600

    def test_no_stop_before_min_steps(self):
        # Flat from the 2nd eval: patience is exhausted at step 400 but the
        # stop is held until min_steps_before_stop (500).
        _, stop_at = _run([30, 30, 30, 30, 30, 30], S)
        assert stop_at == 500

    def test_sub_threshold_creep_counts_as_plateau(self):
        # 0.2%/eval gains: 0.6% cumulative over the reference after 3 evals,
        # under the 1% threshold, so patience runs out; the reported running
        # minimum still tracks them.
        ppls = [50, 40, 30, 25, 22, 20, 19.96, 19.92, 19.88]
        state, stop_at = _run(ppls, S)
        assert stop_at == 900
        assert state.min_ppl == 19.88 and state.best_step == 900
        assert state.ref_ppl == 20

    def test_creep_is_cumulative_against_the_reference(self):
        # 0.6%/eval: the 2nd step of creep is 1.2% below the reference, a
        # significant improvement, so a steady slow gain is not a plateau.
        ppls = [50, 40, 30, 25, 22, 20, 19.88, 19.76, 19.64, 19.52, 19.40]
        state, stop_at = _run(ppls, S)
        assert stop_at is None
        assert state.ref_ppl == 19.52

    def test_significant_gain_resets_patience(self):
        # Two flat evals, then a >1% gain, then flat again.
        ppls = [50, 40, 30, 25, 22, 20, 20, 20, 18, 18, 18, 18]
        state, stop_at = _run(ppls, S)
        assert stop_at == 1200
        assert state.best_step == 900

    def test_non_finite_is_a_bad_eval_and_never_the_min(self):
        ppls = [50, 40, 30, 25, 22, 20, float("nan"), float("inf"), None]
        state, stop_at = _run(ppls, S)
        assert stop_at == 900
        assert state.min_ppl == 20

    def test_all_non_finite_keeps_min_infinite(self):
        state, _ = _run([float("nan")] * 3, S)
        assert math.isinf(state.min_ppl) and state.best_step == -1

    def test_zero_threshold_needs_strict_improvement(self):
        s0 = rs.RaterSweepSettings(eval_every_steps=100, patience_evals=2,
                                   min_steps_before_stop=0, min_rel_improvement=0.0)
        _, stop_at = _run([10, 9.99, 9.99, 9.99], s0)
        assert stop_at == 400

    def test_is_eval_step(self):
        s = rs.RaterSweepSettings(eval_every_steps=250)
        assert [x for x in range(0, 1001) if rs.is_eval_step(x, s)] == [250, 500, 750, 1000]


# --------------------------------------------------------------------------- #
# render-config CLI
# --------------------------------------------------------------------------- #
class TestRenderConfig:
    def test_render(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        rc = rs.main(["render-config", "--base", str(RATER_BASE),
                      "--corpus-dir", "/mnt/x/impoverish_case_en", "--out", str(out)])
        assert rc == 0
        cfg = yaml.safe_load(out.read_text())
        assert "rater_sweep" not in cfg
        assert {cfg["data"][k] for k in ("training_corpus", "source_corpus", "test_corpus")} \
            == {"/mnt/x/impoverish_case_en"}
        # Cache-key inputs other than the corpus are the baseline's.
        base = _base_cfg()
        assert cfg["tokenizer"] == base["tokenizer"]
        assert cfg["data"]["max_sequence_length"] == base["data"]["max_sequence_length"]
        assert cfg["dataset_manipulation"] == base["dataset_manipulation"]
        ExperimentConfig(**cfg)

    def test_refuses_non_rater_base(self, tmp_path):
        rc = rs.main(["render-config",
                      "--base", str(REPO / "configs/sweeps/baselines/bert_large_en.yaml"),
                      "--corpus-dir", "/mnt/x/base", "--out", str(tmp_path / "c.yaml")])
        assert rc == 2
        assert not (tmp_path / "c.yaml").exists()


# --------------------------------------------------------------------------- #
# W&B sweep specs
# --------------------------------------------------------------------------- #
class TestSweepSpecs:
    SPECS = sorted((REPO / "configs" / "sweeps").glob("rater_bert_large_*.yaml"))
    # Constants + the HP fields scripts/sweep_agent_lm.py overlays.
    ALLOWED = {"arch", "lang", "base_config", "condition", "learning_rate",
               "warmup_ratio", "dropout", "attention_dropout",
               "effective_batch_size", "adam_beta2"}

    def test_one_spec_per_corpus(self):
        assert sorted(p.stem.replace("rater_bert_large_", "") for p in self.SPECS) \
            == sorted(rs.KNOWN_CORPORA)

    @pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
    def test_spec(self, path):
        spec = yaml.safe_load(path.read_text())
        corpus = path.stem.replace("rater_bert_large_", "")
        assert "early_terminate" not in spec            # see commit 38621840
        assert spec["method"] == "bayes"
        assert spec["metric"] == {"name": "proxy/min_held_out_perplexity", "goal": "minimize"}
        assert spec["program"] == "scripts/sweep_agent_lm.py"
        assert spec["project"] == "subject-drop-rater-sweeps"
        params = spec["parameters"]
        assert set(params) <= self.ALLOWED
        assert params["condition"] == {"value": corpus}
        base = REPO / params["base_config"]["value"]
        assert base == RATER_BASE
        assert set(params["effective_batch_size"]["values"]) == {128, 256}
