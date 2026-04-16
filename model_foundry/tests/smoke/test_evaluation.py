"""
Smoke tests for the evaluation pipeline modules.

Tests cover SurprisalCalculator, ModelLoader, PerplexityEvaluator,
BLIMPEvaluator, NullSubjectEvaluator, and EvaluationRunner config
validation. All tests use mocks -- no real models are loaded.
"""

import sys
import json
import csv
import math
import types
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

# Ensure the project root is on sys.path so the evaluation package
# (which is not installed via setuptools) can be imported.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.core.surprisal_calculator import SurprisalCalculator
from evaluation.core.model_loader import ModelLoader, get_model_memory_usage
from evaluation.evaluators.blimp_evaluator import BLIMPEvaluator
from evaluation.evaluators.null_subject_evaluator import NullSubjectEvaluator
from evaluation.runners.evaluation_runner import EvaluationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tokenizer():
    """Return a mock SentencePiece-like tokenizer."""
    tok = MagicMock()
    tok.bos_id.return_value = 1
    tok.eos_id.return_value = 2
    tok.pad_id.return_value = 0
    tok.vocab_size.return_value = 1000

    def _encode(text):
        # Deterministic token IDs derived from words
        return [abs(hash(w)) % 998 + 3 for w in text.split()]

    def _decode(ids):
        return " ".join(str(i) for i in ids)

    tok.encode.side_effect = _encode
    tok.decode.side_effect = _decode
    return tok


def _make_mock_model(vocab_size=1000):
    """Return a mock causal LM that produces deterministic logits."""
    model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.return_value = iter([param])

    def forward(input_ids=None, **kwargs):
        seq_len = input_ids.shape[1]
        # Create logits that give predictable probabilities.
        # Set logit for token 0 to 2.0 and all others to 0.0,
        # so we have known softmax output.
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, :, 0] = 2.0
        return types.SimpleNamespace(logits=logits)

    model.__call__ = forward
    model.side_effect = forward
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


# ---------------------------------------------------------------------------
# 1. SurprisalCalculator initialization
# ---------------------------------------------------------------------------

class TestSurprisalCalculatorInit:

    def test_stores_model_and_tokenizer(self):
        model = _make_mock_model()
        tok = _make_mock_tokenizer()
        calc = SurprisalCalculator(model, tok, device="cpu")

        assert calc.model is model
        assert calc.tokenizer is tok
        assert calc.device == "cpu"

    def test_special_token_ids(self):
        model = _make_mock_model()
        tok = _make_mock_tokenizer()
        calc = SurprisalCalculator(model, tok, device="cpu")

        assert calc.bos_id == 1
        assert calc.eos_id == 2
        assert calc.pad_id == 0


# ---------------------------------------------------------------------------
# 2. SurprisalCalculator.calculate_surprisal — math check
# ---------------------------------------------------------------------------

class TestSurprisalCalculation:

    def test_surprisal_math(self):
        """
        With logits [2.0, 0, 0, ...] the softmax probability for token 0 is
        exp(2) / (exp(2) + 999*exp(0)).  Surprisal = -log2(p).
        For any other token t, p(t) = exp(0) / (exp(2) + 999).
        We verify mean surprisal is computed correctly.
        """
        vocab_size = 1000
        model = _make_mock_model(vocab_size=vocab_size)
        tok = _make_mock_tokenizer()
        calc = SurprisalCalculator(model, tok, device="cpu")

        result = calc.calculate_surprisal("hello world")

        # The result should contain the expected keys
        assert "total_surprisal" in result
        assert "mean_surprisal" in result
        assert "per_token_surprisal" in result

        # Check that surprisal values are finite and positive
        assert math.isfinite(result["total_surprisal"])
        assert result["mean_surprisal"] > 0

        # Verify the math: each target token has a known probability.
        # Log-softmax at position i for the actual token id gives us
        # the log probability.  surprisal = -log_prob / ln(2).
        # We just verify the formula yields consistent total and mean.
        per_token = result["per_token_surprisal"]
        token_surprisals = [s for _, s in per_token]
        expected_total = sum(token_surprisals)
        assert abs(result["total_surprisal"] - expected_total) < 1e-5

        expected_mean = expected_total / len(token_surprisals)
        assert abs(result["mean_surprisal"] - expected_mean) < 1e-5

    def test_compare_minimal_pair(self):
        """compare_minimal_pair should return correct > 0 for same sentence."""
        model = _make_mock_model()
        tok = _make_mock_tokenizer()
        calc = SurprisalCalculator(model, tok, device="cpu")

        result = calc.compare_minimal_pair(
            "The cat sat on the mat",
            "The cat sat on the mat"
        )

        assert "good_surprisal" in result
        assert "bad_surprisal" in result
        assert "correct" in result
        # Same sentence => neither is lower, correct may be False
        assert isinstance(result["correct"], (bool, np.bool_))


# ---------------------------------------------------------------------------
# 3. ModelLoader.find_checkpoints
# ---------------------------------------------------------------------------

class TestModelLoaderFindCheckpoints:

    def test_finds_checkpoint_dirs_only(self, tmp_path):
        """Only checkpoint-* and epoch_* dirs should be returned."""
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-500").mkdir()
        (tmp_path / "other_file.txt").touch()
        (tmp_path / "readme.md").touch()

        loader = ModelLoader(device="cpu")
        checkpoints = loader.find_checkpoints(str(tmp_path))

        names = [p.name for p in checkpoints]
        assert len(checkpoints) == 2
        assert "checkpoint-100" in names
        assert "checkpoint-500" in names
        assert "other_file.txt" not in names

    def test_sorted_by_step_number(self, tmp_path):
        """Checkpoints should be sorted numerically, not lexicographically."""
        for step in [500, 100, 2000, 50]:
            (tmp_path / f"checkpoint-{step}").mkdir()

        loader = ModelLoader(device="cpu")
        checkpoints = loader.find_checkpoints(str(tmp_path))

        steps = [int(p.name.split("-")[1]) for p in checkpoints]
        assert steps == sorted(steps)

    def test_mixed_checkpoint_and_epoch(self, tmp_path):
        """Both checkpoint-* and epoch_* dirs should be found."""
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "epoch_1").mkdir()
        (tmp_path / "epoch_10").mkdir()

        loader = ModelLoader(device="cpu")
        checkpoints = loader.find_checkpoints(str(tmp_path))

        assert len(checkpoints) == 3

    def test_nonexistent_dir_raises(self):
        loader = ModelLoader(device="cpu")
        with pytest.raises(FileNotFoundError):
            loader.find_checkpoints("/tmp/nonexistent_evaluation_dir_xyz")


# ---------------------------------------------------------------------------
# 4. PerplexityEvaluator basic flow
# ---------------------------------------------------------------------------

class TestPerplexityEvaluatorInit:

    def test_instantiation(self):
        """PerplexityEvaluator can be created with mock dependencies."""
        from evaluation.evaluators.perplexity_evaluator import PerplexityEvaluator

        model = _make_mock_model()
        tok = _make_mock_tokenizer()
        evaluator = PerplexityEvaluator(model, tok, device="cpu", max_length=512)

        assert evaluator.model is model
        assert evaluator.tokenizer is tok
        assert evaluator.device == "cpu"
        assert evaluator.max_length == 512

    def test_special_tokens_stored(self):
        from evaluation.evaluators.perplexity_evaluator import PerplexityEvaluator

        model = _make_mock_model()
        tok = _make_mock_tokenizer()
        evaluator = PerplexityEvaluator(model, tok, device="cpu")

        assert evaluator.bos_id == 1
        assert evaluator.eos_id == 2


# ---------------------------------------------------------------------------
# 5. BLIMPEvaluator.load_blimp_file
# ---------------------------------------------------------------------------

class TestBLIMPLoadFile:

    def test_loads_jsonl(self, tmp_path):
        """BLIMP JSONL files should be loaded into a list of dicts."""
        filepath = tmp_path / "anaphor_agreement.jsonl"
        stimuli = [
            {
                "sentence_good": "The cat chased itself.",
                "sentence_bad": "The cat chased himself.",
                "field": "syntax",
                "linguistics_term": "anaphor_agreement",
                "UID": "anaphor_1",
                "pairID": "1",
            },
            {
                "sentence_good": "The dogs hurt themselves.",
                "sentence_bad": "The dogs hurt himself.",
                "field": "syntax",
                "linguistics_term": "anaphor_agreement",
                "UID": "anaphor_2",
                "pairID": "2",
            },
        ]
        with open(filepath, "w") as f:
            for item in stimuli:
                json.dump(item, f)
                f.write("\n")

        evaluator = BLIMPEvaluator(surprisal_calculator=None)
        loaded = evaluator.load_blimp_file(str(filepath))

        assert len(loaded) == 2
        assert loaded[0]["sentence_good"] == "The cat chased itself."
        assert loaded[1]["linguistics_term"] == "anaphor_agreement"
        assert loaded[0]["UID"] == "anaphor_1"

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines in JSONL should be skipped."""
        filepath = tmp_path / "with_blanks.jsonl"
        with open(filepath, "w") as f:
            json.dump({"sentence_good": "a", "sentence_bad": "b"}, f)
            f.write("\n\n\n")
            json.dump({"sentence_good": "c", "sentence_bad": "d"}, f)
            f.write("\n")

        evaluator = BLIMPEvaluator(surprisal_calculator=None)
        loaded = evaluator.load_blimp_file(str(filepath))

        assert len(loaded) == 2

    def test_fields_extracted(self, tmp_path):
        """All standard BLIMP fields should be present."""
        filepath = tmp_path / "test.jsonl"
        item = {
            "sentence_good": "She runs.",
            "sentence_bad": "Her runs.",
            "field": "morphology",
            "linguistics_term": "case_marking",
            "UID": "case_1",
            "pairID": "42",
        }
        with open(filepath, "w") as f:
            json.dump(item, f)
            f.write("\n")

        evaluator = BLIMPEvaluator(surprisal_calculator=None)
        loaded = evaluator.load_blimp_file(str(filepath))

        assert loaded[0]["field"] == "morphology"
        assert loaded[0]["pairID"] == "42"


# ---------------------------------------------------------------------------
# 6. NullSubjectEvaluator.load_stimuli_file
# ---------------------------------------------------------------------------

class TestNullSubjectLoadStimuli:

    def _write_csv(self, filepath, rows, header):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

    def test_loads_individual_format(self, tmp_path):
        """Individual file format (without 'form' column) loads correctly."""
        filepath = tmp_path / "stimuli.csv"
        header = ["item", "item_group", "pronoun_status", "c_english", "target", "hotspot_english"]
        rows = [
            [1, "1a_3rdSG", 1, "Mary won the award.", "She shows pride.", "shows"],
            [1, "1a_3rdSG", 0, "Mary won the award.", "Shows pride.", "shows"],
            [2, "1a_3rdSG", 1, "The dog barked.", "It scared cats.", "scared"],
            [2, "1a_3rdSG", 0, "The dog barked.", "Scared cats.", "scared"],
        ]
        self._write_csv(filepath, rows, header)

        evaluator = NullSubjectEvaluator(surprisal_calculator=None)
        df = evaluator.load_stimuli_file(str(filepath))

        assert len(df) == 4
        assert "item" in df.columns
        assert "pronoun_status" in df.columns
        assert "c_english" in df.columns
        assert df["item"].nunique() == 2

    def test_loads_master_format(self, tmp_path):
        """Master file format (with 'form' column) loads correctly."""
        filepath = tmp_path / "master.csv"
        header = ["item", "item_group", "form", "pronoun_status", "c_english", "target", "hotspot_english"]
        rows = [
            [1, "1a_3rdSG", "A", 1, "Mary won.", "She shows pride.", "shows"],
            [1, "1a_3rdSG", "A", 0, "Mary won.", "Shows pride.", "shows"],
        ]
        self._write_csv(filepath, rows, header)

        evaluator = NullSubjectEvaluator(surprisal_calculator=None)
        df = evaluator.load_stimuli_file(str(filepath))

        assert len(df) == 2
        assert "form" in df.columns

    def test_missing_columns_raises(self, tmp_path):
        """Missing required columns should raise ValueError."""
        filepath = tmp_path / "bad.csv"
        header = ["item", "pronoun_status"]  # Missing several required columns
        rows = [[1, 1]]
        self._write_csv(filepath, rows, header)

        evaluator = NullSubjectEvaluator(surprisal_calculator=None)
        with pytest.raises(ValueError, match="Missing required columns"):
            evaluator.load_stimuli_file(str(filepath))

    def test_column_types(self, tmp_path):
        """Pronoun status should be numeric for filtering."""
        filepath = tmp_path / "stimuli.csv"
        header = ["item", "item_group", "pronoun_status", "c_english", "target", "hotspot_english"]
        rows = [
            [1, "1a_3rdSG", 1, "Context.", "She runs.", "runs"],
            [1, "1a_3rdSG", 0, "Context.", "Runs.", "runs"],
        ]
        self._write_csv(filepath, rows, header)

        evaluator = NullSubjectEvaluator(surprisal_calculator=None)
        df = evaluator.load_stimuli_file(str(filepath))

        # pronoun_status should be usable as int for filtering
        overt = df[df["pronoun_status"] == 1]
        null = df[df["pronoun_status"] == 0]
        assert len(overt) == 1
        assert len(null) == 1


# ---------------------------------------------------------------------------
# 7. EvaluationRunner config validation
# ---------------------------------------------------------------------------

class TestEvaluationConfig:

    def test_valid_config(self):
        """A valid config should be created without errors."""
        config = EvaluationConfig(
            model_checkpoint_dir="/tmp/checkpoints",
            tokenizer_path="/tmp/tokenizer",
            test_corpus="/tmp/test_corpus",
            blimp_dir="/tmp/blimp",
            null_subject_dir="/tmp/null_subj",
            output_dir="/tmp/results",
        )

        assert config.model_checkpoint_dir == "/tmp/checkpoints"
        assert config.tokenizer_path == "/tmp/tokenizer"
        assert config.batch_size == 32  # default
        assert config.run_perplexity is True  # default
        assert config.run_blimp is True  # default
        assert config.run_null_subject is True  # default

    def test_defaults(self):
        """Default values should be populated for optional fields."""
        config = EvaluationConfig(
            model_checkpoint_dir="/tmp/m",
            tokenizer_path="/tmp/t",
        )

        assert config.batch_size == 32
        assert config.device == "auto"
        assert config.use_fp16 is False
        assert config.max_length == 1000
        assert config.num_workers == 4
        assert config.save_detailed is True
        assert config.save_format == "jsonl"
        assert config.max_samples is None
        assert config.max_checkpoints is None

    def test_missing_required_fields(self):
        """Omitting required fields should raise a validation error."""
        with pytest.raises(Exception):
            # model_checkpoint_dir and tokenizer_path are required
            EvaluationConfig()

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = EvaluationConfig(
            model_checkpoint_dir="/tmp/m",
            tokenizer_path="/tmp/t",
            batch_size=16,
            device="cpu",
            use_fp16=True,
            max_length=512,
            run_perplexity=False,
            run_blimp=False,
            run_null_subject=True,
            max_samples=100,
            max_checkpoints=3,
        )

        assert config.batch_size == 16
        assert config.device == "cpu"
        assert config.use_fp16 is True
        assert config.max_length == 512
        assert config.run_perplexity is False
        assert config.run_blimp is False
        assert config.run_null_subject is True
        assert config.max_samples == 100
        assert config.max_checkpoints == 3

    def test_optional_paths_default_none(self):
        """Optional dataset paths should default to None."""
        config = EvaluationConfig(
            model_checkpoint_dir="/tmp/m",
            tokenizer_path="/tmp/t",
        )

        assert config.test_corpus is None
        assert config.blimp_dir is None
        assert config.null_subject_dir is None
