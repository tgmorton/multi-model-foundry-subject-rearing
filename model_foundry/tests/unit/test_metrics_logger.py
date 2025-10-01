"""
Unit tests for the MetricsLogger class.

Tests cover:
- Metrics file creation and JSONL format
- Step-level metric logging
- Epoch-level summary logging
- Appending vs overwriting
- Metrics history retrieval
- Filtering by step range
- Statistical computations
- Gradient norm logging
- Learning rate tracking
- Throughput metrics
- NaN/Inf handling
- Concurrent write safety
"""

import json
import time
from pathlib import Path
import pytest
import threading

from model_foundry.logging_components import MetricsLogger


class TestMetricsLoggerBasics:
    """Test basic MetricsLogger functionality."""

    def test_creates_metrics_file(self, tmp_path):
        """
        GIVEN: Output directory path
        WHEN: Creating MetricsLogger
        THEN: Should set metrics_file path to <dir>/metrics.jsonl
        """
        logger = MetricsLogger("test_exp", tmp_path)

        assert logger.experiment_name == "test_exp"
        assert logger.output_dir == tmp_path
        assert logger.metrics_file == tmp_path / "metrics.jsonl"

    def test_log_step_writes_jsonl(self, tmp_path):
        """
        GIVEN: MetricsLogger instance
        WHEN: Logging metrics for a training step
        THEN: Should write JSON line with step, epoch, metrics, timestamp
        """
        logger = MetricsLogger("test_exp", tmp_path)

        metrics = {
            "loss": 2.5,
            "lr": 0.001,
            "grad_norm": 1.23
        }
        logger.log_step(step=100, epoch=2, metrics=metrics)

        # Read JSONL file
        assert logger.metrics_file.exists()

        with open(logger.metrics_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)

        # Verify structure
        assert entry["step"] == 100
        assert entry["epoch"] == 2
        assert entry["metrics"]["loss"] == 2.5
        assert entry["metrics"]["lr"] == 0.001
        assert entry["metrics"]["grad_norm"] == 1.23
        assert "timestamp" in entry

    def test_log_step_appends_to_file(self, tmp_path):
        """
        GIVEN: MetricsLogger with existing metrics
        WHEN: Logging additional steps
        THEN: Should append, not overwrite
        """
        logger = MetricsLogger("test_exp", tmp_path)

        logger.log_step(100, 1, {"loss": 2.5})
        logger.log_step(200, 2, {"loss": 2.3})
        logger.log_step(300, 3, {"loss": 2.1})

        # Should have 3 lines
        with open(logger.metrics_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify step numbers
        assert json.loads(lines[0])["step"] == 100
        assert json.loads(lines[1])["step"] == 200
        assert json.loads(lines[2])["step"] == 300


class TestMetricsLoggerAggregation:
    """Test metrics aggregation and retrieval."""

    def test_log_epoch_summary(self, tmp_path):
        """
        GIVEN: Metrics for multiple steps in an epoch
        WHEN: Logging epoch summary
        THEN: Should write summary with aggregate statistics
        """
        logger = MetricsLogger("test_exp", tmp_path)

        summary = {
            "avg_loss": 2.4,
            "min_loss": 2.1,
            "max_loss": 2.8,
            "total_tokens": 1000000
        }
        logger.log_epoch_summary(epoch=5, summary=summary)

        # Read from file
        with open(logger.metrics_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry["epoch"] == 5
        assert "summary" in entry
        assert entry["summary"]["avg_loss"] == 2.4
        assert entry["summary"]["min_loss"] == 2.1
        assert entry["summary"]["total_tokens"] == 1000000

    def test_get_metrics_history(self, tmp_path):
        """
        GIVEN: Multiple logged metrics
        WHEN: Calling get_metrics_history()
        THEN: Should return list of all metric entries
        """
        logger = MetricsLogger("test_exp", tmp_path)

        # Log several steps
        for step in range(0, 500, 100):
            logger.log_step(step, 0, {"loss": 3.0 - step/1000})

        # Retrieve history
        history = logger.get_metrics_history()

        assert len(history) == 5
        assert history[0]["step"] == 0
        assert history[-1]["step"] == 400
        assert all("metrics" in entry for entry in history)

    def test_get_metrics_for_steps(self, tmp_path):
        """
        GIVEN: Metrics logged for steps 0-1000
        WHEN: Filtering for specific step range
        THEN: Should return only metrics within that range
        """
        logger = MetricsLogger("test_exp", tmp_path)

        # Log steps 0-900 in increments of 100
        for step in range(0, 1000, 100):
            logger.log_step(step, step // 100, {"loss": 3.0 - step/1000})

        # Get metrics for steps 200-500
        filtered = logger.get_metrics_for_steps(start=200, end=500)

        assert len(filtered) == 4  # 200, 300, 400, 500
        assert filtered[0]["step"] == 200
        assert filtered[-1]["step"] == 500

    def test_compute_statistics(self, tmp_path):
        """
        GIVEN: Multiple loss values logged
        WHEN: Computing statistics for 'loss' metric
        THEN: Should return mean, min, max, std
        """
        logger = MetricsLogger("test_exp", tmp_path)

        losses = [3.0, 2.8, 2.6, 2.4, 2.2]
        for i, loss in enumerate(losses):
            logger.log_step(i * 100, 0, {"loss": loss})

        stats = logger.compute_statistics("loss")

        assert stats["mean"] == pytest.approx(2.6)
        assert stats["min"] == 2.2
        assert stats["max"] == 3.0
        assert stats["std"] > 0


class TestMetricsLoggerSpecificMetrics:
    """Test logging of specific metric types."""

    def test_log_gradient_norm(self, tmp_path):
        """
        GIVEN: Training step with gradient norm
        WHEN: Logging metrics including grad_norm
        THEN: Should be recorded in metrics
        """
        logger = MetricsLogger("test_exp", tmp_path)

        logger.log_step(100, 1, {
            "loss": 2.5,
            "grad_norm": 1.234
        })

        with open(logger.metrics_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry["metrics"]["grad_norm"] == 1.234

    def test_log_learning_rate_schedule(self, tmp_path):
        """
        GIVEN: Training with learning rate schedule
        WHEN: Logging LR at each step
        THEN: Should track LR changes over time
        """
        logger = MetricsLogger("test_exp", tmp_path)

        # Simulate warmup + decay
        learning_rates = [0.0001, 0.0005, 0.001, 0.0009, 0.0008, 0.0007]

        for i, lr in enumerate(learning_rates):
            logger.log_step(i * 100, 0, {"lr": lr, "loss": 2.5})

        history = logger.get_metrics_history()
        logged_lrs = [h["metrics"]["lr"] for h in history]

        assert logged_lrs == learning_rates

    def test_log_throughput_metrics(self, tmp_path):
        """
        GIVEN: Training step with throughput data
        WHEN: Logging tokens_per_sec, samples_per_sec
        THEN: Should record throughput metrics
        """
        logger = MetricsLogger("test_exp", tmp_path)

        logger.log_step(100, 1, {
            "loss": 2.5,
            "tokens_per_sec": 8500,
            "samples_per_sec": 42
        })

        with open(logger.metrics_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry["metrics"]["tokens_per_sec"] == 8500
        assert entry["metrics"]["samples_per_sec"] == 42


class TestMetricsLoggerEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_nan_inf_values(self, tmp_path):
        """
        GIVEN: Metrics with NaN or Inf values
        WHEN: Logging to JSONL
        THEN: Should handle gracefully (convert to None or string)
        """
        logger = MetricsLogger("test_exp", tmp_path)

        logger.log_step(100, 1, {
            "loss": float('nan'),
            "grad_norm": float('inf'),
            "lr": 0.001
        })

        # Should write successfully
        with open(logger.metrics_file, 'r') as f:
            entry = json.loads(f.readline())

        # NaN becomes None
        assert entry["metrics"]["loss"] is None

        # Inf becomes string
        assert entry["metrics"]["grad_norm"] == "Infinity"

    @pytest.mark.slow
    def test_concurrent_writes_safe(self, tmp_path):
        """
        GIVEN: Multiple threads writing metrics simultaneously
        WHEN: All threads complete
        THEN: All entries should be written without corruption
        """
        logger = MetricsLogger("test_exp", tmp_path)

        def write_metrics(start_step, count):
            for i in range(count):
                logger.log_step(start_step + i, 0, {"loss": 2.5, "thread_id": start_step})
                time.sleep(0.001)  # Small delay

        # Create 5 threads, each writing 10 entries
        threads = [
            threading.Thread(target=write_metrics, args=(i * 100, 10))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 50 entries total
        with open(logger.metrics_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 50

        # All should be valid JSON
        for line in lines:
            entry = json.loads(line)  # Should not raise
            assert "step" in entry
            assert "metrics" in entry
