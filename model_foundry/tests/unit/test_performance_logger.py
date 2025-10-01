"""
Unit tests for the PerformanceLogger class.

Tests cover:
- Timing code blocks
- Logging timing results
- Tracking multiple invocations
- Exception handling in timed blocks
- Timing statistics computation
- Memory usage logging (CPU and GPU)
- Timer reset functionality
- Exporting timing reports
- Nested timing blocks
"""

import time
import logging
import json
import pytest
import torch

from model_foundry.logging_components import PerformanceLogger


class TestPerformanceLoggerTiming:
    """Test timing functionality."""

    def test_time_block_measures_duration(self):
        """
        GIVEN: PerformanceLogger instance
        WHEN: Using time_block context manager
        THEN: Should measure and store execution time
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with perf_logger.time_block("test_operation"):
            time.sleep(0.1)

        assert "test_operation" in perf_logger.timers
        assert len(perf_logger.timers["test_operation"]) == 1

        # Should be approximately 0.1 seconds (with tolerance)
        assert perf_logger.timers["test_operation"][0] >= 0.1
        assert perf_logger.timers["test_operation"][0] < 0.15  # Allow some overhead

    def test_time_block_logs_duration(self, caplog):
        """
        GIVEN: PerformanceLogger with logger at DEBUG level
        WHEN: Timing a code block
        THEN: Should log completion message with duration
        """
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        perf_logger = PerformanceLogger(logger)

        with caplog.at_level(logging.DEBUG):
            with perf_logger.time_block("test_operation"):
                time.sleep(0.05)

        # Should have logged the timing
        assert "test_operation completed in" in caplog.text
        assert "s" in caplog.text  # seconds unit

    def test_time_block_tracks_multiple_calls(self):
        """
        GIVEN: PerformanceLogger
        WHEN: Timing the same operation multiple times
        THEN: Should track all invocations separately
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        for _ in range(5):
            with perf_logger.time_block("repeated_op"):
                time.sleep(0.01)

        assert len(perf_logger.timers["repeated_op"]) == 5

        # All should be around 0.01 seconds
        for duration in perf_logger.timers["repeated_op"]:
            assert duration >= 0.01

    def test_time_block_handles_exceptions(self):
        """
        GIVEN: Code block that raises exception
        WHEN: Using time_block context manager
        THEN: Should still record timing before re-raising
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with pytest.raises(ValueError):
            with perf_logger.time_block("failing_op"):
                time.sleep(0.01)
                raise ValueError("Test error")

        # Should still have timing recorded
        assert "failing_op" in perf_logger.timers
        assert len(perf_logger.timers["failing_op"]) == 1
        assert perf_logger.timers["failing_op"][0] >= 0.01


class TestPerformanceLoggerStatistics:
    """Test timing statistics computation."""

    def test_get_timing_statistics(self):
        """
        GIVEN: Multiple timing measurements for an operation
        WHEN: Computing statistics
        THEN: Should return count, mean, min, max, std
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        # Manually add some timings for testing
        perf_logger.timers["test_op"] = [0.1, 0.2, 0.15, 0.18, 0.12]

        stats = perf_logger.get_timing_statistics("test_op")

        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(0.15)
        assert stats["min"] == 0.1
        assert stats["max"] == 0.2
        assert stats["std"] > 0  # Should have some variance

    def test_get_timing_statistics_single_sample(self):
        """
        GIVEN: Only one timing measurement
        WHEN: Computing statistics
        THEN: Should return stats with std=0
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        perf_logger.timers["single_op"] = [0.5]

        stats = perf_logger.get_timing_statistics("single_op")

        assert stats["count"] == 1
        assert stats["mean"] == 0.5
        assert stats["min"] == 0.5
        assert stats["max"] == 0.5
        assert stats["std"] == 0.0


class TestPerformanceLoggerMemory:
    """Test memory usage logging."""

    def test_log_memory_usage_cpu(self, caplog):
        """
        GIVEN: PerformanceLogger on CPU system
        WHEN: Logging memory usage
        THEN: Should log CPU memory information
        """
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        perf_logger = PerformanceLogger(logger)

        with caplog.at_level(logging.DEBUG):
            perf_logger.log_memory_usage()

        # Should log something
        assert len(caplog.records) > 0

    @pytest.mark.gpu
    def test_log_memory_usage_gpu(self, caplog):
        """
        GIVEN: PerformanceLogger on GPU system
        WHEN: Logging memory usage
        THEN: Should log GPU memory allocated and reserved
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        perf_logger = PerformanceLogger(logger)

        with caplog.at_level(logging.DEBUG):
            perf_logger.log_memory_usage()

        assert "GPU memory" in caplog.text
        assert "Allocated" in caplog.text
        assert "Reserved" in caplog.text


class TestPerformanceLoggerManagement:
    """Test timer management functionality."""

    def test_reset_timers(self):
        """
        GIVEN: PerformanceLogger with timing data
        WHEN: Calling reset_timers()
        THEN: Should clear all timing data
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with perf_logger.time_block("test_op"):
            time.sleep(0.01)

        assert len(perf_logger.timers["test_op"]) == 1

        perf_logger.reset_timers()

        assert len(perf_logger.timers) == 0

    def test_export_timing_report(self, tmp_path):
        """
        GIVEN: PerformanceLogger with multiple operation timings
        WHEN: Exporting timing report
        THEN: Should write JSON file with all statistics
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        # Time several operations
        for _ in range(3):
            with perf_logger.time_block("op1"):
                time.sleep(0.01)
            with perf_logger.time_block("op2"):
                time.sleep(0.02)

        report_file = tmp_path / "timing_report.json"
        perf_logger.export_timing_report(report_file)

        # Verify file exists and is valid JSON
        assert report_file.exists()

        with open(report_file, 'r') as f:
            report = json.load(f)

        # Should have both operations
        assert "op1" in report
        assert "op2" in report

        # Each should have statistics
        assert report["op1"]["count"] == 3
        assert report["op2"]["count"] == 3
        assert "mean" in report["op1"]
        assert "mean" in report["op2"]

    def test_nested_time_blocks(self):
        """
        GIVEN: Nested timing blocks
        WHEN: Timing outer and inner operations
        THEN: Should track both separately, outer > inner duration
        """
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with perf_logger.time_block("outer"):
            time.sleep(0.05)
            with perf_logger.time_block("inner"):
                time.sleep(0.02)
            time.sleep(0.01)

        assert "outer" in perf_logger.timers
        assert "inner" in perf_logger.timers

        # Outer should be longer than inner
        outer_time = perf_logger.timers["outer"][0]
        inner_time = perf_logger.timers["inner"][0]

        assert outer_time > inner_time
        assert outer_time >= 0.08  # 0.05 + 0.02 + 0.01
        assert inner_time >= 0.02
