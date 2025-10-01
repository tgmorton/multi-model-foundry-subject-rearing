"""
Logging components for the Model Foundry framework.

This module provides structured logging, metrics tracking, performance profiling,
and error tracking functionality.

Components:
    - StructuredLogger: JSON-formatted structured logging with context
    - MetricsLogger: Training metrics tracking in JSONL format
    - PerformanceLogger: Timing and memory profiling
    - ErrorTracker: Error aggregation and tracking
"""

import json
import logging
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

import torch

from .utils import get_git_commit_hash, get_device


class StructuredLogger:
    """
    Enhanced logger with structured logging support.

    Provides JSON-formatted logs with consistent context information across
    all log messages. Base context includes experiment name, git hash, and device.

    Example:
        >>> logger = StructuredLogger("model_foundry.trainer", config)
        >>> logger.info("Training started", step=0, epoch=1)
        # Logs: {"message": "Training started", "context": {"experiment": "exp0",
        #         "git_hash": "abc123", "device": "cuda:0", "step": 0, "epoch": 1}}
    """

    def __init__(self, name: str, experiment_config):
        """
        Initialize structured logger.

        Args:
            name: Logger name (e.g., "model_foundry.trainer")
            experiment_config: ExperimentConfig instance for base context
        """
        self.logger = logging.getLogger(name)
        self.context = self._build_base_context(experiment_config)

    def _build_base_context(self, config) -> Dict[str, Any]:
        """
        Build context that appears in all log messages.

        Args:
            config: Experiment configuration

        Returns:
            Dictionary with experiment, git_hash, device
        """
        return {
            "experiment": config.experiment_name,
            "git_hash": get_git_commit_hash(),
            "device": str(get_device())
        }

    def log_structured(self, level: int, message: str, **kwargs):
        """
        Log a message with structured context.

        Args:
            level: Logging level (e.g., logging.INFO)
            message: Log message
            **kwargs: Additional context fields to include
        """
        log_entry = {
            "message": message,
            "context": {**self.context, **kwargs}
        }

        # Handle non-serializable values
        try:
            log_json = json.dumps(log_entry)
        except (TypeError, ValueError):
            # Convert non-serializable values to strings
            sanitized_context = {}
            for k, v in log_entry["context"].items():
                try:
                    json.dumps({k: v})
                    sanitized_context[k] = v
                except (TypeError, ValueError):
                    sanitized_context[k] = str(v)

            log_entry["context"] = sanitized_context
            log_json = json.dumps(log_entry)

        self.logger.log(level, log_json)

    def debug(self, message: str, **kwargs):
        """Log DEBUG level with context."""
        self.log_structured(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log INFO level with context."""
        self.log_structured(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log WARNING level with context."""
        self.log_structured(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log ERROR level with context."""
        self.log_structured(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log CRITICAL level with context."""
        self.log_structured(logging.CRITICAL, message, **kwargs)

    def update_context(self, **kwargs):
        """
        Update base context with new fields.

        These fields will appear in all subsequent log messages.

        Args:
            **kwargs: Fields to add to base context
        """
        self.context.update(kwargs)

    def clear_context_field(self, field: str):
        """
        Remove a field from base context.

        Args:
            field: Field name to remove
        """
        self.context.pop(field, None)


class MetricsLogger:
    """
    Logger specifically for training metrics and performance data.

    Logs metrics to JSONL file for easy parsing and analysis. Supports
    step-level metrics, epoch summaries, and statistical analysis.

    Example:
        >>> metrics_logger = MetricsLogger("exp0", output_dir)
        >>> metrics_logger.log_step(100, 2, {"loss": 2.5, "lr": 0.001})
    """

    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize metrics logger.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to write metrics.jsonl
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.logger = logging.getLogger(f"model_foundry.metrics")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_step(self, step: int, epoch: int, metrics: Dict[str, Any]):
        """
        Log metrics for a single training step.

        Args:
            step: Global training step
            epoch: Current epoch
            metrics: Dictionary of metric values (loss, lr, grad_norm, etc.)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "metrics": self._sanitize_metrics(metrics)
        }

        # Write to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Also log to main logger
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")

    def log_epoch_summary(self, epoch: int, summary: Dict[str, Any]):
        """
        Log summary statistics for an epoch.

        Args:
            epoch: Epoch number
            summary: Dictionary with aggregate statistics (avg_loss, min_loss, etc.)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "summary": self._sanitize_metrics(summary)
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        self.logger.info(f"Epoch {epoch} summary: {summary}")

    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metrics for JSON serialization (handle NaN, Inf).

        Args:
            metrics: Raw metrics dictionary

        Returns:
            Sanitized metrics dictionary
        """
        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                if v != v:  # NaN check
                    sanitized[k] = None
                elif v == float('inf'):
                    sanitized[k] = "Infinity"
                elif v == float('-inf'):
                    sanitized[k] = "-Infinity"
                else:
                    sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve all metrics from file.

        Returns:
            List of all metric entries
        """
        if not self.metrics_file.exists():
            return []

        history = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                history.append(json.loads(line))

        return history

    def get_metrics_for_steps(self, start: int, end: int) -> List[Dict[str, Any]]:
        """
        Filter metrics by step range.

        Args:
            start: Starting step (inclusive)
            end: Ending step (inclusive)

        Returns:
            List of metric entries within range
        """
        history = self.get_metrics_history()
        return [entry for entry in history
                if "step" in entry and start <= entry["step"] <= end]

    def compute_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Compute statistics over a specific metric.

        Args:
            metric_name: Name of metric to analyze (e.g., "loss")

        Returns:
            Dictionary with mean, min, max, std
        """
        history = self.get_metrics_history()
        values = []

        for entry in history:
            if "metrics" in entry and metric_name in entry["metrics"]:
                value = entry["metrics"][metric_name]
                if value is not None and isinstance(value, (int, float)):
                    values.append(float(value))

        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "std": 0}

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }


class PerformanceLogger:
    """
    Logger for performance profiling and bottleneck analysis.

    Provides timing utilities and memory tracking for identifying
    performance bottlenecks in training.

    Example:
        >>> perf_logger = PerformanceLogger(logger)
        >>> with perf_logger.time_block("forward_pass"):
        ...     outputs = model(**inputs)
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance to use for output
        """
        self.logger = logger
        self.timers: Dict[str, List[float]] = {}

    @contextmanager
    def time_block(self, block_name: str, log_level: int = logging.DEBUG):
        """
        Context manager for timing code blocks.

        Args:
            block_name: Name of the code block being timed
            log_level: Logging level for timing message

        Yields:
            None

        Example:
            >>> with perf_logger.time_block("data_loading"):
            ...     data = load_data()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.logger.log(log_level, f"{block_name} completed in {elapsed:.4f}s")

            # Track for summary statistics
            if block_name not in self.timers:
                self.timers[block_name] = []
            self.timers[block_name].append(elapsed)

    def log_memory_usage(self):
        """Log current memory usage (GPU if available, otherwise CPU)."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.logger.debug(f"GPU memory - Allocated: {allocated:.2f}GB, "
                            f"Reserved: {reserved:.2f}GB")
        else:
            # Could add CPU memory tracking here if needed
            self.logger.debug("Memory tracking - CPU mode (GPU not available)")

    def get_timing_statistics(self, operation: str) -> Dict[str, float]:
        """
        Compute statistics for a timed operation.

        Args:
            operation: Name of the operation

        Returns:
            Dictionary with count, mean, min, max, std
        """
        if operation not in self.timers or not self.timers[operation]:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "std": 0}

        values = self.timers[operation]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }

    def reset_timers(self):
        """Clear all timing data."""
        self.timers.clear()

    def export_timing_report(self, output_file: Path):
        """
        Export timing statistics to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        report = {}
        for operation in self.timers:
            report[operation] = self.get_timing_statistics(operation)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)


class ErrorTracker:
    """
    Track and aggregate errors during training.

    Logs errors to JSONL file with full context and traceback,
    and maintains counts by error type.

    Example:
        >>> tracker = ErrorTracker(logger, experiment_dir)
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     tracker.log_error(e, context={"step": 100})
    """

    def __init__(self, logger: logging.Logger, experiment_dir: Path,
                 max_errors: int = 1000):
        """
        Initialize error tracker.

        Args:
            logger: Logger instance to use
            experiment_dir: Directory to write errors.jsonl
            max_errors: Maximum errors to keep in memory (unlimited in file)
        """
        self.logger = logger
        self.error_log = Path(experiment_dir) / "errors.jsonl"
        self.max_errors = max_errors
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Ensure directory exists
        self.error_log.parent.mkdir(parents=True, exist_ok=True)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log an error with full context and traceback.

        Args:
            error: Exception instance
            context: Additional context (step, epoch, etc.)
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }

        # Write to error log
        with open(self.error_log, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')

        # Log to main logger
        self.logger.error(f"{type(error).__name__}: {error}",
                         exc_info=True, extra=context or {})

        # Track counts
        self.error_counts[type(error).__name__] += 1

    def get_error_summary(self) -> Dict[str, int]:
        """
        Get summary of all errors encountered.

        Returns:
            Dictionary mapping error type to count
        """
        return dict(self.error_counts)

    def reset_counters(self):
        """Reset error counters."""
        self.error_counts.clear()

    def get_recent_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the N most recent errors.

        Args:
            n: Number of recent errors to retrieve

        Returns:
            List of error entries (most recent first)
        """
        if not self.error_log.exists():
            return []

        errors = []
        with open(self.error_log, 'r') as f:
            for line in f:
                errors.append(json.loads(line))

        # Return N most recent
        return errors[-n:][::-1]  # Reverse to get most recent first


class WandBLogger:
    """
    Integration with Weights & Biases experiment tracking.

    Provides seamless logging of metrics, system resources, and artifacts
    to WandB for experiment tracking and visualization.

    Example:
        >>> wandb_logger = WandBLogger(
        ...     project="model-foundry",
        ...     name="exp0_baseline",
        ...     config=config.dict()
        ... )
        >>> wandb_logger.log_metrics(100, {"train/loss": 2.5})
        >>> wandb_logger.finish()
    """

    def __init__(self, project: str, name: str, config: Dict[str, Any],
                 entity: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 enabled: bool = True):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name (experiment name)
            config: Configuration dictionary to log
            entity: WandB entity (username or team name)
            tags: Optional tags for organizing runs
            enabled: Whether WandB is enabled
        """
        self.enabled = enabled
        self.wandb = None

        if enabled:
            try:
                import wandb as wandb_module
                self.wandb = wandb_module

                # Initialize WandB run
                self.wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    config=config,
                    tags=tags or [],
                    resume="allow",  # Auto-resume if run exists
                    settings=wandb_module.Settings(code_dir=".")
                )

                # Store run info
                self.run = self.wandb.run
                self.run_id = self.wandb.run.id if self.wandb.run else None

            except ImportError:
                logging.warning("WandB not installed. Install with: pip install wandb")
                self.enabled = False
            except Exception as e:
                logging.warning(f"Failed to initialize WandB: {e}")
                logging.warning("Continuing without WandB logging.")
                self.enabled = False

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Log metrics to WandB.

        Args:
            step: Global training step
            metrics: Dictionary of metric names and values

        Example:
            >>> wandb_logger.log_metrics(100, {
            ...     "train/loss": 2.5,
            ...     "train/lr": 0.001,
            ...     "train/grad_norm": 1.23
            ... })
        """
        if self.enabled and self.wandb:
            self.wandb.log(metrics, step=step)

    def log_system_metrics(self, step: Optional[int] = None):
        """
        Log system resource usage (GPU memory, etc.).

        Args:
            step: Optional step number
        """
        if not self.enabled or not self.wandb:
            return

        metrics = {}

        # GPU metrics
        if torch.cuda.is_available():
            metrics["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            metrics["system/gpu_memory_cached_gb"] = torch.cuda.memory_reserved() / 1e9

        # Log with or without step
        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)

    def watch_model(self, model, log_freq: int = 100, log_graph: bool = True):
        """
        Watch model parameters and gradients.

        Args:
            model: PyTorch model
            log_freq: How often to log histograms (in steps)
            log_graph: Whether to log model graph
        """
        if self.enabled and self.wandb:
            self.wandb.watch(
                model,
                log="all",  # Log gradients and parameters
                log_freq=log_freq,
                log_graph=log_graph
            )

    def log_artifact(self, artifact_path: str, artifact_type: str, name: str,
                    description: Optional[str] = None):
        """
        Log an artifact (model checkpoint, dataset, etc.).

        Args:
            artifact_path: Path to artifact directory or file
            artifact_type: Type (e.g., "model", "dataset", "config")
            name: Artifact name
            description: Optional description

        Example:
            >>> wandb_logger.log_artifact(
            ...     "output/checkpoint-1000",
            ...     "model",
            ...     "checkpoint-1000"
            ... )
        """
        if not self.enabled or not self.wandb:
            return

        artifact = self.wandb.Artifact(
            name=name,
            type=artifact_type,
            description=description or f"{artifact_type} artifact"
        )

        # Add directory or file
        artifact_path_obj = Path(artifact_path)
        if artifact_path_obj.is_dir():
            artifact.add_dir(str(artifact_path))
        elif artifact_path_obj.is_file():
            artifact.add_file(str(artifact_path))

        self.wandb.log_artifact(artifact)

    def log_table(self, name: str, columns: List[str], data: List[List[Any]]):
        """
        Log a table of data.

        Args:
            name: Table name
            columns: Column names
            data: List of rows

        Example:
            >>> wandb_logger.log_table(
            ...     "predictions",
            ...     ["step", "input", "prediction", "target"],
            ...     [[100, "input text", "predicted", "target"]]
            ... )
        """
        if not self.enabled or not self.wandb:
            return

        table = self.wandb.Table(columns=columns, data=data)
        self.wandb.log({name: table})

    def log_image(self, name: str, image, caption: Optional[str] = None):
        """
        Log an image.

        Args:
            name: Image name/key
            image: Image (PIL, matplotlib figure, numpy array, or path)
            caption: Optional caption
        """
        if not self.enabled or not self.wandb:
            return

        self.wandb.log({name: self.wandb.Image(image, caption=caption)})

    def log_histogram(self, name: str, values):
        """
        Log a histogram of values.

        Args:
            name: Histogram name
            values: Values to histogram (list, numpy array, or tensor)
        """
        if not self.enabled or not self.wandb:
            return

        self.wandb.log({name: self.wandb.Histogram(values)})

    def alert(self, title: str, text: str, level: str = "INFO"):
        """
        Send an alert notification.

        Args:
            title: Alert title
            text: Alert message
            level: Alert level ("INFO", "WARN", "ERROR")

        Example:
            >>> wandb_logger.alert(
            ...     "High Loss",
            ...     "Loss spiked to 10.0 at step 500",
            ...     level="WARN"
            ... )
        """
        if not self.enabled or not self.wandb:
            return

        # Map level string to WandB AlertLevel
        level_map = {
            "INFO": self.wandb.AlertLevel.INFO,
            "WARN": self.wandb.AlertLevel.WARN,
            "ERROR": self.wandb.AlertLevel.ERROR
        }

        alert_level = level_map.get(level.upper(), self.wandb.AlertLevel.INFO)

        self.wandb.alert(
            title=title,
            text=text,
            level=alert_level
        )

    def finish(self):
        """Finish the WandB run and upload any remaining data."""
        if self.enabled and self.wandb:
            self.wandb.finish()

    def get_run_url(self) -> Optional[str]:
        """
        Get the URL of the current WandB run.

        Returns:
            URL string or None if not initialized
        """
        if self.enabled and self.wandb and self.wandb.run:
            return self.wandb.run.get_url()
        return None

    def summary(self) -> Optional[Dict[str, Any]]:
        """
        Get run summary (final metrics).

        Returns:
            Dictionary of summary metrics or None
        """
        if self.enabled and self.wandb and self.wandb.run:
            return dict(self.wandb.run.summary)
        return None
