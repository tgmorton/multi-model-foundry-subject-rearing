# Model Foundry Logging System - Comprehensive Plan

## Executive Summary

This document outlines a complete logging architecture for the model_foundry training framework. The current implementation has basic logging utilities but inconsistent usage (mixing `print()` statements with logging), no structured logging, and limited observability for debugging production issues.

**Goals:**
1. Replace all `print()` statements with proper logging
2. Implement structured logging with consistent message formats
3. Add log levels appropriate to message importance
4. Create dedicated loggers for different subsystems
5. Enable filtering, searching, and analysis of logs
6. Support integration with monitoring tools (WandB, TensorBoard)
7. Ensure thread-safe and multiprocessing-safe logging
8. Provide comprehensive unit tests for all logging functionality

---

## Current State Analysis

### Existing Logging Infrastructure

**File: `logging_utils.py`** (248 lines)
- `setup_logging()` - Basic logger with file + console handlers
- `setup_experiment_logging()` - Experiment-specific logging
- `setup_multi_logging()` - Multiple loggers (main, errors, ablation, progress)
- `get_latest_log()` - Retrieve most recent log file
- `list_experiment_logs()` - List experiment logs
- `cleanup_empty_logs()` - Remove empty log files

**Strengths:**
- Good foundation with file rotation by timestamp
- Experiment-scoped log directories
- Multiple logger support for different streams
- Utility functions for log management

**Weaknesses:**
1. **Inconsistent Usage**: 30+ `print()` statements in `data.py` alone
2. **No Structured Logging**: All messages are plain text, hard to parse
3. **No Context Information**: Missing important metadata (step, epoch, git hash)
4. **Duplicate Handler Issue**: `_LOGGERS_CREATED` set only works within single process
5. **No Log Levels Strategy**: No clear guidelines on when to use DEBUG/INFO/WARNING/ERROR
6. **No Metrics Logging**: Training metrics scattered across console output
7. **No Error Tracking**: Exceptions not consistently logged with context
8. **No Performance Logging**: No timing information for bottleneck analysis

### Current Usage Patterns

**Files with `print()` statements:**
- `data.py` (30+ statements) - Data validation, preprocessing progress
- `training/loop.py` - Training progress (though uses logger elsewhere)
- `training/checkpointing.py` - Checkpoint save/load
- `training/tokenization.py` - Tokenizer loading
- `model.py` - Model creation
- `utils.py` - General utilities
- `cli.py` - Command-line interface

**Files with proper logging:**
- `trainer.py` - Uses `setup_logging()`
- `training/loop.py` - Has logger instance, uses it for some messages
- `training/checkpointing.py` - Has logger setup
- `cli.py` - Mix of logging and print

---

## Proposed Architecture

### 1. Logger Hierarchy

```
model_foundry (root)
├── model_foundry.trainer         # Main orchestration
├── model_foundry.data            # Data processing
│   ├── validation                # Data validation
│   ├── chunking                  # Chunking operations
│   └── loading                   # DataLoader operations
├── model_foundry.model           # Model creation
├── model_foundry.training        # Training subsystem
│   ├── loop                      # Training loop
│   ├── checkpointing             # Checkpoint management
│   └── tokenization              # Tokenizer operations
├── model_foundry.metrics         # Metrics tracking
│   ├── loss                      # Loss values
│   ├── performance               # Speed, throughput
│   └── memory                    # Memory usage
└── model_foundry.system          # System-level events
    ├── errors                    # Error tracking
    └── warnings                  # Warning tracking
```

### 2. Log Levels Strategy

| Level | Usage | Examples |
|-------|-------|----------|
| **DEBUG** | Detailed diagnostic info, variable values, loop iterations | "Processing chunk 45/1000", "Gradient norm: 2.34" |
| **INFO** | General progress, milestones, configuration | "Starting epoch 3/10", "Loaded model with 124M parameters" |
| **WARNING** | Unexpected but recoverable situations | "Using CPU (CUDA unavailable)", "Checkpoint file size unusually large" |
| **ERROR** | Errors that may impact results but don't stop execution | "Failed to save intermediate checkpoint", "NaN detected in gradients" |
| **CRITICAL** | Fatal errors requiring immediate attention | "Out of memory error", "Corrupted checkpoint - cannot resume" |

### 3. Structured Logging Format

**Standard Fields (All Messages):**
```python
{
    "timestamp": "2025-09-30 14:32:15.123",
    "level": "INFO",
    "logger": "model_foundry.training.loop",
    "message": "Completed training step",
    "context": {
        "experiment": "exp0_baseline",
        "git_hash": "e7607e6",
        "device": "cuda:0"
    }
}
```

**Training Step Messages:**
```python
{
    "timestamp": "2025-09-30 14:32:15.123",
    "level": "INFO",
    "logger": "model_foundry.training.loop",
    "message": "Training step completed",
    "context": {
        "experiment": "exp0_baseline",
        "step": 1000,
        "epoch": 2,
        "loss": 2.456,
        "lr": 0.0001,
        "tokens_per_sec": 8500,
        "memory_allocated_gb": 3.2,
        "grad_norm": 1.23
    }
}
```

**Error Messages:**
```python
{
    "timestamp": "2025-09-30 14:32:15.123",
    "level": "ERROR",
    "logger": "model_foundry.training.checkpointing",
    "message": "Failed to save checkpoint",
    "context": {
        "experiment": "exp0_baseline",
        "step": 5000,
        "error_type": "IOError",
        "error_message": "Disk full",
        "traceback": "..."
    }
}
```

### 4. New Logging Components

#### A. `StructuredLogger` Class

Wraps Python's logging.Logger with structured logging capabilities:

```python
class StructuredLogger:
    """Enhanced logger with structured logging support."""

    def __init__(self, name: str, experiment_config: ExperimentConfig):
        self.logger = logging.getLogger(name)
        self.context = self._build_base_context(experiment_config)

    def _build_base_context(self, config):
        """Build context that appears in all log messages."""
        return {
            "experiment": config.experiment_name,
            "git_hash": get_git_commit_hash(),
            "device": str(get_device())
        }

    def log_structured(self, level: int, message: str, **kwargs):
        """Log a message with structured context."""
        log_entry = {
            "message": message,
            "context": {**self.context, **kwargs}
        }
        self.logger.log(level, json.dumps(log_entry))

    def info(self, message: str, **kwargs):
        """Log INFO level with context."""
        self.log_structured(logging.INFO, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log DEBUG level with context."""
        self.log_structured(logging.DEBUG, message, **kwargs)

    # ... warning, error, critical methods
```

#### B. `MetricsLogger` Class

Specialized logger for tracking training metrics:

```python
class MetricsLogger:
    """Logger specifically for training metrics and performance data."""

    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.metrics_file = output_dir / "metrics.jsonl"  # JSON Lines format
        self.logger = logging.getLogger(f"model_foundry.metrics")

    def log_step(self, step: int, epoch: int, metrics: dict):
        """Log metrics for a single training step."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics
        }

        # Write to JSONL file for easy analysis
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Also log to main logger
        self.logger.info(f"Step {step}: " +
                        ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def log_epoch_summary(self, epoch: int, summary: dict):
        """Log summary statistics for an epoch."""
        # Similar to log_step but for epoch-level aggregates
        pass
```

#### C. `PerformanceLogger` Class

Tracks timing and resource usage:

```python
class PerformanceLogger:
    """Logger for performance profiling and bottleneck analysis."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}

    @contextmanager
    def time_block(self, block_name: str, log_level=logging.DEBUG):
        """Context manager for timing code blocks."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.logger.log(log_level,
                           f"{block_name} completed in {elapsed:.4f}s")

            # Track for summary statistics
            if block_name not in self.timers:
                self.timers[block_name] = []
            self.timers[block_name].append(elapsed)

    def log_memory_usage(self):
        """Log current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.logger.debug(f"GPU memory - Allocated: {allocated:.2f}GB, "
                             f"Reserved: {reserved:.2f}GB")
```

#### D. `ErrorTracker` Class

Centralized error tracking and reporting:

```python
class ErrorTracker:
    """Track and aggregate errors during training."""

    def __init__(self, logger: logging.Logger, experiment_dir: Path):
        self.logger = logger
        self.error_log = experiment_dir / "errors.jsonl"
        self.error_counts = defaultdict(int)

    def log_error(self, error: Exception, context: dict = None):
        """Log an error with full context and traceback."""
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
                         exc_info=True, extra=context)

        # Track counts
        self.error_counts[type(error).__name__] += 1

    def get_error_summary(self) -> dict:
        """Get summary of all errors encountered."""
        return dict(self.error_counts)
```

### 5. Integration Points

#### A. Replace `print()` in `data.py`

**Before:**
```python
print(f"  ✓ Training dataset loaded successfully")
print(f"    - Training size: {len(train_dataset):,} examples")
```

**After:**
```python
self.logger.info("Training dataset loaded successfully",
                 dataset_size=len(train_dataset),
                 columns=train_dataset.column_names)
```

#### B. Enhanced Training Loop Logging

**Before:**
```python
self.logger.info("Starting training loop...")
```

**After:**
```python
self.logger.info("Starting training loop",
                 epochs=self.config.training.epochs,
                 total_steps=self.config.training.train_steps,
                 batch_size=self.config.data.batch_size,
                 gradient_accumulation=self.config.training.gradient_accumulation_steps,
                 learning_rate=self.config.training.learning_rate,
                 warmup_steps=self.config.training.warmup_steps)

# During training
with self.perf_logger.time_block("forward_pass"):
    outputs = self.model(**inputs)

self.metrics_logger.log_step(
    step=self.global_step,
    epoch=self.epoch,
    metrics={
        "loss": loss.item(),
        "learning_rate": self.lr_scheduler.get_last_lr()[0],
        "gradient_norm": grad_norm,
        "tokens_per_second": tokens_per_sec
    }
)
```

#### C. Checkpoint Save/Load Logging

**Enhanced checkpointing logs:**
```python
def save_checkpoint(self, ...):
    self.logger.info("Saving checkpoint",
                     step=global_step,
                     epoch=epoch,
                     checkpoint_dir=str(checkpoint_dir))

    with self.perf_logger.time_block("save_model"):
        model.save_pretrained(checkpoint_dir)

    with self.perf_logger.time_block("save_optimizer"):
        torch.save(state, checkpoint_dir / "training_state.pt")

    checkpoint_size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*')) / 1e9

    self.logger.info("Checkpoint saved successfully",
                     step=global_step,
                     checkpoint_size_gb=checkpoint_size,
                     save_time_seconds=total_time)
```

### 6. Configuration

Add logging configuration to `ExperimentConfig`:

```python
@dataclass
class LoggingConfig(BaseModel):
    """Configuration for logging behavior."""

    # Log levels
    console_level: str = Field(default="INFO", description="Console log level")
    file_level: str = Field(default="DEBUG", description="File log level")

    # Output formats
    use_structured_logging: bool = Field(default=True, description="Enable JSON structured logs")
    log_to_wandb: bool = Field(default=True, description="Send logs to WandB")

    # Log rotation
    max_log_files: int = Field(default=10, description="Maximum log files to keep")
    max_log_size_mb: int = Field(default=100, description="Maximum size per log file")

    # Metrics logging
    log_metrics_every_n_steps: int = Field(default=10, description="Steps between metric logs")
    log_detailed_metrics_every_n_steps: int = Field(default=100, description="Steps between detailed metrics")

    # Performance logging
    profile_performance: bool = Field(default=False, description="Enable performance profiling")
    log_memory_every_n_steps: int = Field(default=100, description="Steps between memory logs")

    # Error tracking
    max_errors_to_track: int = Field(default=1000, description="Maximum errors to track in memory")
```

### 7. Log File Organization

**Directory Structure:**
```
logs/
├── exp0_baseline/
│   ├── main_2025-09-30_14-30-00.log          # General logs
│   ├── metrics_2025-09-30_14-30-00.jsonl     # Training metrics (JSON Lines)
│   ├── errors_2025-09-30_14-30-00.jsonl      # Error tracking
│   ├── performance_2025-09-30_14-30-00.jsonl # Performance profiling
│   └── debug_2025-09-30_14-30-00.log         # Verbose debug logs
├── exp1_remove_expletives/
│   └── ...
```

**File Formats:**

- `.log` files: Human-readable text logs
- `.jsonl` files: JSON Lines format for programmatic analysis

### 8. Monitoring Integration

#### WandB Integration

```python
class WandBLogger:
    """Integration with Weights & Biases."""

    def __init__(self, config: ExperimentConfig, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            wandb.init(
                project="model_foundry",
                name=config.experiment_name,
                config=config.dict()
            )

    def log_metrics(self, step: int, metrics: dict):
        """Log metrics to WandB."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_system_metrics(self):
        """Log system resource usage."""
        if self.enabled and torch.cuda.is_available():
            wandb.log({
                "system/gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                "system/gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
            })
```

### 9. Testing Strategy

See **Unit Tests Section** below for comprehensive test plan.

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create `StructuredLogger` class
2. Create `MetricsLogger` class
3. Create `PerformanceLogger` class
4. Create `ErrorTracker` class
5. Add `LoggingConfig` to config.py
6. Write unit tests for all new classes

### Phase 2: Integration (Week 2)
1. Replace all `print()` statements in `data.py`
2. Replace all `print()` statements in `model.py`
3. Enhance logging in `training/loop.py`
4. Enhance logging in `training/checkpointing.py`
5. Enhance logging in `training/tokenization.py`
6. Write integration tests

### Phase 3: Advanced Features (Week 3)
1. Implement log rotation
2. Add WandB integration
3. Create log analysis utilities
4. Add performance profiling
5. Write end-to-end tests

### Phase 4: Documentation & Polish (Week 4)
1. Update all documentation
2. Create logging best practices guide
3. Add logging examples to README
4. Conduct code review
5. Performance testing

---

## Migration Guide

### For Developers

**Old Pattern:**
```python
print(f"  ✓ Loaded {len(dataset)} examples")
```

**New Pattern:**
```python
self.logger.info("Dataset loaded",
                 num_examples=len(dataset),
                 columns=dataset.column_names)
```

**Error Handling - Old:**
```python
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")
```

**Error Handling - New:**
```python
try:
    result = risky_operation()
except Exception as e:
    self.error_tracker.log_error(e, context={
        "operation": "risky_operation",
        "step": self.global_step
    })
    raise  # Re-raise if fatal, or handle gracefully
```

---

## Unit Tests - Comprehensive Test Plan

### Test File: `tests/unit/test_logging.py`

**Overview:** 50+ tests covering all logging functionality

#### 1. StructuredLogger Tests (15 tests)

```python
class TestStructuredLogger:
    """Test the StructuredLogger class."""

    def test_creates_logger_with_base_context(self, tiny_config):
        """Should create logger with experiment context."""
        logger = StructuredLogger("test", tiny_config)
        assert logger.context["experiment"] == tiny_config.experiment_name
        assert "git_hash" in logger.context
        assert "device" in logger.context

    def test_log_structured_creates_json_output(self, tiny_config, tmp_path):
        """Should output structured JSON logs."""
        # Setup logger with file handler
        logger = StructuredLogger("test", tiny_config)
        log_file = tmp_path / "test.log"

        handler = logging.FileHandler(log_file)
        logger.logger.addHandler(handler)

        # Log a message
        logger.info("Test message", custom_field="value")
        handler.flush()

        # Verify JSON structure
        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["message"] == "Test message"
        assert log_entry["context"]["experiment"] == tiny_config.experiment_name
        assert log_entry["context"]["custom_field"] == "value"

    def test_info_level_logs_at_info(self, tiny_config):
        """info() should log at INFO level."""
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test message")
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.INFO

    def test_debug_level_logs_at_debug(self, tiny_config):
        """debug() should log at DEBUG level."""
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.debug("Test message")
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.DEBUG

    def test_warning_level_logs_at_warning(self, tiny_config):
        """warning() should log at WARNING level."""
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.warning("Test message")
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.WARNING

    def test_error_level_logs_at_error(self, tiny_config):
        """error() should log at ERROR level."""
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.error("Test message")
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.ERROR

    def test_critical_level_logs_at_critical(self, tiny_config):
        """critical() should log at CRITICAL level."""
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.critical("Test message")
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.CRITICAL

    def test_context_merges_with_base_context(self, tiny_config):
        """Custom context should merge with base context."""
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test", step=100, loss=2.5)

            logged_message = mock_log.call_args[0][1]
            log_entry = json.loads(logged_message)

            # Should have both base and custom context
            assert "experiment" in log_entry["context"]
            assert log_entry["context"]["step"] == 100
            assert log_entry["context"]["loss"] == 2.5

    def test_custom_context_overrides_base_context(self, tiny_config):
        """Custom context values should override base context."""
        logger = StructuredLogger("test", tiny_config)
        original_experiment = logger.context["experiment"]

        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test", experiment="override")

            logged_message = mock_log.call_args[0][1]
            log_entry = json.loads(logged_message)

            assert log_entry["context"]["experiment"] == "override"
            # But base context should remain unchanged
            assert logger.context["experiment"] == original_experiment

    def test_handles_non_serializable_context(self, tiny_config):
        """Should handle context values that aren't JSON serializable."""
        logger = StructuredLogger("test", tiny_config)

        # Should not raise exception
        class NonSerializable:
            pass

        # This should convert to string representation
        logger.info("Test", obj=NonSerializable())
        # Test passes if no exception raised

    def test_multiple_loggers_independent_contexts(self, tiny_config):
        """Multiple StructuredLogger instances should have independent contexts."""
        logger1 = StructuredLogger("test1", tiny_config)
        logger2 = StructuredLogger("test2", tiny_config)

        logger1.context["custom"] = "value1"
        logger2.context["custom"] = "value2"

        assert logger1.context["custom"] == "value1"
        assert logger2.context["custom"] == "value2"

    def test_update_base_context(self, tiny_config):
        """Should allow updating base context."""
        logger = StructuredLogger("test", tiny_config)
        logger.update_context(step=100, epoch=5)

        assert logger.context["step"] == 100
        assert logger.context["epoch"] == 5

        # Should appear in all subsequent logs
        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test")

            logged_message = mock_log.call_args[0][1]
            log_entry = json.loads(logged_message)

            assert log_entry["context"]["step"] == 100
            assert log_entry["context"]["epoch"] == 5

    def test_clear_context_field(self, tiny_config):
        """Should allow removing fields from base context."""
        logger = StructuredLogger("test", tiny_config)
        logger.update_context(step=100)
        assert "step" in logger.context

        logger.clear_context_field("step")
        assert "step" not in logger.context

    def test_log_exception_with_traceback(self, tiny_config, tmp_path):
        """Should log exceptions with full traceback."""
        logger = StructuredLogger("test", tiny_config)
        log_file = tmp_path / "test.log"

        handler = logging.FileHandler(log_file)
        logger.logger.addHandler(handler)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("Exception occurred", exception=str(e), exc_info=True)

        handler.flush()
        log_content = log_file.read_text()

        assert "ValueError" in log_content
        assert "Test error" in log_content
```

#### 2. MetricsLogger Tests (12 tests)

```python
class TestMetricsLogger:
    """Test the MetricsLogger class."""

    def test_creates_metrics_file(self, tmp_path):
        """Should create metrics.jsonl file."""
        logger = MetricsLogger("test_exp", tmp_path)
        assert logger.metrics_file == tmp_path / "metrics.jsonl"

    def test_log_step_writes_jsonl(self, tmp_path):
        """Should write metrics to JSONL file."""
        logger = MetricsLogger("test_exp", tmp_path)

        metrics = {"loss": 2.5, "lr": 0.001}
        logger.log_step(step=100, epoch=2, metrics=metrics)

        # Read JSONL
        with open(logger.metrics_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["step"] == 100
        assert entry["epoch"] == 2
        assert entry["metrics"]["loss"] == 2.5
        assert entry["metrics"]["lr"] == 0.001
        assert "timestamp" in entry

    def test_log_step_appends_to_file(self, tmp_path):
        """Should append to metrics file, not overwrite."""
        logger = MetricsLogger("test_exp", tmp_path)

        logger.log_step(100, 1, {"loss": 2.5})
        logger.log_step(200, 2, {"loss": 2.3})

        with open(logger.metrics_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 100
        assert json.loads(lines[1])["step"] == 200

    def test_log_epoch_summary(self, tmp_path):
        """Should log epoch-level summary statistics."""
        logger = MetricsLogger("test_exp", tmp_path)

        summary = {
            "avg_loss": 2.4,
            "min_loss": 2.1,
            "max_loss": 2.8,
            "total_tokens": 1000000
        }
        logger.log_epoch_summary(epoch=5, summary=summary)

        # Verify written correctly
        with open(logger.metrics_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["epoch"] == 5
        assert "summary" in entry
        assert entry["summary"]["avg_loss"] == 2.4

    def test_get_metrics_history(self, tmp_path):
        """Should retrieve metrics history from file."""
        logger = MetricsLogger("test_exp", tmp_path)

        # Log several steps
        for step in range(0, 500, 100):
            logger.log_step(step, 0, {"loss": 3.0 - step/1000})

        # Retrieve history
        history = logger.get_metrics_history()

        assert len(history) == 5
        assert history[0]["step"] == 0
        assert history[-1]["step"] == 400

    def test_get_metrics_for_steps(self, tmp_path):
        """Should filter metrics by step range."""
        logger = MetricsLogger("test_exp", tmp_path)

        for step in range(0, 1000, 100):
            logger.log_step(step, step // 100, {"loss": 3.0 - step/1000})

        # Get metrics for steps 200-500
        filtered = logger.get_metrics_for_steps(start=200, end=500)

        assert len(filtered) == 4  # 200, 300, 400, 500
        assert filtered[0]["step"] == 200
        assert filtered[-1]["step"] == 500

    def test_compute_statistics(self, tmp_path):
        """Should compute statistics over metrics."""
        logger = MetricsLogger("test_exp", tmp_path)

        losses = [3.0, 2.8, 2.6, 2.4, 2.2]
        for i, loss in enumerate(losses):
            logger.log_step(i * 100, 0, {"loss": loss})

        stats = logger.compute_statistics("loss")

        assert stats["mean"] == pytest.approx(2.6)
        assert stats["min"] == 2.2
        assert stats["max"] == 3.0
        assert stats["std"] > 0

    def test_log_gradient_norm(self, tmp_path):
        """Should log gradient norm with metrics."""
        logger = MetricsLogger("test_exp", tmp_path)

        logger.log_step(100, 1, {
            "loss": 2.5,
            "grad_norm": 1.23
        })

        with open(logger.metrics_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry["metrics"]["grad_norm"] == 1.23

    def test_log_learning_rate_schedule(self, tmp_path):
        """Should track learning rate changes."""
        logger = MetricsLogger("test_exp", tmp_path)

        # Simulate warmup + decay
        lrs = [0.0001, 0.0005, 0.001, 0.0009, 0.0008]
        for i, lr in enumerate(lrs):
            logger.log_step(i * 100, 0, {"lr": lr})

        history = logger.get_metrics_history()
        logged_lrs = [h["metrics"]["lr"] for h in history]

        assert logged_lrs == lrs

    def test_log_throughput_metrics(self, tmp_path):
        """Should log tokens/sec and other throughput metrics."""
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

    def test_handles_nan_inf_values(self, tmp_path):
        """Should handle NaN and Inf metric values gracefully."""
        logger = MetricsLogger("test_exp", tmp_path)

        logger.log_step(100, 1, {
            "loss": float('nan'),
            "grad_norm": float('inf')
        })

        # Should write successfully (JSON supports these)
        with open(logger.metrics_file, 'r') as f:
            entry = json.loads(f.readline())

        assert entry["metrics"]["loss"] is None or entry["metrics"]["loss"] != entry["metrics"]["loss"]  # NaN check

    def test_concurrent_writes_safe(self, tmp_path):
        """Should handle concurrent writes without corruption."""
        logger = MetricsLogger("test_exp", tmp_path)

        # Simulate multiple threads/processes writing
        import threading

        def write_metrics(start_step):
            for i in range(10):
                logger.log_step(start_step + i, 0, {"loss": 2.5})

        threads = [
            threading.Thread(target=write_metrics, args=(i * 100,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 50 entries
        with open(logger.metrics_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 50
        # All should be valid JSON
        for line in lines:
            json.loads(line)  # Should not raise
```

#### 3. PerformanceLogger Tests (10 tests)

```python
class TestPerformanceLogger:
    """Test the PerformanceLogger class."""

    def test_time_block_measures_duration(self):
        """Should measure execution time of code block."""
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with perf_logger.time_block("test_operation"):
            time.sleep(0.1)

        assert "test_operation" in perf_logger.timers
        assert len(perf_logger.timers["test_operation"]) == 1
        assert perf_logger.timers["test_operation"][0] >= 0.1

    def test_time_block_logs_duration(self, caplog):
        """Should log duration to logger."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        perf_logger = PerformanceLogger(logger)

        with caplog.at_level(logging.DEBUG):
            with perf_logger.time_block("test_operation"):
                time.sleep(0.05)

        assert "test_operation completed in" in caplog.text

    def test_time_block_tracks_multiple_calls(self):
        """Should track multiple invocations of same block."""
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        for _ in range(5):
            with perf_logger.time_block("repeated_op"):
                time.sleep(0.01)

        assert len(perf_logger.timers["repeated_op"]) == 5

    def test_time_block_handles_exceptions(self):
        """Should still log timing even if block raises exception."""
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with pytest.raises(ValueError):
            with perf_logger.time_block("failing_op"):
                raise ValueError("Test error")

        # Should still have timing recorded
        assert "failing_op" in perf_logger.timers
        assert len(perf_logger.timers["failing_op"]) == 1

    def test_get_timing_statistics(self):
        """Should compute statistics over multiple timings."""
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        durations = [0.1, 0.2, 0.15, 0.18, 0.12]
        for d in durations:
            with perf_logger.time_block("test_op"):
                time.sleep(d)

        stats = perf_logger.get_timing_statistics("test_op")

        assert stats["count"] == 5
        assert stats["mean"] > 0.1
        assert stats["min"] >= 0.1
        assert stats["max"] >= 0.2

    def test_log_memory_usage_cpu(self, caplog):
        """Should log CPU memory usage."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        perf_logger = PerformanceLogger(logger)

        with caplog.at_level(logging.DEBUG):
            perf_logger.log_memory_usage()

        # Should log something (format depends on CUDA availability)
        assert len(caplog.records) > 0

    @pytest.mark.gpu
    def test_log_memory_usage_gpu(self, caplog):
        """Should log GPU memory usage when CUDA available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        perf_logger = PerformanceLogger(logger)

        with caplog.at_level(logging.DEBUG):
            perf_logger.log_memory_usage()

        assert "GPU memory" in caplog.text
        assert "Allocated" in caplog.text

    def test_reset_timers(self):
        """Should clear all timing data."""
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with perf_logger.time_block("test_op"):
            time.sleep(0.01)

        assert len(perf_logger.timers["test_op"]) == 1

        perf_logger.reset_timers()
        assert len(perf_logger.timers) == 0

    def test_export_timing_report(self, tmp_path):
        """Should export timing data to JSON file."""
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        for _ in range(3):
            with perf_logger.time_block("op1"):
                time.sleep(0.01)
            with perf_logger.time_block("op2"):
                time.sleep(0.02)

        report_file = tmp_path / "timing_report.json"
        perf_logger.export_timing_report(report_file)

        with open(report_file, 'r') as f:
            report = json.load(f)

        assert "op1" in report
        assert "op2" in report
        assert report["op1"]["count"] == 3
        assert report["op2"]["count"] == 3

    def test_nested_time_blocks(self):
        """Should handle nested timing blocks."""
        logger = logging.getLogger("test")
        perf_logger = PerformanceLogger(logger)

        with perf_logger.time_block("outer"):
            time.sleep(0.05)
            with perf_logger.time_block("inner"):
                time.sleep(0.02)

        assert "outer" in perf_logger.timers
        assert "inner" in perf_logger.timers
        # Outer should be longer than inner
        assert perf_logger.timers["outer"][0] > perf_logger.timers["inner"][0]
```

#### 4. ErrorTracker Tests (8 tests)

```python
class TestErrorTracker:
    """Test the ErrorTracker class."""

    def test_log_error_writes_to_file(self, tmp_path):
        """Should write error to JSONL file."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            tracker.log_error(e, context={"step": 100})

        error_log = tmp_path / "errors.jsonl"
        assert error_log.exists()

        with open(error_log, 'r') as f:
            entry = json.loads(f.readline())

        assert entry["error_type"] == "ValueError"
        assert entry["error_message"] == "Test error"
        assert "traceback" in entry
        assert entry["context"]["step"] == 100

    def test_log_error_increments_counter(self, tmp_path):
        """Should track error counts by type."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        for _ in range(3):
            try:
                raise ValueError("Test")
            except ValueError as e:
                tracker.log_error(e)

        for _ in range(2):
            try:
                raise TypeError("Test")
            except TypeError as e:
                tracker.log_error(e)

        summary = tracker.get_error_summary()
        assert summary["ValueError"] == 3
        assert summary["TypeError"] == 2

    def test_log_error_includes_traceback(self, tmp_path):
        """Should include full traceback in error log."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        def nested_function():
            raise RuntimeError("Nested error")

        try:
            nested_function()
        except RuntimeError as e:
            tracker.log_error(e)

        with open(tmp_path / "errors.jsonl", 'r') as f:
            entry = json.loads(f.readline())

        assert "nested_function" in entry["traceback"]
        assert "RuntimeError" in entry["traceback"]

    def test_get_error_summary(self, tmp_path):
        """Should return dictionary of error counts."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        errors = [ValueError, TypeError, ValueError, RuntimeError, ValueError]
        for error_cls in errors:
            try:
                raise error_cls("Test")
            except error_cls as e:
                tracker.log_error(e)

        summary = tracker.get_error_summary()
        assert summary == {
            "ValueError": 3,
            "TypeError": 1,
            "RuntimeError": 1
        }

    def test_log_error_with_no_context(self, tmp_path):
        """Should handle logging error without context."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        try:
            raise ValueError("Test")
        except ValueError as e:
            tracker.log_error(e)  # No context

        with open(tmp_path / "errors.jsonl", 'r') as f:
            entry = json.loads(f.readline())

        assert entry["context"] == {}

    def test_reset_error_counts(self, tmp_path):
        """Should reset error counters."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        try:
            raise ValueError("Test")
        except ValueError as e:
            tracker.log_error(e)

        assert tracker.get_error_summary()["ValueError"] == 1

        tracker.reset_counters()
        assert len(tracker.get_error_summary()) == 0

    def test_max_errors_limit(self, tmp_path):
        """Should limit number of errors tracked in memory."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path, max_errors=10)

        # Log 15 errors
        for i in range(15):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                tracker.log_error(e)

        # Counter should still be accurate
        assert tracker.get_error_summary()["ValueError"] == 15

        # But in-memory storage should be limited
        # (Implementation detail - depends on whether we keep errors in memory)

    def test_get_recent_errors(self, tmp_path):
        """Should retrieve most recent errors."""
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        for i in range(10):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                tracker.log_error(e, context={"index": i})

        recent = tracker.get_recent_errors(n=3)

        assert len(recent) == 3
        # Should be most recent (9, 8, 7)
        assert recent[0]["context"]["index"] == 9
        assert recent[1]["context"]["index"] == 8
        assert recent[2]["context"]["index"] == 7
```

#### 5. LoggingConfig Tests (5 tests)

```python
class TestLoggingConfig:
    """Test the LoggingConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = LoggingConfig()
        assert config.console_level == "INFO"
        assert config.file_level == "DEBUG"
        assert config.use_structured_logging is True
        assert config.log_to_wandb is True
        assert config.max_log_files == 10

    def test_custom_values(self):
        """Should accept custom configuration."""
        config = LoggingConfig(
            console_level="WARNING",
            file_level="INFO",
            use_structured_logging=False,
            max_log_files=5
        )
        assert config.console_level == "WARNING"
        assert config.file_level == "INFO"
        assert config.use_structured_logging is False
        assert config.max_log_files == 5

    def test_validates_log_levels(self):
        """Should validate log level values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            config = LoggingConfig(console_level=level)
            assert config.console_level == level

        # Invalid level should raise
        with pytest.raises(ValidationError):
            LoggingConfig(console_level="INVALID")

    def test_validates_positive_integers(self):
        """Should validate positive integer fields."""
        with pytest.raises(ValidationError):
            LoggingConfig(max_log_files=-1)

        with pytest.raises(ValidationError):
            LoggingConfig(max_log_size_mb=0)

    def test_integrates_with_experiment_config(self, tiny_config):
        """Should integrate with ExperimentConfig."""
        # Add logging config to experiment config
        config_dict = tiny_config.dict()
        config_dict['logging'] = LoggingConfig(console_level="DEBUG").dict()

        # Should validate successfully
        full_config = ExperimentConfig(**config_dict)
        assert full_config.logging.console_level == "DEBUG"
```

### Test File: `tests/integration/test_logging_integration.py`

**Overview:** 15+ integration tests

```python
class TestLoggingIntegration:
    """Integration tests for logging across the training pipeline."""

    @pytest.mark.integration
    def test_trainer_uses_structured_logging(self, tiny_config, temp_workspace):
        """Trainer should use structured logging throughout."""
        # Create trainer with structured logging enabled
        trainer = Trainer(tiny_config, str(temp_workspace))

        # Verify logger is StructuredLogger instance
        assert isinstance(trainer.logger, StructuredLogger)

    @pytest.mark.integration
    def test_training_loop_logs_metrics(self, tiny_config, temp_workspace, mock_tokenizer):
        """Training loop should log metrics at regular intervals."""
        # Run short training
        trainer = Trainer(tiny_config, str(temp_workspace))
        # ... setup and run training ...

        # Check metrics file exists
        metrics_file = temp_workspace / "test" / "output" / "metrics.jsonl"
        assert metrics_file.exists()

        # Verify metrics logged
        with open(metrics_file, 'r') as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) > 0
        assert all("step" in e for e in entries)
        assert all("metrics" in e for e in entries)

    @pytest.mark.integration
    def test_errors_logged_to_error_file(self, tiny_config, temp_workspace):
        """Errors during training should be logged to error file."""
        # Force an error during training
        # ... code to trigger error ...

        error_file = temp_workspace / "test" / "output" / "errors.jsonl"
        assert error_file.exists()

        with open(error_file, 'r') as f:
            entry = json.loads(f.readline())

        assert "error_type" in entry
        assert "traceback" in entry

    # ... more integration tests
```

---

## Success Criteria

### Functional Requirements
- [ ] All `print()` statements replaced with appropriate logging
- [ ] Structured logging implemented and tested
- [ ] Metrics logging captures all training metrics
- [ ] Error tracking captures and aggregates all errors
- [ ] Performance logging tracks bottlenecks
- [ ] WandB integration working
- [ ] Log rotation implemented
- [ ] 100% unit test coverage on logging components

### Performance Requirements
- [ ] Logging overhead < 1% of training time
- [ ] Log file I/O does not block training
- [ ] Memory usage for logging < 100MB

### Quality Requirements
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Code review approved
- [ ] Integration tested with real training runs

---

## Appendix: Log Message Examples

### Startup Logs
```
2025-09-30 14:30:00 [INFO] model_foundry.trainer: Experiment initialized
  experiment: exp0_baseline
  git_hash: e7607e6
  device: cuda:0
  model_params: 124823296

2025-09-30 14:30:02 [INFO] model_foundry.data: Dataset loaded
  train_size: 1000000
  test_size: 10000
  columns: ['input_ids', 'attention_mask']
```

### Training Logs
```
2025-09-30 14:30:15 [INFO] model_foundry.training.loop: Training step completed
  step: 100
  epoch: 0
  loss: 3.245
  lr: 0.0001
  tokens_per_sec: 8500
  grad_norm: 1.23

2025-09-30 14:30:16 [DEBUG] model_foundry.training.loop: forward_pass completed in 0.0234s
2025-09-30 14:30:16 [DEBUG] model_foundry.training.loop: backward_pass completed in 0.0189s
```

### Error Logs
```
2025-09-30 14:35:22 [ERROR] model_foundry.training.checkpointing: Failed to save checkpoint
  step: 5000
  error_type: IOError
  error_message: No space left on device
  checkpoint_dir: /output/checkpoint-5000

2025-09-30 14:36:10 [WARNING] model_foundry.training.loop: Gradient overflow detected
  step: 5050
  scaler_scale: 65536.0
  action: scaled_down
```

---

## Conclusion

This comprehensive logging plan provides:

1. **Structured, parseable logs** for automated analysis
2. **Clear logging hierarchy** organized by subsystem
3. **Comprehensive test coverage** (50+ unit tests, 15+ integration tests)
4. **Performance tracking** to identify bottlenecks
5. **Error aggregation** for debugging production issues
6. **Monitoring integration** with WandB and other tools
7. **Migration path** from current print-based logging

The implementation will significantly improve observability, debuggability, and maintainability of the model_foundry training framework.


---

## Weights & Biases (WandB) Integration

### Quick Setup Guide

**See [WANDB_INTEGRATION_GUIDE.md](WANDB_INTEGRATION_GUIDE.md) for complete instructions.**

### 1. Create WandB Account

1. Visit [https://wandb.ai/signup](https://wandb.ai/signup)
2. Sign up (free tier available)
3. Get API key from [wandb.ai/authorize](https://wandb.ai/authorize)

### 2. Configure API Key

```bash
# Interactive login (recommended)
wandb login

# Or set environment variable
export WANDB_API_KEY="your-40-character-api-key"
```

### 3. Enable in Configuration

```yaml
# config/experiment.yaml
logging:
  use_wandb: true
  wandb_project: "model-foundry-experiments"
  log_metrics_every_n_steps: 10
```

### 4. Run Training

```bash
python -m model_foundry.cli train configs/experiment.yaml
```

### What Gets Logged to WandB

**Metrics (every N steps):**
- Training loss
- Learning rate
- Gradient norm
- Tokens per second
- GPU memory usage

**System Info:**
- Git commit hash
- Configuration (all hyperparameters)
- System metrics (GPU, CPU)

**Artifacts (optional):**
- Model checkpoints
- Training curves
- Evaluation results

### WandBLogger Usage

```python
from model_foundry.logging_components import WandBLogger

# Initialize
wandb_logger = WandBLogger(
    project="model-foundry",
    name="exp0_baseline",
    config=config.dict(),
    tags=["baseline", "gpt2"]
)

# Log metrics
wandb_logger.log_metrics(
    step=100,
    metrics={
        "train/loss": 2.5,
        "train/lr": 0.001,
        "train/grad_norm": 1.23
    }
)

# Log system metrics
wandb_logger.log_system_metrics(step=100)

# Watch model (log gradients/parameters)
wandb_logger.watch_model(model, log_freq=100)

# Log artifacts
wandb_logger.log_artifact(
    "output/checkpoint-1000",
    artifact_type="model",
    name="checkpoint-1000"
)

# Finish run
wandb_logger.finish()
```

### Environment Variables

```bash
# Disable WandB (override config)
export WANDB_MODE=disabled

# Offline mode (sync later)
export WANDB_MODE=offline

# Silent mode (no console output)
export WANDB_SILENT=true

# Custom project
export WANDB_PROJECT=my-experiment
```

### Viewing Results

1. Go to [wandb.ai/home](https://wandb.ai/home)
2. Click on your project
3. View runs with:
   - Real-time metric graphs
   - Configuration comparison
   - System resource monitoring
   - Artifact downloads

### Troubleshooting

**Not logged in:**
```bash
wandb login --relogin
```

**Disable temporarily:**
```bash
export WANDB_MODE=disabled
```

**Offline sync:**
```bash
wandb sync wandb/offline-run-*
```

### Resources

- **Full Guide:** [WANDB_INTEGRATION_GUIDE.md](WANDB_INTEGRATION_GUIDE.md)
- **WandB Docs:** [docs.wandb.ai](https://docs.wandb.ai)
- **Quickstart:** [docs.wandb.ai/quickstart](https://docs.wandb.ai/quickstart)
