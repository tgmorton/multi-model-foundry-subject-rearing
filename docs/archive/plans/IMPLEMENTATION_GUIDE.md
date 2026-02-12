# Streaming Pipeline Implementation Guide

**Companion Document to:** STREAMING_PIPELINE_SPEC.md
**Version:** 1.0
**Date:** 2025-09-30

## Purpose

This document maps the streaming pipeline architecture specified in STREAMING_PIPELINE_SPEC.md to the existing codebase, detailing exactly what changes need to be made to implement the system.

---

## Table of Contents

1. [Current Codebase Structure](#current-codebase-structure)
2. [New Directory Structure Required](#new-directory-structure-required)
3. [Changes to Training Pipeline (wild_west)](#changes-to-training-pipeline-wild_west)
4. [New Components for SSRDE](#new-components-for-ssrde)
5. [Changes to Evaluation Pipeline (SSRDE)](#changes-to-evaluation-pipeline-ssrde)
6. [Changes to Analysis Pipeline (SSRDE)](#changes-to-analysis-pipeline-ssrde)
7. [New Components for Monitoring Server](#new-components-for-monitoring-server)
8. [Configuration Changes](#configuration-changes)
9. [Database Setup](#database-setup)
10. [Deployment Checklist](#deployment-checklist)

---

## Current Codebase Structure

### Existing Training Pipeline

**Entry Point:** `model_foundry/trainer.py`
- Main class: `Trainer`
- Entry point: `main()` function
- Called via: `python -m model_foundry.cli run <config>`

**Training Flow:**
1. Load config from YAML
2. Initialize model, optimizer, data
3. Run training loop via `TrainingLoop.run()`
4. Save checkpoints via `CheckpointManager`
5. Log to W&B if enabled

**Current Checkpoint Saving:**
- Location: `model_foundry/training/checkpointing.py`
- Class: `CheckpointManager`
- Method: `save_checkpoint()` - saves to `training.output_dir`

**Current Logging:**
- W&B integration exists in `trainer.py` (lines 329-336)
- Logging utilities in `model_foundry/logging_utils.py`

### Existing Evaluation Pipeline

**Entry Point:** `evaluation/runners/evaluation_runner.py`
- Main class: `EvaluationRunner`
- Config: `EvaluationConfig` (Pydantic model)
- Called via: `python evaluation/runners/evaluation_runner.py --config <config>`

**Evaluation Flow:**
1. Find checkpoints in `model_checkpoint_dir`
2. For each checkpoint:
   - Load model and tokenizer
   - Run perplexity evaluation (if enabled)
   - Run BLIMP evaluation (if enabled)
   - Run null-subject evaluation (if enabled)
   - Save results to `output_dir`
3. Generate summaries

**Current Output Format:**
- JSON/JSONL files in `evaluation/results/`
- Individual checkpoint results
- Task-specific summaries

### Existing Analysis Pipeline

**Entry Point:** `analysis/scripts/run_complete_analysis.R`
- Runs 6 analysis scripts sequentially
- Reads data from CSV files
- Outputs to `analysis/tables/` and `analysis/figures/`

**Current Workflow:**
- Manual execution: `Rscript analysis/scripts/run_complete_analysis.R`
- No automation or event triggers
- Expects pre-existing evaluation results

---

## New Directory Structure Required

### On Shared Network Drive

Create this structure on the shared network drive (adjust path as needed):

```bash
# Example path: /mnt/shared/subject-drop/
mkdir -p /mnt/shared/subject-drop/{config,models,evaluation,analysis,state,logs,mlflow-artifacts}
mkdir -p /mnt/shared/subject-drop/config/{experiments,evaluation}
mkdir -p /mnt/shared/subject-drop/logs/{training,evaluation,analysis}
mkdir -p /mnt/shared/subject-drop/state/locks
```

### In Existing Codebase

Create new directories for streaming pipeline components:

```bash
# From project root
mkdir -p pipeline/watcher
mkdir -p pipeline/slurm_templates
mkdir -p pipeline/monitoring
mkdir -p pipeline/utils
```

---

## Changes to Training Pipeline (wild_west)

### 1. Modify `model_foundry/trainer.py`

**Location:** `model_foundry/trainer.py`

**Changes Required:**

#### Add Training Completion Hook

**Where:** After line 360 in `_train_loop()` method, after training completes

**What to Add:**

Add a new method to the `Trainer` class:

```python
def _signal_training_complete(self, final_checkpoint_path: str):
    """
    Write signal file to trigger downstream evaluation pipeline.

    This runs on wild_west and signals to SSRDE that training is complete.
    """
    pass  # Implementation details in spec
```

**Integration Point:**

In `_train_loop()` method, after line 360 (after `final_step = self.training_loop.run(...)`), add:

```python
# Signal training completion for streaming pipeline
if hasattr(self.config, 'streaming_pipeline') and self.config.streaming_pipeline.get('enabled', False):
    checkpoint_dir = self.checkpoint_manager.get_final_checkpoint_dir()
    self._signal_training_complete(checkpoint_dir)
```

#### Add MLflow Logging Enhancement

**Where:** Lines 329-336 (W&B initialization)

**What to Add:**

After W&B init, add MLflow init:

```python
# Initialize MLflow logging (in addition to W&B)
if self.config.logging.get('use_mlflow', False):
    import mlflow
    mlflow.set_tracking_uri(self.config.logging.mlflow_tracking_uri)
    mlflow.start_run(run_name=self.config.experiment_name)
    mlflow.log_params(self.config.model_dump())
```

**Integration with Training Loop:**

In `model_foundry/training/loop.py`, add MLflow metric logging alongside existing logging.

---

### 2. Modify `model_foundry/training/checkpointing.py`

**Location:** `model_foundry/training/checkpointing.py`

**Changes Required:**

#### Update Checkpoint Save Location

**Current Behavior:** Saves to local `output_dir`

**New Behavior:** Save to shared network drive location

**Method to Modify:** `CheckpointManager.save_checkpoint()`

**What to Add:**

Add configuration option to use shared drive path:

```python
def _get_checkpoint_path(self, step: int) -> Path:
    """Get checkpoint path, using shared drive if configured."""
    if hasattr(self.config, 'streaming_pipeline') and self.config.streaming_pipeline.get('enabled', False):
        # Use shared drive path
        base_dir = Path(self.config.streaming_pipeline.shared_drive_path)
        checkpoint_dir = base_dir / "models" / self.config.experiment_name / f"checkpoint-{step}"
    else:
        # Use local path (existing behavior)
        checkpoint_dir = Path(self.base_dir) / self.config.training.output_dir / f"checkpoint-{step}"

    return checkpoint_dir
```

---

### 3. Add New Module: `pipeline/utils/signal_writer.py`

**Purpose:** Utility for writing signal files

**Location:** Create new file `pipeline/utils/signal_writer.py`

**Content:** Implementation of signal file writing logic

**Key Functions:**
- `write_training_complete_signal(experiment_name, checkpoint_path, metadata)`
- `update_training_database(experiment_name, status, checkpoint_path)`

**Dependencies:**
- `json` for signal file format
- `sqlite3` for database updates
- `pathlib` for path handling

---

### 4. Update Experiment Configs

**Location:** `configs/experiment_*.yaml`

**Changes Required:**

Add new section to all experiment configs:

```yaml
# Streaming pipeline configuration
streaming_pipeline:
  enabled: true  # Set to false to disable streaming features
  shared_drive_path: "/mnt/shared/subject-drop"

# MLflow logging (in addition to W&B)
logging:
  use_wandb: true
  wandb_project: "just-drop-the-subject"
  use_mlflow: true
  mlflow_tracking_uri: "http://monitoring-server:5000"
  dir: "logs"
  level: "INFO"
```

**Example:** Update `configs/experiment_0_baseline.yaml` to add these sections

---

## New Components for SSRDE

### 1. Create Watcher Daemon: `pipeline/watcher/model_watcher.py`

**Purpose:** Monitor shared drive for training completion signals

**Location:** New file `pipeline/watcher/model_watcher.py`

**Key Components:**

```python
class ModelCompletionHandler(FileSystemEventHandler):
    """Handles filesystem events for training completion signals."""

    def on_created(self, event):
        """Triggered when .training_complete file is created."""
        pass

class EvaluationCompletionHandler(FileSystemEventHandler):
    """Handles filesystem events for evaluation completion signals."""

    def on_created(self, event):
        """Triggered when .eval_complete file is created."""
        pass

def main():
    """Main daemon entry point."""
    # Set up watchdog observers
    # Watch models/ directory for .training_complete
    # Watch evaluation/ directory for .eval_complete
    # Run forever with graceful shutdown
    pass
```

**Dependencies:**
- `watchdog` library for filesystem events
- `subprocess` for Slurm job submission
- Access to `pipeline/slurm_templates/`
- Access to SQLite databases

**Configuration File:** `pipeline/watcher/watcher_config.yaml`

```yaml
shared_drive_path: "/mnt/shared/subject-drop"
watch_paths:
  - "/mnt/shared/subject-drop/models"
  - "/mnt/shared/subject-drop/evaluation"
databases:
  training: "/mnt/shared/subject-drop/state/training.db"
  evaluation: "/mnt/shared/subject-drop/state/evaluation.db"
  analysis: "/mnt/shared/subject-drop/state/analysis.db"
slurm:
  account: "your_slurm_account"
  partition: "compute"
  templates_dir: "pipeline/slurm_templates"
retry:
  max_attempts: 3
  backoff_seconds: 60
logging:
  log_file: "/mnt/shared/subject-drop/logs/watcher.log"
  level: "INFO"
```

---

### 2. Create Slurm Templates: `pipeline/slurm_templates/`

**Purpose:** Template files for Slurm job submission

#### Template 1: `eval_job.sh.template`

**Location:** `pipeline/slurm_templates/eval_job.sh.template`

**Variables to Substitute:**
- `{EXPERIMENT_NAME}` - experiment identifier
- `{CHECKPOINT_PATH}` - path to model checkpoint
- `{SHARED_DRIVE}` - shared drive mount point
- `{JOB_ID}` - Slurm job ID (for logging)
- `{PYTHON_ENV}` - path to Python virtual environment

**Template Content:**

```bash
#!/bin/bash
#SBATCH --job-name=eval_{EXPERIMENT_NAME}
#SBATCH --output={SHARED_DRIVE}/logs/evaluation/%j.log
#SBATCH --error={SHARED_DRIVE}/logs/evaluation/%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account={SLURM_ACCOUNT}

# Activate Python environment
source {PYTHON_ENV}/bin/activate

# Run evaluation
python -m evaluation.runners.evaluation_runner \
    --config {SHARED_DRIVE}/config/evaluation/{EXPERIMENT_NAME}_eval.yaml \
    --checkpoint {CHECKPOINT_PATH}

# Check exit code
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully"
    # Write completion signal
    echo '{"timestamp": "'$(date -Iseconds)'", "status": "complete", "job_id": "'$SLURM_JOB_ID'"}' > \
        {SHARED_DRIVE}/evaluation/{EXPERIMENT_NAME}/.eval_complete
else
    echo "Evaluation failed"
    echo '{"timestamp": "'$(date -Iseconds)'", "status": "failed", "job_id": "'$SLURM_JOB_ID'"}' > \
        {SHARED_DRIVE}/evaluation/{EXPERIMENT_NAME}/.eval_failed
fi
```

#### Template 2: `analysis_job.sh.template`

**Location:** `pipeline/slurm_templates/analysis_job.sh.template`

**Variables to Substitute:**
- `{EXPERIMENT_NAME}` - experiment identifier
- `{SHARED_DRIVE}` - shared drive mount point
- `{R_LIBS}` - R library path

**Template Content:**

```bash
#!/bin/bash
#SBATCH --job-name=analysis_{EXPERIMENT_NAME}
#SBATCH --output={SHARED_DRIVE}/logs/analysis/%j.log
#SBATCH --error={SHARED_DRIVE}/logs/analysis/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account={SLURM_ACCOUNT}

# Load R module (adjust for your system)
module load R/4.3.0

# Set R library path
export R_LIBS_USER={R_LIBS}

# Change to analysis directory
cd {SHARED_DRIVE}/analysis

# Run R targets pipeline
Rscript -e "targets::tar_make()"

# Check exit code and update database
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully"
else
    echo "Analysis failed"
fi
```

---

### 3. Create Job Submitter: `pipeline/utils/slurm_submitter.py`

**Purpose:** Utility for generating and submitting Slurm jobs

**Location:** New file `pipeline/utils/slurm_submitter.py`

**Key Functions:**

```python
def generate_eval_script(experiment_name, checkpoint_path, config):
    """Generate evaluation Slurm script from template."""
    pass

def generate_analysis_script(experiment_name, config):
    """Generate analysis Slurm script from template."""
    pass

def submit_job(script_path):
    """Submit job to Slurm and return job ID."""
    pass

def check_job_status(job_id):
    """Check status of Slurm job."""
    pass
```

---

## Changes to Evaluation Pipeline (SSRDE)

### 1. Modify `evaluation/runners/evaluation_runner.py`

**Changes Required:**

#### Update Output Location

**Current:** Saves to local `evaluation/results/`

**New:** Save to shared drive `{SHARED_DRIVE}/evaluation/{experiment_name}/`

**Method to Modify:** `__init__()` method (lines 70-90)

**Change:**

```python
# In __init__ method
if hasattr(config, 'streaming_pipeline') and config.streaming_pipeline.get('enabled', False):
    # Use shared drive path
    base_output = Path(config.streaming_pipeline.shared_drive_path) / "evaluation" / experiment_name
else:
    # Use local path (existing behavior)
    base_output = Path(config.output_dir)

self.output_dir = base_output
self.output_dir.mkdir(parents=True, exist_ok=True)
```

#### Add Completion Signal Writing

**Where:** After line 502 in `run()` method, after all evaluations complete

**What to Add:**

```python
# Write completion signal for streaming pipeline
if hasattr(self.config, 'streaming_pipeline') and self.config.streaming_pipeline.get('enabled', False):
    self._write_eval_complete_signal()
```

**New Method to Add:**

```python
def _write_eval_complete_signal(self):
    """Write .eval_complete signal file to trigger analysis pipeline."""
    signal_file = self.output_dir / ".eval_complete"
    signal_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment': self.output_dir.name,
        'job_id': os.environ.get('SLURM_JOB_ID', 'unknown'),
        'status': 'complete'
    }

    with open(signal_file, 'w') as f:
        json.dump(signal_data, f, indent=2)
```

#### Add Parquet Output Format

**Where:** Throughout file, add Parquet output alongside JSONL

**Changes:**

In `run_blimp_evaluation()` (line 162), `run_null_subject_evaluation()` (line 189), etc.:

**Current:** Saves to JSONL only

**New:** Save to both JSONL (for backwards compatibility) and Parquet (for analysis)

**Add Method:**

```python
def _save_to_parquet(self, df, experiment_name, task_name):
    """Save results to partitioned Parquet format."""
    output_path = self.output_dir / "metrics" / f"{task_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add experiment metadata
    df['experiment'] = experiment_name
    df['timestamp'] = datetime.now().isoformat()

    # Save as Parquet
    df.to_parquet(output_path, index=False, compression='snappy')
```

---

### 2. Create Evaluation Config Generator

**Location:** New file `pipeline/utils/generate_eval_config.py`

**Purpose:** Generate evaluation configs from training configs

**Function:**

```python
def generate_eval_config(experiment_name, checkpoint_path, shared_drive_path):
    """
    Generate evaluation config YAML from experiment name and checkpoint.

    Saves to: {shared_drive_path}/config/evaluation/{experiment_name}_eval.yaml
    """
    pass
```

**Called By:** Watcher daemon when training completes

---

## Changes to Analysis Pipeline (SSRDE)

### 1. Create R targets Pipeline: `analysis/_targets.R`

**Purpose:** Convert existing R scripts to targets pipeline

**Location:** New file `analysis/_targets.R`

**Structure:**

```r
library(targets)

# Source all analysis scripts as functions
source("analysis/scripts/analysis_with_models.R")
source("analysis/scripts/first_epoch_analysis.R")
# ... etc

# Define pipeline
list(
  # Target 1: Load evaluation data
  tar_target(
    evaluation_data,
    load_evaluation_parquet("/mnt/shared/subject-drop/evaluation")
  ),

  # Target 2: Main acquisition analysis
  tar_target(
    acquisition_results,
    run_acquisition_analysis(evaluation_data)
  ),

  # Target 3: First epoch analysis
  tar_target(
    first_epoch_results,
    run_first_epoch_analysis(evaluation_data)
  ),

  # ... etc for all 6 scripts

  # Final target: Generate reports
  tar_target(
    final_reports,
    generate_comprehensive_reports(
      acquisition_results,
      first_epoch_results,
      # ... other results
    )
  )
)
```

---

### 2. Modify Existing R Scripts

**Changes Required for ALL scripts in `analysis/scripts/`:**

#### Make Scripts Function-Based

**Current:** Scripts run when sourced (imperative style)

**New:** Scripts define functions that return results

**Example Transformation:**

**Before** (in `analysis_with_models.R`):
```r
# Load data
data <- read_csv("analysis/tables/some_data.csv")

# Run analysis
results <- lmer(...)

# Save results
write_csv(results, "analysis/tables/output.csv")
```

**After:**
```r
run_acquisition_analysis <- function(evaluation_data) {
  # Analysis logic here
  results <- lmer(...)

  # Return results (targets will handle saving)
  return(results)
}
```

#### Update Data Loading

**Current:** Hardcoded paths to CSV files

**New:** Read from Parquet files on shared drive

**Add Helper Function in `analysis/scripts/utils.R`:**

```r
load_evaluation_parquet <- function(base_path) {
  library(arrow)

  # Read all parquet files from evaluation directory
  # Combine into single dataframe
  # Return for use in targets pipeline
}
```

#### Update Output Saving

**Current:** Each script saves directly to `analysis/tables/`

**New:** Targets handles saving, but maintain backwards compatibility

**Pattern:**

```r
run_analysis_step <- function(data) {
  # Analysis logic
  results <- perform_analysis(data)

  # Save to shared drive for backwards compatibility
  if (Sys.getenv("SAVE_OUTPUTS") == "true") {
    write_csv(results, file.path(Sys.getenv("OUTPUT_DIR"), "results.csv"))
  }

  # Return for targets pipeline
  return(results)
}
```

---

### 3. Update `run_complete_analysis.R`

**Current Purpose:** Sequential script runner

**New Purpose:** Wrapper that calls targets pipeline

**Location:** `analysis/scripts/run_complete_analysis.R`

**New Content:**

```r
# Set environment for backwards compatibility
Sys.setenv(SAVE_OUTPUTS = "true")
Sys.setenv(OUTPUT_DIR = "analysis/tables")

# Run targets pipeline
library(targets)
tar_make()

# Generate summary report
cat("Analysis pipeline completed via targets\n")
```

---

## New Components for Monitoring Server

### 1. MLflow Server Setup

**Location:** Monitoring server

**Installation:**

```bash
pip install mlflow
```

**Systemd Service:** Create `/etc/systemd/system/mlflow.service`

```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=mlflow
WorkingDirectory=/opt/mlflow
ExecStart=/opt/mlflow/venv/bin/mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////mnt/shared/subject-drop/state/mlflow.db \
    --default-artifact-root /mnt/shared/subject-drop/mlflow-artifacts
Restart=always

[Install]
WantedBy=multi-user.target
```

**Start Service:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
```

---

### 2. Job Monitoring Dashboard

**Location:** New directory `pipeline/monitoring/`

**Structure:**

```
pipeline/monitoring/
├── dashboard.py          # FastAPI application
├── templates/
│   ├── dashboard.html    # Main dashboard template
│   └── base.html         # Base template
└── static/
    ├── style.css         # Dashboard styles
    └── script.js         # Auto-refresh logic
```

#### Create `pipeline/monitoring/dashboard.py`

**Purpose:** Simple web dashboard for job monitoring

**Key Features:**
- Display current training status (from training.db)
- Show Slurm queue status (via SSH to SSRDE)
- List recent evaluation jobs
- Show analysis job status
- Links to MLflow runs

**Implementation:**

```python
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import sqlite3
import subprocess
from pathlib import Path

app = FastAPI()
app.mount("/static", StaticFiles(directory="pipeline/monitoring/static"), name="static")
templates = Jinja2Templates(directory="pipeline/monitoring/templates")

SHARED_DRIVE = "/mnt/shared/subject-drop"

@app.get("/")
async def dashboard(request: Request):
    # Get training status
    training_status = get_training_status()

    # Get evaluation queue
    eval_queue = get_evaluation_queue()

    # Get Slurm jobs
    slurm_jobs = get_slurm_status()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "training_status": training_status,
        "eval_queue": eval_queue,
        "slurm_jobs": slurm_jobs
    })

def get_training_status():
    """Query training.db for current training runs."""
    pass

def get_evaluation_queue():
    """Query evaluation.db for pending/running jobs."""
    pass

def get_slurm_status():
    """SSH to SSRDE and get squeue output."""
    pass
```

#### Create `pipeline/monitoring/templates/dashboard.html`

**Basic Template:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>ML Pipeline Monitor</title>
    <meta http-equiv="refresh" content="30">
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>ML Pipeline Monitor</h1>

    <section>
        <h2>Training Status (wild_west)</h2>
        <table>
            {% for run in training_status %}
            <tr>
                <td>{{ run.experiment }}</td>
                <td>{{ run.status }}</td>
                <td><a href="{{ mlflow_url }}/{{ run.run_id }}">View in MLflow</a></td>
            </tr>
            {% endfor %}
        </table>
    </section>

    <section>
        <h2>Evaluation Queue (SSRDE)</h2>
        <table>
            {% for job in eval_queue %}
            <tr class="{{ job.status }}">
                <td>{{ job.experiment }}</td>
                <td>{{ job.status }}</td>
                <td>{{ job.slurm_job_id }}</td>
            </tr>
            {% endfor %}
        </table>
    </section>

    <section>
        <h2>Slurm Jobs</h2>
        <pre>{{ slurm_jobs }}</pre>
    </section>
</body>
</html>
```

#### Systemd Service for Dashboard

**Create:** `/etc/systemd/system/pipeline-dashboard.service`

```ini
[Unit]
Description=Pipeline Monitoring Dashboard
After=network.target

[Service]
Type=simple
User=monitoring
WorkingDirectory=/path/to/subject-drop
ExecStart=/path/to/venv/bin/uvicorn pipeline.monitoring.dashboard:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Configuration Changes

### 1. Environment Variables

**Create:** `.env` file in project root (not committed to git)

```bash
# Shared drive configuration
SHARED_DRIVE_PATH=/mnt/shared/subject-drop

# MLflow configuration
MLFLOW_TRACKING_URI=http://monitoring-server:5000

# Slurm configuration (SSRDE)
SLURM_ACCOUNT=your_account_name
SLURM_PARTITION=compute

# SSH configuration (for monitoring server → SSRDE)
SSRDE_HOST=ssrde.yourdomain.edu
SSRDE_USER=your_username

# Python environment paths
WILD_WEST_PYTHON_ENV=/path/to/wild_west/venv
SSRDE_PYTHON_ENV=/path/to/ssrde/venv

# R environment (SSRDE)
R_LIBS_USER=/path/to/R/library
```

---

### 2. Watcher Configuration

**Create:** `pipeline/watcher/watcher_config.yaml`

```yaml
# Shared drive paths
shared_drive_path: "/mnt/shared/subject-drop"

# Paths to watch
watch_paths:
  models: "/mnt/shared/subject-drop/models"
  evaluation: "/mnt/shared/subject-drop/evaluation"

# Database paths
databases:
  training: "/mnt/shared/subject-drop/state/training.db"
  evaluation: "/mnt/shared/subject-drop/state/evaluation.db"
  analysis: "/mnt/shared/subject-drop/state/analysis.db"

# Slurm configuration
slurm:
  account: "your_account"
  partition: "compute"
  templates_dir: "pipeline/slurm_templates"
  python_env: "/path/to/ssrde/venv"
  r_libs: "/path/to/R/library"

# Retry configuration
retry:
  max_attempts: 3
  initial_backoff_seconds: 60
  max_backoff_seconds: 600

# Logging
logging:
  log_file: "/mnt/shared/subject-drop/logs/watcher.log"
  level: "INFO"
  max_bytes: 10485760  # 10MB
  backup_count: 5
```

---

## Database Setup

### 1. Create SQLite Schemas

**Script:** `pipeline/utils/setup_databases.py`

**Purpose:** Initialize all SQLite databases

**Location:** New file `pipeline/utils/setup_databases.py`

**Content:**

```python
import sqlite3
from pathlib import Path

def create_training_db(db_path):
    """Create training.db schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            checkpoint_path TEXT,
            config_path TEXT,
            hostname TEXT,
            user TEXT,
            mlflow_run_id TEXT,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_run_id INTEGER,
            checkpoint_path TEXT,
            step INTEGER,
            epoch INTEGER,
            created_at TIMESTAMP,
            metrics TEXT,
            FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
        )
    """)

    conn.commit()
    conn.close()

def create_evaluation_db(db_path):
    """Create evaluation.db schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            training_run_id INTEGER,
            status TEXT NOT NULL,
            slurm_job_id TEXT,
            submitted_at TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            checkpoint_path TEXT,
            mlflow_run_id TEXT,
            error_message TEXT,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_job_id INTEGER,
            dataset_name TEXT,
            metric_name TEXT,
            metric_value REAL,
            evaluated_at TIMESTAMP,
            FOREIGN KEY (evaluation_job_id) REFERENCES evaluation_jobs(id)
        )
    """)

    conn.commit()
    conn.close()

def create_analysis_db(db_path):
    """Create analysis.db schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            evaluation_job_id INTEGER,
            status TEXT NOT NULL,
            slurm_job_id TEXT,
            submitted_at TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            targets_manifest_path TEXT,
            error_message TEXT,
            FOREIGN KEY (evaluation_job_id) REFERENCES evaluation_jobs(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_job_id INTEGER,
            output_type TEXT,
            output_path TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (analysis_job_id) REFERENCES analysis_jobs(id)
        )
    """)

    conn.commit()
    conn.close()

def main():
    """Set up all databases."""
    import os

    shared_drive = os.environ.get('SHARED_DRIVE_PATH', '/mnt/shared/subject-drop')
    state_dir = Path(shared_drive) / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)

    create_training_db(state_dir / 'training.db')
    create_evaluation_db(state_dir / 'evaluation.db')
    create_analysis_db(state_dir / 'analysis.db')

    print(f"Databases created in {state_dir}")

if __name__ == '__main__':
    main()
```

**Run Once:** `python pipeline/utils/setup_databases.py`

---

### 2. Enable WAL Mode

**Script:** `pipeline/utils/enable_wal.py`

```python
import sqlite3
from pathlib import Path
import os

def enable_wal(db_path):
    """Enable WAL mode for better concurrency."""
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=5000')  # 5 second timeout
    conn.close()
    print(f"WAL mode enabled for {db_path}")

def main():
    shared_drive = os.environ.get('SHARED_DRIVE_PATH', '/mnt/shared/subject-drop')
    state_dir = Path(shared_drive) / 'state'

    for db_file in ['training.db', 'evaluation.db', 'analysis.db', 'mlflow.db']:
        db_path = state_dir / db_file
        if db_path.exists():
            enable_wal(db_path)

if __name__ == '__main__':
    main()
```

---

## Deployment Checklist

### Phase 1: Shared Drive Setup

**On Shared Network Drive:**

- [ ] Create directory structure (`/mnt/shared/subject-drop/...`)
- [ ] Set appropriate permissions (writable by wild_west and SSRDE users)
- [ ] Verify mount is accessible from all three servers
- [ ] Run `pipeline/utils/setup_databases.py` to create SQLite databases
- [ ] Run `pipeline/utils/enable_wal.py` to enable WAL mode

### Phase 2: Training Pipeline (wild_west)

**Code Changes:**

- [ ] Modify `model_foundry/trainer.py` to add completion hook
- [ ] Modify `model_foundry/training/checkpointing.py` for shared drive paths
- [ ] Create `pipeline/utils/signal_writer.py`
- [ ] Update experiment configs to add `streaming_pipeline` section

**Testing:**

- [ ] Run test training with streaming enabled
- [ ] Verify checkpoint saved to shared drive
- [ ] Verify `.training_complete` signal file created
- [ ] Verify training.db updated

### Phase 3: Watcher Daemon (SSRDE)

**Code Changes:**

- [ ] Create `pipeline/watcher/model_watcher.py`
- [ ] Create `pipeline/watcher/watcher_config.yaml`
- [ ] Create `pipeline/slurm_templates/eval_job.sh.template`
- [ ] Create `pipeline/slurm_templates/analysis_job.sh.template`
- [ ] Create `pipeline/utils/slurm_submitter.py`

**Deployment:**

- [ ] Install watchdog library: `pip install watchdog`
- [ ] Create systemd service for watcher (or use screen/tmux)
- [ ] Start watcher daemon
- [ ] Monitor logs: `tail -f /mnt/shared/subject-drop/logs/watcher.log`

**Testing:**

- [ ] Trigger training completion manually
- [ ] Verify watcher detects signal file
- [ ] Verify Slurm job submitted
- [ ] Verify evaluation.db updated

### Phase 4: Evaluation Pipeline (SSRDE)

**Code Changes:**

- [ ] Modify `evaluation/runners/evaluation_runner.py` for shared drive output
- [ ] Add `.eval_complete` signal writing
- [ ] Add Parquet output format
- [ ] Create `pipeline/utils/generate_eval_config.py`

**Configuration:**

- [ ] Create evaluation config templates in `{SHARED_DRIVE}/config/evaluation/`

**Testing:**

- [ ] Run evaluation job via Slurm manually
- [ ] Verify outputs saved to shared drive
- [ ] Verify `.eval_complete` signal created
- [ ] Verify Parquet files created

### Phase 5: Analysis Pipeline (SSRDE)

**Code Changes:**

- [ ] Create `analysis/_targets.R`
- [ ] Refactor existing R scripts to be function-based
- [ ] Update `run_complete_analysis.R` to call targets
- [ ] Add Parquet reading utilities

**R Package Installation (SSRDE):**

- [ ] Install targets: `install.packages("targets")`
- [ ] Install arrow: `install.packages("arrow")`

**Testing:**

- [ ] Run analysis job via Slurm manually
- [ ] Verify targets pipeline executes
- [ ] Verify outputs saved to shared drive

### Phase 6: Monitoring Server

**MLflow Setup:**

- [ ] Install MLflow: `pip install mlflow`
- [ ] Create systemd service
- [ ] Start MLflow server
- [ ] Verify accessible at `http://monitoring-server:5000`
- [ ] Test logging from wild_west

**Dashboard Setup:**

- [ ] Create `pipeline/monitoring/dashboard.py`
- [ ] Create HTML templates
- [ ] Install FastAPI: `pip install fastapi uvicorn`
- [ ] Create systemd service
- [ ] Start dashboard
- [ ] Verify accessible at `http://monitoring-server:8000`

**Security:**

- [ ] Set up reverse proxy (nginx/caddy) with authentication
- [ ] Enable HTTPS
- [ ] Configure firewall rules

### Phase 7: End-to-End Testing

**Full Pipeline Test:**

- [ ] Start fresh training run on wild_west
- [ ] Monitor watcher logs on SSRDE
- [ ] Verify evaluation job submitted and completes
- [ ] Verify analysis job submitted and completes
- [ ] Check MLflow UI for all metrics
- [ ] Check dashboard for job status
- [ ] Verify all outputs in correct locations

**Error Testing:**

- [ ] Test training failure (no signal file)
- [ ] Test evaluation failure (verify error handling)
- [ ] Test watcher restart (verify it picks up missed signals)
- [ ] Test database locking (concurrent access)

---

## Migration Strategy

### Option 1: Gradual Migration (Recommended)

**Week 1-2:**
- Set up shared drive structure
- Implement training completion hook (backwards compatible)
- Test signal writing

**Week 3:**
- Implement watcher daemon
- Test evaluation job submission

**Week 4:**
- Migrate evaluation to Parquet output
- Set up R targets pipeline

**Week 5:**
- Deploy monitoring server
- Full end-to-end testing

### Option 2: Parallel Systems

**Run both systems in parallel:**
- Keep existing manual workflow
- Add streaming pipeline as opt-in (via config flag)
- Test thoroughly before switching default

### Option 3: Single Experiment Pilot

**Test with one experiment:**
- Choose experiment_0_baseline as pilot
- Run through full streaming pipeline
- Compare results with manual workflow
- Once validated, migrate remaining experiments

---

## Rollback Plan

**If issues arise:**

1. **Disable streaming pipeline:** Set `streaming_pipeline.enabled: false` in configs
2. **Training continues normally:** No impact on wild_west operations
3. **Manual evaluation:** Use existing `evaluation/runners/evaluation_runner.py` locally
4. **Manual analysis:** Use existing `analysis/scripts/run_complete_analysis.R`

**All existing workflows remain functional** - streaming pipeline is purely additive.

---

## Dependencies to Install

### Python Packages (all servers)

```bash
pip install mlflow watchdog pydantic pyyaml fastapi uvicorn
```

### R Packages (SSRDE)

```r
install.packages(c("targets", "arrow", "tidyverse", "lme4", "emmeans"))
```

### System Packages

**SSRDE:**
- Slurm client tools (typically already installed)
- R 4.0+ with development headers

**Monitoring Server:**
- nginx or caddy (for reverse proxy)
- SQLite command-line tools

---

## Summary of Files to Create

**New Python Files:**
1. `pipeline/watcher/model_watcher.py`
2. `pipeline/utils/signal_writer.py`
3. `pipeline/utils/slurm_submitter.py`
4. `pipeline/utils/generate_eval_config.py`
5. `pipeline/utils/setup_databases.py`
6. `pipeline/utils/enable_wal.py`
7. `pipeline/monitoring/dashboard.py`

**New Config Files:**
8. `pipeline/watcher/watcher_config.yaml`
9. `.env` (not committed)

**New Templates:**
10. `pipeline/slurm_templates/eval_job.sh.template`
11. `pipeline/slurm_templates/analysis_job.sh.template`
12. `pipeline/monitoring/templates/dashboard.html`

**New R Files:**
13. `analysis/_targets.R`
14. `analysis/scripts/utils.R` (helper functions)

**Modified Files:**
15. `model_foundry/trainer.py`
16. `model_foundry/training/checkpointing.py`
17. `evaluation/runners/evaluation_runner.py`
18. All files in `configs/experiment_*.yaml`
19. All R scripts in `analysis/scripts/` (refactor to functions)
20. `analysis/scripts/run_complete_analysis.R`

---

## Key Integration Points

### 1. Training → Evaluation Trigger

**File:** `model_foundry/trainer.py` line ~360
**Action:** Write `.training_complete` signal file
**Triggered by:** `TrainingLoop.run()` completion
**Detects:** Watcher on SSRDE
**Result:** Slurm evaluation job submitted

### 2. Evaluation → Analysis Trigger

**File:** `evaluation/runners/evaluation_runner.py` line ~502
**Action:** Write `.eval_complete` signal file
**Triggered by:** All evaluations complete
**Detects:** Watcher on SSRDE
**Result:** Slurm analysis job submitted

### 3. All Stages → MLflow

**Files:**
- `model_foundry/trainer.py` (training metrics)
- `evaluation/runners/evaluation_runner.py` (eval metrics)
- R scripts (analysis results)

**Action:** Log metrics to MLflow
**Result:** Visible in MLflow UI on monitoring server

### 4. All Stages → Dashboard

**Source:** SQLite databases in `{SHARED_DRIVE}/state/`
**Read by:** Dashboard on monitoring server
**Display:** Current status of all pipeline stages

---

## Troubleshooting Guide

### Issue: Signal file not detected

**Check:**
1. Watcher daemon running: `ps aux | grep model_watcher`
2. File permissions on shared drive
3. Watcher log: `tail -f /mnt/shared/subject-drop/logs/watcher.log`

### Issue: Slurm job fails to start

**Check:**
1. Slurm account and partition correct in config
2. Template file exists and is readable
3. Job submission logs in watcher.log
4. Slurm limits: `squeue -u $USER`

### Issue: Evaluation outputs not found

**Check:**
1. Shared drive mounted on SSRDE compute nodes
2. Output directory writable
3. Slurm job logs: `{SHARED_DRIVE}/logs/evaluation/{job_id}.log`

### Issue: Database locked errors

**Check:**
1. WAL mode enabled: `sqlite3 training.db 'PRAGMA journal_mode;'`
2. Busy timeout set
3. Not too many concurrent writes

---

_End of Implementation Guide_
