# Streaming Pipeline Architecture Specification

**Version:** 1.0
**Date:** 2025-09-30
**Status:** Implementation Ready

## Executive Summary

This document specifies a modern streaming pipeline architecture for coordinating machine learning training, evaluation, and analysis across three servers sharing a network drive. The system enables manual training on a GPU server (wild_west), automated evaluation via Slurm (SSRDE), and real-time monitoring via a web interface (monitoring server).

**Key Design Principles:**
- Event-driven architecture using file-based signaling
- Minimal footprint on shared resources
- Clear separation of concerns across servers
- Observable, debuggable state at all times
- Graceful handling of failures and retries

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHARED NETWORK DRIVE                          │
│                  /mnt/shared/subject-drop/                       │
│                                                                   │
│  📁 models/          ← Training outputs (wild_west writes)       │
│  📁 evaluation/      ← Evaluation results (SSRDE writes)         │
│  📁 analysis/        ← Analysis outputs (SSRDE writes)           │
│  📁 state/           ← SQLite DBs for coordination               │
│  📁 config/          ← Shared configurations                     │
│  📁 logs/            ← All server logs                           │
└─────────────────────────────────────────────────────────────────┘
         │                        │                    │
         │                        │                    │
         ▼                        ▼                    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WILD_WEST     │    │     SSRDE       │    │   MONITORING    │
│                 │    │                 │    │                 │
│ 4x A6000 GPUs   │    │ Slurm Cluster   │    │ Web Services    │
│ Manual Training │    │ Auto Eval/Anal  │    │ Internet-Facing │
│                 │    │                 │    │                 │
│ Components:     │    │ Components:     │    │ Components:     │
│ • Training      │    │ • Watcher       │    │ • MLflow UI     │
│   scripts       │    │ • Slurm jobs    │    │ • Dashboard     │
│ • Completion    │    │ • Eval workers  │    │                 │
│   hooks         │    │ • R pipeline    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Three-Server Architecture

### Server 1: wild_west (Training)

**Purpose:** Manual model training with GPU resources

**Characteristics:**
- Shared server with informal use agreement
- 4x NVIDIA A6000 GPUs
- Manual job execution (no automation)
- Must be respectful of other users

**Responsibilities:**
- Execute training scripts when requested
- Save model checkpoints to shared drive
- Write completion signals for downstream automation
- Log training metrics to MLflow

**Does NOT Run:**
- No daemon processes
- No background watchers
- No automated job scheduling

**Resource Footprint:**
- Active only during training
- Cleanup after training completes
- Minimal disk I/O to shared drive during training

---

### Server 2: SSRDE (Evaluation & Analysis)

**Purpose:** Automated evaluation and analysis via Slurm

**Characteristics:**
- Robust job queuing with Slurm
- Shared cluster resource
- Automated job submission and execution
- Runs intensive R analysis workloads

**Responsibilities:**
- Watch for training completion signals
- Submit evaluation jobs to Slurm
- Submit analysis jobs to Slurm
- Maintain state databases
- Log evaluation/analysis metrics to MLflow

**Runs:**
- Single lightweight watcher daemon (minimal CPU/memory)
- Slurm-scheduled evaluation workers
- Slurm-scheduled R analysis pipeline
- Optional: Redis Streams for internal coordination

**Resource Footprint:**
- Watcher: <100MB RAM, <1% CPU
- All compute via Slurm (fair scheduling)
- Respects Slurm quotas and limits

---

### Server 3: monitoring (Web Interface)

**Purpose:** Real-time monitoring and experiment tracking

**Characteristics:**
- Internet-facing web server
- Read-only access to shared state
- No computational workloads

**Responsibilities:**
- Host MLflow tracking server
- Provide web dashboard for job monitoring
- Display Slurm queue status
- Track training progress and checkpoints

**Runs:**
- MLflow server (experiment tracking UI)
- Custom dashboard (job queue monitoring)
- Optional: notification services

**Resource Footprint:**
- Minimal: web services only
- No data processing
- Read-only database access

---

## Directory Structure

### Shared Network Drive Layout

```
/mnt/shared/subject-drop/
│
├── config/                          # Shared configurations
│   ├── experiments/                 # Training configs
│   │   ├── experiment_01.yaml
│   │   └── experiment_02.yaml
│   └── evaluation/                  # Evaluation configs
│       ├── base_eval.yaml
│       └── dataset_configs/
│
├── models/                          # Training outputs (wild_west)
│   └── {experiment_name}/
│       ├── checkpoints/
│       │   ├── checkpoint-1000/
│       │   │   ├── model.safetensors
│       │   │   ├── optimizer.pt
│       │   │   └── metadata.json
│       │   ├── checkpoint-2000/
│       │   └── checkpoint-3000/
│       ├── final/
│       │   └── model.safetensors
│       ├── .training_complete       # Signal file (JSON)
│       └── training_metadata.json   # Full training metadata
│
├── evaluation/                      # Evaluation outputs (SSRDE)
│   └── {experiment_name}/
│       ├── metrics/
│       │   ├── accuracy.parquet     # Partitioned Parquet files
│       │   ├── perplexity.parquet
│       │   └── loss.parquet
│       ├── predictions/
│       │   └── {dataset_name}.parquet
│       ├── .eval_complete           # Signal file (JSON)
│       └── eval_metadata.json
│
├── analysis/                        # Analysis outputs (SSRDE)
│   ├── _targets/                    # R targets cache
│   ├── figures/
│   │   └── {experiment_name}/
│   ├── tables/
│   │   └── {experiment_name}/
│   ├── reports/
│   │   └── {experiment_name}/
│   └── _targets.R                   # R targets pipeline
│
├── state/                           # Coordination state (SSRDE manages)
│   ├── training.db                  # SQLite: training run tracking
│   ├── evaluation.db                # SQLite: evaluation queue/status
│   ├── analysis.db                  # SQLite: analysis status
│   ├── mlflow.db                    # MLflow backend store
│   └── locks/                       # File-based locks if needed
│
├── mlflow-artifacts/                # MLflow artifact storage
│   └── {run_id}/
│
└── logs/                            # All server logs
    ├── training/                    # From wild_west
    │   └── {experiment_name}.log
    ├── evaluation/                  # From SSRDE eval jobs
    │   └── {slurm_job_id}.log
    ├── analysis/                    # From SSRDE analysis jobs
    │   └── {slurm_job_id}.log
    └── watcher.log                  # SSRDE watcher daemon
```

---

## Event Flow Architecture

### End-to-End Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: TRAINING (wild_west)                                    │
└──────────────────────────────────────────────────────────────────┘
    │
    │ 1. User starts training manually
    │    $ python train.py --config experiment_01.yaml
    │
    │ 2. Training script runs
    │    • Logs metrics to MLflow (real-time)
    │    • Saves checkpoints to /mnt/shared/.../models/
    │    • Updates training.db periodically
    │
    │ 3. Training completes
    │    • Saves final model
    │    • Writes .training_complete signal file
    │    • Logs final metrics to MLflow
    │
    ▼

┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: EVALUATION TRIGGER (SSRDE watcher)                      │
└──────────────────────────────────────────────────────────────────┘
    │
    │ 4. Watcher detects .training_complete
    │    • Filesystem event triggers handler
    │    • Reads signal file metadata
    │    • Validates model files exist
    │
    │ 5. Generate Slurm submission script
    │    • Creates batch script for evaluation
    │    • Specifies resources (CPU, memory, time)
    │    • Sets up environment and paths
    │
    │ 6. Submit to Slurm queue
    │    • sbatch evaluation_script.sh
    │    • Capture job ID
    │    • Update evaluation.db (status: queued)
    │    • Log to MLflow
    │
    ▼

┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3: EVALUATION EXECUTION (SSRDE compute node)               │
└──────────────────────────────────────────────────────────────────┘
    │
    │ 7. Slurm allocates resources
    │    • Job moves from pending → running
    │    • Update evaluation.db (status: running)
    │
    │ 8. Evaluation worker runs
    │    • Loads model from checkpoint
    │    • Runs evaluation on configured datasets
    │    • Streams metrics to Parquet files
    │    • Logs to MLflow (linked to training run)
    │
    │ 9. Evaluation completes
    │    • Writes .eval_complete signal file
    │    • Updates evaluation.db (status: complete)
    │    • Logs final metrics to MLflow
    │
    ▼

┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4: ANALYSIS TRIGGER (SSRDE watcher)                        │
└──────────────────────────────────────────────────────────────────┘
    │
    │ 10. Watcher detects .eval_complete
    │     • Filesystem event triggers handler
    │     • Validates evaluation outputs exist
    │     • Checks if analysis is needed
    │
    │ 11. Submit R analysis job to Slurm
    │     • Creates Slurm script for R pipeline
    │     • sbatch analysis_script.sh
    │     • Update analysis.db (status: queued)
    │
    ▼

┌──────────────────────────────────────────────────────────────────┐
│ PHASE 5: ANALYSIS EXECUTION (SSRDE compute node)                 │
└──────────────────────────────────────────────────────────────────┘
    │
    │ 12. R targets pipeline runs
    │     • Reads Parquet evaluation data
    │     • Generates figures
    │     • Creates tables
    │     • Renders reports
    │
    │ 13. Analysis completes
    │     • Writes outputs to analysis/
    │     • Updates analysis.db (status: complete)
    │     • Logs to MLflow
    │
    ▼

┌──────────────────────────────────────────────────────────────────┐
│ OBSERVABLE AT ALL TIMES (monitoring server)                      │
└──────────────────────────────────────────────────────────────────┘
    │
    │ • MLflow UI shows all runs, metrics, artifacts
    │ • Dashboard shows current queue status
    │ • All logs accessible in shared drive
    │ • SQLite databases queryable for debugging
```

---

## Signal File Specifications

### .training_complete

**Location:** `/mnt/shared/subject-drop/models/{experiment_name}/.training_complete`

**Format:** JSON

**Required Fields:**
- `timestamp`: ISO 8601 timestamp of completion
- `experiment`: Experiment name (matches directory)
- `checkpoint_path`: Absolute path to final checkpoint directory
- `hostname`: Server where training ran (should be "wild_west")
- `user`: Username who ran training

**Optional Fields:**
- `config_path`: Path to training config file
- `num_epochs`: Total epochs trained
- `final_loss`: Final training loss
- `total_steps`: Total training steps
- `git_commit`: Git commit hash at training time

**Purpose:**
- Triggers evaluation pipeline on SSRDE
- Provides metadata for evaluation job setup
- Enables debugging of training completion

**Example Structure:**
```json
{
  "timestamp": "2025-09-30T14:32:15Z",
  "experiment": "experiment_01",
  "checkpoint_path": "/mnt/shared/subject-drop/models/experiment_01/final",
  "hostname": "wild_west",
  "user": "tmorton",
  "config_path": "/mnt/shared/subject-drop/config/experiments/experiment_01.yaml",
  "num_epochs": 10,
  "final_loss": 2.34,
  "total_steps": 50000,
  "git_commit": "a94429c"
}
```

---

### .eval_complete

**Location:** `/mnt/shared/subject-drop/evaluation/{experiment_name}/.eval_complete`

**Format:** JSON

**Required Fields:**
- `timestamp`: ISO 8601 timestamp of completion
- `experiment`: Experiment name
- `job_id`: Slurm job ID that ran evaluation
- `status`: "complete" or "failed"

**Optional Fields:**
- `datasets_evaluated`: List of dataset names
- `metrics_summary`: Key metrics (accuracy, perplexity, etc.)
- `eval_duration_seconds`: Time taken for evaluation
- `error_message`: If status is "failed", error details

**Purpose:**
- Triggers analysis pipeline on SSRDE
- Provides metadata for analysis job setup
- Enables tracking of evaluation status

---

## State Management

### SQLite Database Schemas

#### training.db

**Purpose:** Track training runs across all experiments

**Table: training_runs**
- `id`: INTEGER PRIMARY KEY
- `experiment_name`: TEXT UNIQUE NOT NULL
- `status`: TEXT (started, running, completed, failed)
- `started_at`: TIMESTAMP
- `completed_at`: TIMESTAMP
- `checkpoint_path`: TEXT
- `config_path`: TEXT
- `hostname`: TEXT
- `user`: TEXT
- `mlflow_run_id`: TEXT
- `metadata`: TEXT (JSON blob)

**Table: checkpoints**
- `id`: INTEGER PRIMARY KEY
- `training_run_id`: INTEGER FOREIGN KEY
- `checkpoint_path`: TEXT
- `step`: INTEGER
- `epoch`: INTEGER
- `created_at`: TIMESTAMP
- `metrics`: TEXT (JSON blob)

---

#### evaluation.db

**Purpose:** Track evaluation job queue and status

**Table: evaluation_jobs**
- `id`: INTEGER PRIMARY KEY
- `experiment_name`: TEXT NOT NULL
- `training_run_id`: INTEGER FOREIGN KEY
- `status`: TEXT (queued, running, completed, failed)
- `slurm_job_id`: TEXT
- `submitted_at`: TIMESTAMP
- `started_at`: TIMESTAMP
- `completed_at`: TIMESTAMP
- `checkpoint_path`: TEXT
- `mlflow_run_id`: TEXT
- `error_message`: TEXT
- `metadata`: TEXT (JSON blob)

**Table: evaluation_results**
- `id`: INTEGER PRIMARY KEY
- `evaluation_job_id`: INTEGER FOREIGN KEY
- `dataset_name`: TEXT
- `metric_name`: TEXT
- `metric_value`: REAL
- `evaluated_at`: TIMESTAMP

---

#### analysis.db

**Purpose:** Track analysis pipeline status

**Table: analysis_jobs**
- `id`: INTEGER PRIMARY KEY
- `experiment_name`: TEXT NOT NULL
- `evaluation_job_id`: INTEGER FOREIGN KEY
- `status`: TEXT (queued, running, completed, failed)
- `slurm_job_id`: TEXT
- `submitted_at`: TIMESTAMP
- `started_at`: TIMESTAMP
- `completed_at`: TIMESTAMP
- `targets_manifest_path`: TEXT
- `error_message`: TEXT

**Table: analysis_outputs**
- `id`: INTEGER PRIMARY KEY
- `analysis_job_id`: INTEGER FOREIGN KEY
- `output_type`: TEXT (figure, table, report)
- `output_path`: TEXT
- `created_at`: TIMESTAMP

---

## Component Specifications

### Component 1: Training Completion Hook (wild_west)

**Language:** Python

**Integration Point:** End of training script

**Responsibilities:**
1. Save final model checkpoint
2. Collect training metadata
3. Write .training_complete signal file
4. Update training.db
5. Log final metrics to MLflow

**Error Handling:**
- Retry signal file write if network issues
- Validate file was written successfully
- Log errors but don't fail training

**Dependencies:**
- Access to shared network drive
- MLflow Python client
- SQLite3 Python library

**Configuration:**
- Path to shared drive (environment variable)
- MLflow tracking URI
- Experiment name (from training config)

---

### Component 2: Model Watcher Daemon (SSRDE)

**Language:** Python

**Runtime:** Long-running daemon process

**Responsibilities:**
1. Watch for .training_complete signal files
2. Validate model files exist and are complete
3. Generate Slurm evaluation job scripts
4. Submit jobs to Slurm via sbatch
5. Update evaluation.db with job status
6. Watch for .eval_complete signal files
7. Generate Slurm analysis job scripts
8. Submit analysis jobs to Slurm

**Implementation Strategy:**
- Use watchdog library for filesystem events
- Debounce events (wait for file write completion)
- Validate signal file JSON before processing
- Idempotent job submission (check if already queued)
- Graceful shutdown on SIGTERM
- Auto-restart on crash (via systemd/supervisor)

**Error Handling:**
- Log all errors to watcher.log
- Retry failed Slurm submissions (exponential backoff)
- Alert on repeated failures
- Continue watching even if individual jobs fail

**Performance Considerations:**
- Minimal CPU/memory usage (<100MB)
- Efficient inotify-based file watching
- Non-blocking I/O for database updates
- Rate limiting for Slurm submissions

**Configuration:**
- Paths to watch (models/, evaluation/ directories)
- Slurm submission templates
- Retry policies
- Database connection strings

---

### Component 3: Evaluation Worker (SSRDE, Slurm-scheduled)

**Language:** Python

**Runtime:** Batch job (Slurm)

**Responsibilities:**
1. Load model from checkpoint
2. Load evaluation datasets
3. Run evaluation metrics
4. Stream results to Parquet files
5. Write .eval_complete signal file
6. Update evaluation.db
7. Log metrics to MLflow

**Input:**
- Experiment name (from command-line arg)
- Checkpoint path (from command-line arg)
- Evaluation config (from shared config directory)

**Output:**
- Parquet files with metrics
- .eval_complete signal file
- Updated evaluation.db
- MLflow logged metrics

**Error Handling:**
- Comprehensive error logging
- Partial evaluation recovery (save what succeeded)
- Write error details to .eval_failed if critical failure
- Update evaluation.db with failure status

**Resource Requirements:**
- CPUs: 8-16 cores
- Memory: 32-64 GB
- Time limit: 2-4 hours
- GPU: Optional (depending on model size)

**Optimization:**
- Batch evaluation where possible
- Stream results to disk (don't hold in memory)
- Use Arrow/Parquet for efficient I/O
- Parallel dataset evaluation

---

### Component 4: Analysis Pipeline (SSRDE, Slurm-scheduled)

**Language:** R

**Runtime:** Batch job (Slurm)

**Framework:** R targets

**Responsibilities:**
1. Read evaluation Parquet files
2. Run statistical analysis
3. Generate publication-quality figures
4. Create summary tables
5. Render reports (Quarto/RMarkdown)
6. Update analysis.db

**Input:**
- Experiment name (from environment variable)
- Evaluation results path
- Analysis config

**Output:**
- Figures (PNG, PDF)
- Tables (CSV, LaTeX)
- Reports (HTML, PDF)
- Updated analysis.db

**R targets Pipeline Structure:**
- Target: load_evaluation_data
- Target: compute_statistics
- Target: generate_figures
- Target: create_tables
- Target: render_report
- Target: finalize_outputs

**Error Handling:**
- Targets caching enables partial re-runs
- Log R errors to analysis job log
- Update analysis.db with error status
- Generate partial outputs where possible

**Resource Requirements:**
- CPUs: 16-32 cores (for parallel computation)
- Memory: 64-128 GB
- Time limit: 4-8 hours

---

### Component 5: MLflow Tracking Server (monitoring)

**Service:** MLflow

**Runtime:** Long-running web service

**Configuration:**
- Backend store: SQLite at `/mnt/shared/subject-drop/state/mlflow.db`
- Artifact store: `/mnt/shared/subject-drop/mlflow-artifacts/`
- Host: 0.0.0.0 (internet-facing)
- Port: 5000

**Responsibilities:**
1. Serve MLflow web UI
2. Accept metric logging from all servers
3. Store experiment metadata
4. Provide model registry
5. Enable artifact browsing

**Features to Enable:**
- Experiment comparison
- Metric visualization
- Artifact download
- Model versioning
- Run linking (training → evaluation)

**Access Control:**
- Basic auth (username/password)
- Or reverse proxy with authentication
- HTTPS via nginx/caddy

---

### Component 6: Job Monitoring Dashboard (monitoring)

**Language:** Python (FastAPI) + HTML/JavaScript

**Runtime:** Long-running web service

**Responsibilities:**
1. Display current training status
2. Show Slurm queue for SSRDE
3. List recent evaluation jobs
4. Show analysis job status
5. Provide quick links to MLflow runs

**Data Sources:**
- SQLite databases (training.db, evaluation.db, analysis.db)
- SSH to SSRDE for `squeue` output
- Filesystem checks for active training

**UI Requirements:**
- Auto-refresh every 30 seconds
- Color-coded status (green=complete, yellow=running, red=failed)
- Links to logs
- Links to MLflow runs
- Search/filter by experiment name

**Implementation Approach:**
- FastAPI backend (lightweight)
- Simple HTML templates (Jinja2)
- Minimal JavaScript (or htmx for live updates)
- No database writes (read-only)

**Access Control:**
- Same as MLflow (basic auth or reverse proxy)

---

## Data Formats

### Parquet for Evaluation Metrics

**Why Parquet:**
- Columnar storage (efficient for R/dplyr)
- Excellent compression
- Schema enforcement
- Fast I/O with Arrow
- Append-friendly via partitioning

**Partitioning Strategy:**
```
evaluation/{experiment_name}/metrics/
├── dataset=blimp/metric=accuracy/data.parquet
├── dataset=blimp/metric=perplexity/data.parquet
├── dataset=blimp_supplement/metric=accuracy/data.parquet
└── dataset=blimp_supplement/metric=perplexity/data.parquet
```

**Schema Example (accuracy.parquet):**
- `experiment`: string
- `checkpoint`: string
- `dataset`: string
- `item_group`: string (e.g., "b1_number")
- `form`: string (e.g., "forms_vs_default")
- `accuracy`: double
- `timestamp`: timestamp

**Writing:**
- Use PyArrow or pandas with partitioning
- Append new results without overwriting
- Validate schema before writing

**Reading (R):**
- Use arrow package
- Automatic partition filtering
- Lazy evaluation for large datasets

---

### JSON for Signal Files

**Why JSON:**
- Human-readable
- Easy to validate
- Widely supported
- Self-describing

**Standards:**
- ISO 8601 timestamps
- UTF-8 encoding
- Pretty-printed for readability
- Schema validation before processing

---

### SQLite for State

**Why SQLite:**
- File-based (works on network drive)
- No server process
- ACID transactions
- Excellent Python/R support
- WAL mode for better concurrency

**Configuration:**
- Enable WAL mode (Write-Ahead Logging)
- Set busy timeout (handle concurrent access)
- Regular backups
- VACUUM periodically

**Concurrency Handling:**
- Use WAL mode
- Retry on SQLITE_BUSY
- Short transactions
- Read-heavy workload (rarely contended)

---

## Slurm Job Templates

### Evaluation Job Template

**Filename:** `slurm_eval_{experiment_name}.sh`

**Required Directives:**
- `#SBATCH --job-name=eval_{experiment_name}`
- `#SBATCH --output=/mnt/shared/subject-drop/logs/evaluation/%j.log`
- `#SBATCH --error=/mnt/shared/subject-drop/logs/evaluation/%j.err`
- `#SBATCH --time=02:00:00` (adjust based on model size)
- `#SBATCH --mem=32G` (adjust based on dataset size)
- `#SBATCH --cpus-per-task=8`
- Optional: `#SBATCH --gres=gpu:1` (if GPU evaluation)

**Environment Setup:**
- Load required modules (Python, CUDA if needed)
- Activate virtual environment
- Set PYTHONPATH

**Execution:**
- Change to working directory
- Run evaluation script with arguments
- Capture exit code
- Update database on completion/failure

**Cleanup:**
- Remove temporary files
- Compress logs if needed

---

### Analysis Job Template

**Filename:** `slurm_analysis_{experiment_name}.sh`

**Required Directives:**
- `#SBATCH --job-name=analysis_{experiment_name}`
- `#SBATCH --output=/mnt/shared/subject-drop/logs/analysis/%j.log`
- `#SBATCH --error=/mnt/shared/subject-drop/logs/analysis/%j.err`
- `#SBATCH --time=04:00:00`
- `#SBATCH --mem=64G`
- `#SBATCH --cpus-per-task=16`

**Environment Setup:**
- Load R module
- Set R library path
- Load required R packages

**Execution:**
- Change to analysis directory
- Run R targets pipeline
- Capture exit code

**Cleanup:**
- Archive temporary targets cache
- Generate manifest of outputs

---

## Error Handling and Recovery

### Training Failures (wild_west)

**Scenario:** Training script crashes before completion

**Detection:**
- No .training_complete signal file
- Incomplete checkpoint directories
- Error in training logs

**Recovery:**
- Manual: User inspects logs and restarts
- Automatic: None (training is manual)

**Prevention:**
- Checkpoint frequently
- Log errors comprehensively
- Validate configs before training

---

### Watcher Daemon Failures (SSRDE)

**Scenario:** Watcher daemon crashes or stops

**Detection:**
- Process not running
- No recent log entries
- Systemd/supervisor health check

**Recovery:**
- Automatic restart via systemd
- On restart, scan for missed signals
- Process any pending .training_complete files

**Prevention:**
- Robust error handling
- Graceful degradation
- Resource limits (prevent OOM)

---

### Slurm Job Failures (SSRDE)

**Scenario:** Evaluation or analysis job fails

**Detection:**
- Non-zero exit code
- Slurm job state = FAILED
- No .eval_complete or .analysis_complete signal

**Recovery:**
- Check evaluation.db for status
- Inspect job logs
- Retry job with adjusted resources if needed

**Prevention:**
- Resource requests with headroom
- Input validation before job start
- Checkpoint intermediate results

---

### Network Drive Issues

**Scenario:** Network filesystem becomes unavailable

**Detection:**
- I/O errors on file operations
- Stale file handles
- Timeouts on reads/writes

**Recovery:**
- Retry with exponential backoff
- Fall back to local disk for temporary files
- Alert if prolonged outage

**Prevention:**
- Graceful handling of I/O errors
- Avoid holding files open unnecessarily
- Use local disk for temporary computation

---

### Database Corruption

**Scenario:** SQLite database becomes corrupted

**Detection:**
- SQLite integrity check fails
- Cannot open database
- Corrupt data returned

**Recovery:**
- Restore from backup
- Rebuild from signal files and logs
- Manual intervention required

**Prevention:**
- Regular automated backups
- WAL mode (reduces corruption risk)
- Don't kill processes mid-transaction

---

## Monitoring and Observability

### Metrics to Track

**Training Metrics (wild_west → MLflow):**
- Loss per epoch
- Learning rate
- Gradient norms
- Training throughput (samples/sec)
- GPU utilization
- Checkpoint sizes

**Evaluation Metrics (SSRDE → MLflow):**
- Accuracy per dataset
- Perplexity
- F1 scores
- Evaluation duration
- Dataset coverage

**System Metrics (all servers):**
- Disk usage on shared drive
- Network I/O to shared drive
- Slurm queue depth
- Job wait times
- Job failure rates

**Pipeline Metrics:**
- Training → evaluation latency
- Evaluation → analysis latency
- End-to-end pipeline duration
- Signal detection latency

---

### Logging Standards

**All Log Files:**
- Timestamps in ISO 8601 format
- Structured logging (JSON where possible)
- Log levels (DEBUG, INFO, WARN, ERROR)
- Context: experiment name, job ID, hostname

**Log Locations:**
- Training: `/mnt/shared/subject-drop/logs/training/{experiment_name}.log`
- Evaluation: `/mnt/shared/subject-drop/logs/evaluation/{slurm_job_id}.log`
- Analysis: `/mnt/shared/subject-drop/logs/analysis/{slurm_job_id}.log`
- Watcher: `/mnt/shared/subject-drop/logs/watcher.log`

**Log Rotation:**
- Rotate daily or at 100MB
- Keep 30 days of logs
- Compress old logs

---

### Alerting

**Critical Alerts:**
- Watcher daemon down for >5 minutes
- Repeated Slurm job failures (>3 in a row)
- Disk space <10% on shared drive
- Database corruption detected

**Warning Alerts:**
- Slurm queue depth >50 jobs
- Job wait time >24 hours
- Evaluation taking >2x expected time
- Missing expected signal files

**Alert Channels:**
- Email notifications
- Dashboard warnings
- Optional: Slack/Discord webhooks

---

## Security Considerations

### Access Control

**Shared Drive:**
- User/group permissions on directories
- Training outputs: writable by wild_west user
- Evaluation outputs: writable by SSRDE user
- All readable by monitoring server

**MLflow:**
- Basic authentication (username/password)
- Or reverse proxy (nginx) with OAuth
- HTTPS required for internet access

**Dashboard:**
- Same authentication as MLflow
- Read-only access to databases
- No write operations exposed

---

### Data Protection

**Sensitive Data:**
- No credentials in signal files
- No PII in logs
- Sanitize error messages

**Network Security:**
- Firewall rules for monitoring server
- SSH key-based auth for watcher → SSRDE
- TLS for MLflow if internet-facing

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Objectives:**
- Set up directory structure on shared drive
- Create SQLite database schemas
- Implement training completion hook

**Deliverables:**
- Directory structure created
- Empty databases initialized
- Training hook integrated and tested

**Testing:**
- Run training, verify signal file created
- Verify database updated
- Verify MLflow metrics logged

---

### Phase 2: Automation Core (Week 2)

**Objectives:**
- Implement watcher daemon
- Create Slurm job templates
- Test evaluation job submission

**Deliverables:**
- Watcher daemon code
- Slurm templates for evaluation
- Systemd service file for watcher

**Testing:**
- Trigger training completion, verify eval job submitted
- Verify Slurm job runs successfully
- Verify .eval_complete signal created

---

### Phase 3: Analysis Pipeline (Week 3)

**Objectives:**
- Set up R targets pipeline
- Create Slurm job template for analysis
- Test end-to-end pipeline

**Deliverables:**
- R targets configuration
- Analysis Slurm template
- Complete pipeline test

**Testing:**
- Run full pipeline: training → eval → analysis
- Verify all outputs generated
- Verify all databases updated

---

### Phase 4: Monitoring (Week 4)

**Objectives:**
- Deploy MLflow server
- Build job monitoring dashboard
- Configure alerts

**Deliverables:**
- MLflow running on monitoring server
- Dashboard accessible via web
- Alert system configured

**Testing:**
- Access MLflow UI from internet
- Verify dashboard shows current jobs
- Trigger test alert

---

### Phase 5: Hardening (Week 5)

**Objectives:**
- Add error handling and retries
- Implement backup strategy
- Document operations

**Deliverables:**
- Comprehensive error handling
- Automated backups
- Operations runbook

**Testing:**
- Fault injection testing
- Disaster recovery drill
- Performance testing under load

---

## Operational Procedures

### Starting a Training Run

**On wild_west:**

1. Prepare training config in `config/experiments/{experiment_name}.yaml`
2. Ensure GPU resources available (check with other users)
3. Run training script: `python train.py --config {experiment_name}.yaml`
4. Monitor via MLflow UI
5. Training hook automatically triggers downstream pipeline

**No manual intervention needed after training starts.**

---

### Monitoring Job Progress

**Via MLflow UI:**
- Navigate to `http://monitoring-server:5000`
- View experiment runs
- Inspect metrics, logs, artifacts

**Via Dashboard:**
- Navigate to `http://monitoring-server:8000`
- View current Slurm queue
- Check job statuses

**Via Command Line:**
- SSH to SSRDE: `ssh ssrde`
- Check queue: `squeue -u $USER`
- View logs: `tail -f /mnt/shared/subject-drop/logs/evaluation/{job_id}.log`

---

### Handling Failed Jobs

**Evaluation Job Failed:**

1. Check job log: `logs/evaluation/{job_id}.log`
2. Identify error (resource limit, code bug, data issue)
3. Fix underlying issue
4. Manually resubmit or trigger via watcher
5. Update evaluation.db if needed

**Analysis Job Failed:**

1. Check job log: `logs/analysis/{job_id}.log`
2. Check R targets cache for partial results
3. Fix error in R pipeline
4. Rerun: `Rscript -e "targets::tar_make()"`
5. Slurm resubmission if needed

---

### Database Maintenance

**Weekly Tasks:**
- Run `VACUUM` on SQLite databases
- Verify database integrity
- Check disk space usage

**Monthly Tasks:**
- Archive old completed jobs (>90 days)
- Rotate logs
- Review performance metrics

---

### Backup Strategy

**What to Backup:**
- SQLite databases (state/)
- MLflow backend database
- Configurations (config/)
- Critical logs

**Backup Frequency:**
- Databases: Daily
- Logs: Weekly
- Configs: On change (git)

**Backup Location:**
- Separate network location
- Not on same shared drive

**Restore Testing:**
- Monthly restore drill
- Document restore procedure

---

## Performance Optimization

### Filesystem I/O

**Strategies:**
- Batch file writes
- Use local disk for temporary files
- Avoid unnecessary stat() calls
- Stream large files (don't read entirely into memory)

**Monitoring:**
- Track I/O wait times
- Monitor network filesystem latency
- Alert on slow I/O

---

### Slurm Efficiency

**Strategies:**
- Right-size resource requests
- Use array jobs for batch evaluation
- Pre-empt low-priority jobs if needed
- Monitor queue depth

**Monitoring:**
- Track job wait times
- Measure queue utilization
- Identify resource bottlenecks

---

### Database Performance

**Strategies:**
- Use WAL mode
- Index frequently queried columns
- Periodic VACUUM
- Connection pooling if needed

**Monitoring:**
- Query execution times
- Database size growth
- Lock contention

---

## Future Enhancements

### Potential Additions

**Redis Streams (if needed):**
- For complex event routing on SSRDE
- Better separation of watcher and job submitter
- Multiple consumer groups

**Notification System:**
- Email on training completion
- Slack/Discord integration
- SMS for critical failures

**Web API:**
- REST API for job submission
- Programmatic access to status
- Webhook support for external tools

**Advanced Analysis:**
- Automated model comparison
- Statistical significance testing
- Automated report generation

**Containerization:**
- Docker containers for reproducibility
- Singularity on Slurm for consistency

---

## Appendix A: Configuration Files

### Environment Variables

**All Servers:**
```bash
export SHARED_DRIVE="/mnt/shared/subject-drop"
export MLFLOW_TRACKING_URI="http://monitoring-server:5000"
```

**SSRDE Only:**
```bash
export WATCHER_CONFIG="/mnt/shared/subject-drop/config/watcher.yaml"
export SLURM_ACCOUNT="your_account"
```

---

### watcher.yaml (SSRDE)

**Purpose:** Configuration for watcher daemon

**Contents:**
- Paths to watch
- Slurm submission parameters
- Retry policies
- Alert configuration
- Database connection strings

---

## Appendix B: Troubleshooting Guide

### Problem: Signal file not detected

**Possible Causes:**
- Watcher daemon not running
- Filesystem event not fired
- Signal file in wrong location

**Diagnosis:**
1. Check watcher process: `ps aux | grep watcher`
2. Check watcher log for errors
3. Verify signal file exists and is readable

**Solution:**
- Restart watcher if stopped
- Manually trigger job submission
- Check filesystem mount

---

### Problem: Slurm job stuck in pending

**Possible Causes:**
- Resource constraints
- Account quota exceeded
- Dependency not satisfied

**Diagnosis:**
1. Check job status: `squeue -j {job_id}`
2. Check reason: `scontrol show job {job_id}`
3. Check account limits: `sacctmgr show assoc where user=$USER`

**Solution:**
- Adjust resource requests
- Wait for resources to free
- Contact SSRDE admin if quota issue

---

### Problem: MLflow not logging metrics

**Possible Causes:**
- MLflow server down
- Network connectivity issue
- Incorrect tracking URI

**Diagnosis:**
1. Check MLflow server: `curl http://monitoring-server:5000`
2. Check network: `ping monitoring-server`
3. Verify MLFLOW_TRACKING_URI

**Solution:**
- Restart MLflow server
- Fix network routing
- Correct environment variable

---

## Appendix C: Glossary

**Signal File:** JSON file used to trigger downstream processes (e.g., .training_complete)

**Watcher Daemon:** Long-running process that monitors filesystem for signal files

**Slurm:** Job scheduler and workload manager for compute clusters

**MLflow:** Platform for ML lifecycle management (tracking, models, deployment)

**R targets:** Pipeline toolkit for R that ensures reproducibility

**Parquet:** Columnar storage format optimized for analytics

**WAL Mode:** Write-Ahead Logging mode for SQLite (better concurrency)

**SSRDE:** Shared compute server with Slurm scheduling

**wild_west:** GPU server for manual training workloads

---

## Document History

**v1.0 (2025-09-30):** Initial specification for implementation

---

## Contact and Support

**For Questions:**
- Review this specification document
- Check logs in `/mnt/shared/subject-drop/logs/`
- Query SQLite databases for state information
- Inspect MLflow UI for run details

**For Issues:**
- File bug reports with logs attached
- Include experiment name and relevant timestamps
- Provide steps to reproduce

---

_End of Specification_
