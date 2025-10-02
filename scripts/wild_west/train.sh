#!/usr/bin/env bash
#
# Wild-West Training Script
# Simple, safe training on shared GPU servers without SLURM
#
# Usage:
#   ./scripts/wild_west/train.sh [OPTIONS] <config_path>
#
# Options:
#   --gpus <ids>        Comma-separated GPU IDs (overrides CUDA_VISIBLE_DEVICES)
#   --lock-gpus         Lock GPUs before training
#   --check-gpus        Check GPU availability before starting
#   --help              Show this help message
#
# Examples:
#   ./scripts/wild_west/train.sh configs/gpt2_small.yaml
#   ./scripts/wild_west/train.sh --lock-gpus configs/experiment_0_baseline.yaml
#   ./scripts/wild_west/train.sh --gpus 2,3 --lock-gpus configs/bert_base.yaml
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_DIR="/tmp/gpu_locks"
LOG_DIR="$PROJECT_DIR/logs"
TIMEOUT="72h"  # Maximum training time

# Defaults
GPUS="${CUDA_VISIBLE_DEVICES:-}"
LOCK_GPUS=false
CHECK_GPUS=false
CONFIG_PATH=""

# ============================================================================
# Functions
# ============================================================================

log() {
    printf "[%(%F %T)T] %s\n" -1 "$*" | tee -a "$LOG_FILE"
}

show_help() {
    head -n 20 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

check_gpu_available() {
    local gpu_id=$1
    local mem_free

    mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")

    if [ "$mem_free" -lt 10000 ]; then
        echo "ERROR: GPU $gpu_id has <10GB free ($mem_free MB)"
        return 1
    fi

    return 0
}

lock_gpu() {
    local gpu_id=$1
    local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"

    if [ -e "$lock_file" ]; then
        echo "ERROR: GPU $gpu_id is already locked:"
        cat "$lock_file"
        return 1
    fi

    mkdir -p "$LOCK_DIR"
    {
        echo "user=$(whoami)"
        echo "host=$(hostname)"
        echo "time=$(date -Iseconds)"
        echo "pid=$$"
        echo "config=$CONFIG_PATH"
        echo "gpus=$GPUS"
    } > "$lock_file"

    log "Locked GPU $gpu_id"
}

unlock_gpu() {
    local gpu_id=$1
    local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"
    rm -f "$lock_file"
    log "Unlocked GPU $gpu_id"
}

cleanup() {
    local exit_code=$?
    trap - INT TERM EXIT

    # Kill entire process group
    if [ -n "${PGID:-}" ]; then
        log "Cleaning up process group $PGID"
        kill -TERM -"$PGID" 2>/dev/null || true
        sleep 2
        kill -KILL -"$PGID" 2>/dev/null || true
    fi

    # Unlock GPUs
    if [ "$LOCK_GPUS" = true ] && [ -n "$GPUS" ]; then
        IFS=',' read -ra GPU_ARR <<< "$GPUS"
        for gpu in "${GPU_ARR[@]}"; do
            unlock_gpu "$gpu"
        done
    fi

    log "Training ended with exit code $exit_code"
    exit "$exit_code"
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --lock-gpus)
            LOCK_GPUS=true
            shift
            ;;
        --check-gpus)
            CHECK_GPUS=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            show_help
            ;;
        *)
            CONFIG_PATH="$1"
            shift
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: Config path required"
    show_help
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Get experiment name from config
EXP_NAME=$(basename "$CONFIG_PATH" .yaml)
LOG_FILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

log "=========================================="
log "Model Foundry Training - Wild-West"
log "=========================================="
log "Config: $CONFIG_PATH"
log "Experiment: $EXP_NAME"
log "User: $(whoami)"
log "Host: $(hostname)"
log "Project: $PROJECT_DIR"

# ============================================================================
# GPU Management
# ============================================================================

if [ -z "$GPUS" ]; then
    # Try to auto-detect available GPU
    log "No GPUs specified, checking availability..."
    GPUS=$("$PROJECT_DIR/scripts/wild_west/gpu_monitor.sh" available | head -1 || echo "")

    if [ -z "$GPUS" ]; then
        echo "ERROR: No GPUs available and none specified"
        exit 1
    fi

    log "Auto-selected GPU: $GPUS"
fi

log "Using GPUs: $GPUS"

# Check GPU availability
if [ "$CHECK_GPUS" = true ]; then
    log "Checking GPU availability..."
    IFS=',' read -ra GPU_ARR <<< "$GPUS"
    for gpu in "${GPU_ARR[@]}"; do
        if ! check_gpu_available "$gpu"; then
            exit 1
        fi
        log "GPU $gpu is available"
    done
fi

# Lock GPUs
if [ "$LOCK_GPUS" = true ]; then
    log "Locking GPUs..."
    IFS=',' read -ra GPU_ARR <<< "$GPUS"
    for gpu in "${GPU_ARR[@]}"; do
        if ! lock_gpu "$gpu"; then
            # Unlock any we already locked
            for locked_gpu in "${GPU_ARR[@]}"; do
                [ "$locked_gpu" = "$gpu" ] && break
                unlock_gpu "$locked_gpu"
            done
            exit 1
        fi
    done
fi

# Set trap for cleanup
trap cleanup INT TERM EXIT

# ============================================================================
# Environment Setup
# ============================================================================

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_CUDA_MEMORY_FRACTION="0.95"
export CUDA_LAUNCH_BLOCKING="0"
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"

log "Environment:"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
log "  OMP_NUM_THREADS=$OMP_NUM_THREADS"

# ============================================================================
# Training Execution
# ============================================================================

log "Starting training..."
log "Command: python -m model_foundry.train $CONFIG_PATH"

cd "$PROJECT_DIR"

# Launch in its own session for proper cleanup
set +e
setsid python -m model_foundry.train "$CONFIG_PATH" &
CHILD=$!
PGID=$(ps -o pgid= "$CHILD" | tr -d ' ')
log "Child PID: $CHILD, PGID: $PGID"

# Wait with timeout
timeout --preserve-status --signal=TERM --kill-after=30s "$TIMEOUT" \
    bash -c "wait $CHILD"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 124 ]; then
    log "ERROR: Training timed out after $TIMEOUT"
elif [ $EXIT_CODE -eq 0 ]; then
    log "Training completed successfully"
else
    log "ERROR: Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
