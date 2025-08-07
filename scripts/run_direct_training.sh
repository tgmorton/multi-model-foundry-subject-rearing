#!/bin/bash

# --- Direct Execution Script for Model Training ---
# This script runs the model training directly on the cluster GPU.

# Exit on any error
set -e

# Function to cleanup on exit
cleanup() {
    echo "=== Cleaning up on exit ==="
    
    # Get the current process ID and its children
    SCRIPT_PID=$$
    
    # Kill any remaining Python processes that might be holding GPU memory
    echo "  - Terminating Python processes..."
    pkill -f "python.*model_foundry" || true
    pkill -f "python.*trainer" || true
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Force kill any remaining processes if needed
    pkill -9 -f "python.*model_foundry" || true
    pkill -9 -f "python.*trainer" || true
    
    # Clear CUDA cache if nvidia-smi is available
    echo "  - Clearing GPU memory..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset || true
        # Also try to clear cache
        python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    fi
    
    echo "  - Cleanup completed"
}

# Set up trap to call cleanup on script exit
trap cleanup EXIT INT TERM

# Function to monitor GPU usage
monitor_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "=== GPU Status ==="
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        echo "================="
    fi
}

# --- Script Usage ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_name_without_yaml_extension> [--resume]"
    echo "Example: ./run_direct_training.sh experiment_1_remove_expletives"
    echo "Example with resume: ./run_direct_training.sh experiment_1_remove_expletives --resume"
    exit 1
fi

CONFIG_NAME="$1"
RESUME_FLAG=""

# Check for resume flag
if [ "$2" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi

# === Environment Setup ===
echo "=== Training Script Started: $(date) ==="
echo "Config: ${CONFIG_NAME}"
if [ -n "$RESUME_FLAG" ]; then
    echo "Resume mode: enabled"
fi

module load singularity/4.1.1 || echo "Warning: singularity/4.1.1 module not found. Ensure Singularity is in your PATH."
module load cuda/11.8 || echo "Warning: cuda/11.8 module not found. Ensure CUDA is correctly configured."

# --- Define Host and Container Paths ---
# !!! UPDATE THIS TO YOUR PROJECT'S ROOT DIRECTORY !!!
HOST_PROJECT_DIR="/labs/ferreiralab/thmorton/subject-drop-rearing"
HOST_TRAINING_SIF_PATH="${HOST_PROJECT_DIR}/singularity/training.sif"

# Construct the full path to the config file from the name
HOST_CONFIG_FILE="${HOST_PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

# --- Preparations ---
echo "Project Directory: ${HOST_PROJECT_DIR}"
echo "Training SIF Path: ${HOST_TRAINING_SIF_PATH}"

# Check for required files
if [ ! -f "$HOST_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $HOST_CONFIG_FILE"
    exit 1
fi
if [ ! -f "$HOST_TRAINING_SIF_PATH" ]; then
    echo "ERROR: Training singularity image not found at $HOST_TRAINING_SIF_PATH"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${HOST_PROJECT_DIR}/logs"

# === Training Execution ===
echo "Starting model training inside Singularity container..."
echo "Using config file: ${CONTAINER_CONFIG_FILE}"

# Show initial GPU status
monitor_gpu

# Set PyTorch CUDA Allocator Config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the training script inside the container with timeout
echo "Starting training with automatic cleanup on exit..."
timeout 24h singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_TRAINING_SIF_PATH}" \
    bash -c "cd /workspace && python -m model_foundry.cli run ${CONTAINER_CONFIG_FILE} ${RESUME_FLAG}"

# Check if timeout occurred
if [ $? -eq 124 ]; then
    echo "⚠️  Training was terminated due to timeout (24 hours)"
    cleanup
fi

# Show final GPU status
echo "=== Final GPU Status ==="
monitor_gpu

# === Script Completion ===
echo "=== Training Script Finished: $(date) ===" 