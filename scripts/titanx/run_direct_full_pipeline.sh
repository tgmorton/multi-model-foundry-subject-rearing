#!/bin/bash

# --- Direct Execution Script for Complete Experimental Pipeline ---
# This script runs the full pipeline: preprocess → train-tokenizer → tokenize-dataset → run
# directly on the TitanX GPU without SLURM.

# Exit on any error
set -e

# --- Script Usage ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_name_without_yaml_extension> [--resume]"
    echo "Example: ./run_direct_full_pipeline.sh experiment_1_remove_expletives"
    echo "Example with resume: ./run_direct_full_pipeline.sh experiment_1_remove_expletives --resume"
    exit 1
fi

CONFIG_NAME="$1"
RESUME_FLAG=""

# Check for resume flag
if [ "$2" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi

# === Environment Setup ===
echo "=== Full Pipeline Script Started: $(date) ==="
echo "Config: ${CONFIG_NAME}"
echo "Hardware: TitanX (Headnode)"
if [ -n "$RESUME_FLAG" ]; then
    echo "Resume mode: enabled"
fi

module load singularity/4.1.1 || echo "Warning: singularity/4.1.1 module not found. Ensure Singularity is in your PATH."
module load cuda/11.8 || echo "Warning: cuda/11.8 module not found. Ensure CUDA is correctly configured."

# --- Define Host and Container Paths ---
# !!! UPDATE THIS TO YOUR PROJECT'S ROOT DIRECTORY !!!
HOST_PROJECT_DIR="/labs/ferreiralab/thmorton/subject-drop-rearing"
HOST_ABLATION_SIF_PATH="${HOST_PROJECT_DIR}/singularity/ablation.sif"
HOST_TRAINING_SIF_PATH="${HOST_PROJECT_DIR}/singularity/training.sif"

# Construct the full path to the config file from the name
HOST_CONFIG_FILE="${HOST_PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

# --- Preparations ---
echo "Project Directory: ${HOST_PROJECT_DIR}"
echo "Ablation SIF Path: ${HOST_ABLATION_SIF_PATH}"
echo "Training SIF Path: ${HOST_TRAINING_SIF_PATH}"

# Check for required files
if [ ! -f "$HOST_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $HOST_CONFIG_FILE"
    exit 1
fi
if [ ! -f "$HOST_ABLATION_SIF_PATH" ]; then
    echo "ERROR: Ablation singularity image not found at $HOST_ABLATION_SIF_PATH"
    exit 1
fi
if [ ! -f "$HOST_TRAINING_SIF_PATH" ]; then
    echo "ERROR: Training singularity image not found at $HOST_TRAINING_SIF_PATH"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${HOST_PROJECT_DIR}/logs"

# Set PyTorch CUDA Allocator Config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# === Step 1: Preprocessing ===
echo "========================================================"
echo "Step 1/4: Preprocessing (Expletive Removal)"
echo "========================================================"
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_ABLATION_SIF_PATH}" \
    bash -c "cd /workspace && python -m model_foundry.cli preprocess ${CONTAINER_CONFIG_FILE}"

# === Step 2: Train Tokenizer ===
echo "========================================================"
echo "Step 2/4: Training Tokenizer"
echo "========================================================"
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_ABLATION_SIF_PATH}" \
    bash -c "cd /workspace && python -m model_foundry.cli train-tokenizer ${CONTAINER_CONFIG_FILE}"

# === Step 3: Tokenize Dataset ===
echo "========================================================"
echo "Step 3/4: Tokenizing Dataset"
echo "========================================================"
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_ABLATION_SIF_PATH}" \
    bash -c "cd /workspace && python -m model_foundry.cli tokenize-dataset ${CONTAINER_CONFIG_FILE}"

# === Step 4: Train Model ===
echo "========================================================"
echo "Step 4/4: Training Model"
echo "========================================================"
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_TRAINING_SIF_PATH}" \
    bash -c "cd /workspace && python -m model_foundry.cli run ${CONTAINER_CONFIG_FILE}"

echo "========================================================"
echo "Full Pipeline Completed: $(date)"
echo "Config: ${CONFIG_NAME}"
echo "Hardware: TitanX (Headnode)"
echo "========================================================" 