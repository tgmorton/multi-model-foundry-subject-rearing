#!/bin/bash

# --- Direct Execution Script for Tokenizer Operations ---
# This script runs tokenizer training and dataset tokenization directly on the cluster GPU.

# Exit on any error
set -e

# --- Script Usage ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_name_without_yaml_extension> <operation>"
    echo "Operations: train-tokenizer, tokenize-dataset, both"
    echo "Example: ./run_direct_tokenizer.sh experiment_1_remove_expletives train-tokenizer"
    echo "Example: ./run_direct_tokenizer.sh experiment_1_remove_expletives tokenize-dataset"
    echo "Example: ./run_direct_tokenizer.sh experiment_1_remove_expletives both"
    exit 1
fi

CONFIG_NAME="$1"
OPERATION="$2"

# === Environment Setup ===
echo "=== Tokenizer Script Started: $(date) ==="
echo "Config: ${CONFIG_NAME}"
echo "Operation: ${OPERATION}"

module load singularity/4.1.1 || echo "Warning: singularity/4.1.1 module not found. Ensure Singularity is in your PATH."
module load cuda/11.8 || echo "Warning: cuda/11.8 module not found. Ensure CUDA is correctly configured."

# --- Define Host and Container Paths ---
# !!! UPDATE THIS TO YOUR PROJECT'S ROOT DIRECTORY !!!
HOST_PROJECT_DIR="/labs/ferreiralab/thmorton/subject-drop-rearing"
HOST_ABLATION_SIF_PATH="${HOST_PROJECT_DIR}/singularity/ablation.sif"

# Construct the full path to the config file from the name
HOST_CONFIG_FILE="${HOST_PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

# --- Preparations ---
echo "Project Directory: ${HOST_PROJECT_DIR}"
echo "Ablation SIF Path: ${HOST_ABLATION_SIF_PATH}"

# Check for required files
if [ ! -f "$HOST_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $HOST_CONFIG_FILE"
    exit 1
fi
if [ ! -f "$HOST_ABLATION_SIF_PATH" ]; then
    echo "ERROR: Ablation singularity image not found at $HOST_ABLATION_SIF_PATH"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${HOST_PROJECT_DIR}/logs"

# Set PyTorch CUDA Allocator Config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# === Execute Operations ===
case $OPERATION in
    "train-tokenizer")
        echo "========================================================"
        echo "Training SentencePiece Tokenizer"
        echo "========================================================"
        singularity exec --nv \
            --bind "${HOST_PROJECT_DIR}":/workspace \
            "${HOST_ABLATION_SIF_PATH}" \
            bash -c "cd /workspace && python -m model_foundry.cli train-tokenizer ${CONTAINER_CONFIG_FILE}"
        ;;
    
    "tokenize-dataset")
        echo "========================================================"
        echo "Tokenizing Dataset"
        echo "========================================================"
        singularity exec --nv \
            --bind "${HOST_PROJECT_DIR}":/workspace \
            "${HOST_ABLATION_SIF_PATH}" \
            bash -c "cd /workspace && python -m model_foundry.cli tokenize-dataset ${CONTAINER_CONFIG_FILE}"
        ;;
    
    "both")
        echo "========================================================"
        echo "Step 1/2: Training SentencePiece Tokenizer"
        echo "========================================================"
        singularity exec --nv \
            --bind "${HOST_PROJECT_DIR}":/workspace \
            "${HOST_ABLATION_SIF_PATH}" \
            bash -c "cd /workspace && python -m model_foundry.cli train-tokenizer ${CONTAINER_CONFIG_FILE}"
        
        echo "========================================================"
        echo "Step 2/2: Tokenizing Dataset"
        echo "========================================================"
        singularity exec --nv \
            --bind "${HOST_PROJECT_DIR}":/workspace \
            "${HOST_ABLATION_SIF_PATH}" \
            bash -c "cd /workspace && python -m model_foundry.cli tokenize-dataset ${CONTAINER_CONFIG_FILE}"
        ;;
    
    *)
        echo "ERROR: Invalid operation '$OPERATION'"
        echo "Valid operations: train-tokenizer, tokenize-dataset, both"
        exit 1
        ;;
esac

# === Script Completion ===
echo "=== Tokenizer Script Finished: $(date) ===" 