#!/bin/bash

# --- Direct Execution Script for Individual Phases ---
# This script runs individual phases of the experimental pipeline
# directly on the TitanX GPU without SLURM.

# Exit on any error
set -e

# --- Script Usage ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_name_without_yaml_extension> <phase> [--resume]"
    echo "Phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
    echo "Example: ./run_direct_phase.sh experiment_1_remove_expletives preprocess"
    echo "Example with resume: ./run_direct_phase.sh experiment_2_baseline run --resume"
    exit 1
fi

CONFIG_NAME="$1"
PHASE="$2"
RESUME_FLAG=""

# Check for resume flag
if [ "$3" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi

# === Environment Setup ===
echo "=== Direct Phase Script Started: $(date) ==="
echo "Config: ${CONFIG_NAME}"
echo "Phase: ${PHASE}"
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

# --- Function to run command in container ---
run_in_container() {
    local container_path="$1"
    local command="$2"
    local description="$3"
    
    echo "========================================================"
    echo "Running: $description"
    echo "Container: $container_path"
    echo "Command: $command"
    echo "========================================================"
    
    if [ ! -f "$container_path" ]; then
        echo "ERROR: Container not found at $container_path"
        exit 1
    fi
    
    # Execute the command inside the container
    singularity exec --nv \
        --bind "${HOST_PROJECT_DIR}":/workspace \
        "$container_path" \
        bash -c "cd /workspace && $command"
}

# --- Phase-specific execution ---
case $PHASE in
    "preprocess")
        echo "Running preprocessing phase..."
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli preprocess ${CONTAINER_CONFIG_FILE}" \
            "Dataset preprocessing and ablation"
        ;;
    "train-tokenizer")
        echo "Running tokenizer training phase..."
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli train-tokenizer ${CONTAINER_CONFIG_FILE}" \
            "SentencePiece tokenizer training"
        ;;
    "tokenize-dataset")
        echo "Running dataset tokenization phase..."
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli tokenize-dataset ${CONTAINER_CONFIG_FILE}" \
            "Dataset tokenization"
        ;;
    "run")
        echo "Running model training phase..."
        run_in_container "$HOST_TRAINING_SIF_PATH" \
            "python -m model_foundry.cli run ${CONTAINER_CONFIG_FILE} ${RESUME_FLAG}" \
            "Model training"
        ;;
    "full-pipeline")
        echo "Running full pipeline..."
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli preprocess ${CONTAINER_CONFIG_FILE}" \
            "Dataset preprocessing and ablation"
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli train-tokenizer ${CONTAINER_CONFIG_FILE}" \
            "SentencePiece tokenizer training"
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli tokenize-dataset ${CONTAINER_CONFIG_FILE}" \
            "Dataset tokenization"
        run_in_container "$HOST_TRAINING_SIF_PATH" \
            "python -m model_foundry.cli run ${CONTAINER_CONFIG_FILE}" \
            "Model training"
        ;;
    *)
        echo "ERROR: Unknown phase '$PHASE'"
        echo "Valid phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
        exit 1
        ;;
esac

echo "========================================================"
echo "Phase Completed: $(date)"
echo "Config: $CONFIG_NAME"
echo "Phase: $PHASE"
echo "Hardware: TitanX (Headnode)"
echo "========================================================" 