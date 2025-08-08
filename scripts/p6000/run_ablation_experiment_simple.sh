#!/bin/bash
#SBATCH --job-name=subject-drop-ablative    # Descriptive job name
#SBATCH --gres=gpu:1       # P6000 GPU partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=0-1:00:00                    # Time limit (D-HH:MM:SS)
#SBATCH --output=../../logs/%x-%j.out          # Standard output log
#SBATCH --error=../../logs/%x-%j.err           # Standard error log

# Exit on any error
set -e

# --- Hardcoded Configuration ---
CONFIG_NAME="experiment_1_remove_expletives"
PHASE="preprocess"

# --- Environment Setup ---
echo "========================================================"
echo "Job Started: $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_NAME"
echo "Phase: $PHASE"
echo "Hardware: P6000"
echo "========================================================"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8

# --- Download spaCy model if not present ---
SPACY_MODEL_DIR="${HOST_PROJECT_DIR}/spacy_models"
EN_CORE_WEB_SM_DIR="${SPACY_MODEL_DIR}/en_core_web_sm"

echo "Checking for spaCy model..."
if [ ! -d "$EN_CORE_WEB_SM_DIR" ]; then
    echo "Downloading en_core_web_sm model..."
    mkdir -p "$SPACY_MODEL_DIR"
    
    # Create a temporary container to download the model
    singularity exec --bind "${HOST_PROJECT_DIR}":/workspace \
        "${HOST_ABLATION_SIF_PATH}" \
        bash -c "cd /workspace && python -m spacy download en_core_web_sm --target /workspace/spacy_models"
    
    echo "Model downloaded to: $EN_CORE_WEB_SM_DIR"
else
    echo "Model already exists at: $EN_CORE_WEB_SM_DIR"
fi

# --- Define Paths ---
HOST_PROJECT_DIR="/labs/ferreiralab/thmorton/subject-drop-rearing"
HOST_ABLATION_SIF_PATH="/labs/ferreiralab/thmorton/subject-drop-rearing/singularity/ablation.sif"
HOST_TRAINING_SIF_PATH="/labs/ferreiralab/thmorton/subject-drop-rearing/singularity/training.sif"

# Construct the full path to the config file from the name
HOST_CONFIG_FILE="${HOST_PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "Ablation SIF Path (Host): ${HOST_ABLATION_SIF_PATH}"
echo "Training SIF Path (Host): ${HOST_TRAINING_SIF_PATH}"

# Check for required files
if [ ! -f "$HOST_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $HOST_CONFIG_FILE"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${HOST_PROJECT_DIR}/logs"

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
    
    # Set PyTorch CUDA Allocator Config for better memory management
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Execute the command inside the container
    srun singularity exec --nv \
        --bind "${HOST_PROJECT_DIR}":/workspace \
        --bind "${SPACY_MODEL_DIR}":/workspace/spacy_models \
        "$container_path" \
        bash -c "cd /workspace && export SPACY_DATA_PATH=/workspace/spacy_models && $command"
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
            "python -m model_foundry.cli run ${CONTAINER_CONFIG_FILE}" \
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
echo "Job Completed: $(date)"
echo "Config: $CONFIG_NAME"
echo "Phase: $PHASE"
echo "Hardware: P6000"
echo "========================================================" 