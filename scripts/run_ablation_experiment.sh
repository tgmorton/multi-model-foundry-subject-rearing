#!/bin/bash
#SBATCH --job-name=subject-drop-ablative    # Descriptive job name
#SBATCH --partition=general_gpu_p6000       # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-0:00:00                    # Time limit (D-HH:MM:SS)
#SBATCH --output=../logs/%x-%j.out          # Standard output log
#SBATCH --error=../logs/%x-%j.err           # Standard error log

# Exit on any error
set -e

# --- Configuration ---
# The script takes a config NAME and a PHASE
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: sbatch $0 <config_name_without_yaml_extension> <phase>"
    echo "Phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
    echo "Example: sbatch $0 experiment_1_remove_expletives preprocess"
    exit 1
fi
CONFIG_NAME="$1"
PHASE="$2"

# --- Environment Setup ---
echo "========================================================"
echo "Job Started: $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_NAME"
echo "Phase: $PHASE"
echo "========================================================"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8

# --- Define Paths ---
# !!! IMPORTANT: UPDATE THESE PATHS TO MATCH YOUR ENVIRONMENT !!!
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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    
    # Build setup commands
    local setup_commands="cd /workspace"
    
    # Add flash-attention installation for training commands
    if [[ "$command" == *"run"* ]] || [[ "$description" == *"training"* ]]; then
        echo "Installing flash-attention for training..."
        setup_commands="$setup_commands && pip install flash-attn --no-build-isolation --quiet"
    fi
    
    # Execute the command inside the container
    srun singularity exec --nv \
        --bind "${HOST_PROJECT_DIR}":/workspace \
        "$container_path" \
        bash -c "$setup_commands && $command"
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
        echo "Running complete experimental pipeline..."
        
        # Step 1: Preprocessing
        echo "Step 1/4: Preprocessing..."
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli preprocess ${CONTAINER_CONFIG_FILE}" \
            "Dataset preprocessing and ablation"
        
        # Step 2: Train tokenizer
        echo "Step 2/4: Training tokenizer..."
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli train-tokenizer ${CONTAINER_CONFIG_FILE}" \
            "SentencePiece tokenizer training"
        
        # Step 3: Tokenize dataset
        echo "Step 3/4: Tokenizing dataset..."
        run_in_container "$HOST_ABLATION_SIF_PATH" \
            "python -m model_foundry.cli tokenize-dataset ${CONTAINER_CONFIG_FILE}" \
            "Dataset tokenization"
        
        # Step 4: Train model
        echo "Step 4/4: Training model..."
        run_in_container "$HOST_TRAINING_SIF_PATH" \
            "python -m model_foundry.cli run ${CONTAINER_CONFIG_FILE}" \
            "Model training"
        ;;
    
    *)
        echo "ERROR: Invalid phase '$PHASE'"
        echo "Valid phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
        exit 1
        ;;
esac

echo "========================================================"
echo "Job Finished: $(date)"
echo "========================================================" 