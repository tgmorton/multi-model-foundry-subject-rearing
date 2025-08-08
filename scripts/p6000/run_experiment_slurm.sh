#!/bin/bash
#SBATCH --job-name=subject-drop-experiment
#SBATCH --partition=p6000
#SBATCH --gres=gpu:p6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# P6000 SLURM Experiment Runner
# Adapted from wild_west/run_experiment.sh with SLURM best practices
# Handles complete experiment pipelines with proper process management

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default values
DEFAULT_PHASE="full-pipeline"
DEFAULT_BATCH_SIZE=""
DEFAULT_NUM_EPOCHS=""

# Function to show usage
show_usage() {
    echo "Usage: sbatch $0 [OPTIONS] -- <config_name>"
    echo ""
    echo "SLURM Options (before --):"
    echo "  --job-name=<name>         Job name"
    echo "  --time=<time>             Time limit (default: 48:00:00)"
    echo "  --mem=<memory>            Memory allocation (default: 32G)"
    echo ""
    echo "Script Options (after --):"
    echo "  -p, --phase <phase>       Phase to run (default: $DEFAULT_PHASE)"
    echo "  -b, --batch-size <size>   Batch size override"
    echo "  -e, --epochs <num>        Number of epochs override"
    echo "  --wandb-mode <mode>       W&B mode: online|offline|disabled (default: disabled)"
    echo "  --wandb-api-key <key>     W&B API key"
    echo "  -v, --verbose             Verbose output"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
    echo ""
    echo "Examples:"
    echo "  sbatch --job-name=exp1 $0 -- experiment_1_remove_expletives"
    echo "  sbatch --time=24:00:00 $0 -- -p run experiment_0_baseline"
    echo "  sbatch $0 -- -p preprocess -b 64 experiment_2_baseline"
}

# Parse command line arguments (after --)
PHASE="$DEFAULT_PHASE"
BATCH_SIZE=""
NUM_EPOCHS=""
WANDB_MODE="disabled"
WANDB_API_KEY=""
VERBOSE=false
CONFIG_NAME=""

# Skip to arguments after --
while [[ $# -gt 0 ]] && [[ "$1" != "--" ]]; do
    shift
done
[[ "$1" == "--" ]] && shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--phase)
            PHASE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --wandb-mode)
            WANDB_MODE="$2"
            shift 2
            ;;
        --wandb-api-key)
            WANDB_API_KEY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            CONFIG_NAME="$1"
            shift
            ;;
    esac
done

# Check if config name is provided
if [ -z "$CONFIG_NAME" ]; then
    echo -e "${RED}ERROR: Config name is required${NC}"
    show_usage
    exit 1
fi

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${BLUE}[$timestamp] INFO: $message${NC}"
            ;;
        "WARN")
            echo -e "${YELLOW}[$timestamp] WARN: $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}"
            ;;
    esac
}

# Function to setup SLURM environment
setup_slurm_environment() {
    log "INFO" "Setting up SLURM environment..."
    
    # Log SLURM job information
    log "INFO" "Job ID: ${SLURM_JOB_ID:-N/A}"
    log "INFO" "Job Name: ${SLURM_JOB_NAME:-N/A}"
    log "INFO" "Node: ${SLURMD_NODENAME:-N/A}"
    log "INFO" "Partition: ${SLURM_JOB_PARTITION:-N/A}"
    log "INFO" "GPUs: ${CUDA_VISIBLE_DEVICES:-All available}"
    
    # Load necessary modules
    log "INFO" "Loading system modules..."
    module load singularity/4.1.1 cuda/11.8 2>/dev/null || log "WARN" "Could not load modules"
    
    # Set PyTorch CUDA allocator config for better memory management
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
    export TORCH_CUDA_MEMORY_FRACTION=0.95
    export CUDA_LAUNCH_BLOCKING=0
    
    # Set NCCL environment for better error handling
    export NCCL_DEBUG=WARN
    export NCCL_ASYNC_ERROR_HANDLING=1
    
    # Container runtime detection
    if command -v apptainer >/dev/null 2>&1; then
        CONTAINER_BIN="apptainer"
    elif command -v singularity >/dev/null 2>&1; then
        CONTAINER_BIN="singularity"
    else
        log "ERROR" "Neither apptainer nor singularity found in PATH."
        exit 1
    fi
    log "INFO" "Using container runtime: $CONTAINER_BIN"
}

# Function to build command with overrides
build_command() {
    local base_command="$1"
    local config_file="$2"
    
    local command="$base_command $config_file"
    
    # Add batch size override if specified
    if [ -n "$BATCH_SIZE" ]; then
        command="$command --batch-size $BATCH_SIZE"
    fi
    
    # Add epochs override if specified
    if [ -n "$NUM_EPOCHS" ]; then
        command="$command --epochs $NUM_EPOCHS"
    fi
    
    echo "$command"
}

# Function to run command in container with SLURM
run_in_container() {
    local container_path="$1"
    local command="$2"
    local description="$3"
    
    log "INFO" "Running: $description"
    log "INFO" "Container: $container_path"
    log "INFO" "Command: $command"
    
    if [ ! -f "$container_path" ]; then
        log "ERROR" "Container not found at $container_path"
        return 1
    fi
    
    # Set up wandb environment variables
    local env_flags=""
    case "$WANDB_MODE" in
        "disabled")
            env_flags="WANDB_MODE=disabled,WANDB_SILENT=true"
            ;;
        "offline")
            env_flags="WANDB_MODE=offline"
            ;;
        "online")
            env_flags="WANDB_MODE=online"
            ;;
    esac
    
    # Add API key if provided
    if [ -n "$WANDB_API_KEY" ]; then
        env_flags="${env_flags},WANDB_API_KEY=${WANDB_API_KEY}"
    fi
    
    # Execute with srun for proper SLURM integration
    srun --exclusive --ntasks=1 --gpus-per-task=1 \
        "$CONTAINER_BIN" exec --nv --pid --contain --cleanenv \
        --env "$env_flags" \
        --bind "${PROJECT_DIR}":/workspace \
        "$container_path" \
        bash -c "
            set -euo pipefail
            cd /workspace
            # Download spaCy model if needed (cached after first run)
            python -m spacy download en_core_web_sm --quiet 2>/dev/null || true
            
            # Execute the actual command
            exec $command
        "
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "ERROR" "Command failed with status $status"
        return $status
    fi
    return 0
}

# Main execution
main() {
    log "INFO" "========================================================"
    log "INFO" "Starting experiment: $CONFIG_NAME"
    log "INFO" "Phase: $PHASE"
    log "INFO" "Start time: $(date)"
    log "INFO" "========================================================"
    
    # Setup SLURM environment
    setup_slurm_environment
    
    # Define paths
    HOST_ABLATION_SIF_PATH="$PROJECT_DIR/singularity/ablation.sif"
    HOST_TRAINING_SIF_PATH="$PROJECT_DIR/singularity/training.sif"
    HOST_CONFIG_FILE="$PROJECT_DIR/configs/${CONFIG_NAME}.yaml"
    CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"
    
    # Check for required files
    if [ ! -f "$HOST_CONFIG_FILE" ]; then
        log "ERROR" "Config file not found at $HOST_CONFIG_FILE"
        exit 1
    fi
    
    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs"
    
    # Phase-specific execution
    case $PHASE in
        "preprocess")
            log "INFO" "Running preprocessing phase..."
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli preprocess" "$CONTAINER_CONFIG_FILE")" \
                "Dataset preprocessing and ablation"
            ;;
        "train-tokenizer")
            log "INFO" "Running tokenizer training phase..."
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli train-tokenizer" "$CONTAINER_CONFIG_FILE")" \
                "SentencePiece tokenizer training"
            ;;
        "tokenize-dataset")
            log "INFO" "Running dataset tokenization phase..."
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli tokenize-dataset" "$CONTAINER_CONFIG_FILE")" \
                "Dataset tokenization"
            ;;
        "run")
            log "INFO" "Running model training phase..."
            run_in_container "$HOST_TRAINING_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli run" "$CONTAINER_CONFIG_FILE")" \
                "Model training"
            ;;
        "full-pipeline")
            log "INFO" "Running full pipeline..."
            
            # Run each phase sequentially
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli preprocess" "$CONTAINER_CONFIG_FILE")" \
                "Dataset preprocessing and ablation" || exit 1
                
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli train-tokenizer" "$CONTAINER_CONFIG_FILE")" \
                "SentencePiece tokenizer training" || exit 1
                
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli tokenize-dataset" "$CONTAINER_CONFIG_FILE")" \
                "Dataset tokenization" || exit 1
                
            run_in_container "$HOST_TRAINING_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli run" "$CONTAINER_CONFIG_FILE")" \
                "Model training" || exit 1
            ;;
        *)
            log "ERROR" "Unknown phase '$PHASE'"
            log "ERROR" "Valid phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
            exit 1
            ;;
    esac
    
    log "SUCCESS" "========================================================"
    log "SUCCESS" "Experiment completed: $CONFIG_NAME"
    log "SUCCESS" "Phase: $PHASE"
    log "SUCCESS" "End time: $(date)"
    log "SUCCESS" "========================================================"
}

# Run main function
main "$@"