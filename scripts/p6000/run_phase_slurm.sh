#!/bin/bash
#SBATCH --job-name=subject-drop-phase
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# P6000 SLURM Single Phase Runner
# Adapted from wild_west/run_phase.sh with SLURM best practices
# Runs individual phases of experiments with proper resource management

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - Use actual cluster paths
HOST_PROJECT_DIR="/labs/ferreiralab/thmorton/subject-drop-rearing"
SCRIPT_DIR="$HOST_PROJECT_DIR/scripts/p6000"
PROJECT_DIR="$HOST_PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$HOST_PROJECT_DIR/logs"

# Set SLURM output/error files dynamically
if [ -n "${SLURM_JOB_ID:-}" ]; then
    exec 1> "$HOST_PROJECT_DIR/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID}.out"
    exec 2> "$HOST_PROJECT_DIR/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID}.err"
fi

# Function to show usage
show_usage() {
    echo "Usage: sbatch $0 [SLURM_OPTIONS] -- <config_name> <phase> [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  <config_name>             Configuration file name (without .yaml)"
    echo "  <phase>                   Phase to run"
    echo ""
    echo "SLURM Options (before --):"
    echo "  --job-name=<name>         Job name"
    echo "  --time=<time>             Time limit (default: 12:00:00)"
    echo "  --mem=<memory>            Memory allocation (default: 32G)"
    echo ""
    echo "Script Options (after phase):"
    echo "  -b, --batch-size <size>   Batch size override"
    echo "  -e, --epochs <num>        Number of epochs override"
    echo "  --wandb-mode <mode>       W&B mode: online|offline|disabled (default: disabled)"
    echo "  --wandb-api-key <key>     W&B API key"
    echo "  -v, --verbose             Verbose output"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Phases:"
    echo "  preprocess                Dataset preprocessing and ablation"
    echo "  train-tokenizer           SentencePiece tokenizer training"
    echo "  tokenize-dataset          Dataset tokenization"
    echo "  run                       Model training"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 -- experiment_1_remove_expletives preprocess"
    echo "  sbatch --time=24:00:00 $0 -- experiment_0_baseline run -b 64"
    echo "  sbatch --job-name=tokenizer $0 -- experiment_2_baseline train-tokenizer"
}

# Parse command line arguments (after --)
CONFIG_NAME=""
PHASE=""
BATCH_SIZE=""
NUM_EPOCHS=""
WANDB_MODE="disabled"
WANDB_API_KEY=""
VERBOSE=false

# Skip to arguments after --
while [[ $# -gt 0 ]] && [[ "$1" != "--" ]]; do
    shift
done
[[ "$1" == "--" ]] && shift

# First two arguments after -- are config_name and phase
if [ $# -lt 2 ]; then
    echo -e "${RED}ERROR: Config name and phase are required${NC}"
    show_usage
    exit 1
fi

CONFIG_NAME="$1"
PHASE="$2"
shift 2

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Unexpected argument: $1"
            show_usage
            exit 1
            ;;
    esac
done

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

# Function to check GPU availability
check_gpu_availability() {
    log "INFO" "Checking GPU availability..."
    
    # Use nvidia-smi to check GPU status
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
              --format=csv,noheader,nounits || {
        log "ERROR" "Failed to query GPU status"
        return 1
    }
    
    # Check available memory
    local mem_info=$(nvidia-smi --query-gpu=memory.used,memory.total \
                     --format=csv,noheader,nounits | head -n1)
    local mem_used=$(echo "$mem_info" | cut -d',' -f1 | tr -d ' ')
    local mem_total=$(echo "$mem_info" | cut -d',' -f2 | tr -d ' ')
    local mem_available=$((mem_total - mem_used))
    
    log "INFO" "GPU Memory: ${mem_available}MB available (${mem_used}MB used / ${mem_total}MB total)"
    
    if [ "$mem_available" -lt 5120 ]; then
        log "WARN" "Low GPU memory available (${mem_available}MB)"
        if [ "$PHASE" == "run" ]; then
            log "ERROR" "Insufficient memory for training phase"
            return 1
        fi
    fi
    
    return 0
}

# Function to setup SLURM environment
setup_slurm_environment() {
    log "INFO" "Setting up SLURM environment..."
    
    # Log SLURM job information
    log "INFO" "========================================================"
    log "INFO" "Job ID: ${SLURM_JOB_ID:-N/A}"
    log "INFO" "Job Name: ${SLURM_JOB_NAME:-N/A}"
    log "INFO" "Node: ${SLURMD_NODENAME:-N/A}"
    log "INFO" "Partition: ${SLURM_JOB_PARTITION:-N/A}"
    log "INFO" "Task ID: ${SLURM_PROCID:-0}"
    log "INFO" "GPUs: ${CUDA_VISIBLE_DEVICES:-All available}"
    log "INFO" "========================================================"
    
    # Load necessary modules
    log "INFO" "Loading system modules..."
    module load singularity/4.1.1 cuda/11.8 2>/dev/null || log "WARN" "Could not load modules"
    
    # Set PyTorch CUDA allocator config based on phase
    if [ "$PHASE" == "run" ]; then
        # Training phase - maximize memory efficiency
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
        export TORCH_CUDA_MEMORY_FRACTION=0.95
    else
        # Preprocessing phases - standard allocation
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TORCH_CUDA_MEMORY_FRACTION=0.90
    fi
    
    export CUDA_LAUNCH_BLOCKING=0
    export NCCL_DEBUG=WARN
    
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

# Function to determine container based on phase
get_container_path() {
    local phase="$1"
    
    case $phase in
        "preprocess"|"train-tokenizer"|"tokenize-dataset")
            echo "$HOST_PROJECT_DIR/singularity/ablation.sif"
            ;;
        "run")
            echo "$HOST_PROJECT_DIR/singularity/training.sif"
            ;;
        *)
            log "ERROR" "Unknown phase: $phase"
            return 1
            ;;
    esac
}

# Function to estimate time requirement for phase
estimate_phase_time() {
    local phase="$1"
    
    case $phase in
        "preprocess")
            echo "2:00:00"
            ;;
        "train-tokenizer")
            echo "1:00:00"
            ;;
        "tokenize-dataset")
            echo "1:00:00"
            ;;
        "run")
            echo "24:00:00"
            ;;
        *)
            echo "12:00:00"
            ;;
    esac
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
        if [ -n "$env_flags" ]; then
            env_flags="${env_flags},WANDB_API_KEY=${WANDB_API_KEY}"
        else
            env_flags="WANDB_API_KEY=${WANDB_API_KEY}"
        fi
    fi
    
    # Add environment flags to container command if any are set
    local env_option=""
    if [ -n "$env_flags" ]; then
        env_option="--env $env_flags"
    fi
    
    # Execute with srun for proper SLURM integration
    srun --exclusive --ntasks=1 --gpus-per-task=1 \
        "$CONTAINER_BIN" exec --nv --pid --contain --cleanenv \
        $env_option \
        --bind "${HOST_PROJECT_DIR}":/workspace \
        "$container_path" \
        bash -c "
            set -euo pipefail
            cd /workspace
            
            # Download spaCy model if needed (only for preprocessing)
            if [ '$PHASE' == 'preprocess' ]; then
                python -m spacy download en_core_web_sm --quiet 2>/dev/null || true
            fi
            
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
    log "INFO" "Starting phase: $PHASE"
    log "INFO" "Config: $CONFIG_NAME"
    log "INFO" "Start time: $(date)"
    log "INFO" "========================================================"
    
    # Setup SLURM environment
    setup_slurm_environment
    
    # Check GPU availability
    check_gpu_availability || {
        log "ERROR" "GPU check failed, aborting"
        exit 1
    }
    
    # Define paths - using cluster absolute paths
    HOST_CONFIG_FILE="$HOST_PROJECT_DIR/configs/${CONFIG_NAME}.yaml"
    CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"
    CONTAINER_PATH=$(get_container_path "$PHASE")
    
    # Check for required files
    if [ ! -f "$HOST_CONFIG_FILE" ]; then
        log "ERROR" "Config file not found at $HOST_CONFIG_FILE"
        exit 1
    fi
    
    # Create logs directory (already done at top)
    # mkdir -p "$HOST_PROJECT_DIR/logs"
    
    # Log estimated time for this phase
    local estimated_time=$(estimate_phase_time "$PHASE")
    log "INFO" "Estimated time for $PHASE: $estimated_time"
    
    # Phase-specific execution
    case $PHASE in
        "preprocess")
            log "INFO" "Running preprocessing phase..."
            run_in_container "$CONTAINER_PATH" \
                "$(build_command "python -m model_foundry.cli preprocess" "$CONTAINER_CONFIG_FILE")" \
                "Dataset preprocessing and ablation"
            ;;
        "train-tokenizer")
            log "INFO" "Running tokenizer training phase..."
            run_in_container "$CONTAINER_PATH" \
                "$(build_command "python -m model_foundry.cli train-tokenizer" "$CONTAINER_CONFIG_FILE")" \
                "SentencePiece tokenizer training"
            ;;
        "tokenize-dataset")
            log "INFO" "Running dataset tokenization phase..."
            run_in_container "$CONTAINER_PATH" \
                "$(build_command "python -m model_foundry.cli tokenize-dataset" "$CONTAINER_CONFIG_FILE")" \
                "Dataset tokenization"
            ;;
        "run")
            log "INFO" "Running model training phase..."
            run_in_container "$CONTAINER_PATH" \
                "$(build_command "python -m model_foundry.cli run" "$CONTAINER_CONFIG_FILE")" \
                "Model training"
            ;;
        *)
            log "ERROR" "Unknown phase '$PHASE'"
            log "ERROR" "Valid phases: preprocess, train-tokenizer, tokenize-dataset, run"
            exit 1
            ;;
    esac
    
    log "SUCCESS" "========================================================"
    log "SUCCESS" "Phase completed: $PHASE"
    log "SUCCESS" "Config: $CONFIG_NAME"
    log "SUCCESS" "End time: $(date)"
    log "SUCCESS" "========================================================"
}

# Run main function
main "$@"