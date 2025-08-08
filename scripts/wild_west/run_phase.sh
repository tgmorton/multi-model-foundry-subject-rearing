#!/bin/bash

# Single-GPU Phase Runner for Wild-West Server
# Runs individual phases of experiments on a single GPU

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOCK_DIR="/tmp/gpu_locks"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <config_name> <phase>"
    echo ""
    echo "Options:"
    echo "  -g, --gpu <gpu_id>        GPU ID to use (default: 1)"
    echo "  -b, --batch-size <size>   Batch size override"
    echo "  -e, --epochs <num>        Number of epochs override"
    echo "  -l, --lock-gpu            Lock GPU before running"
    echo "  -u, --unlock-gpu          Unlock GPU after running"
    echo "  -c, --check-gpu           Check GPU availability before running"
    echo "  -v, --verbose             Verbose output"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Phases:"
    echo "  preprocess      - Dataset preprocessing and ablation"
    echo "  train-tokenizer - SentencePiece tokenizer training"
    echo "  tokenize-dataset- Dataset tokenization"
    echo "  chunk-data      - Dataset chunking into fixed-length blocks"
    echo "  run            - Model training"
    echo ""
    echo "Examples:"
    echo "  $0 experiment_1_remove_expletives preprocess"
    echo "  $0 -g 2 experiment_2_baseline run"
    echo "  $0 -g 1 -l -c experiment_3_remove_articles train-tokenizer"
}

# Parse command line arguments
GPU_ID="1"
BATCH_SIZE=""
NUM_EPOCHS=""
LOCK_GPU=false
UNLOCK_GPU=false
CHECK_GPU=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU_ID="$2"
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
        -l|--lock-gpu)
            LOCK_GPU=true
            shift
            ;;
        -u|--unlock-gpu)
            UNLOCK_GPU=true
            shift
            ;;
        -c|--check-gpu)
            CHECK_GPU=true
            shift
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
            if [ -z "$CONFIG_NAME" ]; then
                CONFIG_NAME="$1"
            elif [ -z "$PHASE" ]; then
                PHASE="$1"
            else
                echo "Too many arguments"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$CONFIG_NAME" ] || [ -z "$PHASE" ]; then
    echo -e "${RED}ERROR: Config name and phase are required${NC}"
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

# Function to check GPU availability
check_gpu_availability() {
    log "INFO" "Checking GPU $GPU_ID availability..."
    
    # Get GPU memory info
    local memory_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | grep "^$GPU_ID,")
    
    if [ -z "$memory_info" ]; then
        log "ERROR" "GPU $GPU_ID not found"
        return 1
    fi
    
    local memory_used=$(echo "$memory_info" | cut -d',' -f2 | tr -d ' ')
    local memory_total=$(echo "$memory_info" | cut -d',' -f3 | tr -d ' ')
    local available_gb=$((memory_total - memory_used))
    
    log "INFO" "GPU $GPU_ID: ${available_gb}GB available (${memory_used}GB used / ${memory_total}GB total)"
    
    if [ "$available_gb" -lt 5120 ]; then  # Less than 5GB available
        log "WARN" "GPU $GPU_ID has very limited memory (${available_gb}GB available)"
        return 1
    elif [ "$available_gb" -lt 10240 ]; then  # Less than 10GB available
        log "WARN" "GPU $GPU_ID has limited memory (${available_gb}GB available)"
    fi
    
    return 0
}

# Function to lock GPU
lock_gpu() {
    log "INFO" "Locking GPU $GPU_ID"
    
    local lock_file="$LOCK_DIR/gpu_${GPU_ID}.lock"
    
    if [ -f "$lock_file" ]; then
        log "ERROR" "GPU $GPU_ID is already locked by:"
        cat "$lock_file"
        return 1
    fi
    
    echo "User: $(whoami)" > "$lock_file"
    echo "Time: $(date)" >> "$lock_file"
    echo "PID: $$" >> "$lock_file"
    echo "Config: $CONFIG_NAME" >> "$lock_file"
    echo "Phase: $PHASE" >> "$lock_file"
    
    log "SUCCESS" "Locked GPU $GPU_ID"
    return 0
}

# Function to unlock GPU
unlock_gpu() {
    log "INFO" "Unlocking GPU $GPU_ID"
    
    local lock_file="$LOCK_DIR/gpu_${GPU_ID}.lock"
    
    if [ -f "$lock_file" ]; then
        rm "$lock_file"
        log "SUCCESS" "Unlocked GPU $GPU_ID"
    else
        log "WARN" "No lock found on GPU $GPU_ID"
    fi
}

# Function to set up environment
setup_environment() {
    log "INFO" "Setting up environment..."
    
    # Set CUDA visible devices
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    log "INFO" "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
    
    # Set PyTorch CUDA allocator config
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Load modules if available
    if command -v module &> /dev/null; then
        log "INFO" "Loading system modules..."
        module load singularity/4.1.1 cuda/11.8 2>/dev/null || log "WARN" "Could not load modules"
    fi
    
    # Check if singularity is available
    if ! command -v singularity &> /dev/null; then
        log "WARN" "Singularity not found in PATH. Make sure it's installed and loaded."
    fi
}

# Function to run command in container
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
    
    # Execute the command inside the container with spaCy model download
    singularity exec --nv \
        --bind "${PROJECT_DIR}":/workspace \
        "$container_path" \
        bash -c "cd /workspace && python -m spacy download en_core_web_sm --quiet && $command"
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
        "preprocess"|"train-tokenizer"|"tokenize-dataset"|"chunk-data")
            echo "$PROJECT_DIR/singularity/ablation.sif"
            ;;
        "run")
            echo "$PROJECT_DIR/singularity/training.sif"
            ;;
        *)
            log "ERROR" "Unknown phase: $phase"
            return 1
            ;;
    esac
}

# Main execution
main() {
    log "INFO" "Starting phase: $PHASE"
    log "INFO" "Config: $CONFIG_NAME"
    log "INFO" "GPU: $GPU_ID"
    
    # Check GPU availability if requested
    if [ "$CHECK_GPU" = true ]; then
        check_gpu_availability || exit 1
    fi
    
    # Lock GPU if requested
    if [ "$LOCK_GPU" = true ]; then
        lock_gpu || exit 1
    fi
    
    # Set up trap to unlock GPU on exit
    if [ "$UNLOCK_GPU" = true ]; then
        trap 'unlock_gpu' EXIT
    fi
    
    # Setup environment
    setup_environment
    
    # Define paths
    HOST_CONFIG_FILE="$PROJECT_DIR/configs/${CONFIG_NAME}.yaml"
    CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"
    CONTAINER_PATH=$(get_container_path "$PHASE")
    
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
        "chunk-data")
            log "INFO" "Running dataset chunking phase..."
            run_in_container "$CONTAINER_PATH" \
                "$(build_command "python -m model_foundry.cli preprocess-data" "$CONTAINER_CONFIG_FILE")" \
                "Dataset chunking"
            ;;
        "run")
            log "INFO" "Running model training phase..."
            run_in_container "$CONTAINER_PATH" \
                "$(build_command "python -m model_foundry.cli run" "$CONTAINER_CONFIG_FILE")" \
                "Model training"
            ;;
        *)
            log "ERROR" "Unknown phase '$PHASE'"
            log "ERROR" "Valid phases: preprocess, train-tokenizer, tokenize-dataset, chunk-data, run"
            exit 1
            ;;
    esac
    
    log "SUCCESS" "Phase completed: $PHASE"
    log "SUCCESS" "Config: $CONFIG_NAME"
    log "SUCCESS" "GPU used: $GPU_ID"
}

# Run main function
main "$@" 