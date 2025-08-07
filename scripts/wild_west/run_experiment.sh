#!/bin/bash

# Wild-West Experiment Runner
# Flexible script for running ablation experiments on specific GPUs

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

# Default values
DEFAULT_GPUS="1,2"  # Default to GPUs 1 and 2
DEFAULT_PHASE="full-pipeline"
DEFAULT_BATCH_SIZE=""
DEFAULT_NUM_EPOCHS=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <config_name>"
    echo ""
    echo "Options:"
    echo "  -g, --gpus <gpu_ids>     Comma-separated GPU IDs (default: $DEFAULT_GPUS)"
    echo "  -p, --phase <phase>       Phase to run (default: $DEFAULT_PHASE)"
    echo "  -b, --batch-size <size>   Batch size override"
    echo "  -e, --epochs <num>        Number of epochs override"
    echo "  -d, --distributed         Enable distributed training"
    echo "  -n, --nodes <num>         Number of nodes for distributed training"
    echo "  -r, --rank <rank>         Node rank for distributed training"
    echo "  -w, --world-size <size>   World size for distributed training"
    echo "  -l, --lock-gpus           Lock GPUs before running"
    echo "  -u, --unlock-gpus         Unlock GPUs after running"
    echo "  -c, --check-gpus          Check GPU availability before running"
    echo "  --wandb-mode <mode>       Weights & Biases mode (online, offline, disabled) (default: disabled)"
    echo "  --wandb-api-key <key>     Weights & Biases API key"
    echo "  -v, --verbose             Verbose output"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
    echo ""
    echo "Examples:"
    echo "  $0 experiment_1_remove_expletives"
    echo "  $0 -g 1,2 -p run experiment_2_baseline"
    echo "  $0 -g 1 -d -n 2 -r 0 experiment_3_remove_articles"
    echo "  $0 -g 1,2 -l -c experiment_4_lemmatize_verbs"
    echo "  $0 -g 0 --wandb-mode online --wandb-api-key YOUR_API_KEY experiment_0_baseline"
}

# Parse command line arguments
GPUS="$DEFAULT_GPUS"
PHASE="$DEFAULT_PHASE"
BATCH_SIZE=""
NUM_EPOCHS=""
DISTRIBUTED=false
NODES=1
RANK=0
WORLD_SIZE=1
LOCK_GPUS=false
UNLOCK_GPUS=false
CHECK_GPUS=false
WANDB_MODE="disabled"
WANDB_API_KEY=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
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
        -d|--distributed)
            DISTRIBUTED=true
            shift
            ;;
        -n|--nodes)
            NODES="$2"
            shift 2
            ;;
        -r|--rank)
            RANK="$2"
            shift 2
            ;;
        -w|--world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        -l|--lock-gpus)
            LOCK_GPUS=true
            shift
            ;;
        -u|--unlock-gpus)
            UNLOCK_GPUS=true
            shift
            ;;
        -c|--check-gpus)
            CHECK_GPUS=true
            shift
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

# Function to check GPU availability
check_gpu_availability() {
    log "INFO" "Checking GPU availability..."
    
    # Convert comma-separated GPUs to array
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        
        # Get GPU memory info
        local memory_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | grep "^$gpu_id,")
        
        if [ -z "$memory_info" ]; then
            log "ERROR" "GPU $gpu_id not found"
            return 1
        fi
        
        local memory_used=$(echo "$memory_info" | cut -d',' -f2 | tr -d ' ')
        local memory_total=$(echo "$memory_info" | cut -d',' -f3 | tr -d ' ')
        local available_gb=$((memory_total - memory_used))
        
        log "INFO" "GPU $gpu_id: ${available_gb}GB available (${memory_used}GB used / ${memory_total}GB total)"
        
        if [ "$available_gb" -lt 10240 ]; then  # Less than 10GB available
            log "WARN" "GPU $gpu_id has limited memory (${available_gb}GB available)"
        fi
    done
    
    return 0
}

# Function to check for defunct processes
check_defunct_processes() {
    log "INFO" "Checking for defunct processes..."
    
    # Look for defunct processes
    local defunct_count=$(ps aux | grep -c "defunct\|<defunct>" || echo "0")
    
    if [ "$defunct_count" -gt 0 ]; then
        log "WARN" "Found $defunct_count defunct processes:"
        ps aux | grep "defunct\|<defunct>" | head -5
        log "WARN" "Consider cleaning up defunct processes manually"
        return 1
    else
        log "INFO" "No defunct processes found"
        return 0
    fi
}

# Function to monitor GPU status
monitor_gpu_status() {
    log "INFO" "=== GPU Status ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    fi
    log "INFO" "================="
}

# Function to lock GPUs
lock_gpus() {
    log "INFO" "Locking GPUs: $GPUS"
    
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"
        
        if [ -f "$lock_file" ]; then
            log "ERROR" "GPU $gpu_id is already locked by:"
            cat "$lock_file"
            return 1
        fi
        
        echo "User: $(whoami)" > "$lock_file"
        echo "Time: $(date)" >> "$lock_file"
        echo "PID: $$" >> "$lock_file"
        echo "Config: $CONFIG_NAME" >> "$lock_file"
        echo "Phase: $PHASE" >> "$lock_file"
        
        log "SUCCESS" "Locked GPU $gpu_id"
    done
}

# Function to unlock GPUs
unlock_gpus() {
    log "INFO" "Unlocking GPUs: $GPUS"
    
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    for gpu_id in "${GPU_ARRAY[@]}"; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"
        
        if [ -f "$lock_file" ]; then
            rm "$lock_file"
            log "SUCCESS" "Unlocked GPU $gpu_id"
        else
            log "WARN" "No lock found on GPU $gpu_id"
        fi
    done
}

# Function to set up environment
setup_environment() {
    log "INFO" "Setting up environment..."
    
    # Set CUDA visible devices
    export CUDA_VISIBLE_DEVICES="$GPUS"
    log "INFO" "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
    
    # Set PyTorch CUDA allocator config
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Set distributed training environment variables if needed
    if [ "$DISTRIBUTED" = true ]; then
        export MASTER_ADDR="localhost"
        export MASTER_PORT="12355"
        export WORLD_SIZE="$WORLD_SIZE"
        export RANK="$RANK"
        export NCCL_DEBUG=INFO
        
        log "INFO" "Distributed training enabled:"
        log "INFO" "  World Size: $WORLD_SIZE"
        log "INFO" "  Rank: $RANK"
        log "INFO" "  Master Addr: $MASTER_ADDR"
        log "INFO" "  Master Port: $MASTER_PORT"
    fi
    
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

# Function to cleanup wandb processes specifically
cleanup_wandb() {
    log "INFO" "Cleaning up wandb processes..."
    
    # Find and kill wandb processes
    local wandb_pids=$(pgrep -f "wandb" || echo "")
    if [ -n "$wandb_pids" ]; then
        log "INFO" "Found wandb processes: $wandb_pids"
        echo "$wandb_pids" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        echo "$wandb_pids" | xargs kill -KILL 2>/dev/null || true
        log "INFO" "Wandb processes terminated"
    else
        log "INFO" "No wandb processes found"
    fi
}

# Function to cleanup processes and GPU memory
cleanup_processes() {
    log "INFO" "Cleaning up processes and GPU memory..."
    
    # Clean up wandb processes first (they can become orphaned)
    cleanup_wandb
    
    # Kill any remaining Python processes that might be holding GPU memory
    log "INFO" "Terminating Python processes..."
    pkill -f "python.*model_foundry" || true
    pkill -f "python.*trainer" || true
    pkill -f "wandb" || true
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Force kill any remaining processes if needed
    pkill -9 -f "python.*model_foundry" || true
    pkill -9 -f "python.*trainer" || true
    pkill -9 -f "wandb" || true
    
    # Clear CUDA cache if nvidia-smi is available
    log "INFO" "Clearing GPU memory..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset || true
        # Also try to clear cache
        python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    fi
    
    log "INFO" "Cleanup completed"
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
    
    # Set up wandb environment variables
    local wandb_env=""
    case "$WANDB_MODE" in
        "disabled")
            wandb_env="WANDB_MODE=disabled WANDB_SILENT=true"
            ;;
        "offline")
            wandb_env="WANDB_MODE=offline"
            ;;
        "online")
            wandb_env="WANDB_MODE=online"
            ;;
    esac
    
    # Add API key if provided
    if [ -n "$WANDB_API_KEY" ]; then
        wandb_env="$wandb_env WANDB_API_KEY=$WANDB_API_KEY"
    fi
    
    # Create a wrapper script for proper signal handling inside the container
    local wrapper_script=$(mktemp)
    cat > "$wrapper_script" << 'EOF'
#!/bin/bash
set -e

# Function to cleanup inside container
cleanup_container() {
    echo "Cleaning up inside container..."
    
    # Kill wandb processes first (they can become orphaned)
    pkill -f "wandb" || true
    sleep 1
    pkill -9 -f "wandb" || true
    
    # Kill any Python processes
    pkill -f "python.*model_foundry" || true
    pkill -f "python.*trainer" || true
    sleep 1
    pkill -9 -f "python.*model_foundry" || true
    pkill -9 -f "python.*trainer" || true
    
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    
    echo "Container cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup_container INT TERM

# Change to workspace directory
cd /workspace

# Download spaCy model
python -m spacy download en_core_web_sm --quiet

# Execute the actual command
exec "$@"
EOF
    
    chmod +x "$wrapper_script"
    
    # Execute the command inside the container with proper signal handling and timeout
    log "INFO" "Starting container execution with 24-hour timeout..."
    timeout 24h singularity exec --nv --cleanenv \
        --bind "${PROJECT_DIR}":/workspace \
        --bind "$wrapper_script":/tmp/wrapper.sh \
        "$container_path" \
        bash -c "/tmp/wrapper.sh bash -c '$wandb_env $command'"
    
    # Check if timeout occurred
    local exit_code=$?
    if [ $exit_code -eq 124 ]; then
        log "WARN" "Container execution was terminated due to timeout (24 hours)"
        cleanup_processes
        return 1
    elif [ $exit_code -ne 0 ]; then
        log "ERROR" "Container execution failed with exit code $exit_code"
        cleanup_processes
        return $exit_code
    fi
    
    # Clean up wrapper script
    rm -f "$wrapper_script"
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
    
    # Add distributed training flags if enabled
    if [ "$DISTRIBUTED" = true ]; then
        command="$command --distributed --world-size $WORLD_SIZE --rank $RANK"
    fi
    
    echo "$command"
}

# Main execution
main() {
    # Set up trap to call cleanup on script exit
    trap 'log "INFO" "Received interrupt signal, cleaning up..."; cleanup_processes; unlock_gpus; exit' INT TERM EXIT
    
    log "INFO" "Starting experiment: $CONFIG_NAME"
    log "INFO" "Phase: $PHASE"
    log "INFO" "GPUs: $GPUS"
    
    # Check GPU availability if requested
    if [ "$CHECK_GPUS" = true ]; then
        check_gpu_availability || exit 1
    fi
    
    # Always check for defunct processes at start
    check_defunct_processes
    
    # Show initial GPU status
    monitor_gpu_status
    
    # Lock GPUs if requested
    if [ "$LOCK_GPUS" = true ]; then
        lock_gpus || exit 1
    fi
    
    # Note: GPU unlocking is now handled in the main trap above
    
    # Setup environment
    setup_environment
    
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
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli preprocess" "$CONTAINER_CONFIG_FILE")" \
                "Dataset preprocessing and ablation"
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli train-tokenizer" "$CONTAINER_CONFIG_FILE")" \
                "SentencePiece tokenizer training"
            run_in_container "$HOST_ABLATION_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli tokenize-dataset" "$CONTAINER_CONFIG_FILE")" \
                "Dataset tokenization"
            run_in_container "$HOST_TRAINING_SIF_PATH" \
                "$(build_command "python -m model_foundry.cli run" "$CONTAINER_CONFIG_FILE")" \
                "Model training"
            ;;
        *)
            log "ERROR" "Unknown phase '$PHASE'"
            log "ERROR" "Valid phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
            exit 1
            ;;
    esac
    
    # Final monitoring
    log "INFO" "=== Final Status ==="
    check_defunct_processes
    monitor_gpu_status
    
    log "SUCCESS" "Experiment completed: $CONFIG_NAME"
    log "SUCCESS" "Phase: $PHASE"
    log "SUCCESS" "GPUs used: $GPUS"
}

# Run main function
main "$@" 