#!/bin/bash
# Experiment 0 Baseline Runner for Wild-West Server
# This script runs the complete workflow for the baseline experiment with 90M data
# Uses Singularity containers and GPU management

set -e

# Configuration
EXPERIMENT_NAME="exp0_baseline_90M"
CONFIG_FILE="configs/experiment_0_baseline_90M.yaml"
SINGULARITY_IMAGE="singularity/training.sif"
BASE_DIR="$(pwd)"

# Default GPU settings
DEFAULT_GPUS="1,2"
GPUS=${GPUS:-$DEFAULT_GPUS}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Singularity image exists
check_singularity_image() {
    if [ ! -f "$SINGULARITY_IMAGE" ]; then
        error "Singularity image not found: $SINGULARITY_IMAGE"
        error "Please build the image first: singularity build $SINGULARITY_IMAGE singularity/training.def"
        exit 1
    fi
    success "Singularity image found: $SINGULARITY_IMAGE"
}

# Check if config file exists
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    success "Configuration file found: $CONFIG_FILE"
}

# Check GPU availability
check_gpus() {
    log "Checking GPU availability..."
    available_gpus=$(./scripts/wild_west/gpu_monitor.sh available | grep -o '[0-9]' | tr '\n' ',' | sed 's/,$//')
    
    if [ -z "$available_gpus" ]; then
        error "No available GPUs found"
        exit 1
    fi
    
    log "Available GPUs: $available_gpus"
    log "Requested GPUs: $GPUS"
    
    # Check if requested GPUs are available
    for gpu in $(echo $GPUS | tr ',' ' '); do
        if ! echo "$available_gpus" | grep -q "$gpu"; then
            error "GPU $gpu is not available"
            exit 1
        fi
    done
    
    success "All requested GPUs are available"
}

# Lock GPUs
lock_gpus() {
    log "Locking GPUs: $GPUS"
    for gpu in $(echo $GPUS | tr ',' ' '); do
        ./scripts/wild_west/gpu_monitor.sh lock $gpu
    done
    success "GPUs locked successfully"
}

# Unlock GPUs
unlock_gpus() {
    log "Unlocking GPUs: $GPUS"
    for gpu in $(echo $GPUS | tr ',' ' '); do
        ./scripts/wild_west/gpu_monitor.sh unlock $gpu
    done
    success "GPUs unlocked successfully"
}

# Run command in Singularity container
run_in_container() {
    local cmd="$1"
    local description="$2"
    
    log "Running: $description"
    log "Command: $cmd"
    
    # Set CUDA_VISIBLE_DEVICES for GPU selection
    export CUDA_VISIBLE_DEVICES=$GPUS
    
    # Run in Singularity container
    singularity exec --nv \
        -B "$BASE_DIR:/workspace" \
        -B "$HOME/.cache:/root/.cache" \
        "$SINGULARITY_IMAGE" \
        bash -c "cd /workspace && $cmd"
    
    if [ $? -eq 0 ]; then
        success "$description completed successfully"
    else
        error "$description failed"
        return 1
    fi
}

# Main execution function
run_experiment() {
    log "=== Starting Experiment 0: Baseline with 90M Data ==="
    log "Experiment: $EXPERIMENT_NAME"
    log "GPUs: $GPUS"
    log "Config: $CONFIG_FILE"
    
    # Pre-flight checks
    check_singularity_image
    check_config
    check_gpus
    
    # Lock GPUs
    lock_gpus
    
    # Trap to ensure GPUs are unlocked on exit
    trap 'unlock_gpus' EXIT
    
    # Step 1: Generate checkpoint schedule
    log "=== Step 1: Generating checkpoint schedule ==="
    run_in_container \
        "python scripts/generate_checkpoint_schedule.py $CONFIG_FILE" \
        "Checkpoint schedule generation"
    
    # Step 2: Train tokenizer
    log "=== Step 2: Training tokenizer ==="
    run_in_container \
        "python -m model_foundry.tokenizer.train_tokenizer --config $CONFIG_FILE --base_dir /workspace" \
        "Tokenizer training"
    
    # Step 3: Tokenize dataset
    log "=== Step 3: Tokenizing dataset ==="
    run_in_container \
        "python -m model_foundry.tokenizer.tokenize_dataset --config $CONFIG_FILE --base_dir /workspace" \
        "Dataset tokenization"
    
    # Step 4: Run training
    log "=== Step 4: Running training ==="
    run_in_container \
        "python -m model_foundry.trainer $CONFIG_FILE" \
        "Model training"
    
    # Unlock GPUs (trap will handle this on exit)
    unlock_gpus
    
    log "=== Experiment 0 Baseline Complete ==="
    success "All steps completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -g, --gpus GPUS     Comma-separated GPU IDs (default: $DEFAULT_GPUS)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -g 1,2"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the experiment
run_experiment 