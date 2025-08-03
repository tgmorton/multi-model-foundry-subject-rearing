#!/bin/bash
# Generate All Experiment Configurations for Wild-West Server
# This script generates configuration files for all experiments based on processed data

set -e

# Configuration
BASE_DIR="$(pwd)"
PYTHON_SCRIPT="scripts/generate_experiment_configs.py"

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

# Check if Python script exists
check_python_script() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    success "Python script found: $PYTHON_SCRIPT"
}

# Check if processed data directory exists
check_processed_data() {
    local processed_dir="$BASE_DIR/data/processed"
    if [ ! -d "$processed_dir" ]; then
        error "Processed data directory not found: $processed_dir"
        exit 1
    fi
    
    # List available experiments
    log "Available experiments in processed data:"
    for exp_dir in "$processed_dir"/*; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            log "  - $exp_name"
        fi
    done
}

# Generate configurations
generate_configs() {
    log "=== Generating Experiment Configurations ==="
    
    # Run the Python script
    python "$PYTHON_SCRIPT" \
        --base-dir "$BASE_DIR" \
        --output-dir "configs" \
        --create-scripts \
        --scripts-dir "scripts"
    
    if [ $? -eq 0 ]; then
        success "Configuration generation completed successfully"
    else
        error "Configuration generation failed"
        exit 1
    fi
}

# Create Experiment 0 baseline config if it doesn't exist
create_exp0_config() {
    local exp0_config="configs/experiment_0_baseline_90M.yaml"
    
    if [ ! -f "$exp0_config" ]; then
        log "Creating Experiment 0 baseline configuration..."
        
        cat > "$exp0_config" << 'EOF'
# ===================================================================
# EXPERIMENT 0: Baseline with 90M Training Data
# GOAL: Train the base model on the unaltered BabyLM 90M corpus.
# This serves as the control against which all ablations are measured.
# ===================================================================

experiment_name: "exp0_baseline_90M"

# --- Data Configuration ---
data:
  # Path to the raw, unprocessed training data.
  source_corpus: "data/raw/train_90M/"
  # The final training corpus is the same as the source in this case.
  training_corpus: "data/raw/train_90M/"
  # Configuration for the dataset loader.
  batch_size: 256
  max_sequence_length: 128

# --- Dataset Manipulation Pipeline ---
# This section is empty because this is the baseline experiment.
# No ablations are performed.
dataset_manipulation: []

# --- Tokenizer Configuration ---
tokenizer:
  # Where to save the trained tokenizer for this experiment.
  output_dir: "tokenizers/exp0_baseline_90M/"
  vocab_size: 50004

# --- Model Architecture ---
# Parameters for the GPT-2 model, based on your proposal's Table 5.
model:
  layers: 12
  embedding_size: 768
  hidden_size: 768
  intermediate_hidden_size: 3072
  attention_heads: 12
  activation_function: "GELU"
  dropout: 0.1
  attention_dropout: 0.1

# --- Training Procedure ---
training:
  # Path to save model checkpoints.
  output_dir: "models/exp0_baseline_90M/"
  # Training parameters from Table 5.
  learning_rate: 0.0001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-6
  warmup_steps: 10000
  train_steps: 1000000
  epochs: 20
  
  # Enhanced training features
  use_amp: true  # Enable automatic mixed precision
  gradient_accumulation_steps: 1  # Number of steps to accumulate gradients
  use_tf32: true  # Enable TF32 for faster training on Ampere+ GPUs
  use_gradient_checkpointing: false  # Enable gradient checkpointing to save memory
  
  # Checkpoint generation parameters
  auto_generate_checkpoints: true
  first_epoch_checkpoints: 20  # Number of checkpoints in first epoch
  subsequent_epochs_spacing: "log"  # "linear" or "log"
  log_base: 2  # Base for logarithmic spacing (default 2)
  linear_interval: null  # Steps between checkpoints for linear spacing
  min_checkpoint_interval: 100  # Minimum interval between checkpoints
  resume_from_checkpoint: false

# --- Logging & Reproducibility ---
logging:
  # Integrate with Weights & Biases as per the proposal.
  use_wandb: true
  wandb_project: "just-drop-the-subject"
  level: "INFO"
  dir: "logs"

# A random seed for ensuring reproducibility between experiments.
random_seed: 42
EOF
        
        success "Created Experiment 0 baseline configuration: $exp0_config"
    else
        log "Experiment 0 baseline configuration already exists: $exp0_config"
    fi
}

# Create run scripts for all experiments
create_run_scripts() {
    log "=== Creating Run Scripts ==="
    
    # Create directory for run scripts
    local scripts_dir="scripts/wild_west/experiments"
    mkdir -p "$scripts_dir"
    
    # Find all config files
    for config_file in configs/experiment_*.yaml; do
        if [ -f "$config_file" ]; then
            exp_name=$(basename "$config_file" .yaml)
            script_file="$scripts_dir/run_${exp_name}.sh"
            
            log "Creating run script for $exp_name"
            
            # Create the script content
            cat > "$script_file" << 'SCRIPT_EOF'
#!/bin/bash
# Run script for experiment
# Generated automatically by generate_all_configs.sh

set -e

# Configuration
EXPERIMENT_NAME="SCRIPT_EXPERIMENT_NAME"
CONFIG_FILE="configs/SCRIPT_CONFIG_NAME.yaml"
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
    log "=== Starting $EXPERIMENT_NAME ==="
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
    
    log "=== $EXPERIMENT_NAME Complete ==="
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
SCRIPT_EOF
            
            # Replace placeholders with actual values
            sed -i "s/SCRIPT_EXPERIMENT_NAME/$exp_name/g" "$script_file"
            sed -i "s/SCRIPT_CONFIG_NAME/$exp_name/g" "$script_file"
            
            chmod +x "$script_file"
            success "Created run script: $script_file"
        fi
    done
}

# Main execution
main() {
    log "=== Generating All Experiment Configurations ==="
    log "Base directory: $BASE_DIR"
    
    # Pre-flight checks
    check_python_script
    check_processed_data
    
    # Generate configurations
    generate_configs
    
    # Create Experiment 0 config if needed
    create_exp0_config
    
    # Create run scripts
    create_run_scripts
    
    log "=== Configuration Generation Complete ===""
    success "All configurations and scripts generated successfully!"
    log ""
    log "Next steps:"
    log "  1. Review the generated configs in configs/"
    log "  2. Run experiments using: ./scripts/wild_west/experiments/run_<experiment>.sh"
    log "  3. For Experiment 0: ./scripts/wild_west/run_exp0_baseline.sh"
}

# Run main function
main 