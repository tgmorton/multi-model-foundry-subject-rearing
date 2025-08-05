#!/bin/bash
# Generate Experiment Configurations
# This script generates YAML configs for all processed experiments

set -e

# Configuration
BASE_DIR="$(cd .. && pwd)"
PYTHON_SCRIPT="$BASE_DIR/scripts/generate_experiment_configs.py"
SINGULARITY_IMAGE="$BASE_DIR/singularity/training.sif"

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

# Check if Python script exists
check_python_script() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    success "Python script found: $PYTHON_SCRIPT"
}

# Main execution
main() {
    log "=== Generating Experiment Configurations ==="
    log "Base directory: $BASE_DIR"
    
    # Pre-flight checks
    check_singularity_image
    check_python_script
    
    # Run the Python script in Singularity container
    log "Running configuration generation..."
    
    singularity exec \
        -B "$BASE_DIR:/workspace" \
        -B "$HOME/.cache:/root/.cache" \
        "$SINGULARITY_IMAGE" \
        bash -c "cd /workspace && python $PYTHON_SCRIPT \
            --base-dir /workspace \
            --output-dir configs/processed_experiments \
            --generate-schedules"
    
    if [ $? -eq 0 ]; then
        success "Configuration generation completed successfully"
    else
        error "Configuration generation failed"
        exit 1
    fi
    
    log "=== Configuration Generation Complete ===""
    success "All configurations generated successfully!"
    log ""
    log "Generated configs are in: $BASE_DIR/configs/processed_experiments/"
    log ""
    log "Next steps:"
    log "  1. Review the generated configs"
    log "  2. Run experiments using: ./scripts/wild_west/run_experiment.sh <config_name>"
    log "  3. Example: ./scripts/wild_west/run_experiment.sh configs/processed_experiments/experiment_1_remove_expletives.yaml"
}

# Run main function
main 