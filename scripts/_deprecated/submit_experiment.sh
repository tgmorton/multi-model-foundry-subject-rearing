#!/bin/bash
# Helper script for submitting experimental jobs to SLURM

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    print_error "Missing required arguments."
    echo "Usage: $0 <config_name_without_yaml_extension> <phase> [job_name_suffix]"
    echo ""
    echo "Arguments:"
    echo "  config_name: Name of the config file without .yaml extension"
    echo "  phase: One of preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
    echo "  job_name_suffix: Optional suffix for the job name (default: none)"
    echo ""
    echo "Examples:"
    echo "  $0 experiment_1_remove_expletives preprocess"
    echo "  $0 experiment_2_impoverish_determiners full-pipeline"
    echo "  $0 experiment_0_baseline run baseline_run"
    exit 1
fi

CONFIG_NAME="$1"
PHASE="$2"
JOB_SUFFIX="$3"

# Validate phase
VALID_PHASES=("preprocess" "train-tokenizer" "tokenize-dataset" "run" "full-pipeline")
VALID_PHASE=false
for phase in "${VALID_PHASES[@]}"; do
    if [ "$PHASE" = "$phase" ]; then
        VALID_PHASE=true
        break
    fi
done

if [ "$VALID_PHASE" = false ]; then
    print_error "Invalid phase '$PHASE'"
    echo "Valid phases: ${VALID_PHASES[*]}"
    exit 1
fi

# Check if config file exists
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 configs/*.yaml 2>/dev/null | sed 's/configs\///' | sed 's/\.yaml$//' || echo "No config files found"
    exit 1
fi

# Construct job name
if [ -n "$JOB_SUFFIX" ]; then
    JOB_NAME="subject-drop-${CONFIG_NAME}-${PHASE}-${JOB_SUFFIX}"
else
    JOB_NAME="subject-drop-${CONFIG_NAME}-${PHASE}"
fi

print_info "Submitting job: $JOB_NAME"
print_info "Config: $CONFIG_NAME"
print_info "Phase: $PHASE"

# Submit the job
JOB_ID=$(sbatch --job-name="$JOB_NAME" scripts/run_ablation_experiment.sh "$CONFIG_NAME" "$PHASE" | grep -o '[0-9]\+')

if [ -n "$JOB_ID" ]; then
    print_success "Job submitted successfully!"
    print_info "Job ID: $JOB_ID"
    print_info "Job Name: $JOB_NAME"
    print_info "Monitor with: squeue -j $JOB_ID"
    print_info "View logs with: tail -f logs/${JOB_NAME}-${JOB_ID}.out"
else
    print_error "Failed to submit job"
    exit 1
fi 