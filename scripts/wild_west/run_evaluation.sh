#!/usr/bin/env bash
# Wild-West Evaluation Runner
# Evaluate trained language models using the evaluation pipeline

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOCK_DIR="/tmp/gpu_locks"

# Defaults
DEFAULT_GPUS="1"  # Evaluation typically needs less GPU power
DEFAULT_CONFIG="configs/evaluation_config.yaml"

# Logging function
log() {
    local level="$1"; shift
    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
    case "$level" in
        INFO)    echo -e "${BLUE}[$ts] INFO: $*${NC}" ;;
        WARN)    echo -e "${YELLOW}[$ts] WARN: $*${NC}" ;;
        ERROR)   echo -e "${RED}[$ts] ERROR: $*${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$ts] SUCCESS: $*${NC}" ;;
    esac
}

# Usage information
show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [experiment_name]

Run evaluation pipeline on trained language models.

Options:
  -g, --gpus <ids>           GPU IDs to use (default: $DEFAULT_GPUS)
  -c, --config <config>      Evaluation config file (default: $DEFAULT_CONFIG)
  -e, --experiment <exp>     Experiment to evaluate (e.g., exp0_baseline)
  -k, --checkpoint <path>    Specific checkpoint to evaluate
  -t, --tasks <tasks>        Tasks to run: perplexity,blimp,null_subject (default: all)
  -s, --samples <num>        Max samples per task (for testing)
  -m, --max-checkpoints <n>  Max checkpoints to evaluate
  -o, --output <dir>         Output directory override
  -l, --lock-gpus            Lock GPUs before running
  -u, --unlock-gpus          Unlock GPUs after running
  -f, --fast                 Fast evaluation (reduced samples)
  -v, --verbose              Verbose output
  -h, --help                 Show this help

Examples:
  # Evaluate exp0_baseline experiment
  $0 exp0_baseline

  # Evaluate with specific config
  $0 -c configs/eval_exp1.yaml exp1_remove_expletives

  # Fast evaluation for testing
  $0 -f -s 100 -m 3 exp0_baseline

  # Evaluate specific checkpoint
  $0 -k models/exp0_baseline/epoch_10/ exp0_baseline

  # Run only BLIMP evaluation
  $0 -t blimp exp0_baseline
EOF
}

# GPU management functions
GPU_MONITORING=true
source "$SCRIPT_DIR/gpu_monitor.sh" 2>/dev/null || {
    log WARN "Cannot source gpu_monitor.sh - GPU management disabled"
    GPU_MONITORING=false
}

# Check GPU availability
check_gpu_availability() {
    local gpu_ids="$1"
    
    if [[ "$GPU_MONITORING" == "false" ]]; then
        log WARN "GPU monitoring disabled, skipping availability check"
        return 0
    fi
    
    log INFO "Checking GPU availability for GPUs: $gpu_ids"
    
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        local status
        status=$("$SCRIPT_DIR/gpu_monitor.sh" check "$gpu_id" 2>/dev/null || echo "UNKNOWN")
        
        case "$status" in
            AVAILABLE|LIMITED)
                log INFO "GPU $gpu_id: $status"
                ;;
            OCCUPIED)
                log ERROR "GPU $gpu_id is occupied. Use different GPU or wait."
                return 1
                ;;
            *)
                log WARN "GPU $gpu_id status unknown: $status"
                ;;
        esac
    done
    
    return 0
}

# Lock GPUs
lock_gpus() {
    local gpu_ids="$1"
    
    if [[ "$GPU_MONITORING" == "false" ]]; then
        log WARN "GPU monitoring disabled, skipping GPU locking"
        return 0
    fi
    
    log INFO "Locking GPUs: $gpu_ids"
    
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        "$SCRIPT_DIR/gpu_monitor.sh" lock "$gpu_id" || {
            log ERROR "Failed to lock GPU $gpu_id"
            return 1
        }
    done
}

# Unlock GPUs
unlock_gpus() {
    local gpu_ids="$1"
    
    if [[ "$GPU_MONITORING" == "false" ]]; then
        return 0
    fi
    
    log INFO "Unlocking GPUs: $gpu_ids"
    
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        "$SCRIPT_DIR/gpu_monitor.sh" unlock "$gpu_id" || {
            log WARN "Failed to unlock GPU $gpu_id"
        }
    done
}

# Create evaluation config
create_eval_config() {
    local experiment="$1"
    local config_template="$2"
    local output_config="$3"
    local checkpoint_path="$4"
    local tasks="$5"
    local max_samples="$6"
    local max_checkpoints="$7"
    local output_dir="$8"
    
    log INFO "Creating evaluation config: $output_config"
    
    # Copy base config
    cp "$config_template" "$output_config"
    
    # Create temporary Python script to modify config
    local modify_script="/tmp/modify_eval_config_$$.py"
    cat > "$modify_script" <<EOF
import yaml
import sys

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Ensure evaluation section exists
if 'evaluation' not in config:
    config = {'evaluation': config}

eval_config = config['evaluation']

# Update paths
eval_config['model_checkpoint_dir'] = "$checkpoint_path"
eval_config['tokenizer_path'] = "tokenizers/$experiment/"

if "$output_dir":
    eval_config['output_dir'] = "$output_dir"
else:
    eval_config['output_dir'] = f"evaluation/results/$experiment/"

# Update task selection
if "$tasks" and "$tasks" != "all":
    tasks = "$tasks".split(',')
    eval_config['run_perplexity'] = 'perplexity' in tasks
    eval_config['run_blimp'] = 'blimp' in tasks  
    eval_config['run_null_subject'] = 'null_subject' in tasks

# Update testing parameters
if "$max_samples":
    eval_config['max_samples'] = int("$max_samples")

if "$max_checkpoints":
    eval_config['max_checkpoints'] = int("$max_checkpoints")

# Save modified config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)
EOF
    
    python3 "$modify_script" "$output_config"
    rm "$modify_script"
    
    log INFO "Configuration updated for experiment: $experiment"
}

# Run evaluation
run_evaluation() {
    local config_file="$1"
    local gpu_ids="$2"
    local checkpoint_path="$3"
    local verbose="$4"
    
    log INFO "Starting evaluation with config: $config_file"
    
    # Set up environment
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Build command
    local cmd_args=("--config" "$config_file")
    
    if [[ -n "$checkpoint_path" ]]; then
        cmd_args+=("--checkpoint" "$checkpoint_path")
    fi
    
    if [[ "$verbose" == "true" ]]; then
        cmd_args+=("--debug")
    fi
    
    # Use Singularity container if available
    local container_cmd=""
    if [[ -f "$PROJECT_DIR/singularity/training.sif" ]]; then
        container_cmd="singularity exec --nv --bind $PROJECT_DIR:/workspace $PROJECT_DIR/singularity/training.sif"
        log INFO "Using Singularity container for evaluation"
    else
        log WARN "Singularity container not found, running directly"
    fi
    
    # Run evaluation
    cd "$PROJECT_DIR"
    
    if [[ -n "$container_cmd" ]]; then
        $container_cmd python evaluation/evaluation_runner.py "${cmd_args[@]}"
    else
        python evaluation/evaluation_runner.py "${cmd_args[@]}"
    fi
}

# Export results for R
export_results() {
    local output_dir="$1"
    local experiment="$2"
    
    log INFO "Exporting results for R analysis..."
    
    local results_file="$output_dir/evaluation_results.jsonl"
    if [[ ! -f "$results_file" ]]; then
        log ERROR "Results file not found: $results_file"
        return 1
    fi
    
    # Use Python to export results
    local export_script="/tmp/export_results_$$.py"
    cat > "$export_script" <<EOF
import sys
import os
sys.path.insert(0, '$PROJECT_DIR')

from evaluation.result_aggregator import ResultAggregator

try:
    aggregator = ResultAggregator('$output_dir')
    files = aggregator.export_for_r('$results_file', '${experiment}_evaluation')
    
    print("Exported files:")
    for data_type, filepath in files.items():
        print(f"  {data_type}: {filepath}")
        
except Exception as e:
    print(f"Error exporting results: {e}")
    sys.exit(1)
EOF
    
    if [[ -f "$PROJECT_DIR/singularity/training.sif" ]]; then
        cd "$PROJECT_DIR"
        singularity exec --bind "$PROJECT_DIR:/workspace" "$PROJECT_DIR/singularity/training.sif" python3 "$export_script"
    else
        cd "$PROJECT_DIR"
        python3 "$export_script"
    fi
    
    rm "$export_script"
    
    log SUCCESS "Results exported for R analysis"
}

# Main function
main() {
    local gpu_ids="$DEFAULT_GPUS"
    local config="$DEFAULT_CONFIG"
    local experiment=""
    local checkpoint_path=""
    local tasks="all"
    local max_samples=""
    local max_checkpoints=""
    local output_dir=""
    local lock_gpus_flag=false
    local unlock_gpus_flag=false
    local fast_mode=false
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -g|--gpus)
                gpu_ids="$2"
                shift 2
                ;;
            -c|--config)
                config="$2"
                shift 2
                ;;
            -e|--experiment)
                experiment="$2"
                shift 2
                ;;
            -k|--checkpoint)
                checkpoint_path="$2"
                shift 2
                ;;
            -t|--tasks)
                tasks="$2"
                shift 2
                ;;
            -s|--samples)
                max_samples="$2"
                shift 2
                ;;
            -m|--max-checkpoints)
                max_checkpoints="$2"
                shift 2
                ;;
            -o|--output)
                output_dir="$2"
                shift 2
                ;;
            -l|--lock-gpus)
                lock_gpus_flag=true
                shift
                ;;
            -u|--unlock-gpus)
                unlock_gpus_flag=true
                shift
                ;;
            -f|--fast)
                fast_mode=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                log ERROR "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$experiment" ]]; then
                    experiment="$1"
                else
                    log ERROR "Unexpected argument: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$experiment" ]] && [[ -z "$checkpoint_path" ]]; then
        log ERROR "Must specify either experiment name or checkpoint path"
        show_usage
        exit 1
    fi
    
    # Set experiment name from checkpoint if not provided
    if [[ -z "$experiment" ]] && [[ -n "$checkpoint_path" ]]; then
        experiment=$(basename "$(dirname "$checkpoint_path")")
    fi
    
    # Handle fast mode
    if [[ "$fast_mode" == "true" ]]; then
        max_samples="${max_samples:-100}"
        max_checkpoints="${max_checkpoints:-3}"
        log INFO "Fast mode enabled: max_samples=$max_samples, max_checkpoints=$max_checkpoints"
    fi
    
    # Set checkpoint path if not provided
    if [[ -z "$checkpoint_path" ]]; then
        checkpoint_path="models/$experiment/"
    fi
    
    # Validate paths
    if [[ ! -f "$config" ]]; then
        log ERROR "Config file not found: $config"
        exit 1
    fi
    
    if [[ ! -d "$checkpoint_path" ]]; then
        log ERROR "Checkpoint directory not found: $checkpoint_path"
        exit 1
    fi
    
    log INFO "Starting evaluation for experiment: $experiment"
    log INFO "Checkpoint path: $checkpoint_path"
    log INFO "Using GPUs: $gpu_ids"
    log INFO "Tasks: $tasks"
    
    # Check GPU availability
    if ! check_gpu_availability "$gpu_ids"; then
        log ERROR "GPU availability check failed"
        exit 1
    fi
    
    # Lock GPUs if requested
    if [[ "$lock_gpus_flag" == "true" ]]; then
        if ! lock_gpus "$gpu_ids"; then
            log ERROR "Failed to lock GPUs"
            exit 1
        fi
        trap "unlock_gpus '$gpu_ids'" EXIT
    fi
    
    # Create temporary config
    local temp_config="/tmp/eval_config_${experiment}_$$.yaml"
    create_eval_config "$experiment" "$config" "$temp_config" "$checkpoint_path" "$tasks" "$max_samples" "$max_checkpoints" "$output_dir"
    
    # Run evaluation
    if run_evaluation "$temp_config" "$gpu_ids" "$checkpoint_path" "$verbose"; then
        log SUCCESS "Evaluation completed successfully"
        
        # Export results
        local final_output_dir
        final_output_dir=$(python3 -c "import yaml; config=yaml.safe_load(open('$temp_config')); print(config['evaluation']['output_dir'])")
        export_results "$final_output_dir" "$experiment"
        
    else
        log ERROR "Evaluation failed"
        exit 1
    fi
    
    # Cleanup
    rm -f "$temp_config"
    
    # Unlock GPUs if requested
    if [[ "$unlock_gpus_flag" == "true" ]]; then
        unlock_gpus "$gpu_ids"
    fi
    
    log SUCCESS "Evaluation pipeline completed for $experiment"
}

# Handle script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi