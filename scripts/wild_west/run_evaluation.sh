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
  -g, --gpus <ids>           GPU IDs to use via CUDA_VISIBLE_DEVICES (default: $DEFAULT_GPUS)
  -c, --config <config>      Evaluation config file (default: $DEFAULT_CONFIG)
  -e, --experiment <exp>     Experiment to evaluate (e.g., exp0_baseline)
  -k, --checkpoint <path>    Specific checkpoint to evaluate
  -t, --tasks <tasks>        Tasks to run: perplexity,blimp,null_subject (default: all)
  -s, --samples <num>        Max samples per task (for testing)
  -m, --max-checkpoints <n>  Max checkpoints to evaluate
  -o, --output <dir>         Output directory override
  -w, --workers <num>        Number of parallel threads (default: 4)
  -P, --parallel             Use parallel evaluation with threading
  -l, --lock-gpus            Lock GPUs before running
  -u, --unlock-gpus          Unlock GPUs after running
  -f, --fast                 Fast evaluation (reduced samples)
  -r, --force-rerun          Re-evaluate checkpoints even if results exist
  -v, --verbose              Verbose output
  -h, --help                 Show this help

Examples:
  # Evaluate exp0_baseline experiment (default: 4 threads, GPU 1)
  $0 exp0_baseline

  # Use original serial evaluation (no parallelization, no timeout)
  $0 -S exp0_baseline

  # Run on GPU 0 instead of default GPU 1
  $0 -g 0 exp0_baseline

  # Run with 8 threads on GPU 2
  $0 -w 8 -g 2 exp0_baseline

  # Serial evaluation on GPU 0 (exactly as before this conversation)
  $0 -S -g 0 exp0_baseline

  # Evaluate with specific config
  $0 -c configs/eval_exp1.yaml exp1_remove_expletives

  # Fast evaluation for testing
  $0 -f -s 100 -m 3 exp0_baseline

  # Force re-evaluation of existing checkpoints
  $0 -r exp0_baseline

  # Evaluate specific checkpoint
  $0 -k models/exp0_baseline/epoch_10/ exp0_baseline

  # Run only BLIMP evaluation
  $0 -t blimp exp0_baseline
EOF
}

# GPU management functions - simplified without gpu_monitor
GPU_MONITORING=true

# Check GPU availability using nvidia-smi
check_gpu_availability() {
    local gpu_ids="$1"
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log WARN "nvidia-smi not found, skipping GPU availability check"
        return 0
    fi
    
    log INFO "Checking GPU availability for GPUs: $gpu_ids"
    
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        # Check if GPU ID is valid (0-3 for this system)
        if [[ ! "$gpu_id" =~ ^[0-3]$ ]]; then
            log ERROR "Invalid GPU ID: $gpu_id (must be 0-3)"
            return 1
        fi
        
        # Check GPU memory usage
        local mem_info
        mem_info=$(nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "")
        
        if [[ -z "$mem_info" ]]; then
            log WARN "Cannot query GPU $gpu_id"
            continue
        fi
        
        local mem_free mem_used
        IFS=', ' read -r mem_free mem_used <<< "$mem_info"
        
        if [[ $mem_free -lt 5000 ]]; then
            log WARN "GPU $gpu_id has limited memory available: ${mem_free}MB free"
        else
            log INFO "GPU $gpu_id: ${mem_free}MB free, ${mem_used}MB used"
        fi
    done
    
    return 0
}

# Lock GPUs - simplified (no actual locking)
lock_gpus() {
    local gpu_ids="$1"
    log INFO "GPU locking disabled - using GPUs: $gpu_ids"
    return 0
}

# Unlock GPUs - simplified (no actual unlocking)
unlock_gpus() {
    local gpu_ids="$1"
    log INFO "GPU unlocking disabled - released GPUs: $gpu_ids"
    return 0
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
# Keep original tokenizer_path from config file (don't override it)

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

# Run evaluation with proper process management and cleanup
run_evaluation() {
    local config_file="$1"
    local gpu_ids="$2"
    local checkpoint_path="$3"
    local verbose="$4"
    local force_rerun="$5"
    local parallel_workers="$6"
    local use_serial="$7"
    
    log INFO "Starting evaluation with config: $config_file"
    
    # Build command
    local cmd_args=("--config" "$config_file")
    
    # Only pass --checkpoint if it's a specific checkpoint directory (not experiment directory)
    if [[ -n "$checkpoint_path" ]] && [[ $(basename "$checkpoint_path") =~ ^checkpoint- ]]; then
        cmd_args+=("--checkpoint" "$checkpoint_path")
    fi
    
    if [[ "$verbose" == "true" ]]; then
        cmd_args+=("--debug")
    fi
    
    if [[ "$force_rerun" == "true" ]]; then
        cmd_args+=("--force-rerun")
    fi
    
    # Environment variables for containers
    local env_flags=(
        --env "CUDA_VISIBLE_DEVICES=${gpu_ids}"
        --env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
        --env "TORCH_CUDA_MEMORY_FRACTION=${TORCH_CUDA_MEMORY_FRACTION:-0.95}"
        --env "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
        --env "NCCL_ASYNC_ERROR_HANDLING=1"
        --env "WANDB_MODE=${WANDB_MODE:-offline}"
        --env "OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}"
        --env "MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}"
    )
    
    # Process cleanup function
    local CHILD_PID=""
    local CHILD_PGID=""
    
    cleanup_processes() {
        if [[ -n "$CHILD_PGID" ]]; then
            log INFO "Cleaning up process group: $CHILD_PGID"
            kill -TERM -"$CHILD_PGID" 2>/dev/null || true
            sleep 2
            kill -KILL -"$CHILD_PGID" 2>/dev/null || true
        fi
    }
    
    # Set up signal trapping for process cleanup
    trap cleanup_processes INT TERM
    
    cd "$PROJECT_DIR"
    
    # Use Singularity container if available
    if [[ -f "$PROJECT_DIR/singularity/training.sif" ]]; then
        log INFO "Using Singularity container for evaluation"
        
        # Pick container runtime
        local runtime="singularity"
        if command -v apptainer >/dev/null 2>&1; then
            runtime="apptainer"
        fi
        
        # Convert absolute paths to container paths for command args
        local container_cmd_args=()
        for arg in "${cmd_args[@]}"; do
            if [[ "$arg" == "$PROJECT_DIR"* ]]; then
                # Replace project dir path with /workspace
                container_arg="/workspace${arg#$PROJECT_DIR}"
                container_cmd_args+=("$container_arg")
            else
                container_cmd_args+=("$arg")
            fi
        done
        
        # Launch in own session
        set +e
        if [[ "$use_serial" == "true" ]]; then
            setsid "$runtime" exec --nv --pid --contain --cleanenv \
                "${env_flags[@]}" \
                --bind "$PROJECT_DIR":/workspace \
                "$PROJECT_DIR/singularity/training.sif" \
                bash -lc "set -euo pipefail; cd /workspace; exec python -m evaluation.evaluation_runner ${container_cmd_args[*]}" &
        else
            setsid "$runtime" exec --nv --pid --contain --cleanenv \
                "${env_flags[@]}" \
                --bind "$PROJECT_DIR":/workspace \
                "$PROJECT_DIR/singularity/training.sif" \
                bash -lc "set -euo pipefail; cd /workspace; exec python -m evaluation.parallel_evaluation_runner ${container_cmd_args[*]} --parallel-workers $parallel_workers" &
        fi
        
        CHILD_PID=$!
        CHILD_PGID=$(ps -o pgid= "$CHILD_PID" 2>/dev/null | tr -d ' ' || echo "")
        
        log INFO "Evaluation started: PID=$CHILD_PID PGID=$CHILD_PGID"
        
        wait "$CHILD_PID"
        local exit_code=$?
        set -e
        
    else
        log WARN "Singularity container not found, running directly"
        
        # Set environment for direct execution
        export CUDA_VISIBLE_DEVICES="$gpu_ids"
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb=512}"
        export TORCH_CUDA_MEMORY_FRACTION="${TORCH_CUDA_MEMORY_FRACTION:-0.95}"
        export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
        export NCCL_ASYNC_ERROR_HANDLING=1
        export WANDB_MODE="${WANDB_MODE:-offline}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
        
        # Launch evaluation
        set +e
        if [[ "$use_serial" == "true" ]]; then
            setsid python -m evaluation.evaluation_runner "${cmd_args[@]}" &
        else
            setsid python -m evaluation.parallel_evaluation_runner "${cmd_args[@]}" --parallel-workers $parallel_workers &
        fi
        
        CHILD_PID=$!
        CHILD_PGID=$(ps -o pgid= "$CHILD_PID" 2>/dev/null | tr -d ' ' || echo "")
        
        log INFO "Evaluation started: PID=$CHILD_PID PGID=$CHILD_PGID"
        
        wait "$CHILD_PID"
        local exit_code=$?
        set -e
    fi
    
    # Clean up signal handlers
    trap - INT TERM
    
    if [[ $exit_code -ne 0 ]]; then
        log ERROR "Evaluation failed with exit code: $exit_code"
        return $exit_code
    fi
    
    log INFO "Evaluation completed successfully"
    return 0
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
    local parallel_workers="4"
    local use_serial=false
    local lock_gpus_flag=false
    local unlock_gpus_flag=false
    local fast_mode=false
    local force_rerun=false
    local verbose=false
    
    # Global cleanup function
    cleanup_main() {
        if [[ "$lock_gpus_flag" == "true" ]] || [[ "$unlock_gpus_flag" == "true" ]]; then
            unlock_gpus "$gpu_ids"
        fi
        # Clean up temp config if it exists  
        [[ -n "${temp_config:-}" ]] && rm -f "$temp_config" 2>/dev/null || true
    }
    
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
            -w|--workers)
                parallel_workers="$2"
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
            -r|--force-rerun)
                force_rerun=true
                shift
                ;;
            -S|--serial)
                use_serial=true
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
    
    # Auto-detect config file if not specified and using default
    if [[ "$config" == "$DEFAULT_CONFIG" ]] && [[ -n "$experiment" ]]; then
        # Try to find a matching eval config file
        local auto_config=""
        
        # Case 1: experiment is like "exp0_baseline" -> look for "configs/eval_exp0_baseline.yaml"
        if [[ -f "configs/eval_${experiment}.yaml" ]]; then
            auto_config="configs/eval_${experiment}.yaml"
        # Case 2: experiment is like "eval_exp0_baseline" -> look for "configs/eval_exp0_baseline.yaml"  
        elif [[ "$experiment" =~ ^eval_ ]] && [[ -f "configs/${experiment}.yaml" ]]; then
            auto_config="configs/${experiment}.yaml"
        fi
        
        if [[ -n "$auto_config" ]]; then
            config="$auto_config"
            log INFO "Auto-detected config file: $config"
        else
            log WARN "No matching config found for experiment '$experiment', using default: $config"
        fi
    fi
    
    # Handle fast mode
    if [[ "$fast_mode" == "true" ]]; then
        max_samples="${max_samples:-100}"
        max_checkpoints="${max_checkpoints:-3}"
        log INFO "Fast mode enabled: max_samples=$max_samples, max_checkpoints=$max_checkpoints"
    fi
    
    # Validate config file exists first
    if [[ ! -f "$config" ]]; then
        log ERROR "Config file not found: $config"
        exit 1
    fi
    
    # Set checkpoint path if not provided - read from config file
    if [[ -z "$checkpoint_path" ]]; then
        checkpoint_path=$(python3 -c "
import yaml
with open('$config', 'r') as f:
    config = yaml.safe_load(f)
if 'evaluation' in config:
    print(config['evaluation']['model_checkpoint_dir'])
else:
    print(config['model_checkpoint_dir'])
" 2>/dev/null || echo "models/$experiment/")
    fi
    
    # Validate checkpoint path
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
    
    # Create temporary config in project directory (accessible to container)
    local temp_config="$PROJECT_DIR/.tmp_eval_config_${experiment}_$$.yaml"
    
    # Lock GPUs if requested
    if [[ "$lock_gpus_flag" == "true" ]]; then
        if ! lock_gpus "$gpu_ids"; then
            log ERROR "Failed to lock GPUs"
            exit 1
        fi
        trap cleanup_main INT TERM EXIT
    fi
    
    # Create the evaluation config
    create_eval_config "$experiment" "$config" "$temp_config" "$checkpoint_path" "$tasks" "$max_samples" "$max_checkpoints" "$output_dir"
    
    # Run evaluation
    if run_evaluation "$temp_config" "$gpu_ids" "$checkpoint_path" "$verbose" "$force_rerun" "$parallel_workers" "$use_serial"; then
        log SUCCESS "Evaluation completed successfully"
        
        # Export results
        local final_output_dir
        final_output_dir=$(python3 -c "import yaml; config=yaml.safe_load(open('$temp_config')); print(config['evaluation']['output_dir'])")
        export_results "$final_output_dir" "$experiment"
        
    else
        log ERROR "Evaluation failed"
        exit 1
    fi
    
    # Cleanup temporary files
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