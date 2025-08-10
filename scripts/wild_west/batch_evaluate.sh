#!/usr/bin/env bash
# Batch Evaluation Runner
# Run evaluation on multiple experiments sequentially or in parallel

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Logging
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

# Usage
show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [experiment1] [experiment2] ...

Run evaluation on multiple experiments.

Options:
  -f, --experiments-file <file>  File containing experiment names (one per line)
  -g, --gpus <ids>              GPU IDs to use (default: round-robin available GPUs)
  -p, --parallel <n>            Run n evaluations in parallel (default: 1)
  -t, --tasks <tasks>           Tasks to run (default: all)
  -s, --samples <num>           Max samples per task (for testing)
  -m, --max-checkpoints <n>     Max checkpoints per experiment
  -o, --output-base <dir>       Base output directory (default: evaluation/results/)
  --fast                        Fast evaluation mode
  --dry-run                     Show what would be run without executing
  -v, --verbose                 Verbose output
  -h, --help                    Show this help

Examples:
  # Evaluate all baseline experiments
  $0 exp0_baseline exp1_remove_expletives exp2_poor_determiners

  # Parallel evaluation using 2 processes
  $0 -p 2 -g 0,1 exp0_baseline exp1_remove_expletives

  # Fast evaluation from file
  $0 --fast -f experiments.txt

  # Dry run to see what would be executed
  $0 --dry-run exp0_baseline exp1_remove_expletives
EOF
}

# Find available experiments
find_experiments() {
    local models_dir="$PROJECT_DIR/models"
    
    if [[ ! -d "$models_dir" ]]; then
        log ERROR "Models directory not found: $models_dir"
        return 1
    fi
    
    log INFO "Available experiments:"
    for exp_dir in "$models_dir"/*; do
        if [[ -d "$exp_dir" ]]; then
            local exp_name=$(basename "$exp_dir")
            echo "  $exp_name"
        fi
    done
}

# Get available GPUs
get_available_gpus() {
    local gpu_monitor="$SCRIPT_DIR/gpu_monitor.sh"
    
    if [[ -f "$gpu_monitor" ]]; then
        "$gpu_monitor" available 2>/dev/null | tr '\n' ',' | sed 's/,$//' || echo "0,1"
    else
        echo "0,1"  # Default fallback
    fi
}

# Assign GPU to process
assign_gpu() {
    local process_id="$1"
    local available_gpus="$2"
    
    IFS=',' read -ra GPU_ARRAY <<< "$available_gpus"
    local gpu_count=${#GPU_ARRAY[@]}
    
    if [[ $gpu_count -eq 0 ]]; then
        echo "0"  # Fallback
        return
    fi
    
    local gpu_index=$((process_id % gpu_count))
    echo "${GPU_ARRAY[$gpu_index]}"
}

# Run single evaluation
run_single_evaluation() {
    local experiment="$1"
    local gpu_id="$2"
    local tasks="$3"
    local max_samples="$4"
    local max_checkpoints="$5"
    local output_base="$6"
    local fast_mode="$7"
    local verbose="$8"
    local process_id="$9"
    
    local log_file="/tmp/eval_${experiment}_gpu${gpu_id}_$$.log"
    
    log INFO "Starting evaluation of $experiment on GPU $gpu_id (process $process_id)"
    
    # Build command
    local cmd=("$SCRIPT_DIR/run_evaluation.sh")
    cmd+=("-g" "$gpu_id")
    cmd+=("-e" "$experiment")
    cmd+=("-l")  # Lock GPU
    cmd+=("-u")  # Unlock GPU after
    
    if [[ -n "$tasks" ]]; then
        cmd+=("-t" "$tasks")
    fi
    
    if [[ -n "$max_samples" ]]; then
        cmd+=("-s" "$max_samples")
    fi
    
    if [[ -n "$max_checkpoints" ]]; then
        cmd+=("-m" "$max_checkpoints")
    fi
    
    if [[ -n "$output_base" ]]; then
        cmd+=("-o" "$output_base/$experiment")
    fi
    
    if [[ "$fast_mode" == "true" ]]; then
        cmd+=("-f")
    fi
    
    if [[ "$verbose" == "true" ]]; then
        cmd+=("-v")
    fi
    
    # Run command
    if "${cmd[@]}" &> "$log_file"; then
        log SUCCESS "Completed evaluation of $experiment on GPU $gpu_id"
        rm -f "$log_file"
        return 0
    else
        log ERROR "Failed evaluation of $experiment on GPU $gpu_id (see $log_file)"
        return 1
    fi
}

# Create comparison report
create_comparison_report() {
    local experiments=("$@")
    local output_dir="$PROJECT_DIR/evaluation/results"
    local report_file="$output_dir/cross_experiment_comparison.md"
    
    log INFO "Creating comparison report: $report_file"
    
    mkdir -p "$output_dir"
    
    cat > "$report_file" <<EOF
# Cross-Experiment Evaluation Comparison

Generated: $(date)

## Experiments Evaluated

EOF
    
    for exp in "${experiments[@]}"; do
        echo "- $exp" >> "$report_file"
    done
    
    cat >> "$report_file" <<EOF

## Results Summary

| Experiment | Final Perplexity | BLIMP Accuracy | Null-Subj Overt Pref | Status |
|------------|------------------|----------------|----------------------|--------|
EOF
    
    # Extract results for each experiment
    for exp in "${experiments[@]}"; do
        local results_file="$output_dir/$exp/evaluation_results.jsonl"
        if [[ -f "$results_file" ]]; then
            local status="✓ Complete"
            # Extract final metrics using Python
            local metrics
            metrics=$(python3 - <<EOF 2>/dev/null || echo "N/A N/A N/A"
import json
import sys

try:
    with open('$results_file', 'r') as f:
        lines = f.readlines()
        if lines:
            last_result = json.loads(lines[-1])
            
            ppl = last_result.get('perplexity', {}).get('perplexity', 'N/A')
            blimp = last_result.get('blimp', {}).get('overall_accuracy', 'N/A')
            null_subj = last_result.get('null_subject', {}).get('overt_preference_rate', 'N/A')
            
            print(f"{ppl} {blimp} {null_subj}")
        else:
            print("N/A N/A N/A")
except:
    print("N/A N/A N/A")
EOF
)
            read ppl blimp null_subj <<< "$metrics"
        else
            local status="✗ Missing"
            local ppl="N/A"
            local blimp="N/A" 
            local null_subj="N/A"
        fi
        
        echo "| $exp | $ppl | $blimp | $null_subj | $status |" >> "$report_file"
    done
    
    cat >> "$report_file" <<EOF

## Analysis Notes

Add your analysis notes here after reviewing the results.

EOF
    
    log SUCCESS "Comparison report created: $report_file"
}

# Main function
main() {
    local experiments=()
    local experiments_file=""
    local gpus=""
    local parallel=1
    local tasks=""
    local max_samples=""
    local max_checkpoints=""
    local output_base=""
    local fast_mode=false
    local dry_run=false
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--experiments-file)
                experiments_file="$2"
                shift 2
                ;;
            -g|--gpus)
                gpus="$2"
                shift 2
                ;;
            -p|--parallel)
                parallel="$2"
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
            -o|--output-base)
                output_base="$2"
                shift 2
                ;;
            --fast)
                fast_mode=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --list-experiments)
                find_experiments
                exit 0
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
                experiments+=("$1")
                shift
                ;;
        esac
    done
    
    # Load experiments from file if specified
    if [[ -n "$experiments_file" ]]; then
        if [[ ! -f "$experiments_file" ]]; then
            log ERROR "Experiments file not found: $experiments_file"
            exit 1
        fi
        
        while IFS= read -r line; do
            line=$(echo "$line" | xargs)  # Trim whitespace
            if [[ -n "$line" ]] && [[ ! "$line" =~ ^# ]]; then
                experiments+=("$line")
            fi
        done < "$experiments_file"
    fi
    
    # Validate we have experiments to run
    if [[ ${#experiments[@]} -eq 0 ]]; then
        log ERROR "No experiments specified"
        log INFO "Available experiments:"
        find_experiments
        exit 1
    fi
    
    # Get available GPUs if not specified
    if [[ -z "$gpus" ]]; then
        gpus=$(get_available_gpus)
        log INFO "Auto-detected available GPUs: $gpus"
    fi
    
    # Validate parallel setting
    if [[ $parallel -lt 1 ]]; then
        log ERROR "Parallel must be >= 1"
        exit 1
    fi
    
    log INFO "Batch evaluation settings:"
    log INFO "  Experiments: ${experiments[*]}"
    log INFO "  GPUs: $gpus"
    log INFO "  Parallel processes: $parallel"
    log INFO "  Tasks: ${tasks:-all}"
    log INFO "  Fast mode: $fast_mode"
    
    # Dry run
    if [[ "$dry_run" == "true" ]]; then
        log INFO "DRY RUN - Commands that would be executed:"
        for i in "${!experiments[@]}"; do
            local exp="${experiments[$i]}"
            local gpu_id
            gpu_id=$(assign_gpu "$((i % parallel))" "$gpus")
            echo "  GPU $gpu_id: $exp"
        done
        exit 0
    fi
    
    # Run evaluations
    local pids=()
    local failed_experiments=()
    local successful_experiments=()
    
    for i in "${!experiments[@]}"; do
        local exp="${experiments[$i]}"
        
        # Wait if we've reached parallel limit
        while [[ ${#pids[@]} -ge $parallel ]]; do
            local new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                else
                    wait "$pid"
                    if [[ $? -eq 0 ]]; then
                        log INFO "Background evaluation process completed successfully"
                    else
                        log ERROR "Background evaluation process failed"
                    fi
                fi
            done
            pids=("${new_pids[@]}")
            
            if [[ ${#pids[@]} -ge $parallel ]]; then
                sleep 5
            fi
        done
        
        # Assign GPU (round-robin)
        local gpu_id
        gpu_id=$(assign_gpu "$((i % parallel))" "$gpus")
        
        # Start evaluation in background if parallel > 1
        if [[ $parallel -gt 1 ]]; then
            run_single_evaluation "$exp" "$gpu_id" "$tasks" "$max_samples" "$max_checkpoints" "$output_base" "$fast_mode" "$verbose" "$i" &
            pids+=($!)
            log INFO "Started background evaluation of $exp on GPU $gpu_id (PID: $!)"
        else
            # Run synchronously
            if run_single_evaluation "$exp" "$gpu_id" "$tasks" "$max_samples" "$max_checkpoints" "$output_base" "$fast_mode" "$verbose" "$i"; then
                successful_experiments+=("$exp")
            else
                failed_experiments+=("$exp")
            fi
        fi
    done
    
    # Wait for remaining background processes
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            log INFO "Background evaluation completed successfully (PID: $pid)"
        else
            log ERROR "Background evaluation failed (PID: $pid)"
        fi
    done
    
    # Create comparison report
    if [[ ${#successful_experiments[@]} -gt 0 ]] || [[ $parallel -gt 1 ]]; then
        create_comparison_report "${experiments[@]}"
    fi
    
    # Final summary
    log INFO "Batch evaluation completed"
    log INFO "  Total experiments: ${#experiments[@]}"
    
    if [[ $parallel -eq 1 ]]; then
        log INFO "  Successful: ${#successful_experiments[@]}"
        log INFO "  Failed: ${#failed_experiments[@]}"
        
        if [[ ${#failed_experiments[@]} -gt 0 ]]; then
            log ERROR "Failed experiments: ${failed_experiments[*]}"
            exit 1
        fi
    else
        log INFO "  Check individual logs for parallel execution results"
    fi
    
    log SUCCESS "All evaluations completed!"
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi