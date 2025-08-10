#!/usr/bin/env bash
# Evaluation Monitoring Script
# Monitor running evaluations and show progress

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
Usage: $0 [OPTIONS] [COMMAND]

Monitor evaluation processes and progress.

Commands:
  status        Show current evaluation status (default)
  progress      Show detailed progress for running evaluations
  results       Show latest results summary
  watch         Continuously monitor evaluations
  cleanup       Clean up old evaluation logs and temp files
  kill          Kill running evaluation processes

Options:
  -e, --experiment <exp>    Focus on specific experiment
  -i, --interval <sec>      Update interval for watch mode (default: 10)
  -v, --verbose             Verbose output
  -h, --help               Show this help

Examples:
  # Show current status
  $0 status

  # Monitor specific experiment
  $0 -e exp0_baseline progress

  # Watch all evaluations
  $0 watch

  # Show latest results
  $0 results
EOF
}

# Find running evaluation processes
find_evaluation_processes() {
    local experiment_filter="$1"
    
    # Find Python processes running evaluation_runner.py
    local pids
    pids=$(pgrep -f "evaluation_runner.py" 2>/dev/null || echo "")
    
    if [[ -z "$pids" ]]; then
        return 0
    fi
    
    echo "$pids" | while read -r pid; do
        if [[ -n "$pid" ]]; then
            local cmdline
            cmdline=$(ps -p "$pid" -o cmd --no-headers 2>/dev/null || echo "")
            
            if [[ -n "$experiment_filter" ]]; then
                if echo "$cmdline" | grep -q "$experiment_filter"; then
                    echo "$pid:$cmdline"
                fi
            else
                echo "$pid:$cmdline"
            fi
        fi
    done
}

# Get GPU usage for evaluation processes
get_gpu_usage() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    
    # Get GPU processes
    nvidia-smi --query-compute-apps=pid,name,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null | \
    while IFS=', ' read -r pid name gpu_uuid mem_used; do
        # Check if it's an evaluation process
        if ps -p "$pid" -o cmd --no-headers 2>/dev/null | grep -q "evaluation_runner.py"; then
            local gpu_id
            gpu_id=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | grep "$gpu_uuid" | cut -d',' -f1 || echo "?")
            echo "GPU $gpu_id: PID $pid using ${mem_used}MB"
        fi
    done
}

# Show evaluation status
show_status() {
    local experiment_filter="$1"
    
    echo "=== Evaluation Status ==="
    echo
    
    # Running processes
    local running_procs
    running_procs=$(find_evaluation_processes "$experiment_filter")
    
    if [[ -n "$running_procs" ]]; then
        echo "Running Evaluations:"
        echo "$running_procs" | while IFS=':' read -r pid cmdline; do
            local start_time
            start_time=$(ps -p "$pid" -o lstart --no-headers 2>/dev/null || echo "Unknown")
            echo "  PID $pid (started: $start_time)"
            echo "    Command: $cmdline"
        done
        echo
        
        # GPU usage
        echo "GPU Usage:"
        local gpu_usage
        gpu_usage=$(get_gpu_usage)
        if [[ -n "$gpu_usage" ]]; then
            echo "$gpu_usage" | while read -r line; do
                echo "  $line"
            done
        else
            echo "  No GPU usage detected"
        fi
        echo
    else
        echo "No running evaluations found."
        echo
    fi
    
    # Recent results
    echo "Recent Results:"
    local results_dir="$PROJECT_DIR/evaluation/results"
    if [[ -d "$results_dir" ]]; then
        find "$results_dir" -name "evaluation_results.jsonl" -mtime -1 2>/dev/null | \
        while read -r results_file; do
            local exp_name
            exp_name=$(basename "$(dirname "$results_file")")
            local mod_time
            mod_time=$(stat -c %y "$results_file" 2>/dev/null || stat -f "%Sm" "$results_file" 2>/dev/null || echo "Unknown")
            local line_count
            line_count=$(wc -l < "$results_file" 2>/dev/null || echo "0")
            echo "  $exp_name: $line_count checkpoints evaluated (modified: $mod_time)"
        done
    else
        echo "  No results directory found"
    fi
}

# Show detailed progress
show_progress() {
    local experiment_filter="$1"
    local verbose="$2"
    
    echo "=== Evaluation Progress ==="
    echo
    
    # For each experiment directory
    local results_dir="$PROJECT_DIR/evaluation/results"
    if [[ ! -d "$results_dir" ]]; then
        echo "No results directory found."
        return 0
    fi
    
    for exp_dir in "$results_dir"/*; do
        if [[ ! -d "$exp_dir" ]]; then
            continue
        fi
        
        local exp_name
        exp_name=$(basename "$exp_dir")
        
        # Skip if filtering and doesn't match
        if [[ -n "$experiment_filter" ]] && [[ "$exp_name" != *"$experiment_filter"* ]]; then
            continue
        fi
        
        echo "Experiment: $exp_name"
        
        # Check for results file
        local results_file="$exp_dir/evaluation_results.jsonl"
        if [[ -f "$results_file" ]]; then
            local total_checkpoints
            total_checkpoints=$(wc -l < "$results_file" 2>/dev/null || echo "0")
            
            if [[ $total_checkpoints -gt 0 ]]; then
                echo "  Checkpoints evaluated: $total_checkpoints"
                
                # Get latest metrics if verbose
                if [[ "$verbose" == "true" ]]; then
                    python3 - <<EOF 2>/dev/null || echo "  Could not parse results"
import json

try:
    with open('$results_file', 'r') as f:
        lines = f.readlines()
        if lines:
            latest = json.loads(lines[-1])
            
            print(f"  Latest checkpoint: {latest.get('checkpoint', 'unknown')}")
            
            if 'perplexity' in latest:
                ppl = latest['perplexity'].get('perplexity', 'N/A')
                print(f"  Perplexity: {ppl}")
            
            if 'blimp' in latest:
                acc = latest['blimp'].get('overall_accuracy', 'N/A')
                print(f"  BLIMP accuracy: {acc}")
            
            if 'null_subject' in latest:
                overt = latest['null_subject'].get('overt_preference_rate', 'N/A')
                print(f"  Null-subject overt pref: {overt}")
except:
    pass
EOF
                fi
            else
                echo "  No completed checkpoints"
            fi
        else
            echo "  No results file found"
        fi
        
        # Check for running evaluation
        local exp_procs
        exp_procs=$(find_evaluation_processes "$exp_name")
        if [[ -n "$exp_procs" ]]; then
            echo "  Status: RUNNING"
            if [[ "$verbose" == "true" ]]; then
                echo "$exp_procs" | while IFS=':' read -r pid cmdline; do
                    echo "    PID: $pid"
                done
            fi
        else
            echo "  Status: IDLE"
        fi
        
        echo
    done
}

# Show results summary
show_results() {
    local experiment_filter="$1"
    
    echo "=== Results Summary ==="
    echo
    
    local results_dir="$PROJECT_DIR/evaluation/results"
    if [[ ! -d "$results_dir" ]]; then
        echo "No results directory found."
        return 0
    fi
    
    # Table header
    printf "%-20s %-12s %-12s %-12s %-10s\n" "Experiment" "Perplexity" "BLIMP Acc" "Null-Subj" "Status"
    printf "%-20s %-12s %-12s %-12s %-10s\n" "----------" "----------" "---------" "--------" "------"
    
    for exp_dir in "$results_dir"/*; do
        if [[ ! -d "$exp_dir" ]]; then
            continue
        fi
        
        local exp_name
        exp_name=$(basename "$exp_dir")
        
        # Skip if filtering
        if [[ -n "$experiment_filter" ]] && [[ "$exp_name" != *"$experiment_filter"* ]]; then
            continue
        fi
        
        local results_file="$exp_dir/evaluation_results.jsonl"
        if [[ -f "$results_file" ]]; then
            # Extract metrics
            local metrics
            metrics=$(python3 - <<EOF 2>/dev/null || echo "N/A N/A N/A UNKNOWN"
import json

try:
    with open('$results_file', 'r') as f:
        lines = f.readlines()
        if lines:
            result = json.loads(lines[-1])
            
            ppl = result.get('perplexity', {}).get('perplexity', 'N/A')
            if ppl != 'N/A':
                ppl = f"{float(ppl):.1f}"
            
            blimp = result.get('blimp', {}).get('overall_accuracy', 'N/A')
            if blimp != 'N/A':
                blimp = f"{float(blimp):.3f}"
            
            null_subj = result.get('null_subject', {}).get('overt_preference_rate', 'N/A')
            if null_subj != 'N/A':
                null_subj = f"{float(null_subj):.3f}"
            
            print(f"{ppl} {blimp} {null_subj} COMPLETE")
        else:
            print("N/A N/A N/A EMPTY")
except:
    print("N/A N/A N/A ERROR")
EOF
)
            read ppl blimp null_subj status <<< "$metrics"
        else
            local ppl="N/A"
            local blimp="N/A"
            local null_subj="N/A"
            local status="MISSING"
        fi
        
        # Check if running
        local exp_procs
        exp_procs=$(find_evaluation_processes "$exp_name")
        if [[ -n "$exp_procs" ]]; then
            status="RUNNING"
        fi
        
        printf "%-20s %-12s %-12s %-12s %-10s\n" "$exp_name" "$ppl" "$blimp" "$null_subj" "$status"
    done
}

# Watch mode
watch_evaluations() {
    local experiment_filter="$1"
    local interval="$2"
    local verbose="$3"
    
    echo "Watching evaluations (press Ctrl+C to exit)..."
    echo "Update interval: ${interval}s"
    echo
    
    while true; do
        clear
        echo "=== Evaluation Monitor === $(date)"
        echo
        
        show_status "$experiment_filter"
        echo
        show_progress "$experiment_filter" "$verbose"
        
        sleep "$interval"
    done
}

# Cleanup old files
cleanup_files() {
    log INFO "Cleaning up old evaluation files..."
    
    # Clean temporary configs
    local cleaned=0
    for temp_file in /tmp/eval_config_*_*.yaml /tmp/modify_eval_config_*.py /tmp/export_results_*.py; do
        if [[ -f "$temp_file" ]]; then
            local age
            age=$(find "$temp_file" -mtime +1 2>/dev/null | wc -l)
            if [[ $age -gt 0 ]]; then
                rm -f "$temp_file"
                ((cleaned++))
            fi
        fi
    done
    
    # Clean old log files
    for log_file in /tmp/eval_*_gpu*_*.log; do
        if [[ -f "$log_file" ]]; then
            local age
            age=$(find "$log_file" -mtime +7 2>/dev/null | wc -l)
            if [[ $age -gt 0 ]]; then
                rm -f "$log_file"
                ((cleaned++))
            fi
        fi
    done
    
    log SUCCESS "Cleaned up $cleaned old files"
}

# Kill evaluation processes
kill_evaluations() {
    local experiment_filter="$1"
    
    local running_procs
    running_procs=$(find_evaluation_processes "$experiment_filter")
    
    if [[ -z "$running_procs" ]]; then
        log INFO "No running evaluations found"
        return 0
    fi
    
    echo "Found running evaluations:"
    echo "$running_procs" | while IFS=':' read -r pid cmdline; do
        echo "  PID $pid: $cmdline"
    done
    echo
    
    read -p "Kill these processes? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$running_procs" | while IFS=':' read -r pid cmdline; do
            if kill "$pid" 2>/dev/null; then
                log SUCCESS "Killed process $pid"
            else
                log ERROR "Failed to kill process $pid"
            fi
        done
    else
        log INFO "Cancelled"
    fi
}

# Main function
main() {
    local command="status"
    local experiment_filter=""
    local interval="10"
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--experiment)
                experiment_filter="$2"
                shift 2
                ;;
            -i|--interval)
                interval="$2"
                shift 2
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
                command="$1"
                shift
                ;;
        esac
    done
    
    # Execute command
    case "$command" in
        status)
            show_status "$experiment_filter"
            ;;
        progress)
            show_progress "$experiment_filter" "$verbose"
            ;;
        results)
            show_results "$experiment_filter"
            ;;
        watch)
            watch_evaluations "$experiment_filter" "$interval" "$verbose"
            ;;
        cleanup)
            cleanup_files
            ;;
        kill)
            kill_evaluations "$experiment_filter"
            ;;
        *)
            log ERROR "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi