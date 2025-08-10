#!/usr/bin/env bash
# Quick Evaluation Script
# Fast evaluation for testing and debugging

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
Usage: $0 [OPTIONS] <experiment>

Quick evaluation with minimal samples for testing.

Options:
  -g, --gpu <id>            GPU ID to use (default: first available)
  -s, --samples <num>       Max samples per task (default: 50)  
  -c, --checkpoints <num>   Max checkpoints to evaluate (default: 3)
  -t, --task <task>         Single task to run (perplexity|blimp|null_subject)
  --no-lock                 Don't lock GPU
  --verbose                 Verbose output
  -h, --help               Show this help

Examples:
  # Quick test of exp0_baseline
  $0 exp0_baseline

  # Test only BLIMP evaluation
  $0 -t blimp exp0_baseline

  # Minimal test with 10 samples
  $0 -s 10 -c 1 exp0_baseline
EOF
}

# Find first available GPU
find_available_gpu() {
    local gpu_monitor="$SCRIPT_DIR/gpu_monitor.sh"
    
    if [[ -f "$gpu_monitor" ]]; then
        local available
        available=$("$gpu_monitor" available 2>/dev/null | grep -E '^[0-3]$' | head -n1)
        if [[ -n "$available" ]]; then
            echo "$available"
            return 0
        fi
    fi
    
    # Fallback: check nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "")
        
        while IFS=', ' read -r gpu_id mem_free; do
            if [[ $gpu_id -ge 0 && $gpu_id -le 3 && $mem_free -gt 10000 ]]; then  # GPU 0-3 with >10GB free
                echo "$gpu_id"
                return 0
            fi
        done <<< "$gpu_info"
    fi
    
    # Final fallback
    echo "0"
}

# Run quick evaluation test
run_quick_test() {
    local experiment="$1"
    local gpu_id="$2"
    local max_samples="$3"
    local max_checkpoints="$4"
    local task="$5"
    local use_lock="$6"
    local verbose="$7"
    
    log INFO "Running quick evaluation test:"
    log INFO "  Experiment: $experiment"
    log INFO "  GPU: $gpu_id"
    log INFO "  Max samples: $max_samples"
    log INFO "  Max checkpoints: $max_checkpoints"
    log INFO "  Task: ${task:-all}"
    
    # Build command
    local cmd=("$SCRIPT_DIR/run_evaluation.sh")
    cmd+=("-g" "$gpu_id")
    cmd+=("-e" "$experiment")
    cmd+=("-s" "$max_samples")
    cmd+=("-m" "$max_checkpoints")
    
    if [[ -n "$task" ]]; then
        cmd+=("-t" "$task")
    fi
    
    if [[ "$use_lock" == "true" ]]; then
        cmd+=("-l" "-u")
    fi
    
    if [[ "$verbose" == "true" ]]; then
        cmd+=("-v")
    fi
    
    # Add fast mode
    cmd+=("-f")
    
    log INFO "Running command: ${cmd[*]}"
    
    # Run evaluation
    local start_time
    start_time=$(date +%s)
    
    if "${cmd[@]}"; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log SUCCESS "Quick evaluation completed in ${duration}s"
        
        # Show quick results summary
        show_results_summary "$experiment"
    else
        log ERROR "Quick evaluation failed"
        return 1
    fi
}

# Show results summary
show_results_summary() {
    local experiment="$1"
    local results_file="$PROJECT_DIR/evaluation/results/$experiment/evaluation_results.jsonl"
    
    if [[ ! -f "$results_file" ]]; then
        log WARN "Results file not found: $results_file"
        return
    fi
    
    log INFO "Quick Results Summary:"
    
    # Extract last result using Python
    python3 - <<EOF
import json
import sys

try:
    with open('$results_file', 'r') as f:
        lines = f.readlines()
        if not lines:
            print("No results found")
            sys.exit(0)
        
        result = json.loads(lines[-1])
        
        print(f"Checkpoint: {result.get('checkpoint', 'unknown')}")
        
        # Perplexity
        if 'perplexity' in result:
            ppl = result['perplexity'].get('perplexity', 'N/A')
            print(f"Perplexity: {ppl}")
        
        # BLIMP  
        if 'blimp' in result:
            acc = result['blimp'].get('overall_accuracy', 'N/A')
            total = result['blimp'].get('total_stimuli', 'N/A')
            print(f"BLIMP Accuracy: {acc} ({total} stimuli)")
        
        # Null-subject
        if 'null_subject' in result:
            overt_pref = result['null_subject'].get('overt_preference_rate', 'N/A')
            total_pairs = result['null_subject'].get('total_pairs', 'N/A')
            print(f"Null-subject Overt Preference: {overt_pref} ({total_pairs} pairs)")
            
except Exception as e:
    print(f"Error reading results: {e}")
EOF
}

# Test evaluation setup
test_setup() {
    log INFO "Testing evaluation setup..."
    
    # Check Python imports
    log INFO "Checking Python dependencies..."
    python3 - <<EOF
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    
    import transformers  
    print(f"Transformers: {transformers.__version__}")
    
    import pandas
    print(f"Pandas: {pandas.__version__}")
    
    import yaml
    print("YAML: OK")
    
    import sentencepiece
    print("SentencePiece: OK")
    
    print("All dependencies available!")
    
except ImportError as e:
    print(f"Missing dependency: {e}")
    exit(1)
EOF
    
    if [[ $? -ne 0 ]]; then
        log ERROR "Python dependency check failed"
        return 1
    fi
    
    # Check evaluation module
    log INFO "Testing evaluation module import..."
    cd "$PROJECT_DIR"
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from evaluation.evaluation_runner import EvaluationConfig
    print('Evaluation module import: OK')
except Exception as e:
    print(f'Evaluation module import failed: {e}')
    sys.exit(1)
"
    
    if [[ $? -ne 0 ]]; then
        log ERROR "Evaluation module test failed"
        return 1
    fi
    
    log SUCCESS "Evaluation setup test passed"
}

# Main function
main() {
    local experiment=""
    local gpu_id=""
    local max_samples="50"
    local max_checkpoints="3"
    local task=""
    local use_lock=true
    local verbose=false
    local test_setup_flag=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -g|--gpu)
                gpu_id="$2"
                shift 2
                ;;
            -s|--samples)
                max_samples="$2"
                shift 2
                ;;
            -c|--checkpoints)
                max_checkpoints="$2"
                shift 2
                ;;
            -t|--task)
                task="$2"
                shift 2
                ;;
            --no-lock)
                use_lock=false
                shift
                ;;
            --test-setup)
                test_setup_flag=true
                shift
                ;;
            --verbose)
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
    
    # Test setup if requested
    if [[ "$test_setup_flag" == "true" ]]; then
        test_setup
        exit $?
    fi
    
    # Validate experiment
    if [[ -z "$experiment" ]]; then
        log ERROR "Must specify experiment name"
        show_usage
        exit 1
    fi
    
    # Find GPU if not specified
    if [[ -z "$gpu_id" ]]; then
        gpu_id=$(find_available_gpu)
        log INFO "Auto-selected GPU: $gpu_id"
    fi
    
    # Validate experiment directory exists
    local exp_dir="$PROJECT_DIR/models/$experiment"
    if [[ ! -d "$exp_dir" ]]; then
        log ERROR "Experiment directory not found: $exp_dir"
        log INFO "Available experiments:"
        ls -1 "$PROJECT_DIR/models/" 2>/dev/null | grep -E "^exp[0-9]" | head -10 || echo "  No experiments found"
        exit 1
    fi
    
    # Run quick evaluation
    run_quick_test "$experiment" "$gpu_id" "$max_samples" "$max_checkpoints" "$task" "$use_lock" "$verbose"
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi