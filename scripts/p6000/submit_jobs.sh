#!/bin/bash

# P6000 Job Submission Helper
# Simplifies submitting SLURM jobs for experiments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$*"
    
    case $level in
        "INFO")
            echo -e "${BLUE}INFO: $message${NC}"
            ;;
        "WARN")
            echo -e "${YELLOW}WARN: $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}ERROR: $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}SUCCESS: $message${NC}"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  experiment <config> [phase]    Submit a full experiment or specific phase
  phase <config> <phase>         Submit a single phase
  batch <config_list>            Submit multiple experiments from a list
  status                         Show status of all your jobs
  cancel <job_id>                Cancel a specific job
  cancel-all                     Cancel all your jobs
  logs <job_id>                  Show logs for a specific job
  
Options for 'experiment' and 'phase':
  --time <HH:MM:SS>              Override time limit
  --mem <memory>                 Override memory (e.g., 32G)
  --batch-size <size>            Override batch size
  --epochs <num>                 Override number of epochs
  --wandb-mode <mode>            Set W&B mode (online/offline/disabled)
  --job-name <name>              Custom job name
  --dependency <job_id>          Wait for another job to complete
  --array <start-end>            Submit as array job
  
Examples:
  # Submit full pipeline for baseline experiment
  $0 experiment experiment_0_baseline
  
  # Submit only training phase with custom settings
  $0 phase experiment_0_baseline run --time 48:00:00 --batch-size 64
  
  # Submit multiple experiments in sequence
  $0 batch experiments.txt
  
  # Check job status
  $0 status
  
  # View logs for a job
  $0 logs 12345
  
Config files should be in: $PROJECT_DIR/configs/
Logs will be saved to: $PROJECT_DIR/logs/

Available configurations:
EOF
    
    # List available config files
    if [ -d "$PROJECT_DIR/configs" ]; then
        echo ""
        for config in "$PROJECT_DIR"/configs/*.yaml; do
            if [ -f "$config" ]; then
                basename "$config" .yaml | sed 's/^/  - /'
            fi
        done
    fi
}

# Function to submit experiment
submit_experiment() {
    local config_name="$1"
    shift
    
    local phase="full-pipeline"
    local slurm_opts=""
    local script_opts=""
    local job_name="exp-${config_name}"
    
    # Parse additional options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --time)
                slurm_opts="$slurm_opts --time=$2"
                shift 2
                ;;
            --mem)
                slurm_opts="$slurm_opts --mem=$2"
                shift 2
                ;;
            --batch-size)
                script_opts="$script_opts -b $2"
                shift 2
                ;;
            --epochs)
                script_opts="$script_opts -e $2"
                shift 2
                ;;
            --wandb-mode)
                script_opts="$script_opts --wandb-mode $2"
                shift 2
                ;;
            --job-name)
                job_name="$2"
                slurm_opts="$slurm_opts --job-name=$2"
                shift 2
                ;;
            --dependency)
                slurm_opts="$slurm_opts --dependency=afterok:$2"
                shift 2
                ;;
            --array)
                slurm_opts="$slurm_opts --array=$2"
                shift 2
                ;;
            *)
                if [ -z "$phase" ]; then
                    phase="$1"
                else
                    log "ERROR" "Unknown option: $1"
                    return 1
                fi
                shift
                ;;
        esac
    done
    
    # Check if config exists
    if [ ! -f "$PROJECT_DIR/configs/${config_name}.yaml" ]; then
        log "ERROR" "Config file not found: ${config_name}.yaml"
        return 1
    fi
    
    # Set default time based on phase
    if [[ ! "$slurm_opts" =~ "--time" ]]; then
        case $phase in
            "preprocess")
                slurm_opts="$slurm_opts --time=2:00:00"
                ;;
            "train-tokenizer"|"tokenize-dataset")
                slurm_opts="$slurm_opts --time=1:00:00"
                ;;
            "run")
                slurm_opts="$slurm_opts --time=24:00:00"
                ;;
            "full-pipeline")
                slurm_opts="$slurm_opts --time=48:00:00"
                ;;
        esac
    fi
    
    # Submit the job
    log "INFO" "Submitting experiment: $config_name (phase: $phase)"
    
    local cmd="sbatch --partition=p6000 --gres=gpu:p6000:1 \
        --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
        --mem=32G --output=$PROJECT_DIR/logs/%x-%j.out \
        --error=$PROJECT_DIR/logs/%x-%j.err \
        --job-name=$job_name \
        $slurm_opts \
        $SCRIPT_DIR/run_experiment_slurm.sh \
        -- -p $phase $script_opts $config_name"
    
    if [ -n "${VERBOSE:-}" ]; then
        log "INFO" "Command: $cmd"
    fi
    
    # Execute and capture job ID
    local output=$(eval $cmd 2>&1)
    local status=$?
    
    if [ $status -eq 0 ]; then
        local job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
        log "SUCCESS" "Job submitted with ID: $job_id"
        echo "$job_id"
    else
        log "ERROR" "Failed to submit job: $output"
        return 1
    fi
}

# Function to submit single phase
submit_phase() {
    local config_name="$1"
    local phase="$2"
    shift 2
    
    local slurm_opts=""
    local script_opts=""
    local job_name="phase-${config_name}-${phase}"
    
    # Parse additional options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --time)
                slurm_opts="$slurm_opts --time=$2"
                shift 2
                ;;
            --mem)
                slurm_opts="$slurm_opts --mem=$2"
                shift 2
                ;;
            --batch-size)
                script_opts="$script_opts -b $2"
                shift 2
                ;;
            --epochs)
                script_opts="$script_opts -e $2"
                shift 2
                ;;
            --wandb-mode)
                script_opts="$script_opts --wandb-mode $2"
                shift 2
                ;;
            --job-name)
                job_name="$2"
                slurm_opts="$slurm_opts --job-name=$2"
                shift 2
                ;;
            --dependency)
                slurm_opts="$slurm_opts --dependency=afterok:$2"
                shift 2
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                return 1
                ;;
        esac
    done
    
    # Check if config exists
    if [ ! -f "$PROJECT_DIR/configs/${config_name}.yaml" ]; then
        log "ERROR" "Config file not found: ${config_name}.yaml"
        return 1
    fi
    
    # Set default time based on phase
    if [[ ! "$slurm_opts" =~ "--time" ]]; then
        case $phase in
            "preprocess")
                slurm_opts="$slurm_opts --time=2:00:00"
                ;;
            "train-tokenizer"|"tokenize-dataset")
                slurm_opts="$slurm_opts --time=1:00:00"
                ;;
            "run")
                slurm_opts="$slurm_opts --time=24:00:00"
                ;;
        esac
    fi
    
    # Submit the job
    log "INFO" "Submitting phase: $phase for $config_name"
    
    local cmd="sbatch --partition=p6000 --gres=gpu:p6000:1 \
        --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
        --mem=32G --output=$PROJECT_DIR/logs/%x-%j.out \
        --error=$PROJECT_DIR/logs/%x-%j.err \
        --job-name=$job_name \
        $slurm_opts \
        $SCRIPT_DIR/run_phase_slurm.sh \
        -- $config_name $phase $script_opts"
    
    if [ -n "${VERBOSE:-}" ]; then
        log "INFO" "Command: $cmd"
    fi
    
    # Execute and capture job ID
    local output=$(eval $cmd 2>&1)
    local status=$?
    
    if [ $status -eq 0 ]; then
        local job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
        log "SUCCESS" "Job submitted with ID: $job_id"
        echo "$job_id"
    else
        log "ERROR" "Failed to submit job: $output"
        return 1
    fi
}

# Function to submit batch of experiments
submit_batch() {
    local config_list="$1"
    
    if [ ! -f "$config_list" ]; then
        log "ERROR" "Config list file not found: $config_list"
        return 1
    fi
    
    log "INFO" "Submitting batch jobs from: $config_list"
    
    local prev_job_id=""
    while IFS= read -r config_name || [ -n "$config_name" ]; do
        # Skip empty lines and comments
        [[ -z "$config_name" || "$config_name" =~ ^# ]] && continue
        
        # Submit with dependency on previous job if exists
        if [ -n "$prev_job_id" ]; then
            prev_job_id=$(submit_experiment "$config_name" --dependency "$prev_job_id")
        else
            prev_job_id=$(submit_experiment "$config_name")
        fi
        
        if [ $? -ne 0 ]; then
            log "ERROR" "Failed to submit: $config_name"
            return 1
        fi
    done < "$config_list"
    
    log "SUCCESS" "All jobs submitted successfully"
}

# Function to show job status
show_status() {
    log "INFO" "Your current jobs on P6000:"
    squeue -u "$USER" -p p6000 -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %.20R"
}

# Function to cancel jobs
cancel_job() {
    local job_id="$1"
    log "INFO" "Cancelling job: $job_id"
    scancel "$job_id"
    if [ $? -eq 0 ]; then
        log "SUCCESS" "Job cancelled: $job_id"
    else
        log "ERROR" "Failed to cancel job: $job_id"
    fi
}

cancel_all_jobs() {
    log "WARN" "Cancelling all your P6000 jobs..."
    scancel -u "$USER" -p p6000
    if [ $? -eq 0 ]; then
        log "SUCCESS" "All jobs cancelled"
    else
        log "ERROR" "Failed to cancel jobs"
    fi
}

# Function to show logs
show_logs() {
    local job_id="$1"
    local log_file="$PROJECT_DIR/logs/*-${job_id}.out"
    local err_file="$PROJECT_DIR/logs/*-${job_id}.err"
    
    log "INFO" "Showing logs for job: $job_id"
    
    # Find and display output log
    if ls $log_file 1> /dev/null 2>&1; then
        echo -e "\n${GREEN}=== Output Log ===${NC}"
        tail -n 50 $log_file
    else
        log "WARN" "Output log not found"
    fi
    
    # Find and display error log
    if ls $err_file 1> /dev/null 2>&1; then
        echo -e "\n${RED}=== Error Log ===${NC}"
        tail -n 20 $err_file
    else
        log "INFO" "No errors found"
    fi
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    local command="$1"
    shift
    
    case $command in
        experiment)
            [ $# -lt 1 ] && { log "ERROR" "Config name required"; show_usage; exit 1; }
            submit_experiment "$@"
            ;;
        phase)
            [ $# -lt 2 ] && { log "ERROR" "Config name and phase required"; show_usage; exit 1; }
            submit_phase "$@"
            ;;
        batch)
            [ $# -lt 1 ] && { log "ERROR" "Config list file required"; show_usage; exit 1; }
            submit_batch "$@"
            ;;
        status)
            show_status
            ;;
        cancel)
            [ $# -lt 1 ] && { log "ERROR" "Job ID required"; show_usage; exit 1; }
            cancel_job "$@"
            ;;
        cancel-all)
            cancel_all_jobs
            ;;
        logs)
            [ $# -lt 1 ] && { log "ERROR" "Job ID required"; show_usage; exit 1; }
            show_logs "$@"
            ;;
        help|-h|--help)
            show_usage
            ;;
        *)
            log "ERROR" "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"