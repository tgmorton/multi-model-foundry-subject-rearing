#!/bin/bash

# P6000 Job Monitoring Script
# Provides real-time monitoring of SLURM jobs

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
REFRESH_INTERVAL=30

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$*"
    
    case $level in
        "INFO")
            echo -e "${BLUE}$message${NC}"
            ;;
        "WARN")
            echo -e "${YELLOW}$message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}$message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}$message${NC}"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -i, --interval <seconds>    Refresh interval (default: 30)
  -j, --job <job_id>          Monitor specific job
  -f, --follow                Follow log file of most recent job
  -s, --summary               Show summary only (no continuous monitoring)
  -g, --gpu                   Show GPU utilization
  -h, --help                  Show this help

Examples:
  # Monitor all jobs with default settings
  $0
  
  # Monitor with 10 second refresh
  $0 -i 10
  
  # Monitor specific job
  $0 -j 12345
  
  # Follow most recent job's log
  $0 -f
  
  # Show one-time summary
  $0 -s
EOF
}

# Function to get job details
get_job_details() {
    local job_id="$1"
    scontrol show job "$job_id" 2>/dev/null
}

# Function to format time
format_time() {
    local seconds="$1"
    local days=$((seconds / 86400))
    local hours=$(((seconds % 86400) / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $days -gt 0 ]; then
        printf "%dd %02d:%02d:%02d" $days $hours $minutes $secs
    else
        printf "%02d:%02d:%02d" $hours $minutes $secs
    fi
}

# Function to show job summary
show_job_summary() {
    clear
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    P6000 Job Monitor                           ${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get current time
    echo -e "${BLUE}Current Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Count jobs by state
    local running=$(squeue -u "$USER" -p p6000 -t RUNNING -h | wc -l)
    local pending=$(squeue -u "$USER" -p p6000 -t PENDING -h | wc -l)
    local total=$((running + pending))
    
    echo -e "${GREEN}Running:${NC} $running  ${YELLOW}Pending:${NC} $pending  ${BLUE}Total:${NC} $total"
    echo ""
    
    # Show job details
    echo -e "${CYAN}Job Details:${NC}"
    echo "--------------------------------------------------------------------------------"
    printf "%-10s %-30s %-10s %-15s %-15s\n" "JOB ID" "NAME" "STATE" "TIME" "NODE"
    echo "--------------------------------------------------------------------------------"
    
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            local job_id=$(echo "$line" | awk '{print $1}')
            local job_name=$(echo "$line" | awk '{print $3}')
            local state=$(echo "$line" | awk '{print $5}')
            local time=$(echo "$line" | awk '{print $6}')
            local node=$(echo "$line" | awk '{print $8}')
            
            # Truncate long names
            if [ ${#job_name} -gt 28 ]; then
                job_name="${job_name:0:25}..."
            fi
            
            # Color code by state
            case $state in
                "R")
                    state_color="${GREEN}RUNNING${NC}"
                    ;;
                "PD")
                    state_color="${YELLOW}PENDING${NC}"
                    ;;
                "CG")
                    state_color="${BLUE}COMPLETING${NC}"
                    ;;
                *)
                    state_color="$state"
                    ;;
            esac
            
            printf "%-10s %-30s %-20b %-15s %-15s\n" \
                "$job_id" "$job_name" "$state_color" "$time" "$node"
        fi
    done < <(squeue -u "$USER" -p p6000 -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %.20R" | tail -n +2)
    
    echo "--------------------------------------------------------------------------------"
}

# Function to show GPU utilization
show_gpu_utilization() {
    echo ""
    echo -e "${CYAN}GPU Utilization:${NC}"
    echo "--------------------------------------------------------------------------------"
    
    # Get list of nodes with running jobs
    local nodes=$(squeue -u "$USER" -p p6000 -t RUNNING -h -o "%N" | sort -u)
    
    if [ -z "$nodes" ]; then
        echo "No GPUs currently in use"
    else
        for node in $nodes; do
            echo -e "${BLUE}Node:${NC} $node"
            # Try to get GPU info from the node
            srun --nodelist="$node" --ntasks=1 --time=00:00:10 \
                nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null || echo "  Unable to query GPU"
        done
    fi
    
    echo "--------------------------------------------------------------------------------"
}

# Function to monitor specific job
monitor_job() {
    local job_id="$1"
    
    while true; do
        clear
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}                 Monitoring Job: $job_id                        ${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo ""
        
        # Get job details
        local details=$(get_job_details "$job_id")
        
        if [ -z "$details" ]; then
            log "ERROR" "Job $job_id not found or completed"
            break
        fi
        
        # Parse key information
        local state=$(echo "$details" | grep "JobState=" | cut -d'=' -f2 | cut -d' ' -f1)
        local name=$(echo "$details" | grep "JobName=" | cut -d'=' -f2)
        local runtime=$(echo "$details" | grep "RunTime=" | cut -d'=' -f2 | cut -d' ' -f1)
        local timelimit=$(echo "$details" | grep "TimeLimit=" | cut -d'=' -f2 | cut -d' ' -f1)
        local node=$(echo "$details" | grep "NodeList=" | cut -d'=' -f2)
        local workdir=$(echo "$details" | grep "WorkDir=" | cut -d'=' -f2)
        
        echo -e "${BLUE}Job Name:${NC} $name"
        echo -e "${BLUE}State:${NC} $state"
        echo -e "${BLUE}Runtime:${NC} $runtime / $timelimit"
        echo -e "${BLUE}Node:${NC} $node"
        echo -e "${BLUE}Work Dir:${NC} $workdir"
        echo ""
        
        # Show last lines of output log if running
        if [ "$state" == "RUNNING" ]; then
            local log_file="$PROJECT_DIR/logs/*-${job_id}.out"
            if ls $log_file 1> /dev/null 2>&1; then
                echo -e "${CYAN}Recent Output:${NC}"
                echo "--------------------------------------------------------------------------------"
                tail -n 10 $log_file 2>/dev/null || echo "No output yet"
                echo "--------------------------------------------------------------------------------"
            fi
        fi
        
        echo ""
        echo "Refreshing in $REFRESH_INTERVAL seconds... (Press Ctrl+C to exit)"
        sleep $REFRESH_INTERVAL
    done
}

# Function to follow log file
follow_log() {
    # Get most recent job
    local job_id=$(squeue -u "$USER" -p p6000 -h -o "%i" | head -n1)
    
    if [ -z "$job_id" ]; then
        log "ERROR" "No active jobs found"
        return 1
    fi
    
    local log_file="$PROJECT_DIR/logs/*-${job_id}.out"
    
    if ls $log_file 1> /dev/null 2>&1; then
        log "INFO" "Following log for job $job_id (Press Ctrl+C to exit)"
        tail -f $log_file
    else
        log "ERROR" "Log file not found for job $job_id"
        return 1
    fi
}

# Main monitoring loop
monitor_loop() {
    while true; do
        show_job_summary
        
        if [ "${SHOW_GPU:-}" == "true" ]; then
            show_gpu_utilization
        fi
        
        echo ""
        echo "Refreshing in $REFRESH_INTERVAL seconds... (Press Ctrl+C to exit)"
        sleep $REFRESH_INTERVAL
    done
}

# Main execution
main() {
    local mode="monitor"
    local job_id=""
    SHOW_GPU=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--interval)
                REFRESH_INTERVAL="$2"
                shift 2
                ;;
            -j|--job)
                mode="job"
                job_id="$2"
                shift 2
                ;;
            -f|--follow)
                mode="follow"
                shift
                ;;
            -s|--summary)
                mode="summary"
                shift
                ;;
            -g|--gpu)
                SHOW_GPU=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute based on mode
    case $mode in
        monitor)
            monitor_loop
            ;;
        job)
            monitor_job "$job_id"
            ;;
        follow)
            follow_log
            ;;
        summary)
            show_job_summary
            if [ "$SHOW_GPU" == "true" ]; then
                show_gpu_utilization
            fi
            ;;
    esac
}

# Set up clean exit
trap 'echo ""; log "INFO" "Monitoring stopped"; exit 0' INT TERM

# Run main function
main "$@"