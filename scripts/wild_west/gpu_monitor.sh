#!/bin/bash

# GPU Monitor for Wild-West Server
# Provides real-time GPU status and resource coordination

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOG_FILE="/tmp/gpu_monitor_$(whoami).log"
LOCK_DIR="/tmp/gpu_locks"

# Create lock directory
mkdir -p "$LOCK_DIR"

# Function to get GPU status
get_gpu_status() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
}

# Function to get GPU processes
get_gpu_processes() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
}

# Function to get process owner
get_process_owner() {
    local pid="$1"
    if [ -n "$pid" ] && [ "$pid" != "N/A" ]; then
        # Try to get owner from /proc
        if [ -f "/proc/$pid/status" ]; then
            local owner=$(grep "^Uid:" "/proc/$pid/status" | cut -f2 | tr -d '\t')
            # Convert UID to username
            local username=$(id -un "$owner" 2>/dev/null || echo "unknown")
            echo "$username"
        else
            echo "unknown"
        fi
    else
        echo "N/A"
    fi
}

# Function to check if GPU is available (less than 10GB used)
is_gpu_available() {
    local gpu_id="$1"
    local memory_used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | grep "^$gpu_id," | cut -d',' -f2)
    local memory_total=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | grep "^$gpu_id," | cut -d',' -f2)
    local available_gb=$((memory_total - memory_used))
    
    if [ "$available_gb" -gt 10240 ]; then  # More than 10GB available
        echo "true"
    else
        echo "false"
    fi
}

# Function to display GPU status
show_gpu_status() {
    echo -e "${BLUE}=== GPU Status at $(date) ===${NC}"
    echo ""
    
    # Get GPU status
    local gpu_status=$(get_gpu_status)
    
    echo -e "${YELLOW}GPU ID | Memory Used/Total | Utilization | Temp | Power | Status${NC}"
    echo "-------|-------------------|------------|------|-------|--------"
    
    while IFS=',' read -r gpu_id name memory_used memory_total utilization temp power; do
        # Clean up values
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        memory_used=$(echo "$memory_used" | tr -d ' ')
        memory_total=$(echo "$memory_total" | tr -d ' ')
        utilization=$(echo "$utilization" | tr -d ' ')
        temp=$(echo "$temp" | tr -d ' ')
        power=$(echo "$power" | tr -d ' ')
        
        # Calculate available memory
        local available_gb=$((memory_total - memory_used))
        local memory_pct=0
        if [ "$memory_total" -gt 0 ]; then
            memory_pct=$((memory_used * 100 / memory_total))
        fi
        
        # Determine status
        if [ "$available_gb" -gt 20480 ]; then  # More than 20GB available
            status="${GREEN}AVAILABLE${NC}"
        elif [ "$available_gb" -gt 10240 ]; then  # More than 10GB available
            status="${YELLOW}LIMITED${NC}"
        else
            status="${RED}OCCUPIED${NC}"
        fi
        
        printf "%-6s | %6s/%6s GB | %10s%% | %4s°C | %5sW | %s\n" \
               "$gpu_id" "$memory_used" "$memory_total" "$utilization" "$temp" "$power" "$status"
    done <<< "$gpu_status"
    
    echo ""
}

# Function to show GPU processes
show_gpu_processes() {
    echo -e "${BLUE}=== GPU Processes ===${NC}"
    echo ""
    
    local processes=$(get_gpu_processes)
    
    if [ -z "$processes" ]; then
        echo -e "${GREEN}No GPU processes found${NC}"
        return
    fi
    
    echo -e "${YELLOW}GPU | PID | Owner | Process | Memory${NC}"
    echo "----|-----|-------|---------|-------"
    
    while IFS=',' read -r gpu_uuid pid process memory; do
        # Clean up values
        gpu_uuid=$(echo "$gpu_uuid" | tr -d ' ')
        pid=$(echo "$pid" | tr -d ' ')
        process=$(echo "$process" | tr -d ' ')
        memory=$(echo "$memory" | tr -d ' ')
        
        # Get GPU index from UUID
        local gpu_index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | grep "$gpu_uuid" | cut -d',' -f1 | tr -d ' ')
        
        # Get process owner
        local owner=$(get_process_owner "$pid")
        
        printf "%-3s | %-4s | %-5s | %-7s | %s MiB\n" "$gpu_index" "$pid" "$owner" "$process" "$memory"
    done <<< "$processes"
    
    echo ""
}

# Function to show GPU usage summary by user
show_gpu_usage_summary() {
    echo -e "${BLUE}=== GPU Usage Summary by User ===${NC}"
    echo ""
    
    local processes=$(get_gpu_processes)
    
    if [ -z "$processes" ]; then
        echo -e "${GREEN}No GPU processes found${NC}"
        return
    fi
    
    # Create associative array to track user GPU usage
    declare -A user_gpus
    declare -A user_memory
    
    while IFS=',' read -r gpu_uuid pid process memory; do
        # Clean up values
        pid=$(echo "$pid" | tr -d ' ')
        memory=$(echo "$memory" | tr -d ' ')
        
        # Get process owner
        local owner=$(get_process_owner "$pid")
        
        if [ "$owner" != "N/A" ] && [ "$owner" != "unknown" ]; then
            # Count GPUs per user
            if [ -z "${user_gpus[$owner]}" ]; then
                user_gpus[$owner]=1
                user_memory[$owner]=$memory
            else
                user_gpus[$owner]=$((${user_gpus[$owner]} + 1))
                user_memory[$owner]=$((${user_memory[$owner]} + memory))
            fi
        fi
    done <<< "$processes"
    
    if [ ${#user_gpus[@]} -eq 0 ]; then
        echo -e "${YELLOW}No user processes found${NC}"
        return
    fi
    
    echo -e "${YELLOW}User | GPUs | Total Memory${NC}"
    echo "-----|------|-------------"
    
    for user in "${!user_gpus[@]}"; do
        local gpu_count=${user_gpus[$user]}
        local total_memory=${user_memory[$user]}
        printf "%-4s | %-4s | %s MiB\n" "$user" "$gpu_count" "$total_memory"
    done
    
    echo ""
}

# Function to find available GPUs
find_available_gpus() {
    echo -e "${BLUE}=== Available GPUs ===${NC}"
    echo ""
    
    local gpu_status=$(get_gpu_status)
    local available_gpus=()
    
    while IFS=',' read -r gpu_id name memory_used memory_total utilization temp power; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        memory_used=$(echo "$memory_used" | tr -d ' ')
        memory_total=$(echo "$memory_total" | tr -d ' ')
        
        local available_gb=$((memory_total - memory_used))
        
        if [ "$available_gb" -gt 10240 ]; then  # More than 10GB available
            available_gpus+=("$gpu_id")
            echo -e "${GREEN}GPU $gpu_id: ${available_gb}GB available${NC}"
        fi
    done <<< "$gpu_status"
    
    if [ ${#available_gpus[@]} -eq 0 ]; then
        echo -e "${RED}No GPUs with sufficient memory available${NC}"
        return 1
    fi
    
    echo ""
    echo -e "${YELLOW}Recommended GPUs for your experiments:${NC}"
    printf "%s\n" "${available_gpus[@]}"
    
    return 0
}

# Function to create GPU lock
create_gpu_lock() {
    local gpu_id="$1"
    local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"
    local user=$(whoami)
    local timestamp=$(date)
    
    if [ -f "$lock_file" ]; then
        echo -e "${RED}GPU $gpu_id is already locked by:${NC}"
        cat "$lock_file"
        return 1
    fi
    
    echo "User: $user" > "$lock_file"
    echo "Time: $timestamp" >> "$lock_file"
    echo "PID: $$" >> "$lock_file"
    
    echo -e "${GREEN}Locked GPU $gpu_id for user $user${NC}"
    return 0
}

# Function to release GPU lock
release_gpu_lock() {
    local gpu_id="$1"
    local lock_file="$LOCK_DIR/gpu_${gpu_id}.lock"
    
    if [ -f "$lock_file" ]; then
        rm "$lock_file"
        echo -e "${GREEN}Released lock on GPU $gpu_id${NC}"
    else
        echo -e "${YELLOW}No lock found on GPU $gpu_id${NC}"
    fi
}

# Function to show all locks
show_locks() {
    echo -e "${BLUE}=== GPU Locks ===${NC}"
    echo ""
    
    if [ ! "$(ls -A $LOCK_DIR)" ]; then
        echo -e "${GREEN}No GPU locks found${NC}"
        return
    fi
    
    for lock_file in "$LOCK_DIR"/*.lock; do
        if [ -f "$lock_file" ]; then
            local gpu_id=$(basename "$lock_file" .lock | sed 's/gpu_//')
            echo -e "${YELLOW}GPU $gpu_id:${NC}"
            cat "$lock_file"
            echo ""
        fi
    done
}

# Main menu
case "${1:-status}" in
    "status")
        show_gpu_status
        show_gpu_processes
        ;;
    "summary")
        show_gpu_usage_summary
        ;;
    "available")
        find_available_gpus
        ;;
    "lock")
        if [ -z "$2" ]; then
            echo "Usage: $0 lock <gpu_id>"
            exit 1
        fi
        create_gpu_lock "$2"
        ;;
    "unlock")
        if [ -z "$2" ]; then
            echo "Usage: $0 unlock <gpu_id>"
            exit 1
        fi
        release_gpu_lock "$2"
        ;;
    "locks")
        show_locks
        ;;
    "watch")
        echo -e "${BLUE}Watching GPU status (Ctrl+C to stop)...${NC}"
        while true; do
            clear
            show_gpu_status
            sleep 5
        done
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  status    - Show current GPU status (default)"
        echo "  summary   - Show GPU usage summary by user"
        echo "  available - Find available GPUs"
        echo "  lock <id> - Lock a GPU for your use"
        echo "  unlock <id> - Release a GPU lock"
        echo "  locks     - Show all GPU locks"
        echo "  watch     - Watch GPU status in real-time"
        echo "  help      - Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 