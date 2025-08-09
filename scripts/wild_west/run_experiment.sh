#!/usr/bin/env bash
# Wild-West Experiment Runner — hardened

set -euo pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOCK_DIR="/tmp/gpu_locks"

# Defaults
DEFAULT_GPUS="1,2"
DEFAULT_PHASE="full-pipeline"

# Logging
log() {
  local level="$1"; shift
  local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
  case "$level" in
    INFO)    echo -e "${BLUE}[$ts] INFO: $*${NC}";;
    WARN)    echo -e "${YELLOW}[$ts] WARN: $*${NC}";;
    ERROR)   echo -e "${RED}[$ts] ERROR: $*${NC}";;
    SUCCESS) echo -e "${GREEN}[$ts] SUCCESS: $*${NC}";;
  esac
}

show_usage() {
  cat <<EOF
Usage: $0 [OPTIONS] <config_name>

Options:
  -g, --gpus <ids>           Comma-separated GPU IDs (default: $DEFAULT_GPUS)
  -p, --phase <phase>        Phase: preprocess | train-tokenizer | tokenize-dataset | chunk-data | run | full-pipeline (default: $DEFAULT_PHASE)
  -b, --batch-size <size>    Override batch size
  -e, --epochs <num>         Override epochs
  -d, --distributed          Enable distributed training
  -n, --nodes <num>          Num nodes (default 1)
  -r, --rank <rank>          Node rank (default 0)
  -w, --world-size <size>    World size (default 1)
  -l, --lock-gpus            Lock GPUs before running
  -u, --unlock-gpus          Unlock GPUs after running
  -c, --check-gpus           Check GPU availability before running
  --wandb-mode <mode>        W&B mode: online | offline | disabled (default: disabled)
  --wandb-api-key <key>      W&B API key
  -v, --verbose              Verbose output
  -h, --help                 Show this help
EOF
}

# Args
GPUS="$DEFAULT_GPUS"; PHASE="$DEFAULT_PHASE"; BATCH_SIZE=""; NUM_EPOCHS=""
DISTRIBUTED=false; NODES=1; RANK=0; WORLD_SIZE=1
LOCK_GPUS=false; UNLOCK_GPUS=false; CHECK_GPUS=false
WANDB_MODE="disabled"; WANDB_API_KEY=""; VERBOSE=false
CONFIG_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--gpus) GPUS="$2"; shift 2;;
    -p|--phase) PHASE="$2"; shift 2;;
    -b|--batch-size) BATCH_SIZE="$2"; shift 2;;
    -e|--epochs) NUM_EPOCHS="$2"; shift 2;;
    -d|--distributed) DISTRIBUTED=true; shift;;
    -n|--nodes) NODES="$2"; shift 2;;
    -r|--rank) RANK="$2"; shift 2;;
    -w|--world-size) WORLD_SIZE="$2"; shift 2;;
    -l|--lock-gpus) LOCK_GPUS=true; shift;;
    -u|--unlock-gpus) UNLOCK_GPUS=true; shift;;
    -c|--check-gpus) CHECK_GPUS=true; shift;;
    --wandb-mode) WANDB_MODE="$2"; shift 2;;
    --wandb-api-key) WANDB_API_KEY="$2"; shift 2;;
    -v|--verbose) VERBOSE=true; shift;;
    -h|--help) show_usage; exit 0;;
    -* ) echo "Unknown option: $1"; show_usage; exit 1;;
    *  ) CONFIG_NAME="$1"; shift;;
  esac
done

[[ -z "${CONFIG_NAME:-}" ]] && { log ERROR "Config name is required"; show_usage; exit 1; }

# Ensure lock dir
mkdir -p "$LOCK_DIR"

# Container runtime
if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_BIN="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_BIN="singularity"
else
  log ERROR "Neither apptainer nor singularity found in PATH."
  exit 1
fi

# GPU check
check_gpu_availability() {
  log INFO "Checking GPU availability..."
  IFS=',' read -ra ARR <<< "$GPUS"
  for gpu_id in "${ARR[@]}"; do
    gpu_id=$(echo "$gpu_id" | tr -d ' ')
    local info
    info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' -v id="$gpu_id" '$1==id{print}')
    [[ -z "$info" ]] && { log ERROR "GPU $gpu_id not found"; return 1; }
    local used total
    used=$(echo "$info" | cut -d',' -f2 | tr -d ' ')
    total=$(echo "$info" | cut -d',' -f3 | tr -d ' ')
    local avail=$((total - used))
    log INFO "GPU $gpu_id: ${avail}MB available (${used}MB used / ${total}MB total)"
    if (( avail < 10240 )); then log WARN "GPU $gpu_id low free memory (${avail}MB)"; fi
  done
}

# Lock/unlock
lock_gpus() {
  log INFO "Locking GPUs: $GPUS"
  IFS=',' read -ra ARR <<< "$GPUS"
  for gpu_id in "${ARR[@]}"; do
    gpu_id=$(echo "$gpu_id" | tr -d ' ')
    local f="$LOCK_DIR/gpu_${gpu_id}.lock"
    if [[ -f "$f" ]]; then log ERROR "GPU $gpu_id already locked by:"; cat "$f"; return 1; fi
    {
      echo "User: $(whoami)"
      echo "Time: $(date)"
      echo "PID: $$"
      echo "Config: $CONFIG_NAME"
      echo "Phase: $PHASE"
    } > "$f"
    log SUCCESS "Locked GPU $gpu_id"
  done
}
unlock_gpus() {
  log INFO "Unlocking GPUs: $GPUS"
  IFS=',' read -ra ARR <<< "$GPUS"
  for gpu_id in "${ARR[@]}"; do
    gpu_id=$(echo "$gpu_id" | tr -d ' ')
    local f="$LOCK_DIR/gpu_${gpu_id}.lock"
    if [[ -f "$f" ]]; then rm -f "$f"; log SUCCESS "Unlocked GPU $gpu_id"; else log WARN "No lock found on GPU $gpu_id"; fi
  done
}

# Build training command
build_command() {
  local base="$1"; local cfg="$2"; local cmd="$base $cfg"
  [[ -n "$BATCH_SIZE" ]] && cmd="$cmd --batch-size $BATCH_SIZE"
  [[ -n "$NUM_EPOCHS" ]] && cmd="$cmd --epochs $NUM_EPOCHS"
  $DISTRIBUTED && cmd="$cmd --distributed --world-size $WORLD_SIZE --rank $RANK"
  echo "$cmd"
}

# Run inside container with robust signal/PGID handling
run_in_container() {
  local sif="$1"; local run_cmd="$2"; local desc="$3"
  log INFO "Running: $desc"
  log INFO "Container: $sif"
  log INFO "Command: $run_cmd"
  [[ ! -f "$sif" ]] && { log ERROR "Container not found: $sif"; return 1; }

  # Env to pass into container (we use multiple --env flags to avoid comma parsing issues)
  local env_flags=()
  env_flags+=(--env "CUDA_VISIBLE_DEVICES=${GPUS}")
  env_flags+=(--env "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False,max_split_size_mb:512}")
  env_flags+=(--env "TORCH_CUDA_MEMORY_FRACTION=${TORCH_CUDA_MEMORY_FRACTION:-0.95}")
  env_flags+=(--env "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}")
  env_flags+=(--env "NCCL_DEBUG=${NCCL_DEBUG:-WARN}")
  env_flags+=(--env "NCCL_ASYNC_ERROR_HANDLING=1")

  case "$WANDB_MODE" in
    disabled) env_flags+=(--env "WANDB_MODE=disabled" --env "WANDB_SILENT=true");;
    offline)  env_flags+=(--env "WANDB_MODE=offline");;
    online)   env_flags+=(--env "WANDB_MODE=online");;
  esac
  [[ -n "$WANDB_API_KEY" ]] && env_flags+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

  if $DISTRIBUTED; then
    env_flags+=(--env "MASTER_ADDR=${MASTER_ADDR:-localhost}")
    env_flags+=(--env "MASTER_PORT=${MASTER_PORT:-12355}")
    env_flags+=(--env "WORLD_SIZE=${WORLD_SIZE}")
    env_flags+=(--env "RANK=${RANK}")
  fi

  # Prepare setup commands
  local setup_commands="set -euo pipefail; cd /workspace"
  
  # Add flash-attention installation for training commands
  if [[ "$run_cmd" == *"run"* ]] || [[ "$desc" == *"training"* ]]; then
    log INFO "Installing flash-attention for training..."
    setup_commands="$setup_commands; (pip install flash-attn --no-build-isolation --quiet || echo 'Warning: Flash Attention installation failed, falling back to standard attention')"
  fi

  # Launch in its own session so we can kill the whole PGID
  setsid "$CONTAINER_BIN" exec --nv --pid --contain --cleanenv \
    "${env_flags[@]}" \
    --bind "${PROJECT_DIR}":/workspace \
    "$sif" bash -lc "
      $setup_commands
      # Avoid doing downloads on every run; ensure base image has these
      # python -m spacy download en_core_web_sm --quiet || true
      exec $run_cmd
    " &

  local child=$!
  local pgid; pgid=$(ps -o pgid= "$child" | tr -d ' ')
  log INFO "Container child PID: $child  PGID: $pgid"

  # Trap to kill the entire container process group on exit/signals
  trap 'kill -TERM -'"$pgid"' 2>/dev/null || true; sleep 2; kill -KILL -'"$pgid"' 2>/dev/null || true; wait '"$child"' 2>/dev/null || true' INT TERM EXIT

  # Foreground wait
  wait "$child"
  local status=$?

  # Clear trap for next phase
  trap - INT TERM EXIT
  return $status
}

main() {
  # Global trap ONLY handles unlocks; cleanup is per-container launch
  if $LOCK_GPUS; then
    trap 'unlock_gpus' EXIT
  fi

  log INFO "Starting experiment: $CONFIG_NAME"
  log INFO "Phase: $PHASE  GPUs: $GPUS"

  $CHECK_GPUS && check_gpu_availability

  $LOCK_GPUS && lock_gpus

  # Set env in host (also passed inside container)
  export CUDA_VISIBLE_DEVICES="$GPUS"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False,max_split_size_mb:512}"
  export TORCH_CUDA_MEMORY_FRACTION="${TORCH_CUDA_MEMORY_FRACTION:-0.95}"
  export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

  # Paths
  HOST_ABLATION_SIF_PATH="$PROJECT_DIR/singularity/ablation.sif"
  HOST_TRAINING_SIF_PATH="$PROJECT_DIR/singularity/training.sif"
  HOST_CONFIG_FILE="$PROJECT_DIR/configs/${CONFIG_NAME}.yaml"
  CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

  [[ -f "$HOST_CONFIG_FILE" ]] || { log ERROR "Config file not found: $HOST_CONFIG_FILE"; exit 1; }
  mkdir -p "$PROJECT_DIR/logs"

  case "$PHASE" in
    preprocess)
      log INFO "Preprocess..."
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli preprocess" "$CONTAINER_CONFIG_FILE")" \
        "Dataset preprocessing and ablation"
      ;;
    train-tokenizer)
      log INFO "Train tokenizer..."
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli train-tokenizer" "$CONTAINER_CONFIG_FILE")" \
        "SentencePiece tokenizer training"
      ;;
    tokenize-dataset)
      log INFO "Tokenize dataset..."
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli tokenize-dataset" "$CONTAINER_CONFIG_FILE")" \
        "Dataset tokenization"
      ;;
    chunk-data)
      log INFO "Chunk data..."
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli preprocess-data" "$CONTAINER_CONFIG_FILE")" \
        "Dataset chunking"
      ;;
    run)
      log INFO "Train model..."
      run_in_container "$HOST_TRAINING_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli run" "$CONTAINER_CONFIG_FILE")" \
        "Model training"
      ;;
    full-pipeline)
      log INFO "Full pipeline..."
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli preprocess" "$CONTAINER_CONFIG_FILE")" \
        "Dataset preprocessing and ablation"
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli train-tokenizer" "$CONTAINER_CONFIG_FILE")" \
        "SentencePiece tokenizer training"
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli tokenize-dataset" "$CONTAINER_CONFIG_FILE")" \
        "Dataset tokenization"
      run_in_container "$HOST_ABLATION_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli preprocess-data" "$CONTAINER_CONFIG_FILE")" \
        "Dataset chunking"
      run_in_container "$HOST_TRAINING_SIF_PATH" \
        "$(build_command "python -m model_foundry.cli run" "$CONTAINER_CONFIG_FILE")" \
        "Model training"
      ;;
    *)
      log ERROR "Unknown phase '$PHASE'"; exit 1;;
  esac

  log SUCCESS "Experiment completed: $CONFIG_NAME"
  $UNLOCK_GPUS && unlock_gpus
}

main "$@"
