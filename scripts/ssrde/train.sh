#!/usr/bin/env bash
#SBATCH --job-name=model_foundry_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#
# SLURM Training Script for SSRDE Cluster
# Unified script for A5000 and P6000 nodes
#
# Usage:
#   sbatch [SLURM_OPTIONS] scripts/ssrde/train.sh <config_path>
#
# Common SLURM options:
#   --gres=gpu:N              Number of GPUs (1-4)
#   --time=HH:MM:SS           Time limit
#   --partition=PARTITION     Partition (a5000, p6000, or general)
#   --job-name=NAME           Job name
#   --mail-type=TYPE          Email notification (BEGIN,END,FAIL,ALL)
#   --mail-user=EMAIL         Email address
#
# Examples:
#   sbatch scripts/ssrde/train.sh configs/gpt2_small.yaml
#   sbatch --gres=gpu:4 --time=48:00:00 scripts/ssrde/train.sh configs/gpt2_large.yaml
#   sbatch --partition=a5000 scripts/ssrde/train.sh configs/experiment_0_baseline.yaml
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

CONFIG_PATH="${1:-}"

if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: Config path required"
    echo "Usage: sbatch $0 <config_path>"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# Get experiment name
EXP_NAME=$(basename "$CONFIG_PATH" .yaml)

# ============================================================================
# SLURM Job Information
# ============================================================================

echo "=========================================="
echo "Model Foundry Training - SLURM"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Config: $CONFIG_PATH"
echo "Experiment: $EXP_NAME"
echo "User: $USER"
echo "Working Dir: $PWD"
echo "Start Time: $(date)"
echo "=========================================="

# ============================================================================
# GPU Information
# ============================================================================

echo ""
echo "GPU Configuration:"
echo "  GPUs Allocated: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
fi

# ============================================================================
# Environment Setup
# ============================================================================

echo "Setting up environment..."

# Load modules if needed (adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate conda/virtual environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate model_foundry

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_CUDA_MEMORY_FRACTION="0.95"
export CUDA_LAUNCH_BLOCKING="0"
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
export NCCL_DEBUG="WARN"  # Set to INFO for debugging distributed training

# For multi-GPU training
if [ "${SLURM_GPUS_ON_NODE:-1}" -gt 1 ]; then
    export MASTER_ADDR=$(hostname)
    export MASTER_PORT=29500
    export WORLD_SIZE=${SLURM_GPUS_ON_NODE}
    export RANK=0
fi

echo "Environment variables set:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
if [ "${SLURM_GPUS_ON_NODE:-1}" -gt 1 ]; then
    echo "  MASTER_ADDR=$MASTER_ADDR"
    echo "  WORLD_SIZE=$WORLD_SIZE"
fi
echo ""

# ============================================================================
# Training Execution
# ============================================================================

echo "Starting training..."
echo "Command: python -m model_foundry.train $CONFIG_PATH"
echo ""

# Run training
START_TIME=$(date +%s)

python -m model_foundry.train "$CONFIG_PATH"
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo "End Time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED"
fi

echo "=========================================="

# ============================================================================
# Final GPU Status
# ============================================================================

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Final GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
fi

exit $EXIT_CODE
