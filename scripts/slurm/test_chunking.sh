#!/bin/bash
#SBATCH --job-name=test-chunking
#SBATCH --output=logs/test-chunking-%j.out
#SBATCH --error=logs/test-chunking-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=ferreiralab

# Script to test chunking behavior for both experiments
# This will help debug why exp1 has only 582 chunks vs exp0's ~5750

set -e
set -u

echo "========================================================="
echo "Starting chunking analysis test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================="

# Set up environment
module load cuda/11.8
module load singularity

# Path configurations
PROJECT_ROOT="/labs/ferreiralab/thmorton/subject-drop-rearing"
CONTAINER="${PROJECT_ROOT}/singularity/ablation.sif"
SCRIPT_PATH="scripts/test_chunking.py"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Change to project directory
cd "${PROJECT_ROOT}"

echo ""
echo "Testing chunking WITHOUT actual re-chunking (simulation only):"
echo "================================================================"
singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    python "${SCRIPT_PATH}" \
        --exp0-config configs/experiment_0_baseline.yaml \
        --exp1-config configs/experiment_1_remove_expletives.yaml \
        --chunk-size 1000

echo ""
echo "========================================================="
echo "Would you like to test with ACTUAL re-chunking?"
echo "This will recreate the chunked datasets."
echo "Uncomment the section below to enable:"
echo "========================================================="

# Uncomment this section to test with actual re-chunking
# This will DELETE and RECREATE the chunked datasets!
# 
# echo ""
# echo "Testing with ACTUAL re-chunking (will recreate chunks):"
# echo "================================================================"
# singularity exec --nv \
#     --bind "${PROJECT_ROOT}:/workspace" \
#     --pwd /workspace \
#     "${CONTAINER}" \
#     python "${SCRIPT_PATH}" \
#         --exp0-config configs/experiment_0_baseline.yaml \
#         --exp1-config configs/experiment_1_remove_expletives.yaml \
#         --chunk-size 1000 \
#         --test-actual

echo ""
echo "========================================================="
echo "Test completed: $(date)"
echo "Check the output above for the analysis results"
echo "Results also saved to: chunking_analysis_results.json"
echo "========================================================="