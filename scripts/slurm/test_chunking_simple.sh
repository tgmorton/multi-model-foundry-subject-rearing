#!/bin/bash
#SBATCH --job-name=test-chunk
#SBATCH --output=logs/test-chunk-%j.out
#SBATCH --error=logs/test-chunk-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=ferreiralab

# Simple chunking test script
# Usage: sbatch test_chunking_simple.sh <config_file>
# Example: sbatch test_chunking_simple.sh configs/experiment_1_remove_expletives.yaml

set -e
set -u

# Check if config file was provided
if [ $# -eq 0 ]; then
    echo "ERROR: No config file provided"
    echo "Usage: sbatch test_chunking_simple.sh <config_file>"
    echo "Example: sbatch test_chunking_simple.sh configs/experiment_1_remove_expletives.yaml"
    exit 1
fi

CONFIG_FILE="$1"

echo "========================================================="
echo "Chunking Test"
echo "Config: ${CONFIG_FILE}"
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

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Change to project directory
cd "${PROJECT_ROOT}"

# Run the chunking test
echo ""
echo "Running chunking test for: ${CONFIG_FILE}"
echo "========================================================="
singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    python scripts/test_chunking_simple.py "${CONFIG_FILE}"

echo ""
echo "========================================================="
echo "Test completed: $(date)"
echo "========================================================="