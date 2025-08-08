#!/bin/bash
# Test both exp0 and exp1 chunking in sequence
# Usage: ./test_chunking_both.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Check if we're on the cluster
if [ ! -d "/labs/ferreiralab/thmorton/subject-drop-rearing" ]; then
    echo -e "${RED}ERROR: This script must be run on the cluster${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo -e "${MAGENTA}=========================================================${NC}"
echo -e "${MAGENTA}Testing Chunking for Both Experiments${NC}"
echo -e "${MAGENTA}=========================================================${NC}"
echo -e "Start time: ${YELLOW}$(date)${NC}"
echo ""

# Path configurations
PROJECT_ROOT="/labs/ferreiralab/thmorton/subject-drop-rearing"
CONTAINER="${PROJECT_ROOT}/singularity/ablation.sif"

# Check if container exists
if [ ! -f "${CONTAINER}" ]; then
    echo -e "${RED}ERROR: Container not found at ${CONTAINER}${NC}"
    exit 1
fi

# Change to project directory
cd "${PROJECT_ROOT}"

# Load modules if available
module load cuda/11.8 2>/dev/null || true
module load singularity 2>/dev/null || true

# Check GPU availability once
echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader || echo "No GPU detected"
else
    echo "nvidia-smi not found - GPU may not be available"
fi
echo ""

# Install tqdm if not available
singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    bash -c "pip list | grep -q tqdm || pip install -q tqdm"

# Test exp0_baseline
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}TESTING: experiment_0_baseline${NC}"
echo -e "${BLUE}=========================================================${NC}"

singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    python -u scripts/test_chunking_simple.py configs/experiment_0_baseline.yaml

EXP0_EXIT=$?

if [ $EXP0_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ exp0_baseline test completed${NC}"
else
    echo -e "${RED}✗ exp0_baseline test failed with exit code: $EXP0_EXIT${NC}"
fi

echo ""
echo ""

# Test exp1_remove_expletives
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}TESTING: experiment_1_remove_expletives${NC}"
echo -e "${BLUE}=========================================================${NC}"

singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    python -u scripts/test_chunking_simple.py configs/experiment_1_remove_expletives.yaml

EXP1_EXIT=$?

if [ $EXP1_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ exp1_remove_expletives test completed${NC}"
else
    echo -e "${RED}✗ exp1_remove_expletives test failed with exit code: $EXP1_EXIT${NC}"
fi

# Final summary
echo ""
echo -e "${MAGENTA}=========================================================${NC}"
echo -e "${MAGENTA}SUMMARY${NC}"
echo -e "${MAGENTA}=========================================================${NC}"

if [ $EXP0_EXIT -eq 0 ]; then
    echo -e "exp0_baseline: ${GREEN}✓ SUCCESS${NC}"
else
    echo -e "exp0_baseline: ${RED}✗ FAILED${NC}"
fi

if [ $EXP1_EXIT -eq 0 ]; then
    echo -e "exp1_remove_expletives: ${GREEN}✓ SUCCESS${NC}"
else
    echo -e "exp1_remove_expletives: ${RED}✗ FAILED${NC}"
fi

echo -e "End time: ${YELLOW}$(date)${NC}"
echo -e "${MAGENTA}=========================================================${NC}"

# Exit with failure if either test failed
if [ $EXP0_EXIT -ne 0 ] || [ $EXP1_EXIT -ne 0 ]; then
    exit 1
fi

exit 0