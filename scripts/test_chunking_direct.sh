#!/bin/bash
# Direct chunking test script - runs on head node or any node with GPU
# Usage: ./test_chunking_direct.sh <config_file>
# Example: ./test_chunking_direct.sh configs/experiment_1_remove_expletives.yaml

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if config file was provided
if [ $# -eq 0 ]; then
    echo -e "${RED}ERROR: No config file provided${NC}"
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/experiment_1_remove_expletives.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Check if we're on the cluster
if [ ! -d "/labs/ferreiralab/thmorton/subject-drop-rearing" ]; then
    echo -e "${RED}ERROR: This script must be run on the cluster${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Direct Chunking Test${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo -e "Config: ${GREEN}${CONFIG_FILE}${NC}"
echo -e "Start time: ${YELLOW}$(date)${NC}"
echo -e "${BLUE}=========================================================${NC}"

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

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}ERROR: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# Load modules if available (won't error if not available)
module load cuda/11.8 2>/dev/null || true
module load singularity 2>/dev/null || true

# Check GPU availability
echo ""
echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader || echo "No GPU detected"
else
    echo "nvidia-smi not found - GPU may not be available"
fi

# Run the chunking test
echo ""
echo -e "${GREEN}Running chunking test for: ${CONFIG_FILE}${NC}"
echo -e "${BLUE}=========================================================${NC}"

singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    python scripts/test_chunking_simple.py "${CONFIG_FILE}"

EXIT_CODE=$?

echo ""
echo -e "${BLUE}=========================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Test completed successfully${NC}"
else
    echo -e "${RED}✗ Test failed with exit code: $EXIT_CODE${NC}"
fi
echo -e "End time: ${YELLOW}$(date)${NC}"
echo -e "${BLUE}=========================================================${NC}"

exit $EXIT_CODE