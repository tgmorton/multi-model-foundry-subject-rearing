#!/bin/bash
# Script to tokenize exp1 dataset
# Usage: ./tokenize_exp1.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're on the cluster
if [ ! -d "/labs/ferreiralab/thmorton/subject-drop-rearing" ]; then
    echo -e "${RED}ERROR: This script must be run on the cluster${NC}"
    exit 1
fi

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Tokenizing Experiment 1 Dataset${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""

PROJECT_ROOT="/labs/ferreiralab/thmorton/subject-drop-rearing"
CONTAINER="${PROJECT_ROOT}/singularity/ablation.sif"
CONFIG="configs/experiment_1_remove_expletives.yaml"

cd "${PROJECT_ROOT}"

# Load modules
module load cuda/11.8 2>/dev/null || true
module load singularity 2>/dev/null || true

# Check if processed data exists
if [ ! -d "data/processed/exp1_remove_expletives" ]; then
    echo -e "${RED}ERROR: Processed data not found at data/processed/exp1_remove_expletives${NC}"
    echo "Please run the preprocessing step first to create the ablated dataset."
    exit 1
fi

echo -e "${YELLOW}Checking processed data...${NC}"
echo -n "  Counting files in processed directory: "
file_count=$(find data/processed/exp1_remove_expletives -name "*.train" -o -name "*.test" | wc -l)
echo "$file_count files found"

# Check if tokenizer exists
if [ ! -d "tokenizers/exp1_remove_expletives" ]; then
    echo -e "${YELLOW}Tokenizer not found. Training tokenizer first...${NC}"
    
    singularity exec --nv \
        --bind "${PROJECT_ROOT}:/workspace" \
        --pwd /workspace \
        "${CONTAINER}" \
        python -m model_foundry.cli train-tokenizer "${CONFIG}"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Tokenizer training failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Tokenizer trained successfully${NC}"
else
    echo -e "${GREEN}✓ Tokenizer already exists${NC}"
fi

# Run tokenization
echo ""
echo -e "${YELLOW}Running tokenization...${NC}"
echo -e "${BLUE}=========================================================${NC}"

singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    "${CONTAINER}" \
    python -m model_foundry.cli tokenize-dataset "${CONFIG}"

EXIT_CODE=$?

echo ""
echo -e "${BLUE}=========================================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Tokenization completed successfully${NC}"
    
    # Check the output
    if [ -d "data/tokenized/exp1_remove_expletives" ]; then
        echo ""
        echo "Tokenized data created at: data/tokenized/exp1_remove_expletives"
        echo -n "  Dataset size: "
        du -sh data/tokenized/exp1_remove_expletives | cut -f1
    fi
    
    echo ""
    echo "Next step: Run the chunking test"
    echo "  ./scripts/test_chunking_direct.sh configs/experiment_1_remove_expletives.yaml"
else
    echo -e "${RED}✗ Tokenization failed with exit code: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi