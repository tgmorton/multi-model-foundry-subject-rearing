#!/bin/bash
# Quick script to check the structure of tokenized datasets

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/labs/ferreiralab/thmorton/subject-drop-rearing"
cd "${PROJECT_ROOT}"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Tokenized Dataset Structure Check${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""

# Check exp0
echo -e "${YELLOW}=== EXP0_BASELINE_MODEL ===${NC}"
echo "Main directory contents:"
ls -la data/tokenized/exp0_baseline_model/ | head -10

if [ -d "data/tokenized/exp0_baseline_model/train" ]; then
    echo ""
    echo "Train subdirectory contents:"
    ls -la data/tokenized/exp0_baseline_model/train/ | head -10
fi

echo ""
echo -e "${YELLOW}=== EXP1_REMOVE_EXPLETIVES ===${NC}"
echo "Main directory contents:"
ls -la data/tokenized/exp1_remove_expletives/ | head -10

if [ -d "data/tokenized/exp1_remove_expletives/train" ]; then
    echo ""
    echo "Train subdirectory contents:"
    ls -la data/tokenized/exp1_remove_expletives/train/ | head -10
fi

echo ""
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Testing if datasets can be loaded${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Test loading with Python
singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --pwd /workspace \
    singularity/ablation.sif \
    python -c "
from datasets import load_from_disk
import os

print('\nTesting exp0 loading:')
try:
    # Try direct load
    ds = load_from_disk('data/tokenized/exp0_baseline_model')
    print(f'  ✓ Direct load successful: {len(ds)} samples')
except:
    try:
        # Try train subdirectory
        ds = load_from_disk('data/tokenized/exp0_baseline_model/train')
        print(f'  ✓ Train subdir load successful: {len(ds)} samples')
    except Exception as e:
        print(f'  ✗ Failed to load: {e}')

print('\nTesting exp1 loading:')
try:
    # Try direct load
    ds = load_from_disk('data/tokenized/exp1_remove_expletives')
    print(f'  ✓ Direct load successful: {len(ds)} samples')
except:
    try:
        # Try train subdirectory
        ds = load_from_disk('data/tokenized/exp1_remove_expletives/train')
        print(f'  ✓ Train subdir load successful: {len(ds)} samples')
    except Exception as e:
        print(f'  ✗ Failed to load: {e}')
"

echo ""
echo -e "${BLUE}=========================================================${NC}"