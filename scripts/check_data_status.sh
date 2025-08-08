#!/bin/bash
# Script to check the status of data directories for both experiments
# Usage: ./check_data_status.sh

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
    exit 1
fi

PROJECT_ROOT="/labs/ferreiralab/thmorton/subject-drop-rearing"
cd "${PROJECT_ROOT}"

echo -e "${MAGENTA}=========================================================${NC}"
echo -e "${MAGENTA}Data Directory Status Check${NC}"
echo -e "${MAGENTA}=========================================================${NC}"
echo ""

# Function to check directory
check_directory() {
    local dir=$1
    local label=$2
    
    echo -e "${BLUE}Checking: ${label}${NC}"
    echo "  Path: $dir"
    
    if [ -d "$dir" ]; then
        echo -e "  Status: ${GREEN}✓ EXISTS${NC}"
        
        # Count files
        file_count=$(find "$dir" -type f 2>/dev/null | wc -l)
        echo "  Files: $file_count"
        
        # Get size
        dir_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  Size: $dir_size"
        
        # Check for specific files
        if [ -f "$dir/dataset_info.json" ]; then
            echo -e "  Dataset info: ${GREEN}✓${NC}"
        fi
        if [ -f "$dir/state.json" ]; then
            echo -e "  State file: ${GREEN}✓${NC}"
        fi
        if [ -d "$dir/data" ]; then
            echo -e "  Data folder: ${GREEN}✓${NC}"
        fi
        
        # List immediate contents
        echo "  Contents:"
        ls -la "$dir" 2>/dev/null | head -5 | sed 's/^/    /'
        
    else
        echo -e "  Status: ${RED}✗ NOT FOUND${NC}"
    fi
    echo ""
}

# Check exp0 directories
echo -e "${YELLOW}=== EXPERIMENT 0 (BASELINE) ===${NC}"
echo ""

check_directory "data/raw/train_90M" "Raw training data"
check_directory "data/tokenized/exp0_baseline_model" "Tokenized data"
check_directory "data/tokenized/exp0_baseline" "Tokenized data (alt name)"
check_directory "data/chunked/exp0_baseline_model" "Chunked data"
check_directory "data/chunked/exp0_baseline" "Chunked data (alt name)"

# Check exp1 directories
echo -e "${YELLOW}=== EXPERIMENT 1 (REMOVE EXPLETIVES) ===${NC}"
echo ""

check_directory "data/processed/exp1_remove_expletives" "Processed data"
check_directory "data/tokenized/exp1_remove_expletives" "Tokenized data"
check_directory "data/chunked/exp1_remove_expletives" "Chunked data"

# Check tokenizers
echo -e "${YELLOW}=== TOKENIZERS ===${NC}"
echo ""

check_directory "tokenizers/exp0_baseline" "Exp0 tokenizer"
check_directory "tokenizers/exp0_baseline_model" "Exp0 tokenizer (alt)"
check_directory "tokenizers/exp1_remove_expletives" "Exp1 tokenizer"

# Summary
echo -e "${MAGENTA}=========================================================${NC}"
echo -e "${MAGENTA}SUMMARY${NC}"
echo -e "${MAGENTA}=========================================================${NC}"

# Quick status checks
echo ""
echo "Quick Status:"

# Exp0
if [ -d "data/tokenized/exp0_baseline_model" ] || [ -d "data/tokenized/exp0_baseline" ]; then
    echo -e "  Exp0 tokenized: ${GREEN}✓${NC}"
else
    echo -e "  Exp0 tokenized: ${RED}✗${NC}"
fi

if [ -d "data/chunked/exp0_baseline_model" ] || [ -d "data/chunked/exp0_baseline" ]; then
    echo -e "  Exp0 chunked: ${GREEN}✓${NC}"
else
    echo -e "  Exp0 chunked: ${RED}✗${NC}"
fi

# Exp1
if [ -d "data/tokenized/exp1_remove_expletives" ]; then
    echo -e "  Exp1 tokenized: ${GREEN}✓${NC}"
else
    echo -e "  Exp1 tokenized: ${RED}✗ - Need to run tokenization${NC}"
fi

if [ -d "data/chunked/exp1_remove_expletives" ]; then
    echo -e "  Exp1 chunked: ${GREEN}✓${NC}"
else
    echo -e "  Exp1 chunked: ${RED}✗${NC}"
fi

echo ""
echo -e "${MAGENTA}=========================================================${NC}"