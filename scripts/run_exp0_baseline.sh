#!/bin/bash
# Experiment 0 Baseline Runner
# This script runs the complete workflow for the baseline experiment with 90M data

set -e

echo "=== Running Experiment 0: Baseline with 90M Data ==="

# Set up paths
BASE_DIR="$(pwd)"
CONFIG_FILE="$BASE_DIR/configs/experiment_0_baseline_90M.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "Configuration: $CONFIG_FILE"
echo "Base directory: $BASE_DIR"

# Step 1: Generate checkpoint schedule
echo ""
echo "=== Step 1: Generating checkpoint schedule ==="
python scripts/generate_checkpoint_schedule.py "$CONFIG_FILE"

# Step 2: Train tokenizer
echo ""
echo "=== Step 2: Training tokenizer ==="
python -m model_foundry.tokenizer.train_tokenizer \
    --config "$CONFIG_FILE" \
    --base_dir "$BASE_DIR"

# Step 3: Tokenize dataset
echo ""
echo "=== Step 3: Tokenizing dataset ==="
python -m model_foundry.tokenizer.tokenize_dataset \
    --config "$CONFIG_FILE" \
    --base_dir "$BASE_DIR"

# Step 4: Run training
echo ""
echo "=== Step 4: Running training ==="
python -m model_foundry.trainer "$CONFIG_FILE"

echo ""
echo "=== Experiment 0 Baseline Complete ==="
echo "✓ All steps completed successfully!" 