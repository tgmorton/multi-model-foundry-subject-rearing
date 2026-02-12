#!/bin/bash

# --- Submit Experiment Script for P6000 ---
# This script submits jobs to the P6000 partition

# Exit on any error
set -e

# --- Script Usage ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_name_without_yaml_extension> <phase> [job_name]"
    echo "Phases: preprocess, train-tokenizer, tokenize-dataset, run, full-pipeline"
    echo "Example: $0 experiment_1_remove_expletives preprocess"
    echo "Example: $0 experiment_0_baseline run baseline_training"
    exit 1
fi

CONFIG_NAME="$1"
PHASE="$2"
JOB_NAME="${3:-subject-drop-${CONFIG_NAME}-${PHASE}}"

# --- Environment Setup ---
echo "=== Submitting P6000 Job ==="
echo "Config: $CONFIG_NAME"
echo "Phase: $PHASE"
echo "Job Name: $JOB_NAME"
echo "Hardware: P6000"
echo "========================================================"

# --- Submit Job ---
echo "Submitting job to P6000 partition..."
sbatch \
    --job-name="$JOB_NAME" \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=24:00:00 \
    --output=../logs/%x-%j.out \
    --error=../logs/%x-%j.err \
    run_ablation_experiment.sh "$CONFIG_NAME" "$PHASE"

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully!"
    echo "Monitor with: squeue -u $USER"
    echo "View logs with: tail -f logs/${JOB_NAME}-<job_id>.out"
else
    echo "✗ Job submission failed!"
    exit 1
fi 