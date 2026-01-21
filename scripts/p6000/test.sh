#!/bin/bash
#SBATCH --job-name=test-modules
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=0-0:10:00

echo "Testing module availability..."

# Test singularity
if module load singularity/4.1.1 2>/dev/null; then
    echo "✓ singularity/4.1.1 available"
    module unload singularity/4.1.1
else
    echo "✗ singularity/4.1.1 not available"
fi

# Test cuda
if module load cuda/11.8 2>/dev/null; then
    echo "✓ cuda/11.8 available"
    module unload cuda/11.8
else
    echo "✗ cuda/11.8 not available"
fi

# List available modules
echo "Available modules:"
module avail 2>&1 | grep -E "(singularity|cuda)" || echo "No singularity or cuda modules found"