# Hardware-Specific Scripts for Subject-Drop Rearing Experiments

This directory contains scripts organized by hardware type for running ablative experiments on different systems.

## Directory Structure

```
scripts/
├── p6000/                    # P6000 GPU cluster scripts
│   ├── run_ablation_experiment.sh
│   └── submit_experiment.sh
├── a5000/                    # A5000 GPU cluster scripts
│   ├── run_ablation_experiment.sh
│   └── submit_experiment.sh
├── titanx/                   # TitanX headnode scripts
│   └── run_direct_full_pipeline.sh
├── 3070ti/                   # RTX 3070 Ti Windows scripts
│   ├── README.md
│   ├── run_preprocessing.bat
│   ├── run_tokenizer.bat
│   ├── run_tokenization.bat
│   └── run_full_preprocessing.bat
└── README.md                 # This file
```

## Hardware-Specific Scripts

### P6000 Cluster Scripts (`scripts/p6000/`)

**Use for**: P6000 GPU nodes on the cluster
**Resources**: 8 CPUs, 32GB RAM, P6000 GPU
**Partition**: `general_gpu_p6000`

```bash
# Submit preprocessing job
./scripts/p6000/submit_experiment.sh experiment_1_remove_expletives preprocess

# Submit full pipeline job
./scripts/p6000/submit_experiment.sh experiment_1_remove_expletives full-pipeline

# Submit training-only job
./scripts/p6000/submit_experiment.sh experiment_0_baseline run
```

### A5000 Cluster Scripts (`scripts/a5000/`)

**Use for**: A5000 GPU nodes on the cluster
**Resources**: 12 CPUs, 64GB RAM, A5000 GPU
**Partition**: `general_gpu_a5000`

```bash
# Submit preprocessing job
./scripts/a5000/submit_experiment.sh experiment_1_remove_expletives preprocess

# Submit full pipeline job
./scripts/a5000/submit_experiment.sh experiment_1_remove_expletives full-pipeline

# Submit training-only job
./scripts/a5000/submit_experiment.sh experiment_0_baseline run
```

### TitanX Headnode Scripts (`scripts/titanx/`)

**Use for**: Direct execution on headnode with TitanX GPU
**Resources**: Direct access to TitanX GPU
**No SLURM**: Runs directly on headnode

```bash
# Run full pipeline directly
./scripts/titanx/run_direct_full_pipeline.sh experiment_1_remove_expletives

# Run with resume option
./scripts/titanx/run_direct_full_pipeline.sh experiment_1_remove_expletives --resume
```

### RTX 3070 Ti Windows Scripts (`scripts/3070ti/`)

**Use for**: Windows machine with RTX 3070 Ti
**Purpose**: Preprocessing tasks only (corpus ablations, tokenizer training, tokenization)
**OS**: Windows 10/11

```cmd
# Run preprocessing only
scripts\3070ti\run_preprocessing.bat experiment_1_remove_expletives

# Run tokenizer training
scripts\3070ti\run_tokenizer.bat experiment_1_remove_expletives

# Run dataset tokenization
scripts\3070ti\run_tokenization.bat experiment_1_remove_expletives

# Run complete preprocessing pipeline
scripts\3070ti\run_full_preprocessing.bat experiment_1_remove_expletives
```

## Available Phases

1. **preprocess**: Run dataset manipulation pipeline (corpus ablations)
2. **train-tokenizer**: Train SentencePiece tokenizer
3. **tokenize-dataset**: Tokenize training data
4. **run**: Execute model training
5. **full-pipeline**: Run all phases in sequence

## Job Submission Examples

### Cluster Jobs (P6000/A5000)

```bash
# Run preprocessing for expletive removal experiment
./scripts/a5000/submit_experiment.sh experiment_1_remove_expletives preprocess

# Run full pipeline for baseline experiment
./scripts/p6000/submit_experiment.sh experiment_0_baseline full-pipeline

# Run training only with custom job name
./scripts/a5000/submit_experiment.sh experiment_0_baseline run baseline_training
```

### Direct Execution (TitanX)

```bash
# Run complete pipeline on headnode
./scripts/titanx/run_direct_full_pipeline.sh experiment_1_remove_expletives
```

### Windows Preprocessing (3070 Ti)

```cmd
# Run complete preprocessing pipeline
scripts\3070ti\run_full_preprocessing.bat experiment_1_remove_expletives
```

## Monitoring Jobs

### Cluster Jobs

```bash
# Check job status
squeue -u $USER

# View job logs
tail -f logs/subject-drop-experiment_1_remove_expletives-preprocess-<job_id>.out

# Cancel a job
scancel <job_id>
```

### Windows Jobs

Monitor through:
- Task Manager (GPU usage)
- Command prompt output
- Log files in `logs/` directory

## Configuration

### Required Paths

Update these paths in the respective scripts to match your environment:

**Cluster Scripts**:
```bash
HOST_PROJECT_DIR="/labs/ferreiralab/thmorton/subject-drop-rearing"
HOST_ABLATION_SIF_PATH="/labs/ferreiralab/thmorton/subject-drop-rearing/singularity/ablation.sif"
HOST_TRAINING_SIF_PATH="/labs/ferreiralab/thmorton/subject-drop-rearing/singularity/training.sif"
```

**Windows Scripts**:
```batch
set PROJECT_DIR=C:\path\to\your\subject-drop-rearing
```

### SLURM Parameters

| Hardware | Partition | CPUs | Memory | Time Limit |
|----------|-----------|------|--------|------------|
| P6000 | `general_gpu_p6000` | 8 | 32GB | 7 days |
| A5000 | `general_gpu_a5000` | 12 | 64GB | 7 days |

## Container Requirements

### Ablation Container (ablation.sif)
- Used for: preprocessing, tokenizer training, dataset tokenization
- Contains: spaCy, transformers, sentencepiece, and other preprocessing dependencies
- Includes: en_core_web_trf spaCy model for linguistic analysis

### Training Container (training.sif)
- Used for: model training
- Contains: PyTorch with CUDA support, transformers, and training dependencies
- Optimized for: GPU training with CUDA 11.8

## Experiment Workflow

### Cluster Workflow

1. **Create Experiment Config**: Add a new `.yaml` file to `configs/`
2. **Submit Preprocessing**: Run dataset manipulation pipeline
3. **Submit Tokenizer Training**: Train experiment-specific tokenizer
4. **Submit Dataset Tokenization**: Tokenize training data
5. **Submit Model Training**: Train the language model
6. **Evaluate Results**: Use evaluation scripts on trained models

### Windows Workflow (3070 Ti)

1. **Setup Windows Environment**: Follow `scripts/3070ti/README.md`
2. **Run Preprocessing**: Execute corpus ablations
3. **Train Tokenizer**: Create experiment-specific tokenizer
4. **Tokenize Dataset**: Process training data
5. **Transfer to Cluster**: Move processed data for training
6. **Cluster Training**: Use cluster scripts for model training

## Troubleshooting

### Common Issues

1. **Container not found**: Ensure singularity images are in the correct path
2. **Config file not found**: Check that config files exist in `configs/`
3. **Permission denied**: Make scripts executable with `chmod +x scripts/*.sh`
4. **Out of memory**: Reduce batch size or increase memory allocation
5. **Time limit exceeded**: Increase time limit or split into smaller jobs

### Debugging

```bash
# Test script locally (without SLURM)
bash scripts/a5000/run_ablation_experiment.sh experiment_0_baseline validate-config

# Check container functionality
singularity exec --bind .:/workspace singularity/ablation.sif python -m model_foundry.cli info

# Validate config file
python -m model_foundry.cli validate-config configs/experiment_0_baseline.yaml
```

## Example Experiments

### Baseline Experiment
- **Config**: `experiment_0_baseline.yaml`
- **Goal**: Control experiment with no ablations
- **Command**: `./scripts/a5000/submit_experiment.sh experiment_0_baseline full-pipeline`

### Expletive Removal Experiment
- **Config**: `experiment_1_remove_expletives.yaml`
- **Goal**: Test effect of expletive presence on subject-drop acquisition
- **Command**: `./scripts/a5000/submit_experiment.sh experiment_1_remove_expletives full-pipeline`

## Log Files

Logs are saved to `logs/` directory with naming pattern:
- `subject-drop-<config>-<phase>-<job_id>.out` (stdout)
- `subject-drop-<config>-<phase>-<job_id>.err` (stderr)

Example: `logs/subject-drop-experiment_1_remove_expletives-preprocess-12345.out`

## Hardware Recommendations

### For Preprocessing Tasks
- **Windows 3070 Ti**: Good for preprocessing, tokenizer training, dataset tokenization
- **Cluster A5000**: Best for all tasks, especially training
- **Cluster P6000**: Good for all tasks, slightly less memory than A5000

### For Training Tasks
- **Cluster A5000**: Recommended for model training
- **Cluster P6000**: Good for model training
- **TitanX Headnode**: Good for quick experiments or testing
- **Windows 3070 Ti**: Not recommended for training (limited VRAM)

## Performance Expectations

| Task | 3070 Ti | P6000 | A5000 | TitanX |
|------|---------|-------|-------|--------|
| Preprocessing | 2-4 hours | 1-2 hours | 1-2 hours | 1-2 hours |
| Tokenizer | 30 min | 15 min | 15 min | 15 min |
| Tokenization | 1 hour | 30 min | 30 min | 30 min |
| Training | Not recommended | 2-4 days | 2-4 days | 3-5 days | 