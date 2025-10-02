# Cross-Architecture Comparison Guide

This guide explains how to use Model Foundry's token counting and checkpoint alignment features to fairly compare different architectures (GPT-2, BERT, LSTM, GRU, RNN, Mamba) trained on the same data.

## Token Counting in Checkpoints

Every checkpoint now includes detailed token metrics in `metadata.json`:

```json
{
  "experiment_name": "gpt2_10M",
  "global_step": 5000,
  "epoch": 2,
  "token_metrics": {
    "total_tokens_processed": 25600000,
    "tokens_per_step": 5120,
    "effective_batch_size": 128,
    "sequence_length": 512,
    "estimated_tokens_at_step": 25600000
  },
  "training_config": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "gradient_accumulation_steps": 4
  },
  "model_config": {
    "architecture": "gpt2",
    "layers": 12,
    "hidden_size": 768,
    "attention_heads": 12
  }
}
```

### Token Metrics Explained

- **total_tokens_processed**: Actual count of tokens seen by the model (updated each step)
- **tokens_per_step**: Batch size × gradient accumulation × sequence length
- **effective_batch_size**: Batch size × gradient accumulation steps
- **sequence_length**: Maximum sequence length from config
- **estimated_tokens_at_step**: Theoretical token count (step × tokens_per_step)

## Fair Comparison Strategy

### 1. Standardize Training Configuration

Use identical hyperparameters across all architectures:

```yaml
# master_training_config.yaml (copy to all experiment configs)
training:
  learning_rate: 0.0001
  epochs: 10
  train_steps: null  # Auto-calculated from epochs × steps_per_epoch
  gradient_accumulation_steps: 4
  auto_generate_checkpoints: true
  first_epoch_checkpoints: 20
  subsequent_epochs_spacing: "log"
  min_checkpoints_per_epoch: 5

data:
  training_corpus: "data/processed/10M_tokens"
  batch_size: 32
  max_sequence_length: 512
```

### 2. Ensure Equal Token Exposure

The critical formula for fair comparison:

```
tokens_per_step = batch_size × gradient_accumulation_steps × sequence_length
```

**Example**:
- Batch size: 32
- Gradient accumulation: 4
- Sequence length: 512
- **Tokens per step**: 32 × 4 × 512 = **65,536 tokens/step**

All architectures with these settings will see exactly 65,536 tokens per training step.

### 3. Aligned Checkpoint Schedules

Generate identical checkpoint schedules:

```bash
# Generate schedule for first architecture
python scripts/generate_checkpoint_schedule.py \
  configs/gpt2_experiment.yaml \
  --first-epoch 20 \
  --spacing log \
  --min-per-epoch 5

# Copy the checkpoint_schedule list to all other configs
# configs/bert_experiment.yaml
# configs/mamba_experiment.yaml
# configs/lstm_experiment.yaml
# etc.
```

All models will checkpoint at the same training steps (e.g., step 100, 500, 1000, etc.).

### 4. Evaluate at Aligned Checkpoints

Compare models at identical token counts:

```bash
# All at step 5000 (same tokens seen)
python run_evaluation.py configs/gpt2_experiment.yaml --checkpoint checkpoint-5000
python run_evaluation.py configs/bert_experiment.yaml --checkpoint checkpoint-5000
python run_evaluation.py configs/mamba_experiment.yaml --checkpoint checkpoint-5000
python run_evaluation.py configs/lstm_experiment.yaml --checkpoint checkpoint-5000
```

### 5. Compare Using Token Metrics

Extract and compare token metrics from checkpoint metadata:

```python
import json
from pathlib import Path

def compare_checkpoints(checkpoint_paths):
    """Compare token metrics across architecture checkpoints."""
    for path in checkpoint_paths:
        metadata_path = Path(path) / "metadata.json"
        with open(metadata_path) as f:
            meta = json.load(f)

        arch = meta['model_config']['architecture']
        tokens = meta['token_metrics']['total_tokens_processed']
        step = meta['global_step']

        print(f"{arch:10s} | Step: {step:6d} | Tokens: {tokens:,} ({tokens/1e6:.2f}M)")

# Example usage
checkpoints = [
    "output/gpt2/checkpoint-5000",
    "output/bert/checkpoint-5000",
    "output/mamba/checkpoint-5000",
    "output/lstm/checkpoint-5000"
]
compare_checkpoints(checkpoints)
```

Output:
```
gpt2       | Step:   5000 | Tokens: 327,680,000 (327.68M)
bert       | Step:   5000 | Tokens: 327,680,000 (327.68M)
mamba      | Step:   5000 | Tokens: 327,680,000 (327.68M)
lstm       | Step:   5000 | Tokens: 327,680,000 (327.68M)
```

## WandB Integration

Track token metrics across architectures in WandB:

```yaml
logging:
  use_wandb: true
  project: "architecture-comparison"
  tags: ["10M_tokens", "fair_comparison"]
```

WandB will log `tokens_processed` at each step, allowing you to create comparison plots normalized by token count rather than wall-clock time.

## Example: Complete Experimental Setup

### Step 1: Create Master Config Template

```yaml
# configs/templates/base_experiment.yaml
experiment_name: "REPLACE_WITH_ARCHITECTURE_10M"

data:
  source_corpus: "data/raw/10M_tokens"
  training_corpus: "data/processed/10M_tokens"
  batch_size: 32
  max_sequence_length: 512

tokenizer:
  vocab_size: 32000
  tokenizer_type: "sentencepiece"  # Override for BERT (use "wordpiece")

training:
  learning_rate: 0.0001
  epochs: 10
  gradient_accumulation_steps: 4
  auto_generate_checkpoints: true
  first_epoch_checkpoints: 20
  subsequent_epochs_spacing: "log"
  min_checkpoints_per_epoch: 5

logging:
  use_wandb: true
  project: "architecture-comparison-10M"
  log_interval: 10
```

### Step 2: Create Architecture-Specific Configs

```yaml
# configs/gpt2_10M.yaml
experiment_name: "gpt2_10M"
# ... (copy from base_experiment.yaml)
model:
  architecture: "gpt2"
  transformer:
    layers: 12
    embedding_size: 768
    hidden_size: 768
    intermediate_hidden_size: 3072
    attention_heads: 12
    activation_function: "gelu"
    dropout: 0.1
    attention_dropout: 0.1
```

```yaml
# configs/mamba_10M.yaml
experiment_name: "mamba_10M"
# ... (copy from base_experiment.yaml)
model:
  architecture: "mamba"
  mamba:
    d_model: 768
    n_layers: 24
    d_state: 16
    d_conv: 4
    expand: 2
    dropout: 0.1
```

### Step 3: Generate Aligned Checkpoint Schedule

```bash
# Generate for one architecture
python scripts/generate_checkpoint_schedule.py configs/gpt2_10M.yaml

# Copy the generated checkpoint_schedule to all other configs
# Or use a script to sync them:
python scripts/sync_checkpoint_schedules.py configs/gpt2_10M.yaml configs/*.yaml
```

### Step 4: Train All Architectures

```bash
python model_foundry/train.py configs/gpt2_10M.yaml
python model_foundry/train.py configs/bert_10M.yaml
python model_foundry/train.py configs/mamba_10M.yaml
python model_foundry/train.py configs/lstm_10M.yaml
python model_foundry/train.py configs/gru_10M.yaml
python model_foundry/train.py configs/rnn_10M.yaml
```

### Step 5: Evaluate at Aligned Checkpoints

```bash
# Create evaluation script
for arch in gpt2 bert mamba lstm gru rnn; do
  for step in 1000 5000 10000 20000 50000; do
    python run_evaluation.py configs/${arch}_10M.yaml \
      --checkpoint checkpoint-${step} \
      --output-dir results/${arch}/step-${step}
  done
done
```

### Step 6: Analyze Results

```python
import json
import pandas as pd
from pathlib import Path

results = []
for arch in ['gpt2', 'bert', 'mamba', 'lstm', 'gru', 'rnn']:
    for step in [1000, 5000, 10000, 20000, 50000]:
        checkpoint = f"output/{arch}_10M/checkpoint-{step}"
        metadata_path = Path(checkpoint) / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)

            results.append({
                'architecture': arch,
                'step': step,
                'tokens_processed': meta['token_metrics']['total_tokens_processed'],
                'epoch': meta['epoch'],
                # Add evaluation metrics if available
            })

df = pd.DataFrame(results)
print(df.pivot(index='step', columns='architecture', values='tokens_processed'))
```

## Key Principles for Fair Comparison

1. **Same tokens per step** across all architectures
2. **Same checkpoint schedule** (steps, not epochs)
3. **Same hyperparameters** (learning rate, batch size, etc.)
4. **Same data** (identical training corpus)
5. **Compare at same token counts**, not wall-clock time or epochs

## Token Count Verification

Always verify token counts match across checkpoints:

```bash
# Quick check script
for arch in gpt2 bert mamba lstm; do
  tokens=$(jq '.token_metrics.total_tokens_processed' \
    output/${arch}_10M/checkpoint-5000/metadata.json)
  echo "$arch: $tokens tokens"
done
```

Expected output (all should match):
```
gpt2: 327680000 tokens
bert: 327680000 tokens
mamba: 327680000 tokens
lstm: 327680000 tokens
```

If token counts differ, check:
- Batch size is identical
- Gradient accumulation steps match
- Sequence length is the same
- Training started from step 0 (not resumed mid-training)

## Advanced: Resuming Training

When resuming from checkpoint, token counting continues correctly:

```python
# In training loop (automatic)
if resume_from_checkpoint:
    state = load_training_state()
    total_tokens_processed = state['total_tokens_processed']  # Restored
    # Training continues, incrementing token count
```

The `total_tokens_processed` is saved in `training_state.pt` and restored automatically.
