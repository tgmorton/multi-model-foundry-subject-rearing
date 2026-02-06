# Corpus Descriptive Analysis — NRP/Nautilus Deployment

## Prerequisites

- `kubectl` configured for your NRP namespace
- GPU quota in namespace (1x T4 or better per job)

## How It Works

Each job uses a stock `pytorch/pytorch` image. An init container clones the repo,
then the main container installs Python deps and downloads the spaCy model at
startup. No custom Docker image build required.

## Setup

### 1. Create PVC for data storage

```bash
kubectl apply -f k8s/pvc.yaml
```

### 2. Upload corpus data

```bash
# Start a temporary pod to copy data
kubectl run data-upload --image=busybox --restart=Never --overrides='
{
  "spec": {
    "containers": [{
      "name": "data-upload",
      "image": "busybox",
      "command": ["sleep", "3600"],
      "volumeMounts": [{"name": "data", "mountPath": "/mnt/data"}]
    }],
    "volumes": [{
      "name": "data",
      "persistentVolumeClaim": {"claimName": "corpus-analysis-data"}
    }]
  }
}'

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/data-upload --timeout=60s

# Copy data
kubectl cp data/raw/ data-upload:/mnt/data/raw/

# Create output directories
kubectl exec data-upload -- mkdir -p /mnt/data/output/train_90M /mnt/data/output/test_10M /mnt/data/output/pull_10M
kubectl exec data-upload -- mkdir -p /mnt/data/checkpoints/train_90M /mnt/data/checkpoints/test_10M /mnt/data/checkpoints/pull_10M

# Clean up
kubectl delete pod data-upload
```

## Running Jobs

### Analyzer Mode (Aggregated Counts)

```bash
# Submit one at a time (they share the PVC with ReadWriteOnce)
kubectl apply -f k8s/job-train-90m.yaml

# Monitor
kubectl logs -f job/corpus-analysis-train-90m

# After completion, submit next
kubectl apply -f k8s/job-test-10m.yaml
kubectl apply -f k8s/job-pull-10m.yaml
```

### Annotation Mode (Per-Sentence Parquet)

Annotation mode produces sentence-level Parquet files for downstream queries
and influence function analysis. Uses more memory (6-8Gi) due to Parquet buffering.

```bash
# Run annotation pipeline
kubectl apply -f k8s/job-annotate-train-90m.yaml

# Monitor
kubectl logs -f job/corpus-annotate-train-90m

# After completion
kubectl apply -f k8s/job-annotate-test-10m.yaml
```

Output structure:
```
/mnt/data/output/{split}/annotated_corpus/
├── base/                      # Core annotations (sentence_id, text, tokens)
│   └── {split}.parquet
├── layers/                    # Per-layer annotations
│   ├── clause_structure/
│   ├── that_trace/
│   ├── pronouns/
│   └── ...
└── metadata.json
```

## Retrieving Output

```bash
# Start retrieval pod
kubectl run data-download --image=busybox --restart=Never --overrides='
{
  "spec": {
    "containers": [{
      "name": "data-download",
      "image": "busybox",
      "command": ["sleep", "3600"],
      "volumeMounts": [{"name": "data", "mountPath": "/mnt/data"}]
    }],
    "volumes": [{
      "name": "data",
      "persistentVolumeClaim": {"claimName": "corpus-analysis-data"}
    }]
  }
}'

kubectl wait --for=condition=Ready pod/data-download --timeout=60s
kubectl cp data-download:/mnt/data/output/ ./analysis/output/corpus_descriptives/data/
kubectl delete pod data-download
```

## Resource Notes

- `en_core_web_trf` uses ~4GB VRAM; T4 (16GB) is sufficient
- Memory request 8Gi, limit 12Gi covers spaCy + data in memory
- Each job processes one split sequentially through all 6 genre files
- Checkpointing is enabled; jobs can resume after preemption
- pip install + spaCy model download adds ~2-3 min startup overhead per job
