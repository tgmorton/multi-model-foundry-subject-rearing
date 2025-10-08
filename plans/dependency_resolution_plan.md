# Dependency Resolution Plan

## Problem Statement

There are two conflicting dependency requirements in the project:

1. **spaCy 3.8.7** (used by preprocessing) requires **numpy >= 2.0**
2. **transformers** (used by model_foundry) requires **numpy < 2.0**

Currently, if numpy 2.x is installed, preprocessing works but model_foundry tests fail. If numpy 1.x is installed, model_foundry works but spaCy may have issues.

## Current Status

- ✅ Preprocessing module is now **independent** of model_foundry (no cross-imports)
- ✅ All 79 preprocessing tests pass with numpy 2.2.6
- ❌ model_foundry tests fail with numpy 2.2.6 due to transformers incompatibility

## Solution Options

### Option 1: Separate Virtual Environments (Recommended)

**Approach**: Use separate conda/venv environments for preprocessing and model training.

**Implementation**:
```bash
# Environment 1: Preprocessing
conda create -n preprocessing python=3.10
conda activate preprocessing
pip install spacy>=3.8 numpy>=2.0 tqdm pydantic
python -m spacy download en_core_web_sm

# Environment 2: Model Training
conda create -n model_foundry python=3.10
conda activate model_foundry
pip install torch transformers "numpy<2.0" pydantic
```

**Workflow**:
1. Activate `preprocessing` environment → Run ablation pipelines → Generate processed corpora
2. Activate `model_foundry` environment → Train models on processed corpora

**Pros**:
- Clean separation of concerns
- No dependency conflicts
- Each module uses optimal versions
- Easy to maintain

**Cons**:
- Requires environment switching
- Slightly more setup overhead

### Option 2: Downgrade spaCy (Temporary Workaround)

**Approach**: Use an older spaCy version (< 3.8) that works with numpy < 2.0.

**Implementation**:
```bash
pip install "spacy<3.8" "numpy<2.0"
python -m spacy download en_core_web_sm
```

**Pros**:
- Single environment
- Quick fix

**Cons**:
- Miss out on spaCy 3.8+ improvements
- Not a long-term solution (spaCy will require numpy 2.x going forward)
- May have compatibility issues with newer Python versions

### Option 3: Wait for transformers to Support numpy 2.x

**Approach**: Monitor transformers releases and upgrade when numpy 2.x support is added.

**Timeline**: Likely Q2-Q3 2025 based on HuggingFace roadmap

**Implementation**:
- Keep preprocessing and model_foundry separate for now
- Use Option 1 (separate environments) in the interim
- Merge environments once transformers supports numpy 2.x

**Pros**:
- Eventually allows single environment
- Gets both libraries at latest versions

**Cons**:
- Requires waiting (uncertain timeline)
- Still need separate environments until then

### Option 4: Docker Containers (Production-Ready)

**Approach**: Create separate Docker containers for preprocessing and training workflows.

**Implementation**:
```dockerfile
# Dockerfile.preprocessing
FROM python:3.10
RUN pip install spacy>=3.8 numpy>=2.0 tqdm pydantic
RUN python -m spacy download en_core_web_sm
COPY preprocessing/ /app/preprocessing/
WORKDIR /app
ENTRYPOINT ["python", "-m", "preprocessing.run_ablation"]

# Dockerfile.training
FROM python:3.10
RUN pip install torch transformers "numpy<2.0" pydantic
COPY model_foundry/ /app/model_foundry/
WORKDIR /app
ENTRYPOINT ["python", "-m", "model_foundry.train"]
```

**Pros**:
- Production-ready
- Fully isolated dependencies
- Reproducible builds
- Easy deployment to SLURM/cloud

**Cons**:
- More infrastructure overhead
- Requires Docker knowledge
- Larger initial setup time

## Recommended Approach

**Phase 1 (Immediate)**: Option 1 - Separate Virtual Environments
- Quickest solution with no code changes
- Leverages existing independence of preprocessing module
- Works well with current workflow

**Phase 2 (6-12 months)**: Option 3 - Wait for transformers numpy 2.x support
- Monitor HuggingFace transformers releases
- When available, merge to single environment

**Phase 3 (Production)**: Option 4 - Docker Containers
- Once workflows are stable, containerize for deployment
- Especially useful for SLURM cluster deployment

## Implementation Steps for Option 1

### 1. Create Environment Configuration Files

Create `environment_preprocessing.yml`:
```yaml
name: preprocessing
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - spacy>=3.8.7
    - numpy>=2.0
    - tqdm
    - pydantic>=2.0
    - pytest
```

Create `environment_model_foundry.yml`:
```yaml
name: model_foundry
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - pip
  - pip:
    - transformers
    - "numpy<2.0"
    - pydantic>=2.0
    - pytest
```

### 2. Update Documentation

Add to `README.md`:
```markdown
## Environment Setup

This project uses two separate environments due to dependency conflicts:

### Preprocessing Environment
conda env create -f environment_preprocessing.yml
conda activate preprocessing
python -m spacy download en_core_web_sm

### Model Training Environment
conda env create -f environment_model_foundry.yml
conda activate model_foundry
```

### 3. Create Helper Scripts

Create `scripts/run_preprocessing.sh`:
```bash
#!/bin/bash
conda activate preprocessing
python -m preprocessing.run_ablation "$@"
```

Create `scripts/run_training.sh`:
```bash
#!/bin/bash
conda activate model_foundry
python scripts/train_*.py "$@"
```

### 4. Update CI/CD

If using GitHub Actions or similar:
```yaml
jobs:
  test-preprocessing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
        with:
          environment-file: environment_preprocessing.yml
      - run: pytest preprocessing/tests/

  test-model-foundry:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
        with:
          environment-file: environment_model_foundry.yml
      - run: pytest model_foundry/tests/
```

## Testing the Solution

### Verify Preprocessing Environment
```bash
conda activate preprocessing
python -m pytest preprocessing/tests/ -v
# Should see: 79 passed
```

### Verify Model Training Environment
```bash
conda activate model_foundry
python -m pytest model_foundry/tests/unit/ -v
# Should see: all model_foundry tests passing
```

## Monitoring transformers Updates

Watch for numpy 2.x support:
- GitHub: https://github.com/huggingface/transformers/issues
- Releases: https://github.com/huggingface/transformers/releases
- Search for: "numpy 2" or "numpy 2.0 support"

## Notes

- The preprocessing module's independence from model_foundry (achieved in Phase 1-2) makes this separation clean and maintainable
- No code changes required - purely an environment/deployment concern
- Both modules can continue to evolve independently
- When transformers adds numpy 2.x support, simply merge the environment files

## Estimated Timeline

- **Option 1 Setup**: 1-2 hours (creating environments, testing)
- **Option 4 Setup**: 1-2 days (Docker setup, testing, documentation)
- **transformers numpy 2.x support**: Unknown (check Q2-Q3 2025)

## Priority

**Medium** - Current workaround (separate environments) is viable and doesn't block development. Can be formalized when convenient.
