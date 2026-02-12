# Windows Setup for RTX 3070 Ti - Preprocessing Pipeline

This guide explains how to set up the subject-drop rearing codebase on a Windows machine with an RTX 3070 Ti for running preprocessing (ablation) tasks.

## System Requirements

- **OS**: Windows 10/11
- **GPU**: RTX 3070 Ti (8GB VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space
- **Python**: 3.8-3.10

## Installation Steps

### 1. Install Python and Git

1. **Download Python 3.10** from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation
   - Install for all users

2. **Download Git** from [git-scm.com](https://git-scm.com/download/win)
   - Use default settings

3. **Verify installation**:
   ```cmd
   python --version
   git --version
   ```

### 2. Install CUDA and cuDNN

1. **Download CUDA Toolkit 11.8** from [NVIDIA](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - Choose Windows → x86_64 → 10/11 → exe (local)
   - Install with default settings

2. **Download cuDNN 8.6** from [NVIDIA](https://developer.nvidia.com/cudnn)
   - Requires free NVIDIA account
   - Extract and copy files to CUDA installation directory

3. **Verify CUDA installation**:
   ```cmd
   nvcc --version
   ```

### 3. Clone the Repository

```cmd
git clone https://github.com/your-username/subject-drop-rearing.git
cd subject-drop-rearing
```

### 4. Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### 5. Install Dependencies

```cmd
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 6. Install spaCy and Models

```cmd
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

### 7. Install Additional Dependencies

```cmd
pip install sentencepiece transformers datasets tqdm pyyaml typer
```

## Configuration

### 1. Update Paths

Edit `scripts/3070ti/run_preprocessing.bat` and update the paths:

```batch
set PROJECT_DIR=C:\path\to\your\subject-drop-rearing
set CONFIG_FILE=configs\experiment_1_remove_expletives.yaml
```

### 2. Create Data Directory Structure

```cmd
mkdir data\raw
mkdir data\processed
mkdir data\tokenized
mkdir data\chunked
mkdir logs
```

### 3. Download BabyLM Dataset

1. **Download** the BabyLM dataset from [Hugging Face](https://huggingface.co/datasets/babylm/babylm_100M)
2. **Extract** to `data/raw/train_90M/`
3. **Verify** the structure:
   ```
   data/raw/train_90M/
   ├── open_subtitles.train
   ├── bnc_spoken.train
   ├── gutenberg.train
   ├── childes.train
   ├── simple_wiki.train
   └── switchboard.train
   ```

## Usage

### Running Preprocessing Tasks

1. **Activate virtual environment**:
   ```cmd
   venv\Scripts\activate
   ```

2. **Run preprocessing for experiment 1**:
   ```cmd
   scripts\3070ti\run_preprocessing.bat experiment_1_remove_expletives
   ```

3. **Run tokenizer training**:
   ```cmd
   scripts\3070ti\run_tokenizer.bat experiment_1_remove_expletives
   ```

4. **Run dataset tokenization**:
   ```cmd
   scripts\3070ti\run_tokenization.bat experiment_1_remove_expletives
   ```

### Available Scripts

- `run_preprocessing.bat` - Run corpus ablations
- `run_tokenizer.bat` - Train SentencePiece tokenizer
- `run_tokenization.bat` - Tokenize dataset
- `run_full_preprocessing.bat` - Run all preprocessing steps

## Hardware-Specific Optimizations

### Memory Management

The RTX 3070 Ti has 8GB VRAM. For optimal performance:

1. **Reduce batch size** in config files:
   ```yaml
   data:
     batch_size: 128  # Reduced from 256
   ```

2. **Use gradient checkpointing**:
   ```yaml
   model:
     gradient_checkpointing: true
   ```

3. **Enable memory optimization**:
   ```cmd
   set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

### Performance Tips

1. **Close other applications** while running preprocessing
2. **Monitor GPU memory** with Task Manager
3. **Use SSD storage** for faster I/O
4. **Increase page file** if needed

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size
   - Close other applications
   - Restart computer

2. **Python not found**:
   - Check PATH environment variable
   - Reinstall Python with "Add to PATH"

3. **spaCy model not found**:
   ```cmd
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_trf
   ```

4. **Permission denied**:
   - Run Command Prompt as Administrator
   - Check antivirus settings

### Debugging

1. **Check GPU status**:
   ```cmd
   nvidia-smi
   ```

2. **Test PyTorch CUDA**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

3. **Test spaCy**:
   ```python
   import spacy
   nlp = spacy.load("en_core_web_sm")
   doc = nlp("This is a test sentence.")
   print([(token.text, token.pos_) for token in doc])
   ```

## File Structure

After setup, your directory should look like:

```
subject-drop-rearing/
├── data/
│   ├── raw/train_90M/
│   ├── processed/
│   ├── tokenized/
│   └── chunked/
├── configs/
│   ├── experiment_0_baseline.yaml
│   ├── experiment_1_remove_expletives.yaml
│   └── ...
├── scripts/
│   └── 3070ti/
│       ├── README.md
│       ├── run_preprocessing.bat
│       ├── run_tokenizer.bat
│       └── run_tokenization.bat
├── preprocessing/
│   ├── remove_expletives.py
│   ├── impoverish_determiners.py
│   └── ...
├── venv/
└── requirements.txt
```

## Performance Expectations

### RTX 3070 Ti Performance

- **Preprocessing speed**: ~100-200 MB/min
- **Tokenizer training**: ~10-30 minutes
- **Dataset tokenization**: ~30-60 minutes
- **Memory usage**: 4-6 GB VRAM typical

### Comparison with Cluster

| Task | 3070 Ti | Cluster (A5000) |
|------|---------|-----------------|
| Preprocessing | 2-4 hours | 1-2 hours |
| Tokenizer | 30 min | 15 min |
| Tokenization | 1 hour | 30 min |

## Next Steps

After preprocessing is complete:

1. **Transfer processed data** to cluster for training
2. **Use cluster scripts** for model training
3. **Run evaluation** on cluster or locally

## Support

For issues specific to Windows setup:

1. Check this README first
2. Verify all dependencies are installed
3. Check GPU drivers are up to date
4. Monitor system resources during execution 