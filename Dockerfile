# Comprehensive training image: torch + CUDA extensions + all project deps.
# Source code is NOT in the image — it is cloned at pod startup by
# scripts/docker-entrypoint.sh. This image only needs to be rebuilt when
# Dockerfile / requirements.txt / pyproject.toml / docker-entrypoint.sh
# changes; see .gitlab-ci.yml.
#
# Pinned for reproducibility:
#   CUDA 12.1.1 (devel, for nvcc)  +  cuDNN 8  +  Ubuntu 22.04
#   Python 3.11
#   torch 2.5.1+cu121  (matches pytorch/pytorch:2.5.1-cuda12.1 used historically)
#   flash-attn 2.7.4.post1   (cu12 / torch2.5 / cp311 / cxx11abi=FALSE)
#   causal-conv1d 1.6.0      (same ABI matrix)
#   mamba-ssm 2.3.0          (same ABI matrix)
#
# PyTorch's prebuilt cu121 wheels are built with the pre-cxx11 ABI, so all
# CUDA-extension wheels here MUST be cxx11abiFALSE. Mismatched ABIs produce
# undefined-symbol errors at import time.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# --- System packages -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential ninja-build \
        git wget curl ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
 && pip install --upgrade pip setuptools wheel

# --- PyTorch (pinned to CUDA 12.1) -----------------------------------------
RUN pip install \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

# --- Project Python deps ---------------------------------------------------
# requirements.txt is the source of truth for analysis/training deps.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Extras observed across job YAMLs (sentencepiece, accelerate, duckdb, ...)
# Kept here so a single image covers train + eval + analysis + annotate.
RUN pip install \
        accelerate>=0.26.0 \
        python-dotenv \
        sentencepiece \
        duckdb \
        pandas \
        joblib \
        spacy-transformers \
        einops \
        pytest

# --- CUDA extensions (prebuilt wheels) -------------------------------------
# We use prebuilt wheels rather than `pip install flash-attn` because the
# from-source build of flash-attn is ~30 min and frequently OOMs CI runners.
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG CAUSAL_CONV1D_VERSION=1.6.0
ARG MAMBA_SSM_VERSION=2.3.0

RUN pip install \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

RUN pip install \
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/causal_conv1d-${CAUSAL_CONV1D_VERSION}+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

RUN pip install \
        "https://github.com/state-spaces/mamba/releases/download/v${MAMBA_SSM_VERSION}/mamba_ssm-${MAMBA_SSM_VERSION}+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# Smoke-test the CUDA extensions at build time so a broken image fails CI
# instead of failing 30 min into a training pod.
RUN python -c "import torch; assert torch.version.cuda.startswith('12.1'), torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
 && python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" \
 && python -c "import causal_conv1d; print('causal_conv1d ok')" \
 && python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)"

# --- spaCy models ----------------------------------------------------------
# All models referenced in code, across EN / IT / ES. The trf models are
# large (~500MB each) but downloading them at pod startup costs minutes per
# run; baking them in is a one-time cost.
RUN python -m spacy download en_core_web_sm \
 && python -m spacy download en_core_web_md \
 && python -m spacy download en_core_web_lg \
 && python -m spacy download en_core_web_trf \
 && python -m spacy download it_core_news_sm \
 && python -m spacy download it_core_news_lg \
 && python -m spacy download es_core_news_sm \
 && python -m spacy download es_core_news_lg \
 && python -m spacy download es_core_news_trf

# --- Entrypoint ------------------------------------------------------------
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN mkdir -p /opt/repo /mnt/data /scratch
WORKDIR /opt/repo
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
