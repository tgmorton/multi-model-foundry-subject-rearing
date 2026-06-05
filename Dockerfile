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
#
# TODO(provenance, G4): pin the base image by @sha256 digest at the next
# rebuild (FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:<digest>).
# The floating tag re-resolves to whatever the registry currently serves, so
# the build is not reproducible from this Dockerfile alone until it is pinned.

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
# TODO(provenance, G4): exact-pin the still-bare/ranged names below
# (accelerate, python-dotenv, duckdb, pandas, joblib, spacy-transformers,
# einops, pytest) from a live pod's pip freeze at next rebuild — their
# versions are not in the known image inventory, so don't guess them here.
RUN pip install \
        accelerate>=0.26.0 \
        python-dotenv \
        sentencepiece==0.2.0 \
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
# All models referenced in code, across EN / IT / ES. Pinned to spaCy
# 3.7-compatible releases. We install via direct wheel URLs rather than
# `python -m spacy download`: the download command relies on spaCy's
# online compatibility lookup, which has been intermittently returning
# empty version strings and producing 404s. Direct URLs are deterministic
# and have no startup network dependency.
#
# Note: there is no es_core_news_trf in explosion/spacy-models. Code
# references to it appear to be dead config — drop and re-add a custom
# model if it ever becomes real.
ARG SPACY_MODELS_BASE=https://github.com/explosion/spacy-models/releases/download
RUN pip install \
        ${SPACY_MODELS_BASE}/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl \
        ${SPACY_MODELS_BASE}/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl \
        ${SPACY_MODELS_BASE}/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl \
        ${SPACY_MODELS_BASE}/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl \
        ${SPACY_MODELS_BASE}/it_core_news_sm-3.7.0/it_core_news_sm-3.7.0-py3-none-any.whl \
        ${SPACY_MODELS_BASE}/it_core_news_lg-3.7.0/it_core_news_lg-3.7.0-py3-none-any.whl \
        ${SPACY_MODELS_BASE}/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl \
        ${SPACY_MODELS_BASE}/es_core_news_lg-3.7.0/es_core_news_lg-3.7.0-py3-none-any.whl

# Smoke-test spaCy models load: catches missing-data-files errors at
# build time rather than at first analysis run.
RUN python -c "import spacy; [spacy.load(m) for m in ['en_core_web_sm','en_core_web_md','en_core_web_lg','en_core_web_trf','it_core_news_sm','it_core_news_lg','es_core_news_sm','es_core_news_lg']]; print('all spaCy models load ok')"

# --- Entrypoint ------------------------------------------------------------
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN mkdir -p /opt/repo /mnt/data /scratch
WORKDIR /opt/repo
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
