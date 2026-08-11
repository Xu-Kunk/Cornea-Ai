# Corneal-FM Docker image for GX10 (NVIDIA DGX Spark / GB10 Grace-Blackwell, ARM64, sm_121)
# Build: docker build -t corneal-fm .
# Run:   docker run --gpus all -v ./data:/app/data -v ./save:/app/save corneal-fm
#
# Base image: NGC pytorch:25.11-py3 (CUDA 13 + torch built for sm_121) — matches
# docs/GX10-deployment-checklist.md and config/env.yml. Do NOT use <=24.12: it ships
# CUDA 12.6 / torch 2.6 with no Blackwell-consumer (sm_12x) codegen (NVIDIA optimizes
# containers for Blackwell starting 25.01; DGX Spark playbooks pin 25.11-py3).
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps.
# requirements-core.txt pins a CPU torch that would REPLACE the container's
# CUDA-enabled sm_121 torch wheel, so we strip torch and keep the container build.
COPY requirements-core.txt .
RUN grep -viE '^torch==' requirements-core.txt > /tmp/req-notorch.txt \
    && pip install --no-cache-dir -r /tmp/req-notorch.txt \
    && rm -f /tmp/req-notorch.txt

# Source
COPY corneal_fm/ ./corneal_fm/
COPY config/ ./config/
COPY scripts/ ./scripts/

# GB10 sm_121 build/compile env (used when building extensions / triton kernels)
ENV TORCH_CUDA_ARCH_LIST="12.1+PTX" \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "scripts/run_all.py"]
