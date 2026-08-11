#!/usr/bin/env bash
# Smoke validation of the GX10 speed fixes before the overnight λ-grid.
# Runs ONE λ (0.5) for ONE epoch with the production caft_gf.py config
# (bf16, batch=8×grad_accum=4, gradient_checkpointing) and records wall time.
# PASS criteria (judged from the log afterwards):
#   - reaches CAFT training (RUN_STATUS: TRAINED) without rc=-9 / OOM
#   - single-epoch wall time consistent with ~60-90 min/epoch (not 289 min)
#   - GPU sustains high util, system memory stays bounded (no swap)
# Usage: bash scripts/smoke_caft_1ep.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=python
LOG="$ROOT/artifacts/smoke/smoke_1ep.log"
mkdir -p "$ROOT/artifacts/smoke"

cd "$ROOT" || exit 1

echo "[$(date '+%F %T')] SMOKE start: λ=0.5 × 1 epoch" | tee -a "$LOG"

# Environment must be set before any torch import (mirrors run_lambda_grid.py).
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_ENDPOINT=https://hf-mirror.com
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.1+PTX"
export CORNEAL_FM_DATA="$ROOT/data"

START=$(date +%s)
"$PY" scripts/caft_gf.py \
    --caft_lambda 0.5 \
    --data_file corneal_100k_subsample.h5ad \
    --epochs 1 \
    --seed 42 >> "$LOG" 2>&1
RC=$?
END=$(date +%s)

echo "[$(date '+%F %T')] SMOKE exit rc=$RC elapsed_min=$(( (END - START) / 60 ))" | tee -a "$LOG"
exit $RC
