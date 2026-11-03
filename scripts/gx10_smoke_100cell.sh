#!/usr/bin/env bash
# gx10_smoke_100cell.sh — Deterministic bounded 100-cell GX10 smoke validation.
#
# This script:
#   1. Creates a timestamped smoke output directory
#   2. Generates a 100-cell deterministic subset
#   3. Runs preflight checks
#   4. Applies GX10 runtime guardrails
#   5. Launches a single CAFT-GF training run (λ=0, 1 epoch, batch≤4, seed=42)
#   6. On CAFT_DONE, runs paired GPR comparison
#   7. Validates GPR contract and reports verdict
#
# Usage:
#   bash scripts/gx10_smoke_100cell.sh
#
# Must be run as root or with sufficient privileges for cgroup/swapoff.
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
readonly SMOKE_DIR="${PROJECT_ROOT}/smoke_${TIMESTAMP}"

readonly SRC_H5AD="${PROJECT_ROOT}/artifacts/day3-20260723/S6/corneal_30k_layer_subsample.h5ad"
readonly PINNED_PYTHON="/home/dq/venv-ds/bin/python"
readonly SUBSET_H5AD="${SMOKE_DIR}/smoke_100cells.h5ad"
readonly CGROUP_LAUNCHER="${SCRIPT_DIR}/gx10_cgroup_run.sh"
readonly GPR_RECOVERY="${SCRIPT_DIR}/gx10_gpr_recovery.sh"

readonly TRAIN_EPOCHS=1
readonly TRAIN_BATCH=4
readonly TRAIN_LAMBDA=0.0
readonly TRAIN_SEED=42
readonly TRAIN_MEMORY_MAX="20G"
readonly TRAIN_MEMORY_SWAP_MAX="0"
readonly TRAIN_RUNTIME_MAX="2700"   # 45 min
readonly TRAIN_OOM_SCORE_ADJUST="500"

declare -g SMOKE_LOG="${SMOKE_DIR}/smoke.log"

# ──────────────────────────────────────────────────────────────────────────────

_write_status() {
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$SMOKE_LOG"
}

_bail() {
    _write_status "BLOCKED: $*"
    cat > "${SMOKE_DIR}/SMOKE_BLOCKED" <<< "$(date +%Y-%m-%dT%H:%M:%S) — $*"
    exit 1
}

_pass() {
    _write_status "PASS: $*"
    cat > "${SMOKE_DIR}/SMOKE_PASS" <<< "$(date +%Y-%m-%dT%H:%M:%S) — $*"
    exit 0
}

_monitor_loop() {
    # Background monitor: log MemAvailable + GPU temp every 15s.
    # Kills the entire process group if temp >= 80C or MemAvailable < 20GiB.
    local pid_to_watch="$1"
    local interval=15
    local temp_limit=80
    local mem_limit_kb=20971520  # 20 GiB in KiB

    while kill -0 "$pid_to_watch" 2>/dev/null; do
        local mem_kb gib temp
        mem_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
        gib="$(( mem_kb / 1024 / 1024 ))"
        temp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 || echo 0)"
        temp="${temp// /}"

        _write_status "MONITOR: MemAvailable=${gib}GiB  GPU_temp=${temp}C"

        if [[ "$temp" -ge "$temp_limit" ]]; then
            _write_status "CRITICAL: GPU temp ${temp}C >= ${temp_limit}C — KILLING"
            kill -TERM -"$pid_to_watch" 2>/dev/null || true
            sleep 2
            kill -KILL -"$pid_to_watch" 2>/dev/null || true
            _bail "GPU temperature ${temp}C >= ${temp_limit}C limit"
        fi

        if [[ "$mem_kb" -lt "$mem_limit_kb" ]]; then
            _write_status "CRITICAL: MemAvailable ${gib}GiB < 20GiB — KILLING"
            kill -TERM -"$pid_to_watch" 2>/dev/null || true
            sleep 2
            kill -KILL -"$pid_to_watch" 2>/dev/null || true
            _bail "MemAvailable ${gib}GiB < 20GiB limit"
        fi

        sleep "$interval"
    done
}

# ──────────────────────────────────────────────────────────────────────────────
# Step 0: Create smoke directory
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p "$SMOKE_DIR"
_write_status "=== GX10 100-CELL SMOKE VALIDATION ==="
_write_status "Timestamp:  $TIMESTAMP"
_write_status "Smoke dir:  $SMOKE_DIR"
_write_status "Source:     $SRC_H5AD"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Generate deterministic 100-cell subset
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 1: Generate 100-cell subset ---"
"$PINNED_PYTHON" -m scripts.deterministic_subset \
    --src "$SRC_H5AD" \
    --out "$SUBSET_H5AD" \
    --seed 42 2>&1 | tee -a "$SMOKE_LOG"

if [[ ! -f "$SUBSET_H5AD" ]]; then
    _bail "Subset generation failed — no output file"
fi
# Verify exactly 100 cells
cell_count=$("$PINNED_PYTHON" -c "
import anndata as ad
a = ad.read_h5ad('$SUBSET_H5AD', backed='r')
print(a.n_obs)
")
if [[ "$cell_count" != "100" ]]; then
    _bail "Subset has $cell_count cells, expected 100"
fi
_write_status "Subset generated: $cell_count cells ✓"

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Preflight checks
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 2: Preflight checks ---"

# Check CUDA
if ! nvidia-smi &>/dev/null; then
    _bail "nvidia-smi not available — CUDA check failed"
fi
_write_status "CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Check cgroup v2
cg_fs="$(stat -fc%T /sys/fs/cgroup 2>/dev/null || echo unknown)"
if [[ "$cg_fs" != "cgroup2fs" ]]; then
    _bail "cgroup v2 not detected (got: $cg_fs)"
fi
_write_status "cgroup v2: ✓"

# Check MemAvailable
mem_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
mem_gib="$(( mem_kb / 1024 / 1024 ))"
if [[ "$mem_gib" -lt 20 ]]; then
    _bail "MemAvailable ${mem_gib}GiB < 20GiB"
fi
_write_status "MemAvailable: ${mem_gib}GiB ✓"

# Check no active GPU compute
gpu_procs="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^$' | grep -v '^pid,' | tr -d ' ' || true)"
if [[ -n "$gpu_procs" ]]; then
    _bail "Active GPU compute processes: $gpu_procs"
fi
_write_status "No active GPU compute: ✓"

# Check pinned Python
if [[ ! -x "$PINNED_PYTHON" ]]; then
    _bail "Pinned Python not found: $PINNED_PYTHON"
fi
_write_status "Pinned Python: $PINNED_PYTHON ✓"

# Check no vLLM
if pgrep -f "vllm\|vllm.entrypoints\|llama" >/dev/null 2>&1; then
    _bail "Active vLLM/LLM process detected"
fi
_write_status "No vLLM/compute: ✓"

# Check source h5ad
if [[ ! -f "$SRC_H5AD" ]]; then
    _bail "Source h5ad not found: $SRC_H5AD"
fi
_write_status "Source h5ad: ✓"

# Dry-run the cgroup launcher to verify argument parsing
_write_status "--- Cgroup launcher dry-run ---"
bash "$CGROUP_LAUNCHER" --mode service --memory-max "$TRAIN_MEMORY_MAX" \
    --memory-swap-max "$TRAIN_MEMORY_SWAP_MAX" --runtime-max "$TRAIN_RUNTIME_MAX" \
    --oom-score-adjust "$TRAIN_OOM_SCORE_ADJUST" \
    --unit "smoke-check" --log-dir "$SMOKE_DIR/logs" \
    --dry-run -- echo OK 2>&1 | tee -a "$SMOKE_LOG"
_write_status "Cgroup launcher dry-run: ✓"

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Apply GX10 runtime guardrails
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 3: Apply GX10 runtime guardrails ---"

# swapoff
if swapon --show 2>/dev/null | grep -q .; then
    if swapoff -a 2>/dev/null; then
        _write_status "swapoff: applied ✓"
    else
        _bail "swapoff failed — cannot proceed with swap active"
    fi
else
    _write_status "swapoff: already off ✓"
fi

# swappiness=1
if [[ -w /proc/sys/vm/swappiness ]]; then
    echo 1 > /proc/sys/vm/swappiness 2>/dev/null || true
    actual="$(cat /proc/sys/vm/swappiness)"
    if [[ "$actual" != "1" ]]; then
        _write_status "WARN: swappiness set to $actual (wanted 1) — non-fatal, continuing"
    else
        _write_status "swappiness=1: applied ✓"
    fi
else
    _write_status "WARN: cannot set swappiness — non-fatal, continuing"
fi

# dirty_bytes=268435456
if [[ -w /proc/sys/vm/dirty_bytes ]]; then
    echo 268435456 > /proc/sys/vm/dirty_bytes 2>/dev/null || true
    actual="$(cat /proc/sys/vm/dirty_bytes)"
    if [[ "$actual" != "268435456" ]]; then
        _write_status "WARN: dirty_bytes set to $actual (wanted 268435456) — non-fatal, continuing"
    else
        _write_status "dirty_bytes=268435456: applied ✓"
    fi
else
    _write_status "WARN: cannot set dirty_bytes — non-fatal, continuing"
fi

# GPU persistence mode and clock 300-2200 (non-fatal if permission denied)
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -pm 1 2>/dev/null && _write_status "GPU persistence: enabled ✓" || \
        _write_status "WARN: GPU persistence mode not available (permission?) — continuing"
    nvidia-smi -lgc 300,2200 2>/dev/null && _write_status "GPU clock 300-2200: applied ✓" || \
        _write_status "WARN: GPU clock lock not available (permission?) — continuing"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Launch CAFT-GF training (λ=0, 1 epoch, batch=4)
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 4: Launch CAFT-GF training ---"

readonly TRAIN_UNIT="gx10-smoke-${TIMESTAMP}"
readonly TRAIN_LOG_DIR="${SMOKE_DIR}/logs"

mkdir -p "$TRAIN_LOG_DIR"

_write_status "Training: epochs=$TRAIN_EPOCHS batch=$TRAIN_BATCH lambda=$TRAIN_LAMBDA seed=$TRAIN_SEED"
_write_status "Cgroup: MemoryMax=$TRAIN_MEMORY_MAX MemorySwapMax=$TRAIN_MEMORY_SWAP_MAX RuntimeMax=${TRAIN_RUNTIME_MAX}s"

# Launch training via cgroup service launcher, background the monitoring
bash "$CGROUP_LAUNCHER" \
    --mode service \
    --memory-max "$TRAIN_MEMORY_MAX" \
    --memory-swap-max "$TRAIN_MEMORY_SWAP_MAX" \
    --runtime-max "$TRAIN_RUNTIME_MAX" \
    --oom-score-adjust "$TRAIN_OOM_SCORE_ADJUST" \
    --unit "$TRAIN_UNIT" \
    --log-dir "$TRAIN_LOG_DIR" \
    -- \
    env HOME="/home/dq" HF_HOME="/home/dq/.cache/huggingface" "$PINNED_PYTHON" "$SCRIPT_DIR/caft_gf.py" \
        --data_file "$SUBSET_H5AD" \
        --caft_lambda "$TRAIN_LAMBDA" \
        --epochs "$TRAIN_EPOCHS" \
        --batch_size "$TRAIN_BATCH" \
        --seed "$TRAIN_SEED" \
        \
    &
TRAIN_PID=$!
_monitor_loop "$TRAIN_PID" &
MONITOR_PID=$!

wait "$TRAIN_PID" || true
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

# Find the save directory (caft_gf.py uses get_save_dir)
readonly SAVE_GLOB="${PROJECT_ROOT}/save/caft_gf_lam${TRAIN_LAMBDA}_seed${TRAIN_SEED}_*"
save_dir=$(ls -dt ${SAVE_GLOB} 2>/dev/null | head -1 || echo "")

if [[ -z "$save_dir" ]]; then
    _bail "CAFT training did not produce a save directory (glob: ${SAVE_GLOB})"
fi
_write_status "Save directory: $save_dir"

# Check for CAFT_DONE
if [[ ! -f "$save_dir/CAFT_DONE" ]]; then
    # Check for ERROR_TRACEBACK
    if [[ -f "$save_dir/ERROR_TRACEBACK.txt" ]]; then
        _write_status "ERROR_TRACEBACK:"
        cat "$save_dir/ERROR_TRACEBACK.txt" | tee -a "$SMOKE_LOG"
        _bail "CAFT training failed — ERROR_TRACEBACK.txt present"
    fi
    _bail "CAFT training failed — CAFT_DONE not found in $save_dir"
fi
_write_status "CAFT_DONE found: $(cat "$save_dir/CAFT_DONE") ✓"

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Locate before/after checkpoints
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 5: Locate checkpoints ---"

# "Before" checkpoint: Geneformer-V2-104M base model (no fine-tuning).
# BertForSequenceClassification.from_pretrained with num_labels=N will load
# the shared BERT weights and initialise a fresh classifier head — exactly
# what load_pretrained_classifier does.  The preflight checks for
# model.safetensors + config.json, both present in the base snapshot.
BEFORE_CKPT=$(ls -d ~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M/ 2>/dev/null | head -1 || echo "")
if [[ -z "$BEFORE_CKPT" ]]; then
    _bail "Geneformer base model not found in HF cache"
fi
_write_status "Before checkpoint (base): $BEFORE_CKPT"
if [[ ! -f "$BEFORE_CKPT/model.safetensors" ]] || [[ ! -f "$BEFORE_CKPT/config.json" ]]; then
    _bail "Before checkpoint lacks model.safetensors or config.json"
fi
_write_status "Before checkpoint files: ✓"

# "After" checkpoint: latest checkpoint-* from training
readonly AFTER_CKPT=$(ls -dt "$save_dir"/model/checkpoint-* 2>/dev/null | head -1 || echo "")
if [[ -z "$AFTER_CKPT" ]]; then
    _bail "No after checkpoint found in $save_dir/model/"
fi
_write_status "After checkpoint: $AFTER_CKPT"

# Tokenized dataset
readonly TOKENIZED_DIR="${save_dir}/corneal_layer_subset_tk/cell.dataset"
if [[ ! -d "$TOKENIZED_DIR" ]]; then
    _bail "Tokenized dataset not found: $TOKENIZED_DIR"
fi
_write_status "Tokenized dataset: $TOKENIZED_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: GPR dry-run via gx10_gpr_recovery.sh
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 6: GPR recovery dry-run ---"
readonly GPR_OUT="${SMOKE_DIR}/gpr_smoke.json"

bash "$GPR_RECOVERY" \
    --before-checkpoint "$BEFORE_CKPT" \
    --after-checkpoint "$AFTER_CKPT" \
    --tokenized "$TOKENIZED_DIR" \
    --out "$GPR_OUT" \
    --dry-run \
    --quiet 2>&1 | tee -a "$SMOKE_LOG"
_write_status "GPR dry-run: ✓"

# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Run paired GPR comparison via gx10_gpr_recovery.sh
# n_boot=1000 (recovery default; ~1-2s for 100 cells, appropriate for smoke)
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 7: Run paired GPR comparison (n_boot=1000 via recovery) ---"

mkdir -p "${SMOKE_DIR}/gpr_logs"

# gx10_gpr_recovery.sh exec's into gx10_cgroup_run.sh --wait, so this blocks.
_write_status "Launching GPR via $GPR_RECOVERY with cgroup service..."
bash "$GPR_RECOVERY" \
    --before-checkpoint "$BEFORE_CKPT" \
    --after-checkpoint "$AFTER_CKPT" \
    --tokenized "$TOKENIZED_DIR" \
    --out "$GPR_OUT" \
    --quiet 2>&1 | tee -a "$SMOKE_LOG"

if [[ ! -f "$GPR_OUT" ]]; then
    _bail "GPR comparison did not produce output"
fi
_write_status "GPR output: $GPR_OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Step 8: Validate GPR contract
# ──────────────────────────────────────────────────────────────────────────────
_write_status "--- Step 8: Validate GPR contract ---"

"$PINNED_PYTHON" -c "
import json, sys
from pathlib import Path

with open('${GPR_OUT}') as f:
    data = json.load(f)

ok = True
errors = []

# Top-level keys
for key in ('contract', 'primary', 'sensitivity', 'diagnostics'):
    if key not in data:
        errors.append(f'missing top-level key: {key}')
        ok = False

# contract
ct = data.get('contract', {})
for section in ('embedding', 'metric', 'dataset', 'checkpoints', 'resampling'):
    if section not in ct:
        errors.append(f'contract missing: {section}')
        ok = False

# primary
pr = data.get('primary', {})
for field in ('metric_name', 'cs_before', 'cs_after', 'gpr', 'absolute_delta', 'paired_stratified_bootstrap_95ci'):
    if field not in pr:
        errors.append(f'primary missing: {field}')
        ok = False

# sensitivity
sens = data.get('sensitivity', {})
if 'mean_pool' not in sens:
    errors.append('sensitivity missing: mean_pool')
    ok = False

# diagnostics
diag = data.get('diagnostics', {})
if 'anisotropy' not in diag:
    errors.append('diagnostics missing: anisotropy')
    ok = False

# type checks
if not isinstance(ct.get('resampling', {}).get('method', ''), str):
    errors.append('resampling.method not a string')
    ok = False

if ok:
    print('Contract validation: PASS')
    print(f'  Primary raw CLS CS_before: {pr.get(\"cs_before\")}')
    print(f'  Primary raw CLS CS_after:  {pr.get(\"cs_after\")}')
    print(f'  GPR:                       {pr.get(\"gpr\")}')
    print(f'  Absolute delta:            {pr.get(\"absolute_delta\")}')
    print(f'  Bootstrap 95%CI:           {pr.get(\"paired_stratified_bootstrap_95ci\")}')
    a = diag.get('anisotropy', {})
    print(f'  Anisotropy raw_cls before: {a.get(\"raw_cls\", {}).get(\"before\", \"N/A\")}')
    print(f'  Anisotropy raw_cls after:  {a.get(\"raw_cls\", {}).get(\"after\", \"N/A\")}')
else:
    print('Contract validation: FAIL')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
" 2>&1 | tee -a "$SMOKE_LOG"

# ──────────────────────────────────────────────────────────────────────────────
# Step 9: Final verdict
# ──────────────────────────────────────────────────────────────────────────────
_write_status "=== SMOKE VALIDATION COMPLETE ==="
_pass "All gates passed. 100-cell smoke validated. GPR comparison produced. Authorize 30k."
