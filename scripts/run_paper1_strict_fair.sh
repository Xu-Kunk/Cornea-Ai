#!/usr/bin/env bash
# ============================================================================
# P1-O2 strict-fair CAFT training launcher (Paper 1, Objective O2)
#
# Protocol:
#   Paper 1 requires a donor-disjoint, strictly-fair CAFT fine-tune of
#   Geneformer V2-104M. O2 (Objective O2) trains the final 6-checkpoint grid:
#   λ ∈ {0.0, 0.5} × seed ∈ {42, 123, 456}, 3 epochs each, on the
#   donor-disjoint train split (corneal_100k_donor_split.h5ad), gated by the
#   fair-split manifest (artifacts/paper1_fair_split/manifest.json) via
#   scripts/caft_gf.py --strict-fair (fail-closed: refuses anything that is
#   not donor-disjoint).
#
#   Training is sequential (single GPU; ~11-13 h/run per run_lambda_grid notes).
#   Each run streams stdout+stderr to
#   artifacts/paper1_strict_fair/caft_lam<λ>_seed<seed>.log.
#   Success is confirmed by the CAFT_DONE marker in the run's save dir
#   (save/caft_gf_lam<λ>_seed<seed>_<ts>/). A failed run is logged and the
#   grid CONTINUES with the next run. A summary table (λ, seed, status,
#   save_dir) and a COMPLETE / PARTIAL banner are printed at the end.
#
# Pre-flight (abort on any failure):
#   swap off (swapon --show empty)
#   vLLM stopped (systemctl --user is-active vllm.service != active AND
#                 docker ps --filter name=vllm empty)
#   disk free > 50 GB
#   GPU temp < 60 °C (poll every 30 s, max 10 min)
#   corneal_fm.evaluation importable (Tier-B evaluator present)
#
# Usage (<compute-node>, clean shell, background):
#   cd <local>/ && chmod +x scripts/run_paper1_strict_fair.sh
#   nohup bash scripts/run_paper1_strict_fair.sh > artifacts/paper1_strict_fair/launcher.out 2>&1 &
#   # monitor: tail -f artifacts/paper1_strict_fair/launcher.log
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# ── Configuration (overridable via env) ──────────────────────────────────
PY="${PY:-<venv>/bin/python3}"
LAMBDAS="${LAMBDAS:-0.0 0.5}"
SEEDS="${SEEDS:-42 123 456}"
EPOCHS="${EPOCHS:-3}"
DATA_FILE="${DATA_FILE:-corneal_100k_donor_split.h5ad}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/paper1_fair_split/manifest.json}"
RUN_LOG_DIR="$ROOT/artifacts/paper1_strict_fair"
RUN_TIMEOUT_S=172800      # 48 h per run (matches run_lambda_grid.py CAFT_TIMEOUT)
COOL_TARGET_C=60          # GPU temp target (< 60 °C)
TEMP_POLL_S=30
TEMP_WAIT_MAX_S=600       # 10 min max wait
MIN_DISK_GB=50

mkdir -p "$RUN_LOG_DIR"
LOG="$RUN_LOG_DIR/launcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }

# ── Environment (must be set before any torch import) ─────────────────────
# Same set as run_lambda_grid.py: OMP threads pinned, no expandable segments
# (OOM control on the GB10's shared CPU/GPU memory), SDPA on, HF mirror for
# China network, ptxas path + arch list for the Blackwell <compute-node> (sm_121).
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCH_CUDNN_SDPA_ENABLED=1
export HF_ENDPOINT=https://hf-mirror.com
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST=12.1+PTX

# ── Pre-flight ────────────────────────────────────────────────────────────

preflight() {
    log "── Pre-flight checks ──"
    command -v "$PY" >/dev/null 2>&1 || die "python not found: $PY"

    # 1) swap must be OFF (a swap-in during training can freeze the GB10 host)
    if [ -n "$(swapon --show 2>/dev/null)" ]; then
        die "swap is ON; run 'swapoff -a' before training"
    fi
    log "OK: swap off"

    # 2) vLLM must be stopped (occupies ~74 GB unified memory → OOM risk)
    local vllm_state
    vllm_state=$(systemctl --user is-active vllm.service 2>/dev/null || echo "inactive")
    if [ "$vllm_state" = "active" ]; then
        die "vllm.service is ACTIVE; stop it: systemctl --user stop vllm"
    fi
    log "OK: vllm.service not active ($vllm_state)"
    if command -v docker >/dev/null 2>&1; then
        if [ -n "$(docker ps --filter name=vllm --format '{{.ID}}' 2>/dev/null)" ]; then
            die "docker vllm container is running; stop it before training"
        fi
        log "OK: no docker vllm container"
    else
        log "note: docker not found; skipping container check"
    fi

    # 3) disk free > 50 GB
    local avail_kb avail_gb
    avail_kb=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
    avail_gb=$(( avail_kb / 1024 / 1024 ))
    if [ "$avail_gb" -le "$MIN_DISK_GB" ]; then
        die "only ${avail_gb} GB free; need > ${MIN_DISK_GB} GB"
    fi
    log "OK: disk free = ${avail_gb} GB"

    # 4) GPU temp < 60 °C (wait loop: 30 s poll, max 10 min)
    local waited=0 max_temp=0 temps t
    while :; do
        temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
        [ -n "$temps" ] || die "nvidia-smi returned no GPU temperatures"
        max_temp=0
        for t in $temps; do
            [ "$t" -gt "$max_temp" ] && max_temp=$t
        done
        if [ "$max_temp" -lt "$COOL_TARGET_C" ]; then
            break
        fi
        if [ "$waited" -ge "$TEMP_WAIT_MAX_S" ]; then
            die "GPU temp still ${max_temp}C (>= ${COOL_TARGET_C}C) after ${TEMP_WAIT_MAX_S}s; aborting"
        fi
        log "GPU ${max_temp}C >= ${COOL_TARGET_C}C; sleeping ${TEMP_POLL_S}s (${waited}s elapsed)..."
        sleep "$TEMP_POLL_S"
        waited=$(( waited + TEMP_POLL_S ))
    done
    log "OK: GPU temp ${max_temp}C < ${COOL_TARGET_C}C"

    # 5) Tier-B evaluator must be importable (corneal_fm/evaluation.py)
    if ! "$PY" -c "import corneal_fm.evaluation as ev; assert callable(ev.run_static_checks) and callable(ev.run_leakage_checks)" >/dev/null 2>&1; then
        die "corneal_fm.evaluation import failed (Tier-B evaluator missing/broken)"
    fi
    log "OK: corneal_fm.evaluation importable (Tier-B evaluator present)"

    # 6) sanity: donor-disjoint train data + fair-split manifest present
    [ -f "$ROOT/data/$DATA_FILE" ] || die "missing train data: data/$DATA_FILE"
    [ -f "$ROOT/$SPLIT_MANIFEST" ] || die "missing split manifest: $SPLIT_MANIFEST"
    log "OK: train data + split manifest present"
    log "── All pre-flight checks passed ──"
}

# ── Single training run ───────────────────────────────────────────────────
# Runs caft_gf.py synchronously (48 h timeout), then locates the run's save
# dir (newest matching save/caft_gf_lam<λ>_seed<seed>_*) and requires the
# CAFT_DONE marker. On success sets global RUN_SAVE_DIR; returns 0.
# On failure logs the error and returns 1 (caller continues the grid).

RUN_SAVE_DIR=""

run_caft() {
    local lam="$1" seed="$2"
    local logf="$RUN_LOG_DIR/caft_lam${lam}_seed${seed}.log"
    log "──── λ=$lam seed=$seed  training start ────"
    local before
    before=$(ls -dt "$ROOT"/save/caft_gf_lam${lam}_seed${seed}_* 2>/dev/null | head -1 || true)

    timeout "$RUN_TIMEOUT_S" "$PY" scripts/caft_gf.py \
        --caft_lambda "$lam" --epochs "$EPOCHS" --seed "$seed" \
        --data_file "$DATA_FILE" --strict-fair \
        --split_manifest "$SPLIT_MANIFEST" > "$logf" 2>&1
    local rc=$?

    local save_dir
    save_dir=$(ls -dt "$ROOT"/save/caft_gf_lam${lam}_seed${seed}_* 2>/dev/null | head -1 || true)
    if [ -n "$before" ] && [ "$save_dir" = "$before" ]; then
        save_dir=""   # no NEW save dir appeared → run never wrote one
    fi

    if [ "$rc" -eq 124 ]; then
        log "FAIL λ=$lam seed=$seed  (timed out after $((RUN_TIMEOUT_S/3600))h); log: $logf"
        return 1
    fi
    if [ "$rc" -ne 0 ]; then
        log "FAIL λ=$lam seed=$seed  (rc=$rc${save_dir:+; latest dir: }$save_dir); log: $logf"
        return 1
    fi
    if [ -z "$save_dir" ] || [ ! -f "$save_dir/CAFT_DONE" ]; then
        log "FAIL λ=$lam seed=$seed  (CAFT_DONE marker missing${save_dir:+ in }$save_dir); log: $logf"
        return 1
    fi
    RUN_SAVE_DIR="$save_dir"
    log "OK   λ=$lam seed=$seed  done (CAFT_DONE) → $save_dir"
    return 0
}

# ── Main ──────────────────────────────────────────────────────────────────

log "===================================================="
log "P1-O2 strict-fair CAFT launcher"
log "grid: λ∈{$LAMBDAS} × seed∈{$SEEDS} | epochs=$EPOCHS | data=$DATA_FILE"
log "manifest: $SPLIT_MANIFEST | logs: $RUN_LOG_DIR"
log "===================================================="

preflight

LAM_HIST=(); SEED_HIST=(); STATUS=(); SAVE_DIRS=()
for lam in $LAMBDAS; do
    for seed in $SEEDS; do
        LAM_HIST+=("$lam"); SEED_HIST+=("$seed")
        RUN_SAVE_DIR=""
        if run_caft "$lam" "$seed"; then
            STATUS+=("OK"); SAVE_DIRS+=("$RUN_SAVE_DIR")
        else
            STATUS+=("FAIL")
            SAVE_DIRS+=("$(ls -dt "$ROOT"/save/caft_gf_lam${lam}_seed${seed}_* 2>/dev/null | head -1 || true)")
        fi
    done
done

# ── Summary ───────────────────────────────────────────────────────────────
NOK=0
TOTAL=${#STATUS[@]}
log ""
log "===================================================="
log "P1-O2 strict-fair training summary"
log "===================================================="
printf '%-8s %-6s %-8s %s\n' "lambda" "seed" "status" "save_dir" | tee -a "$LOG"
for i in "${!STATUS[@]}"; do
    printf '%-8s %-6s %-8s %s\n' "${LAM_HIST[$i]}" "${SEED_HIST[$i]}" \
        "${STATUS[$i]}" "${SAVE_DIRS[$i]:--}" | tee -a "$LOG"
    [ "${STATUS[$i]}" = "OK" ] && NOK=$(( NOK + 1 ))
done
log "===================================================="
if [ "$NOK" -eq "$TOTAL" ]; then
    log "P1-O2 TRAINING COMPLETE"
    log "===================================================="
    exit 0
else
    log "P1-O2 TRAINING PARTIAL ($NOK/$TOTAL succeeded)"
    log "===================================================="
    exit 1
fi
