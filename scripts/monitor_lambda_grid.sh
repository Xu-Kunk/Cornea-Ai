#!/usr/bin/env bash
# CAFT λ-grid overnight monitor — runs every 30 min, independent of any terminal session.
# Usage: setsid bash scripts/monitor_lambda_grid.sh &
# Log:   artifacts/lambda_grid/monitor.log

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/artifacts/lambda_grid/monitor.log"
SAVE="$ROOT/save"
TSV="$ROOT/artifacts/lambda_grid/results.tsv"

mkdir -p "$ROOT/artifacts/lambda_grid"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=========================================="
log "Monitor started (PID $$)"
log "=========================================="

while true; do
    log "--- check ---"

    # GPU status
    GPU=$(nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,clocks.sm --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed")
    log "GPU: $GPU"

    # Memory
    MEM=$(free -h | awk '/^内存/{print "used="$3" free="$4" avail="$7}')
    log "MEM: $MEM"

    # Active CAFT processes
    PROCS=$(ps aux | grep -E "caft_gf|run_lambda" | grep -v grep | awk '{print "PID="$2, "CPU="$3"%", "TIME="$10}')
    if [ -z "$PROCS" ]; then
        log "PROCS: NONE — λ grid may have finished or crashed!"
    else
        log "PROCS: $PROCS"
    fi

    # Latest RUN_STATUS for each active save dir
    for d in "$SAVE"/caft_gf_lam*_seed42_2026072*; do
        [ -f "$d/RUN_STATUS.txt" ] || continue
        LAST=$(tail -1 "$d/RUN_STATUS.txt")
        CKPT=$(ls "$d/model/" 2>/dev/null | grep "checkpoint" | sort -t- -k2 -n | tail -1)
        DONE=$([ -f "$d/CAFT_DONE" ] && echo "YES" || echo "no")
        log "  $(basename $d): last='$LAST' | ckpt=${CKPT:-none} | CAFT_DONE=$DONE"
    done

    # results.tsv
    if [ -f "$TSV" ]; then
        NROWS=$(($(wc -l < "$TSV") - 1))
        log "results.tsv: $NROWS λ completed"
        tail -n +2 "$TSV" | while IFS=$'\t' read -r lam cs gpr rest; do
            log "  λ=$lam  gpr=$gpr  cs_after=$cs"
        done
    else
        log "results.tsv: not yet created"
    fi

    # Alert conditions
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$TEMP" ] && [ "$TEMP" -ge 85 ] 2>/dev/null; then
        log "⚠️  ALERT: GPU temp ${TEMP}°C >= 85°C — thermal risk!"
    fi
    if [ -z "$PROCS" ]; then
        log "⚠️  ALERT: No CAFT process running — check grid.log for errors"
        tail -5 "$ROOT/artifacts/lambda_grid/grid.log" >> "$LOG" 2>/dev/null
    fi

    sleep 1800   # 30 minutes
done
