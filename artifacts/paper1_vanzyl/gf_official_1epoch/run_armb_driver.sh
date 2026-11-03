#!/bin/bash
# run_armb_driver.sh v3 — PREREG §9 Arm B + E3: GF official 1-epoch control.
# v3 (2026-08-19): SEEDS parameterizable (default "42 123 456"; E3 passes
# "789 2024 31415"); wait_gpu switched to free >= 4096 MiB (user E3 guard).
# RESUME-CAPABLE (v2): per-seed completion markers
#   seed{S}_TRAIN_DONE / seed{S}_EVAL_DONE  in $OUT.
#   - both present -> seed skipped
#   - TRAIN_DONE only -> training skipped, eval phases run
#   - sub-phase robustness: tier-b eval skipped if results_0.0_{S}.json exists;
#     extra eval skipped if eval_armb_seed{S}.json exists; tokenized-cache
#     detection for --skip-tokenize is file-based (restart-safe).
# Science code untouched; only orchestration changed.
set -u
cd <home>/corneal_scgpt || exit 9
PY=venv/bin/python
OUT=artifacts/paper1_vanzyl/gf_official_1epoch
TKCACHE=artifacts/paper1_tier_b_eval/external_tokenized/cell.dataset
SEEDS=${1:-"42 123 456"}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/armb_driver.log"; }

wait_gpu() {  # GPU free mem >= 4096 MiB, up to 600 x 30s (~5h) before abort
  local free
  for i in $(seq 1 600); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${free:-0}" -ge 4096 ] && return 0
    sleep 30
  done
  return 1
}

train_completed() {  # marker file OR any seed save dir with CAFT_DONE
  local s=$1
  [ -f "$OUT/seed${s}_TRAIN_DONE" ] && return 0
  ls -d save/caft_gf_lam0.0_seed${s}_*/CAFT_DONE 2>/dev/null | grep -q . && return 0
  return 1
}

log "driver v3 start: seeds=[$SEEDS] (E3: GF 1-epoch extended seeds)"
for seed in $SEEDS; do
  TDONE=$OUT/seed${seed}_TRAIN_DONE
  EDONE=$OUT/seed${seed}_EVAL_DONE
  if [ -f "$TDONE" ] && [ -f "$EDONE" ]; then
    log "seed=$seed already complete (markers); skip"
    continue
  fi
  # ---------------- TRAIN ----------------
  if ! train_completed "$seed"; then
    log "seed=$seed phase=train begin"
    rc=1
    for attempt in 1 2 3 4 5; do
      wait_gpu || { log "seed=$seed GPU busy too long; abort"; exit 7; }
      $PY scripts/caft_gf_arm_b.py \
          --data_file corneal_100k_donor_split.h5ad \
          --epochs 1 --lr 5e-5 --caft_lambda 0.0 --seed "$seed" \
          --strict_fair --split_manifest artifacts/paper1_fair_split/manifest.json \
          > "$OUT/train_seed${seed}.log" 2>&1
      rc=$?
      log "seed=$seed attempt=$attempt train rc=$rc"
      [ $rc -eq 0 ] && break
      sleep 30
    done
    if [ $rc -ne 0 ]; then log "seed=$seed train failed all attempts"; exit 8; fi
    touch "$TDONE"
    log "seed=$seed TRAIN_DONE marker"
  else
    log "seed=$seed training already done (marker or CAFT_DONE); skip train"
  fi
  # ---------------- EVAL tier_b ----------------
  if [ -f "artifacts/paper1_tier_b_eval/results_0.0_${seed}.json" ]; then
    log "seed=$seed eval_tierb already done (results_0.0_${seed}.json); skip"
  else
    log "seed=$seed phase=eval_tierb begin"
    wait_gpu || { log "seed=$seed GPU busy too long; abort"; exit 7; }
    if [ -f "$TKCACHE" ]; then
      $PY scripts/run_tier_b_eval.py --runs 0.0:$seed --skip-tokenize > "$OUT/eval_seed${seed}.log" 2>&1
    else
      $PY scripts/run_tier_b_eval.py --runs 0.0:$seed > "$OUT/eval_seed${seed}.log" 2>&1
    fi
    rc=$?
    log "seed=$seed eval_tierb rc=$rc"
    if [ $rc -ne 0 ]; then log "seed=$seed eval_tierb failed; abort (resume-safe)"; exit 8; fi
  fi
  # ---------------- EVAL extra ----------------
  if [ -f "$OUT/eval_armb_seed${seed}.json" ]; then
    log "seed=$seed eval_extra already done; skip"
  else
    log "seed=$seed phase=eval_extra begin"
    wait_gpu || { log "seed=$seed GPU busy too long; abort"; exit 7; }
    $PY scripts/eval_armb_extra.py --seed "$seed" > "$OUT/eval_armb_seed${seed}.log" 2>&1
    rc=$?
    log "seed=$seed eval_extra rc=$rc"
    if [ $rc -ne 0 ]; then log "seed=$seed eval_extra failed; abort (resume-safe)"; exit 8; fi
  fi
  touch "$EDONE"
  log "seed=$seed EVAL_DONE marker"
done
$PY scripts/merge_armb_summary.py > "$OUT/merge_summary.log" 2>&1
log "summary rc=$?"
log "ARM_B_DONE"
exit 0
