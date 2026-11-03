#!/bin/bash
# gf_retry_launch.sh — auto-stagger GF frozen diagnosis against GPU contention
# with the scGPT session (diag5.py). Waits for GPU mem <= threshold, launches
# gf_frozen_diag_vanzyl.py --batch 16; on OOM (exit 3) waits and relaunches.
# History in gf_retry.log; final script log kept clean (truncated per run).
cd <home>/corneal_scgpt || exit 9
OUT=artifacts/paper1_vanzyl/g1_gf
RETRY=$OUT/gf_retry.log
THRESH_MB=2000
for attempt in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "[$(date +%H:%M:%S)] attempt=$attempt GPU_used=${used}MiB waiting <=${THRESH_MB}MiB" >> "$RETRY"
  while [ "${used:-99999}" -gt "$THRESH_MB" ]; do
    sleep 15
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  done
  echo "[$(date +%H:%M:%S)] attempt=$attempt window free (${used}MiB); launching GF" >> "$RETRY"
  venv/bin/python scripts/gf_frozen_diag_vanzyl.py \
      --out "$OUT" --batch 16 > "$OUT/gf_frozen_diag_vanzyl.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] attempt=$attempt GF exit=$rc" >> "$RETRY"
  [ "$rc" -eq 0 ] && echo "[$(date +%H:%M:%S)] SUCCESS" >> "$RETRY" && exit 0
  [ "$rc" -eq 3 ] && sleep 10 && continue   # OOM contention -> next window
  echo "[$(date +%H:%M:%S)] non-OOM failure rc=$rc; giving up" >> "$RETRY"
  exit $rc
done
echo "[$(date +%H:%M:%S)] gave up after 30 attempts" >> "$RETRY"
exit 7
