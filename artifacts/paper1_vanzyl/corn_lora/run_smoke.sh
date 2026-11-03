#!/bin/bash
# E5 smoke: identity-check + dry-run (guarded free>=4GiB)
cd <home>/corneal_scgpt
PY=repo/.venv/bin/python
SCRIPT=scripts/scgpt_lora_corn_vanzyl.py
OUT=artifacts/paper1_vanzyl/corn_lora
LOG=$OUT/smoke.log
> $LOG
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "[$(date -u +%H:%M:%S)] GPU free ${FREE} MiB" >> $LOG
  [ "$FREE" -ge 4096 ] && break
  sleep 60
done
echo "=== identity-check ===" >> $LOG
$PY $SCRIPT --identity-check --seed 42 >> $LOG 2>&1
echo "rc=$?" >> $LOG
echo "=== dry-run (3000 cells, 1 epoch) ===" >> $LOG
$PY $SCRIPT --seed 42 --dry-run --max-cells 3000 --out $OUT >> $LOG 2>&1
echo "rc=$?" >> $LOG
echo "SMOKE_DONE" >> $LOG
