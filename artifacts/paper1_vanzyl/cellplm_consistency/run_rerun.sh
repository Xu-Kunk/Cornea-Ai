#!/bin/bash
set -euo pipefail
WRK=/home/summer/corneal_scgpt/artifacts/paper1_vanzyl/cellplm_consistency
PY=/home/summer/corneal_scgpt/venv/bin/python
DRIVER=/home/summer/cellplm_vanzyl/scripts/run_cellplm_cross_driver.py
LOG="$WRK/driver_rerun.log"
HB="$WRK/heartbeat_rerun.log"

cd /home/summer/corneal_scgpt

# background heartbeat: one line per minute while this script runs
(
  while true; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) HEARTBEAT gpu=\"$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)\" MB" >> "$HB"
    sleep 60
  done
) &
HB_PID=$!

# run driver under setsid nohup
setsid nohup "$PY" "$DRIVER" >> "$LOG" 2>&1 &
DPID=$!

# record PIDs
echo "DRIVER_PID=$DPID" >> "$LOG"
echo "HEARTBEAT_PID=$HB_PID" >> "$LOG"

# wait for driver to finish
wait $DPID
RC=$?

# stop heartbeat
kill $HB_PID 2>/dev/null || true
wait $HB_PID 2>/dev/null || true

echo "DRIVER_EXIT=$RC" >> "$LOG"
exit $RC
