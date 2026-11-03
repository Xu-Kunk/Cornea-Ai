#!/usr/bin/env bash
# Waits for the head-init driver (run_headinit_vanzyl.py) to exit, then runs the
# CellPLM layerwise re-run (frozen-seed fix). Detached on p5070.
set -u
CELLPLM_PY=/home/summer/corneal_scgpt/venv/bin/python
OUT=/home/summer/cellplm_vanzyl/artifacts/paper1_vanzyl/cellplm_layerwise
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "{\"ts\": \"$NOW\", \"event\": \"WATCHER_START\", \"msg\": \"waiting for head-init driver\"}" >> "$OUT/lane.log" 2>/dev/null || true

# wait until the head-init runner is gone (driver + any training subprocess)
while ps aux | grep -E "run_headinit_vanzyl\.py|scgpt_cls_vanzyl_headinit\.py" | grep -v grep > /dev/null; do
    sleep 300
done
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "{\"ts\": \"$NOW\", \"event\": \"WATCHER_HEADINIT_DONE\", \"msg\": \"launching cellplm layerwise (frozen-seed fix)\"}" >> "$OUT/lane.log" 2>/dev/null || true

cd /home/summer/cellplm_vanzyl/scripts || exit 2
"$CELLPLM_PY" cellplm_layerwise_e1.py --device cuda >> "$OUT/e1_run_frozenfix.log" 2>&1
RC=$?
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "{\"ts\": \"$NOW\", \"event\": \"WATCHER_CELLPLM_DONE\", \"msg\": \"rc=$RC\"}" >> "$OUT/lane.log" 2>/dev/null || true
exit $RC
