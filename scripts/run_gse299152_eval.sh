#!/usr/bin/env bash
# GSE299152 second external test — Tier-B evaluation of 9 locked checkpoints.
# Queues until GPU is free (energy_anchor may occupy it), then runs with nohup.
# Output: artifacts/paper1_tier_b_eval_gse299152/
# Usage: nohup bash scripts/run_gse299152_eval.sh > artifacts/paper1_tier_b_eval_gse299152/queue.log 2>&1 &
set -euo pipefail

EVAL=<venv>/bin/python3
SCRIPT=<repo>/scripts/run_tier_b_eval.py
OUTDIR=<repo>/artifacts/paper1_tier_b_eval_gse299152
mkdir -p "$OUTDIR"

RUNS="caft_gf:0.0:42,caft_gf:0.0:123,caft_gf:0.0:456,ordinal_gf:0.0:42,ordinal_gf:0.0:123,ordinal_gf:0.0:456,ordinal_gf:0.5:42,ordinal_gf:0.5:123,ordinal_gf:0.5:456"

echo "[$(date)] waiting for GPU to free (poll 5 min, cap 12 h)..."
wait_until=0
while :; do
    # 1) no compute process using GPU memory
    compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -cE '^[0-9]+$' || true)
    # 2) GPU util below 30%
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' %' | head -1)
    if [ "${compute:-0}" -le 1 ] && [ "${util:-100}" -lt 30 ]; then
        echo "[$(date)] GPU free (compute=$compute util=$util). launching eval."
        break
    fi
    wait_until=$((wait_until + 1))
    if [ "$wait_until" -ge 144 ]; then  # 144 * 5 min = 12 h
        echo "[$(date)] TIMEOUT waiting for GPU; exiting without running." >&2
        exit 1
    fi
    sleep 300
done

# Give the energy_anchor process a moment to fully release GPU memory.
sleep 60

echo "[$(date)] running: $EVAL $SCRIPT --accession gse299152 --runs $RUNS"
$EVAL "$SCRIPT" --accession gse299152 --runs "$RUNS" 2>&1 | tee "$OUTDIR/eval_run.log"
echo "[$(date)] eval finished, exit=$?"
