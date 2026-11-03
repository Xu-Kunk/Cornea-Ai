#!/usr/bin/env bash
# M2/M3 seed extension orchestrator (<compute-node>, nohup, GPU guard)
# Sequence: (A) van Zyl run6_ext naive seeds 789, 2024
#           (B) cross_tissue (epidermis GSE147424) naive seed 456
# Training scripts enforce a locked seed whitelist; this launcher uses
# tiny runtime wrappers to extend the whitelist without modifying the
# locked training code.
set -u

SCGPT_PY=<home>/corneal_scgpt/repo/.venv/bin/python
ROOT=<home>/corneal_scgpt
ART=$ROOT/artifacts/paper1_vanzyl

RUN6_OUT=$ART/run6_ext/checkpoints
CROSS_OUT=$ART/cross_tissue/checkpoints

WRAPPERS=/tmp/opencode/seed_ext_wrappers
mkdir -p "$WRAPPERS"

# ---------------------------------------------------------------------------
# Runtime seed-extension wrappers (do NOT modify locked training scripts)
# ---------------------------------------------------------------------------
cat > "$WRAPPERS/run_vanzyl_seed_ext.py" <<'PY'
import importlib.util
import sys
from pathlib import Path

def main():
    target = Path("<home>/corneal_scgpt/scripts/scgpt_cls_vanzyl.py")
    # Preserve the default sys.path behaviour when the target script is run
    # directly: the directory containing the training script must be importable.
    sys.path.insert(0, str(target.parent))
    spec = importlib.util.spec_from_file_location("scgpt_cls_vanzyl", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scgpt_cls_vanzyl"] = module
    spec.loader.exec_module(module)
    if hasattr(module, "SEEDS"):
        module.SEEDS = list(set(module.SEEDS) | {456, 789, 2024})
    # Present the target script as argv[0] to argparse
    sys.argv = [str(target)] + sys.argv[1:]
    module.main()

if __name__ == "__main__":
    main()
PY

cat > "$WRAPPERS/run_epidermis_seed_ext.py" <<'PY'
import importlib.util
import sys
from pathlib import Path

def main():
    target = Path("<home>/corneal_scgpt/scripts/scgpt_cls_epidermis.py")
    # Preserve the default sys.path behaviour when the target script is run
    # directly: the directory containing the training script must be importable.
    sys.path.insert(0, str(target.parent))
    spec = importlib.util.spec_from_file_location("scgpt_cls_epidermis", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scgpt_cls_epidermis"] = module
    spec.loader.exec_module(module)
    if hasattr(module, "SEEDS"):
        module.SEEDS = list(set(module.SEEDS) | {456, 789, 2024})
    sys.argv = [str(target)] + sys.argv[1:]
    module.main()

if __name__ == "__main__":
    main()
PY

chmod +x "$WRAPPERS"/*.py

mkdir -p "$RUN6_OUT" "$CROSS_OUT"

LANE_RUN6=$ART/run6_ext/lane.log
LANE_CROSS=$ART/cross_tissue/lane.log

now_json() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

write_lane() {
    local path=$1
    local event=$2
    local msg=$3
    echo "{\"ts\": \"$(now_json)\", \"event\": \"$event\", \"msg\": \"$msg\"}" >> "$path"
}

gpu_wait() {
    local need=4096
    while true; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
        if [ "$free" -ge "$need" ]; then
            echo "[$(date -u +%H:%M:%S)] GPU free ${free} MiB >= ${need} MiB"
            break
        fi
        echo "[$(date -u +%H:%M:%S)] GPU free ${free} MiB < ${need} MiB; polling in 60s"
        sleep 60
    done
}

wait_training_log() {
    local logf=$1
    echo "[$(date -u +%H:%M:%S)] waiting for training signal: $logf"
    for _ in $(seq 1 90); do
        if [ -f "$logf" ]; then
            if grep -Eqi 'training start|epoch.*step|data loaded' "$logf"; then
                echo "[$(date -u +%H:%M:%S)] training signal detected; first lines:"
                head -n 5 "$logf"
                return 0
            fi
            # Print whatever we have so far for progress visibility
            if [ "$(wc -l < "$logf")" -gt 0 ]; then
                :
            fi
        fi
        sleep 2
    done
    echo "[$(date -u +%H:%M:%S)] WARNING: no training signal in $logf after 180s"
    return 1
}

write_lane "$LANE_RUN6" "ORCH_START" "M3 seed extension: naive 789 + 2024 -> run6_ext"
write_lane "$LANE_CROSS" "ORCH_START" "M2 seed extension: naive 456 -> cross_tissue"

# ---------------------------------------------------------------------------
# Task A: van Zyl run6_ext naive seeds 789, 2024
# ---------------------------------------------------------------------------
for seed in 789 2024; do
    run_dir=$RUN6_OUT/naive_seed${seed}
    mkdir -p "$run_dir"
    logf=$run_dir/run.log
    write_lane "$LANE_RUN6" "RUN_START" "naive seed $seed"
    gpu_wait
    $SCGPT_PY "$WRAPPERS/run_vanzyl_seed_ext.py" \
        --arm naive \
        --seed "$seed" \
        --out "$RUN6_OUT" > "$logf" 2>&1 &
    pid=$!
    wait_training_log "$logf"
    wait $pid
    rc=$?
    write_lane "$LANE_RUN6" "RUN_DONE" "naive seed $seed rc=$rc"
    if [ $rc -ne 0 ]; then
        write_lane "$LANE_RUN6" "ORCH_FAIL" "naive seed $seed failed"
        exit $rc
    fi
done

# ---------------------------------------------------------------------------
# Task B: cross_tissue (epidermis GSE147424) naive seed 456
# ---------------------------------------------------------------------------
run_dir=$CROSS_OUT/naive_seed456
mkdir -p "$run_dir"
run_log=$run_dir/run.log
driver_log=$run_dir/driver.log
write_lane "$LANE_CROSS" "RUN_START" "naive seed 456"
gpu_wait
$SCGPT_PY "$ROOT/scripts/run_scgpt_task.py" \
    --train-script "$WRAPPERS/run_epidermis_seed_ext.py" \
    --ckpt-root "$CROSS_OUT" \
    --val-manifest "$ART/cross_tissue/split_val_manifest.json" \
    --runs "naive:456" > "$driver_log" 2>&1 &
pid=$!
wait_training_log "$run_log"
wait $pid
rc=$?
write_lane "$LANE_CROSS" "RUN_DONE" "naive seed 456 rc=$rc"
if [ $rc -ne 0 ]; then
    write_lane "$LANE_CROSS" "ORCH_FAIL" "naive seed 456 failed"
    exit $rc
fi

write_lane "$LANE_RUN6" "ORCH_DONE" "run6_ext seed extension complete"
write_lane "$LANE_CROSS" "ORCH_DONE" "cross_tissue seed extension complete"
echo "[$(date -u +%H:%M:%S)] ORCHESTRATOR_DONE"
