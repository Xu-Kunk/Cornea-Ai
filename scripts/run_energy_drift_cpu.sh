#!/usr/bin/env bash
# Overnight CPU chain for the Paper 1 energy-model extension (GPU owned by fix-7):
#   1) A1 feasibility: EnergyAnchorTrainer CPU forward/backward checks
#      -> artifacts/paper1_energy_drift/feasibility_a1.json
#   2) A5 energy-drift layerwise curves (3000 cells, all 13 hidden states)
#      -> artifacts/paper1_energy_drift/energy_drift_{arm}.json + summary.json
#
# Runs sequentially (no GPU, no contention with the A1 grid launcher which
# waits for the GPU). Launch from the project root:
#   setsid bash scripts/run_energy_drift_cpu.sh > artifacts/paper1_energy_drift/cpu_chain.log 2>&1 &

set -u
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-<venv>/bin/python3}"
LOG_DIR="${PROJECT}/artifacts/paper1_energy_drift"
mkdir -p "${LOG_DIR}"

ts() { date +"[%H:%M:%S]"; }

echo "$(ts) [1/2] A1 feasibility (EnergyAnchorTrainer CPU forward/backward)..."
"${PY}" -u "${PROJECT}/scripts/feasibility_energy_anchor.py" || echo "$(ts) A1 feasibility FAILED (rc=$?)"

echo "$(ts) [2/2] A5 layerwise energy drift (3000 cells, CPU)..."
"${PY}" -u "${PROJECT}/scripts/measure_energy_drift.py" \
    --source layerwise --max-cells 3000 --batch-size 32 --device cpu \
    || echo "$(ts) A5 layerwise FAILED (rc=$?)"

echo "$(ts) chain complete -> ${LOG_DIR}"
