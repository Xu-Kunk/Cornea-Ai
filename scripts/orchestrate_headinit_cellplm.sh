#!/usr/bin/env bash
# Head-init perturbation + CellPLM layerwise orchestrator (<compute-node>, FROZEN 2026-08-20)
# Sequence: (1) CellPLM layerwise (forward, GPU guard inside, ~30 min after GPU frees)
#           (2) scGPT head-init runner (3 variants x 15 epochs, GPU guard per variant, ~7h)
#           (3) head-init test-donor eval (GPU guard inside)
# Both scripts embed the GPU guard (free >= 4 GiB, poll 60 s, wait budget 5 h).
set -u
SCGPT_PY=<home>/corneal_scgpt/repo/.venv/bin/python
CELLPLM_PY=<home>/corneal_scgpt/venv/bin/python
LANE=<home>/corneal_scgpt/artifacts/paper1_vanzyl/head_init/lane.log
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "{\"ts\": \"$NOW\", \"event\": \"ORCH_START\", \"msg\": \"head_init + cellplm_layerwise\"}" >> "$LANE"

echo "[$(date -u +%H:%M:%S)] STEP 1/2: CellPLM layerwise (frozen/naive/corn)"
cd <home>/cellplm_vanzyl/scripts || exit 2
"$CELLPLM_PY" cellplm_layerwise_e1.py --device cuda \
    >> <home>/cellplm_vanzyl/artifacts/paper1_vanzyl/cellplm_layerwise/e1_run.log 2>&1
RC1=$?
echo "[$(date -u +%H:%M:%S)] CellPLM layerwise rc=$RC1"
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "{\"ts\": \"$NOW\", \"event\": \"ORCH_CELLPLM_LAYERWISE\", \"msg\": \"rc=$RC1\"}" >> "$LANE"

echo "[$(date -u +%H:%M:%S)] STEP 2/2: scGPT head-init runner (3 variants)"
cd <home>/corneal_scgpt/scripts || exit 2
"$SCGPT_PY" run_headinit_vanzyl.py
RC2=$?
echo "[$(date -u +%H:%M:%S)] head-init runner rc=$RC2"
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "{\"ts\": \"$NOW\", \"event\": \"ORCH_HEADINIT\", \"msg\": \"rc=$RC2\"}" >> "$LANE"
echo "{\"ts\": \"$NOW\", \"event\": \"ORCH_DONE\", \"msg\": \"rc1=$RC1 rc2=$RC2\"}" >> "$LANE"
exit $((RC1 != 0 || RC2 != 0))
