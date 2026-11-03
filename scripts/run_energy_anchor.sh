#!/usr/bin/env bash
# EnergyAnchor-GF grid launcher (Paper 1 energy-model A1 arm) — queue behind fix-7.
#
# Queues A1 arms sequentially on the single GB10 GPU, mirroring
# run_hp_aft.sh / run_ordinal_grid.py conventions:
#   arm A1  : --energy-anchor --lambda-energy 0.5 (MMD anchor, penultimate L11)
#   x seeds {42,123,456}  (--first => seed 42 only, feasibility set)
#
# Each run: waits for GPU cool + RAM headroom (same guards as run_hp_aft.sh),
# streams stdout+stderr to save/energy_anchor_grid_logs/, is skipped if its
# ENERGY_ANCHOR_DONE marker already exists (idempotent), and appends a status
# line to save/energy_anchor_grid_logs/runs_status.tsv. On any failure the
# grid CONTINUES to the next run and the exit code is reported.
#
# LAUNCH (GPU is owned by fix-7 right now; each run waits for it to free):
#   nohup bash scripts/run_energy_anchor.sh \
#       > save/energy_anchor_grid_logs/grid_launch.log 2>&1 &
#   tail -f save/energy_anchor_grid_logs/grid_launch.log
#
# Full grid is the default (seeds 42 123 456); --first => seed 42 only.
#
# Environment overrides: PY (interpreter), LAMBDAS, SEEDS, DATA_FILE,
# SPLIT_MANIFEST, EPOCHS, KERNEL_BANDWIDTH, ANCHOR_HIDDEN.

set -u

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/dq/venv-ds/bin/python3}"
LOG_DIR="${PROJECT}/save/energy_anchor_grid_logs"
STATUS_TSV="${LOG_DIR}/runs_status.tsv"
mkdir -p "${LOG_DIR}"

LAMBDAS="${LAMBDAS:-0.5}"
SEEDS="${SEEDS:-42 123 456}"
DATA_FILE="${DATA_FILE:-corneal_100k_donor_split.h5ad}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/paper1_fair_split/manifest.json}"
EPOCHS="${EPOCHS:-3}"
KERNEL_BANDWIDTH="${KERNEL_BANDWIDTH:-}"
ANCHOR_HIDDEN="${ANCHOR_HIDDEN:-11}"
RUN_TIMEOUT_S=172800   # 48 h per run (measured ~7.6 h for the CAFT baseline)

MIN_AVAIL_GB=40.0
COOL_TEMP_C=80
TEMP_WAIT_MAX_S="${TEMP_WAIT_MAX_S:-43200}"   # 12 h per-run wait budget (fix-7 owns the GPU)

ts() { date +"[%H:%M:%S]"; }

gpu_temp_c() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{m=$1} END {print m+0}'
}

mem_available_gb() {
  awk '/MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo
}

wait_until_ready() {
  local tag="$1" t0 now temp avail nprocs reason
  t0=$(date +%s)
  while :; do
    temp=$(gpu_temp_c); avail=$(mem_available_gb)
    # any GPU compute process during the wait phase is FOREIGN (this grid runs
    # no training while waiting) — fix-7's job must be fully gone, not just cool.
    nprocs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | grep -cE "[0-9]+" || true)
    if [ -n "${temp}" ] && [ "${temp}" -lt "${COOL_TEMP_C}" ] \
       && [ "${nprocs}" -eq 0 ] \
       && awk "BEGIN{exit !(${avail} >= ${MIN_AVAIL_GB})}"; then
      echo "$(ts) ${tag}  ready (GPU ${temp} C, no foreign GPU procs, ${avail} GiB avail)"
      return 0
    fi
    now=$(date +%s)
    if [ $((now - t0)) -gt "${TEMP_WAIT_MAX_S}" ]; then
      echo "$(ts) ${tag}  ERROR not ready after $((TEMP_WAIT_MAX_S/60)) min (GPU ${temp} C, ${nprocs} foreign procs, ${avail} GiB avail); aborting grid."
      exit 1
    fi
    reason="GPU ${temp} C, ${nprocs} foreign procs"
    awk "BEGIN{exit !(${avail} < ${MIN_AVAIL_GB})}" && reason="${reason}, ${avail} GiB avail"
    echo "$(ts) ${tag}  waiting (${reason})..."
    sleep 60
  done
}

run_one() {
  local lam="$1" seed="$2"
  local tag="lam${lam}_seed${seed}"
  local log="${LOG_DIR}/energy_anchor_${tag}.log" save_dir
  save_dir=$(ls -d "${PROJECT}"/save/energy_anchor_gf_lam${lam}_seed${seed}_* 2>/dev/null | tail -1)
  if [ -n "${save_dir}" ] && [ -f "${save_dir}/ENERGY_ANCHOR_DONE" ]; then
    echo "$(ts) ${tag}  already trained (ENERGY_ANCHOR_DONE) -> skip"
    echo -e "${tag}\tSKIP\t${save_dir}" >> "${STATUS_TSV}"
    return 0
  fi
  wait_until_ready "${tag}"
  echo "$(ts) ${tag}  training starting ..."
  local t0 rc bw_args=()
  if [ -n "${KERNEL_BANDWIDTH}" ]; then
    bw_args=(--kernel-bandwidth "${KERNEL_BANDWIDTH}")
  fi
  t0=$(date +%s)
  "${PY}" -u "${PROJECT}/scripts/run_energy_anchor.py" \
    --energy-anchor --lambda-energy "${lam}" --anchor-hidden "${ANCHOR_HIDDEN}" \
    "${bw_args[@]}" \
    --data_file "${DATA_FILE}" --epochs "${EPOCHS}" --seed "${seed}" \
    --strict-fair --split_manifest "${SPLIT_MANIFEST}" \
    > "${log}" 2>&1 &
  local pid=$!
  # 48 h watchdog per run
  local waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 300; waited=$((waited + 300))
    if [ "${waited}" -ge "${RUN_TIMEOUT_S}" ]; then
      echo "$(ts) ${tag}  TIMEOUT after ${RUN_TIMEOUT_S}s; killing."
      kill -9 "${pid}" 2>/dev/null
    fi
  done
  wait "${pid}"; rc=$?
  local wall=$(( ($(date +%s) - t0) / 60 ))
  save_dir=$(ls -d "${PROJECT}"/save/energy_anchor_gf_lam${lam}_seed${seed}_* 2>/dev/null | tail -1)
  if [ "${rc}" -eq 0 ] && [ -n "${save_dir}" ] && [ -f "${save_dir}/ENERGY_ANCHOR_DONE" ]; then
    echo "$(ts) ${tag}  TRAINED (${wall} min) -> ${save_dir}"
    echo -e "${tag}\tOK\t${wall}\t${save_dir}" >> "${STATUS_TSV}"
  else
    echo "$(ts) ${tag}  FAILED (rc=${rc}, ${wall} min); log: ${log}"
    tail -40 "${log}" 2>/dev/null
    echo -e "${tag}\tFAIL\t${wall}\t${log}" >> "${STATUS_TSV}"
  fi
}

main() {
  local first=0
  for a in "$@"; do
    case "${a}" in
      --first) first=1 ;;
      -h|--help) sed -n '1,42p' "$0"; exit 0 ;;
      *) echo "unknown arg: ${a}"; exit 2 ;;
    esac
  done

  echo "$(ts) ============================================================"
  echo "$(ts) EnergyAnchor-GF (A1) grid: lambda_e x seeds"
  if [ "${first}" -eq 1 ]; then
    SEEDS="42"; echo "$(ts) --first mode: seed 42 only (feasibility set)"
  fi
  for lam in ${LAMBDAS}; do
    for s in ${SEEDS}; do
      echo "$(ts)   lambda_e=${lam} anchor_hidden=${ANCHOR_HIDDEN} seed=${s}"
    done
  done
  echo "$(ts) NOTE: fix-7 owns the GPU right now; each run waits"
  echo "$(ts) for GPU temp < ${COOL_TEMP_C} C and >= ${MIN_AVAIL_GB} GiB RAM."
  echo "$(ts) ============================================================"

  for lam in ${LAMBDAS}; do
    for s in ${SEEDS}; do
      run_one "${lam}" "${s}"
    done
  done
  echo "$(ts) grid complete -> ${STATUS_TSV}"
}

main "$@"
