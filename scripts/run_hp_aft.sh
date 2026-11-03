#!/usr/bin/env bash
# HPA-FT grid launcher (Paper 1 main method) — queue behind fix-7 (D5 ordinal).
#
# Queues HPA-FT arms sequentially on the single GB10 GPU, mirroring
# run_ordinal_grid.py / run_lambda_grid.py conventions:
#   arm A  lwc        : --mode lwc        --lambda 0.5 --cont-layers 3-10
#   arm B  headfreeze : --mode headfreeze --lambda 0.5 (L5-L12 frozen by default)
# x seeds {42,123,456}  (--first => seed 42 only, both arms, feasibility set)
#
# Each run: waits for GPU cool + RAM headroom (same guards as the lambda
# grid), streams stdout+stderr to artifacts/paper1_hp_aft/logs/, is skipped if
# its HPA_DONE marker already exists (idempotent), and appends a status line
# to artifacts/paper1_hp_aft/runs_status.tsv. On any failure the grid
# CONTINUES to the next run and the exit code is reported.
#
# LAUNCH (only after fix-7 has finished with the GPU):
#   nohup bash scripts/run_hp_aft.sh --first \
#       > artifacts/paper1_hp_aft/grid_launch.log 2>&1 &
#   tail -f artifacts/paper1_hp_aft/grid_launch.log
#
# Full grid:
#   nohup bash scripts/run_hp_aft.sh > artifacts/paper1_hp_aft/grid_launch.log 2>&1 &
#
# Environment overrides: PY (interpreter), LAMBDAS, SEEDS, CONT_LWC,
# CONT_HF, DATA_FILE, SPLIT_MANIFEST, EPOCHS.

set -u

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/dq/venv-ds/bin/python3}"
ART="${PROJECT}/artifacts/paper1_hp_aft"
LOG_DIR="${ART}/logs"
STATUS_TSV="${ART}/runs_status.tsv"
mkdir -p "${LOG_DIR}" "${ART}"

LAMBDAS="${LAMBDAS:-0.5}"
SEEDS="${SEEDS:-42 123 456}"
CONT_LWC="${CONT_LWC:-3-10}"
CONT_HF="${CONT_HF:-2-4}"
DATA_FILE="${DATA_FILE:-corneal_100k_donor_split.h5ad}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/paper1_fair_split/manifest.json}"
EPOCHS="${EPOCHS:-3}"
RUN_TIMEOUT_S=172800   # 48 h per run (measured ~7.6 h for the CAFT baseline)

MIN_AVAIL_GB=40.0
COOL_TEMP_C=80
TEMP_WAIT_MAX_S=5400

ts() { date +"[%H:%M:%S]"; }

gpu_temp_c() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{m=$1} END {print m+0}'
}

mem_available_gb() {
  awk '/MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo
}

wait_until_ready() {
  local tag="$1" t0 now temp avail reason
  t0=$(date +%s)
  while :; do
    temp=$(gpu_temp_c); avail=$(mem_available_gb)
    if [ -n "${temp}" ] && [ "${temp}" -lt "${COOL_TEMP_C}" ] \
       && awk "BEGIN{exit !(${avail} >= ${MIN_AVAIL_GB})}"; then
      echo "$(ts) ${tag}  ready (GPU ${temp} C, ${avail} GiB avail)"
      return 0
    fi
    now=$(date +%s)
    if [ $((now - t0)) -gt "${TEMP_WAIT_MAX_S}" ]; then
      echo "$(ts) ${tag}  ERROR not ready after $((TEMP_WAIT_MAX_S/60)) min (GPU ${temp} C, ${avail} GiB); aborting grid."
      exit 1
    fi
    reason="GPU ${temp} C"
    awk "BEGIN{exit !(${avail} < ${MIN_AVAIL_GB})}" && reason="${reason}, ${avail} GiB avail"
    echo "$(ts) ${tag}  waiting (${reason})..."
    sleep 60
  done
}

run_one() {
  local mode="$1" lam="$2" seed="$3" tag="${mode}_lam${lam}_seed${seed}"
  local log="${LOG_DIR}/hp_aft_${tag}.log" save_dir
  save_dir=$(ls -d "${PROJECT}"/save/hp_aft_gf_${mode}_lam${lam}_seed${seed}_* 2>/dev/null | tail -1)
  if [ -n "${save_dir}" ] && [ -f "${save_dir}/HPA_DONE" ]; then
    echo "$(ts) ${tag}  already trained (HPA_DONE) -> skip"
    echo -e "${tag}\tSKIP\t${save_dir}" >> "${STATUS_TSV}"
    return 0
  fi
  wait_until_ready "${tag}"
  echo "$(ts) ${tag}  training starting ..."
  local t0 rc
  t0=$(date +%s)
  if [ "${mode}" = "lwc" ]; then
    "${PY}" -u "${PROJECT}/scripts/run_hp_aft.py" \
      --mode lwc --lambda "${lam}" --cont-layers "${CONT_LWC}" \
      --data_file "${DATA_FILE}" --epochs "${EPOCHS}" --seed "${seed}" \
      --strict-fair --split_manifest "${SPLIT_MANIFEST}" \
      > "${log}" 2>&1 &
  else
    "${PY}" -u "${PROJECT}/scripts/run_hp_aft.py" \
      --mode headfreeze --lambda "${lam}" --cont-layers "${CONT_HF}" \
      --data_file "${DATA_FILE}" --epochs "${EPOCHS}" --seed "${seed}" \
      --strict-fair --split_manifest "${SPLIT_MANIFEST}" \
      > "${log}" 2>&1 &
  fi
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
  save_dir=$(ls -d "${PROJECT}"/save/hp_aft_gf_${mode}_lam${lam}_seed${seed}_* 2>/dev/null | tail -1)
  if [ "${rc}" -eq 0 ] && [ -n "${save_dir}" ] && [ -f "${save_dir}/HPA_DONE" ]; then
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
      -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
      *) echo "unknown arg: ${a}"; exit 2 ;;
    esac
  done

  echo "$(ts) ============================================================"
  echo "$(ts) HPA-FT grid: mode x lambda x seeds"
  if [ "${first}" -eq 1 ]; then
    SEEDS="42"; echo "$(ts) --first mode: seed 42 only (feasibility set)"
  fi
  for lam in ${LAMBDAS}; do
    for s in ${SEEDS}; do
      echo "$(ts)   lwc        lam=${lam} cont=${CONT_LWC} seed=${s}"
      echo "$(ts)   headfreeze lam=${lam} cont=${CONT_HF} seed=${s} (L5-L12 frozen)"
    done
  done
  echo "$(ts) NOTE: fix-7 (D5 ordinal) owns the GPU right now; each run waits"
  echo "$(ts) for GPU temp < ${COOL_TEMP_C} C and >= ${MIN_AVAIL_GB} GiB RAM."
  echo "$(ts) ============================================================"

  for lam in ${LAMBDAS}; do
    for s in ${SEEDS}; do
      run_one lwc "${lam}" "${s}"
      run_one headfreeze "${lam}" "${s}"
    done
  done
  echo "$(ts) grid complete -> ${STATUS_TSV}"
}

main "$@"
