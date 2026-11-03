#!/usr/bin/env bash
# gx10_gpr_recovery.sh — GPR paired-comparison recovery preflight wrapper.
# Sources gx10_gpr_preflight.sh for all validation gates, then delegates
# execution through gx10_cgroup_run.sh --mode service.
#
# Usage:
#   gx10_gpr_recovery.sh --before-checkpoint PATH --after-checkpoint PATH \
#       --tokenized PATH --out PATH [--num-classes N] [--dry-run] [--quiet]
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gx10_gpr_preflight.sh"

readonly CGROUP_LAUNCHER="${SCRIPT_DIR}/gx10_cgroup_run.sh"
readonly PINNED_PYTHON="/home/dq/venv-ds/bin/python"
readonly GPR_MEMORY_MAX="100G"
readonly GPR_MEMORY_SWAP_MAX="0"
readonly GPR_RUNTIME_MAX="86400"
readonly GPR_OOM_SCORE_ADJUST="500"

BEFORE_CHECKPOINT="" AFTER_CHECKPOINT="" TOKENIZED_PATH="" OUT_PATH=""
NUM_CLASSES="" DRY_RUN=false QUIET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --before-checkpoint) BEFORE_CHECKPOINT="$2"; shift 2 ;;
        --after-checkpoint)  AFTER_CHECKPOINT="$2";  shift 2 ;;
        --tokenized)         TOKENIZED_PATH="$2";    shift 2 ;;
        --out)               OUT_PATH="$2";           shift 2 ;;
        --num-classes)       NUM_CLASSES="$2";        shift 2 ;;
        --dry-run)           DRY_RUN=true;            shift   ;;
        --quiet)             QUIET=true;              shift   ;;
        *) echo "ERROR: unknown argument: $1" >&2
           echo "Usage: $0 --before-checkpoint PATH --after-checkpoint PATH --tokenized PATH --out PATH [--num-classes N] [--dry-run] [--quiet]" >&2
           exit 1 ;;
    esac
done

if [[ -z "$BEFORE_CHECKPOINT" || -z "$AFTER_CHECKPOINT" || -z "$TOKENIZED_PATH" || -z "$OUT_PATH" ]]; then
    echo "ERROR: all four path arguments are required" >&2; exit 1
fi
for p in "$BEFORE_CHECKPOINT" "$AFTER_CHECKPOINT" "$TOKENIZED_PATH"; do
    [[ "$p" == /* ]] || fail "$p is not an absolute path (required)"
done

# --num-classes: default 20, must be positive integer
if [[ -z "$NUM_CLASSES" ]]; then
    NUM_CLASSES=20
elif ! [[ "$NUM_CLASSES" =~ ^[0-9]+$ ]] || [[ "$NUM_CLASSES" -eq 0 ]]; then
    echo "ERROR: --num-classes must be a positive integer (got '$NUM_CLASSES')" >&2; exit 1
fi

_run_preflight_all "$BEFORE_CHECKPOINT" "$AFTER_CHECKPOINT" "$TOKENIZED_PATH" "$PINNED_PYTHON"

out_abs="$(realpath "$OUT_PATH" 2>/dev/null || echo "$OUT_PATH")"
log_dir="$(dirname "$out_abs")/gpr_recovery_logs"

CGROUP_ARGS=(
    "--mode" "service"
    "--memory-max" "$GPR_MEMORY_MAX" "--memory-swap-max" "$GPR_MEMORY_SWAP_MAX"
    "--runtime-max" "$GPR_RUNTIME_MAX" "--oom-score-adjust" "$GPR_OOM_SCORE_ADJUST"
    "--log-dir" "$log_dir" "--unit" "gx10-gpr-recovery-$(date +%Y%m%d-%H%M%S)"
    "--" "$PINNED_PYTHON" "-m" "corneal_fm.gpr"
    "--before_checkpoint" "$BEFORE_CHECKPOINT" "--after_checkpoint" "$AFTER_CHECKPOINT"
    "--tokenized" "$TOKENIZED_PATH" "--out" "$OUT_PATH"
    "--seed" "42" "--n_boot" "1000"
    "--num_classes" "$NUM_CLASSES"
)

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "PREFLIGHT: all gates passed (dry-run). Would invoke cgroup launcher:"
    echo "  $CGROUP_LAUNCHER \\"
    for ((i = 0; i < ${#CGROUP_ARGS[@]}; i++)); do
        printf '    %q' "${CGROUP_ARGS[$i]}"
        if (( (i + 1) % 6 == 0 )) && (( i + 1 < ${#CGROUP_ARGS[@]} )); then
            printf ' \\'
        fi
        echo ""
    done
    echo ""
    exit 0
fi

echo "Launching GPR recovery via $CGROUP_LAUNCHER ..."
exec bash "$CGROUP_LAUNCHER" "${CGROUP_ARGS[@]}"
