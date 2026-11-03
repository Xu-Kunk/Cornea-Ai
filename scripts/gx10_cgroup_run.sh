#!/usr/bin/env bash
# gx10_cgroup_run.sh — Fail-fast GX10 cgroup launcher for Corneal-FM training.
#
# Two modes:
#   service  Transient systemd service (supports OOMScoreAdjust + all exec properties).
#             Uses systemd-run --service-type=exec with full property set.
#   scope    systemd-run --scope (cgroup resource properties only).
#             Rejects --oom-score-adjust locally before calling systemd.
#
# Usage:
#   gx10_cgroup_run.sh --mode service|scope [--dry-run] [--unit NAME] \
#       [--memory-max SIZE] [--memory-swap-max SIZE] \
#       [--runtime-max SECONDS] [--log-dir PATH] \
#       [--oom-score-adjust VALUE]   # service only
#       -- COMMAND [ARGS...]
#
# Exit codes:
#   0   Success (dry-run printed; or job launched)
#   1   Argument / validation error (fail-fast, before any systemd call)
#   2   systemd-run returned non-zero (only when not --dry-run)
#
# GX10 context: GB10 aarch64 / SM121 / CUDA13 / 128GB UMA.
# systemd-run --scope -p OOMScoreAdjust=500 FAILS because scope units
# lack ExecProperties=. Use service mode when OOMScoreAdjust is needed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
readonly DEFAULT_MEMORY_HIGH="100G"
readonly DEFAULT_TIMEOUT_STOP_SEC=120
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE=""
DRY_RUN=false
UNIT=""
MEMORY_MAX=""
MEMORY_SWAP_MAX=""
RUNTIME_MAX=""
LOG_DIR=""
OOM_SCORE_ADJUST=""

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --unit)
            UNIT="$2"; shift 2 ;;
        --memory-max)
            MEMORY_MAX="$2"; shift 2 ;;
        --memory-swap-max)
            MEMORY_SWAP_MAX="$2"; shift 2 ;;
        --runtime-max)
            RUNTIME_MAX="$2"; shift 2 ;;
        --log-dir)
            LOG_DIR="$2"; shift 2 ;;
        --oom-score-adjust)
            OOM_SCORE_ADJUST="$2"; shift 2 ;;
        --)
            shift
            POSITIONAL_ARGS=("$@")
            break ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required (service | scope)" >&2
    exit 1
fi

if [[ "$MODE" != "service" && "$MODE" != "scope" ]]; then
    echo "ERROR: --mode must be 'service' or 'scope', got '$MODE'" >&2
    exit 1
fi

if [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: no command provided after -- separator" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Scope + OOMScoreAdjust rejection (fail-fast, BEFORE any systemd call)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "scope" && -n "$OOM_SCORE_ADJUST" ]]; then
    echo "ERROR: --oom-score-adjust is NOT supported with --mode scope." >&2
    echo "  Scope units lack ExecProperties=; OOMScoreAdjust is an exec property." >&2
    echo "  Use --mode service for OOMScoreAdjust support." >&2
    echo "  systemd-run --scope -p OOMScoreAdjust=N → 'Unknown assignment OOMScoreAdjust'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Generate unit name
# ---------------------------------------------------------------------------
if [[ -z "$UNIT" ]]; then
    cmd_name="${POSITIONAL_ARGS[0]##*/}"  # basename of command
    ts="$(date +%Y%m%d-%H%M%S)"
    UNIT="gx10-${MODE}-${cmd_name}-${ts}"
fi

# ---------------------------------------------------------------------------
# Build systemd-run command
# ---------------------------------------------------------------------------
SYSTEMD_CMD=("systemd-run")

# --- Mode-specific wrapper ---
if [[ "$MODE" == "service" ]]; then
    SYSTEMD_CMD+=(
        "--service-type=exec"
        "--wait"
        "--collect"
    )
elif [[ "$MODE" == "scope" ]]; then
    SYSTEMD_CMD+=("--scope")
fi

# --- Unit ---
SYSTEMD_CMD+=("--unit" "$UNIT")

# --- Working directory ---
SYSTEMD_CMD+=("--working-directory" "$PROJECT_ROOT")

# --- Cgroup resource properties (both modes) ---
if [[ -n "$MEMORY_MAX" ]]; then
    SYSTEMD_CMD+=("-p" "MemoryMax=$MEMORY_MAX")
    # MemoryHigh defaults to 100G when MemoryMax is set and no explicit high
    SYSTEMD_CMD+=("-p" "MemoryHigh=$DEFAULT_MEMORY_HIGH")
else
    # Without MemoryMax, still set MemoryHigh as ceiling
    SYSTEMD_CMD+=("-p" "MemoryHigh=$DEFAULT_MEMORY_HIGH")
fi

if [[ -n "$MEMORY_SWAP_MAX" ]]; then
    SYSTEMD_CMD+=("-p" "MemorySwapMax=$MEMORY_SWAP_MAX")
fi

SYSTEMD_CMD+=(
    "-p" "MemoryAccounting=yes"
    "-p" "TasksAccounting=yes"
    "-p" "OOMPolicy=kill"
)

# --- Runtime limit (service only via RuntimeMaxSec) ---
if [[ -n "$RUNTIME_MAX" ]]; then
    if [[ "$MODE" == "service" ]]; then
        SYSTEMD_CMD+=("-p" "RuntimeMaxSec=$RUNTIME_MAX")
        SYSTEMD_CMD+=("-p" "TimeoutStopSec=$DEFAULT_TIMEOUT_STOP_SEC")
    else
        # Scope mode: RuntimeMaxSec is unsupported; log a warning but continue
        echo "WARNING: --runtime-max ignored in scope mode (RuntimeMaxSec is an exec property)" >&2
    fi
fi

# --- OOMScoreAdjust (service only) ---
if [[ -n "$OOM_SCORE_ADJUST" ]]; then
    if [[ "$MODE" == "service" ]]; then
        SYSTEMD_CMD+=("-p" "OOMScoreAdjust=$OOM_SCORE_ADJUST")
    fi
fi

# --- Logging ---
if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    log_stdout="${LOG_DIR}/${UNIT}.out"
    log_stderr="${LOG_DIR}/${UNIT}.err"
    SYSTEMD_CMD+=(
        "-p" "StandardOutput=file:$log_stdout"
        "-p" "StandardError=file:$log_stderr"
    )
fi

# --- Environment: PyTorch allocator for unified memory ---
SYSTEMD_CMD+=(
    "-E" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
    "-E" "CORNEAL_FM_DATA=${PROJECT_ROOT}/data"
)

# --- User command ---
SYSTEMD_CMD+=("--")
SYSTEMD_CMD+=("${POSITIONAL_ARGS[@]}")

# ---------------------------------------------------------------------------
# Execute or print
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    echo "# Dry-run: generated systemd-run command"
    echo "# Unit:    $UNIT"
    echo "# Mode:    $MODE"
    echo "# Working: $PROJECT_ROOT"
    echo ""
    # Print command with proper quoting for copy-paste
    printf '%q ' "${SYSTEMD_CMD[@]}"
    echo ""
    exit 0
fi

# --- Real execution ---
echo "Launching $MODE unit '$UNIT' ..."
"${SYSTEMD_CMD[@]}"
