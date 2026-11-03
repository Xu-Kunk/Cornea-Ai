#!/bin/bash
# eval_with_diagnostics.sh — run an epi4-CE evaluation so a failure is diagnosable.
#
# Why (2026-10-21): the first seed-42 evaluation died with rc=1 after 32 s
# (a KeyError in the eval script, since fixed) but the driver's
# `> logs/eval_seed42.log 2>&1` truncated the traceback on the retry, leaving
# only the torch deprecation line. This wrapper never overwrites evidence:
#
#   * stdout and stderr go to SEPARATE files;
#   * the python process runs with `-X faulthandler` (native frames on SIGSEGV/
#     SIGABRT and on timeout) and unbuffered output;
#   * an append-only `*.attempts.log` records a timestamped header per attempt,
#     the exact command, the exit status, and the signal name when rc >= 128;
#   * the raw stdout/stderr of attempt N are kept as `*_attemptN.{out,err}.log`.
#
# Usage:
#   bash scripts/eval_with_diagnostics.sh SEED [EXTRA epi4_ce_eval.py args...]
#   e.g. bash scripts/eval_with_diagnostics.sh 42 --skip-tokenize \
#            --out-dir results/paper1_vanzyl/epi4_ce
set -u
cd <home>/epi4_ce || exit 9
PY=<home>/corneal_scgpt/venv/bin/python3
LOGS=logs
SEED=${1:?usage: eval_with_diagnostics.sh SEED [args...]}
shift
mkdir -p "$LOGS"
ATT="$LOGS/eval_seed${SEED}.attempts.log"
N=$(( $(grep -c '^=== attempt' "$ATT" 2>/dev/null || echo 0) + 1 ))
OUT="$LOGS/eval_seed${SEED}_attempt${N}.out.log"
ERR="$LOGS/eval_seed${SEED}_attempt${N}.err.log"
CMD=($PY -X faulthandler -u scripts/epi4_ce_eval.py --seeds "$SEED" "$@")

{
  echo "=== attempt ${N} $(date '+%F %T %Z')"
  echo "cmd: ${CMD[*]}"
  echo "stdout: ${OUT}"
  echo "stderr: ${ERR}"
} >> "$ATT"

"${CMD[@]}" > "$OUT" 2> "$ERR"
rc=$?
sig=""
if [ "$rc" -ge 128 ]; then
  signo=$(( rc - 128 ))
  sig=$(kill -l "$signo" 2>/dev/null || echo "signal ${signo}")
fi
{
  echo "rc=${rc}${sig:+ signal=${sig}} end=$(date '+%F %T %Z')"
  echo
} >> "$ATT"

echo "rc=${rc}${sig:+ signal=${sig}}"
echo "attempt log: ${ATT}"
exit "$rc"
