#!/usr/bin/env bash
# Tests for scripts/gx10_gpr_recovery.sh — preflight gates, cgroup delegation.
set -euo pipefail

readonly RECOVERY="scripts/gx10_gpr_recovery.sh"
readonly PREFLIGHT="scripts/gx10_gpr_preflight.sh"
readonly LAUNCHER="scripts/gx10_cgroup_run.sh"
readonly TMP_OUT="$(mktemp -t gx10_recovery_test.XXXXXX)"
trap 'rm -f "$TMP_OUT"' EXIT

failures=0 passes=0
pass() { passes=$((passes + 1)); echo -e "  \033[0;32mPASS\033[0m $*"; }
fail() { failures=$((failures + 1)); echo -e "  \033[0;31mFAIL\033[0m $*"; }

echo "=== test_gx10_recovery.sh ==="
echo ""

[[ -f "$RECOVERY" ]] && pass "recovery script exists" || fail "recovery missing"
[[ -f "$PREFLIGHT" ]] && pass "preflight helper exists" || fail "preflight helper missing"

RECOVERY_TMP="$(mktemp -d -t gx10_rec_test.XXXXXX)"

_before_cp="$RECOVERY_TMP/before/model"
mkdir -p "$_before_cp"; touch "$_before_cp"/{model.safetensors,config.json}
_after_base="$RECOVERY_TMP/after"
_after_cp="$_after_base/model/checkpoint-1500"
mkdir -p "$_after_cp"; touch "$_after_cp"/{model.safetensors,config.json}
touch "$_after_base/CAFT_DONE"
_tokenized="$RECOVERY_TMP/dataset"
mkdir -p "$_tokenized"; touch "$_tokenized"/dataset_info.json
_out="$RECOVERY_TMP/gpr_result.json"

# stale 020334
set +e; bash "$RECOVERY" --dry-run --quiet \
    --before-checkpoint "$RECOVERY_TMP/stale_020334_run/model" \
    --after-checkpoint "$_after_cp" --tokenized "$_tokenized" --out "$_out" >"$TMP_OUT" 2>&1
rc=$?; set -e
[ "$rc" -eq 1 ] && pass "rejects stale 020334" || fail "stale 020334 should exit 1 (got $rc)"

# missing CAFT_DONE
_no_done="$RECOVERY_TMP/nodone/model/checkpoint-99"
mkdir -p "$_no_done"; touch "$_no_done"/{model.safetensors,config.json}
set +e; bash "$RECOVERY" --dry-run --quiet \
    --before-checkpoint "$_before_cp" --after-checkpoint "$_no_done" \
    --tokenized "$_tokenized" --out "$_out" >"$TMP_OUT" 2>&1
rc=$?; set -e
[ "$rc" -eq 1 ] && pass "rejects missing CAFT_DONE" || fail "missing CAFT_DONE should exit 1 (got $rc)"

# valid dry-run
set +e; bash "$RECOVERY" --dry-run --quiet \
    --before-checkpoint "$_before_cp" --after-checkpoint "$_after_cp" \
    --tokenized "$_tokenized" --out "$_out" >"$TMP_OUT" 2>&1
rc=$?; set -e
[ "$rc" -eq 0 ] && pass "dry-run passes with valid fixtures" || fail "dry-run should exit 0 (got $rc)"
grep -q "gx10_cgroup_run.sh" "$TMP_OUT" && pass "dry-run mentions cgroup launcher" || fail "dry-run missing cgroup launcher"

# string checks on source
grep -q "/home/dq/venv-ds/bin/python" "$RECOVERY" && pass "pins Python interpreter" || fail "must pin Python"
grep -q "query-compute-apps" "$PREFLIGHT" && pass "checks query-compute-apps" || fail "must check query-compute-apps"
grep -q "expandable_segments:False" "$LAUNCHER" && pass "launcher uses False" || fail "launcher must use False"

# --num-classes passthrough (default 20)
set +e; bash "$RECOVERY" --dry-run --quiet \
    --before-checkpoint "$_before_cp" --after-checkpoint "$_after_cp" \
    --tokenized "$_tokenized" --out "$_out" >"$TMP_OUT" 2>&1
rc=$?; set -e
[ "$rc" -eq 0 ] && pass "default dry-run passes" || fail "default dry-run failed (rc=$rc)"
grep -q -- "--num_classes" "$TMP_OUT" && pass "default dry-run has --num_classes flag" || fail "missing --num_classes flag"
grep -qF "20" <(grep -A1 -- '--num_classes' "$TMP_OUT") 2>/dev/null && pass "default --num_classes value is 20" || {
    # arguments may be on separate lines from multi-line %q output
    awk '/--num_classes/{found=1; next} found{print; found=0}' "$TMP_OUT" | grep -qF "20" && \
        pass "default --num_classes value is 20" || fail "default --num_classes value not 20"
}
grep -qF "1000" <(grep -A1 -- '--n_boot' "$TMP_OUT") 2>/dev/null && pass "default --n_boot 1000 in dry-run" || {
    awk '/--n_boot/{found=1; next} found{print; found=0}' "$TMP_OUT" | grep -qF "1000" && \
        pass "default --n_boot 1000 in dry-run" || fail "missing --n_boot 1000"
}

# --num-classes 8 explicit
set +e; bash "$RECOVERY" --dry-run --quiet \
    --before-checkpoint "$_before_cp" --after-checkpoint "$_after_cp" \
    --tokenized "$_tokenized" --out "$_out" \
    --num-classes 8 >"$TMP_OUT" 2>&1
rc=$?; set -e
[ "$rc" -eq 0 ] && pass "--num-classes 8 dry-run passes" || fail "--num-classes 8 dry-run failed (rc=$rc)"
grep -qF "8" <(grep -A1 -- '--num_classes' "$TMP_OUT") 2>/dev/null && pass "--num_classes=8 rendered" || {
    awk '/--num_classes/{found=1; next} found{print; found=0}' "$TMP_OUT" | grep -qF "8" && \
        pass "--num_classes=8 rendered" || fail "--num_classes=8 not found"
}

# --num-classes 0 rejected
set +e; bash "$RECOVERY" --dry-run --quiet \
    --before-checkpoint "$_before_cp" --after-checkpoint "$_after_cp" \
    --tokenized "$_tokenized" --out "$_out" \
    --num-classes 0 >"$TMP_OUT" 2>&1
rc=$?; set -e
[ "$rc" -eq 1 ] && pass "rejects --num-classes 0" || fail "--num-classes 0 should exit 1 (got $rc)"

# --num-classes abc rejected
set +e; bash "$RECOVERY" --dry-run --quiet \
    --before-checkpoint "$_before_cp" --after-checkpoint "$_after_cp" \
    --tokenized "$_tokenized" --out "$_out" \
    --num-classes abc >"$TMP_OUT" 2>&1
rc=$?; set -e
[ "$rc" -eq 1 ] && pass "rejects --num-classes abc" || fail "--num-classes abc should exit 1 (got $rc)"

# preflight helper under 250 LOC
_ploc=$(awk '!/^[[:space:]]*$/ && !/^[[:space:]]*#/' "$PREFLIGHT" | wc -l)
[ "$_ploc" -le 250 ] && pass "preflight helper $_ploc LOC ≤ 250" || fail "preflight helper $_ploc LOC > 250"

# recovery wrapper under 250 LOC
_rloc=$(awk '!/^[[:space:]]*$/ && !/^[[:space:]]*#/' "$RECOVERY" | wc -l)
[ "$_rloc" -le 250 ] && pass "recovery wrapper $_rloc LOC ≤ 250" || fail "recovery wrapper $_rloc LOC > 250"

rm -rf "$RECOVERY_TMP"
echo ""
echo "============================================"
echo "Results: $passes passed, $failures failed"
echo "============================================"
[ "$failures" -gt 0 ] && exit 1 || exit 0
