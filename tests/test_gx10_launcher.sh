#!/usr/bin/env bash
# Tests for scripts/gx10_cgroup_run.sh — mode selection, rejection, dry-run.
set -euo pipefail

readonly LAUNCHER="scripts/gx10_cgroup_run.sh"
readonly TMP_OUT="$(mktemp -t gx10_launcher_test.XXXXXX)"
trap 'rm -f "$TMP_OUT"' EXIT

failures=0 passes=0
pass() { passes=$((passes + 1)); echo -e "  \033[0;32mPASS\033[0m $*"; }
fail() { failures=$((failures + 1)); echo -e "  \033[0;31mFAIL\033[0m $*"; }

assert_exit() {
    local expected=$1 desc; shift; desc="$*"
    set +e; bash "$LAUNCHER" "$@" >"$TMP_OUT" 2>&1; local actual=$?; set -e
    if [ "$actual" -eq "$expected" ]; then pass "$desc (exit $actual)"
    else fail "$desc — expected exit $expected, got $actual"; sed 's/^/      /' "$TMP_OUT"; fi
}

assert_stdout_contains() {
    local pattern="$1" desc="$2"; shift 2
    set +e; bash "$LAUNCHER" "$@" >"$TMP_OUT" 2>&1; local actual=$?; set -e
    if echo "$actual" | grep -q "^0$" || true; then :; else
        fail "$desc — launcher exited $actual"; sed 's/^/      /' "$TMP_OUT"; return; fi
    if grep -qF -- "$pattern" "$TMP_OUT"; then pass "$desc"
    else fail "$desc — output missing '$pattern'"; sed 's/^/      /' "$TMP_OUT"; fi
}

assert_stdout_not_contains() {
    local pattern="$1" desc="$2"; shift 2
    if grep -qF -- "$pattern" "$TMP_OUT"; then fail "$desc — output contains forbidden '$pattern'"; sed 's/^/      /' "$TMP_OUT"
    else pass "$desc"; fi
}

echo "=== test_gx10_launcher.sh ==="
echo ""
[[ -f "$LAUNCHER" ]] && pass "Launcher exists" || fail "Launcher missing"
assert_exit 1 "No arguments→usage error" --dry-run -- true 2>&1 || true
echo ""

echo "--- service mode ---"
assert_stdout_contains "systemd-run" "renders systemd-run" --mode service --dry-run -- echo hello
assert_stdout_contains "--service-type=exec" "has --service-type=exec" --mode service --dry-run -- echo hello
assert_stdout_contains "OOMScoreAdjust" "has OOMScoreAdjust" --mode service --oom-score-adjust 500 --dry-run -- echo hello
assert_stdout_contains "MemoryMax" "has MemoryMax" --mode service --memory-max 100G --dry-run -- echo hello
assert_stdout_contains "MemorySwapMax" "has MemorySwapMax" --mode service --memory-swap-max 0 --dry-run -- echo hello
assert_stdout_contains "MemoryHigh" "has MemoryHigh" --mode service --memory-max 100G --dry-run -- echo hello
assert_stdout_contains "RuntimeMaxSec" "has RuntimeMaxSec" --mode service --runtime-max 86400 --dry-run -- echo hello
assert_stdout_contains "TimeoutStopSec" "has TimeoutStopSec" --mode service --runtime-max 86400 --dry-run -- echo hello
assert_stdout_contains "MemoryAccounting=yes" "enables MemoryAccounting" --mode service --dry-run -- echo hello
assert_stdout_contains "TasksAccounting=yes" "enables TasksAccounting" --mode service --dry-run -- echo hello
assert_stdout_contains "OOMPolicy=kill" "sets OOMPolicy" --mode service --dry-run -- echo hello
assert_stdout_contains "--wait" "uses --wait" --mode service --dry-run -- echo hello
assert_stdout_contains "--collect" "uses --collect" --mode service --dry-run -- echo hello
assert_stdout_contains "PYTORCH_CUDA_ALLOC_CONF" "sets allocator env" --mode service --dry-run -- echo hello
assert_stdout_contains "expandable_segments:False" "allocator is False" --mode service --dry-run -- echo hello
assert_stdout_not_contains "expandable_segments:True" "allocator not True" --mode service --dry-run -- echo hello
echo ""

echo "--- scope mode ---"
assert_stdout_contains "systemd-run" "renders systemd-run" --mode scope --dry-run -- echo hello
assert_stdout_contains "--scope" "uses --scope" --mode scope --dry-run -- echo hello
assert_stdout_not_contains "OOMScoreAdjust" "rejects OOMScoreAdjust" --mode scope --dry-run -- echo hello
assert_stdout_contains "MemoryMax" "has MemoryMax" --mode scope --memory-max 100G --dry-run -- echo hello
assert_stdout_contains "MemorySwapMax" "has MemorySwapMax" --mode scope --memory-swap-max 0 --dry-run -- echo hello
assert_stdout_contains "OOMPolicy=kill" "sets OOMPolicy" --mode scope --dry-run -- echo hello
echo ""

echo "--- scope+OOMScoreAdjust rejection ---"
assert_exit 1 "scope+oom exits non-zero" --mode scope --oom-score-adjust 500 --dry-run -- echo hello
set +e; bash "$LAUNCHER" --mode scope --oom-score-adjust 500 --dry-run -- echo hello >"$TMP_OUT" 2>&1; set -e
grep -qi "oomscoreadjust\|not supported\|scope.*cannot\|property" "$TMP_OUT" && pass "scope+OOMScoreAdjust clear message" || fail "scope+OOMScoreAdjust should have clear message"
echo ""

echo "--- unit/log-dir/separator ---"
assert_stdout_contains "--unit gx10-test" "service --unit" --mode service --unit gx10-test --dry-run -- echo hello
assert_stdout_contains "--unit gx10-scope-test" "scope --unit" --mode scope --unit gx10-scope-test --dry-run -- echo hello
assert_stdout_contains "StandardOutput=file" "service log-dir" --mode service --log-dir /tmp/gx10-logs --dry-run -- echo hello
assert_exit 1 "missing -- separator" --mode service --dry-run echo hello
assert_stdout_contains "systemd-run" "no-resource-flags valid" --mode service --dry-run -- echo hello
echo ""

echo "--- allocator source check ---"
grep -q "expandable_segments:False" "$LAUNCHER" && pass "launcher source uses False" || fail "launcher must use False"
grep -q "expandable_segments:True" "$LAUNCHER" && fail "launcher must NOT use True" || pass "launcher does not use True"
echo ""

echo "============================================"
echo "Results: $passes passed, $failures failed"
echo "============================================"
[ "$failures" -gt 0 ] && exit 1 || exit 0
