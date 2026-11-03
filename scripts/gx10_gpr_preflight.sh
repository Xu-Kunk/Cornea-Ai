# gx10_gpr_preflight.sh — Sourced preflight gates for GPR recovery.
# Must be sourced from gx10_gpr_recovery.sh; expects QUIET, BEFORE_CHECKPOINT,
# AFTER_CHECKPOINT, TOKENIZED_PATH, PINNED_PYTHON to be set by the caller.
set -euo pipefail

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
warn() { if [[ "$QUIET" == false ]]; then echo "PREFLIGHT WARN: $*" >&2; fi; true; }
ok()   { if [[ "$QUIET" == false ]]; then echo "PREFLIGHT OK:   $*"; fi; true; }

_run_preflight_all() {
    local before="$1" after="$2" tokenized="$3" python="$4"

    _gate_no_020334 "$before" "$after"
    _gate_caft_done "$after"
    _gate_no_legacy_pca "$before"
    _gate_tokenized "$tokenized"
    _gate_before_checkpoint "$before"
    _gate_after_checkpoint "$after"
    _gate_python_pin "$python"
    _gate_no_vllm
    _gate_no_gpu_compute
    _gate_cuda
    _gate_memory
    _gate_cgroup_v2
}

_gate_no_020334() {
    local before="$1" after="$2"
    for path in "$before" "$after"; do
        if [[ "$path" == *"020334"* ]]; then
            fail "Checkpoint path contains '020334' — this is a known stale/incomplete run. Rejected."
        fi
    done
    ok "No stale 020334 in checkpoint paths"
}

_gate_caft_done() {
    local after="$1" parent="$1"
    local found=false
    while [[ "$parent" != "/" && "$parent" != "." ]]; do
        if [[ -f "$parent/CAFT_DONE" ]]; then
            found=true
            ok "CAFT_DONE found at $parent/CAFT_DONE"
            break
        fi
        parent="$(dirname "$parent")"
    done
    if [[ "$found" == false ]]; then
        fail "CAFT_DONE not found in any parent of $after — training incomplete or directory mis-specified"
    fi
}

_gate_no_legacy_pca() {
    local before="$1"
    if [[ -f "$before/cs_before_results.json" ]]; then
        local content
        content="$(head -c 200 "$before/cs_before_results.json" 2>/dev/null || true)"
        if echo "$content" | grep -qi '"embedding".*"PCA"'; then
            fail "$before/cs_before_results.json is a legacy PCA artifact; use paired checkpoint comparison instead"
        fi
    fi
    ok "No legacy PCA artifact detected"
}

_gate_tokenized() {
    local tokenized="$1"
    if [[ ! -d "$tokenized" ]]; then
        fail "Tokenized dataset $tokenized does not exist or is not a directory"
    fi
    local has_dataset=false
    [[ -f "$tokenized/dataset_info.json" ]] && has_dataset=true
    if ls "$tokenized"/data-*.arrow &>/dev/null 2>&1; then has_dataset=true; fi
    if [[ "$has_dataset" == false ]]; then
        fail "Tokenized dataset $tokenized lacks dataset_info.json or data-*.arrow files"
    fi
    ok "Tokenized dataset found and valid"
}

_gate_before_checkpoint() {
    local before="$1"
    [[ -d "$before" ]] || fail "Before checkpoint $before does not exist"
    if ! ls "$before"/model.safetensors "$before"/config.json &>/dev/null 2>&1; then
        fail "Before checkpoint $before lacks model.safetensors or config.json"
    fi
    ok "Before checkpoint exists with model files"
}

_gate_after_checkpoint() {
    local after="$1"
    [[ -d "$after" ]] || fail "After checkpoint $after does not exist"
    if ! ls "$after"/model.safetensors "$after"/config.json &>/dev/null 2>&1; then
        fail "After checkpoint $after lacks model.safetensors or config.json"
    fi
    ok "After checkpoint exists with model files"
}

_gate_python_pin() {
    local python="$1"
    if [[ ! -x "$python" ]]; then
        fail "Python interpreter $python not found or not executable"
    fi
    ok "Python interpreter $python found and executable"
}

_gate_no_vllm() {
    if pgrep -f "vllm\|vllm.entrypoints\|llama\|aphrodite" >/dev/null 2>&1; then
        fail "Active vLLM or LLM inference process detected — stop before running GPR"
    fi
    ok "No active vLLM/compute process (pgrep)"
}

_gate_no_gpu_compute() {
    local compute
    compute="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^$' || true)"
    compute="$(echo "$compute" | grep -v '^pid,' || true)"
    compute="$(echo "$compute" | tr -d ' ')"
    if [[ -n "$compute" ]]; then
        fail "Active GPU compute processes detected (nvidia-smi --query-compute-apps): $compute"
    fi
    ok "No active GPU compute processes (nvidia-smi)"
}

_gate_cuda() {
    if ! command -v nvidia-smi &>/dev/null; then
        fail "nvidia-smi not found — CUDA not available"
    fi
    if ! nvidia-smi &>/dev/null 2>&1; then
        fail "nvidia-smi returned error — CUDA may not be available"
    fi
    ok "CUDA available (nvidia-smi ok)"
}

_gate_memory() {
    local kb gib
    kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    gib="$(( kb / 1024 / 1024 ))"
    if [[ "$gib" -lt 20 ]]; then
        fail "MemAvailable is ${gib}GiB; need ≥20 GiB for GPR model loading (unified memory)"
    fi
    ok "MemAvailable ${gib}GiB (>=20GiB)"
}

_gate_cgroup_v2() {
    local fs
    fs="$(stat -fc%T /sys/fs/cgroup 2>/dev/null || echo unknown)"
    if [[ "$fs" != "cgroup2fs" ]]; then
        local cg_type
        cg_type="$(grep cgroup /proc/filesystems 2>/dev/null | head -1 || echo unknown)"
        if [[ "$cg_type" != *"cgroup2"* ]]; then
            fail "cgroup v2 (unified hierarchy) not detected; required for GX10 memory protection"
        fi
    fi
    ok "cgroup v2 available"
}
