#!/usr/bin/env python
"""Run the this study method-arm grid: axis_ft / soft_spearman / flow x lambdas x
seeds, then aggregate Phase A results.

Phase A (feasibility, seed 42):
    axis_ft       {0.5, 1.0}  s42
    soft_spearman {0.5, 1.0}  s42
    flow          {0.5}       s42   (feasibility-gated arm)

Phase B (extension to seeds 123, 456 at the caller-chosen lambda):
    --arm axis_ft --lambdas "<chosen>" --seeds "123,456"
    --arm soft_spearman --lambdas "<chosen>" --seeds "123,456"

The lambda SELECTION step is intentionally OUT of this script — the
orchestrator/user picks the lambda from the Phase A results
(--summary -> artifacts/paper1_method_arms/phase_a_summary.csv). The flow
arm is additionally feasibility-GATED: Phase B flow runs are refused unless
the Phase A flow s42 run satisfies the stability rule (training loss history
has no NaN and, once eval results exist, axis conservation rho_L12 > 0.5 at
lambda_f = 0.5).

Usage:
    python scripts/run_method_arms.py                        # Phase A (all arms)
    python scripts/run_method_arms.py --summary              # aggregate Phase A
    python scripts/run_method_arms.py --arm axis_ft --lambdas "0.5,1.0" --seeds "42"
    python scripts/run_method_arms.py --arm flow --lambdas "0.5" --seeds "123,456"
    python scripts/run_method_arms.py --arm soft_spearman --lambdas "1.0" --seeds "42,123,456"

Sequential on the single GB10 GPU (mirrors run_ordinal_grid.py). Each run
streams stdout+stderr to artifacts/paper1_method_arms/logs/<arm>_lam<λ>_seed<s>.log.
Success is confirmed by the METHOD_ARM_DONE marker in the run's save dir
(save/method_<arm>_lam<λ>_seed<s>_<ts>/); failed runs are logged and the grid
CONTINUES. A runs_status.json table is written to the artifacts dir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

# ── Environment (must be set before any torch import) ────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "artifacts" / "paper1_method_arms"
LOG_DIR = RESULTS_DIR / "logs"
STATUS_JSON = RESULTS_DIR / "runs_status.json"
SUMMARY_CSV = RESULTS_DIR / "phase_a_summary.csv"

ARMS = ("axis_ft", "soft_spearman", "flow")
# Phase A defaults (feasibility): per-arm lambda grid x seed 42.
DEFAULT_PHASE_A_LAMBDAS = {"axis_ft": [0.5, 1.0], "soft_spearman": [0.5, 1.0], "flow": [0.5]}
DEFAULT_PHASE_A_SEEDS = [42]
# Flow stability gate (Phase A seed 42 -> extension rule).
FLOW_GATE_LAMBDA = 0.5
FLOW_GATE_SEED = 42
FLOW_GATE_RHO = 0.5

EPOCHS = 3
DATA_FILE = "corneal_100k_donor_split.h5ad"
SPLIT_MANIFEST = "artifacts/paper1_fair_split/manifest.json"
PY = os.environ.get("PY", "<venv>/bin/python3")
RUN_TIMEOUT_S = 172800  # 48 h per run

# Robust <compute-node> guards (same as run_ordinal_grid.py): never start a run hot or
# memory-pressured on the GB10's shared 128 GB CPU/GPU memory.
MIN_AVAIL_GB = 40.0
COOL_TEMP_C = 80
TEMP_WAIT_MAX_S = 5400


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def gpu_temp_c() -> int | None:
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    temps = [int(line.strip()) for line in res.stdout.strip().split("\n") if line.strip()]
    return max(temps) if temps else None


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().split("\n"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    return 0.0


def wait_until_ready(arm: str, lam: float, seed: int) -> None:
    t0 = time.time()
    while True:
        temp = gpu_temp_c()
        avail = mem_available_gb()
        if temp is not None and temp < COOL_TEMP_C and avail >= MIN_AVAIL_GB:
            print(f"{_ts()} {arm} λ={lam} seed={seed}  ready (GPU {temp} °C, "
                  f"{avail:.0f} GiB avail)\n", flush=True)
            return
        if time.time() - t0 > TEMP_WAIT_MAX_S:
            print(f"ERROR: {arm} λ={lam} seed={seed}  not ready after "
                  f"{TEMP_WAIT_MAX_S // 60} min (GPU {temp} °C, {avail:.0f} GiB avail); "
                  f"aborting grid.", flush=True)
            sys.exit(1)
        reason = []
        if temp is not None and temp >= COOL_TEMP_C:
            reason.append(f"GPU {temp} °C")
        if avail < MIN_AVAIL_GB:
            reason.append(f"{avail:.0f} GiB avail")
        print(f"{_ts()} {arm} λ={lam} seed={seed}  waiting ({', '.join(reason)})…", flush=True)
        time.sleep(60)


# ── Pre-flight checks (mirror run_ordinal_grid.py) ────────────────────────


def preflight() -> None:
    print(f"{_ts()} pre-flight checks …", flush=True)

    swap = Path("/proc/swaps").read_text().strip().split("\n")
    if len(swap) > 1:
        print(f"ERROR: swap active ({len(swap) - 1} entries); aborting.", flush=True)
        sys.exit(1)

    res = subprocess.run(["systemctl", "is-active", "vllm.service"],
                         capture_output=True, text=True, timeout=10)
    if res.stdout.strip() == "active":
        print("ERROR: vllm.service is active; aborting.", flush=True)
        sys.exit(1)
    res_docker = subprocess.run(
        ["docker", "ps", "--filter", "name=vllm", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=10,
    )
    if res_docker.stdout.strip():
        print(f"ERROR: vLLM docker container running ({res_docker.stdout.strip()}); aborting.",
              flush=True)
        sys.exit(1)

    free_gb = shutil.disk_usage(PROJECT).free / (1024 ** 3)
    if free_gb < 50:
        print(f"ERROR: only {free_gb:.1f} GB free (< 50 GB); aborting.", flush=True)
        sys.exit(1)

    t0 = time.time()
    while True:
        temp = gpu_temp_c()
        if temp is None:
            print("ERROR: cannot read GPU temperature; aborting.", flush=True)
            sys.exit(1)
        if temp < COOL_TEMP_C:
            print(f"  ✓ GPU temp: {temp} °C", flush=True)
            break
        if time.time() - t0 > TEMP_WAIT_MAX_S:
            print(f"ERROR: GPU still {temp} °C (>= {COOL_TEMP_C} °C) after "
                  f"{TEMP_WAIT_MAX_S // 60} min of waiting; aborting.", flush=True)
            sys.exit(1)
        print(f"  … GPU {temp} °C >= {COOL_TEMP_C} °C, cooling… ", flush=True)
        time.sleep(60)

    print(f"{_ts()} all pre-flight checks passed.\n", flush=True)


# ── Directory helpers ────────────────────────────────────────────────────


def find_save_dir(arm: str, lam: float, seed: int) -> Path | None:
    dirs = sorted(glob(str(PROJECT / "save" / f"method_{arm}_lam{lam}_seed{seed}_*")))
    return Path(dirs[-1]) if dirs else None


def is_done(arm: str, lam: float, seed: int) -> bool:
    save_dir = find_save_dir(arm, lam, seed)
    return save_dir is not None and (save_dir / "METHOD_ARM_DONE").exists()


# ── Flow feasibility gate ─────────────────────────────────────────────────


def flow_gate_pass() -> tuple[bool, str]:
    """Phase A -> Phase B stability rule for the exploratory flow arm.

    The gate PASSES only when the Phase A flow run (lambda_f = 0.5, seed 42)
    (a) completed (METHOD_ARM_DONE), (b) has no NaN training loss, and
    (c) IF an eval results JSON exists (results_flow_0.5_42.json, written by
    the future eval script mirroring run_ordinal_eval.py) preserves the axis
    (marker-gradient rho at layer 12 > 0.5). When eval results are not yet
    present the gate passes on (a)+(b) with an explicit warning that the rho
    check is deferred to eval time.
    """
    save_dir = find_save_dir("flow", FLOW_GATE_LAMBDA, FLOW_GATE_SEED)
    if save_dir is None or not (save_dir / "METHOD_ARM_DONE").exists():
        return False, (
            f"flow Phase A run λ={FLOW_GATE_LAMBDA} s{FLOW_GATE_SEED} not done "
            f"(no METHOD_ARM_DONE) — run Phase A first"
        )
    # (b) NaN check on the training loss history
    args_json = save_dir / "method_arm_args.json"
    losses: list[float] = []
    if args_json.is_file():
        try:
            history = json.loads(args_json.read_text()).get("log_history_tail", [])
            losses = [
                float(h.get("loss", math.nan))
                for h in history
                if isinstance(h, dict) and "loss" in h
            ]
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            losses = []
    no_nan = bool(losses) and all(math.isfinite(v) for v in losses)
    if not no_nan:
        return False, (
            f"flow Phase A λ={FLOW_GATE_LAMBDA} s{FLOW_GATE_SEED} loss history has "
            f"NaN or is empty ({losses}) — stability rule FAILED"
        )
    # (c) rho preservation (only when eval results exist)
    res_json = RESULTS_DIR / f"results_flow_{FLOW_GATE_LAMBDA}_{FLOW_GATE_SEED}.json"
    if res_json.is_file():
        try:
            res = json.loads(res_json.read_text())
            se = res.get("secondary_endpoints", {})
            rho = se.get("marker_gradient_rank_correlation")
            rho = rho if rho is not None else (res.get("layerwise", {}) or {}).get("marker_gradient_rho", [None])[-1]
        except (json.JSONDecodeError, OSError, IndexError, TypeError):
            rho = None
        if rho is None or not math.isfinite(float(rho)):
            return False, "flow eval results present but rho_L12 unreadable"
        if float(rho) <= FLOW_GATE_RHO:
            return False, (
                f"flow λ={FLOW_GATE_LAMBDA} s{FLOW_GATE_SEED} rho_L12={float(rho):.3f} "
                f"<= {FLOW_GATE_RHO} — axis not preserved, Phase B refused"
            )
        return True, f"loss NaN-free and rho_L12={float(rho):.3f} > {FLOW_GATE_RHO}"
    return True, (
        f"loss NaN-free (rho check deferred: no results_flow_{FLOW_GATE_LAMBDA}_"
        f"{FLOW_GATE_SEED}.json yet)"
    )


# ── Task runner ───────────────────────────────────────────────────────────


def run_arm_training(arm: str, lam: float, seed: int) -> Path | None:
    """Train one arm run; returns the save dir on success (METHOD_ARM_DONE)."""
    cmd = [
        PY, "-u",
        str(PROJECT / "scripts" / "method_arm_gf.py"),
        "--arm", arm,
        "--lambda-arm", str(lam),
        "--data_file", DATA_FILE,
        "--epochs", str(EPOCHS),
        "--seed", str(seed),
        "--strict-fair",
        "--split_manifest", SPLIT_MANIFEST,
    ]
    print(f"{_ts()} {arm} λ={lam} seed={seed}  training starting …", flush=True)
    log_path = LOG_DIR / f"{arm}_lam{lam}_seed{seed}.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S)
    wall = (time.time() - t0) / 60

    if proc.returncode != 0:
        print(f"{_ts()} {arm} λ={lam} seed={seed}  TRAIN FAILED (rc={proc.returncode}, "
              f"{wall:.1f} min); log: {log_path}", flush=True)
        try:
            tail = log_path.read_text()[-2000:]
        except OSError:
            tail = ""
        if tail:
            print(tail, flush=True)
        return None

    save_dir = find_save_dir(arm, lam, seed)
    if save_dir is None or not (save_dir / "METHOD_ARM_DONE").exists():
        print(f"{_ts()} {arm} λ={lam} seed={seed}  METHOD_ARM_DONE marker missing; "
              f"incomplete.", flush=True)
        return None
    print(f"{_ts()} {arm} λ={lam} seed={seed}  TRAINED ({wall:.1f} min)  → {save_dir.name}",
          flush=True)
    return save_dir


# ── Phase A summary ───────────────────────────────────────────────────────


def build_phase_a_summary() -> Path:
    """Aggregate Phase A results into phase_a_summary.csv.

    Columns: arm, lambda, seed, macro_f1, rho_L12, ECE, axis_conservation,
    status. Metric columns are read from the eval results JSON
    (results_<arm>_<lam>_<seed>.json, the convention the future eval script
    mirrors from run_ordinal_eval.py); runs without eval results carry NaN
    metrics and their training status from runs_status.json.
    """
    import pandas as pd

    status: dict[tuple[str, float, int], str] = {}
    if STATUS_JSON.is_file():
        try:
            for r in json.loads(STATUS_JSON.read_text()).get("runs", []):
                status[(str(r.get("arm")), float(r.get("lambda")), int(r.get("seed")))] = (
                    f"train:{r.get('train')} eval:{r.get('eval')}"
                )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            status = {}

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for lam in DEFAULT_PHASE_A_LAMBDAS[arm]:
            for seed in DEFAULT_PHASE_A_SEEDS:
                row: dict[str, Any] = {
                    "arm": arm, "lambda": lam, "seed": seed,
                    "macro_f1": math.nan, "rho_L12": math.nan,
                    "ECE": math.nan, "axis_conservation": math.nan,
                    "status": status.get((arm, lam, seed), "no_run"),
                }
                res = RESULTS_DIR / f"results_{arm}_{lam}_{seed}.json"
                if res.is_file():
                    try:
                        r = json.loads(res.read_text())
                        pe = r.get("primary_endpoint", {})
                        se = r.get("secondary_endpoints", {})
                        row["macro_f1"] = pe.get("macro_f1", math.nan)
                        row["rho_L12"] = se.get("marker_gradient_rank_correlation", math.nan)
                        row["ECE"] = se.get("expected_calibration_error", math.nan)
                        row["axis_conservation"] = row["rho_L12"]
                        row["status"] = "evaluated"
                    except (json.JSONDecodeError, OSError, TypeError):
                        row["status"] = "eval_json_unreadable"
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n{_ts()} summary -> {SUMMARY_CSV}")
    print(df.to_string(index=False))
    return SUMMARY_CSV


# ── Main ──────────────────────────────────────────────────────────────────


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=None,
                        help="Arms to run (repeatable or comma-separated); "
                             f"default: all of {ARMS} (Phase A).")
    parser.add_argument("--lambdas", default=None,
                        help="Comma-separated lambda grid (default per arm: "
                             f"{DEFAULT_PHASE_A_LAMBDAS}).")
    parser.add_argument("--seeds", default=None,
                        help=f"Comma-separated seeds (default Phase A: {DEFAULT_PHASE_A_SEEDS}; "
                             f"Phase B: 123,456).")
    parser.add_argument("--summary", action="store_true",
                        help="Aggregate Phase A results into phase_a_summary.csv and exit.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip GPU/swap/disk pre-flight checks (CI/testing).")
    args = parser.parse_args()

    if args.summary:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        build_phase_a_summary()
        return

    if not args.skip_preflight:
        preflight()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    arms = _parse_csv(",".join(args.arm) if args.arm else None, list(ARMS))
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print(f"ERROR: unknown arm(s) {bad}; valid: {ARMS}", flush=True)
        sys.exit(1)
    seeds = [int(s) for s in _parse_csv(args.seeds, [str(s) for s in DEFAULT_PHASE_A_SEEDS])]

    # Flow feasibility gate: Phase B (any seed beyond the Phase A seed 42)
    # requires the Phase A stability rule to pass.
    phase_b_flow = "flow" in arms and set(seeds) - set(DEFAULT_PHASE_A_SEEDS)
    if phase_b_flow:
        gate_ok, reason = flow_gate_pass()
        if not gate_ok:
            print(f"{_ts()} FLOW GATE REFUSED: {reason}", flush=True)
            print(f"{_ts()} flow Phase B extension cancelled (other arms continue).", flush=True)
            arms = [a for a in arms if a != "flow"]
            if not arms:
                sys.exit(1)
        else:
            print(f"{_ts()} FLOW GATE PASSED: {reason}", flush=True)

    runs: list[tuple[str, float, int]] = []
    for arm in arms:
        lambdas = [float(v) for v in _parse_csv(args.lambdas, [str(x) for x in DEFAULT_PHASE_A_LAMBDAS[arm]])]
        runs.extend((arm, lam, seed) for lam in lambdas for seed in seeds)
    print(f"{_ts()} grid: {len(runs)} runs -> {runs}", flush=True)

    status: list[dict[str, Any]] = []
    for arm, lam, seed in runs:
        trained = is_done(arm, lam, seed)
        if trained:
            print(f"{_ts()} {arm} λ={lam} seed={seed}  already trained "
                  f"(METHOD_ARM_DONE) → skip\n", flush=True)
        else:
            wait_until_ready(arm, lam, seed)
            save_dir = run_arm_training(arm, lam, seed)
            trained = save_dir is not None
            if not trained:
                status.append({"arm": arm, "lambda": lam, "seed": seed,
                               "train": "FAIL", "eval": "skipped", "save_dir": None})
                continue
        status.append({
            "arm": arm,
            "lambda": lam,
            "seed": seed,
            "train": "OK" if trained else "FAIL",
            "eval": "pending",
            "save_dir": str(find_save_dir(arm, lam, seed)) if trained else None,
            "updated_utc": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    with open(STATUS_JSON, "w") as f:
        json.dump({
            "grid": [f"{a}_lam{lam}_seed{s}" for a, lam, s in runs],
            "mode": "train",
            "runs": status,
        }, f, indent=2)
    print(f"\n{_ts()} status -> {STATUS_JSON}")

    n_ok = sum(1 for s in status if s["train"] == "OK")
    n_total = len(status)
    print(f"\n{'=' * 60}")
    print(f"this study method arms: {n_ok}/{n_total} runs trained")
    print("=" * 60)
    if n_ok < n_total:
        print("Run complete but PARTIAL; relaunch to resume (idempotent).")
    print("After Phase A, pick lambdas and run: --summary  /  --arm <arm> "
          "--lambdas \"<chosen>\" --seeds \"123,456\"")


if __name__ == "__main__":
    main()
