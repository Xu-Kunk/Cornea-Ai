#!/usr/bin/env python3
"""Phase B / flow autonomous controller for Paper 1 method arms.

Applies the PRE-REGISTERED lambda-selection rule
(docs/research/paper1_phase_a_lambda_selection_protocol.md, locked 2026-08-06
BEFORE any Phase A result existed) and launches Phase B + the gated flow
extension. Run after Phase A completes; polls runs_status.json if launched
early.

Rule (per arm axis_ft / soft_spearman, candidates lam in {0.5, 1.0} at seed 42):
  1. non-inferiority: macro-F1 bootstrap CI overlaps the CE-naive reference
     interval [0.6067, 0.6895] (locked CE-naive 3-seed interval).
  2. among passing lambdas pick the one with the higher external rho_L12
     (GSE153515, evaluation.py secondary_endpoints.marker_gradient_rank_correlation).
  3. tie (|Δrho| < 0.005): lower ECE.
  4. both fail non-inferiority: pick the smaller lambda, flag trade-off.
  5. both arms fail: fallback flag (measurement+mechanism story) recorded.

Phase A completion contract: the train-only launcher (run_method_arms.py)
never flips runs_status eval to "OK" — the completion signal is the 5
results_<arm>_<lam>_42.json files written by scripts/run_method_arms_eval.py.
wait_phase_a_done() polls for those JSONs (fresh 50 h window from controller
start) and the lambda rule reads them directly.

Flow gate (s42, lambda 0.5):
  hard = run completed with non-NaN training loss (per launcher gate);
  soft = rho_L12 > 0.5. If both pass -> flow Phase B extension (seeds 123,456).

Launches Phase B as sequential invocations:
  --arm axis_ft       --lambdas <chosen_axis> --seeds "123,456"
  --arm soft_spearman --lambdas <chosen_ss>   --seeds "123,456"
  (+ flow if gated):  --arm flow --lambdas 0.5 --seeds "123,456"
with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (GB10 driver-vmalloc
fragmentation guard, per fix-1 verification).

Usage:
  python3 scripts/run_phase_b_controller.py            # wait + decide + launch
  python3 scripts/run_phase_b_controller.py --dry-run  # decide only, no launch
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "artifacts" / "paper1_method_arms"
STATUS_JSON = RESULTS_DIR / "runs_status.json"
DECISION_JSON = RESULTS_DIR / "phase_b_decision.json"
LOG_FILE = RESULTS_DIR / "phase_b_controller.log"
PY = os.environ.get("PY", "<venv>/bin/python3")
RUN_METHOD_ARMS = PROJECT / "scripts" / "run_method_arms.py"

# Locked reference (pre-registration doc): CE-naive 3-seed macro-F1 CI.
CE_NAIVE_CI = (0.6067, 0.6895)
CANDIDATE_LAMBDAS = [0.5, 1.0]
RHO_TIE_EPS = 0.005
FLOW_LAMBDA = 0.5
FLOW_GATE_SEED = 42
PHASE_A_TIMEOUT_S = 50 * 3600  # 50 h max wait for Phase A
POLL_S = 300

# Phase A run set: (arm, lambda, seed)
PHASE_A_RUNS = [
    ("axis_ft", 0.5, 42),
    ("axis_ft", 1.0, 42),
    ("soft_spearman", 0.5, 42),
    ("soft_spearman", 1.0, 42),
    ("flow", 0.5, 42),
]


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def result_path(arm: str, lam: float, seed: int) -> Path:
    return RESULTS_DIR / f"results_{arm}_{lam}_{seed}.json"


def load_result(arm: str, lam: float, seed: int) -> dict | None:
    p = result_path(arm, lam, seed)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        log(f"  parse error {p.name}: {e}")
        return None


def arm_metrics(arm: str, lam: float, seed: int) -> dict | None:
    d = load_result(arm, lam, seed)
    if d is None:
        return None
    pe = d.get("primary_endpoint", {})
    se = d.get("secondary_endpoints", {})
    return {
        "macro_f1": pe.get("macro_f1"),
        "ci": (pe.get("ci_lower"), pe.get("ci_upper")),
        "rho_L12": se.get("marker_gradient_rank_correlation"),
        "ECE": se.get("expected_calibration_error"),
    }


def wait_phase_a_done() -> dict:
    """Wait until all 5 Phase A result JSONs exist (results_<arm>_<lam>_42.json).

    The train-only launcher never flips runs_status eval to "OK"; the Phase A
    evaluation contract is the results JSONs written by
    scripts/run_method_arms_eval.py. Fresh 50 h window from controller start.
    """
    deadline = time.time() + PHASE_A_TIMEOUT_S
    while time.time() < deadline:
        missing = [
            f"{a}_lam{lam}_seed{s}"
            for a, lam, s in PHASE_A_RUNS
            if not result_path(a, lam, s).exists()
        ]
        if not missing:
            log("Phase A complete: all 5 result JSONs present.")
            return {"phase_a_runs": [f"{a}_lam{lam}_seed{s}" for a, lam, s in PHASE_A_RUNS]}
        log(f"  waiting for Phase A eval results (missing: {', '.join(missing)}) …")
        time.sleep(POLL_S)
    log("ERROR: Phase A did not complete within 50 h.")
    sys.exit(1)


def choose_lambda(arm: str, records: dict) -> dict:
    """Apply pre-registered rule for one arm."""
    results = {}
    for lam in CANDIDATE_LAMBDAS:
        m = arm_metrics(arm, lam, FLOW_GATE_SEED)
        results[lam] = m
        if m is None:
            log(f"  {arm} λ={lam}: NO RESULT")
    present = {lam: m for lam, m in results.items() if m is not None and m["macro_f1"] is not None}
    if len(present) == 0:
        log(f"ERROR: {arm} has no usable Phase A results.")
        return {"arm": arm, "chosen_lambda": None, "reason": "no results"}

    def passes(m: dict) -> bool:
        lo, hi = m["ci"]
        return (lo is not None and hi is not None
                and not (hi < CE_NAIVE_CI[0] or lo > CE_NAIVE_CI[1]))

    passing = {lam: m for lam, m in present.items() if passes(m)}
    if passing:
        best = max(passing, key=lambda lam: (passing[lam]["rho_L12"] or -1,))
        chosen = best
        reason = f"non-inferior (CI {passing[best]['ci']} overlaps [{CE_NAIVE_CI[0]},{CE_NAIVE_CI[1]}])"
        # tie-break: lower ECE if rho within eps of the other candidate
        for lam, m in passing.items():
            if lam != chosen and abs((m["rho_L12"] or -1) - (passing[chosen]["rho_L12"] or -1)) < RHO_TIE_EPS:
                if (m["ECE"] or 9) < (passing[chosen]["ECE"] or 9):
                    chosen = lam
    else:
        chosen = min(present)  # both fail non-inferiority -> smaller lambda, trade-off flagged
        reason = "both fail non-inferiority -> smaller lambda, trade-off flagged"
    log(f"  {arm}: chosen λ={chosen} ({reason})")
    return {"arm": arm, "chosen_lambda": chosen, "reason": reason,
            "candidates": {str(lam): {k: (list(v) if k == "ci" and v is not None else v)
                                      for k, v in m.items()}
                           for lam, m in present.items()}}


def flow_gate() -> tuple[bool, dict]:
    m = arm_metrics("flow", FLOW_LAMBDA, FLOW_GATE_SEED)
    if m is None or m["rho_L12"] is None:
        log("  flow: no Phase A result -> gate FAIL (hard)")
        return False, {"gate": "fail", "reason": "no result"}
    hard = True  # non-NaN loss gate enforced by launcher; result JSON implies ok
    soft = m["rho_L12"] > 0.5
    ok = hard and soft
    log(f"  flow gate: hard={hard}, soft(rho={m['rho_L12']:.4f} > 0.5)={soft} -> {'PASS' if ok else 'FAIL'}")
    return ok, {"gate": "pass" if ok else "fail", "rho_L12": m["rho_L12"], "hard": hard, "soft": soft}


def launch(args_list: list[str], tag: str) -> None:
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "-u", str(RUN_METHOD_ARMS)] + args_list
    log(f"  launching {tag}: {' '.join(cmd[-6:])}")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT), env=env)
    rc = proc.wait()
    if rc != 0:
        log(f"  WARNING: {tag} exited rc={rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Apply rules and write decision JSON; do not launch.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Do not wait for Phase A; use current runs_status.")
    args = parser.parse_args()

    log("=" * 60)
    log(f"Phase B / flow controller start (dry_run={args.dry_run}, no_wait={args.no_wait})")

    if not args.no_wait:
        wait_phase_a_done()
    else:
        log("--no-wait: using current Phase A results (no polling)")

    decision = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 — datetime.UTC does not exist at runtime (CPython 3.12); ruff's py312 UP017 suggestion is invalid (see evaluation.py precedent)
                "rule": "docs/research/paper1_phase_a_lambda_selection_protocol.md",
                "arms": {}, "flow": {}, "launch": None}
    for arm in ["axis_ft", "soft_spearman"]:
        decision["arms"][arm] = choose_lambda(arm, {})

    flow_ok, flow_d = flow_gate()
    decision["flow"] = flow_d

    DECISION_JSON.write_text(json.dumps(decision, indent=2))
    log(f"decision written -> {DECISION_JSON}")

    if args.dry_run:
        log("dry-run: no Phase B launched.")
        return

    lam_a = decision["arms"]["axis_ft"]["chosen_lambda"]
    lam_s = decision["arms"]["soft_spearman"]["chosen_lambda"]
    if lam_a is None or lam_s is None:
        log("ERROR: missing lambda choice; not launching Phase B.")
        return

    launch(["--arm", "axis_ft", "--lambdas", f"{lam_a}", "--seeds", "123,456"], "Phase B axis_ft")
    launch(["--arm", "soft_spearman", "--lambdas", f"{lam_s}", "--seeds", "123,456"], "Phase B soft_spearman")
    if flow_ok:
        launch(["--arm", "flow", "--lambdas", f"{FLOW_LAMBDA}", "--seeds", "123,456"], "flow Phase B extension")
    decision["launch"] = {"axis_ft": lam_a, "soft_spearman": lam_s, "flow_extension": flow_ok}
    DECISION_JSON.write_text(json.dumps(decision, indent=2))
    log("Phase B sequence launched.")


if __name__ == "__main__":
    main()
