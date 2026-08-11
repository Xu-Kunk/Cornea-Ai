#!/usr/bin/env python3
"""Option A retrain chain: wait EA done -> retrain AA-FT + SS s42 (new centroid_pca axis) -> eval -> summarize.

Approved plan (user 2026-08-10): axis re-fit to per-layer-centroid-PCA (evaluation
metric's training-side analog), retrain the two novel arms at the chosen lambdas
(axis_ft 1.0, soft_spearman 0.5) seed 42, evaluate on GSE153515, and report the
external rho. The 3-seed extension (seeds 123,456) is NOT auto-launched — the
s42 rho band verdict is logged for the orchestrator to decide (per the
user-approved gate: extend only if rho lands in the expected 0.5-0.7 band).

Run: nohup python3 scripts/run_option_a_chain.py > /dev/null 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "artifacts" / "paper1_method_arms"
CHAIN_LOG = RESULTS_DIR / "option_a_chain.log"
PY = os.environ.get("PY", "python3")
ENV = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
POLL_S = 300
EA_TIMEOUT_S = 12 * 3600

# Approved commands (fix-4, verified against the actual CLI)
RETRAINS = [
    ["scripts/method_arm_gf.py", "--arm", "axis_ft", "--lambda-arm", "1.0",
     "--data_file", "corneal_100k_donor_split.h5ad", "--epochs", "3", "--seed", "42",
     "--strict-fair", "--split_manifest", "artifacts/paper1_fair_split/manifest.json",
     "--axis-mode", "centroid_pca"],
    ["scripts/method_arm_gf.py", "--arm", "soft_spearman", "--lambda-arm", "0.5",
     "--data_file", "corneal_100k_donor_split.h5ad", "--epochs", "3", "--seed", "42",
     "--strict-fair", "--split_manifest", "artifacts/paper1_fair_split/manifest.json",
     "--axis-mode", "centroid_pca"],
]
EVAL_RUNS = "axis_ft:1.0:42,soft_spearman:0.5:42"
OLD_RESULTS = [
    RESULTS_DIR / "results_axis_ft_1.0_42.json",
    RESULTS_DIR / "results_soft_spearman_0.5_42.json",
]


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}"  # noqa: UP017 — datetime.UTC does not exist at runtime (CPython 3.12)
    print(line, flush=True)
    with open(CHAIN_LOG, "a") as f:
        f.write(line + "\n")


def wait_ea_done() -> None:
    t0 = time.time()
    while time.time() - t0 < EA_TIMEOUT_S:
        # EA done when ENERGY_ANCHOR_DONE exists OR the EA python process is gone
        ea_dirs = sorted(PROJECT.glob("save/energy_anchor_gf_lam0.5_seed42_*"))
        done = any((d / "ENERGY_ANCHOR_DONE").exists() for d in ea_dirs)
        proc = subprocess.run(["pgrep", "-f", "run_energy_anchor[.]py"],
                              capture_output=True, text=True)
        alive = proc.returncode == 0
        if done or not alive:
            log(f"EA finished (done={done}, process_alive={alive}).")
            time.sleep(60)  # CUDA-context grace
            return
        log(f"waiting for EA to finish... ({(time.time() - t0) / 3600:.1f}h)")
        time.sleep(POLL_S)
    log("ERROR: EA did not finish within 12 h; aborting.")
    raise SystemExit(1)


def run(cmd: list[str], tag: str) -> None:
    log(f"launching {tag} ...")
    rc = subprocess.run([PY, "-u"] + cmd, cwd=str(PROJECT), env=ENV).returncode
    log(f"{tag} finished rc={rc}")
    if rc != 0:
        raise SystemExit(f"{tag} failed rc={rc}")


def main() -> None:
    log("=" * 60)
    log("Option A chain start (centroid_pca axis, s42 retrain)")
    wait_ea_done()

    log("--- step 1: retrain axis_ft lambda=1.0 s42 (new axis) ---")
    run(RETRAINS[0], "axis_ft retrain")
    log("--- step 2: retrain soft_spearman lambda=0.5 s42 (new axis) ---")
    run(RETRAINS[1], "soft_spearman retrain")

    log("--- step 3: clear old results + eval ---")
    for p in OLD_RESULTS:
        p.unlink(missing_ok=True)
        log(f"removed {p.name}")
    run(["scripts/run_method_arms_eval.py", "--runs", EVAL_RUNS], "eval s42")

    log("--- step 4: summarize rho ---")
    summary = {}
    for arm, lam in [("axis_ft", 1.0), ("soft_spearman", 0.5)]:
        f = RESULTS_DIR / f"results_{arm}_{lam}_42.json"
        try:
            d = json.loads(f.read_text())
            se = d["secondary_endpoints"]
            pe = d["primary_endpoint"]
            summary[f"{arm}_{lam}"] = {
                "rho_L12": se["marker_gradient_rank_correlation"],
                "macro_f1": pe["macro_f1"],
                "ECE": se["expected_calibration_error"],
            }
            log(f"{arm} lam={lam}: rho={summary[f'{arm}_{lam}']['rho_L12']:.4f} "
                f"F1={summary[f'{arm}_{lam}']['macro_f1']:.4f}")
        except Exception as e:  # noqa: BLE001
            log(f"{arm} lam={lam}: missing/error ({e})")
    rho_ax = summary.get("axis_ft_1.0", {}).get("rho_L12")
    rho_ss = summary.get("soft_spearman_0.5", {}).get("rho_L12")
    band = [r for r in (rho_ax, rho_ss) if r is not None]
    verdict = "BAND_PASS (extend 3-seed)" if band and max(band) >= 0.5 else "BAND_FAIL (report as-is)"
    log(f"VERDICT: {verdict} (axis_ft rho={rho_ax}, ss rho={rho_ss})")
    (RESULTS_DIR / "option_a_s42_summary.json").write_text(
        json.dumps({"summary": summary, "verdict": verdict,
                    "generated_utc": datetime.now(timezone.utc).isoformat()}, indent=2))  # noqa: UP017 — datetime.UTC does not exist at runtime (CPython 3.12)
    log("Option A s42 chain complete.")


if __name__ == "__main__":
    main()
