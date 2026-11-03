#!/usr/bin/env python3
"""Autonomous Paper 1 chain: Phase A complete -> method-arm eval -> controller v2 -> EA.

Replaces the fragile setsid'd bash sentinels (which silently died) with one
robust, logged chain process. Steps (each waits/block on its predecessor):

  1. WAIT: artifacts/paper1_method_arms/phase_a_launch.log contains "Run complete"
     (poll 300 s, timeout 36 h).
  2. EVAL: run scripts/run_method_arms_eval.py (writes results_<arm>_<lam>_<seed>.json,
     flips runs_status eval->OK). GPU is free after Phase A.
  3. CONTROLLER v2: run scripts/run_phase_b_controller.py (waits for the 5 result
     JSONs -> pre-registered lambda decision -> Phase B arms -> flow extension;
     blocking).
  4. EA (after controller returns = Phase B/flow done): run run_energy_anchor.sh
     --first (its own GPU guards + ENERGY_ANCHOR_DONE skip).

Every step logs a line with timestamp to artifacts/paper1_method_arms/autonomous_chain.log;
a heartbeat line is appended each poll so liveness is auditable.

Usage: nohup python3 scripts/run_autonomous_chain.py > /dev/null 2>&1 &
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LAUNCH_LOG = PROJECT / "artifacts" / "paper1_method_arms" / "phase_a_launch.log"
CHAIN_LOG = PROJECT / "artifacts" / "paper1_method_arms" / "autonomous_chain.log"
PY = os.environ.get("PY", "/home/dq/venv-ds/bin/python3")
ENV = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

POLL_S = 300
PHASE_A_TIMEOUT_S = 36 * 3600


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}"  # noqa: UP017 — datetime.UTC does not exist at runtime (CPython 3.12); ruff's suggestion is invalid
    print(line, flush=True)
    with open(CHAIN_LOG, "a") as f:
        f.write(line + "\n")


def wait_phase_a_done() -> None:
    t0 = time.time()
    while time.time() - t0 < PHASE_A_TIMEOUT_S:
        try:
            if "runs trained" in LAUNCH_LOG.read_text():  # success ("5/5 runs trained") and partial ("Run complete but PARTIAL ... N/5 runs trained") both contain this
                log("Phase A complete (Run complete detected).")
                return
        except OSError:
            pass
        log(f"waiting for Phase A... (elapsed {(time.time() - t0) / 3600:.1f}h)")
        time.sleep(POLL_S)
    log("ERROR: Phase A did not complete within 36 h; chain aborted.")
    sys.exit(1)


def run(cmd: list[str], tag: str) -> None:
    log(f"launching {tag}: {' '.join(cmd[:3])} ...")
    proc = subprocess.run(cmd, cwd=str(PROJECT), env=ENV)
    log(f"{tag} finished rc={proc.returncode}")


def main() -> None:
    log("=" * 60)
    log("autonomous chain start")
    wait_phase_a_done()
    time.sleep(60)  # grace: let the Phase A launcher release its CUDA context before the eval step

    log("--- step 2: method-arm eval ---")
    run([PY, "-u", str(PROJECT / "scripts" / "run_method_arms_eval.py")], "method-arms eval")

    log("--- step 3: controller v2 (lambda decision + Phase B + flow) ---")
    run([PY, "-u", str(PROJECT / "scripts" / "run_phase_b_controller.py")], "controller v2")

    log("--- step 4: energy anchor (after Phase B/flow) ---")
    run(["bash", str(PROJECT / "scripts" / "run_energy_anchor.sh"), "--first"], "energy anchor")

    log("autonomous chain complete: all Paper 1 model runs finished.")


if __name__ == "__main__":
    main()
