#!/usr/bin/env python
"""Cross-arms driver: scGPT cross-tissue + scGPT LOO + CellPLM consistency
(approved supplementary experiments; reviewer points: single-tissue /
single-donor bias).

Sequential phases (each idempotent; re-launch resumes):
  P1 scGPT epidermis (cross_tissue): 2 seeds x 15 epochs naive CE
  P2 scGPT epidermis final eval (best-val + final-epoch, held-out test)
  P3 scGPT LOO Hu0220: 2 seeds x 15 epochs naive CE
  P4 scGPT LOO final eval
  P5 CellPLM consistency (epidermis + LOO220): 2 seeds each

GPU guard (free >= 4 GiB, poll 60 s, budget 12 h) inside each phase; OOM
retry x3. lane.log event blocks written per artifact dir. Runs detached on
<compute-node> (setsid nohup; driver rc -> driver.rc).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("<home>/corneal_scgpt")
SCRIPTS = ROOT / "scripts"
PY = ROOT / "repo" / ".venv" / "bin" / "python"
CELLPLM_PY = "<home>/corneal_scgpt/venv/bin/python"

CROSS = ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue"
LOO = ROOT / "artifacts" / "paper1_vanzyl" / "loo"
CELLPLM = ROOT / "artifacts" / "paper1_vanzyl" / "cellplm_consistency"
DONE_MARKERS = {
    "P1": CROSS / "P1_SCGPT_TRAIN_DONE",
    "P2": CROSS / "P2_SCGPT_EVAL_DONE",
    "P3": LOO / "P3_SCGPT_TRAIN_DONE",
    "P4": LOO / "P4_SCGPT_EVAL_DONE",
    "P5": CELLPLM / "P5_CELLPLM_DONE",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def lane_event(path: Path, kind: str, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": now_utc(), "event": kind, "msg": msg},
                      ensure_ascii=False)
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], label: str) -> int:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = (time.time() - t0) / 60
    print(f"[{label}] rc={proc.returncode} wall={wall:.1f} min", flush=True)
    if proc.returncode != 0:
        print(f"[{label}] STDOUT tail:\n{proc.stdout[-3000:]}")
        print(f"[{label}] STDERR tail:\n{proc.stderr[-3000:]}")
    return proc.returncode


def main() -> int:
    lane_event(CROSS / "lane.log", "START",
               "Cross-arms driver: P1-P5 (scGPT cross-tissue + LOO + CellPLM)")

    # P1: scGPT epidermis training (2 seeds)
    if not DONE_MARKERS["P1"].is_file():
        lane_event(CROSS / "lane.log", "P1_START",
                   "scGPT epidermis naive CE x seeds {42,123} x 15 epochs")
        rc = run([PY, str(SCRIPTS / "run_scgpt_task.py"),
                  "--train-script", str(SCRIPTS / "scgpt_cls_epidermis.py"),
                  "--ckpt-root", str(CROSS / "checkpoints"),
                  "--val-manifest", str(CROSS / "split_val_manifest.json"),
                  "--runs", "naive:42,naive:123"],
                 "P1")
        if rc != 0:
            lane_event(CROSS / "lane.log", "P1_FAIL", f"rc={rc}")
            return 1
        DONE_MARKERS["P1"].write_text(now_utc() + "\n")
        lane_event(CROSS / "lane.log", "P1_DONE", "scGPT epidermis 2 seeds done")
    else:
        lane_event(CROSS / "lane.log", "P1_SKIP", "marker present")

    # P2: scGPT epidermis eval
    if not DONE_MARKERS["P2"].is_file():
        lane_event(CROSS / "lane.log", "P2_START", "final eval (best-val + final)")
        rc = run([PY, str(SCRIPTS / "scgpt_eval_epidermis.py")], "P2")
        if rc != 0:
            lane_event(CROSS / "lane.log", "P2_FAIL", f"rc={rc}")
            return 1
        DONE_MARKERS["P2"].write_text(now_utc() + "\n")
        lane_event(CROSS / "lane.log", "P2_DONE", "scGPT epidermis eval done")
    else:
        lane_event(CROSS / "lane.log", "P2_SKIP", "marker present")

    # P3: scGPT LOO training (2 seeds)
    if not DONE_MARKERS["P3"].is_file():
        lane_event(LOO / "lane.log", "P3_START",
                   "scGPT LOO (Hu0220) naive CE x seeds {42,123} x 15 epochs")
        rc = run([PY, str(SCRIPTS / "run_scgpt_task.py"),
                  "--train-script", str(SCRIPTS / "scgpt_cls_vanzyl_loo220.py"),
                  "--ckpt-root", str(LOO / "checkpoints"),
                  "--val-manifest", str(LOO / "split_val_manifest.json"),
                  "--runs", "naive:42,naive:123"],
                 "P3")
        if rc != 0:
            lane_event(LOO / "lane.log", "P3_FAIL", f"rc={rc}")
            return 1
        DONE_MARKERS["P3"].write_text(now_utc() + "\n")
        lane_event(LOO / "lane.log", "P3_DONE", "scGPT LOO 2 seeds done")
    else:
        lane_event(LOO / "lane.log", "P3_SKIP", "marker present")

    # P4: scGPT LOO eval
    if not DONE_MARKERS["P4"].is_file():
        lane_event(LOO / "lane.log", "P4_START", "final eval (best-val + final)")
        rc = run([PY, str(SCRIPTS / "scgpt_eval_vanzyl_loo220.py")], "P4")
        if rc != 0:
            lane_event(LOO / "lane.log", "P4_FAIL", f"rc={rc}")
            return 1
        DONE_MARKERS["P4"].write_text(now_utc() + "\n")
        lane_event(LOO / "lane.log", "P4_DONE", "scGPT LOO eval done")
    else:
        lane_event(LOO / "lane.log", "P4_SKIP", "marker present")

    # P5: CellPLM consistency (epidermis + LOO220)
    if not DONE_MARKERS["P5"].is_file():
        lane_event(CELLPLM / "lane.log", "P5_START",
                   "CellPLM epidermis + LOO220 naive CE x 2 seeds each")
        rc = run([CELLPLM_PY, str(SCRIPTS / "run_cellplm_cross_driver.py")], "P5")
        if rc != 0:
            lane_event(CELLPLM / "lane.log", "P5_FAIL", f"rc={rc}")
            return 1
        DONE_MARKERS["P5"].write_text(now_utc() + "\n")
        lane_event(CELLPLM / "lane.log", "P5_DONE", "CellPLM consistency done")
    else:
        lane_event(CELLPLM / "lane.log", "P5_SKIP", "marker present")

    lane_event(CROSS / "lane.log", "DONE", "all phases complete")
    lane_event(LOO / "lane.log", "DONE", "all phases complete")
    print("CROSS_ARMS_DRIVER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
