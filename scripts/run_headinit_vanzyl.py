#!/usr/bin/env python
"""Head-init perturbation runner (basin-selection test, FROZEN 2026-08-20).

Sequentially runs the 3 locked head-init variants (training seed 42 fixed;
only cls_decoder init varies) x 15 epochs. Idempotent: a variant is skipped
when its run_dir contains a runs_status DONE marker (with dry_run=false).
GPU guard: waits for free GPU memory >= --require-free-mib (default 4 GiB)
before each run; OOM auto-retry (up to 3 attempts, 60s apart); other failures
stop the runner. After all variants: launches the locked test-donor eval
(scgpt_eval_vanzyl_headinit.py).

Lane log: artifacts/paper1_vanzyl/head_init/lane.log (event blocks, same
format as the CellPLM lane).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("<home>/corneal_scgpt")
PYTHON = ROOT / "repo" / ".venv" / "bin" / "python"
TRAIN_SCRIPT = ROOT / "scripts" / "scgpt_cls_vanzyl_headinit.py"
EVAL_SCRIPT = ROOT / "scripts" / "scgpt_eval_vanzyl_headinit.py"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "head_init"
CHECKPOINTS = OUT / "checkpoints"
VAL_MANIFEST = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence" / "split_val_manifest.json"
VARIANTS = ["headscale_0.01", "headscale_1.0", "headseed_999"]
LOCKED = {"n_train": 38235, "n_val": 4247}
MAX_OOM_RETRIES = 3
REQUIRE_FREE_MIB = 4 * 1024
MAX_WAIT_MIN = 300  # 5 h GPU-guard budget


def log(msg: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "lane.log").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.now(UTC).isoformat(), "event": "LOG",
                             "msg": msg}, ensure_ascii=False) + "\n")


def lane_event(event: str, msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(UTC).isoformat(), "event": event, "msg": msg}
    with (OUT / "lane.log").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {event}: {msg}",
          flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def gpu_mem() -> tuple[int, int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        used, total = (int(x) for x in out.stdout.strip().split(","))
        return used, total
    except Exception:
        return -1, -1


def run_done(run_dir: Path) -> bool:
    rs = run_dir / "runs_status"
    if rs.is_file() and "DONE" in rs.read_text(encoding="utf-8"):
        aj = run_dir / "args.json"
        if aj.is_file():
            try:
                if json.loads(aj.read_text(encoding="utf-8")).get("dry_run") is False:
                    return True
            except Exception:
                return False
    return False


def wait_for_gpu() -> bool:
    waited = 0
    while True:
        used, total = gpu_mem()
        free = total - used if used >= 0 else -1
        if free >= REQUIRE_FREE_MIB:
            lane_event("GPU_GUARD_PASS", f"free {free} MiB >= {REQUIRE_FREE_MIB} MiB")
            return True
        if waited >= MAX_WAIT_MIN * 60:
            lane_event("GPU_GUARD_TIMEOUT",
                       f"free {free} MiB < {REQUIRE_FREE_MIB} after {MAX_WAIT_MIN} min")
            return False
        lane_event("GPU_GUARD_WAIT",
                   f"free {free} MiB < {REQUIRE_FREE_MIB}; waited {waited // 60} min")
        time.sleep(60)
        waited += 60


def run_one(variant: str) -> int:
    run_dir = CHECKPOINTS / f"naive_seed42_{variant}"
    if run_done(run_dir):
        log(f"SKIP {variant}: already DONE")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    lane_event("VARIANT_START", f"variant={variant} train_seed=42 (15 epochs)")
    run_log = run_dir / "run.log"
    for attempt in range(1, MAX_OOM_RETRIES + 1):
        if not wait_for_gpu():
            lane_event("VARIANT_GPU_TIMEOUT", variant)
            return 3
        cmd = [str(PYTHON), str(TRAIN_SCRIPT), "--variant", variant,
               "--out", str(CHECKPOINTS)]
        t0 = time.time()
        with run_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  text=True)
        wall_min = (time.time() - t0) / 60.0
        if proc.returncode == 0:
            lane_event("VARIANT_DONE",
                       f"{variant}: rc=0 in {wall_min:.1f} min")
            return 0
        oom = False
        try:
            txt = run_log.read_text(encoding="utf-8")
            oom = "out of memory" in txt.lower() or "OOM" in txt
        except OSError:
            txt = ""
        if oom and attempt < MAX_OOM_RETRIES:
            lane_event("VARIANT_OOM_RETRY",
                       f"{variant} attempt {attempt} rc={proc.returncode} "
                       f"OOM; backoff 60s")
            time.sleep(60)
            continue
        lane_event("VARIANT_FAIL",
                   f"{variant} attempt {attempt} rc={proc.returncode} "
                   f"after {wall_min:.1f} min (see {run_log})")
        return 1
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-eval", action="store_true",
                    help="only train variants, do not launch the test eval")
    ap.add_argument("--variant", choices=VARIANTS, default=None,
                    help="restrict to one variant (resume aid)")
    args = ap.parse_args()

    if not VAL_MANIFEST.is_file():
        log("FATAL: split_val_manifest.json missing (lock requirement)")
        return 2
    vm = json.loads(VAL_MANIFEST.read_text(encoding="utf-8"))
    if vm["n_train"] != LOCKED["n_train"] or vm["n_val"] != LOCKED["n_val"]:
        log(f"FATAL: manifest drift {vm['n_train']}/{vm['n_val']}")
        return 2
    log(f"val manifest OK: train={vm['n_train']} val={vm['n_val']}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    selected = [v for v in VARIANTS if args.variant is None or v == args.variant]
    log(f"plan: {len(selected)} variants -> {selected}")

    for variant in selected:
        rc = run_one(variant)
        if rc != 0:
            lane_event("DRIVER_EXIT", f"rc={rc}")
            return rc

    if not args.skip_eval:
        lane_event("EVAL_START", "test-donor eval (best_val + final_epoch)")
        if not wait_for_gpu():
            lane_event("EVAL_GPU_TIMEOUT", "")
            return 3
        eval_log = OUT / "head_init_eval.log"
        with eval_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run([str(PYTHON), str(EVAL_SCRIPT)],
                                  stdout=fh, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            lane_event("EVAL_FAIL", f"rc={proc.returncode} (see {eval_log})")
            return 1
        lane_event("EVAL_DONE", "head_init_summary.json written")

    lane_event("DRIVER_EXIT", "rc=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
