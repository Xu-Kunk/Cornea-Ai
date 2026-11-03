#!/usr/bin/env python
"""E3 + E4 orchestrated runner (<compute-node>, sequential, idempotent).

E3: scGPT convergence arm, 2 new seeds {789, 2024} x 15 epochs (naive CE,
locked §9 protocol, per-epoch val trajectory + best-val selection).
E4: scGPT LoRA fine-tuning (QKV r=8 alpha=16, base frozen), seeds {42, 123}
x 15 epochs, same protocol. Both evaluate best-val + final-epoch on the
locked test donor inline.

GPU guard: start when free memory >= 4 GiB (parallel GF lane possible);
OOM exit auto-retry (3 attempts). Markers: VANZYL_E3_DONE /
VANZYL_E4_DONE per run; interrupted runs resume.
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
CONV_SCRIPT = ROOT / "scripts" / "scgpt_cls_vanzyl_convergence.py"
LORA_SCRIPT = ROOT / "scripts" / "scgpt_lora_vanzyl.py"
E3_OUT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence" / "checkpoints"
E4_OUT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "lora"
RUN_LOG = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "e3e4_runner.log"
E3_SEEDS = [789, 2024]
E4_SEEDS = [42, 123]
MAX_OOM_RETRIES = 3


def log(msg: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu_mem() -> tuple[int, int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        used, total = (int(x) for x in out.stdout.strip().split(","))
        return used, total
    except Exception:
        return -1, -1


def run_done(marker: Path) -> bool:
    return marker.is_file()


def gpu_guard(require_free_mib: int = 4096, max_wait_min: int = 300) -> bool:
    waited = 0
    while True:
        used, total = gpu_mem()
        free = total - used if used >= 0 else -1
        if free >= require_free_mib:
            return True
        if waited >= max_wait_min * 60:
            log(f"ABORT: GPU free {free} MiB < {require_free_mib} after "
                f"{max_wait_min} min")
            return False
        log(f"GPU free {free} MiB < {require_free_mib}; wait 60s "
            f"({waited // 60} min)")
        time.sleep(60)
        waited += 60


def run_with_retry(cmd: list[str], run_log: Path, label: str) -> bool:
    attempt = 0
    while True:
        t0 = time.time()
        with run_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  text=True)
        wall_min = (time.time() - t0) / 60.0
        if proc.returncode == 0:
            log(f"DONE {label}: {wall_min:.1f} min")
            return True
        txt = run_log.read_text(encoding="utf-8", errors="ignore")
        is_oom = ("out of memory" in txt.lower() or "oom" in txt.lower()
                  or "cuda error" in txt.lower())
        attempt += 1
        if is_oom and attempt <= MAX_OOM_RETRIES:
            log(f"OOM {label} (attempt {attempt}); retry in 60s")
            time.sleep(60)
            continue
        log(f"FAIL {label}: exit={proc.returncode} after {wall_min:.1f} min")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e3-only", action="store_true")
    ap.add_argument("--e4-only", action="store_true")
    ap.add_argument("--require-free-mib", type=int, default=4096)
    args = ap.parse_args()
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ---- E3: convergence, seeds 789/2024 ----
    if not args.e4_only:
        for seed in E3_SEEDS:
            marker = E3_OUT / f"naive_seed{seed}" / "VANZYL_E3_DONE"
            if run_done(marker):
                log(f"SKIP E3 naive_seed{seed}: already DONE")
                continue
            if not gpu_guard(args.require_free_mib):
                return 3
            log(f"=== E3 START naive seed{seed} (15 epochs) ===")
            run_dir = E3_OUT / f"naive_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            ok = run_with_retry(
                [str(PYTHON), str(CONV_SCRIPT), "--arm", "naive",
                 "--seed", str(seed), "--out", str(E3_OUT)],
                run_dir / "run.log", f"E3 naive seed{seed}")
            if not ok:
                return 1
            marker.write_text(f"DONE\narm=naive seed={seed}\n",
                              encoding="utf-8")

    # ---- E4: LoRA, seeds 42/123 ----
    if not args.e3_only:
        for seed in E4_SEEDS:
            marker = E4_OUT / "checkpoints" / f"lora_seed{seed}" / "VANZYL_E4_DONE"
            if run_done(marker):
                log(f"SKIP E4 lora_seed{seed}: already DONE")
                continue
            if not gpu_guard(args.require_free_mib):
                return 3
            log(f"=== E4 START LoRA seed{seed} (15 epochs) ===")
            run_dir = E4_OUT / "checkpoints" / f"lora_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            ok = run_with_retry(
                [str(PYTHON), str(LORA_SCRIPT), "--seed", str(seed),
                 "--out", str(E4_OUT)],
                run_dir / "run.log", f"E4 lora seed{seed}")
            if not ok:
                return 1
            marker.write_text(f"DONE\nlora seed={seed}\n", encoding="utf-8")

    log("E3+E4 runner finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
