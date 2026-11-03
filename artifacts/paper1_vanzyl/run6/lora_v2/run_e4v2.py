#!/usr/bin/env python
"""E4-v2 runner: corrected LoRA fine-tuning (adapters re-enabled, 2026-08-20).

Seeds 42/123/456 x 15 epochs -> run6/lora_v2/checkpoints/lora_seed{S} (the v1
runs silently trained the head only and are preserved as the audit trail).
GPU guard free>=4GiB, OOM retry x3, VANZYL_E4V2_DONE markers, per-run logs.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("<home>/corneal_scgpt")
PYTHON = ROOT / "repo" / ".venv" / "bin" / "python"
LORA_SCRIPT = ROOT / "scripts" / "scgpt_lora_vanzyl.py"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "lora_v2"
LOG = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "e4v2_runner.log"
SEEDS = [42, 123, 456]


def log(m: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu_free() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=15)
        used, total = (int(x) for x in out.stdout.strip().split(","))
        return total - used
    except Exception:
        return -1


def main() -> int:
    for seed in SEEDS:
        run_dir = OUT / "checkpoints" / f"lora_seed{seed}"
        marker = run_dir / "VANZYL_E4V2_DONE"
        if marker.is_file():
            log(f"SKIP lora_seed{seed}: already DONE")
            continue
        waited = 0
        while True:
            free = gpu_free()
            if free >= 4096:
                break
            if waited >= 300 * 60:
                log(f"ABORT: GPU busy after 300 min")
                return 3
            log(f"GPU free {free} MiB < 4096; wait 60s ({waited // 60} min)")
            time.sleep(60)
            waited += 60
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== E4v2 START LoRA seed{seed} (15 epochs) ===")
        attempt = 0
        while True:
            rl = run_dir / "run.log"
            t0 = time.time()
            with rl.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(
                    [str(PYTHON), str(LORA_SCRIPT), "--seed", str(seed),
                     "--out", str(OUT)], stdout=fh, stderr=subprocess.STDOUT,
                    text=True)
            wall = (time.time() - t0) / 60
            if proc.returncode == 0:
                break
            txt = rl.read_text(encoding="utf-8", errors="ignore")
            is_oom = ("out of memory" in txt.lower() or "cuda error" in txt.lower())
            attempt += 1
            if is_oom and attempt <= 3:
                log(f"OOM seed{seed} (attempt {attempt}); retry in 60s")
                time.sleep(60)
                continue
            log(f"FAIL seed{seed}: exit={proc.returncode} after {wall:.1f} min")
            return 1
        marker.write_text(f"DONE\nseed={seed}\n", encoding="utf-8")
        log(f"DONE E4v2 lora_seed{seed}: {wall:.1f} min")
    log("E4v2 runner finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
