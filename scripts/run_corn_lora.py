#!/usr/bin/env python
"""E5 runner: CORN + LoRA combination fine-tuning (adapters + ordinal head).

Seeds 42/123/456 x 15 epochs -> artifacts/paper1_vanzyl/corn_lora/checkpoints/
corn_lora_seed{S}. GPU guard free>=4GiB (coexists with fix-2/fix-12 lanes),
OOM retry x3, VANZYL_CORN_LORA_DONE markers, per-run logs. After each seed
the per-seed args.json (incl. inline test-donor eval) is copied to
corn_lora/seed{S}_args.json.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("<home>/corneal_scgpt")
PYTHON = ROOT / "repo" / ".venv" / "bin" / "python"
SCRIPT = ROOT / "scripts" / "scgpt_lora_corn_vanzyl.py"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "corn_lora"
LOG = OUT / "corn_lora_runner.log"
SEEDS = [42, 123, 456]


def log(m: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
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
        run_dir = OUT / "checkpoints" / f"corn_lora_seed{seed}"
        marker = run_dir / "VANZYL_CORN_LORA_DONE"
        if marker.is_file():
            log(f"SKIP corn_lora_seed{seed}: already DONE")
            continue
        waited = 0
        while True:
            free = gpu_free()
            if free >= 4096:
                break
            if waited >= 300 * 60:
                log("ABORT: GPU busy after 300 min")
                return 3
            log(f"GPU free {free} MiB < 4096; wait 60s ({waited // 60} min)")
            time.sleep(60)
            waited += 60
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== E5 START CORN+LoRA seed{seed} (15 epochs) ===")
        attempt = 0
        while True:
            rl = run_dir / "run.log"
            t0 = time.time()
            with rl.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(
                    [str(PYTHON), str(SCRIPT), "--seed", str(seed),
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
        # per-seed json at the lane root (mirrors run6/lora_v2/seed{S}_args.json)
        shutil.copyfile(run_dir / "args.json", OUT / f"seed{seed}_args.json")
        log(f"DONE E5 corn_lora_seed{seed}: {wall:.1f} min")
    log("E5 runner finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
