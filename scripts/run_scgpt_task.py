#!/usr/bin/env python
"""Generic scGPT naive-CE task runner (cross-tissue / LOO arms).

Parameterized version of run_vanzyl_convergence.py (PREREG §9) — identical
GPU guard (free >= 4 GiB, poll 60 s), OOM auto-retry (max 3, 60 s apart),
idempotent per-run DONE markers, runner_summary.json. Science logic
unchanged; only paths/seeds are configurable.

Usage:
  run_scgpt_task.py --train-script scripts/scgpt_cls_epidermis.py \
      --ckpt-root artifacts/paper1_vanzyl/cross_tissue/checkpoints \
      --val-manifest artifacts/paper1_vanzyl/cross_tissue/split_val_manifest.json \
      --runs naive:42,naive:123 \
      [--require-free-mib 4096] [--max-gpu-wait-min 720]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/home/summer/corneal_scgpt")
PYTHON = ROOT / "repo" / ".venv" / "bin" / "python"
MAX_OOM_RETRIES = 3


def log(msg: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)


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
    if (run_dir / "runs_status").is_file():
        rs = (run_dir / "runs_status").read_text(encoding="utf-8")
        if "DONE" in rs:
            aj = run_dir / "args.json"
            if aj.is_file():
                try:
                    if json.loads(aj.read_text(encoding="utf-8")).get("dry_run") is False:
                        return True
                except Exception:
                    return False
    return False


def summary_update(ckpt_root: Path, entry: dict) -> None:
    sp = ckpt_root / "runner_summary.json"
    data = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {"runs": {}}
    data["runs"][f"{entry['arm']}_{entry['seed']}"] = entry
    data["updated_utc"] = now_utc()
    sp.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-script", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--runs", required=True,
                    help="comma list arm:seed, e.g. naive:42,naive:123")
    ap.add_argument("--require-free-mib", type=int, default=4096)
    ap.add_argument("--max-gpu-wait-min", type=int, default=720)
    args = ap.parse_args()

    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = ROOT / train_script
    ckpt_root = Path(args.ckpt_root)
    if not ckpt_root.is_absolute():
        ckpt_root = ROOT / ckpt_root
    val_manifest = Path(args.val_manifest)
    if not val_manifest.is_absolute():
        val_manifest = ROOT / val_manifest

    if not val_manifest.is_file():
        log(f"FATAL: val manifest missing ({val_manifest})")
        return 2
    vm = json.loads(val_manifest.read_text(encoding="utf-8"))

    runs = []
    for r in args.runs.split(","):
        arm, seed = r.split(":")
        runs.append((arm, int(seed)))
    log(f"val manifest OK: train={vm['n_train']} val={vm['n_val']} "
        f"sha_train={vm['sha256']['scgpt_vanzyl_train.h5ad'][:8]}")
    log(f"plan: {len(runs)} runs -> {runs}")

    ckpt_root.mkdir(parents=True, exist_ok=True)
    for arm, seed in runs:
        run_dir = ckpt_root / f"{arm}_seed{seed}"
        if run_done(run_dir):
            log(f"SKIP {arm} seed{seed}: already DONE")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== START {arm} seed{seed} (15 epochs) ===")

        waited = 0
        while True:
            used, total = gpu_mem()
            free = total - used if used >= 0 else -1
            if free >= args.require_free_mib:
                break
            if waited >= args.max_gpu_wait_min * 60:
                log(f"ABORT {arm} seed{seed}: free {free} MiB < "
                    f"{args.require_free_mib} after {args.max_gpu_wait_min} min")
                return 3
            log(f"GPU free {free} MiB < {args.require_free_mib} "
                f"({used} MiB used); wait 60s ({waited // 60} min)")
            time.sleep(60)
            waited += 60

        attempt = 0
        while True:
            run_log = run_dir / "run.log"
            cmd = [str(PYTHON), str(train_script), "--arm", arm,
                   "--seed", str(seed), "--out", str(ckpt_root)]
            t0 = time.time()
            with run_log.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                      text=True)
            wall_min = (time.time() - t0) / 60.0
            if proc.returncode == 0:
                break
            txt = run_log.read_text(encoding="utf-8", errors="ignore")
            is_oom = ("out of memory" in txt.lower()
                      or "oom" in txt.lower() or "cuda error" in txt.lower())
            attempt += 1
            if is_oom and attempt <= MAX_OOM_RETRIES:
                log(f"OOM exit (attempt {attempt}/{MAX_OOM_RETRIES}) "
                    f"{arm} seed{seed} after {wall_min:.1f} min; retry in 60s")
                time.sleep(60)
                continue
            log(f"FAIL {arm} seed{seed}: exit={proc.returncode} after "
                f"{wall_min:.1f} min (see {run_log}); runner stops")
            summary_update(ckpt_root, {"arm": arm, "seed": seed,
                                       "status": "FAILED", "exit": proc.returncode,
                                       "wall_min": round(wall_min, 1),
                                       "ts": now_utc()})
            return 1

        ckpt = run_dir / "model_best_val.pt"
        if not ckpt.is_file():
            log(f"FAIL {arm} seed{seed}: exit=0 but model_best_val.pt missing")
            return 1
        (run_dir / "DONE").write_text(
            f"DONE\n{now_utc()}\narm={arm} seed={seed}\n", encoding="utf-8")
        log(f"DONE {arm} seed{seed}: {wall_min:.1f} min; best-val ckpt {ckpt}")
        summary_update(ckpt_root, {"arm": arm, "seed": seed, "status": "DONE",
                                   "wall_min": round(wall_min, 1),
                                   "checkpoint": str(ckpt), "ts": now_utc()})

    log("runner finished (all selected runs processed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
