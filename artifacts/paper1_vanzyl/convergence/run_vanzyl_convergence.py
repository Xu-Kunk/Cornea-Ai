#!/usr/bin/env python
"""van Zyl Arm A convergence runner (PREREG §9, LOCKED).

Sequentially runs naive CE x seeds {42,123,456} x 15 epochs. Idempotent
(VANZYL_CONV_DONE marker; interrupted runs resume). GPU guard: waits for
free GPU memory (<= 1 GiB used) before each run — the GF 1-epoch lane may be
running concurrently on the same GPU. OOM auto-retry: a non-zero exit whose
run.log mentions out-of-memory is retried (up to 3 attempts, 60s apart);
other failures stop the runner.
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
TRAIN_SCRIPT = ROOT / "scripts" / "scgpt_cls_vanzyl_convergence.py"
CONV = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence"
CHECKPOINTS = CONV / "checkpoints"
VAL_MANIFEST = CONV / "split_val_manifest.json"
RUNS = [("naive", 42), ("naive", 123), ("naive", 456)]
LOCKED = {"n_train": 38235, "n_val": 4247}
MAX_OOM_RETRIES = 3


def log(msg: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with (CONV / "runner.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def gpu_mem() -> tuple[int, int]:
    """Return (used_mib, total_mib); (-1, -1) on failure."""
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
    if (run_dir / "VANZYL_CONV_DONE").is_file():
        return True
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
    return False


def summary_update(entry: dict) -> None:
    sp = CONV / "runner_summary.json"
    data = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {"runs": {}}
    data["runs"][f"{entry['arm']}_{entry['seed']}"] = entry
    data["updated_utc"] = now_utc()
    sp.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-gpu-wait-min", type=int, default=480)
    ap.add_argument("--require-free-mib", type=int, default=4096,
                    help="start when free GPU memory >= this (PREREG §9: the "
                         "two arms may run in parallel; verified headroom)")
    ap.add_argument("--seed", type=int, default=None, help="restrict (resume aid)")
    args = ap.parse_args()

    if not VAL_MANIFEST.is_file():
        log("FATAL: split_val_manifest.json missing (lock requirement)")
        return 2
    vm = json.loads(VAL_MANIFEST.read_text(encoding="utf-8"))
    if vm["n_train"] != LOCKED["n_train"] or vm["n_val"] != LOCKED["n_val"]:
        log(f"FATAL: manifest drift {vm['n_train']}/{vm['n_val']}")
        return 2
    log(f"val manifest OK: train={vm['n_train']} val={vm['n_val']} "
        f"sha_train={vm['sha256']['scgpt_vanzyl_train.h5ad'][:8]}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    selected = [r for r in RUNS if args.seed is None or r[1] == args.seed]
    log(f"plan: {len(selected)} runs -> {selected}")

    for arm, seed in selected:
        run_dir = CHECKPOINTS / f"{arm}_seed{seed}"
        if run_done(run_dir):
            log(f"SKIP {arm} seed{seed}: already DONE")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== START {arm} seed{seed} (15 epochs, ~2.1h) ===")

        # GPU guard: free memory >= require-free-mib (PREREG §9 parallel arms;
        # GF 1-epoch lane may be running concurrently)
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
                f"(GF lane {used} MiB); wait 60s ({waited // 60} min)")
            time.sleep(60)
            waited += 60

        attempt = 0
        while True:
            run_log = run_dir / "run.log"
            cmd = [str(PYTHON), str(TRAIN_SCRIPT), "--arm", arm, "--seed", str(seed),
                   "--out", str(CHECKPOINTS)]
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
            summary_update({"arm": arm, "seed": seed, "status": "FAILED",
                            "exit": proc.returncode,
                            "wall_min": round(wall_min, 1), "ts": now_utc()})
            return 1

        ckpt = run_dir / "model_best_val.pt"
        if not ckpt.is_file():
            log(f"FAIL {arm} seed{seed}: exit=0 but model_best_val.pt missing")
            return 1
        (run_dir / "VANZYL_CONV_DONE").write_text(
            f"DONE\n{now_utc()}\narm={arm} seed={seed}\n", encoding="utf-8")
        log(f"DONE {arm} seed{seed}: {wall_min:.1f} min; best-val ckpt {ckpt}")
        summary_update({"arm": arm, "seed": seed, "status": "DONE",
                        "wall_min": round(wall_min, 1), "checkpoint": str(ckpt),
                        "ts": now_utc()})

    log("runner finished (all selected runs processed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
