#!/usr/bin/env python
"""Run the Paper 1 D5 ordinal grid: ordinal-naive (λ=0) + ordinal-CAFT (λ=0.5)
x seeds {42, 123, 456} = 6 runs, then Tier-B evaluate each.

Usage:
    python scripts/run_ordinal_grid.py --first      # seed 42 both arms only (feasibility)
    python scripts/run_ordinal_grid.py              # full 6-run grid
    python scripts/run_ordinal_grid.py --eval-only  # evaluate already-trained checkpoints

Sequential on the single GB10 GPU (mirrors run_lambda_grid.py / the Paper 1
strict-fair launcher). Each run streams stdout+stderr to
artifacts/paper1_ordinal_eval/logs/ordinal_lam<λ>_seed<seed>.log. Success is
confirmed by the ORDINAL_DONE marker in the run's save dir
(save/ordinal_gf_lam<λ>_seed<seed>_<ts>/); failed runs are logged and the
grid CONTINUES. A runs_status.json table is written to the artifacts dir.

Feasibility gate: --first trains seed 42 both arms, then evaluates them. If
the ordinal arms preserve the differentiation gradient (rho not collapsed),
relaunch WITHOUT --first for the remaining 4 runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

# ── Environment (must be set before any torch import) ────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "artifacts" / "paper1_ordinal_eval"
LOG_DIR = RESULTS_DIR / "logs"
STATUS_JSON = RESULTS_DIR / "runs_status.json"

LAMBDAS = [0.0, 0.5]
SEEDS = [42, 123, 456]
EPOCHS = 3
DATA_FILE = "corneal_100k_donor_split.h5ad"
SPLIT_MANIFEST = "artifacts/paper1_fair_split/manifest.json"
PY = os.environ.get("PY", "python3")
RUN_TIMEOUT_S = 172800  # 48 h per run

# Robust GX10 guards (same as run_lambda_grid.py): never start a run hot or
# memory-pressured on the GB10's shared 128 GB CPU/GPU memory.
MIN_AVAIL_GB = 40.0
COOL_TEMP_C = 80
TEMP_WAIT_MAX_S = 5400


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def gpu_temp_c() -> int | None:
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    temps = [int(line.strip()) for line in res.stdout.strip().split("\n") if line.strip()]
    return max(temps) if temps else None


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().split("\n"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    return 0.0


def wait_until_ready(lam: float, seed: int) -> None:
    t0 = time.time()
    while True:
        temp = gpu_temp_c()
        avail = mem_available_gb()
        if temp is not None and temp < COOL_TEMP_C and avail >= MIN_AVAIL_GB:
            print(f"{_ts()} λ={lam} seed={seed}  ready (GPU {temp} °C, {avail:.0f} GiB avail)\n",
                  flush=True)
            return
        if time.time() - t0 > TEMP_WAIT_MAX_S:
            print(f"ERROR: λ={lam} seed={seed}  not ready after {TEMP_WAIT_MAX_S // 60} min "
                  f"(GPU {temp} °C, {avail:.0f} GiB avail); aborting grid.", flush=True)
            sys.exit(1)
        reason = []
        if temp is not None and temp >= COOL_TEMP_C:
            reason.append(f"GPU {temp} °C")
        if avail < MIN_AVAIL_GB:
            reason.append(f"{avail:.0f} GiB avail")
        print(f"{_ts()} λ={lam} seed={seed}  waiting ({', '.join(reason)})…", flush=True)
        time.sleep(60)


# ── Pre-flight checks (mirror run_lambda_grid.py) ────────────────────────


def preflight() -> None:
    print(f"{_ts()} pre-flight checks …", flush=True)

    swap = Path("/proc/swaps").read_text().strip().split("\n")
    if len(swap) > 1:
        print(f"ERROR: swap active ({len(swap) - 1} entries); aborting.", flush=True)
        sys.exit(1)

    res = subprocess.run(["systemctl", "is-active", "vllm.service"],
                         capture_output=True, text=True, timeout=10)
    if res.stdout.strip() == "active":
        print("ERROR: vllm.service is active; aborting.", flush=True)
        sys.exit(1)
    res_docker = subprocess.run(
        ["docker", "ps", "--filter", "name=vllm", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=10,
    )
    if res_docker.stdout.strip():
        print(f"ERROR: vLLM docker container running ({res_docker.stdout.strip()}); aborting.",
              flush=True)
        sys.exit(1)

    free_gb = shutil.disk_usage(PROJECT).free / (1024 ** 3)
    if free_gb < 50:
        print(f"ERROR: only {free_gb:.1f} GB free (< 50 GB); aborting.", flush=True)
        sys.exit(1)

    t0 = time.time()
    while True:
        temp = gpu_temp_c()
        if temp is None:
            print("ERROR: cannot read GPU temperature; aborting.", flush=True)
            sys.exit(1)
        if temp < COOL_TEMP_C:
            print(f"  ✓ GPU temp: {temp} °C", flush=True)
            break
        if time.time() - t0 > TEMP_WAIT_MAX_S:
            print(f"ERROR: GPU still {temp} °C (>= {COOL_TEMP_C} °C) after "
                  f"{TEMP_WAIT_MAX_S // 60} min of waiting; aborting.", flush=True)
            sys.exit(1)
        print(f"  … GPU {temp} °C >= {COOL_TEMP_C} °C, cooling… ", flush=True)
        time.sleep(60)

    print(f"{_ts()} all pre-flight checks passed.\n", flush=True)


# ── Directory helpers ────────────────────────────────────────────────────


def find_save_dir(lam: float, seed: int) -> Path | None:
    dirs = sorted(glob(str(PROJECT / "save" / f"ordinal_gf_lam{lam}_seed{seed}_*")))
    return Path(dirs[-1]) if dirs else None


def find_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(
        glob(str(save_dir / "model" / "checkpoint-*")),
        key=lambda p: int(p.rsplit("checkpoint-", 1)[-1]),
    )
    return Path(ckpts[-1]) if ckpts else None


# ── Task runners ─────────────────────────────────────────────────────────


def run_ordinal_training(lam: float, seed: int) -> Path | None:
    """Train one ordinal arm; returns the save dir on success (ORDINAL_DONE)."""
    cmd = [
        PY, "-u",
        str(PROJECT / "scripts" / "ordinal_gf.py"),
        "--ordinal_lambda", str(lam),
        "--data_file", DATA_FILE,
        "--epochs", str(EPOCHS),
        "--seed", str(seed),
        "--strict-fair",
        "--split_manifest", SPLIT_MANIFEST,
    ]
    print(f"{_ts()} λ={lam} seed={seed}  ordinal training starting …", flush=True)
    log_path = LOG_DIR / f"ordinal_lam{lam}_seed{seed}.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S)
    wall = (time.time() - t0) / 60

    if proc.returncode != 0:
        print(f"{_ts()} λ={lam} seed={seed}  TRAIN FAILED (rc={proc.returncode}, "
              f"{wall:.1f} min); log: {log_path}", flush=True)
        try:
            tail = log_path.read_text()[-2000:]
        except OSError:
            tail = ""
        if tail:
            print(tail, flush=True)
        return None

    save_dir = find_save_dir(lam, seed)
    if save_dir is None or not (save_dir / "ORDINAL_DONE").exists():
        print(f"{_ts()} λ={lam} seed={seed}  ORDINAL_DONE marker missing; incomplete.",
              flush=True)
        return None
    print(f"{_ts()} λ={lam} seed={seed}  TRAINED ({wall:.1f} min)  → {save_dir.name}",
          flush=True)
    return save_dir


def run_ordinal_eval(lam: float, seed: int) -> bool:
    """Evaluate one ordinal checkpoint; returns True if results JSON written."""
    out_json = RESULTS_DIR / f"results_{lam}_{seed}.json"
    if out_json.exists():
        print(f"{_ts()} λ={lam} seed={seed}  eval already present → skip", flush=True)
        return True
    cmd = [
        PY, "-u",
        str(PROJECT / "scripts" / "run_ordinal_eval.py"),
        "--runs", f"{lam}:{seed}",
    ]
    print(f"{_ts()} λ={lam} seed={seed}  evaluating …", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if res.returncode != 0 or not out_json.exists():
        print(f"{_ts()} λ={lam} seed={seed}  EVAL FAILED (rc={res.returncode})", flush=True)
        if res.stderr:
            print(res.stderr[-2000:], flush=True)
        return False
    print(f"{_ts()} λ={lam} seed={seed}  evaluated → {out_json.name}", flush=True)
    return True


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", action="store_true",
                        help="Feasibility: only seed 42, both arms.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate already-trained checkpoints.")
    args = parser.parse_args()

    if not args.eval_only:
        preflight()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [42] if args.first else SEEDS
    runs = [(lam, seed) for lam in LAMBDAS for seed in seeds]
    print(f"{_ts()} grid: λ∈{LAMBDAS} × seed∈{seeds} → {len(runs)} runs "
          f"(mode={'eval-only' if args.eval_only else 'train+eval'})", flush=True)

    status: list[dict[str, Any]] = []
    for lam, seed in runs:
        save_dir = find_save_dir(lam, seed)
        trained = save_dir is not None and (save_dir / "ORDINAL_DONE").exists()

        if not args.eval_only:
            if trained:
                print(f"{_ts()} λ={lam} seed={seed}  already trained (ORDINAL_DONE) → skip\n",
                      flush=True)
            else:
                wait_until_ready(lam, seed)
                save_dir = run_ordinal_training(lam, seed)
                trained = save_dir is not None
                if not trained:
                    status.append({"lambda": lam, "seed": seed, "train": "FAIL",
                                   "eval": "skipped", "save_dir": None})
                    continue
        else:
            if not trained:
                print(f"{_ts()} λ={lam} seed={seed}  no ORDINAL_DONE save dir; skipping.",
                      flush=True)
                status.append({"lambda": lam, "seed": seed, "train": "missing",
                               "eval": "skipped", "save_dir": str(save_dir) if save_dir else None})
                continue

        # Training → eval (idempotent; skips existing results JSON).
        if args.eval_only and not trained:
            continue
        eval_ok = run_ordinal_eval(lam, seed) if trained else False
        status.append({
            "lambda": lam,
            "seed": seed,
            "train": "OK" if trained else "FAIL",
            "eval": "OK" if eval_ok else "FAIL",
            "save_dir": str(save_dir) if save_dir else None,
            "updated_utc": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    with open(STATUS_JSON, "w") as f:
        json.dump({
            "grid": f"λ∈{LAMBDAS} × seed∈{seeds}",
            "mode": "eval-only" if args.eval_only else "train+eval",
            "runs": status,
        }, f, indent=2)
    print(f"\n{_ts()} status -> {STATUS_JSON}")

    n_ok = sum(1 for s in status if s["train"] == "OK" and s["eval"] == "OK")
    n_total = len(status)
    print(f"\n{'=' * 60}")
    print(f"D5 ordinal grid: {n_ok}/{n_total} runs trained+evaluated")
    print(f"{'=' * 60}")
    if n_ok < n_total:
        print("Run complete but PARTIAL; relaunch to resume (idempotent).")


if __name__ == "__main__":
    main()
