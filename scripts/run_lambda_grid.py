#!/usr/bin/env python
"""Run CAFT λ-grid: fine-tune Geneformer at each λ, measure GPR via paired checkpoint.

Usage:
    python scripts/run_lambda_grid.py

Sequentially processes λ ∈ {0.01, 0.1, 0.5, 1.0, 5.0, 10.0}.
Skips λ values already present in results.tsv (idempotent).
On failure for any λ, logs the error and continues to the next.
"""

from __future__ import annotations

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

# ── Fixed paths ──────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
PRETRAINED = (
    Path.home()
    / ".cache/huggingface/hub/models--ctheodoris--Geneformer"
    / "snapshots"
    / "04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
    / "Geneformer-V2-104M"
)
TOKENIZED_30K = (
    PROJECT
    / "save/caft_gf_lam0.0_seed42_20260724-085100"
    / "corneal_layer_subset_tk"
    / "cell.dataset"
)
RESULTS_DIR = PROJECT / "artifacts" / "lambda_grid"
RESULTS_TSV = RESULTS_DIR / "results.tsv"
LAMBDA_GRID = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
SEED = 42

# Robust <compute-node> guards: the GB10 shares 128 GB between CPU and GPU, and a global
# OOM can freeze the whole host (NVIDIA forum / pytorch#174358). Keep a generous
# headroom and never start a run hot or memory-pressured.
MIN_AVAIL_GB = 40.0          # refuse to (re)start a λ unless >= this much RAM is available
COOL_TEMP_C = 80             # wait until GPU cools below this before each run (user-approved 2026-07-28; sustained load 77-80C)
TEMP_WAIT_MAX_S = 5400       # give up if it has not cooled within 90 min

TSV_HEADER = (
    "lambda\tcs_after\tgpr\tgpr_ci_low\tgpr_ci_high\telapsed_min\tsave_dir\ttimestamp\n"
)


def _ts() -> str:
    """Return current time formatted as [HH:MM:SS]."""
    return datetime.now().strftime("[%H:%M:%S]")


def gpu_temp_c() -> int | None:
    """Return the max GPU temperature in °C, or None if it cannot be read."""
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    temps = [int(line.strip()) for line in res.stdout.strip().split("\n") if line.strip()]
    return max(temps) if temps else None


def mem_available_gb() -> float:
    """Return available system memory in GiB (from /proc/meminfo)."""
    for line in Path("/proc/meminfo").read_text().split("\n"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    return 0.0


def wait_until_ready(lam: float) -> None:
    """Block until the GPU is cool and enough RAM is available; then proceed."""
    t0 = time.time()
    while True:
        temp = gpu_temp_c()
        avail = mem_available_gb()
        if temp is not None and temp < COOL_TEMP_C and avail >= MIN_AVAIL_GB:
            print(
                f"{_ts()} λ={lam}  ready (GPU {temp} °C, {avail:.0f} GiB avail)\n",
                flush=True,
            )
            return
        if time.time() - t0 > TEMP_WAIT_MAX_S:
            print(
                f"ERROR: λ={lam}  not ready after {TEMP_WAIT_MAX_S // 60} min "
                f"(GPU {temp} °C, {avail:.0f} GiB avail); aborting grid.",
                flush=True,
            )
            sys.exit(1)
        reason = []
        if temp is not None and temp >= COOL_TEMP_C:
            reason.append(f"GPU {temp} °C")
        if avail < MIN_AVAIL_GB:
            reason.append(f"{avail:.0f} GiB avail")
        print(
            f"{_ts()} λ={lam}  waiting ({', '.join(reason)})…",
            flush=True,
        )
        time.sleep(60)


# ── Pre-flight checks ────────────────────────────────────────────────────


def preflight() -> None:
    """Verify <compute-node> prerequisites; exit(1) if any check fails."""
    print(f"{_ts()} pre-flight checks …", flush=True)

    # 1. swap must be off (only header line present)
    swap = Path("/proc/swaps").read_text().strip().split("\n")
    if len(swap) > 1:
        print(f"ERROR: swap active ({len(swap) - 1} entries); aborting.", flush=True)
        sys.exit(1)
    print("  ✓ swapoff", flush=True)

    # 2. vllm must not be running (systemd service OR docker container)
    res = subprocess.run(
        ["systemctl", "is-active", "vllm.service"],
        capture_output=True, text=True, timeout=10,
    )
    if res.stdout.strip() == "active":
        print("ERROR: vllm.service is active; aborting.", flush=True)
        sys.exit(1)
    res_docker = subprocess.run(
        ["docker", "ps", "--filter", "name=vllm", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=10,
    )
    if res_docker.stdout.strip():
        print(f"ERROR: vLLM docker container running ({res_docker.stdout.strip()}); aborting.", flush=True)
        sys.exit(1)
    print("  ✓ vllm inactive", flush=True)

    # 3. disk free > 50 GB
    usage = shutil.disk_usage(PROJECT)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 50:
        print(f"ERROR: only {free_gb:.1f} GB free (< 50 GB); aborting.", flush=True)
        sys.exit(1)
    print(f"  ✓ disk free: {free_gb:.1f} GB", flush=True)

    # 4. GPU temperature: wait until cool enough (instead of aborting on a warm
    #    GPU left over from a previous run).
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
            print(
                f"ERROR: GPU still {temp} °C (>= {COOL_TEMP_C} °C) after "
                f"{TEMP_WAIT_MAX_S // 60} min of waiting; aborting.",
                flush=True,
            )
            sys.exit(1)
        print(f"  … GPU {temp} °C >= {COOL_TEMP_C} °C, cooling… ", flush=True)
        time.sleep(60)

    print(f"{_ts()} all pre-flight checks passed.\n", flush=True)


# ── Directory helpers ────────────────────────────────────────────────────


def find_save_dir(lam: float) -> Path | None:
    """Return the most recent save directory for *lam*, or *None*."""
    pattern = str(PROJECT / "save" / f"caft_gf_lam{lam}_seed{SEED}_*")
    dirs = sorted(glob(pattern))
    return Path(dirs[-1]) if dirs else None


def find_checkpoint(save_dir: Path) -> Path | None:
    """Return the highest-numbered checkpoint inside ``model/``."""
    ckpts = sorted(
        glob(str(save_dir / "model" / "checkpoint-*")),
        key=lambda p: int(p.rsplit("checkpoint-", 1)[-1]),
    )
    return Path(ckpts[-1]) if ckpts else None


# ── Task runners ─────────────────────────────────────────────────────────


def run_caft(lam: float) -> Path | None:
    """Run ``scripts/caft_gf.py`` for *lam*, return the save dir on success.

    caft's stdout/stderr are streamed LIVE to ``artifacts/lambda_grid/caft_lam{lam}.log``
    (instead of being captured in memory) so training progress — the tqdm
    ``N/2500`` step line and the per-10-step loss logs — is observable in real
    time. ``-u`` forces unbuffered output so step lines land in the log promptly
    rather than sitting in a block buffer (the earlier "is it alive?" confusion).
    """
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT / "scripts" / "caft_gf.py"),
        "--caft_lambda", str(lam),
        "--data_file", "corneal_100k_subsample.h5ad",
        "--epochs", "3",
        "--seed", str(SEED),
    ]
    print(f"{_ts()} λ={lam}  CAFT starting …", flush=True)
    log_path = RESULTS_DIR / f"caft_lam{lam}.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=172800)  # 48h — <compute-node> trains ~11-13h/λ
    wall = (time.time() - t0) / 60

    if proc.returncode != 0:
        print(
            f"{_ts()} λ={lam}  CAFT FAILED (rc={proc.returncode}, {wall:.1f} min); log: {log_path}",
            flush=True,
        )
        try:
            tail = log_path.read_text()[-2000:]
        except OSError:
            tail = ""
        if tail:
            print(tail, flush=True)
        return None

    save_dir = find_save_dir(lam)
    if save_dir is None:
        print(f"{_ts()} λ={lam}  no save dir found after training!", flush=True)
        return None

    if not (save_dir / "CAFT_DONE").exists():
        print(f"{_ts()} λ={lam}  CAFT_DONE marker missing; incomplete.", flush=True)
        return None

    print(f"{_ts()} λ={lam}  CAFT done ({wall:.1f} min)  → {save_dir.name}", flush=True)
    return save_dir


def run_gpr(ckpt: Path, out_json: Path) -> dict[str, Any] | None:
    """Run ``corneal_fm.gpr_cli`` (paired mode), return the primary result dict."""
    cmd = [
        sys.executable, "-m", "corneal_fm.gpr_cli",
        "--before_checkpoint", str(PRETRAINED),
        "--after_checkpoint", str(ckpt),
        "--tokenized", str(TOKENIZED_30K),
        "--num_classes", "31",
        "--n_boot", "1000",
        "--seed", str(SEED),
        "--out", str(out_json),
    ]
    ckpt_rel = f"{ckpt.parent.parent.parent.name}/{ckpt.parent.name}/{ckpt.name}"
    print(f"{_ts()} GPR  measuring {ckpt_rel} …", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # GPR ~2500s on a free GPU; 2h margin
    wall = (time.time() - t0) / 60

    if res.returncode != 0:
        print(f"{_ts()} GPR  FAILED (rc={res.returncode}, {wall:.1f} min)", flush=True)
        if res.stderr:
            print(res.stderr[-2000:], flush=True)
        return None

    if not out_json.exists():
        print(f"{_ts()} GPR  output not found: {out_json}", flush=True)
        return None

    data = json.loads(out_json.read_text())
    primary = data.get("primary")
    if primary is None:
        print(f"{_ts()} GPR  output missing 'primary' field.", flush=True)
        return None

    print(
        f"{_ts()} GPR  done ({wall:.1f} min)  "
        f"cs_after={primary['cs_after']:.4f}  "
        f"gpr={primary['gpr']:.4f}",
        flush=True,
    )
    return primary


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    preflight()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results for idempotency
    done: set[float] = set()
    if RESULTS_TSV.exists():
        for line in RESULTS_TSV.read_text().strip().split("\n")[1:]:
            if line.strip():
                done.add(float(line.split("\t")[0]))
        print(f"{_ts()} existing results for λ: {sorted(done)}\n", flush=True)

    for lam in LAMBDA_GRID:
        if lam in done:
            print(f"{_ts()} λ={lam}  already in results.tsv → skip", flush=True)
            continue

        # Guard: do not start a run while the GPU is hot or memory is tight.
        wait_until_ready(lam)

        t_lambda = time.time()

        # — Train ─────────────────────────────────────────────────────────
        save_dir = run_caft(lam)
        if save_dir is None:
            print(f"{_ts()} λ={lam}  skipped after CAFT failure.\n", flush=True)
            continue

        # — Locate final checkpoint ───────────────────────────────────────
        ckpt = find_checkpoint(save_dir)
        if ckpt is None:
            print(f"{_ts()} λ={lam}  no checkpoint found; skipped.\n", flush=True)
            continue

        # — Measure GPR ───────────────────────────────────────────────────
        out_json = RESULTS_DIR / f"gpr_lam{lam}.json"
        primary = run_gpr(ckpt, out_json)
        if primary is None:
            print(f"{_ts()} λ={lam}  skipped after GPR failure.\n", flush=True)
            continue

        # — Extract 95 % CI ──────────────────────────────────────────────
        ci = primary.get("paired_stratified_bootstrap_95ci", {})
        gpr_ci = ci.get("gpr", {})

        # — Append result row ─────────────────────────────────────────────
        elapsed_min = round((time.time() - t_lambda) / 60, 2)
        row = (
            f"{lam}\t"
            f"{primary['cs_after']}\t"
            f"{primary['gpr']}\t"
            f"{gpr_ci.get('ci_low', '')}\t"
            f"{gpr_ci.get('ci_high', '')}\t"
            f"{elapsed_min}\t"
            f"{save_dir.name}\t"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}\n"
        )

        if not RESULTS_TSV.exists():
            RESULTS_TSV.write_text(TSV_HEADER)
        with open(RESULTS_TSV, "a") as fh:
            fh.write(row)
        print(f"{_ts()} λ={lam}  recorded to results.tsv\n", flush=True)

    # — Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"All done. Results: {RESULTS_TSV}")
    print(f"{'=' * 60}")
    if RESULTS_TSV.exists():
        sys.stdout.write(RESULTS_TSV.read_text())
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
