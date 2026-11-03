#!/usr/bin/env python
"""Cornea2 scGPT driver — frozen eval -> 2 seeds x 15 epochs per dataset.

Mirrors run_e3e4.py: GPU guard (free >= 4 GiB, wait up to 12 h — E3/E4
occupy <compute-node> until tomorrow morning), OOM exit auto-retry (3 attempts),
idempotent markers, detached-friendly (setsid nohup via the launch command
below).

Sequence per dataset: frozen baseline eval -> naive seed 42 -> naive seed
123 -> eval (best-val + final-epoch) -> summary append. Collapse is
reported from trajectory + eval json per seed.

Launch (on <compute-node>):
    setsid nohup repo/.venv/bin/python scripts/run_cornea2_driver.py \
        > artifacts/paper1_vanzyl/cornea2/driver.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / "repo" / ".venv" / "bin" / "python"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
OUT = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cornea2"
LOG_PATH = OUT / "driver.log"

DATASETS = ["GSE186433", "GSE155683"]
SEEDS = [42, 123]
MAX_OOM_RETRIES = 3
REQUIRE_FREE_MIB = 4096
MAX_WAIT_MIN = 720  # 12 h wait budget (E3/E4 until tomorrow morning)


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


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


def gpu_guard(require_free_mib: int = REQUIRE_FREE_MIB,
              max_wait_min: int = MAX_WAIT_MIN) -> bool:
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
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== cornea2 driver start (python={PYTHON}) ===")

    for ds in DATASETS:
        tag = ds.lower()
        frz_out = OUT / f"{tag}_frozen_baseline.json"
        if not run_done(frz_out):
            if not gpu_guard():
                return 3
            log(f"=== frozen eval {ds} ===")
            ok = run_with_retry(
                [str(PYTHON), str(SCRIPT_DIR / "run_cornea2_frozen_eval.py"),
                 "--dataset", ds],
                OUT / f"{tag}_frozen_run.log", f"frozen {ds}")
            if not ok:
                return 1
        frz = json.loads(frz_out.read_text())
        frozen_rho = frz["rho"]["rho_normed"]
        log(f"frozen {ds}: rho_normed={frozen_rho:.4f} "
            f"z={frz['rho']['null_z']:.1f}")

        # GPU dry-run smoke (torch 2.13 CPU path broken on CUDA-visible host)
        smoke_marker = OUT / f"{tag}_smoke_done"
        if not run_done(smoke_marker):
            if not gpu_guard():
                return 3
            log(f"=== dry-run smoke {ds} ===")
            ok = run_with_retry(
                [str(PYTHON), str(SCRIPT_DIR / "scgpt_cls_cornea2.py"),
                 "--dataset", ds, "--seed", "42", "--frozen-rho",
                 str(frozen_rho), "--dry-run", "--max-cells", "300"],
                OUT / f"{tag}_smoke.log", f"smoke {ds}")
            if not ok:
                return 1
            smoke_marker.write_text("DONE\n", encoding="utf-8")

        for seed in SEEDS:
            run_dir = OUT / ds / f"naive_seed{seed}"
            marker = run_dir / "CORNEATRAIN_DONE"
            final_ckpt = run_dir / "model_epoch15.pt"
            if run_done(marker) and final_ckpt.is_file():
                log(f"SKIP train {ds} seed{seed}: already DONE "
                    f"(15 epochs complete)")
            else:
                if run_done(marker):
                    n_ep = len(list(run_dir.glob("model_epoch*.pt")))
                    log(f"RE-TRAIN {ds} seed{seed}: marker present but "
                        f"model_epoch15.pt missing ({n_ep} epochs found; "
                        "incomplete run)")
                if not gpu_guard():
                    return 3
                log(f"=== train {ds} naive seed{seed} (15 epochs) ===")
                run_dir.mkdir(parents=True, exist_ok=True)
                ok = run_with_retry(
                    [str(PYTHON), str(SCRIPT_DIR / "scgpt_cls_cornea2.py"),
                     "--dataset", ds, "--seed", str(seed),
                     "--frozen-rho", str(frozen_rho)],
                    run_dir / "run.log", f"train {ds} seed{seed}")
                if not ok:
                    return 1
            ev_out = run_dir / "eval.json"
            ev_stale = (not run_done(ev_out)) or (
                final_ckpt.is_file()
                and ev_out.stat().st_mtime < final_ckpt.stat().st_mtime)
            if ev_stale:
                if run_done(ev_out):
                    log(f"RE-EVAL {ds} seed{seed}: eval.json older than "
                        "model_epoch15.pt (stale after re-train)")
                if not gpu_guard():
                    return 3
                log(f"=== eval {ds} seed{seed} (best-val + final-epoch) ===")
                ok = run_with_retry(
                    [str(PYTHON), str(SCRIPT_DIR / "run_cornea2_eval.py"),
                     "--dataset", ds, "--seed", str(seed)],
                    run_dir / "eval.log", f"eval {ds} seed{seed}")
                if not ok:
                    return 1

    # summary
    summary = {"timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
               "datasets": {}}
    for ds in DATASETS:
        tag = ds.lower()
        frz = json.loads((OUT / f"{tag}_frozen_baseline.json").read_text())
        entry = {"frozen": frz["rho"]["rho_normed"],
                 "frozen_z": frz["rho"]["null_z"],
                 "probe_f1_frozen": frz["probe_f1"]["f1_macro"],
                 "seeds": {}}
        for seed in SEEDS:
            ev = json.loads(
                (OUT / ds / f"naive_seed{seed}" / "eval.json").read_text())
            def _safe(ev_: dict, ck: str, k: str):
                c = ev_.get(ck)
                return c.get(k) if isinstance(c, dict) else None

            entry["seeds"][f"s{seed}"] = {
                "best_val": {
                    "rho_normed": _safe(ev, "best_val", "rho_normed"),
                    "null_z": _safe(ev, "best_val", "null_z"),
                    "macro_f1": _safe(ev, "best_val", "macro_f1"),
                    "ci95": _safe(ev, "best_val", "bootstrap_ci95"),
                },
                "final_epoch": {
                    "rho_normed": _safe(ev, "final_epoch", "rho_normed"),
                    "null_z": _safe(ev, "final_epoch", "null_z"),
                    "macro_f1": _safe(ev, "final_epoch", "macro_f1"),
                    "ci95": _safe(ev, "final_epoch", "bootstrap_ci95"),
                },
            }
        summary["datasets"][ds] = entry
        log(f"SUMMARY {ds}: frozen rho={entry['frozen']:.4f} | "
            + " | ".join(
                f"s{seed} bv_rho={entry['seeds'][f's{seed}']['best_val']['rho_normed']:.4f}"
                for seed in SEEDS))
    (OUT / "summary_cornea2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log("=== cornea2 driver DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
