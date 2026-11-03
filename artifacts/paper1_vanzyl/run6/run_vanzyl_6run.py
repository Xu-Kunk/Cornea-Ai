#!/usr/bin/env python
"""van Zyl 6-run idempotent runner (PREREG-2026-08-11 §8, LOCKED).

Sequentially executes the 6 locked runs:
    naive:{42,123,456}, ordinal:{42,123,456} (arm x seed order per §8 listing).

Idempotency: a run is skipped when its run_dir contains the VANZYL_RUN_DONE
marker (or the training script's own runs_status DONE). Interrupted runs
resume from the first non-DONE run. Fail-fast: any non-zero exit stops the
runner (safe to re-launch later).

Each run:
    python scripts/scgpt_cls_vanzyl.py --arm <arm> --seed <seed>
        --out artifacts/paper1_vanzyl/run6/checkpoints
    stdout/stderr -> <run_dir>/run.log
    artifacts     -> <run_dir>/{model.pt, args.json, runs_status, train_log.jsonl}
    runner marks  -> <run_dir>/VANZYL_RUN_DONE (after success)

GPU guard: waits (up to --max-gpu-wait-min) for the GPU to be ~free before
each run, so a parallel lane cannot corrupt the runs.
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
TRAIN_SCRIPT = ROOT / "scripts" / "scgpt_cls_vanzyl.py"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "run6"
CHECKPOINTS = OUT / "checkpoints"
MANIFEST = OUT / "split_manifest.json"
RUNS = [("naive", 42), ("naive", 123), ("naive", 456),
        ("ordinal", 42), ("ordinal", 123), ("ordinal", 456)]
LOCKED_N_TRAIN = 42_482


def log(msg: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with (OUT / "runner.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def gpu_free_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def run_done(run_dir: Path) -> bool:
    if (run_dir / "VANZYL_RUN_DONE").is_file():
        return True
    rs = run_dir / "runs_status"
    if rs.is_file() and "DONE" in rs.read_text(encoding="utf-8"):
        # only trust a REAL (non dry-run) completion; a smoke dry-run writes
        # runs_status DONE with args.json dry_run=true and must be redone
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
    sp = OUT / "runner_summary.json"
    data = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {"runs": {}}
    data["runs"][f"{entry['arm']}_{entry['seed']}"] = entry
    data["updated_utc"] = now_utc()
    sp.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-gpu-wait-min", type=int, default=30)
    ap.add_argument("--arm", choices=["naive", "ordinal"], default=None,
                    help="restrict to one arm (resume aid); default all")
    ap.add_argument("--seed", type=int, default=None, help="restrict to one seed")
    args = ap.parse_args()

    if not MANIFEST.is_file():
        log(f"FATAL: split manifest missing: {MANIFEST} (lock requirement)")
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("n_train_cells") != LOCKED_N_TRAIN:
        log(f"FATAL: manifest n_train {manifest.get('n_train_cells')} != "
            f"locked {LOCKED_N_TRAIN}")
        return 2
    if not TRAIN_SCRIPT.is_file():
        log(f"FATAL: train script missing: {TRAIN_SCRIPT}")
        return 2
    log(f"manifest OK: test_donor={manifest['test_donor']} "
        f"n_train={manifest['n_train_cells']} n_test={manifest['n_test_cells']}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    selected = [(a, s) for a, s in RUNS
                if (args.arm is None or a == args.arm)
                and (args.seed is None or s == args.seed)]
    log(f"plan: {len(selected)} runs -> {selected}")

    for arm, seed in selected:
        run_dir = CHECKPOINTS / f"{arm}_seed{seed}"
        if run_done(run_dir):
            log(f"SKIP {arm} seed{seed}: already DONE")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== START {arm} seed{seed} ===")
        # GPU guard
        waited = 0
        while True:
            used = gpu_free_mib()
            if used < 0 or used <= 1024:
                break
            if waited >= args.max_gpu_wait_min * 60:
                log(f"ABORT {arm} seed{seed}: GPU busy {used} MiB after "
                    f"{args.max_gpu_wait_min} min wait")
                return 3
            log(f"GPU busy ({used} MiB); wait 60s ({waited // 60} min)")
            time.sleep(60)
            waited += 60
        run_log = run_dir / "run.log"
        cmd = [str(PYTHON), str(TRAIN_SCRIPT), "--arm", arm, "--seed", str(seed),
               "--out", str(CHECKPOINTS)]
        t0 = time.time()
        with run_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  text=True)
        wall_min = (time.time() - t0) / 60.0
        if proc.returncode != 0:
            log(f"FAIL {arm} seed{seed}: exit={proc.returncode} "
                f"after {wall_min:.1f} min (see {run_log}); runner stops")
            summary_update({"arm": arm, "seed": seed, "status": "FAILED",
                            "exit": proc.returncode, "wall_min": round(wall_min, 1),
                            "ts": now_utc()})
            return 1
        ckpt = run_dir / "model.pt"
        if not ckpt.is_file():
            log(f"FAIL {arm} seed{seed}: exit=0 but model.pt missing; runner stops")
            return 1
        (run_dir / "VANZYL_RUN_DONE").write_text(
            f"DONE\n{now_utc()}\narm={arm} seed={seed}\n", encoding="utf-8")
        log(f"DONE {arm} seed{seed}: {wall_min:.1f} min, checkpoint {ckpt}")
        summary_update({"arm": arm, "seed": seed, "status": "DONE",
                        "wall_min": round(wall_min, 1), "checkpoint": str(ckpt),
                        "ts": now_utc()})

    log("runner finished (all selected runs processed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
