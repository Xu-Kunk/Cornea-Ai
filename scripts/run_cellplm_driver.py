#!/usr/bin/env python
"""CellPLM arm C3 driver: 3-seed naive-CE fine-tune + PCS eval (GPU guarded).

Sequential seeds 42 / 123 / 456; each seed waits for GPU free >= 4 GiB
(poll 60 s, budget 5 h). Appends lane.log event blocks. Aggregates
per-seed jsons into summary_vanzyl_cellplm.json with mean +/- sd.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("<local>/")
RUNNER = PROJECT_ROOT / "scripts" / "run_cellplm_vanzyl.py"
OUT_ROOT = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm"
LANE_LOG = OUT_ROOT / "lane.log"
SEEDS = [42, 123, 456]
PY = "<repo>/.venv-cellplm/bin/python"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def lane_event(kind: str, msg: str) -> None:
    line = json.dumps({"ts": now_utc(), "event": kind, "msg": msg}, ensure_ascii=False)
    print(line, flush=True)
    with open(LANE_LOG, "a") as f:
        f.write(line + "\n")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lane_event("START", "CellPLM arm C3: 3-seed naive CE, test donor Hu0216, seeds 42/123/456")

    per_seed = {}
    for seed in SEEDS:
        lane_event("SEED_START", f"seed {seed}")
        t0 = time.time()
        proc, rc = None, -1
        for attempt in range(1, 4):  # OOM/transient failure retry (max 3)
            proc = subprocess.run(
                [PY, str(RUNNER), "--seed", str(seed), "--out-root", str(OUT_ROOT), "--device", "cuda"],
                capture_output=True, text=True,
            )
            rc = proc.returncode
            if rc == 0:
                break
            lane_event("SEED_RETRY", f"seed {seed} attempt {attempt} rc={rc}; backoff 120 s")
            time.sleep(120)
        el = (time.time() - t0) / 60
        if rc != 0:
            lane_event("SEED_FAIL", f"seed {seed} rc={rc} after {el:.1f} min (3 attempts)")
            print("STDOUT:\n", proc.stdout[-4000:])
            print("STDERR:\n", proc.stderr[-4000:])
            return 1
        res = json.loads((OUT_ROOT / f"seed{seed}" / f"seed{seed}.json").read_text())
        per_seed[str(seed)] = res
        lane_event(
            "SEED_DONE",
            f"seed {seed}: F1={res['eval']['macro_f1']:.4f} CI=[{res['eval']['bootstrap_ci95'][0]:.4f},"
            f"{res['eval']['bootstrap_ci95'][1]:.4f}] rho_n={res['eval']['rho_normed']:.4f} "
            f"z={res['eval']['null_z']:.1f} epochs={res['training']['epochs_actual']} ({el:.1f} min)",
        )

    # ---------- summary ----------
    f1 = [per_seed[s]["eval"]["macro_f1"] for s in per_seed]
    rho = [per_seed[s]["eval"]["rho_normed"] for s in per_seed]
    z = [per_seed[s]["eval"]["null_z"] for s in per_seed]
    rho_frz = [per_seed[s]["frozen_baseline"]["rho_normed"] for s in per_seed]
    summary = {
        "arm": "CellPLM (cell-as-token, 85M) naive CE — third architecture cross-collapse check",
        "recipe": "official CellTypeAnnotationPipeline, out_dim=3, lr 5e-3 (embedder 5e-4), wd 1e-7, "
        "plat scheduler, epochs 2000 cap, es=200, patience 25, hvg 3000, best-valid-F1 ckpt",
        "eval_set": "sanes2022_Hu0216 (20,158 cells, LOCKED PREREG §8)",
        "seeds": SEEDS,
        "per_seed": {s: {"macro_f1": per_seed[s]["eval"]["macro_f1"],
                         "ci95": per_seed[s]["eval"]["bootstrap_ci95"],
                         "rho_normed": per_seed[s]["eval"]["rho_normed"],
                         "rho_raw": per_seed[s]["eval"]["rho_raw"],
                         "null_z": per_seed[s]["eval"]["null_z"],
                         "epochs": per_seed[s]["training"]["epochs_actual"],
                         "best_valid_f1": per_seed[s]["training"]["best_valid_f1"],
                         "frozen_rho_normed": per_seed[s]["frozen_baseline"]["rho_normed"],
                         "frozen_z": per_seed[s]["frozen_baseline"]["null_z"]}
                      for s in per_seed},
        "mean_sd": {
            "macro_f1": [float(np.mean(f1)), float(np.std(f1, ddof=1))],
            "rho_normed": [float(np.mean(rho)), float(np.std(rho, ddof=1))],
            "null_z": [float(np.mean(z)), float(np.std(z, ddof=1))],
            "frozen_rho_normed": [float(np.mean(rho_frz)), float(np.std(rho_frz, ddof=1))],
        },
        "comparison_arms": {
            "gf_frozen_rho_normed": 0.8292,
            "scgpt_frozen_rho_normed": 0.7887,
            "scgpt_naive_rho_normed_mean": 0.5554,
            "scgpt_naive_macro_f1_mean": 0.9226,
        },
        "timestamp_utc": now_utc(),
    }
    (OUT_ROOT / "summary_vanzyl_cellplm.json").write_text(json.dumps(summary, indent=2))
    lane_event("DONE", f"summary written: F1 mean={np.mean(f1):.4f}±{np.std(f1, ddof=1):.4f} "
                       f"rho_n mean={np.mean(rho):.4f}±{np.std(rho, ddof=1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
