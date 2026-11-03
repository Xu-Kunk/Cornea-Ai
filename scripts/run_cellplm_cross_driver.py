#!/usr/bin/env python
"""CellPLM consistency driver: 2 arms x 2 seeds naive-CE fine-tune + PCS
eval (approved supplementary).

Arms (identical cells to the scGPT cross-tissue / LOO arms):
  * epidermis:  data/epidermis_cellplm.h5ad   (4-layer axis, out_dim 4,
                 test = frozen 10% stratified hold-out)
  * loo220:     data/vanzyl_cellplm_loo220.h5ad (3-tier axis, out_dim 3,
                 test donor sanes2022_Hu0220)

Sequential seeds 42 / 123 per arm; each seed waits for GPU free >= 4 GiB
(poll 60 s, budget 12 h); OOM/transient failure retried (max 3 attempts,
120 s backoff). Idempotent: seed DONE marker skips. Appends lane.log event
blocks; aggregates per-seed jsons into summary_{arm}.json (mean +/- sd).

Runs detached on <compute-node> (setsid nohup, driver rc recorded).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("<home>/cellplm_vanzyl")
RUNNER = PROJECT_ROOT / "scripts" / "run_cellplm_cross.py"
CKPT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "ckpt"
OUT_ROOT = Path("<home>/corneal_scgpt") / "artifacts" / "paper1_vanzyl" / "cellplm_consistency"
LANE_LOG = OUT_ROOT / "lane.log"
PY = "<home>/corneal_scgpt/venv/bin/python"

ARMS = [
    {
        "name": "epidermis",
        "data": "<home>/corneal_scgpt/data/epidermis_cellplm.h5ad",
        "out_dim": 4,
        "layer_order": "basal,suprabasal,spinous,differentiated",
        "test_donor": "epidermis",
        "seeds": [42, 123],
    },
    {
        "name": "loo220",
        "data": "<home>/corneal_scgpt/data/vanzyl_cellplm_loo220.h5ad",
        "out_dim": 3,
        "layer_order": "LSC,basal,differentiated",
        "test_donor": "sanes2022_Hu0220",
        "seeds": [42, 123],
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def lane_event(kind: str, msg: str) -> None:
    line = json.dumps({"ts": now_utc(), "event": kind, "msg": msg},
                      ensure_ascii=False)
    print(line, flush=True)
    with open(LANE_LOG, "a") as f:
        f.write(line + "\n")


def seed_done(arm_out: Path, seed: int) -> bool:
    return (arm_out / f"seed{seed}" / "DONE").is_file()


def summary_for(arm: dict, per_seed: dict) -> dict:
    rho = [per_seed[s]["eval"]["rho_normed"] for s in per_seed]
    rho_raw = [per_seed[s]["eval"]["rho_raw"] for s in per_seed]
    z = [per_seed[s]["eval"]["null_z"] for s in per_seed]
    f1 = [per_seed[s]["eval"]["macro_f1"] for s in per_seed]
    frz = [per_seed[s]["frozen_baseline"]["rho_normed"] for s in per_seed]
    epochs = [per_seed[s]["training"]["epochs_actual"] for s in per_seed]
    delta = [per_seed[s].get("delta_rho", {}).get("delta_rho") for s in per_seed]
    collapse_rate = sum(1 for d in delta if d is not None and d > 0.15) / len(delta)
    return {
        "arm": f"CellPLM consistency: {arm['name']}",
        "test_donor": arm["test_donor"],
        "layer_order": arm["layer_order"].split(","),
        "seeds": sorted(per_seed.keys()),
        "rho_normed": {
            "mean": float(np.mean(rho)), "sd": float(np.std(rho, ddof=1))
            if len(rho) > 1 else 0.0,
            "per_seed": {str(s): float(rho[i]) for i, s in enumerate(per_seed)},
        },
        "rho_raw_mean": float(np.mean(rho_raw)),
        "null_z_mean": float(np.mean(z)),
        "macro_f1": {
            "mean": float(np.mean(f1)), "sd": float(np.std(f1, ddof=1))
            if len(f1) > 1 else 0.0,
            "per_seed": {str(s): float(f1[i]) for i, s in enumerate(per_seed)},
        },
        "frozen_rho_mean": float(np.mean(frz)),
        "delta_rho_per_seed": {str(s): (float(d) if d is not None else None)
                               for s, d in zip(per_seed, delta)},
        "collapse_rate_delta_gt_0.15": round(collapse_rate, 4),
        "epochs_actual": {str(s): int(e) for s, e in zip(per_seed, epochs)},
        "timestamp_utc": now_utc(),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lane_event("START", "CellPLM consistency: epidermis + LOO(Hu0220), "
                        "2 seeds each (42/123)")
    all_summaries = {}
    for arm in ARMS:
        arm_out = OUT_ROOT / arm["name"]
        arm_out.mkdir(parents=True, exist_ok=True)
        lane_event("ARM_START", f"{arm['name']}: test_donor={arm['test_donor']} "
                                f"out_dim={arm['out_dim']} "
                                f"layer_order={arm['layer_order']}")
        per_seed: dict = {}
        for seed in arm["seeds"]:
            if seed_done(arm_out, seed):
                lane_event("SEED_SKIP", f"{arm['name']} seed {seed}: DONE marker")
                per_seed[str(seed)] = json.loads(
                    (arm_out / f"seed{seed}" / f"seed{seed}.json").read_text())
                continue
            lane_event("SEED_START", f"{arm['name']} seed {seed}")
            t0 = time.time()
            proc, rc = None, -1
            for attempt in range(1, 4):  # OOM/transient retry (max 3)
                proc = subprocess.run(
                    [PY, str(RUNNER), "--seed", str(seed),
                     "--data", arm["data"],
                     "--ckpt", str(CKPT_DIR),
                     "--out-root", str(arm_out),
                     "--out-dim", str(arm["out_dim"]),
                     "--layer-order", arm["layer_order"],
                     "--test-donor", arm["test_donor"],
                     "--device", "cuda"],
                    capture_output=True, text=True,
                )
                rc = proc.returncode
                if rc == 0:
                    break
                lane_event("SEED_RETRY", f"{arm['name']} seed {seed} attempt "
                                         f"{attempt} rc={rc}; backoff 120 s")
                time.sleep(120)
            el = (time.time() - t0) / 60
            if rc != 0:
                lane_event("SEED_FAIL", f"{arm['name']} seed {seed} rc={rc} "
                                        f"after {el:.1f} min (3 attempts)")
                print("STDOUT:\n", proc.stdout[-4000:])
                print("STDERR:\n", proc.stderr[-4000:])
                return 1
            res = json.loads(
                (arm_out / f"seed{seed}" / f"seed{seed}.json").read_text())
            per_seed[str(seed)] = res
            lane_event(
                "SEED_DONE",
                f"{arm['name']} seed {seed}: F1={res['eval']['macro_f1']:.4f} "
                f"CI=[{res['eval']['bootstrap_ci95'][0]:.4f},"
                f"{res['eval']['bootstrap_ci95'][1]:.4f}] "
                f"rho_n={res['eval']['rho_normed']:.4f} "
                f"z={res['eval']['null_z']:.1f} "
                f"epochs={res['training']['epochs_actual']} ({el:.1f} min)",
            )
        summ = summary_for(arm, per_seed)
        all_summaries[arm["name"]] = summ
        (OUT_ROOT / f"summary_{arm['name']}.json").write_text(
            json.dumps(summ, indent=2), encoding="utf-8")
        lane_event("ARM_DONE", f"{arm['name']}: {json.dumps(summ, ensure_ascii=False)}")

    (OUT_ROOT / "summary_all.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8")
    lane_event("DONE", "CellPLM consistency driver finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
