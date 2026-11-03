#!/usr/bin/env python
"""Head-init perturbation: final test-donor evaluation (FROZEN 2026-08-20).

Evaluates the 3 head-init variants (naive_seed42_headscale_0.01 /
headscale_1.0 / headseed_999) x 2 checkpoints (best-val by val macro-F1,
final-epoch model_epoch15.pt) on the Hu0216 test donor (20,158 cells):
macro-F1 + cell-level bootstrap 95% CI (n_boot=1000, seed=42),
rho(L2-normed) + rho(raw) + label-shuffle z (n=100).

Metric code imported UNCHANGED from scgpt_eval_vanzyl_convergence.py
(macro_f1_ci, rho_and_null) / the 6-run eval suite — no metric code rewritten.

Frozen judgment (basin-selection test): a variant "keeps" when its final-epoch
test rho_normed >= 0.6547 (the locked §9 collapse threshold); "collapses"
otherwise. Reference: seed42 collapses (final -0.091), seed123/456 keep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scgpt_cls  # noqa: E402
from scgpt_eval_vanzyl_convergence import (  # noqa: E402  (locked eval machinery)
    LAYER_ORDER,
    TEST_DONOR,
    TEST_H5AD,
    TRAIN_H5AD,
    macro_f1_ci,
    rho_and_null,
    naive_mapping_90pct,
    eval_checkpoint,
)

HEADINIT_CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "head_init" / "checkpoints"
OUT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "head_init"
SCGPT_PRETRAINED = ROOT / "pretrained" / "scgpt_pretrained"
SIXRUN_FROZEN = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval" / "frozen_baseline.json"
CONV_FINAL = ROOT / "artifacts" / "paper1_vanzyl" / "convergence" / "final_test_eval.json"

VARIANTS = ["headscale_0.01", "headscale_1.0", "headseed_999"]
CKPT_TYPES = [("best_val", "model_best_val.pt"),
              ("final_epoch", "model_epoch15.pt")]
COLLAPSE_THRESHOLD = 0.6547  # LOCKED §9 (frozen baseline 0.8047 - 0.15)


def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def gpu_free_mib() -> int:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        used, total = (int(x) for x in out.stdout.strip().split(","))
        return total - used
    except Exception:
        return -1


def main() -> int:
    import subprocess, time as _t
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # GPU guard: free memory >= 4 GiB (parallel lanes may run on the same GPU)
    waited = 0
    while True:
        free = gpu_free_mib()
        if free >= 4096:
            break
        if waited >= 300 * 60:
            log(f"ABORT: GPU free {free} MiB < 4096 after 300 min")
            return 3
        log(f"GPU free {free} MiB < 4096; wait 60s ({waited // 60} min)")
        _t.sleep(60)
        waited += 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, pargs = scgpt_cls.load_vocab(pretrained_dir=SCGPT_PRETRAINED)
    log(f"device={device} torch={torch.__version__}")

    import anndata as ad
    a = ad.read_h5ad(TEST_H5AD, backed="r")
    y_true = np.asarray(a.obs["cell_class_4"].astype(str))
    if not (a.obs["donor_id"].astype(str) == TEST_DONOR).all():
        raise RuntimeError("test h5ad not exclusively the locked test donor")
    log(f"test: {a.shape}, y dist={dict(Counter(y_true.tolist()))}")

    naive_classes, naive_map, naive_cont = naive_mapping_90pct()
    log(f"train-90% naive classes: {len(naive_classes)}")

    frozen = json.loads(SIXRUN_FROZEN.read_text(encoding="utf-8"))
    frz_rho = frozen["rho"]["rho_normed"]
    frz_z = frozen["rho"]["null_z"]
    frz_f1 = frozen["probe_f1"]["f1_macro"]
    log(f"frozen baseline (test donor): rho_n={frz_rho:.4f} z={frz_z:.1f} "
        f"probe_f1={frz_f1:.4f}")

    conv_ref = json.loads(CONV_FINAL.read_text(encoding="utf-8"))
    ref = {
        "s42": conv_ref["per_seed_per_ckpt"]["s42_final_epoch"],
        "s123": conv_ref["per_seed_per_ckpt"]["s123_final_epoch"],
        "s456": conv_ref["per_seed_per_ckpt"]["s456_final_epoch"],
    }
    log(f"reference (convergence final-epoch): s42 rho_n="
        f"{ref['s42']['rho_normed']:.4f} s123={ref['s123']['rho_normed']:.4f} "
        f"s456={ref['s456']['rho_normed']:.4f}")

    results: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        cdir = HEADINIT_CKPT / f"naive_seed42_{variant}"
        args_json = cdir / "args.json"
        for ctype, fname in CKPT_TYPES:
            ckpt = cdir / fname
            if not ckpt.is_file():
                log(f"SKIP {variant} {ctype}: {fname} missing")
                continue
            log(f"=== eval {variant} {ctype} ===")
            t0 = time.time()
            r = eval_checkpoint(ckpt, args_json, vocab, pargs, device,
                                y_true, naive_classes, naive_map)
            log(f"  F1={r['macro_f1']['macro_f1']:.4f} "
                f"CI=[{r['macro_f1']['ci95_low']:.4f},{r['macro_f1']['ci95_high']:.4f}] "
                f"rho_n={r['rho_normed']:.4f} z={r['null_z']:.1f} "
                f"({time.time() - t0:.0f}s)")
            results[f"{variant}_{ctype}"] = r
            # per-variant json
            (OUT_DIR / f"{variant}_{ctype}.json").write_text(
                json.dumps(r, indent=2), encoding="utf-8")

    # basin-selection judgment (frozen): final-epoch rho vs 0.6547
    judgment: dict[str, Any] = {}
    for variant in VARIANTS:
        key = f"{variant}_final_epoch"
        if key not in results:
            judgment[variant] = {"status": "missing_checkpoint"}
            continue
        rho_n = results[key]["rho_normed"]
        kept = bool(rho_n >= COLLAPSE_THRESHOLD)
        judgment[variant] = {
            "final_rho_normed": rho_n,
            "threshold": COLLAPSE_THRESHOLD,
            "basin": "kept" if kept else "collapsed",
            "basin_selection_supported": kept,
            "best_val_rho_normed": results.get(f"{variant}_best_val", {}).get(
                "rho_normed"),
            "best_val_f1": results.get(f"{variant}_best_val", {}).get(
                "macro_f1", {}).get("macro_f1"),
            "final_f1": results[key]["macro_f1"]["macro_f1"],
        }

    summary = {
        "meta": {
            "task": "head-init perturbation final test-donor evaluation "
                    "(basin-selection test, FROZEN 2026-08-20)",
            "eval_set": TEST_DONOR,
            "n_eval_cells": int(len(y_true)),
            "layer_order": LAYER_ORDER,
            "checkpoints": {"best_val": "model_best_val.pt (val macro-F1)",
                            "final_epoch": "model_epoch15.pt (sensitivity)"},
            "bootstrap": {"n": 1000, "seed": 42, "level": "cell-level"},
            "null": {"n": 100, "seed": 42, "primary": "L2-normed"},
            "frozen_baseline_test_donor": {"rho_normed": frz_rho, "z": frz_z,
                                           "probe_f1_macro": frz_f1},
            "judgment_rule": "final-epoch test rho_normed >= 0.6547 (locked "
                             "§9 collapse threshold) -> 'kept'",
            "reference": {f"s{s}": {"rho_normed": ref[f"s{s}"]["rho_normed"],
                                    "macro_f1": ref[f"s{s}"]["macro_f1"]["macro_f1"]}
                          for s in [42, 123, 456]},
            "timestamp_utc": now_utc(),
        },
        "frozen_baseline": frozen,
        "per_variant_per_ckpt": results,
        "basin_judgment": judgment,
    }
    out = OUT_DIR / "head_init_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("saved " + str(out))
    log(json.dumps(judgment, indent=1))
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
