#!/usr/bin/env python
"""Arm B (PREREG-2026-08-11 §9) per-seed merge + summary.

Reads per seed:
  - save/caft_gf_lam0.0_seed{S}_*/caft_args.json      (training provenance)
  - artifacts/paper1_tier_b_eval/results_0.0_{S}.json (run_tier_b_eval CE arm:
    donor macro-F1 + bootstrap CI + raw rho + LAA + ECE)
  - artifacts/paper1_vanzyl/gf_official_1epoch/eval_armb_seed{S}.json
    (L2-normed rho + label-shuffle z)
and writes gf_official_1epoch_seed{S}.json + gf_official_1epoch_summary.json
with the PREREG §9 Arm B verdict on the 3-seed mean raw rho (same convention
as the 3-epoch reference mean −0.008):
  mean_rho_raw <= 0.2            -> official-budget collapse confirmed
  mean_rho_raw > 0.5             -> overtraining-dependent collapse
  otherwise                      -> intermediate (report as is)
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CORNEAL_ROOT = Path("<home>/corneal_scgpt")
SAVE_DIR = CORNEAL_ROOT / "save"
TIERB_OUT = CORNEAL_ROOT / "artifacts" / "paper1_tier_b_eval"
OUT = CORNEAL_ROOT / "artifacts" / "paper1_vanzyl" / "gf_official_1epoch"

SEEDS = [42, 123, 456, 789, 2024, 31415]  # Arm B (3) + E3 extension (3, 2026-08-19)
REF_3EPOCH_RHO = {"42": -0.04239536391926124, "123": -0.03602503481100148,
                  "456": 0.053929088570711256}  # results_0.0_{S}.json raw
REF_3EPOCH_F1 = {"42": 0.6541863606043621, "123": 0.641991505068463,
                 "456": 0.6515084581425847}


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    draws = np.empty(n)
    for i in range(n):
        draws[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    rows = []
    per_seed = {}
    for seed in SEEDS:
        save_dirs = sorted(SAVE_DIR.glob(f"caft_gf_lam0.0_seed{seed}_*"))
        if not save_dirs:
            print(f"ERROR: no save dir for seed {seed}", flush=True)
            return 1
        save_dir = save_dirs[-1]
        cargs = json.loads((save_dir / "caft_args.json").read_text())
        tb = json.loads((TIERB_OUT / f"results_0.0_{seed}.json").read_text())
        extra = json.loads((OUT / f"eval_armb_seed{seed}.json").read_text())
        seed_json = {
            "seed": seed,
            "save_dir": save_dir.name,
            "training": {
                "epochs": cargs["epochs"],
                "lr": cargs["lr"],
                "per_device_batch": cargs["per_device_train_batch_size"],
                "grad_accum": cargs["gradient_accumulation_steps"],
                "effective_batch": cargs["effective_batch_size_per_device"],
                "warmup_steps": 500,
                "weight_decay": 0.001,
                "train_cells": cargs["train_cells"],
                "num_classes": cargs["num_classes"],
                "caft_lambda": cargs["caft_lambda"],
                "strict_fair": cargs["strict_fair"],
                "manifest_sha256": cargs["manifest_sha256"],
                "elapsed_min": cargs["elapsed_min"],
                "CAFT_DONE": (save_dir / "CAFT_DONE").is_file(),
            },
            "eval": {
                "donor_macro_f1": tb["primary_endpoint"]["macro_f1"],
                "ci_lower": tb["primary_endpoint"]["ci_lower"],
                "ci_upper": tb["primary_endpoint"]["ci_upper"],
                "rho_raw_tierb": tb["secondary_endpoints"][
                    "marker_gradient_rank_correlation"],
                "rho_raw": extra["rho_raw"],
                "rho_normed": extra["rho_normed"],
                "null_z": extra["null_z"],
                "null_pctile": extra["null_pctile"],
                "LAA": tb["secondary_endpoints"]["layer_adjacency_accuracy"],
                "ECE": tb["secondary_endpoints"]["expected_calibration_error"],
            },
            "reference_3epoch": {
                "rho_raw": REF_3EPOCH_RHO.get(str(seed)),
                "donor_macro_f1": REF_3EPOCH_F1.get(str(seed)),
                "note": "3-epoch reference exists only for Arm B seeds "
                        "42/123/456 (E3 seeds: null)",
            },
        }
        per_seed[seed] = seed_json
        rows.append(seed_json)
        (OUT / f"gf_official_1epoch_seed{seed}.json").write_text(
            json.dumps(seed_json, indent=2), encoding="utf-8")

    mean_rho_raw = float(np.mean([r["eval"]["rho_raw"] for r in rows]))
    mean_rho_normed = float(np.mean([r["eval"]["rho_normed"] for r in rows]))
    mean_f1 = float(np.mean([r["eval"]["donor_macro_f1"] for r in rows]))
    # 6-seed aggregation (E3): bootstrap 95% CI over seed-level metrics + collapse
    # rate per the LOCKED §9 threshold (rho_raw <= 0.2 = collapsed).
    rho_raws = [r["eval"]["rho_raw"] for r in rows]
    rho_normeds = [r["eval"]["rho_normed"] for r in rows]
    f1s = [r["eval"]["donor_macro_f1"] for r in rows]
    ci_rho_raw = bootstrap_ci(rho_raws)
    ci_rho_normed = bootstrap_ci(rho_normeds)
    ci_f1 = bootstrap_ci(f1s)
    collapse_rate = float(np.mean([x <= 0.2 for x in rho_raws]))
    if collapse_rate == 1.0:
        e3_verdict = ("DETERMINISTIC_COLLAPSE_CONFIRMED: all 6 seeds rho_raw <= 0.2 "
                      f"(rate 1.0); GF 1-epoch collapse is seed-invariant under "
                      f"the LOCKED budget")
    elif collapse_rate >= 0.5:
        e3_verdict = (f"PARTIAL_COLLAPSE: {int(collapse_rate * 6)}/6 seeds rho_raw <= 0.2 "
                      f"-> deterministic-collapse claim NOT fully supported")
    else:
        e3_verdict = (f"NO_COLLAPSE_MAJORITY: {int(collapse_rate * 6)}/6 seeds rho_raw <= 0.2 "
                      "-> deterministic-collapse claim REJECTED")
    if mean_rho_raw <= 0.2:
        verdict = ("OFFICIAL_BUDGET_COLLAPSE_CONFIRMED: 1-epoch mean raw rho "
                   f"{mean_rho_raw:+.4f} <= 0.2 -> GF collapse holds under the "
                   "official 1-epoch budget (claim strengthened)")
    elif mean_rho_raw > 0.5:
        verdict = ("OVERTRAINING_DEPENDENT: 1-epoch mean raw rho "
                   f"{mean_rho_raw:+.4f} > 0.5 vs 3-epoch -0.008 -> GF collapse "
                   "restated as overtraining-dependent (headline correction)")
    else:
        verdict = (f"INTERMEDIATE: 1-epoch mean raw rho {mean_rho_raw:+.4f} "
                   "in (0.2, 0.5]; report as is")
    summary = {
        "arm": "GF official 1-epoch budget control (PREREG-2026-08-11 §9 Arm B + E3)",
        "recipe": "naive CE 31-class, lr 5e-5, wd 1e-3, warmup 500, "
                  "effective batch 32 (8x4), bf16, 1 epoch",
        "seeds": SEEDS,
        "e3_seed_selection": "789, 2024, 31415 (deterministic, fixed 2026-08-19 "
                             "before any E3 training; arbitrary values, no "
                             "cherry-picking; Arm B seeds 42/123/456 preserved)",
        "train": "GSE153515 donor-disjoint train partition "
                 "(corneal_100k_donor_split.h5ad, manifest bc357c7d...; "
                 "80/20 prepare_data split -> 79,684 train cells)",
        "eval": "GSE153515 LOCKED external test (14,775 epithelial cells, "
                "2 donors; run_tier_b_eval CE arm + L2-normed rho & "
                "label-shuffle z companion)",
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "mean_rho_raw": mean_rho_raw,
        "mean_rho_normed": mean_rho_normed,
        "mean_donor_macro_f1": mean_f1,
        "reference_3epoch_means": {
            "rho_raw": float(np.mean(list(REF_3EPOCH_RHO.values()))),
            "donor_macro_f1": float(np.mean(list(REF_3EPOCH_F1.values()))),
        },
        "aggregate_6seed": {
            "n_seeds": len(rows),
            "rho_raw_mean": mean_rho_raw,
            "rho_raw_ci95_boot": list(ci_rho_raw),
            "rho_normed_mean": mean_rho_normed,
            "rho_normed_ci95_boot": list(ci_rho_normed),
            "donor_macro_f1_mean": mean_f1,
            "donor_macro_f1_ci95_boot": list(ci_f1),
            "collapse_rate_rho_raw_le0.2": collapse_rate,
            "n_collapsed": int(round(collapse_rate * len(rows))),
            "ci_method": "bootstrap n=1000 seed=42 over seed-level metrics",
        },
        "verdict": verdict,
        "e3_verdict": e3_verdict,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (OUT / "gf_official_1epoch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
