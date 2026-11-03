#!/usr/bin/env python
"""Merge cross-arms supplementary results into one summary (runs on <compute-node>
after all phases; also copyable locally).

Combines:
  * cross_tissue/final_test_eval.json      (scGPT epidermis)
  * loo/final_test_eval.json               (scGPT LOO Hu0220)
  * cellplm_consistency/summary_epidermis.json
  * cellplm_consistency/summary_loo220.json

Output: artifacts/paper1_vanzyl/cross/cross_arms_summary.json
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("<home>/corneal_scgpt")
OUT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "cross"


def load(p: Path) -> dict | None:
    if not p.is_file():
        print(f"missing: {p}", flush=True)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    cross = load(ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue" / "final_test_eval.json")
    loo = load(ROOT / "artifacts" / "paper1_vanzyl" / "loo" / "final_test_eval.json")
    cp_epi = load(ROOT / "artifacts" / "paper1_vanzyl" / "cellplm_consistency" / "summary_epidermis.json")
    cp_loo = load(ROOT / "artifacts" / "paper1_vanzyl" / "cellplm_consistency" / "summary_loo220.json")

    summary = {
        "task": "Approved supplementary experiments: scGPT cross-tissue + "
                "scGPT LOO + CellPLM consistency (reviewer points: "
                "single-tissue / single-donor)",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "references": {
            "vanzyl_hu0216_scgpt": {
                "arm": "scGPT van Zyl, test donor Hu0216 (PREREG §9 Arm A)",
                "frozen_rho": 0.7887,
                "best_val_rho_mean": 0.4395,
                "final_epoch_rho_mean": 0.5023,
                "bistable_final_per_seed": [-0.0908, 0.8329, 0.7647],
            },
            "vanzyl_cellplm": {
                "arm": "CellPLM van Zyl, test donor Hu0216 (arm C3)",
                "frozen_rho_mean": 0.84,
                "rho_normed_per_seed": [-0.0464, 0.0187, 0.0064],
                "rho_normed_mean": -0.007,
                "macro_f1_mean": 0.973,
            },
            "gf_epidermis_v14": {
                "arm": "Geneformer cross-tissue epidermis (v14)",
                "frozen_rho_L12_ordinal": 0.2395,
                "ce_naive_rho_L12_ordinal": 0.4551,
            },
            "gf_vanzyl_official_1epoch": {
                "arm": "GF 1-epoch official budget (E3-GF, 6 seeds)",
                "rho_raw_mean": 0.0579,
                "collapse_rate_5of6": "5/6 rho_raw <= 0.2",
            },
        },
        "scgpt_epidermis": cross,
        "scgpt_loo220": loo,
        "cellplm_epidermis": cp_epi,
        "cellplm_loo220": cp_loo,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "cross_arms_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
