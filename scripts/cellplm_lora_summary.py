#!/usr/bin/env python
"""Aggregate the CellPLM LoRA-only seeds (van Zyl, test donor Hu0216).

Reads ``results/paper1_vanzyl/cellplm_lora/seed{42,123,456}/seed{seed}.json``
(written by ``scripts/cellplm_lora_vanzyl.py``: LoRA r=8 alpha=16 on the
Flow_Attention q/k/v projections, trunk frozen, official
``CellTypeAnnotationPipeline.fit()``) and writes ``summary.json`` with the
per-seed rows and the three-seed mean ± sample SD (ddof=1) for rho (normed),
rho (raw) and macro-F1.

Usage:
    python scripts/cellplm_lora_summary.py
"""

from __future__ import annotations

import json
import statistics as st
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "paper1_vanzyl" / "cellplm_lora"
SEEDS = (42, 123, 456)


def _agg(values: dict) -> dict:
    xs = [v for v in values.values() if v is not None]
    return {
        "mean": float(st.mean(xs)) if xs else None,
        "sd": float(st.stdev(xs)) if len(xs) > 1 else None,
        "sd_kind": "sample SD (ddof=1)",
        "n": len(xs),
        "per_seed": values,
    }


def main() -> int:
    per_seed = {}
    for seed in SEEDS:
        path = OUT_DIR / f"seed{seed}" / f"seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        doc = json.loads(path.read_text())
        per_seed[str(seed)] = {
            "rho_normed": doc["eval"]["rho_normed"],
            "rho_raw": doc["eval"]["rho_raw"],
            "macro_f1": doc["eval"]["macro_f1"],
            "acc": doc["eval"]["acc"],
            "bootstrap_ci95": doc["eval"]["bootstrap_ci95"],
            "null_z": doc["eval"]["null_z"],
            "n_eval_cells": doc["eval"]["n_eval_cells"],
            "test_donor": doc["eval"]["test_donor"],
            "epochs_actual": doc["training"]["epochs_actual"],
            "best_valid_f1": doc["training"]["best_valid_f1"],
            "elapsed_min": doc["training"]["elapsed_min"],
            "n_train": doc["training"]["n_train"],
            "n_valid": doc["training"]["n_valid"],
            "frozen_rho_normed": doc.get("frozen_baseline", {}).get("rho_normed"),
            "delta_rho_vs_frozen": doc.get("delta_rho", {}).get("delta_rho"),
            "recipe": doc.get("recipe"),
            "lora": doc.get("lora"),
            "timestamp_utc": doc.get("timestamp_utc"),
        }

    summary = {
        "arm": "CellPLM LoRA-only (LoRA q/k/v r=8 alpha=16, trunk frozen)",
        "dataset": "van Zyl annotated (third architecture, LoRA arm)",
        "test_donor": "sanes2022_Hu0216",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": {
            "rho_normed": _agg({s: r["rho_normed"] for s, r in per_seed.items()}),
            "rho_raw": _agg({s: r["rho_raw"] for s, r in per_seed.items()}),
            "macro_f1": _agg({s: r["macro_f1"] for s, r in per_seed.items()}),
            # Early stopping fired at very different epochs per seed; exposed
            # explicitly because it is the candidate cause of the rho spread.
            "epochs_actual": _agg({s: r["epochs_actual"] for s, r in per_seed.items()}),
            "elapsed_min": _agg({s: r["elapsed_min"] for s, r in per_seed.items()}),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{'seed':<6}{'rho_normed':>12}{'macro_f1':>10}{'acc':>8}{'minutes':>9}{'epochs':>8}")
    for s, r in per_seed.items():
        print(f"{s:<6}{r['rho_normed']:>12.4f}{r['macro_f1']:>10.4f}"
              f"{r['acc']:>8.4f}{r['elapsed_min']:>9.2f}{r['epochs_actual']:>8d}")
    agg = summary["aggregate"]
    print(f"{'mean':<6}{agg['rho_normed']['mean']:>12.4f}{agg['macro_f1']['mean']:>10.4f}"
          f"{'':>8}{'':>9}{agg['epochs_actual']['mean']:>8.1f}")
    print(f"{'sd':<6}{agg['rho_normed']['sd']:>12.4f}{agg['macro_f1']['sd']:>10.4f}"
          f"{'':>8}{'':>9}{agg['epochs_actual']['sd']:>8.1f}")
    print(f"epochs_actual per seed: {agg['epochs_actual']['per_seed']}")
    print(f"wrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
