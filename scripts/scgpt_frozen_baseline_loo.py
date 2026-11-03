#!/usr/bin/env python
"""Frozen baseline for the scGPT LOO arm (test donor Hu0220).

Computes the frozen (unfitted pretrained scGPT) rho/null/probe on the
Hu0220 test cells by donor-masking the LOCKED G1 embeddings
(artifacts/g1_scgpt/vanzyl_cell_emb.npz, 62,640 x 512 — the SAME source the
Hu0216 frozen baseline was derived from; identical metric code). CPU-only.

Output: artifacts/paper1_vanzyl/loo/frozen_baseline_loo220.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("<home>/corneal_scgpt")
sys.path.insert(0, str(ROOT / "scripts"))
from scgpt_frozen_diag_rebuild import (  # noqa: E402  (unchanged metric code)
    fit_eval_probe,
    l2_norm_rows,
    label_shuffle_null,
)
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

G1_NPZ = ROOT / "artifacts" / "g1_scgpt" / "vanzyl_cell_emb.npz"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "loo"
TEST_DONOR = "sanes2022_Hu0220"
LAYER_ORDER = ["LSC", "basal", "differentiated"]
N_NULL = 100
N_PROBE = 5


def main() -> int:
    t0 = time.time()
    z = np.load(G1_NPZ, allow_pickle=True)
    cell_emb = z["cell_emb"]
    labels = z["labels"]
    donors = z["donors"]
    n_total = cell_emb.shape[0]
    mask = donors == TEST_DONOR
    n_test = int(mask.sum())
    print(f"G1 npz: {n_total} cells; {TEST_DONOR} cells: {n_test}", flush=True)
    emb = cell_emb[mask].astype(np.float32)
    y = labels[mask]

    emb_n = l2_norm_rows(emb)
    rho_raw = float(marker_gradient_correlation(emb, y, LAYER_ORDER))
    rho_normed = float(marker_gradient_correlation(emb_n, y, LAYER_ORDER))
    null = label_shuffle_null(emb_n, y, LAYER_ORDER, n=N_NULL, seed=42)
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    null_z = float((rho_normed - null_mean) / max(null_sd, 1e-12))
    null_pctile = float((null <= rho_normed).mean())
    probe = fit_eval_probe(emb, y, LAYER_ORDER, n_repeats=N_PROBE)
    counts = {str(c): int((y == c).sum()) for c in LAYER_ORDER}
    print(f"rho_raw={rho_raw:.4f} rho_normed={rho_normed:.4f} "
          f"z={null_z:.1f} pctile={null_pctile:.2f} "
          f"probe_f1={probe['f1_macro']:.4f}", flush=True)
    print(f"label counts: {counts}", flush=True)

    res = {
        "arm": "frozen",
        "seed": None,
        "n_eval_cells": n_test,
        "test_donor": TEST_DONOR,
        "source": str(G1_NPZ),
        "rho": {
            "rho_raw": rho_raw,
            "rho_normed": rho_normed,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "null_z": null_z,
            "null_pctile": null_pctile,
            "n_null": N_NULL,
        },
        "probe_f1": probe,
        "label_counts": counts,
        "metric_code": "scgpt_frozen_diag_rebuild.py (unchanged; G1 rebuild)",
        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_baseline_loo220.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("saved", OUT / "frozen_baseline_loo220.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
