#!/usr/bin/env python
"""Cornea2 donor-aware train/test split (mirrors run6 split_manifest.json).

Rule (LOCKED, outcome-blind, mirrors PREREG-2026-08-11 §8):
    analysis set  = cell_class_4 in {LSC, basal, differentiated} (3-tier)
    test donor    = donor with the MAX 3-tier epithelial cell count
                    (GSE186433: among single-donor units only — the 07+08
                    limbus pool mixes 2 donors and cannot be a held-out
                    test donor; it goes to TRAIN and is disclosed).
    train         = all other donors' 3-tier cells.

Per dataset outputs:
    data/cornea2/<ds>_split_manifest.json   (LOCKED, outcome-blind)
    data/cornea2/<ds>_train.h5ad / _test.h5ad (scGPT panel, 3-tier cells)
    data/cornea2/<ds>_val_indices.npz       (10% of train, stratified by
                                             donor x cell_class_4, seed 42)
    data/cornea2/<ds>_split_val_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "cornea2"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cornea2"

LAYER_ORDER_3 = ["LSC", "basal", "differentiated"]
VAL_FRAC = 0.10
VAL_SEED = 42
TEST_DONOR_RULE = "max 3-tier epithelial cells among eligible donors (outcome-blind)"


def donor_counts(adata: ad.AnnData, donors: list[str]) -> dict:
    """3-tier cell count per donor."""
    return {
        d: int((adata.obs["donor_id"] == d).sum())
        for d in donors
    }


def stratified_val_split(
    train_adata: ad.AnnData, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """10% stratified val split by donor_id x cell_class_4 (mirrors
    lock_val_split.py: per-stratum proportional draw, rng seed 42)."""
    rng = np.random.default_rng(seed)
    df = train_adata.obs[["donor_id", "cell_class_4"]].copy()
    strata = df.groupby(["donor_id", "cell_class_4"]).size()
    val_mask = np.zeros(train_adata.n_obs, dtype=bool)
    for (d, c), n in strata.items():
        idx = np.where((df["donor_id"] == d) & (df["cell_class_4"] == c))[0]
        n_val = int(round(n * val_frac))
        if n_val >= 1:
            pick = rng.choice(idx, size=n_val, replace=False)
            val_mask[pick] = True
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    return train_idx, val_idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["GSE186433", "GSE155683"],
                    required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    tag = ds.lower()

    scgpt_h5ad = OUT_DIR / f"{tag}_scgpt.h5ad"
    if not scgpt_h5ad.exists():
        print(f"ERROR: {scgpt_h5ad} not found (run protocol_02c first)")
        return 1
    out_train = OUT_DIR / f"{tag}_train.h5ad"
    out_test = OUT_DIR / f"{tag}_test.h5ad"
    if (out_train.exists() or out_test.exists()) and not args.force:
        print(f"ERROR: outputs exist ({out_train}) — use --force")
        return 1

    adata = ad.read_h5ad(scgpt_h5ad)
    # analysis set: 3-tier epithelial only
    mask = adata.obs["cell_class_4"].isin(LAYER_ORDER_3).to_numpy()
    adata = adata[mask].copy()
    print(f"analysis set: {adata.n_obs} 3-tier cells, "
          f"{adata.obs['donor_id'].nunique()} donors")

    donors = sorted(adata.obs["donor_id"].unique().tolist())
    if ds == "GSE186433":
        # 07+08 is a 2-donor limbus pool -> cannot be a held-out test donor
        eligible = [d for d in donors if d != "07+08"]
        pool_donor = "07+08"
        note_pool = (
            f"pool donor {pool_donor} (2-donor limbus pool) -> TRAIN only; "
            "test donor selected among single-donor units"
        )
    else:
        eligible = donors
        pool_donor = None
        note_pool = "all donors are single adult donors"

    counts = donor_counts(adata, eligible)
    test_donor = max(counts, key=counts.get)
    train_donors = [d for d in eligible if d != test_donor]
    if pool_donor is not None:
        train_donors.append(pool_donor)

    manifest = {
        "dataset": ds,
        "rule": TEST_DONOR_RULE,
        "analysis_set_rule": "cell_class_4 in {LSC, basal, differentiated} "
                             "(3-tier; mirrors PREREG-2026-08-11 §7/§8)",
        "layer_order": LAYER_ORDER_3,
        "per_donor_3tier_counts": donor_counts(adata, donors),
        "test_donor": test_donor,
        "test_donor_note": note_pool,
        "train_donors": train_donors,
        "n_cells_total_3tier": int(adata.n_obs),
        "n_train_cells": int((adata.obs["donor_id"].isin(train_donors)).sum()),
        "n_test_cells": int((adata.obs["donor_id"] == test_donor).sum()),
        "panel_genes": int(adata.n_vars),
        "prepped_panel": json.loads(
            (ARTIFACT_DIR / f"{tag}_scgpt_panel_check.json").read_text()
        )["panel_size"],
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (OUT_DIR / f"{tag}_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(json.dumps(manifest, indent=2))

    train_adata = adata[adata.obs["donor_id"].isin(train_donors)].copy()
    test_adata = adata[adata.obs["donor_id"] == test_donor].copy()
    print(f"train: {train_adata.n_obs} cells ({len(train_donors)} donors)")
    print(f"test:  {test_adata.n_obs} cells (donor {test_donor})")

    # val split (10% of train, stratified) — LOCKED manifest
    tr_idx, val_idx = stratified_val_split(train_adata, VAL_FRAC, VAL_SEED)
    np.savez(OUT_DIR / f"{tag}_val_indices.npz",
             train_idx=tr_idx, val_idx=val_idx)
    val_manifest = {
        "dataset": ds,
        "rule": "10% of train, stratified by donor_id x cell_class_4 "
                f"(seed {VAL_SEED}); test donor never enters",
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "sha256": {
            f"{tag}_train.h5ad": None,  # filled after write
        },
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (OUT_DIR / f"{tag}_split_val_manifest.json").write_text(
        json.dumps(val_manifest, indent=2, ensure_ascii=False)
    )

    train_adata.write_h5ad(out_train)
    test_adata.write_h5ad(out_test)
    print(f"saved {out_train} ({train_adata.n_obs} x {train_adata.n_vars})")
    print(f"saved {out_test} ({test_adata.n_obs} x {test_adata.n_vars})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
