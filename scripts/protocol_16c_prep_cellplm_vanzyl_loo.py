#!/usr/bin/env python
"""Protocol 16c — CellPLM data adaptation for the van Zyl LOO arm (approved).

Leave-one-donor-out replicate of protocol_16 (arm C3) with test donor
sanes2022_Hu0220 (FROZEN judgment, same as the scGPT LOO arm — identical
cells, directly comparable).

Input contract identical to protocol_16:
  * vanzyl_annotated.h5ad X is log1p(1e4)-normalized; raw counts in
    layers['counts'] -> X = raw counts (CellPLM embedder log1p contract).
  * Analysis set rule (LOCKED): leiden != '0' AND cell_class_4 in
    {LSC, basal, differentiated}.
  * Test donor Hu0220 strictly held out; train/valid = 90/10 cell-level
    stratified (donor x celltype), seed 42 (official CellPLM convention,
    mirrors protocol_16).
  * Columns: batch (= donor_id), split (train/valid/test),
    celltype (= cell_class_4 3-tier).

Output: data/vanzyl_cellplm_loo220.h5ad (+ manifest with sha256).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_H5AD = PROJECT_ROOT / "data" / "vanzyl_annotated.h5ad"
OUT_H5AD = PROJECT_ROOT / "data" / "vanzyl_cellplm_loo220.h5ad"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm_consistency"
CKPT_DIR = PROJECT_ROOT.parent / "cellplm_vanzyl" / "artifacts" / "paper1_vanzyl" / "cellplm" / "ckpt"

TEST_DONOR = "sanes2022_Hu0220"
LOCKED_TRAIN_CELLS = 62_640 - 16_705  # 45,935 (analysis set minus Hu0220)
THREE_TIER = ["LSC", "basal", "differentiated"]
SPLIT_SEED = 42
VALID_FRAC = 0.1


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("reading source h5ad (backed)")
    src = ad.read_h5ad(SRC_H5AD, backed="r")
    var_names = src.var_names.copy()
    obs = src.obs[["donor_id", "leiden", "cell_class_4"]].copy()
    n_source = src.shape[0]
    del src
    log(f"source {n_source} cells x {len(var_names)} genes")

    mask = (
        (obs["leiden"] != "0") & (obs["cell_class_4"].isin(THREE_TIER))
    ).values
    obs = obs[mask].reset_index(drop=True)
    log(f"analysis set: {len(obs)} cells")

    is_test = (obs["donor_id"] == TEST_DONOR).values
    n_test = int(is_test.sum())
    n_train_donors = int((~is_test).sum())
    assert n_train_donors == LOCKED_TRAIN_CELLS, \
        f"train cells {n_train_donors} != locked {LOCKED_TRAIN_CELLS}"
    log(f"train donors {n_train_donors} cells / test donor {n_test} cells")

    from sklearn.model_selection import train_test_split

    train_idx = np.nonzero(~is_test)[0]
    strat = np.char.add(
        np.char.add(obs["donor_id"].astype(str).values, "|"),
        obs["cell_class_4"].astype(str).values,
    )[train_idx]
    train_sel, valid_sel = train_test_split(
        train_idx, test_size=VALID_FRAC, random_state=SPLIT_SEED,
        stratify=strat,
    )
    split_col = np.full(len(obs), "test", dtype=object)
    split_col[train_sel] = "train"
    split_col[valid_sel] = "valid"
    log(f"train {int((split_col == 'train').sum())} / "
        f"valid {int((split_col == 'valid').sum())} / test {n_test}")

    log("extracting raw counts (layers['counts']) for analysis cells")
    src2 = ad.read_h5ad(SRC_H5AD, backed="r")
    counts = src2.layers["counts"][mask]
    del src2
    log(f"counts extracted: {counts.shape} ({counts.nnz / 1e6:.1f}M nnz)")

    out = ad.AnnData(
        X=sp.csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame({
            "donor_id": obs["donor_id"].values,
            "leiden": obs["leiden"].values,
            "cell_class_4": obs["cell_class_4"].values,
            "batch": obs["donor_id"].values,
            "split": split_col,
            "celltype": obs["cell_class_4"].values,
        }, index=obs.index),
        var=pd.DataFrame(index=var_names),
    )
    log(f"adapted h5ad: {out.shape}")

    cfg = json.load(open(CKPT_DIR / "20230926_85M.config.json"))
    gene_set = set(cfg["gene_list"])
    overlap = float(out.var_names.isin(gene_set).mean())
    log(f"gene overlap with CellPLM {len(gene_set)}-gene set: {overlap:.4f}")

    out.write_h5ad(OUT_H5AD)
    sha = sha256_file(OUT_H5AD)
    log(f"wrote {OUT_H5AD} sha256={sha}")

    manifest = {
        "protocol": "protocol_16c_prep_cellplm_vanzyl_loo.py",
        "source": str(SRC_H5AD),
        "source_sha256": "fad5e9a71ecd1208e0626049b638e789d22b52563d5a66740570d617df67f4e8",
        "output": str(OUT_H5AD),
        "output_sha256": sha,
        "analysis_set_rule": "leiden != '0' AND cell_class_4 in "
                             "{LSC, basal, differentiated} (LOCKED PREREG §7)",
        "test_donor": TEST_DONOR,
        "train_donors": sorted(set(obs.loc[split_col != "test",
                                            "donor_id"].unique())),
        "n_train": int((split_col == "train").sum()),
        "n_valid": int((split_col == "valid").sum()),
        "n_test": n_test,
        "split_rule": "90/10 cell-level train/valid on train-donor cells, "
                      "stratified (donor x celltype), seed 42 (official "
                      "CellPLM tutorial convention, mirrors protocol_16)",
        "valid_frac": VALID_FRAC,
        "split_seed": SPLIT_SEED,
        "columns_added": ["batch", "split", "celltype"],
        "X_source": "layers['counts'] (raw counts; CellPLM embedder log1p "
                    "contract)",
        "n_genes": int(out.shape[1]),
        "gene_overlap_with_cellplm": overlap,
        "cellplm_gene_set_size": len(gene_set),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "data_manifest_loo220.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    log("manifest written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
