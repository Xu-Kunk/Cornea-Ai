#!/usr/bin/env python
"""Protocol 16 — CellPLM data adaptation for van Zyl (this study, arm C3).

Input contract alignment (scientific logic untouched; only the CellPLM
input contract is adapted):

* vanzyl_annotated.h5ad (LOCKED, sha fad5e9a7...): X is log1p(1e4)-normalized;
  raw counts live in layers['counts'] — CellPLM's OmicsEmbedder applies
  log1p internally, so X must be raw counts (official pipeline contract).
* Analysis set rule (LOCKED, PREREG §7 / split_manifest.json):
    leiden != '0' AND cell_class_4 in {LSC, basal, differentiated}
* Test donor (LOCKED, PREREG §8): sanes2022_Hu0216 (20,158 cells);
  train donors = other 4 (42,482 cells). Strictly held out.
* Split (judgment, frozen here): official CellPLM tutorial convention —
  90/10 cell-level train/valid on the 42,482 train-donor cells,
  stratified by (donor x celltype), seed 42.
* Columns added: batch (= donor_id), split (train/valid/test),
  celltype (= cell_class_4 3-tier).

Output: data/vanzyl_cellplm.h5ad (+ sha256 + data_manifest.json).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

PROJECT_ROOT = Path("<local>/")
SRC_H5AD = PROJECT_ROOT / "data" / "vanzyl_annotated.h5ad"
OUT_H5AD = PROJECT_ROOT / "data" / "vanzyl_cellplm.h5ad"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm"

LOCKED_TEST_DONOR = "sanes2022_Hu0216"
LOCKED_TRAIN_CELLS = 42482
LOCKED_TEST_CELLS = 20158
THREE_TIER = ["LSC", "basal", "differentiated"]
SPLIT_SEED = 42  # frozen; only governs train/valid split, NOT model training seeds
VALID_FRAC = 0.1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

    # 1. Read source (backed), capture obs columns + var_names
    log("reading source h5ad (backed)")
    src = ad.read_h5ad(SRC_H5AD, backed="r")
    var_names = src.var_names.copy()
    obs = src.obs[["donor_id", "leiden", "cell_class_4"]].copy()
    n_source = src.shape[0]
    del src
    log(f"source {n_source} cells x {len(var_names)} genes")

    # 2. Analysis set mask (LOCKED rule)
    mask = (
        (obs["leiden"] != "0")
        & (obs["cell_class_4"].isin(THREE_TIER))
    ).values
    obs = obs[mask].reset_index(drop=True)
    log(f"analysis set: {len(obs)} cells (train+test)")

    # 3. Donor split (LOCKED)
    is_test = (obs["donor_id"] == LOCKED_TEST_DONOR).values
    n_test = int(is_test.sum())
    n_train = int((~is_test).sum())
    assert n_train == LOCKED_TRAIN_CELLS, f"train cells {n_train} != locked {LOCKED_TRAIN_CELLS}"
    assert n_test == LOCKED_TEST_CELLS, f"test cells {n_test} != locked {LOCKED_TEST_CELLS}"
    log(f"train donors {n_train} cells / test donor {n_test} cells (LOCKED split verified)")

    # 4. Train/valid split (frozen: official 90/10, stratified by donor x celltype, seed 42)
    from sklearn.model_selection import train_test_split

    train_idx = np.nonzero(~is_test)[0]
    strat = np.char.add(
        np.char.add(obs["donor_id"].astype(str).values, "|"),
        obs["cell_class_4"].astype(str).values,
    )[train_idx]
    train_sel, valid_sel = train_test_split(
        train_idx, test_size=VALID_FRAC, random_state=SPLIT_SEED, stratify=strat
    )
    split_col = np.full(len(obs), "test", dtype=object)
    split_col[train_sel] = "train"
    split_col[valid_sel] = "valid"
    log(f"train {int((split_col=='train').sum())} / valid {int((split_col=='valid').sum())} / test {n_test}")
    for part in ["train", "valid"]:
        counts = obs.loc[split_col == part, "cell_class_4"].value_counts(normalize=True).round(3).to_dict()
        log(f"  {part} class frac: {counts}")

    # 5. Extract raw counts for analysis cells (chunked)
    log("extracting raw counts (layers['counts']) for analysis cells")
    src2 = ad.read_h5ad(SRC_H5AD, backed="r")
    counts = src2.layers["counts"][mask]
    log(f"counts extracted: {counts.shape} ({counts.nnz / 1e6:.1f}M nnz)")
    del src2

    # 6. Build adapted AnnData: X = raw counts (float32 csr), obs with new columns
    out = ad.AnnData(
        X=sp.csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame(
            {
                "donor_id": obs["donor_id"].values,
                "leiden": obs["leiden"].values,
                "cell_class_4": obs["cell_class_4"].values,
                "batch": obs["donor_id"].values,
                "split": split_col,
                "celltype": obs["cell_class_4"].values,
            },
            index=obs.index,
        ),
        var=pd.DataFrame(index=var_names),
    )
    log(f"adapted h5ad: {out.shape}, X dtype {out.X.dtype}")

    # 7. Gene overlap with CellPLM pretrain gene set
    cfg = json.load(open(OUT_DIR / "ckpt" / "20230926_85M.config.json"))
    gene_set = set(cfg["gene_list"])
    overlap = float(out.var_names.isin(gene_set).mean())
    log(f"gene overlap with CellPLM {len(gene_set)}-gene set: {overlap:.4f}")

    # 8. Persist + sha256
    out.write_h5ad(OUT_H5AD)
    sha = sha256_file(OUT_H5AD)
    log(f"wrote {OUT_H5AD} sha256={sha}")

    manifest = {
        "protocol": "protocol_16_prep_cellplm_vanzyl.py",
        "source": str(SRC_H5AD),
        "source_sha256": "fad5e9a71ecd1208e0626049b638e789d22b52563d5a66740570d617df67f4e8",
        "output": str(OUT_H5AD),
        "output_sha256": sha,
        "analysis_set_rule": "leiden != '0' AND cell_class_4 in {LSC, basal, differentiated} (LOCKED PREREG §7)",
        "test_donor": LOCKED_TEST_DONOR,
        "train_donors": sorted(set(obs.loc[split_col != "test", "donor_id"].unique())),
        "n_train": int((split_col == "train").sum()),
        "n_valid": int((split_col == "valid").sum()),
        "n_test": int(n_test),
        "split_rule": "90/10 cell-level train/valid on train-donor cells, stratified (donor x celltype), "
        f"seed {SPLIT_SEED} (official CellPLM tutorial convention)",
        "valid_frac": VALID_FRAC,
        "split_seed": SPLIT_SEED,
        "columns_added": ["batch", "split", "celltype"],
        "X_source": "layers['counts'] (raw counts; CellPLM embedder log1p contract)",
        "n_genes": int(out.shape[1]),
        "gene_overlap_with_cellplm": overlap,
        "cellplm_gene_set_size": len(gene_set),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    log("manifest written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
