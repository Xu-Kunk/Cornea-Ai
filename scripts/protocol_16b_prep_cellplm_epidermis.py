#!/usr/bin/env python
"""Protocol 16b — CellPLM data adaptation for the epidermis arm (approved).

Consumes the SAME frozen split as the scGPT epidermis arm (Task 1, protocol
02e): artifacts/paper1_vanzyl/cross_tissue/split_indices.npz (test/val in
full 18,355-cell space). Identical cells, identical train/val/test — the
scGPT and CellPLM cross-tissue numbers are directly comparable.

CellPLM input contract (official pipeline, ensembl_auto_conversion=False):
  X must be raw-ish counts (OmicsEmbedder applies log1p internally).
  The v14 product X is log1p(1e4)-normalized -> X_cellplm = expm1(X) = 1e4-
  scaled counts (verified: rowsum(expm1(X)) == 10000 for every cell). After
  the internal log1p, CellPLM sees exactly the v14 log1p values.
  DEVIATION D-EPI-3 (recorded): pseudo-counts reconstructed by expm1 of the
  locked log1p(1e4) product (the raw GSM count matrices exist in
  data/epidermis/wang2020_epidermis/extracted/ but were not re-parsed; the
  reconstructed values carry identical information content at 1e4 depth).
  var index: gene symbols -> ENSG via the locked vanzyl_annotated.h5ad var
  mapping (feature_name -> ENSG, first occurrence), intersected with the
  CellPLM 19,374-gene set. Coverage 10,281/18,441 (58%) — disclosed.
  DEVIATION D-EPI-4 (recorded).

obs columns added: batch (= "epidermis"), split (train/valid/test),
celltype (= ordinal_layer 4-class).

Output: data/epidermis_cellplm.h5ad (+ manifest with sha256).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_H5AD = PROJECT_ROOT / "artifacts" / "paper1_epidermis_eval" / "wang2020_processed.h5ad"
OUT_H5AD = PROJECT_ROOT / "data" / "epidermis_cellplm.h5ad"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm_consistency"
CROSS_TISSUE = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue"
VANZYL_ANN = PROJECT_ROOT / "data" / "vanzyl_annotated.h5ad"
CKPT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "ckpt"

LAYER_ORDER = ["basal", "suprabasal", "spinous", "differentiated"]


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
    t0 = time.time()

    # 1. frozen split from the scGPT epidermis arm (identical cells)
    sidx = np.load(CROSS_TISSUE / "split_indices.npz")
    test_idx = sidx["test_idx"].astype(np.int64)
    val_idx = sidx["val_idx"].astype(np.int64)
    split_col = np.full(int(sidx["n_total"]), "test", dtype=object)
    split_col[val_idx] = "valid"
    split_col[test_idx] = "test"  # test first wins by construction
    # val/test are disjoint by construction; train = complement
    tr_mask = np.ones(int(sidx["n_total"]), dtype=bool)
    tr_mask[test_idx] = False
    tr_mask[val_idx] = False
    split_col[tr_mask] = "train"
    log(f"split: train={int((split_col == 'train').sum())} "
        f"valid={int((split_col == 'valid').sum())} "
        f"test={int((split_col == 'test').sum())}")

    # 2. symbol -> ENSG via vanzyl var (first occurrence)
    log("building symbol->ENSG map from vanzyl var")
    vz = ad.read_h5ad(VANZYL_ANN, backed="r")
    sym2ensg: dict[str, str] = {}
    for ensg, sym in zip(vz.var_names, vz.var["feature_name"].astype(str)):
        if sym not in sym2ensg:
            sym2ensg[sym] = ensg
    del vz
    log(f"mapping size: {len(sym2ensg)}")

    cfg = json.load(open(CKPT_DIR / "20230926_85M.config.json"))
    model_gene_set = set(cfg["gene_list"])

    # 3. load source, X -> expm1 pseudo-counts, map genes
    a = ad.read_h5ad(SRC_H5AD)
    x = a.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    x_counts = np.expm1(x)  # 1e4-scaled counts (verified rowsum == 1e4)
    x_counts = np.where(np.isfinite(x_counts), x_counts, 0.0).astype(np.float32)

    ensg_list: list[str] = []
    keep_cols: list[int] = []
    n_unmapped = 0
    n_not_in_model = 0
    for i, sym in enumerate(a.var_names):
        e = sym2ensg.get(str(sym))
        if e is None:
            n_unmapped += 1
            continue
        if e not in model_gene_set:
            n_not_in_model += 1
            continue
        ensg_list.append(e)
        keep_cols.append(i)
    log(f"genes: {a.n_vars} symbols -> {len(ensg_list)} mapped & in model "
        f"(unmapped {n_unmapped}, not-in-model {n_not_in_model})")

    x_csr = sp.csr_matrix(x_counts[:, keep_cols].astype(np.float32))
    obs = pd.DataFrame({
        "celltype": a.obs["ordinal_layer"].astype(str).values,
        "ordinal_layer": a.obs["ordinal_layer"].astype(str).values,
        "batch": np.full(a.n_obs, "epidermis", dtype=object),
        "split": split_col,
        "donor_id": np.full(a.n_obs, "epidermis", dtype=object),
    }, index=a.obs_names)
    out = ad.AnnData(X=x_csr, obs=obs,
                     var=pd.DataFrame(index=pd.Index(ensg_list, name="gene")))
    log(f"adapted: {out.shape}, X dtype {out.X.dtype}, "
        f"nnz {x_csr.nnz / 1e6:.1f}M")

    out.write_h5ad(OUT_H5AD)
    sha = sha256_file(OUT_H5AD)
    log(f"wrote {OUT_H5AD} sha256={sha} ({time.time() - t0:.0f}s)")

    manifest = {
        "protocol": "protocol_16b_prep_cellplm_epidermis.py",
        "source": str(SRC_H5AD),
        "source_sha256": sha256_file(SRC_H5AD),
        "output": str(OUT_H5AD),
        "output_sha256": sha,
        "split_source": str(CROSS_TISSUE / "split_indices.npz"),
        "split_rule": "identical to the scGPT epidermis arm (protocol_02e): "
                      "test 10% + val 10% of remaining per ordinal stratum, "
                      "seed 42",
        "n_train": int((split_col == "train").sum()),
        "n_valid": int((split_col == "valid").sum()),
        "n_test": int((split_col == "test").sum()),
        "layer_order": LAYER_ORDER,
        "columns_added": ["batch", "split", "celltype"],
        "X_source": "expm1(v14 log1p(1e4) X) = 1e4-scaled pseudo-counts "
                    "(CellPLM embedder log1p contract; D-EPI-3)",
        "gene_mapping": {
            "method": "symbol -> ENSG via vanzyl_annotated.h5ad var "
                      "(feature_name -> var_names, first occurrence)",
            "n_symbols_source": int(a.n_vars),
            "n_mapped_and_in_model": len(ensg_list),
            "n_unmapped": n_unmapped,
            "n_mapped_but_not_in_model": n_not_in_model,
            "model_gene_set_size": len(model_gene_set),
        },
        "deviations": [
            "D-EPI-3: pseudo-counts via expm1 of the locked log1p(1e4) "
            "product (identical information at 1e4 depth)",
            "D-EPI-4: gene symbols mapped to ENSG via vanzyl var; "
            "unmapped genes dropped (coverage 58%)",
        ],
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (OUT_DIR / "data_manifest_epidermis.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    log("manifest written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
