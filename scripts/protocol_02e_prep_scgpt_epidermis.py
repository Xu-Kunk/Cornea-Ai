#!/usr/bin/env python
"""Protocol 02e — scGPT data preparation for the cross-tissue epidermis arm.

Supplementary experiment (reviewer point: single-tissue bias), approved.
Consumes the LOCKED v14 processing product
artifacts/paper1_epidermis_eval/wang2020_processed.h5ad (Wang 2020,
GSE147424; 18,355 marker-positive cells; X = log1p(1e4) normalized; 4-layer
ordinal: basal < suprabasal < spinous < differentiated; v14 marker_axis
protocol, epidermis_tpoac_eval.py).

Panel (mirrors protocol_02b, PREREG §4 rule adapted to epidermis):
  HVG ∩ scGPT vocab; default n_top_genes = 1200; raised to 4253 if ANY of
  the 7 epidermis differentiation markers (KRT5/KRT14/TP63/KRT1/KRT10/IVL/
  FLG — the v14 eval markers) is missing from the final panel.
  DEVIATION D-EPI-1 (recorded): the v14 product has NO raw-counts layer
  (X is already log1p(1e4)-normalized), so HVG is computed on the log1p X
  with scanpy flavor='seurat' (protocol_02b used raw counts +
  seurat_v3/seurat). Scientific logic otherwise unchanged.

Split (frozen judgment, outcome-blind, locked pre-run):
  test = 10% per ordinal stratum (seed 42, deterministic rng permutation,
  mirroring lock_val_split.py); val = 10% of the remaining per stratum;
  train = the rest. Stratified by ordinal_layer (perfectly balanced 4-class
  axis). Cell-level split (no donor metadata in the v14 product; samples
  S1-S17, no donor_id column) — disclosed.

Outputs (local, rsynced to p5070):
  artifacts/paper1_vanzyl/cross_tissue/scgpt_epidermis_{train,test}.h5ad
  artifacts/paper1_vanzyl/cross_tissue/val_indices.npz      (val, in train space)
  artifacts/paper1_vanzyl/cross_tissue/split_indices.npz    (test/val, in full space)
  artifacts/paper1_vanzyl/cross_tissue/split_val_manifest.json
  artifacts/paper1_vanzyl/cross_tissue/split_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_H5AD = PROJECT_ROOT / "artifacts" / "paper1_epidermis_eval" / "wang2020_processed.h5ad"
VOCAB_PATH = PROJECT_ROOT / "data" / "scgpt_gene_vocab.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue"

LAYER_ORDER = ["basal", "suprabasal", "spinous", "differentiated"]
EPI_MARKERS = ["KRT5", "KRT14", "TP63", "KRT1", "KRT10", "IVL", "FLG"]
DEFAULT_N_TOP_GENES = 1200
RAISED_N_TOP_GENES = 4253
HVG_SUBSAMPLE = 50_000
RANDOM_SEED = 42
SPLIT_SEED = 42
TEST_FRAC = 0.1
VAL_FRAC_OF_REMAINING = 0.1


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vocab(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def marker_panel_check(symbols: list[str], markers: list[str]) -> dict:
    sym_set = set(symbols)
    present = {m: (m in sym_set) for m in markers}
    missing = [m for m in markers if not present[m]]
    return {
        "n_markers_total": len(markers),
        "n_present": sum(present.values()),
        "missing": missing,
        "all_present": len(missing) == 0,
    }


def build_panel(adata: ad.AnnData, x_log1p, var_df: pd.DataFrame, vocab: dict,
                n_top_genes: int) -> tuple[ad.AnnData, dict]:
    """HVG∩vocab panel on the log1p X (D-EPI-1: no raw layer in v14 product).

    Returns (panel adata, panel stats dict). X stays log1p(1e4) — the scGPT
    tokenizer value contract (continuous input embedding, official args).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_subsample = min(HVG_SUBSAMPLE, x_log1p.shape[0])
    idx = rng.choice(x_log1p.shape[0], n_subsample, replace=False)
    idx.sort()

    tmp = ad.AnnData(X=x_log1p[idx].copy(), var=var_df.copy())
    tmp.var_names = adata.var_names.copy()

    hvg_genes = None
    hvg_mask = None
    for flavor in ("seurat",):  # log1p X -> seurat flavor (D-EPI-1)
        try:
            sc.pp.highly_variable_genes(
                tmp, n_top_genes=n_top_genes, flavor=flavor, subset=False,
            )
            hvg_mask = tmp.var["highly_variable"].values
            hvg_genes = tmp.var_names[hvg_mask].tolist()
            log(f"  ✓ HVG with flavor='{flavor}': {len(hvg_genes)} genes")
            break
        except Exception as exc:
            log(f"  ✗ flavor='{flavor}' failed: {exc}")
    if hvg_genes is None:
        raise RuntimeError(f"HVG selection failed for n_top_genes={n_top_genes}")

    hvg_symbols = tmp.var.loc[hvg_mask, "gene_symbol"].astype(str).tolist()
    vocab_set = set(vocab.keys())
    matched = [s for s in hvg_symbols if s in vocab_set]
    log(f"  {len(matched)}/{len(hvg_symbols)} HVG symbols in scGPT vocab "
        f"({100.0 * len(matched) / len(hvg_symbols):.1f}%)")

    keep = np.array([s in vocab_set for s in hvg_symbols])
    keep_genes = np.asarray(hvg_genes)[keep]
    if len(keep_genes) == 0:
        raise RuntimeError(f"zero matched genes (n_top_genes={n_top_genes})")

    x_full = x_log1p[:].copy()
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.tolist())}
    col_idx = [gene_to_idx[g] for g in keep_genes]

    out = ad.AnnData(
        X=x_full[:, col_idx].copy(),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Series(hvg_symbols)[keep].values),
    )
    out.var["ensg_id"] = keep_genes
    out.var["gene_symbol"] = pd.Series(hvg_symbols)[keep].values
    stats = {
        "n_top_genes_requested": n_top_genes,
        "hvg_selected": len(hvg_genes),
        "n_genes_in_vocab": len(keep_genes),
        "vocab_frac": round(len(matched) / len(hvg_symbols), 4),
        "flavor": "seurat (log1p X; D-EPI-1)",
    }
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out = OUT_DIR / "scgpt_epidermis_train.h5ad"
    test_out = OUT_DIR / "scgpt_epidermis_test.h5ad"
    if train_out.is_file() and test_out.is_file() and not args.force:
        log("outputs exist; use --force to rebuild")
        return 1

    t0 = time.time()
    vocab = load_vocab(VOCAB_PATH)
    log(f"vocab: {len(vocab)} tokens")

    a = ad.read_h5ad(SRC_H5AD)
    x = a.X
    if hasattr(x, "toarray"):
        x = x.toarray().astype(np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)
    log(f"source: {a.shape} X max={x.max():.3f} (log1p1e4 contract)")

    var_df = a.var.copy()
    var_df["gene_symbol"] = a.var_names.astype(str)

    # ---- panel build (1200 -> 4253 if a marker missing) ----
    panel, stats = build_panel(a, x, var_df, vocab, DEFAULT_N_TOP_GENES)
    symbols = panel.var["gene_symbol"].astype(str).tolist()
    check = marker_panel_check(symbols, EPI_MARKERS)
    deviation = None
    if not check["all_present"]:
        log(f"  marker check at {DEFAULT_N_TOP_GENES}: {check['n_present']}/"
            f"{check['n_markers_total']} present — raising to "
            f"{RAISED_N_TOP_GENES} (mirrors PREREG §4 rule)")
        panel, stats = build_panel(a, x, var_df, vocab, RAISED_N_TOP_GENES)
        symbols = panel.var["gene_symbol"].astype(str).tolist()
        check = marker_panel_check(symbols, EPI_MARKERS)
        deviation = {
            "rule": "PREREG §4 max_seq_len rule adapted to epidermis",
            "default_panel": DEFAULT_N_TOP_GENES,
            "raised_panel": RAISED_N_TOP_GENES,
            "reason": "epidermis marker missing at default panel",
            "marker_check_at_default": check,
        }
    log(f"final panel: {panel.n_vars} genes, marker check "
        f"{check['n_present']}/{check['n_markers_total']} present")

    # ---- obs columns the training/eval pipeline requires ----
    panel.obs["author_cell_type"] = panel.obs["ordinal_layer"].astype(str)
    panel.obs["cell_class_4"] = panel.obs["ordinal_layer"].astype(str)
    panel.obs["donor_id"] = "epidermis"
    panel.obs["leiden"] = "0"  # all cells pass the analysis-set filter
    ordinal_idx = panel.obs["ordinal_idx"].astype(int).to_numpy()
    labels = panel.obs["ordinal_layer"].astype(str).to_numpy()
    log(f"panel obs: n={panel.n_obs} labels="
        f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    # ---- frozen split: test 10%, val 10% of remaining, stratified ----
    strata = sorted(set(labels.tolist()))
    rng = np.random.default_rng(SPLIT_SEED)
    test_idx = np.zeros(0, dtype=np.int64)
    val_idx = np.zeros(0, dtype=np.int64)
    stratum_counts: dict[str, dict] = {}
    for c in strata:
        idx = np.where(labels == c)[0]
        perm = rng.permutation(len(idx))
        n_test = max(1, int(round(len(idx) * TEST_FRAC)))
        n_val = max(1, int(round(len(idx) * VAL_FRAC_OF_REMAINING)))
        test_idx = np.concatenate([test_idx, idx[perm[:n_test]]])
        val_idx = np.concatenate([val_idx, idx[perm[n_test:n_test + n_val]]])
        stratum_counts[c] = {
            "n_total": int(len(idx)),
            "n_test": int(n_test),
            "n_val": int(n_val),
            "n_train": int(len(idx) - n_test - n_val),
        }
    test_idx.sort()
    val_idx.sort()
    test_mask = np.zeros(panel.n_obs, dtype=bool)
    val_mask = np.zeros(panel.n_obs, dtype=bool)
    test_mask[test_idx] = True
    val_mask[val_idx] = True
    train_mask = ~(test_mask | val_mask)
    n_test = int(test_mask.sum())
    n_val = int(val_mask.sum())
    n_train = int(train_mask.sum())
    log(f"split: train={n_train} val={n_val} test={n_test} "
        f"(frac test={n_test / panel.n_obs:.4f})")
    log(f"  test class dist: "
        f"{dict(zip(*np.unique(labels[test_mask], return_counts=True)))}")

    np.savez(OUT_DIR / "split_indices.npz",
             test_idx=test_idx.astype(np.int64),
             val_idx=val_idx.astype(np.int64),
             n_total=panel.n_obs)

    # ---- write train / test h5ads ----
    # Train h5ad = all non-test cells (train + val, 16,519) — lock_val_split
    # convention: the val partition lives INSIDE the train h5ad and is masked
    # by val_indices.npz (identical to the van Zyl run6/convergence structure).
    tr = panel[~test_mask].to_memory().copy()
    tr.write(train_out)
    te = panel[test_mask].to_memory().copy()
    te.write(test_out)
    log(f"wrote {train_out.name} {tr.shape} / {test_out.name} {te.shape}")

    # ---- val npz (indices in TRAIN-space rows; lock_val_split convention) ----
    # train h5ad rows = non-test cells in original order; a val cell's row =
    # number of non-test cells before it (injective since val ⊂ non-test).
    val_pos = np.nonzero(val_mask)[0]
    non_test_pos = np.nonzero(~test_mask)[0]
    val_in_train = np.searchsorted(non_test_pos, val_pos).astype(np.int64)
    val_in_train.sort()
    assert len(np.unique(val_in_train)) == len(val_in_train), \
        "val-in-train indices must be unique"
    np.savez(OUT_DIR / "val_indices.npz", val_idx=val_in_train, n_total=len(non_test_pos))
    log(f"val npz (train h5ad space): {len(val_in_train)} unique indices "
        f"of {len(non_test_pos)} h5ad rows")

    # ---- manifests ----
    files = [SRC_H5AD, train_out, test_out,
             OUT_DIR / "split_indices.npz", OUT_DIR / "val_indices.npz"]
    hashes = {f.name: sha256_file(f) for f in files}

    split_manifest = {
        "task": "scGPT cross-tissue epidermis arm (approved supplementary)",
        "dataset": "Wang2020_GSE147424 (v14 product wang2020_processed.h5ad)",
        "source": str(SRC_H5AD),
        "source_sha256": hashes["wang2020_processed.h5ad"],
        "layer_order": LAYER_ORDER,
        "markers_required": EPI_MARKERS,
        "panel": {"n_genes": int(panel.n_vars), **stats},
        "marker_check": check,
        "panel_deviation": deviation,
        "split": {
            "rule": "test 10% + val 10% of remaining per ordinal stratum, "
                    "rng seed 42 (mirrors lock_val_split.py); cell-level "
                    "(no donor metadata in v14 product; samples S1-S17)",
            "n_cells_total": int(panel.n_obs),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_train_h5ad_rows": int(n_train + n_val),
            "stratum_counts": stratum_counts,
            "test_class_dist": {k: int(v) for k, v in zip(
                *np.unique(labels[test_mask], return_counts=True))},
        },
        "h5ad_sha256": {
            "scgpt_epidermis_train.h5ad": hashes["scgpt_epidermis_train.h5ad"],
            "scgpt_epidermis_test.h5ad": hashes["scgpt_epidermis_test.h5ad"],
            "split_indices.npz": hashes["split_indices.npz"],
            "val_indices.npz": hashes["val_indices.npz"],
        },
        "X_contract": "log1p(1e4) normalized (v14 product; scGPT continuous "
                      "input embedding)",
        "deviations": [
            "D-EPI-1: HVG on log1p X with seurat flavor (no raw-counts layer "
            "in the v14 product; protocol_02b used raw counts + "
            "seurat_v3/seurat)",
            "D-EPI-2: cell-level split (no donor_id in v14 product) instead "
            "of donor-level holdout",
        ],
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8")

    val_manifest = {
        "rule": "10% of the non-test cells stratified by ordinal_layer "
                "(seed 42), mirrors lock_val_split.py; val lives inside the "
                "train h5ad (16,519 rows) and is masked by val_indices.npz",
        "source": str(train_out),
        "n_cells_total": int(n_train + n_val),
        "n_train": n_train,
        "n_val": n_val,
        "val_frac": round(n_val / (n_train + n_val), 4),
        "seed": SPLIT_SEED,
        "stratum_counts": stratum_counts,
        "sha256": {
            "scgpt_vanzyl_train.h5ad": hashes["scgpt_epidermis_train.h5ad"],
            "val_indices.npz": hashes["val_indices.npz"],
        },
        "frozen_baseline_for_collapse": {
            "note": "filled after the frozen diagnostic (pre-training); "
                    "threshold = frozen rho - 0.15 (van Zyl convention)",
            "rho_frz_L2normed": None,
            "collapse_threshold": None,
        },
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (OUT_DIR / "split_val_manifest.json").write_text(
        json.dumps(val_manifest, indent=2), encoding="utf-8")

    log(f"manifests written; total wall {time.time() - t0:.0f}s")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
