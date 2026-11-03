#!/usr/bin/env python
"""Protocol 02c — scGPT prep for the two cornea2 datasets.

Mirrors protocol_02b_prep_scgpt_vanzyl.py (PREREG max_seq_len rule:
default panel = top-1200 HVG; raise to 4253 if any of the 13
differentiation markers is missing from HVG ∩ scGPT vocab).

Consumes data/cornea2/<ds>_annotated.h5ad (output of
protocol_15d_annotate_cornea2.py; raw counts in layers['counts'], var
index = gene symbols) and produces data/cornea2/<ds>_scgpt.h5ad with
X = log1p(1e4) normalized and .raw = raw counts.

Deltas vs protocol_02b (recorded; method otherwise locked):
1. var_names are gene symbols directly (no Ensembl feature_name mapping;
   protocol_02b mapped CELLxGENE Ensembl var_names via feature_name).
2. Output per dataset.

Usage
-----
    <venv>/bin/python3 scripts/protocol_02c_prep_scgpt_cornea2.py \
        --dataset GSE186433 [--force]
    <venv>/bin/python3 scripts/protocol_02c_prep_scgpt_cornea2.py \
        --dataset GSE155683 [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
OUT_DIR = DATA / "cornea2"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cornea2"
VOCAB_PATH = DATA / "scgpt_gene_vocab.json"

DEFAULT_N_TOP_GENES = 1200
RAISED_N_TOP_GENES = 4253

DIFF_MARKERS = [
    "KRT3", "KRT12", "KRT15", "TP63", "COL17A1", "ABCG2",
    "S100A7", "S100A8", "S100A9", "TGM1", "IVL", "DSC1", "ALDH3A1",
]

HVG_SUBSAMPLE = 50_000
RANDOM_SEED = 42


def load_vocab(path: Path) -> dict:
    with open(path) as f:
        vocab = json.load(f)
    logger.info("  ✓ Using cached vocab at %s (%d tokens)", path, len(vocab))
    return vocab


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


def build_panel(adata, counts_matrix, vocab, n_top_genes) -> ad.AnnData:
    rng = np.random.default_rng(RANDOM_SEED)
    n_subsample = min(HVG_SUBSAMPLE, counts_matrix.shape[0])
    idx = rng.choice(counts_matrix.shape[0], n_subsample, replace=False)
    idx.sort()

    logger.info("  Loading %d cells raw counts ...", n_subsample)
    x_sub = counts_matrix[idx].copy()
    tmp = ad.AnnData(X=x_sub, var=adata.var.copy())
    tmp.var_names = adata.var_names.copy()

    sc.pp.filter_genes(tmp, min_cells=3)
    logger.info("  After filter_genes(min_cells=3): %s", tmp.shape)

    hvg_genes = None
    for flavor in ("seurat_v3", "seurat"):
        try:
            logger.info("  Trying flavor='%s' ...", flavor)
            sc.pp.highly_variable_genes(
                tmp, n_top_genes=n_top_genes, flavor=flavor, subset=False,
            )
            mask = tmp.var["highly_variable"].values
            hvg_genes = tmp.var_names[mask].tolist()
            logger.info("  ✓ Selected %d HVGs with flavor='%s' "
                        "(n_top_genes=%d)", len(hvg_genes), flavor,
                        n_top_genes)
            break
        except Exception as exc:
            logger.warning("  ✗ flavor='%s' failed: %s", flavor, exc)
            continue
    if hvg_genes is None:
        raise RuntimeError(f"HVG selection failed for n_top_genes={n_top_genes}")

    hvg_symbols = tmp.var.loc[mask, "gene_symbol"].tolist()
    vocab_set = set(vocab.keys())
    matched_symbols = [s for s in hvg_symbols if s in vocab_set]
    pct = 100.0 * len(matched_symbols) / len(hvg_symbols)
    logger.info("  %d/%d HVG symbols found in scGPT vocab (%.1f%%)",
                len(matched_symbols), len(hvg_symbols), pct)

    hvg_sym_series = pd.Series(hvg_symbols, index=hvg_genes)
    keep_genes = hvg_sym_series[hvg_sym_series.isin(vocab_set)].index.tolist()
    logger.info("  Keeping %d genes for output", len(keep_genes))
    if len(keep_genes) == 0:
        raise RuntimeError(f"Zero matched genes (n_top_genes={n_top_genes})")

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.tolist())}
    col_idx = [gene_to_idx[g] for g in keep_genes]
    x_full = counts_matrix[:].copy()

    out = ad.AnnData(
        X=x_full[:, col_idx].copy(),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=keep_genes),
    )
    out.var["gene_symbol"] = [str(g) for g in keep_genes]

    out.raw = ad.AnnData(X=out.X.copy(), var=out.var.copy())

    logger.info("  Normalizing .X (log1p 1e4) ...")
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    logger.info("  Output shape: %d × %d", out.n_obs, out.n_vars)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["GSE186433", "GSE155683"],
                    required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    tag = ds.lower()

    input_path = OUT_DIR / f"{tag}_annotated.h5ad"
    output_path = OUT_DIR / f"{tag}_scgpt.h5ad"
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)
    if output_path.exists() and not args.force:
        logger.error("Output exists: %s (use --force)", output_path)
        raise SystemExit(1)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"STEP 1 — Loading annotated {ds} h5ad")
    logger.info("=" * 60)
    adata = ad.read_h5ad(input_path)
    logger.info("  Shape: %d × %d", adata.n_obs, adata.n_vars)
    if "counts" not in adata.layers:
        logger.error("  layers['counts'] missing")
        raise SystemExit(1)
    counts_matrix = adata.layers["counts"]
    logger.info("  Raw counts nnz=%d", counts_matrix.nnz)
    adata.var["gene_symbol"] = adata.var_names.astype(str)

    logger.info("=" * 60)
    logger.info("STEP 2 — Loading scGPT vocab")
    logger.info("=" * 60)
    vocab = load_vocab(VOCAB_PATH)

    logger.info("=" * 60)
    logger.info("STEP 3 — Panel build + 13-marker check (PREREG §4)")
    logger.info("=" * 60)
    panel = build_panel(adata, counts_matrix, vocab, DEFAULT_N_TOP_GENES)
    panel_symbols = panel.var["gene_symbol"].astype(str).tolist()
    check = marker_panel_check(panel_symbols, DIFF_MARKERS)
    logger.info("  Panel %d: 13-marker %d/%d%s", DEFAULT_N_TOP_GENES,
                check["n_present"], check["n_markers_total"],
                "" if check["all_present"] else f" MISSING {check['missing']}")

    deviation = None
    if not check["all_present"]:
        logger.warning("  ✗ Missing marker at %d — raising panel to %d",
                       DEFAULT_N_TOP_GENES, RAISED_N_TOP_GENES)
        panel = build_panel(adata, counts_matrix, vocab, RAISED_N_TOP_GENES)
        panel_symbols = panel.var["gene_symbol"].astype(str).tolist()
        check = marker_panel_check(panel_symbols, DIFF_MARKERS)
        deviation = {
            "rule": "PREREG-2026-08-11.md §4 max_seq_len",
            "default_panel": DEFAULT_N_TOP_GENES,
            "raised_panel": RAISED_N_TOP_GENES,
            "reason": "differentiation marker missing at default panel",
            "marker_check_at_default": check,
        }

    panel_check_result = {
        "dataset": ds,
        "panel_size": int(panel.n_vars),
        "n_top_genes_requested": (
            DEFAULT_N_TOP_GENES if deviation is None else RAISED_N_TOP_GENES
        ),
        "marker_check": check,
        "deviation": deviation,
        "diff_markers": DIFF_MARKERS,
    }
    (ARTIFACT_DIR / f"{tag}_scgpt_panel_check.json").write_text(
        json.dumps(panel_check_result, indent=2, ensure_ascii=False)
    )

    logger.info("=" * 60)
    logger.info("STEP 4 — Export")
    logger.info("=" * 60)
    panel.write(output_path)
    logger.info("  ✓ Done -> %s (cells=%d genes=%d)", output_path,
                panel.n_obs, panel.n_vars)


if __name__ == "__main__":
    main()
