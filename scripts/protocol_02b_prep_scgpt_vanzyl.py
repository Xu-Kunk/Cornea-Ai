#!/usr/bin/env python
"""scGPT data preparation for the van Zyl 2022 PNAS atlas (mirrors
protocol_02_prep_scgpt.py).

Consumes data/vanzyl_annotated.h5ad (output of
protocol_15c_annotate_vanzyl.py) and produces data/scgpt_vanzyl.h5ad:
HVG ∩ scGPT vocab panel, X = log1p(1e4) normalized, .raw = raw counts.

PREREG max_seq_len rule (PREREG-2026-08-11.md §4, LOCKED)
---------------------------------------------------------
Default panel = top-1200 HVG (scGPT official n_hvg convention). The 13
differentiation markers (KRT3/KRT12/KRT15/TP63/COL17A1/ABCG2/S100A7/
S100A8/S100A9/TGM1/IVL/DSC1/ALDH3A1) must ALL be present in the final
panel (HVG ∩ vocab). If any is missing the panel is raised to 4253 and
the deviation is recorded in the manifest.

Deltas vs protocol_02 (all recorded, method otherwise locked)
-------------------------------------------------------------
1. Input: data/vanzyl_annotated.h5ad; raw counts come from
   layers['counts'] (protocol_02 read adata.raw.X of the CELLxGENE file).
2. N_TOP_GENES default 1200 (protocol_02 used 5000) per PREREG §4; raises
   to 4253 only if a differentiation marker is missing.
3. Marker-panel check implemented here (13-marker set, PREREG §4).
4. Output file data/scgpt_vanzyl.h5ad (protocol_02 wrote
   corneal_scgpt_ready.h5ad).

Outputs
-------
* data/scgpt_vanzyl.h5ad     (HVG∩vocab panel; X=log1p(1e4); .raw=counts)
* artifacts/paper1_vanzyl/scgpt_panel_check.json  (13-marker check result)

Usage
-----
    <venv>/bin/python3 scripts/protocol_02b_prep_scgpt_vanzyl.py \
        [--input data/vanzyl_annotated.h5ad] \
        [--output data/scgpt_vanzyl.h5ad] [--force]
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

# ── paths ────────────────────────────────────────────────────────────────
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
DEFAULT_INPUT = DATA / "vanzyl_annotated.h5ad"
DEFAULT_OUTPUT = DATA / "scgpt_vanzyl.h5ad"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl"
VOCAB_PATH = DATA / "scgpt_gene_vocab.json"

# PREREG §4 panel sizes (LOCKED).
DEFAULT_N_TOP_GENES = 1200
RAISED_N_TOP_GENES = 4253

# 13 differentiation markers, PREREG §4 (LOCKED).
DIFF_MARKERS = [
    "KRT3", "KRT12", "KRT15", "TP63", "COL17A1", "ABCG2",
    "S100A7", "S100A8", "S100A9", "TGM1", "IVL", "DSC1", "ALDH3A1",
]

HVG_SUBSAMPLE = 50_000  # cells used for HVG computation (protocol_02 value)
RANDOM_SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────


def load_vocab(path: Path) -> dict:
    """Load the local scGPT gene vocabulary (already downloaded)."""
    with open(path) as f:
        vocab = json.load(f)
    logger.info("  ✓ Using cached vocab at %s (%d tokens)", path, len(vocab))
    return vocab


def disambiguate_symbols(var_df: pd.DataFrame) -> pd.DataFrame:
    """Append ENSG suffix to duplicate gene symbols (protocol_02 verbatim)."""
    sym = var_df["feature_name"].astype(str)
    dupe_mask = sym.duplicated(keep=False)
    if dupe_mask.sum() == 0:
        var_df["gene_symbol"] = sym.values
        return var_df
    unique_sym = sym.copy()
    unique_sym[dupe_mask.values] = (
        sym[dupe_mask.values] + "__" + var_df.index[dupe_mask.values]
    )
    var_df["gene_symbol"] = unique_sym.values
    n_dupes = int(dupe_mask.sum())
    logger.info("  Disambiguated %d duplicate symbols (suffix: __ENSG)", n_dupes)
    return var_df


def marker_panel_check(symbols: list[str], markers: list[str]) -> dict:
    """Check which of the 13 PREREG markers are in the panel (by symbol)."""
    sym_set = set(symbols)
    present = {m: (m in sym_set) for m in markers}
    missing = [m for m in markers if not present[m]]
    return {
        "n_markers_total": len(markers),
        "n_present": sum(present.values()),
        "missing": missing,
        "all_present": len(missing) == 0,
    }


# ── main pipeline ────────────────────────────────────────────────────────


def build_panel(
    adata: ad.AnnData,
    counts_matrix: np.ndarray,
    var_df: pd.DataFrame,
    vocab: dict,
    n_top_genes: int,
) -> ad.AnnData:
    """Run HVG selection + vocab match; return the panel AnnData (raw counts)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_subsample = min(HVG_SUBSAMPLE, counts_matrix.shape[0])
    idx = rng.choice(counts_matrix.shape[0], n_subsample, replace=False)
    idx.sort()

    logger.info("  Loading %d cells raw counts ...", n_subsample)
    x_sub = counts_matrix[idx].copy()
    logger.info("  Sub-matrix shape: %s, nnz: %d",
                x_sub.shape, x_sub.nnz)

    tmp = ad.AnnData(X=x_sub, var=var_df.copy())
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
        raise RuntimeError(
            f"HVG selection failed for n_top_genes={n_top_genes}"
        )

    hvg_symbols = tmp.var.loc[mask, "gene_symbol"].tolist()

    # Match HVGs to scGPT vocab.
    vocab_set = set(vocab.keys())
    matched_symbols = [s for s in hvg_symbols if s in vocab_set]
    pct = 100.0 * len(matched_symbols) / len(hvg_symbols)
    logger.info("  %d/%d HVG symbols found in scGPT vocab (%.1f%%)",
                len(matched_symbols), len(hvg_symbols), pct)

    hvg_sym_series = pd.Series(hvg_symbols, index=hvg_genes)
    keep_genes = hvg_sym_series[hvg_sym_series.isin(vocab_set)].index.tolist()
    logger.info("  Keeping %d genes for output", len(keep_genes))
    if len(keep_genes) == 0:
        raise RuntimeError(
            f"Zero matched genes (n_top_genes={n_top_genes})"
        )

    # Subset the FULL raw counts to the panel (no subsampling here).
    logger.info("  Building column index for subset ...")
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.tolist())}
    col_idx = [gene_to_idx[g] for g in keep_genes]

    logger.info("  Loading full raw counts into memory ...")
    x_full = counts_matrix[:].copy()
    logger.info("  Full matrix loaded: shape=%s, nnz=%d",
                x_full.shape, x_full.nnz)

    sym_index = var_df.loc[keep_genes, "gene_symbol"].values

    out = ad.AnnData(
        X=x_full[:, col_idx].copy(),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=sym_index),
    )
    out.var["ensg_id"] = keep_genes
    out.var["gene_symbol"] = sym_index

    # Store raw counts in .raw (unchanged).
    out.raw = ad.AnnData(
        X=out.X.copy(),
        var=out.var.copy(),
    )

    # Normalize .X (keep .raw unchanged).
    logger.info("  Normalizing .X (log1p 1e4) ...")
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)

    logger.info("  Output shape: %d × %d", out.n_obs, out.n_vars)
    return out


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="scGPT prep for van Zyl atlas (HVG∩vocab panel, "
                    "PREREG 1200/4253 rule)."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Input annotated h5ad (default: %(default)s)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output h5ad (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it already exists")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)
    if output_path.exists() and not args.force:
        logger.error("Output already exists: %s (use --force to overwrite)",
                     output_path)
        raise SystemExit(1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1 — Loading annotated van Zyl h5ad")
    logger.info("=" * 60)
    adata = ad.read_h5ad(input_path)
    logger.info("  Shape: %d × %d", adata.n_obs, adata.n_vars)
    logger.info("  Layers: %s", list(adata.layers.keys()))

    if "counts" not in adata.layers:
        logger.error("  layers['counts'] missing — raw counts not found. "
                     "Expected from protocol_15c output.")
        raise SystemExit(1)
    counts_matrix = adata.layers["counts"]
    logger.info("  Raw counts from layers['counts']: nnz=%d",
                counts_matrix.nnz)

    var_df = adata.var[["feature_name"]].copy()
    var_df = disambiguate_symbols(var_df)

    logger.info("=" * 60)
    logger.info("STEP 2 — Loading scGPT vocab")
    logger.info("=" * 60)
    vocab = load_vocab(VOCAB_PATH)

    # PREREG §4: try default 1200 panel; raise to 4253 if a 13-marker
    # differentiation marker is missing.
    logger.info("=" * 60)
    logger.info("STEP 3 — Panel build + 13-marker check (PREREG §4)")
    logger.info("=" * 60)

    panel = build_panel(adata, counts_matrix, var_df, vocab,
                        DEFAULT_N_TOP_GENES)
    panel_symbols = panel.var["gene_symbol"].astype(str).tolist()
    check = marker_panel_check(panel_symbols, DIFF_MARKERS)
    logger.info("  Panel size %d: 13-marker check -> %d/%d present%s",
                DEFAULT_N_TOP_GENES, check["n_present"],
                check["n_markers_total"],
                "" if check["all_present"]
                else f" (MISSING: {check['missing']})")

    deviation = None
    if not check["all_present"]:
        logger.warning("  ✗ Missing marker(s) at n_top_genes=%d — raising "
                       "panel to %d (PREREG §4 deviation).",
                       DEFAULT_N_TOP_GENES, RAISED_N_TOP_GENES)
        panel = build_panel(adata, counts_matrix, var_df, vocab,
                            RAISED_N_TOP_GENES)
        panel_symbols = panel.var["gene_symbol"].astype(str).tolist()
        check = marker_panel_check(panel_symbols, DIFF_MARKERS)
        logger.info("  Panel size %d: 13-marker check -> %d/%d present%s",
                    RAISED_N_TOP_GENES, check["n_present"],
                    check["n_markers_total"],
                    "" if check["all_present"]
                    else f" (STILL MISSING: {check['missing']})")
        deviation = {
            "rule": "PREREG-2026-08-11.md §4 max_seq_len",
            "default_panel": DEFAULT_N_TOP_GENES,
            "raised_panel": RAISED_N_TOP_GENES,
            "reason": "differentiation marker missing at default panel",
            "marker_check_at_default": check,
        }

    panel_check_result = {
        "panel_size": int(panel.n_vars),
        "n_top_genes_requested": (
            DEFAULT_N_TOP_GENES if deviation is None else RAISED_N_TOP_GENES
        ),
        "marker_check": check,
        "deviation": deviation,
        "diff_markers": DIFF_MARKERS,
    }
    (ARTIFACT_DIR / "scgpt_panel_check.json").write_text(
        json.dumps(panel_check_result, indent=2, ensure_ascii=False)
    )
    logger.info("  Saved %s", ARTIFACT_DIR / "scgpt_panel_check.json")

    logger.info("=" * 60)
    logger.info("STEP 4 — Export")
    logger.info("=" * 60)
    logger.info("  Writing to %s ...", output_path)
    panel.write(output_path)
    logger.info("  ✓ Done")

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Cells:                     %d", panel.n_obs)
    logger.info("  Panel genes (HVG∩vocab):   %d", panel.n_vars)
    logger.info("  13-marker all present:     %s", check["all_present"])
    if deviation is not None:
        logger.info("  DEVIATION (PREREG §4):     panel raised to %d",
                    RAISED_N_TOP_GENES)
    logger.info("  Output:                    %s", output_path)


if __name__ == "__main__":
    main()
