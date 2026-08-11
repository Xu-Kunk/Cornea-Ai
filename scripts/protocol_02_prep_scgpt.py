"""
scGPT data preparation: HVG filtering + gene vocabulary matching.

Produces:
  - data/corneal_scgpt_ready.h5ad:  Subset to HVGs matching scGPT vocab
  - data/scgpt_gene_vocab.json:     Downloaded scGPT token vocabulary

Strategy:
  - Subsampled cells (~50k) for HVG computation to keep memory manageable
  - raw counts matrix is ~13 GB sparse; full-load would be risky
  - HVG → symbol dedup (append ENSG to duplicate symbols) → vocab match → subset
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import json
import urllib.request
import urllib.error
import os
import ssl
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
SRC = DATA / "30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad"
OUT_H5AD = DATA / "corneal_scgpt_ready.h5ad"
OUT_VOCAB = str(DATA / "scgpt_gene_vocab.json")
VOCAB_URL = (
    "https://raw.githubusercontent.com/bowang-lab/scGPT/main/"
    "scgpt/tokenizer/default_gene_vocab.json"
)
N_TOP_GENES = 5000
HVG_SUBSAMPLE = 50_000  # cells used for HVG computation
RANDOM_SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────


def download_vocab(url: str, outpath: str):
    """Download scGPT gene vocab, return dict or None (fail → soft-fail).

    Tries GitHub Raw first, falls back to GitHub Content API (base64-decoded).
    """
    logger.info("Downloading scGPT gene vocab ...")

    if os.path.exists(outpath):
        with open(outpath) as f:
            vocab = json.load(f)
        logger.info("  ✓ Using cached vocab at %s (%d tokens)", outpath, len(vocab))
        return vocab

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # Method 1: direct raw URL
    for attempt in (url,):
        try:
            ctx = ssl._create_unverified_context()
            req = urllib.request.Request(attempt, headers={"User-Agent": "scGPT-prep"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                if resp.getcode() == 404:
                    raise urllib.error.HTTPError(attempt, 404, "Not Found", {}, None)
                vocab = json.loads(resp.read().decode())
            with open(outpath, "w") as f:
                json.dump(vocab, f)
            logger.info("  ✓ Downloaded %d tokens via raw URL", len(vocab))
            return vocab
        except urllib.error.HTTPError as e:
            if e.code != 404:
                logger.warning("HTTP %d on raw URL: %s", e.code, e.reason)
            break  # try fallback
        except Exception as e:
            logger.warning("Raw URL download failed: %s", e)
            break

    # Method 2: GitHub Content API (base64)
    import base64
    api_url = (
        "https://api.github.com/repos/bowang-lab/scGPT/contents/"
        "scgpt/tokenizer/default_gene_vocab.json"
    )
    try:
        req = urllib.request.Request(
            api_url, headers={"Accept": "application/vnd.github.v3+json",
                              "User-Agent": "scGPT-prep"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            content = base64.b64decode(data["content"]).decode("utf-8")
            vocab = json.loads(content)
        with open(outpath, "w") as f:
            json.dump(vocab, f)
        logger.info("  ✓ Downloaded %d tokens via GitHub API", len(vocab))
        return vocab
    except Exception as e:
        logger.warning("GitHub API download also failed: %s", e)
        logger.warning("All HVGs will be saved; matching deferred to fine-tune.")
        return None


def disambiguate_symbols(var_df: pd.DataFrame) -> pd.DataFrame:
    """Append ENSG suffix to duplicate gene symbols so var_names are unique."""
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
    n_dupes = dupe_mask.sum()
    logger.info("  Disambiguated %d duplicate symbols (suffix: __ENSG)", n_dupes)
    return var_df


# ── main ─────────────────────────────────────────────────────────────────


def main():
    # ================================================================
    # Step 1 — Source info
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading source h5ad (backed)")
    logger.info("=" * 60)
    adata = sc.read_h5ad(SRC, backed="r")
    n_cells, n_genes = adata.shape
    logger.info("  Shape: %d × %d", n_cells, n_genes)
    logger.info("  Has .raw: %s", adata.raw is not None)

    # var for downstream
    var_df = adata.var[["feature_name"]].copy()
    var_df = disambiguate_symbols(var_df)  # adds gene_symbol col

    # ================================================================
    # Step 2 — HVG selection on subsampled raw counts
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 2 — HVG selection (seurat_v3, subsample=%d)", HVG_SUBSAMPLE)
    logger.info("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)
    n_subsample = min(HVG_SUBSAMPLE, n_cells)
    idx = rng.choice(n_cells, n_subsample, replace=False)
    idx.sort()

    logger.info("  Loading %d cells raw counts ...", n_subsample)
    X_sub = adata.raw.X[idx].copy()  # sparse matrix, subset rows
    logger.info("  Sub-matrix shape: %s, nnz: %d", X_sub.shape, X_sub.nnz)
    logger.info("  Memory: data=%.1f MB, indices=%.1f MB",
                X_sub.data.nbytes / 1e6, X_sub.indices.nbytes / 1e6)

    tmp = ad.AnnData(X=X_sub, var=var_df.copy())
    tmp.var_names = adata.raw.var_names.copy()

    # remove genes with < 3 cells — saves seurat_v3 from edge-case issues
    sc.pp.filter_genes(tmp, min_cells=3)
    logger.info("  After filter_genes(min_cells=3): %s", tmp.shape)

    hvg_genes = None
    for flavor in ("seurat_v3", "seurat"):
        try:
            logger.info("  Trying flavor='%s' ...", flavor)
            sc.pp.highly_variable_genes(
                tmp, n_top_genes=N_TOP_GENES, flavor=flavor, subset=False,
            )
            mask = tmp.var["highly_variable"].values
            hvg_genes = tmp.var_names[mask].tolist()
            logger.info("  ✓ Selected %d HVGs with flavor='%s'", len(hvg_genes), flavor)
            break
        except Exception as exc:
            logger.warning("  ✗ flavor='%s' failed: %s", flavor, exc)
            continue

    if hvg_genes is None:
        logger.error("All HVG methods failed — aborting.")
        raise RuntimeError("HVG selection failed")

    hvg_symbols = tmp.var.loc[mask, "gene_symbol"].tolist()

    # ================================================================
    # Step 3 — Download scGPT vocab
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 3 — Download scGPT gene vocabulary")
    logger.info("=" * 60)
    vocab = download_vocab(VOCAB_URL, OUT_VOCAB)

    # ================================================================
    # Step 4 — Match HVGs to vocab
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 4 — Matching HVGs to scGPT vocabulary")
    logger.info("=" * 60)

    if vocab is not None:
        vocab_set = set(vocab.keys())
        matched_symbols = [s for s in hvg_symbols if s in vocab_set]
        n_matched = len(matched_symbols)
        pct = 100.0 * n_matched / len(hvg_symbols)
        logger.info("  %d/%d HVG symbols found in scGPT vocab (%.1f%%)",
                    n_matched, len(hvg_symbols), pct)

        # Build keep-mask over all genes
        hvg_sym_series = pd.Series(hvg_symbols, index=hvg_genes)
        keep_genes = hvg_sym_series[hvg_sym_series.isin(vocab_set)].index.tolist()
        logger.info("  Keeping %d genes for output", len(keep_genes))
    else:
        logger.warning("  No vocab — saving all %d HVGs", len(hvg_genes))
        keep_genes = hvg_genes

    if len(keep_genes) == 0:
        logger.error("No genes survived matching — aborting.")
        raise RuntimeError("Zero matched genes")

    # ================================================================
    # Step 5  — Subset full data and export
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 5 — Subsetting full dataset & exporting")
    logger.info("=" * 60)

    logger.info("  Building column index for subset ...")
    gene_to_idx = {g: i for i, g in enumerate(adata.raw.var_names.tolist())}
    col_idx = [gene_to_idx[g] for g in keep_genes]

    logger.info("  Loading full raw counts into memory ...")
    X_full = adata.raw.X[:].copy()
    logger.info("  Full matrix loaded: shape=%s, nnz=%d", X_full.shape, X_full.nnz)

    # Build output AnnData
    #   var_names = gene symbols (scGPT matches on symbols)
    #   .raw      = raw counts reference for scGPT fine-tune
    #   .X        = log1p-normalized counts for downstream exploration
    sym_index = var_df.loc[keep_genes, "gene_symbol"].values

    out = ad.AnnData(
        X=X_full[:, col_idx].copy(),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=sym_index),
    )
    out.var["ensg_id"] = keep_genes
    out.var["gene_symbol"] = sym_index

    # Store raw counts in .raw
    out.raw = ad.AnnData(
        X=out.X.copy(),
        var=out.var.copy(),
    )

    # Normalize .X (keep .raw unchanged)
    logger.info("  Normalizing .X (log1p) ...")
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)

    logger.info("  Output shape: %d × %d", out.n_obs, out.n_vars)
    logger.info("  Writing to %s ...", OUT_H5AD)
    out.write(OUT_H5AD)
    logger.info("  ✓ Done")

    # ================================================================
    # Summary
    # ================================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Cells (original → output):  %d → %d", n_cells, out.n_obs)
    logger.info("  Genes (original → output):  %d → %d", n_genes, out.n_vars)
    logger.info("  HVG genes selected:          %d", len(hvg_genes))
    if vocab is not None:
        logger.info("  Matched in scGPT vocab:      %d (%.1f%%)",
                    out.n_vars, 100 * out.n_vars / len(hvg_genes))
    logger.info("  Output:                      %s", OUT_H5AD)
    logger.info("  Vocab:                       %s",
                OUT_VOCAB if os.path.exists(OUT_VOCAB) else "N/A (deferred)")


if __name__ == "__main__":
    main()
