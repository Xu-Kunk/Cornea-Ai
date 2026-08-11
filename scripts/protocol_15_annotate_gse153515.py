#!/usr/bin/env python
"""Protocol 15 — Annotate the GSE153515 human limbal stem cell scRNA-seq atlas.

Input `data/gse153515_limbal_stem.h5ad` (17,587 cells x 32,738 genes) ships with
completely EMPTY obs (no columns). This script recovers the study's sample
structure and reconstructs cell-type annotations from published marker genes,
following the original Seurat v3.1 pipeline. The reference paper used CCA
integration (20 PCs); here Harmony integration (scanpy equivalent) removes the
donor batch effect, with clustering resolution 0.6.

References
----------
* Li D-Q, Kim S, et al. Single-cell transcriptomics of human limbal stem cells.
  Ocul Surf. 2021;20:1-13.  doi: 10.1016/j.jtos.2020.12.004  (PMID: 33388438)
* Kim S, et al. Characterization of human limbal stem cells by single-cell
  transcriptomics. IOVS. 2021;62(9):36.  doi: 10.1167/iovs.62.9.36
  (PMID: 34297801)
* Reference pipeline: github.com/sangbaekim/10xRNAseq_LSC (Seurat v3.1, CCA,
  resolution 0.6, 20 PCs).

Sample structure (cellranger aggr barcode suffix):
* `-1` -> Donor 1 (GSM4646295, 7,096 cells)
* `-2` -> Donor 2 (GSM4646296, 10,491 cells)

Outputs
-------
* data/gse153515_annotated.h5ad            (annotated, filtered, preprocessed)
* artifacts/gse153515_annotation/umap_celltype.pdf   (raw + harmonized cell type)
* artifacts/gse153515_annotation/umap_donor.pdf      (donor + leiden cluster)
* artifacts/gse153515_annotation/marker_scores.csv   (per-cluster DEG enrichment scores + top DEGs)

Pipeline order (important)
--------------------------
Marker scoring runs on the FULL gene set BEFORE QC filtering. Rare cell-type
markers (e.g. LSC A2M/ABCG2, proliferation MKI67/TOP2A/PLK1) are expressed in
<3% of cells and would be removed by the min_cells gene filter; scoring first
preserves their signal. Scores are stored in obs and survive the cell-level QC
subsetting unchanged, then QC + clustering run on high-quality cells only:
Step1 donors -> Step2 marker scoring (full genes) -> Step3 QC filter ->
Step4 preprocess + Harmony integration + cluster -> Step5 cell-type assignment
-> Step6 harmonize -> Step7 save.

Harmony (Step 4) is the scanpy equivalent of the reference paper's Seurat CCA
integration: PCA (30 PCs) is computed on the scaled data, then the embedding is
batch-corrected across donors (donor_id) before neighbors/UMAP/leiden, which
removes donor-specific clusters with zero marker enrichment.

Cell-type assignment is DEG-BASED (Step 5), mirroring the reference paper's
Seurat FindAllMarkers -> manual annotation workflow. sc.tl.rank_genes_groups
(wilcoxon) identifies the top 200 DEGs per Leiden cluster; each cell-type
marker set is then scored by enrichment: how many of its markers appear in the
cluster's top-200 DEGs and how high they rank
(enrichment = n_in_top200 / n_total * (1 - mean_rank / 200)). The cluster is
labeled with the best-scoring cell type. Fixed marker-set scoring is
deliberately avoided because the PMC_basal markers (KRT14/15/5) are broadly
expressed in ALL limbal epithelial cells and would dominate every cluster
(e.g. basal ~98%). Non-epithelial types (melanocyte/conjunctival/immune) are
assigned only when a cluster's enrichment exceeds NON_EPI_THRESHOLD (0.3) and
is the maximum across all marker sets; a cluster whose top enrichment is
'Immune' is flagged 'immune_contamination' (contaminating leukocytes/APCs).

Harmonization (4-class scheme)
------------------------------
Cell types are harmonized onto the project-standard 4-class differentiation
axis LAYER_ORDER = ['LSC', 'TAC', 'basal', 'differentiated'] (+ NON_LAYER for
non-epithelial cells). Corneal epithelial differentiation is a continuum
LSC -> TAC -> basal -> wing -> squamous (Cotsarelis et al., Cell 1989,
doi: 10.1016/0092-8674(89)90292-0); because GSE153515 annotations do not
distinguish wing from squamous at the cluster level, these are merged as
"differentiated" following standard limbal epithelial staging.

Usage
-----
    python3 scripts/protocol_15_annotate_gse153515.py \
        [--input data/gse153515_limbal_stem.h5ad] \
        [--output data/gse153515_annotated.h5ad] [--force] \
        [--seed 42] [--resolution 0.6]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse as sp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "gse153515_limbal_stem.h5ad"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "gse153515_annotation"

SEED = 42
RESOLUTION = 0.6
N_HVG = 2000
N_PCS = 30  # PCA components fed to Harmony integration (the paper used 20 PCs with CCA)
N_DEG = 200  # top-N DEGs per cluster used by the Step 5 enrichment scoring

# Enrichment threshold for assigning a cluster to a non-epithelial type (Step 5).
NON_EPI_THRESHOLD = 0.3

# Barcode suffix -> (donor_id, GEO sample accession, expected cell count)
DONOR_MAP = {
    "1": ("donor_1", "GSM4646295", 7096),
    "2": ("donor_2", "GSM4646296", 10491),
}

# Project-standard 4-class corneal/limbal epithelial differentiation axis.
# Corneal epithelial differentiation is a continuum
# LSC -> TAC -> basal -> wing -> squamous (Cotsarelis et al., Cell 1989,
# doi: 10.1016/0092-8674(89)90292-0). Since the GSE153515 annotations do not
# distinguish wing from squamous at the cluster level, these are merged into a
# single "differentiated" class following standard limbal epithelial staging.
LAYER_ORDER = ["LSC", "TAC", "basal", "differentiated"]

# Published marker sets (paper Results + Supplementary Table 5), expanded with
# the paper's actual top DEGs per cluster (e.g. ECSCR/RAMP3/PLVAP/AQP1 for the
# LSC cluster C10, CLU/IGFBP6/ALDH3A1/SPARC/HTRA1/... for basal clusters
# C1/C3/C4, ADIRF/MAL2/CLDN7/LYPD3/MT1X/HSPB1/TACSTD2 for suprabasal clusters
# C0/C2). 'Immune' is a contamination-detection set (PTPRC/CD45, T-cell and
# myeloid markers) — clusters with Immune as top enrichment are flagged.
MARKER_SETS = {
    "LSC": ["A2M", "ABCG2", "ITGA9", "TSPAN7", "SOX17",
            "ECSCR", "RAMP3", "PLVAP", "AQP1"],
    "LPC": ["KRT14", "KRT15", "KRT19", "DCN", "PLIN2", "SLC2A3", "ITM2A", "TCF4"],
    "TAC": ["MKI67", "BIRC5", "RRM2", "TOP2A", "PLK1", "TK1"],
    "PMC_basal": ["KRT14", "KRT15", "KRT5", "COL17A1", "ITGB1",  # post-mitotic basal
                  "CLU", "IGFBP6", "ALDH3A1", "SPARC", "HTRA1",
                  "NQO1", "S100A10", "CITED2"],
    "TDC": ["KRT12", "KRT3", "GJA1", "SPRR2A", "IVL",
            "ADIRF", "MAL2", "CLDN7", "LYPD3", "MT1X", "HSPB1", "TACSTD2"],
    "Melanocyte": ["TYR", "TYRP1", "MLANA", "PMEL", "DCT"],
    "Conjunctival": ["KRT4", "KRT7", "KRT13", "MUC5AC", "TFF3"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD14", "CD68", "FCGR3A", "HLA-DRA"],
}

# Harmonization to the 4-class scheme (NON_LAYER = non-epithelial).
HARMONIZE_MAP = {
    "LSC": "LSC",
    "LPC": "LSC",           # limbal progenitor -> LSC (same niche, KRT15+)
    "TAC": "TAC",
    "PMC_basal": "basal",   # post-mitotic -> basal
    "TDC": "differentiated",  # terminally differentiated (KRT12/KRT3+); wing and
                              # squamous are not separable at the GSE153515
                              # cluster level, so merged as "differentiated"
    "melanocyte": "NON_LAYER",
    "conjunctival": "NON_LAYER",
    "immune_contamination": "NON_LAYER",
}

# ---------------------------------------------------------------------------
# Gene symbol <-> Ensembl ID resolution
# ---------------------------------------------------------------------------
# NOTE: In data/gse153515_limbal_stem.h5ad BOTH the var index and the
# var['gene_symbol'] column store Ensembl IDs (100% of rows match ^ENSG), so a
# direct `var['gene_symbol'] == symbol` lookup returns nothing. Marker symbols
# are therefore resolved to Ensembl IDs through this map. IDs were taken from
# the project's own CELLxGENE file (data/corneal_100k_subsample.h5ad,
# var['feature_name'] -> var_names) and ALL 64 marker Ensembl IDs (37 original
# + 27 added with the expanded marker sets) were verified present in the
# GSE153515 var index AND symbol-correct against the CELLxGENE feature_name
# map (queried directly on 2026-07-31). The alternate IDs sometimes suggested
# for TSPAN7/SOX17/ITGA9/ITM2A (ENSG00000005810, ENSG00000106524,
# ENSG00000138496, ENSG00000135924) resolve to different genes (MYCBP2,
# ANKMY2, PARP9, DNAJB2) and must NOT be used. Likewise, the following IDs
# often proposed for the newly added markers resolve to OTHER genes and must
# NOT be used: ECSCR ENSG00000169242 (=EFNA1), RAMP3 ENSG00000122678 (=POLM),
# PLVAP ENSG00000005302 (=MSL3), IGFBP6 ENSG00000137463 (=MGARP), HTRA1
# ENSG00000117289 (not in map), CITED2 ENSG00000156273 (=BACH1), ADIRF
# ENSG00000170961 (=HAS2), MAL2 ENSG00000137501 (=SYTL2), CLDN7 ENSG00000181856
# (=SLC2A4), LYPD3 ENSG00000105397 (=TYK2), MT1X ENSG00000197093 (=GAL3ST4),
# TFF3 ENSG00000186075 (=ZPBP2), AQP1 ENSG00000240594 (not in map), S100A10
# ENSG00000182130 (not in map). If the input file ever carries real symbols in
# var['gene_symbol'], the script switches to direct symbol matching
# automatically.
ENSEMBL_MAP = {
    # LSC
    "A2M": "ENSG00000175899", "ABCG2": "ENSG00000118777",
    "ITGA9": "ENSG00000144668", "TSPAN7": "ENSG00000156298",
    "SOX17": "ENSG00000164736",
    "ECSCR": "ENSG00000249751", "RAMP3": "ENSG00000122679",
    "PLVAP": "ENSG00000130300", "AQP1": "ENSG00000240583",
    # LPC
    "KRT14": "ENSG00000186847", "KRT15": "ENSG00000171346",
    "KRT19": "ENSG00000171345", "DCN": "ENSG00000011465",
    "PLIN2": "ENSG00000147872", "SLC2A3": "ENSG00000059804",
    "ITM2A": "ENSG00000078596", "TCF4": "ENSG00000196628",
    # TAC
    "MKI67": "ENSG00000148773", "BIRC5": "ENSG00000089685",
    "RRM2": "ENSG00000171848", "TOP2A": "ENSG00000131747",
    "PLK1": "ENSG00000166851", "TK1": "ENSG00000167900",
    # PMC_basal
    "KRT5": "ENSG00000186081", "COL17A1": "ENSG00000065618",
    "ITGB1": "ENSG00000150093",
    "CLU": "ENSG00000120885", "IGFBP6": "ENSG00000167779",
    "ALDH3A1": "ENSG00000108602", "SPARC": "ENSG00000113140",
    "HTRA1": "ENSG00000166033", "NQO1": "ENSG00000181019",
    "S100A10": "ENSG00000197747", "CITED2": "ENSG00000164442",
    # TDC
    "KRT12": "ENSG00000187242", "KRT3": "ENSG00000186442",
    "GJA1": "ENSG00000152661", "SPRR2A": "ENSG00000241794",
    "IVL": "ENSG00000163207",
    "ADIRF": "ENSG00000148671", "MAL2": "ENSG00000147676",
    "CLDN7": "ENSG00000181885", "LYPD3": "ENSG00000124466",
    "MT1X": "ENSG00000187193", "HSPB1": "ENSG00000106211",
    "TACSTD2": "ENSG00000184292",
    # Melanocyte
    "TYR": "ENSG00000077498", "TYRP1": "ENSG00000107165",
    "MLANA": "ENSG00000120215", "PMEL": "ENSG00000185664",
    "DCT": "ENSG00000080166",
    # Conjunctival
    "KRT4": "ENSG00000170477", "KRT7": "ENSG00000135480",
    "KRT13": "ENSG00000171401", "MUC5AC": "ENSG00000215182",
    "TFF3": "ENSG00000160180",
    # Immune (contamination detection)
    "PTPRC": "ENSG00000081237", "CD3D": "ENSG00000167286",
    "CD3E": "ENSG00000198851", "CD14": "ENSG00000170458",
    "CD68": "ENSG00000129226", "FCGR3A": "ENSG00000203747",
    "HLA-DRA": "ENSG00000204287",
}


def log(msg: str) -> None:
    """Print a message with a timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Step 1: Donor assignment from cellranger aggr barcode suffixes
# ---------------------------------------------------------------------------

def assign_donors(adata: ad.AnnData) -> None:
    banner("Step 1: Donor assignment (barcode suffix)")
    suffixes = np.array([b.rsplit("-", 1)[-1] for b in adata.obs_names])
    unknown = set(suffixes) - set(DONOR_MAP)
    if unknown:
        raise ValueError(f"Unexpected barcode suffixes (not -1/-2): {sorted(unknown)!r}")

    adata.obs["donor_id"] = pd.Categorical(
        [DONOR_MAP[s][0] for s in suffixes]
    )
    adata.obs["geo_accession"] = pd.Categorical(
        [DONOR_MAP[s][1] for s in suffixes]
    )

    counts = adata.obs["donor_id"].value_counts()
    for suffix, (donor, geo, expected) in DONOR_MAP.items():
        actual = int(counts.get(donor, 0))
        assert actual == expected, (
            f"Donor count mismatch: {donor} ({geo}) has {actual} cells, "
            f"expected {expected}"
        )
        log(f"  {donor} ({geo}): {actual} cells (expected {expected}) — OK")
    log("  Donor assignment verified (7096 / 10491).")


# ---------------------------------------------------------------------------
# Step 2: Marker-gene scoring on the FULL gene set (before QC)
# ---------------------------------------------------------------------------
# Marker genes for rare cell types (e.g., LSC markers A2M/ABCG2, proliferation
# markers MKI67/TOP2A) are expressed in <3% of cells and would be removed by the
# min_cells QC filter. Scoring on the full gene set preserves these signals.
# This follows standard practice: cluster on high-quality cells, but score
# markers on the full gene set.

def _setup_var_names(adata: ad.AnnData) -> None:
    """Expose gene symbols in var_names when the file actually carries them.

    GSE153515 ships with Ensembl IDs in var['gene_symbol'], in which case
    var_names are left as-is and marker symbols are resolved via ENSEMBL_MAP
    (see resolve_marker_genes). If a future file carries real symbols, this
    makes score_genes match directly.
    """
    gs = adata.var["gene_symbol"].astype(str)
    has_real_symbols = bool((~gs.str.startswith("ENSG")).any())
    if has_real_symbols:
        adata.var["ensembl_id"] = adata.var_names
        adata.var_names = gs
        adata.var_names_make_unique()
        log("  var_names set to gene symbols (original IDs kept in var['ensembl_id']).")
    else:
        log("  var['gene_symbol'] contains Ensembl IDs; keeping var_names as-is "
            "and resolving marker symbols via ENSEMBL_MAP.")


def resolve_marker_genes(adata: ad.AnnData, symbols: list[str]) -> tuple[list[str], list[str]]:
    """Return (available, missing) marker names in adata.var_names space."""
    var_set = set(adata.var_names.astype(str))
    available: list[str] = []
    missing: list[str] = []
    for sym in symbols:
        candidates = [sym]
        ens = ENSEMBL_MAP.get(sym)
        if ens:
            candidates.append(ens)
        hit = next((c for c in candidates if c in var_set), None)
        if hit is not None:
            available.append(hit)
        else:
            missing.append(sym)
    return available, missing


def score_marker_sets(adata: ad.AnnData) -> None:
    """Score marker sets on the FULL gene set BEFORE QC filtering.

    Runs on the complete 32,738-gene matrix (var_names still carry every
    Ensembl ID). Because score_genes writes into adata.obs (one row per cell),
    the scores survive the subsequent cell filtering in Step 3 unchanged.
    """
    banner("Step 2: Marker-gene scoring on FULL gene set (before QC)")
    for name, genes in MARKER_SETS.items():
        avail, missing = resolve_marker_genes(adata, genes)
        if missing:
            log(f"  WARNING [{name}]: {len(missing)}/{len(genes)} genes missing "
                f"({missing}) — continuing with {len(avail)} available.")
        if not avail:
            log(f"  WARNING [{name}]: no genes available; score left as NaN.")
            adata.obs[f"score_{name}"] = np.nan
            continue
        sc.tl.score_genes(adata, gene_list=avail, score_name=f"score_{name}")
        log(f"  score_{name}: {len(avail)} genes -> {avail}")


# ---------------------------------------------------------------------------
# Step 3: QC filtering (mirrors the reference R script)
# ---------------------------------------------------------------------------

def qc_filter(adata: ad.AnnData) -> None:
    banner("Step 3: QC filtering (for clustering; marker scores already computed)")
    n_total = adata.n_obs

    # Genes: keep those detected in >= 3% of cells (R: cut_cell <- round(dim*0.03))
    min_cells = round(n_total * 0.03)
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    log(f"  Gene filter: min_cells={min_cells} -> {adata.n_vars} genes retained "
        f"(was {n_genes_before}).")

    # Cells: keep n_genes within the 5th-95th percentile (R: quantile(nGene, c(.05,.95)))
    n_genes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    lo, hi = np.quantile(n_genes, [0.05, 0.95])
    keep = (n_genes >= lo) & (n_genes <= hi)
    adata.obs["n_genes"] = n_genes.astype(int)
    log(f"  Cell filter: n_genes in [{lo:.0f}, {hi:.0f}] (5th-95th pct) -> "
        f"{int(keep.sum())}/{n_total} cells retained.")
    adata._inplace_subset_obs(keep)

    # Store raw counts BEFORE any normalization.
    adata.layers["counts"] = sp.csr_matrix(adata.X)
    log(f"  Raw counts stored in layers['counts'] ({adata.shape[0]} x {adata.shape[1]}).")


# ---------------------------------------------------------------------------
# Step 4: Preprocessing (scanpy equivalent of the Seurat pipeline)
# ---------------------------------------------------------------------------

def preprocess(adata: ad.AnnData, seed: int, resolution: float) -> None:
    banner("Step 4: Preprocessing (Seurat-equivalent)")

    # Normalize first (required for 'seurat' HVG flavor which operates on
    # log-normalized data — matches R FindVariableGenes on normalized data).
    # NOTE: 'seurat_v3' flavor requires scikit-misc (unavailable on aarch64);
    # 'seurat' flavor is the scanpy equivalent of Seurat v2 FindVariableGenes.
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    adata.layers["normalized"] = adata.X.copy()
    log("  Normalized (target_sum=10_000) + log1p; stored in layers['normalized'].")

    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, flavor="seurat", subset=False
    )
    log(f"  Highly variable genes: {int(adata.var['highly_variable'].sum())}/{adata.n_vars} "
        f"(flavor='seurat', n_top_genes={N_HVG}).")

    sc.pp.scale(adata, max_value=10)
    log("  Scaled (max_value=10).")

    sc.tl.pca(adata, n_comps=N_PCS, random_state=seed)
    log(f"  PCA: {N_PCS} components (random_state={seed}).")

    # Harmony integration (scanpy equivalent of the paper's Seurat CCA):
    # batch-correct the PCA embedding across donors before clustering, which
    # removes donor-batch clusters with zero marker enrichment.
    try:
        import harmonypy as hpy

        ho = hpy.run_harmony(
            adata.obsm["X_pca"], adata.obs, ["donor_id"],
            random_state=seed, max_iter_harmony=20,
        )
        adata.obsm["X_harmony"] = np.asarray(ho.Z_corr)
        log("  Harmony integration (harmonypy 2.x, direct API): corrected "
            "embedding stored in obsm['X_harmony'].")
    except ImportError:
        # Fallback: scanpy's built-in wrapper. NOTE (verified 2026-07-31):
        # sc.external.pp.harmony_integrate ALSO imports harmonypy internally,
        # so this path still requires the package — it only helps if a scanpy
        # build bundles harmonypy differently.
        log("  harmonypy not found — falling back to "
            "sc.external.pp.harmony_integrate(adata, key='donor_id').")
        sc.external.pp.harmony_integrate(adata, key="donor_id",
                                         adjusted_basis="X_harmony")
        log("  Harmony integration (scanpy wrapper) done -> obsm['X_harmony'].")

    sc.pp.neighbors(adata, use_rep="X_harmony", random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    log("  Neighbors (use_rep='X_harmony') + UMAP (random_state={}) done.".format(seed))

    sc.tl.leiden(adata, resolution=resolution, random_state=seed)
    log(f"  Leiden clustering: resolution={resolution}, "
        f"{adata.obs['leiden'].nunique()} clusters (random_state={seed}).")


# ---------------------------------------------------------------------------
# Step 5: Cell-type assignment (DEG enrichment, FindAllMarkers workflow)
# ---------------------------------------------------------------------------

def assign_cell_types(adata: ad.AnnData) -> pd.DataFrame:
    """Assign cell types per Leiden cluster from differential expression.

    Step 5a: sc.tl.rank_genes_groups (wilcoxon, top 200 DEGs per cluster) —
    the scanpy equivalent of Seurat's FindAllMarkers, which is what the
    reference paper (PMID 33388438) actually used for manual annotation.

    Step 5b: for each cluster, score each cell-type marker set by DEG
    enrichment:
        enrichment = (n_markers_in_top200 / n_total_markers) * (1 - mean_rank / 200)
    i.e. how many of the set's markers appear in the cluster's top-200 DEGs
    and how high they rank.

    Step 5c: assign the best-scoring cell type per cluster. Non-epithelial
    types (melanocyte/conjunctival) are assigned only when their enrichment
    exceeds NON_EPI_THRESHOLD and is the maximum across all marker sets. A
    cluster whose top enrichment is the 'Immune' set is flagged
    'immune_contamination' (contaminating leukocytes/APCs) and harmonizes to
    NON_LAYER.

    Fixed marker-set scoring is deliberately avoided because the PMC_basal
    markers (KRT14/15/5) are broadly expressed in ALL limbal epithelial cells
    and dominate any fixed-score comparison (cluster-mean scoring labels
    ~98% of clusters as PMC_basal).

    Returns an enrichment DataFrame (one row per cluster: enrichment per cell
    type, n_cells, assigned type, top-5 DEGs) used by the Step 7 report.
    """
    banner("Step 5: DEG-based cluster cell type assignment (FindAllMarkers workflow)")

    # Step 5a: differential expression per Leiden cluster (wilcoxon).
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", n_genes=N_DEG)
    res = adata.uns["rank_genes_groups"]
    # scanpy stores group names as field names of the structured array,
    # not as a separate 'groups' key.
    deg_groups = list(res["names"].dtype.names)
    deg_by_cluster = {g: list(res["names"][g]) for g in deg_groups}

    # Reverse map (Ensembl ID -> symbol) for displaying the top DEGs; only the
    # 64 curated markers are resolvable — every other var_name stays as-is.
    ensembl_to_symbol = {v: k for k, v in ENSEMBL_MAP.items()}

    def display(name: str) -> str:
        return ensembl_to_symbol.get(name, name)

    def marker_candidates(sym: str) -> list[str]:
        """Candidate var_names for a marker symbol (symbol and/or Ensembl ID)."""
        cands = [sym]
        ens = ENSEMBL_MAP.get(sym)
        if ens:
            cands.append(ens)
        return cands

    epithelial_types = ["LSC", "LPC", "TAC", "PMC_basal", "TDC"]
    cluster_labels = {}
    enrich_rows = {}

    # Steps 5b + 5c: per-cluster enrichment scoring and assignment.
    for cluster_id in sorted(deg_by_cluster, key=lambda x: int(x)):
        deg_names = deg_by_cluster[cluster_id]
        deg_rank = {name: r for r, name in enumerate(deg_names)}  # 0-based rank in top-N_DEG
        n_cells = int((adata.obs["leiden"] == cluster_id).sum())

        enrich = {}
        for ct, symbols in MARKER_SETS.items():
            ranks = []
            for sym in symbols:
                hit = next(
                    (deg_rank[c] for c in marker_candidates(sym) if c in deg_rank),
                    None,
                )
                if hit is not None:
                    ranks.append(hit)
            if not ranks:
                enrich[ct] = 0.0
            else:
                enrich[ct] = (len(ranks) / len(symbols)) * (1.0 - np.mean(ranks) / N_DEG)

        # Non-epithelial priority (threshold-gated). Immune contamination takes
        # precedence, then melanocyte, then conjunctival.
        if enrich["Immune"] > NON_EPI_THRESHOLD and enrich["Immune"] == max(enrich.values()):
            assigned = "immune_contamination"
        elif enrich["Melanocyte"] > NON_EPI_THRESHOLD and enrich["Melanocyte"] == max(enrich.values()):
            assigned = "melanocyte"
        elif enrich["Conjunctival"] > NON_EPI_THRESHOLD and enrich["Conjunctival"] == max(enrich.values()):
            assigned = "conjunctival"
        else:
            assigned = max(epithelial_types, key=lambda t: enrich[t])

        cluster_labels[cluster_id] = assigned
        top5 = [display(x) for x in deg_names[:5]]
        enrich_rows[cluster_id] = {
            **enrich,
            "n_cells": n_cells,
            "assigned": assigned,
            **{f"top_deg_{i + 1}": top5[i] for i in range(len(top5))},
        }

    # Print the full assignment table (cluster, type, n_cells, enrichments, top5 DEGs).
    log("\n  Cluster -> Cell Type assignment (DEG enrichment, top-%d DEGs):" % N_DEG)
    log(f"  {'Cl':>4} {'Assigned':>14} {'n':>7} | "
        + " | ".join(f"{ct:>11}" for ct in MARKER_SETS)
        + " | top 5 DEGs")
    for cluster_id in sorted(cluster_labels, key=lambda x: int(x)):
        r = enrich_rows[cluster_id]
        enrich_str = " | ".join(f"{r[ct]:>11.3f}" for ct in MARKER_SETS)
        log(f"  {cluster_id:>4} {cluster_labels[cluster_id]:>14} {r['n_cells']:>7} | "
            f"{enrich_str} | "
            + ", ".join(f"{r[f'top_deg_{i}']}" for i in range(1, 6)))

    # Apply cluster labels to all cells.
    adata.obs["cell_type_raw"] = adata.obs["leiden"].map(cluster_labels)

    counts = adata.obs["cell_type_raw"].value_counts()
    log("\n  Cell type assignment (DEG enrichment):")
    for ct, n in counts.items():
        log(f"    {ct}: {n} ({100 * n / adata.n_obs:.1f}%)")

    # Provenance: cluster -> celltype mapping and per-cluster enrichment scores.
    adata.uns["annotation"]["cluster_to_celltype"] = cluster_labels
    adata.uns["annotation"]["cluster_enrichment"] = {
        k: {ct: v[ct] for ct in MARKER_SETS} for k, v in enrich_rows.items()
    }
    adata.uns["annotation"]["method"] = "deg_enrichment_cluster_annotation"
    adata.uns["annotation"]["deg_method"] = "wilcoxon"
    adata.uns["annotation"]["deg_n_genes"] = N_DEG

    enrich_df = pd.DataFrame.from_dict(enrich_rows, orient="index")
    enrich_df.index.name = "leiden"
    return enrich_df


# ---------------------------------------------------------------------------
# Step 6: Harmonization to the 4-class scheme
# ---------------------------------------------------------------------------

def harmonize(adata: ad.AnnData) -> None:
    banner("Step 6: Harmonization to 4-class scheme")
    raw = adata.obs["cell_type_raw"].astype(str)
    harmonized = np.array([HARMONIZE_MAP.get(t, "NON_LAYER") for t in raw])
    layer = np.array([LAYER_ORDER.index(h) if h in LAYER_ORDER else -1 for h in harmonized])

    adata.obs["cell_type_harmonized"] = pd.Categorical(harmonized)
    adata.obs["layer"] = layer.astype(int)  # 0-3 for LSC->differentiated, -1 NON_LAYER
    adata.obs["is_epithelial"] = layer >= 0

    n_epi = int(adata.obs["is_epithelial"].sum())
    log(f"  {n_epi} epithelial cells ({100 * n_epi / adata.n_obs:.1f}%), "
        f"{adata.n_obs - n_epi} non-epithelial (NON_LAYER).")


# ---------------------------------------------------------------------------
# Step 7: Save + report
# ---------------------------------------------------------------------------

def save_umap_plots(adata: ad.AnnData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc.pl.umap(adata, color="cell_type_raw", ax=axes[0], show=False,
               title="cell_type_raw", legend_loc="right margin")
    sc.pl.umap(adata, color="cell_type_harmonized", ax=axes[1], show=False,
               title="cell_type_harmonized", legend_loc="right margin")
    fig.savefig(out_dir / "umap_celltype.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {out_dir / 'umap_celltype.pdf'}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc.pl.umap(adata, color="donor_id", ax=axes[0], show=False,
               title="donor_id", legend_loc="right margin")
    sc.pl.umap(adata, color="leiden", ax=axes[1], show=False,
               title=f"leiden (res={adata.uns['annotation']['seurat_pipeline']['resolution']})",
               legend_loc="right margin")
    fig.savefig(out_dir / "umap_donor.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {out_dir / 'umap_donor.pdf'}")


def save_marker_scores(out_dir: Path, enrich_df: pd.DataFrame) -> None:
    # One row per Leiden cluster: enrichment score for each cell-type marker
    # set, n_cells, assigned type, and the cluster's top-5 DEGs.
    out = enrich_df.copy()
    out.index.name = "leiden"
    out.to_csv(out_dir / "marker_scores.csv")
    log(f"  Saved {out_dir / 'marker_scores.csv'} (DEG enrichment per cluster, "
        f"{out.shape[0]} clusters x {out.shape[1]} cols)")


def print_summary(adata: ad.AnnData) -> None:
    banner("Summary")
    print("=== cell_type_raw ===")
    print(adata.obs["cell_type_raw"].value_counts().to_string())
    print("\n=== cell_type_harmonized ===")
    print(adata.obs["cell_type_harmonized"].value_counts().to_string())
    print("\n=== Per donor ===")
    print(adata.obs["donor_id"].value_counts().to_string())
    print("\n=== donor_id x cell_type_harmonized ===")
    print(pd.crosstab(adata.obs["donor_id"], adata.obs["cell_type_harmonized"]).to_string())
    print("\n=== Layer distribution (0-3: LSC/TAC/basal/differentiated; -1: NON_LAYER) ===")
    layer_counts = (
        adata.obs["layer"].value_counts()
        .reindex(list(range(-1, len(LAYER_ORDER))))
        .fillna(0)
        .astype(int)
    )
    print(layer_counts.to_string())
    print("\n=== Leiden cluster sizes ===")
    print(adata.obs["leiden"].value_counts().sort_index().to_string())
    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate GSE153515 human limbal stem cell atlas (donors + cell types)."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Input h5ad (default: %(default)s)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output annotated h5ad (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it already exists")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--resolution", type=float, default=RESOLUTION,
                        help="Leiden clustering resolution (default: %(default)s)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        log(f"ERROR: input file not found: {input_path}")
        sys.exit(1)
    if output_path.exists() and not args.force:
        log(f"ERROR: output already exists: {output_path} (use --force to overwrite)")
        sys.exit(1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    log(f"Loading {input_path} ...")
    adata = ad.read_h5ad(input_path)
    log(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes; "
        f"obs columns: {list(adata.obs.columns)}")

    # Dense float32 counts -> CSR to keep memory in check; dense is ~2.3 GB.
    if not sp.issparse(adata.X):
        log("Converting dense X to CSR sparse.")
        adata.X = sp.csr_matrix(adata.X)

    assign_donors(adata)

    # Step 2: score markers on the FULL gene set BEFORE QC filtering — rare
    # cell-type markers (A2M, ABCG2, MKI67, TOP2A, PLK1, ...) are expressed in
    # <3% of cells and would be dropped by the min_cells gene filter in Step 3.
    # Scores are stored in obs and survive the cell-level QC subsetting.
    _setup_var_names(adata)
    score_marker_sets(adata)

    # Steps 3-4: QC filtering, then preprocessing + clustering on high-quality cells.
    qc_filter(adata)
    preprocess(adata, seed=args.seed, resolution=args.resolution)

    # Provenance dict must exist before Step 5 populates the cluster
    # assignment tables (uns['annotation']['cluster_to_celltype'], ...).
    adata.uns["annotation"] = {
        "protocol": "protocol_15_annotate_gse153515.py",
        "references": {
            "ocul_surf_2021": {"pmid": "33388438", "doi": "10.1016/j.jtos.2020.12.004"},
            "iovs_2021": {"pmid": "34297801", "doi": "10.1167/iovs.62.9.36"},
            "github": "sangbaekim/10xRNAseq_LSC",
        },
        "seurat_pipeline": {
            "version": "v3.1",
            "integration": "CCA",
            "resolution": args.resolution,
            "n_pcs": N_PCS,
            "seed": args.seed,
        },
        "marker_sets": MARKER_SETS,
        "pipeline_order": (
            "Step1 donors -> Step2 marker scoring on FULL gene set (pre-QC) -> "
            "Step3 QC filter (genes min_cells=3%%, cells n_genes 5th-95th pct) -> "
            "Step4 preprocess + Harmony integration (donor batch correction) + "
            "cluster -> Step5 DEG-based cell-type assignment (wilcoxon "
            "rank_genes_groups per Leiden cluster) -> Step6 harmonize -> Step7 save"
        ),
        "scoring_note": (
            "Marker scores are computed on the full 32,738-gene matrix BEFORE "
            "QC filtering, because rare cell-type markers (A2M/ABCG2, "
            "MKI67/TOP2A/PLK1) are expressed in <3% of cells and would be "
            "removed by the min_cells gene filter. Scores live in obs and "
            "survive the cell-level QC subsetting unchanged. Standard practice: "
            "cluster on high-quality cells, score markers on the full gene set."
        ),
        "method": "deg_enrichment_cluster_annotation",
        "deg_method": "wilcoxon",
        "deg_n_genes": N_DEG,
        "method_note": (
            "Cell types are assigned per Leiden cluster from DEG enrichment "
            "(sc.tl.rank_genes_groups, wilcoxon, top %d DEGs), mirroring the "
            "Seurat FindAllMarkers -> manual annotation workflow of the "
            "reference paper (PMID 33388438). For each cluster and marker set: "
            "enrichment = (n_markers_in_top%d / n_total_markers) * "
            "(1 - mean_rank / %d). Fixed marker-set scoring is avoided because "
            "broadly expressed basal markers (KRT14/15/5) dominate all limbal "
            "epithelial cells and would label ~98%% of clusters as PMC_basal. "
            "Non-epithelial types (melanocyte/conjunctival/immune) are assigned "
            "only when the cluster enrichment exceeds NON_EPI_THRESHOLD and is "
            "the maximum across all marker sets; an 'Immune' top enrichment "
            "flags the cluster as immune_contamination." % (N_DEG, N_DEG, N_DEG)
        ),
        "harmonize_map": HARMONIZE_MAP,
        "layer_order": LAYER_ORDER,
        "layer_scheme": {
            "n_classes": 4,
            "axis": "LSC/TAC/basal/differentiated (+ NON_LAYER)",
            "decision": (
                "wing+squamous merged into 'differentiated': GSE153515 clusters "
                "do not separate wing from squamous; corneal epithelial "
                "differentiation is a continuum LSC->TAC->basal->wing->squamous "
                "(Cotsarelis et al., Cell 1989, doi: 10.1016/0092-8674(89)90292-0)."
            ),
        },
        "ensembl_map_source": "data/corneal_100k_subsample.h5ad "
                               "(CELLxGENE feature_name->var_names); all 64 marker "
                               "Ensembl IDs verified present in GSE153515 var index "
                               "and symbol-correct (2026-07-31)",
    }

    # Step 5: DEG-based cell-type assignment returns the per-cluster enrichment
    # table used for marker_scores.csv; Step 6 harmonizes to the 4-class scheme.
    enrich_df = assign_cell_types(adata)
    harmonize(adata)

    save_umap_plots(adata, ARTIFACT_DIR)
    save_marker_scores(ARTIFACT_DIR, enrich_df)

    log(f"Saving annotated h5ad -> {output_path}")
    adata.write_h5ad(output_path)
    log(f"Saved: {output_path}")

    print_summary(adata)
    log("Done.")


if __name__ == "__main__":
    main()
