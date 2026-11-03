#!/usr/bin/env python
"""Protocol 15c — Annotate the van Zyl 2022 PNAS cornea atlas (4-class scheme).

Mirrors protocol_15_annotate_gse153515.py methodology exactly (Harmony
integration + marker sets + DEG enrichment n=200 -> clusters -> 4-class
LSC/TAC/basal/differentiated) and applies it to the CELLxGENE van Zyl et al.
2022 PNAS (DOI 10.1073/pnas.2200914119) anterior-segment atlas, which is the
only cornea dataset inside the scGPT pretraining corpus (census 2023-05-15).

Inputs (CELLxGENE, collection 63d03351-06be-478e-a0db-f7a653b6b19b)
-------------------------------------------------------------------
* data/vanzyl_central_cornea.h5ad  (37,485 cells; Central Cornea partition)
* data/vanzyl_tm_wedge.h5ad        (52,309 cells; Trabecular Meshwork and
                                    Corneal Scleral wedge partition, contains
                                    the corneo-scleral junction = limbal niche)

Both partitions ship with CELLxGENE `obs.cell_type` / `author_cell_type`
annotations. These are used ONLY as a post-hoc cross-check (agreement rate in
the report) and NEVER as ground truth for assignment — cluster assignment is
DEG-enrichment based, exactly like protocol_15.

Partition-exclusion rule (preregistered)
----------------------------------------
If the TM wedge partition has <5% epithelial cells (assessed with its own
`cell_type` annotation), the partition is dropped and disclosed; the pipeline
then runs on Central Cornea alone. Verified 2026-08-18: epithelial
(corneal+basal+conjunctival+goblet) = 11,588/52,309 = 22.15% -> partition KEPT.

Deltas vs protocol_15 (all recorded, method otherwise locked)
-------------------------------------------------------------
1. Input: two CELLxGENE partitions merged with an `partition` obs column
   (protocol_15 consumed one GEO file).
2. Donor: obs already carries `donor_id` (sanes2022_*); `sample` =
   library_uuid. No barcode-suffix reconstruction needed.
3. Marker symbols resolve against var['feature_name'] (CELLxGENE ships real
   symbols); protocol_15 needed an Ensembl-ID map because GSE153515 stored
   Ensembl IDs in gene_symbol.
4. Harmony batch keys = ["partition", "donor_id"] (spec: 分区/donor 批次);
   protocol_15 used ["donor_id"].
5. Cell-level QC uses n_genes 5th-95th percentile of the MERGED object
   (protocol_15 same rule, single object).
6. Cross-check vs CELLxGENE annotations appended to the report; assignment
   itself never consults them.

Outputs
-------
* data/vanzyl_annotated.h5ad   (obs: cluster, cell_class_4, cell_type_raw,
                                donor, sample, partition, is_epithelial, ...)
* artifacts/paper1_vanzyl/annotation_report.md
* artifacts/paper1_vanzyl/umap_celltype.pdf, umap_donor.pdf
* artifacts/paper1_vanzyl/marker_scores.csv
* artifacts/paper1_vanzyl/qc_decisions.json   (consumed by manifest builder)

Usage
-----
    <venv>/bin/python3 scripts/protocol_15c_annotate_vanzyl.py \
        [--input-central data/vanzyl_central_cornea.h5ad] \
        [--input-wedge data/vanzyl_tm_wedge.h5ad] \
        [--output data/vanzyl_annotated.h5ad] [--force] \
        [--seed 42] [--resolution 0.6]
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_INPUT_CENTRAL = PROJECT_ROOT / "data" / "vanzyl_central_cornea.h5ad"
DEFAULT_INPUT_WEDGE = PROJECT_ROOT / "data" / "vanzyl_tm_wedge.h5ad"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "vanzyl_annotated.h5ad"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl"

SEED = 42
RESOLUTION = 0.6
N_HVG = 2000
N_PCS = 30  # PCA components fed to Harmony (protocol_15 locked value)
N_DEG = 200  # top-N DEGs per cluster used by enrichment scoring

# Enrichment threshold for non-epithelial assignment (protocol_15 locked).
NON_EPI_THRESHOLD = 0.3

# Epithelial proportion below which the TM wedge partition is excluded.
EPITHELIAL_KEEP_FRACTION = 0.05

# CELLxGENE cell_type values that count as epithelial for the partition check.
CELLXGENE_EPITHELIAL_TYPES = {
    "corneal epithelial cell",
    "basal cell",
    "conjunctival epithelial cell",
    "goblet cell",
}

# Project-standard 4-class corneal/limbal epithelial differentiation axis.
# Corneal epithelial differentiation is a continuum
# LSC -> TAC -> basal -> wing -> squamous (Cotsarelis et al., Cell 1989,
# doi: 10.1016/0092-8674(89)90292-0). Wing and superficial are merged into a
# single "differentiated" class, exactly like protocol_15.
LAYER_ORDER = ["LSC", "TAC", "basal", "differentiated"]

# Published marker sets — VERBATIM copy of protocol_15 (method locked). See
# protocol_15 docstring for provenance (paper Results + Supplementary Table 5,
# expanded with the paper's actual top DEGs per cluster). 'Immune' is a
# contamination-detection set; clusters with Immune as top enrichment are
# flagged 'immune_contamination'.
MARKER_SETS = {
    "LSC": ["A2M", "ABCG2", "ITGA9", "TSPAN7", "SOX17",
            "ECSCR", "RAMP3", "PLVAP", "AQP1"],
    "LPC": ["KRT14", "KRT15", "KRT19", "DCN", "PLIN2", "SLC2A3", "ITM2A", "TCF4"],
    "TAC": ["MKI67", "BIRC5", "RRM2", "TOP2A", "PLK1", "TK1"],
    "PMC_basal": ["KRT14", "KRT15", "KRT5", "COL17A1", "ITGB1",
                  "CLU", "IGFBP6", "ALDH3A1", "SPARC", "HTRA1",
                  "NQO1", "S100A10", "CITED2"],
    "TDC": ["KRT12", "KRT3", "GJA1", "SPRR2A", "IVL",
            "ADIRF", "MAL2", "CLDN7", "LYPD3", "MT1X", "HSPB1", "TACSTD2"],
    "Melanocyte": ["TYR", "TYRP1", "MLANA", "PMEL", "DCT"],
    "Conjunctival": ["KRT4", "KRT7", "KRT13", "MUC5AC", "TFF3"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD14", "CD68", "FCGR3A", "HLA-DRA"],
}

# Harmonization to the 4-class scheme (NON_LAYER = non-epithelial). VERBATIM
# from protocol_15 (LPC -> LSC; PMC_basal -> basal; TDC -> differentiated;
# wing+squamous merged as "differentiated").
HARMONIZE_MAP = {
    "LSC": "LSC",
    "LPC": "LSC",
    "TAC": "TAC",
    "PMC_basal": "basal",
    "TDC": "differentiated",
    "melanocyte": "NON_LAYER",
    "conjunctival": "NON_LAYER",
    "immune_contamination": "NON_LAYER",
}

# CELLxGENE annotation -> 4-class scheme, used ONLY for the cross-check
# agreement rate in the report. Never used for assignment.
AUTHOR_CELLTYPE_MAP = {
    "Limbal_Epi-Basal": "LSC",
    "K_Epi-TA": "TAC",
    "K_Epi-Basal": "basal",
    "K_Epi-Wing": "differentiated",
    "K_Epi-Superficial": "differentiated",
    "Limbal_Epi-Wing": "differentiated",
}
CELLXGENE_CELLTYPE_MAP = {
    "basal cell": "basal",
    "transit amplifying cell": "TAC",
    "corneal epithelial cell": "differentiated",
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
# Step 0: Load + merge partitions; TM wedge epithelial pre-check
# ---------------------------------------------------------------------------


def check_tm_wedge_epithelial(path: Path) -> tuple[bool, dict]:
    """Return (keep, stats) per the <5% epithelial exclusion rule.

    Assessed with the partition's own CELLxGENE cell_type annotation
    (backed read, cheap). `stats` carries the counts for the manifest.
    """
    banner("Step 0a: TM wedge epithelial pre-check (<5% rule)")
    a = ad.read_h5ad(path, backed="r")
    ct = a.obs["cell_type"].astype(str)
    epi_mask = np.isin(ct, list(CELLXGENE_EPITHELIAL_TYPES))
    n_epi = int(epi_mask.sum())
    frac = n_epi / a.n_obs
    log(f"  TM wedge: {a.n_obs} cells, {n_epi} epithelial "
        f"({100 * frac:.2f}%)")
    keep = frac >= EPITHELIAL_KEEP_FRACTION
    log(f"  Threshold: {100 * EPITHELIAL_KEEP_FRACTION:.0f}% -> "
        f"{'KEEP' if keep else 'EXCLUDE'}")
    a.file.close()
    stats = {
        "n_cells": int(a.n_obs),
        "n_epithelial": n_epi,
        "epithelial_fraction": round(frac, 5),
        "threshold": EPITHELIAL_KEEP_FRACTION,
        "keep": bool(keep),
    }
    return keep, stats


def load_and_merge(
    central_path: Path, wedge_path: Path, keep_wedge: bool
) -> ad.AnnData:
    """Load both partitions and merge with a `partition` obs column."""
    banner("Step 0b: Load + merge partitions")
    central = ad.read_h5ad(central_path)
    log(f"  Central Cornea: {central.n_obs} cells x {central.n_vars} genes")
    central.obs["partition"] = "central_cornea"
    if not sp.issparse(central.X):
        central.X = sp.csr_matrix(central.X)

    if keep_wedge:
        wedge = ad.read_h5ad(wedge_path)
        log(f"  TM wedge: {wedge.n_obs} cells x {wedge.n_vars} genes")
        wedge.obs["partition"] = "tm_wedge"
        if not sp.issparse(wedge.X):
            wedge.X = sp.csr_matrix(wedge.X)
        combined = ad.concat(
            [central, wedge], join="outer", index_unique="-", merge="same"
        )
        log(f"  Merged: {combined.n_obs} cells x {combined.n_vars} genes")
    else:
        combined = central
        log("  TM wedge EXCLUDED (epithelial <5%); Central Cornea only: "
            f"{combined.n_obs} cells")
    return combined


# ---------------------------------------------------------------------------
# Step 1: donor/sample metadata
# ---------------------------------------------------------------------------


def assign_metadata(adata: ad.AnnData) -> None:
    """Normalize donor/sample columns and record per-donor counts."""
    banner("Step 1: Donor/sample metadata")
    adata.obs["donor"] = adata.obs["donor_id"].astype(str)
    adata.obs["sample"] = adata.obs["library_uuid"].astype(str)
    counts = adata.obs["donor"].value_counts()
    for donor, n in counts.items():
        log(f"  {donor}: {n} cells")
    log(f"  {len(counts)} donors, {adata.obs['sample'].nunique()} samples.")


# ---------------------------------------------------------------------------
# Step 2: Marker-gene scoring on the FULL gene set (before QC)
# ---------------------------------------------------------------------------


def resolve_marker_genes(
    adata: ad.AnnData, symbols: list[str]
) -> tuple[list[str], list[str]]:
    """Return (available, missing) marker var_names.

    CELLxGENE ships real gene symbols in var['feature_name']; the file-level
    var_names are Ensembl IDs. Markers (symbols) are resolved to var_names
    via feature_name so scanpy score_genes / DEG lookups match directly.
    """
    fn_to_vn: dict[str, str] = {}
    for vn, fn in zip(
        adata.var_names.astype(str), adata.var["feature_name"].astype(str)
    ):
        if fn not in fn_to_vn:
            fn_to_vn[fn] = vn
    available: list[str] = []
    missing: list[str] = []
    for sym in symbols:
        if sym in fn_to_vn:
            available.append(fn_to_vn[sym])
        else:
            missing.append(sym)
    return available, missing


def score_marker_sets(adata: ad.AnnData) -> None:
    """Score marker sets on the FULL gene set BEFORE QC filtering.

    Runs on the complete gene matrix (var_names still carry every Ensembl ID).
    Because score_genes writes into adata.obs (one row per cell), the scores
    survive the subsequent cell filtering unchanged. Identical rationale and
    order to protocol_15 Step 2.
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
        log(f"  score_{name}: {len(avail)} genes -> {genes}")


# ---------------------------------------------------------------------------
# Step 3: QC filtering (mirrors protocol_15)
# ---------------------------------------------------------------------------


def qc_filter(adata: ad.AnnData) -> None:
    banner("Step 3: QC filtering (for clustering; marker scores already computed)")
    n_total = adata.n_obs

    # Genes: keep those detected in >= 3% of cells (protocol_15 rule).
    min_cells = round(n_total * 0.03)
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    log(f"  Gene filter: min_cells={min_cells} -> {adata.n_vars} genes retained "
        f"(was {n_genes_before}).")

    # Cells: keep n_genes within the 5th-95th percentile (protocol_15 rule).
    # Type guard: coerce X to CSR and count detected genes via astype(bool)
    # (avoids `X > 0` comparisons, which fail on non-comparable X backends
    # such as anndata's backed _CSRDataset or exotic dtypes).
    x_mat = adata.X
    if not sp.issparse(x_mat):
        x_mat = sp.csr_matrix(x_mat)
    n_genes = np.asarray(x_mat.astype(bool).sum(axis=1)).ravel()
    lo, hi = np.quantile(n_genes, [0.05, 0.95])
    keep = (n_genes >= lo) & (n_genes <= hi)
    adata.obs["n_genes"] = n_genes.astype(int)
    log(f"  Cell filter: n_genes in [{lo:.0f}, {hi:.0f}] (5th-95th pct) -> "
        f"{int(keep.sum())}/{n_total} cells retained.")
    adata._inplace_subset_obs(keep)

    # Store raw counts BEFORE any normalization (scGPT prep consumes these).
    adata.layers["counts"] = sp.csr_matrix(adata.X)
    log(f"  Raw counts stored in layers['counts'] "
        f"({adata.shape[0]} x {adata.shape[1]}).")


# ---------------------------------------------------------------------------
# Step 4: Preprocessing (Harmony integration + clustering)
# ---------------------------------------------------------------------------


def preprocess(adata: ad.AnnData, seed: int, resolution: float) -> None:
    banner("Step 4: Preprocessing (Seurat-equivalent + Harmony)")

    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    # NOTE (deviation, recorded in manifest): protocol_15 additionally stored
    # layers['normalized']. On this memory-starved host (only ~6 GB usable
    # headroom, peak of this pipeline ~11 GB) the extra ~1.6 GB copy pushes the
    # run into heavy swap thrash. layers['normalized'] is not consumed by any
    # downstream step (scGPT prep reads layers['counts'] only), so the copy is
    # omitted; the normalized values remain available as adata.X until scale.
    log("  Normalized (target_sum=10_000) + log1p (layers['normalized'] "
        "omitted — memory constraint; see manifest deviation note).")

    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, flavor="seurat", subset=False
    )
    log(f"  Highly variable genes: "
        f"{int(adata.var['highly_variable'].sum())}/{adata.n_vars} "
        f"(flavor='seurat', n_top_genes={N_HVG}).")

    sc.pp.scale(adata, max_value=10)
    log("  Scaled (max_value=10).")

    sc.tl.pca(adata, n_comps=N_PCS, random_state=seed)
    log(f"  PCA: {N_PCS} components (random_state={seed}).")

    # Harmony integration (scanpy equivalent of the paper's Seurat CCA):
    # batch-correct the PCA embedding across partition AND donor before
    # clustering. Deltas vs protocol_15: batch keys are ["partition",
    # "donor_id"] (spec: 分区/donor 批次) instead of ["donor_id"] alone.
    try:
        import harmonypy as hpy

        ho = hpy.run_harmony(
            adata.obsm["X_pca"], adata.obs, ["partition", "donor_id"],
            random_state=seed, max_iter_harmony=20,
        )
        adata.obsm["X_harmony"] = np.asarray(ho.Z_corr)
        log("  Harmony integration (harmonypy 2.x, direct API): batch keys "
            "['partition', 'donor_id'] -> obsm['X_harmony'].")
    except ImportError:
        log("  harmonypy not found — falling back to "
            "sc.external.pp.harmony_integrate(adata, key=['partition', "
            "'donor_id']).")
        sc.external.pp.harmony_integrate(
            adata, key=["partition", "donor_id"], adjusted_basis="X_harmony"
        )
        log("  Harmony integration (scanpy wrapper) done -> obsm['X_harmony'].")

    sc.pp.neighbors(adata, use_rep="X_harmony", random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    log("  Neighbors (use_rep='X_harmony') + UMAP done.")

    sc.tl.leiden(adata, resolution=resolution, random_state=seed)
    log(f"  Leiden clustering: resolution={resolution}, "
        f"{adata.obs['leiden'].nunique()} clusters (random_state={seed}).")


# ---------------------------------------------------------------------------
# Step 5: Cell-type assignment (DEG enrichment, FindAllMarkers workflow)
# ---------------------------------------------------------------------------


def assign_cell_types(adata: ad.AnnData) -> pd.DataFrame:
    """Assign cell types per Leiden cluster from differential expression.

    Identical to protocol_15 Step 5: sc.tl.rank_genes_groups (wilcoxon, top
    N_DEG per cluster) -> enrichment = (n_markers_in_top200 / n_total_markers)
    * (1 - mean_rank / 200) -> best-scoring cell type per cluster, with
    threshold-gated non-epithelial assignment (NON_EPI_THRESHOLD) and
    immune_contamination flagging. Returns the per-cluster enrichment table.
    """
    banner("Step 5: DEG-based cluster cell type assignment (FindAllMarkers workflow)")

    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                            n_genes=N_DEG)
    res = adata.uns["rank_genes_groups"]
    deg_groups = list(res["names"].dtype.names)
    deg_by_cluster = {g: list(res["names"][g]) for g in deg_groups}

    # var_names are Ensembl IDs; map back to symbols via feature_name for the
    # top-DEG display columns. Only resolvable names get a symbol.
    ensembl_to_symbol = dict(
        zip(adata.var_names.astype(str), adata.var["feature_name"].astype(str))
    )

    def display(name: str) -> str:
        return ensembl_to_symbol.get(name, name)

    def marker_candidates(sym: str) -> list[str]:
        """Candidate var_names for a marker symbol.

        Markers are symbols (feature_name space); a marker hits a cluster's
        DEG list if the cluster's top-DEG var_name maps back to that symbol.
        """
        cands = set()
        for vn, fn in ensembl_to_symbol.items():
            if fn == sym:
                cands.add(vn)
        return list(cands)

    epithelial_types = ["LSC", "LPC", "TAC", "PMC_basal", "TDC"]
    cluster_labels = {}
    enrich_rows = {}

    for cluster_id in sorted(deg_by_cluster, key=lambda x: int(x)):
        deg_names = deg_by_cluster[cluster_id]
        deg_rank = {name: r for r, name in enumerate(deg_names)}
        n_cells = int((adata.obs["leiden"] == cluster_id).sum())

        enrich = {}
        for ct, symbols in MARKER_SETS.items():
            ranks = []
            for sym in symbols:
                hit = next(
                    (deg_rank[c] for c in marker_candidates(sym)
                     if c in deg_rank),
                    None,
                )
                if hit is not None:
                    ranks.append(hit)
            if not ranks:
                enrich[ct] = 0.0
            else:
                enrich[ct] = (
                    len(ranks) / len(symbols) * (1.0 - np.mean(ranks) / N_DEG)
                )

        # Non-epithelial priority (threshold-gated), protocol_15 order:
        # immune contamination, then melanocyte, then conjunctival.
        if enrich["Immune"] > NON_EPI_THRESHOLD and \
                enrich["Immune"] == max(enrich.values()):
            assigned = "immune_contamination"
        elif enrich["Melanocyte"] > NON_EPI_THRESHOLD and \
                enrich["Melanocyte"] == max(enrich.values()):
            assigned = "melanocyte"
        elif enrich["Conjunctival"] > NON_EPI_THRESHOLD and \
                enrich["Conjunctival"] == max(enrich.values()):
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

    log(f"\n  Cluster -> Cell Type assignment (DEG enrichment, top-{N_DEG} DEGs):")
    log(f"  {'Cl':>4} {'Assigned':>14} {'n':>7} | "
        + " | ".join(f"{ct:>11}" for ct in MARKER_SETS)
        + " | top 5 DEGs")
    for cluster_id in sorted(cluster_labels, key=lambda x: int(x)):
        r = enrich_rows[cluster_id]
        enrich_str = " | ".join(f"{r[ct]:>11.3f}" for ct in MARKER_SETS)
        log(f"  {cluster_id:>4} {cluster_labels[cluster_id]:>14} "
            f"{r['n_cells']:>7} | {enrich_str} | "
            + ", ".join(f"{r[f'top_deg_{i}']}" for i in range(1, 6)))

    adata.obs["cell_type_raw"] = adata.obs["leiden"].map(cluster_labels)

    counts = adata.obs["cell_type_raw"].value_counts()
    log("\n  Cell type assignment (DEG enrichment):")
    for ct, n in counts.items():
        log(f"    {ct}: {n} ({100 * n / adata.n_obs:.1f}%)")

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
    layer = np.array(
        [LAYER_ORDER.index(h) if h in LAYER_ORDER else -1
         for h in harmonized]
    )

    adata.obs["cell_class_4"] = pd.Categorical(harmonized)
    adata.obs["layer"] = layer.astype(int)  # 0-3 LSC->differentiated, -1 NON_LAYER
    adata.obs["is_epithelial"] = layer >= 0
    adata.obs["cluster"] = adata.obs["leiden"].astype(str)

    n_epi = int(adata.obs["is_epithelial"].sum())
    log(f"  {n_epi} epithelial cells ({100 * n_epi / adata.n_obs:.1f}%), "
        f"{adata.n_obs - n_epi} non-epithelial (NON_LAYER).")


# ---------------------------------------------------------------------------
# Step 7: Cross-check vs CELLxGENE annotations + reports
# ---------------------------------------------------------------------------


def crosscheck_cellxgene(adata: ad.AnnData) -> dict:
    """Agreement of our 4-class labels vs CELLxGENE annotations (report only).

    CELLxGENE `author_cell_type` (van Zyl's own labels) and `cell_type`
    (curated) are mapped onto the 4-class scheme with the maps above and
    compared against our cell_class_4. Both overall and epithelial-subset
    agreement are computed. These numbers are descriptive; the CELLxGENE
    labels were never used for assignment.
    """
    banner("Step 7: Cross-check vs CELLxGENE annotations")

    def mapped(series, mapping: dict[str, str]) -> pd.Series:
        return series.astype(str).map(lambda x: mapping.get(x, "NON_LAYER"))

    author_mapped = mapped(adata.obs["author_cell_type"], AUTHOR_CELLTYPE_MAP)
    cellx_mapped = mapped(adata.obs["cell_type"], CELLXGENE_CELLTYPE_MAP)
    ours = adata.obs["cell_class_4"].astype(str)

    result = {}
    for name, ref in [
        ("author_cell_type", author_mapped),
        ("cell_type", cellx_mapped),
    ]:
        overall = float((ours == ref).mean())
        epi_mask = adata.obs["is_epithelial"].to_numpy()
        epi_agr = float(((ours == ref) & epi_mask).sum() / max(epi_mask.sum(), 1))
        result[name] = {
            "overall_agreement": round(overall, 4),
            "epithelial_subset_agreement": round(epi_agr, 4),
        }
        log(f"  vs {name}: overall agreement {overall:.3f}, "
            f"epithelial-subset {epi_agr:.3f}")

    # Confusion (epithelial classes only) vs author annotation — for report.
    epi_classes = LAYER_ORDER
    cm = pd.crosstab(
        ours[adata.obs["is_epithelial"]],
        author_mapped[adata.obs["is_epithelial"]],
    ).reindex(index=epi_classes, columns=epi_classes, fill_value=0)
    result["confusion_author_epithelial"] = cm.to_dict()
    return result


def save_umap_plots(adata: ad.AnnData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc.pl.umap(adata, color="cell_type_raw", ax=axes[0], show=False,
               title="cell_type_raw", legend_loc="right margin")
    sc.pl.umap(adata, color="cell_class_4", ax=axes[1], show=False,
               title="cell_class_4", legend_loc="right margin")
    fig.savefig(out_dir / "umap_celltype.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {out_dir / 'umap_celltype.pdf'}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc.pl.umap(adata, color="donor", ax=axes[0], show=False,
               title="donor", legend_loc="right margin")
    sc.pl.umap(adata, color="leiden", ax=axes[1], show=False,
               title=f"leiden (res={RESOLUTION})", legend_loc="right margin")
    fig.savefig(out_dir / "umap_donor.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {out_dir / 'umap_donor.pdf'}")


def save_marker_scores(out_dir: Path, enrich_df: pd.DataFrame) -> None:
    out = enrich_df.copy()
    out.index.name = "leiden"
    out.to_csv(out_dir / "marker_scores.csv")
    log(f"  Saved {out_dir / 'marker_scores.csv'} "
        f"({out.shape[0]} clusters x {out.shape[1]} cols)")


def write_qc_decisions(out_dir: Path, qc: dict) -> None:
    (out_dir / "qc_decisions.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False)
    )
    log(f"  Saved {out_dir / 'qc_decisions.json'}")


def print_summary(adata: ad.AnnData) -> None:
    banner("Summary")
    print("=== cell_type_raw ===")
    print(adata.obs["cell_type_raw"].value_counts().to_string())
    print("\n=== cell_class_4 ===")
    print(adata.obs["cell_class_4"].value_counts().to_string())
    print("\n=== Per donor ===")
    print(adata.obs["donor"].value_counts().to_string())
    print("\n=== donor x partition ===")
    print(pd.crosstab(adata.obs["donor"], adata.obs["partition"]).to_string())
    print("\n=== donor x cell_class_4 ===")
    print(pd.crosstab(adata.obs["donor"], adata.obs["cell_class_4"]).to_string())
    print("\n=== partition x cell_class_4 ===")
    print(pd.crosstab(adata.obs["partition"],
                      adata.obs["cell_class_4"]).to_string())
    print("\n=== Layer distribution (0-3: LSC/TAC/basal/differentiated; "
          "-1: NON_LAYER) ===")
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
        description="Annotate van Zyl 2022 PNAS cornea atlas (4-class scheme)."
    )
    parser.add_argument("--input-central", default=str(DEFAULT_INPUT_CENTRAL),
                        help="Central Cornea h5ad (default: %(default)s)")
    parser.add_argument("--input-wedge", default=str(DEFAULT_INPUT_WEDGE),
                        help="TM wedge h5ad (default: %(default)s)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output annotated h5ad (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it already exists")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--resolution", type=float, default=RESOLUTION,
                        help="Leiden clustering resolution (default: %(default)s)")
    args = parser.parse_args()

    central_path = Path(args.input_central)
    wedge_path = Path(args.input_wedge)
    output_path = Path(args.output)
    for p in (central_path, wedge_path):
        if not p.exists():
            log(f"ERROR: input file not found: {p}")
            sys.exit(1)
    if output_path.exists() and not args.force:
        log(f"ERROR: output already exists: {output_path} "
            f"(use --force to overwrite)")
        sys.exit(1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    qc = {"seed": args.seed, "resolution": args.resolution}

    # Step 0a: TM wedge epithelial pre-check (preregistered <5% rule).
    keep_wedge, wedge_stats = check_tm_wedge_epithelial(wedge_path)
    qc["tm_wedge_partition"] = wedge_stats

    # Step 0b: merge partitions.
    adata = load_and_merge(central_path, wedge_path, keep_wedge)
    qc["merged_n_cells"] = int(adata.n_obs)
    qc["merged_n_genes"] = int(adata.n_vars)

    assign_metadata(adata)

    # Step 2: marker scoring on the FULL gene set BEFORE QC (protocol_15 order).
    score_marker_sets(adata)

    # Steps 3-4: QC + preprocessing + Harmony + clustering.
    qc_filter(adata)
    qc["qc_n_cells"] = int(adata.n_obs)
    qc["qc_n_genes"] = int(adata.n_vars)
    preprocess(adata, seed=args.seed, resolution=args.resolution)

    # Provenance dict must exist before Step 5 populates it.
    adata.uns["annotation"] = {
        "protocol": "protocol_15c_annotate_vanzyl.py",
        "mirrors": "protocol_15_annotate_gse153515.py (method locked)",
        "references": {
            "van_zyl_2022": {
                "doi": "10.1073/pnas.2200914119",
                "journal": "PNAS",
            },
        },
        "seurat_pipeline": {
            "version": "v3.1-equivalent",
            "integration": "Harmony (scanpy equivalent of Seurat CCA)",
            "batch_keys": ["partition", "donor_id"],
            "resolution": args.resolution,
            "n_pcs": N_PCS,
            "seed": args.seed,
        },
        "marker_sets": MARKER_SETS,
        "pipeline_order": (
            "Step0 partition merge + TM wedge epithelial pre-check -> "
            "Step1 donor/sample -> Step2 marker scoring on FULL gene set "
            "(pre-QC) -> Step3 QC filter (genes min_cells=3%, cells n_genes "
            "5th-95th pct) -> Step4 preprocess + Harmony integration "
            "(['partition','donor_id']) + cluster -> Step5 DEG-based "
            "assignment (wilcoxon rank_genes_groups) -> Step6 harmonize -> "
            "Step7 cross-check + save"
        ),
        "scoring_note": (
            "Marker scores are computed on the full gene matrix BEFORE QC "
            "filtering, because rare cell-type markers (A2M/ABCG2, "
            "MKI67/TOP2A/PLK1) are expressed in <3% of cells and would be "
            "removed by the min_cells gene filter. Scores live in obs and "
            "survive the cell-level QC subsetting unchanged."
        ),
        "method": "deg_enrichment_cluster_annotation",
        "deg_method": "wilcoxon",
        "deg_n_genes": N_DEG,
        "method_note": (
            "Cell types assigned per Leiden cluster from DEG enrichment "
            "(sc.tl.rank_genes_groups, wilcoxon), mirroring the Seurat "
            "FindAllMarkers -> manual annotation workflow. For each cluster "
            f"and marker set: enrichment = (n_markers_in_top{N_DEG} / "
            f"n_total_markers) * (1 - mean_rank / {N_DEG}). Non-epithelial "
            "types assigned only when enrichment exceeds "
            "NON_EPI_THRESHOLD and is the maximum; 'Immune' top enrichment "
            "flags contamination."
        ),
        "harmonize_map": HARMONIZE_MAP,
        "layer_order": LAYER_ORDER,
        "layer_scheme": {
            "n_classes": 4,
            "axis": "LSC/TAC/basal/differentiated (+ NON_LAYER)",
            "decision": (
                "wing+superficial merged into 'differentiated': van Zyl "
                "clusters are not expected to separate wing from superficial "
                "at the 0.6-resolution level; differentiation is a continuum "
                "LSC->TAC->basal->wing->squamous (Cotsarelis et al., Cell "
                "1989, doi: 10.1016/0092-8674(89)90292-0). Same rule as "
                "protocol_15."
            ),
        },
        "crosscheck_note": (
            "CELLxGENE obs.cell_type / author_cell_type used ONLY for "
            "post-hoc agreement reporting; never as assignment ground truth."
        ),
    }

    enrich_df = assign_cell_types(adata)
    harmonize(adata)

    crosscheck = crosscheck_cellxgene(adata)
    qc["crosscheck"] = crosscheck

    save_umap_plots(adata, ARTIFACT_DIR)
    save_marker_scores(ARTIFACT_DIR, enrich_df)

    # Obs summary stats for the manifest.
    qc["cluster_n"] = int(adata.obs["leiden"].nunique())
    qc["cell_class_4_counts"] = (
        adata.obs["cell_class_4"].value_counts().to_dict()
    )
    qc["assignment_rate_4class"] = round(
        float(adata.obs["is_epithelial"].mean()), 4
    )
    qc["donor_counts"] = adata.obs["donor"].value_counts().to_dict()
    qc["donor_x_partition"] = (
        pd.crosstab(adata.obs["donor"], adata.obs["partition"]).to_dict()
    )
    qc["n_donors"] = int(adata.obs["donor"].nunique())
    write_qc_decisions(ARTIFACT_DIR, qc)

    log(f"Saving annotated h5ad -> {output_path}")
    adata.write_h5ad(output_path)
    log(f"Saved: {output_path}")

    print_summary(adata)
    log("Done.")


if __name__ == "__main__":
    main()
