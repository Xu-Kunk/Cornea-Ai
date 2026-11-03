#!/usr/bin/env python
"""Protocol 15d — Annotate the two second independent cornea datasets.

Reproduces protocol_15c_annotate_vanzyl.py methodology EXACTLY (marker
scoring pre-QC on the full gene set -> QC -> Seurat-equivalent preprocess +
Harmony integration -> Leiden -> DEG-enrichment 4-class assignment) and
applies it to two INDEPENDENT GEO cornea scRNA-seq datasets:

    GSE186433 (Catala et al. 2021 Sci Rep, PMID 34741068)
        19,472 cells / 8 donors (4F/4M; 07+08 = 2-donor limbus pool).
        Inputs: GSE186433_filtered_counts.csv.gz (genes x cells) +
                GSE186433_metadata_percell.csv.gz.
        Harmony batch keys: ["Samplenr"] (donor unit; van Zyl used
        ["partition","donor_id"] — here there is no partition column, so
        donor is the only batch axis; deviation recorded).
        Cross-check reference: obs Layers (Epi/Stroma/Endo/Rest) is an
        anatomical-dissection label, NOT a cell-type annotation — used only
        as a weak descriptive reference, never as assignment ground truth.

    GSE155683 (Collin et al. 2021 Ocul Surf, PMID 33865984)
        Adult subset only: GSE155683_counts_adult_cornea.txt.gz
        21,343 cells / 4 adult donors (barcode suffix _1.._4 = donor).
        No per-cell metadata file exists on GEO -> no cross-check reference;
        donor reconstructed from the barcode suffix (disclosed).
        Harmony batch keys: ["donor"].

Deltas vs protocol_15c (all recorded; method otherwise locked):
1. Input is raw GEO counts (CSV/TXT, genes x cells) instead of CELLxGENE
   h5ad partitions; var_names are gene symbols already.
2. No CELLxGENE/author cell_type annotations exist -> cross-check section
   uses what IS available (GSE186433 Layers dissection label; GSE155683
   nothing) and is clearly labelled as descriptive-only.
3. Batch keys per dataset as above.

Outputs (per dataset)
---------------------
* data/cornea2/<ds>_annotated.h5ad   (obs: cluster, cell_class_4,
                                      cell_type_raw, donor, sample, ...)
* artifacts/paper1_vanzyl/cornea2/<ds>_annotation_report.md
* artifacts/paper1_vanzyl/cornea2/<ds>_qc_decisions.json
* artifacts/paper1_vanzyl/cornea2/<ds>_marker_scores.csv
* artifacts/paper1_vanzyl/cornea2/<ds>_umap_celltype.pdf, _umap_donor.pdf

Usage
-----
    /home/dq/venv-ds/bin/python3 scripts/protocol_15d_annotate_cornea2.py \
        --dataset GSE186433 [--force]
    /home/dq/venv-ds/bin/python3 scripts/protocol_15d_annotate_cornea2.py \
        --dataset GSE155683 [--force]
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "cornea2_download"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cornea2"

SEED = 42
RESOLUTION = 0.6
N_HVG = 2000
N_PCS = 30
N_DEG = 200
NON_EPI_THRESHOLD = 0.3

LAYER_ORDER = ["LSC", "TAC", "basal", "differentiated"]

# VERBATIM copy of protocol_15c marker sets (method locked).
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

DATASETS = {
    "GSE186433": {
        "counts": DATA / "GSE186433_filtered_counts.csv.gz",
        "metadata": DATA / "GSE186433_metadata_percell.csv.gz",
        "harmony_keys": ["Samplenr"],
        "donor_col": "Samplenr",
        "sample_col": "Filename",
        "has_crosscheck": True,
        # Native QC signal: the dataset is dissected BY LAYER. Mirroring the
        # van Zyl partition epithelial pre-check (<5% rule, protocol_15c
        # Step 0a), keep only cells dissected as Epi — the DCN/PLIN2/SLC2A3
        # LPC markers are not specific against stromal fibroblasts, so
        # including Stroma/Endo cells contaminates the LSC/basal clusters
        # (verified 2026-08-19: full-object run -> 100% epithelial, Stroma
        # cells labelled LPC/PMC_basal).
        "epithelial_keep": {"Layers": ["Epi"]},
    },
    "GSE155683": {
        # Downloaded 2026-07-22 into data/ (GEO supplementary); SHA-256
        # archived in cornea2 SHA manifest (030cfd2f…).
        "counts": PROJECT_ROOT / "data" / "GSE155683_counts_adult_cornea.txt.gz",
        "metadata": None,
        "harmony_keys": ["donor"],
        "donor_col": "donor",
        "sample_col": "sample",
        "has_crosscheck": False,
    },
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Loading (raw GEO counts -> AnnData, cells x genes, sparse)
# ---------------------------------------------------------------------------


def load_counts_csv(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load a GEO genes-x-cells counts CSV/TXT into (csr, genes, cells).

    Streams the file row-block by row-block to keep peak memory low (the
    GSE186433 CSV is 39.5 MB gz ~ 700 MB unpacked; the host has ~7 GB
    usable RAM).
    """
    log(f"  Loading counts from {path.name} ...")
    cells: list[str] | None = None
    col_blocks: list[sp.csr_matrix] = []
    genes: list[str] = []
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        header = fh.readline().strip().split(",")
        cells = [c.strip('"') for c in header[1:]]
        n_cols = len(cells)
        block: list[list[float]] = []
        block_genes: list[str] = []
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != n_cols + 1:
                continue
            block_genes.append(parts[0].strip('"'))
            block.append([float(x) for x in parts[1:]])
            if len(block) >= 500:
                col_blocks.append(sp.csr_matrix(np.asarray(block, dtype=np.float32)))
                block = []
                block_genes_out = block_genes
                genes.extend(block_genes)
                block_genes = []
        if block:
            col_blocks.append(sp.csr_matrix(np.asarray(block, dtype=np.float32)))
            genes.extend(block_genes)
    log(f"  Loaded {len(genes)} genes x {len(cells)} cells")
    X = sp.vstack(col_blocks, format="csr").T  # transpose to cells x genes
    log(f"  Matrix {X.shape} nnz={X.nnz}")
    return X, genes, cells


def build_adata(dataset: str, force: bool) -> ad.AnnData:
    cfg = DATASETS[dataset]
    X, genes, cells = load_counts_csv(cfg["counts"])
    var = pd.DataFrame(index=genes)
    var.index.name = "gene_symbol"
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=cells), var=var)

    if cfg["metadata"] is not None:
        md = pd.read_csv(cfg["metadata"], index_col=0)
        md.index = [str(x) for x in md.index]
        if list(md.index) != list(adata.obs_names):
            log("  WARNING: metadata row order != counts cell order; "
                "aligning by index")
            adata = adata[list(md.index)].copy()
        for c in md.columns:
            adata.obs[c] = md[c].astype(str)
        log(f"  Metadata merged: {list(md.columns)}")

    # Native epithelial pre-check (mirrors van Zyl partition rule): keep
    # only cells whose dissection layer matches `epithelial_keep`.
    if "epithelial_keep" in cfg:
        col, vals = next(iter(cfg["epithelial_keep"].items()))
        mask = adata.obs[col].isin(vals).to_numpy()
        n_kept = int(mask.sum())
        log(f"  Epithelial pre-check: {col} in {vals} -> {n_kept}/"
            f"{adata.n_obs} cells kept ({100 * n_kept / adata.n_obs:.1f}%)")
        adata = adata[mask].copy()

    # donor / sample reconstruction
    if dataset == "GSE155683":
        # donor = barcode suffix _1.._4 (Collin adult samples)
        suffix = np.array(
            [str(b).rsplit("_", 1)[-1] for b in adata.obs_names]
        )
        adata.obs["donor"] = pd.Categorical("donor_" + suffix)
        adata.obs["sample"] = pd.Categorical("donor_" + suffix)
        log("  GSE155683: donor reconstructed from barcode suffix "
            f"-> {adata.obs['donor'].nunique()} donors")

    for c in (cfg["donor_col"], cfg["sample_col"]):
        if c not in adata.obs:
            adata.obs[c] = "NA"
    adata.obs["donor_id"] = adata.obs[cfg["donor_col"]].astype(str)
    adata.obs["sample"] = adata.obs[cfg["sample_col"]].astype(str)

    # strip the S-prefix from GSE186433 barcodes? Keep as-is (stable ids).
    adata.obs_names = [str(b) for b in adata.obs_names]
    log(f"  AnnData {adata.shape} donors={adata.obs['donor_id'].nunique()}")
    return adata


# ---------------------------------------------------------------------------
# Steps 2-7: mirror protocol_15c (verbatim logic, adapted inputs)
# ---------------------------------------------------------------------------


def score_marker_sets(adata: ad.AnnData) -> None:
    banner("Step 2: Marker-gene scoring on FULL gene set (before QC)")
    symbol_to_vn = {str(g): g for g in adata.var_names}
    for name, genes in MARKER_SETS.items():
        avail = [symbol_to_vn[g] for g in genes if g in symbol_to_vn]
        missing = [g for g in genes if g not in symbol_to_vn]
        if missing:
            log(f"  WARNING [{name}]: {len(missing)}/{len(genes)} genes "
                f"missing ({missing}) — continuing with available.")
        if not avail:
            log(f"  WARNING [{name}]: no genes available; score = NaN.")
            adata.obs[f"score_{name}"] = np.nan
            continue
        sc.tl.score_genes(adata, gene_list=avail, score_name=f"score_{name}")
        log(f"  score_{name}: {len(avail)} genes")


def qc_filter(adata: ad.AnnData) -> None:
    banner("Step 3: QC filtering (marker scores already computed)")
    n_total = adata.n_obs
    min_cells = round(n_total * 0.03)
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    log(f"  Gene filter: min_cells={min_cells} -> {adata.n_vars} genes "
        f"(was {n_genes_before}).")

    x_mat = adata.X
    if not sp.issparse(x_mat):
        x_mat = sp.csr_matrix(x_mat)
    n_genes = np.asarray(x_mat.astype(bool).sum(axis=1)).ravel()
    lo, hi = np.quantile(n_genes, [0.05, 0.95])
    keep = (n_genes >= lo) & (n_genes <= hi)
    adata.obs["n_genes"] = n_genes.astype(int)
    log(f"  Cell filter: n_genes in [{lo:.0f}, {hi:.0f}] -> "
        f"{int(keep.sum())}/{n_total} cells retained.")
    adata._inplace_subset_obs(keep)

    adata.layers["counts"] = sp.csr_matrix(adata.X)
    log(f"  Raw counts stored in layers['counts'] ({adata.shape}).")


def preprocess(adata: ad.AnnData, seed: int, resolution: float,
               batch_keys: list[str]) -> None:
    banner(f"Step 4: Preprocess + Harmony ({batch_keys}) + clustering")
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, flavor="seurat", subset=False
    )
    log(f"  HVG: {int(adata.var['highly_variable'].sum())}/{adata.n_vars}")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=N_PCS, random_state=seed)
    try:
        import harmonypy as hpy

        ho = hpy.run_harmony(
            adata.obsm["X_pca"], adata.obs, batch_keys,
            random_state=seed, max_iter_harmony=20,
        )
        adata.obsm["X_harmony"] = np.asarray(ho.Z_corr)
        log(f"  Harmony: batch keys {batch_keys} -> obsm['X_harmony']")
    except ImportError:
        sc.external.pp.harmony_integrate(
            adata, key=batch_keys, adjusted_basis="X_harmony"
        )
        log("  Harmony via scanpy wrapper")
    sc.pp.neighbors(adata, use_rep="X_harmony", random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed)
    log(f"  Leiden res={resolution}: {adata.obs['leiden'].nunique()} clusters")


def assign_cell_types(adata: ad.AnnData) -> pd.DataFrame:
    banner("Step 5: DEG-based cluster cell type assignment")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                            n_genes=N_DEG)
    res = adata.uns["rank_genes_groups"]
    deg_groups = list(res["names"].dtype.names)
    deg_by_cluster = {g: list(res["names"][g]) for g in deg_groups}

    epithelial_types = ["LSC", "LPC", "TAC", "PMC_basal", "TDC"]
    cluster_labels = {}
    enrich_rows = {}
    for cluster_id in sorted(deg_by_cluster, key=lambda x: int(x)):
        deg_names = deg_by_cluster[cluster_id]
        deg_rank = {name: r for r, name in enumerate(deg_names)}
        n_cells = int((adata.obs["leiden"] == cluster_id).sum())
        enrich = {}
        for ct, symbols in MARKER_SETS.items():
            ranks = [deg_rank[s] for s in symbols if s in deg_rank]
            if not ranks:
                enrich[ct] = 0.0
            else:
                enrich[ct] = (
                    len(ranks) / len(symbols) * (1.0 - np.mean(ranks) / N_DEG)
                )
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
        enrich_rows[cluster_id] = {
            **enrich,
            "n_cells": n_cells,
            "assigned": assigned,
            **{f"top_deg_{i + 1}": deg_names[i] for i in range(5)},
        }
    adata.obs["cell_type_raw"] = adata.obs["leiden"].map(cluster_labels)
    counts = adata.obs["cell_type_raw"].value_counts()
    log("\n  Cell type assignment:")
    for ct, n in counts.items():
        log(f"    {ct}: {n} ({100 * n / adata.n_obs:.1f}%)")
    adata.uns["annotation"] = {
        "method": "deg_enrichment_cluster_annotation",
        "deg_method": "wilcoxon",
        "deg_n_genes": N_DEG,
        "cluster_to_celltype": cluster_labels,
    }
    enrich_df = pd.DataFrame.from_dict(enrich_rows, orient="index")
    enrich_df.index.name = "leiden"
    return enrich_df


def harmonize(adata: ad.AnnData) -> None:
    banner("Step 6: Harmonization to 4-class scheme")
    raw = adata.obs["cell_type_raw"].astype(str)
    harmonized = np.array([HARMONIZE_MAP.get(t, "NON_LAYER") for t in raw])
    layer = np.array(
        [LAYER_ORDER.index(h) if h in LAYER_ORDER else -1 for h in harmonized]
    )
    adata.obs["cell_class_4"] = pd.Categorical(harmonized)
    adata.obs["layer"] = layer.astype(int)
    adata.obs["is_epithelial"] = layer >= 0
    adata.obs["cluster"] = adata.obs["leiden"].astype(str)
    n_epi = int(adata.obs["is_epithelial"].sum())
    log(f"  {n_epi} epithelial ({100 * n_epi / adata.n_obs:.1f}%), "
        f"{adata.n_obs - n_epi} NON_LAYER.")


# Fibroblast/mesenchymal/endothelial/smooth-muscle contamination gene set.
# Mirrors the van Zyl cluster0 forensics precedent (ciliary-muscle cluster
# misassigned as epithelial, excluded from the analysis set and recorded in
# qc_decisions.json): a cluster whose top-5 DEGs include >= 3 of these genes
# is non-epithelial contamination despite a top epithelial enrichment score.
# Two contamination sources are known for these GEO datasets:
#   (a) the LPC marker set contains DCN/PLIN2/SLC2A3, not specific against
#       stromal fibroblasts/keratocytes (GSE186433 cluster 6, GSE155683
#       clusters 3/7/12);
#   (b) the van Zyl LSC marker set contains A2M/ECSCR/RAMP3/PLVAP/AQP1 —
#       limbal vascular genes — so endothelial/lymphatic/smooth-muscle
#       clusters can score as LSC (GSE155683 clusters 9/11/15; same failure
#       mode as the van Zyl cluster0 ciliary-muscle misassignment).
FIBRO_DEG_GENES = {
    # stroma / keratocyte / fibroblast
    "DCN", "VIM", "APOD", "PTGDS", "SAA1", "COL1A1", "COL3A1", "COL1A2",
    "LUM", "KERA", "ITGBL1", "DPT", "FBLN1", "MFAP4", "SPARC", "GSN",
    "C1QA", "C1QB", "C1S", "C1R", "MGP", "CFH", "CFD", "C3",
    # smooth muscle / pericyte
    "TAGLN", "ACTA2", "MYL9", "CALD1", "NR2F2", "RGS5", "MCAM",
    # endothelial / lymphatic / vascular
    "PECAM1", "VWF", "CD34", "ENG", "PLVAP", "CLDN5", "MMRN1", "CCL21",
    "TFPI", "GNG11", "FLT1", "ESM1", "ACKR1",
    # immune (B/T/myeloid/APC)
    "PTPRC", "CD3D", "CD3E", "CD14", "CD68", "HLA-DRA", "HLA-DRB1",
    "HLA-DPA1", "CD74", "SRGN", "LYZ", "AIF1",
}

FIBRO_TOP5_HITS = 3  # >= 3 of the top-5 DEGs -> contamination


def flag_fibro_contamination(adata: ad.AnnData) -> list[str]:
    """Flag clusters whose top-5 DEGs look fibroblastic/endothelial.

    Assignment is per Leiden cluster from sc.tl.rank_genes_groups (already
    computed in Step 5). Contaminated clusters get cell_class_4 = NON_LAYER
    and is_epithelial = False (excluded from the analysis set), exactly like
    the van Zyl cluster0 ciliary-muscle exclusion. Returned list is recorded
    in qc_decisions.json (judgemental choice, frozen before any split/train).
    """
    res = adata.uns["rank_genes_groups"]
    groups = list(res["names"].dtype.names)
    flagged: list[str] = []
    if "NON_LAYER" not in adata.obs["cell_class_4"].cat.categories:
        adata.obs["cell_class_4"] = adata.obs["cell_class_4"].cat.add_categories("NON_LAYER")
    for g in sorted(groups, key=lambda x: int(x)):
        top5 = [str(x) for x in res["names"][g][:5]]
        hits = [g_ for g_ in top5 if g_ in FIBRO_DEG_GENES]
        if len(hits) >= FIBRO_TOP5_HITS:
            flagged.append(g)
            mask = adata.obs["leiden"] == g
            adata.obs.loc[mask, "cell_class_4"] = "NON_LAYER"
            adata.obs.loc[mask, "layer"] = -1
            adata.obs.loc[mask, "is_epithelial"] = False
            log(f"  FIBRO-CONTAMINATION cluster {g}: top5 {top5} "
                f"({len(hits)} hits) -> NON_LAYER (analysis-set exclusion)")
    return flagged


def crosscheck_layers(adata: ad.AnnData, dataset: str) -> dict:
    """Weak descriptive cross-check (NOT assignment ground truth).

    GSE186433: Layers is an anatomical-dissection label (Epi/Stroma/Endo/
    Rest). Epithelial assignment rate per dissection layer is reported.
    GSE155683: no metadata exists; nothing to cross-check.
    """
    banner("Step 7: Descriptive cross-check (dissection layer / N/A)")
    result = {"dataset": dataset, "note": None}
    if "Layers" in adata.obs:
        ctab = pd.crosstab(
            adata.obs["Layers"], adata.obs["cell_class_4"]
        ).reindex(
            columns=LAYER_ORDER + ["NON_LAYER"], fill_value=0
        )
        epi_frac = (
            ctab[LAYER_ORDER].sum(axis=1) / ctab.sum(axis=1)
        )
        result["per_layer_epithelial_fraction"] = epi_frac.round(4).to_dict()
        result["crosstab"] = ctab.to_dict()
        result["note"] = (
            "Layers is an anatomical-dissection label, not a cell-type "
            "annotation; reported descriptively only."
        )
        log(f"  Epithelial fraction per dissection layer: {epi_frac.to_dict()}")
    else:
        result["note"] = "No per-cell metadata exists on GEO for this dataset."
        log("  No per-cell metadata available; cross-check skipped.")
    return result


def save_umap_plots(adata: ad.AnnData, out_dir: Path, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc.pl.umap(adata, color="cell_type_raw", ax=axes[0], show=False,
               title="cell_type_raw", legend_loc="right margin")
    sc.pl.umap(adata, color="cell_class_4", ax=axes[1], show=False,
               title="cell_class_4", legend_loc="right margin")
    fig.savefig(out_dir / f"{tag}_umap_celltype.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc.pl.umap(adata, color="donor_id", ax=axes[0], show=False,
               title="donor", legend_loc="right margin")
    sc.pl.umap(adata, color="leiden", ax=axes[1], show=False,
               title=f"leiden (res={RESOLUTION})", legend_loc="right margin")
    fig.savefig(out_dir / f"{tag}_umap_donor.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved UMAPs -> {out_dir}")


def write_report(dataset: str, adata: ad.AnnData, enrich_df: pd.DataFrame,
                 qc: dict) -> None:
    out_dir = ARTIFACT_DIR
    tag = dataset.lower()
    lines = [
        f"# {dataset} — Annotation report (protocol_15d, mirrors 15c)",
        "",
        f"- Cells in: {qc['input_n_cells']}; after QC: {adata.n_obs}",
        f"- Genes in: {qc['input_n_genes']}; after QC: {adata.n_vars}",
        f"- Leiden clusters (res={RESOLUTION}): {qc['cluster_n']}",
        "",
        "## cell_class_4 counts",
        "",
        "```",
        adata.obs["cell_class_4"].value_counts().to_string(),
        "```",
        "",
        "## Per-donor cell_class_4",
        "",
        "```",
        pd.crosstab(adata.obs["donor_id"],
                    adata.obs["cell_class_4"]).to_string(),
        "```",
        "",
        "## Cluster assignment (DEG enrichment, wilcoxon top-200)",
        "",
        "```",
        enrich_df.round(3).to_string(),
        "```",
        "",
        "## Cross-check",
        "",
        f"```json",
        json.dumps(qc.get("crosscheck", {}), indent=2),
        "```",
        "",
    ]
    (out_dir / f"{tag}_annotation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    log(f"  Saved report -> {out_dir / f'{tag}_annotation_report.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resolution", type=float, default=RESOLUTION)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    tag = args.dataset.lower()
    out_h5ad = PROJECT_ROOT / "data" / "cornea2" / f"{tag}_annotated.h5ad"
    if out_h5ad.exists() and not args.force:
        log(f"ERROR: output exists {out_h5ad} (use --force)")
        return 1
    out_dir = ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    qc = {"seed": args.seed, "resolution": args.resolution,
          "dataset": args.dataset,
          "mirrors": "protocol_15c_annotate_vanzyl.py (method locked)",
          "harmony_batch_keys": cfg["harmony_keys"]}

    banner(f"Loading {args.dataset}")
    adata = build_adata(args.dataset, args.force)
    qc["input_n_cells"] = int(adata.n_obs)
    qc["input_n_genes"] = int(adata.n_vars)
    if "epithelial_keep" in cfg:
        qc["epithelial_precheck"] = cfg["epithelial_keep"]
        qc["precheck_kept_cells"] = int(adata.n_obs)
        qc["precheck_note"] = (
            "Dataset is dissected by layer; full-object annotation assigns "
            "stromal fibroblasts to LPC/PMC_basal (DCN/PLIN2/SLC2A3 markers "
            "not specific). Mirroring the van Zyl partition epithelial "
            "pre-check, only Layers==Epi cells are annotated (native QC "
            "signal)."
        )

    score_marker_sets(adata)
    qc_filter(adata)
    qc["qc_n_cells"] = int(adata.n_obs)
    qc["qc_n_genes"] = int(adata.n_vars)
    preprocess(adata, args.seed, args.resolution, cfg["harmony_keys"])

    enrich_df = assign_cell_types(adata)
    harmonize(adata)
    fibro_flagged = flag_fibro_contamination(adata)
    qc["fibro_contamination_clusters"] = fibro_flagged
    qc["fibro_contamination_note"] = (
        "Clusters with >=3 of top-5 DEGs in the fibroblast/endothelial "
        "gene set are non-epithelial contamination despite top epithelial "
        "enrichment (LPC markers DCN/PLIN2/SLC2A3 not specific vs stromal "
        "fibroblasts). Mirrors the van Zyl cluster0 ciliary-muscle "
        "exclusion precedent. Frozen before any split/train."
    )
    cross = crosscheck_layers(adata, args.dataset)
    qc["crosscheck"] = cross
    qc["cluster_n"] = int(adata.obs["leiden"].nunique())
    qc["cell_class_4_counts"] = (
        adata.obs["cell_class_4"].value_counts().to_dict()
    )
    qc["assignment_rate_4class"] = round(
        float(adata.obs["is_epithelial"].mean()), 4
    )
    qc["donor_counts"] = adata.obs["donor_id"].value_counts().to_dict()
    qc["donor_x_cell_class4"] = (
        pd.crosstab(adata.obs["donor_id"],
                    adata.obs["cell_class_4"]).to_dict()
    )
    qc["n_donors"] = int(adata.obs["donor_id"].nunique())

    save_umap_plots(adata, out_dir, tag)
    enrich_df.to_csv(out_dir / f"{tag}_marker_scores.csv")
    (out_dir / f"{tag}_qc_decisions.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False)
    )
    write_report(args.dataset, adata, enrich_df, qc)

    banner("Summary")
    print(adata.obs["cell_class_4"].value_counts().to_string())
    print("\n", pd.crosstab(adata.obs["donor_id"],
                            adata.obs["cell_class_4"]).to_string())

    log(f"Saving -> {out_h5ad}")
    adata.write_h5ad(out_h5ad)
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
