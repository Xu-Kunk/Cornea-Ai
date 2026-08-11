#!/usr/bin/env python
"""Generate publication-quality DATA-ONLY figures and tables for the Paper1
manuscript (Ophthalmology Science / Elsevier style).

Outputs (figures/paper1/):
  figure3_gse153515_annotation.pdf/.png   — full width (174 mm), panels A-D
  figureS4_gse153515_qc.pdf/.png          — single column (85 mm), panels A-B
  table1_dataset_summary.csv              — training / external test datasets
  table2_model_arms.csv                   — 4 model arms
  table6_provenance.csv                   — model / tokenizer / data provenance

Notes
-----
* Data-only figures: no model training results required.
* The annotated h5ad stores Ensembl IDs as the var index; gene symbols in
  var['gene_symbols'] are NOT real symbols (they are Ensembl IDs). Marker genes
  for panel C are therefore addressed by Ensembl ID, with symbols shown as
  tick labels via the scanpy dotplot `gene_symbols` mechanism.
* A2M/ABCG2/TSPAN7/SOX17 (LSC) and MKI67/TOP2A/PLK1 (TAC) plus ADIRF and MLANA
  were removed by the protocol's min_cells gene filter and are absent from the
  annotated file. They are recovered from data/gse153515_limbal_stem.h5ad and
  re-normalized with the exact protocol formula
  log1p(counts / total_counts_per_cell * 10000), where total_counts is the
  per-cell sum of the annotated layers['counts'] (the gene-filtered matrix on
  which normalize_total was applied).
* Okabe-Ito colorblind-safe palette; Liberation Sans (metric-compatible with
  Arial) used because Arial/Helvetica are not installed on this system.

Usage
-----
    python3 scripts/generate_paper1_data_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATED = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
RAW = PROJECT_ROOT / "data" / "gse153515_limbal_stem.h5ad"
MARKER_SCORES = PROJECT_ROOT / "artifacts" / "gse153515_annotation" / "marker_scores.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "paper1"

FULL_WIDTH_IN = 174 / 25.4   # 6.850 in
SINGLE_COL_IN = 85 / 25.4    # 3.346 in

# Okabe-Ito colorblind-safe palette
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_BLUEGREEN = "#009E73"
OI_ORANGE = "#E69F00"
OI_SKYBLUE = "#56B4E9"
OI_YELLOW = "#F0E442"
OI_RED = "#D55E00"
OI_GRAY = "#999999"

CELLTYPE_COLORS = {
    "LSC": OI_BLUE,
    "TAC": OI_VERMILLION,
    "basal": OI_BLUEGREEN,
    "differentiated": OI_ORANGE,
    "NON_LAYER": OI_GRAY,
}
DONOR_COLORS = {"donor_1": OI_BLUE, "donor_2": OI_ORANGE}

# Marker genes (Ensembl ID -> symbol) for Figure 3 panel C.
MARKER_GROUPS = {
    "LSC": {
        "ENSG00000175899": "A2M",
        "ENSG00000118777": "ABCG2",
        "ENSG00000156298": "TSPAN7",
        "ENSG00000164736": "SOX17",
    },
    "TAC": {
        "ENSG00000148773": "MKI67",
        "ENSG00000131747": "TOP2A",
        "ENSG00000166851": "PLK1",
    },
    "PMC/basal": {
        "ENSG00000186081": "KRT14",
        "ENSG00000171345": "KRT15",
        "ENSG00000186395": "KRT5",
    },
    "TDC": {
        "ENSG00000170477": "KRT12",
        "ENSG00000196569": "KRT3",
        "ENSG00000204929": "ADIRF",
    },
    "Melanocyte": {
        "ENSG00000130203": "MITF",
        "ENSG00000185269": "MLANA",
    },
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "pdf.fonttype": 42,      # embed TrueType (editable text in vector PDF)
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#000000",
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    """Place a bold uppercase panel label (12 pt) at the top-left of `ax`."""
    pos = ax.get_position()
    fig.text(
        pos.x0 - 0.004,
        pos.y1 + 0.008,
        label,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(fig: plt.Figure, name: str) -> list[Path]:
    out_pdf = OUT_DIR / f"{name}.pdf"
    out_png = OUT_DIR / f"{name}.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return [out_pdf, out_png]


# ---------------------------------------------------------------------------
# Data loading & marker-gene recovery
# ---------------------------------------------------------------------------


def load_annotated():
    adata = sc.read_h5ad(ANNOTATED)
    return adata


def compute_total_counts(adata) -> np.ndarray:
    """Per-cell total counts from layers['counts'] (protocol's gene-filtered
    matrix — the exact denominator normalize_total used)."""
    return np.asarray(adata.layers["counts"].sum(axis=1)).ravel().astype(np.float64)


def recover_missing_markers(adata, ens_ids: list[str]) -> np.ndarray:
    """Recover genes missing from the annotated file from the raw file and
    re-normalize with the protocol formula log1p(cnt/total*1e4)."""
    raw = sc.read_h5ad(RAW, backed="r")
    try:
        missing = [e for e in ens_ids if e not in adata.var_names]
        if not missing:
            return np.zeros((adata.n_obs, len(ens_ids)), dtype=np.float32)
        barcodes = adata.obs_names.tolist()
        sub = raw[barcodes, missing]          # same cell order as annotated
        Xraw = sub.X
        if hasattr(Xraw, "toarray"):
            Xraw = Xraw.toarray()
        cnt = np.asarray(Xraw, dtype=np.float64)
        total = compute_total_counts(adata)   # (n_cells,)
        norm = np.log1p(cnt / total[:, None] * 10_000.0)
        out = np.zeros((adata.n_obs, len(ens_ids)), dtype=np.float32)
        for j, e in enumerate(ens_ids):
            if e in missing:
                out[:, j] = norm[:, missing.index(e)]
        return out
    finally:
        raw.file.close()


def build_dotplot_adata(adata) -> sc.AnnData:
    """Temp AnnData: log-normalized expression of the 15 curated markers
    (16 Leiden clusters in obs)."""
    all_ens, all_sym, all_cols = [], [], []
    for grp in MARKER_GROUPS.values():
        for ens, sym in grp.items():
            all_ens.append(ens)
            all_sym.append(sym)
    present = [e for e in all_ens if e in adata.var_names]
    present_idx = [all_ens.index(e) for e in present]

    # Columns in the full all_ens order; fill present genes from annotated
    # normalized layer, missing genes from the raw file.
    X = np.zeros((adata.n_obs, len(all_ens)), dtype=np.float32)
    if present:
        norm_layer = adata[:, present].layers["normalized"]
        if hasattr(norm_layer, "toarray"):
            norm_layer = norm_layer.toarray()
        norm = np.asarray(norm_layer, dtype=np.float32)
        X[:, present_idx] = norm
    missing = [e for e in all_ens if e not in adata.var_names]
    if missing:
        miss_norm = recover_missing_markers(adata, missing)
        miss_idx = [all_ens.index(e) for e in missing]
        X[:, miss_idx] = miss_norm
    all_cols = all_ens

    obs_df = pd.DataFrame({"leiden": adata.obs["leiden"].astype(str).values},
                          index=adata.obs_names)
    var_df = pd.DataFrame(
        {"gene_symbols": all_sym},
        index=pd.Index(all_cols, name="var_names"),
    )
    tmp = sc.AnnData(X=X, obs=obs_df, var=var_df)
    tmp.var_names_make_unique()
    tmp.obs["leiden"] = pd.Categorical(
        tmp.obs["leiden"], categories=sorted(tmp.obs["leiden"].unique())
    )
    return tmp


# ---------------------------------------------------------------------------
# Figure 3 — external test data annotation (full width, panels A-D)
# ---------------------------------------------------------------------------


def figure3(adata) -> list[Path]:
    umap = np.asarray(adata.obsm["X_umap"])
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.8))
    gs = fig.add_gridspec(
        2, 2, left=0.045, right=0.969, top=0.955, bottom=0.122,
        wspace=0.25, hspace=0.38, height_ratios=[0.95, 1.05],
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ---- Panel A: UMAP colored by cell_type_harmonized ----
    try:
        cats = ["LSC", "TAC", "basal", "differentiated", "NON_LAYER"]
        ct = adata.obs["cell_type_harmonized"].astype(str).values
        colors = np.array([CELLTYPE_COLORS[c] for c in ct])
        axA.scatter(umap[:, 0], umap[:, 1], c=colors, s=1.2, linewidths=0,
                    rasterized=True)
        axA.set_aspect("equal")
        axA.set_xlabel("UMAP1"); axA.set_ylabel("UMAP2")
        axA.set_xticks([]); axA.set_yticks([])
        for sp in axA.spines.values():
            sp.set_visible(False)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CELLTYPE_COLORS[c],
                       markersize=6, label=c)
            for c in cats
        ]
        axA.legend(handles=handles, loc="upper left", frameon=True,
                   framealpha=0.85, fontsize=7, handletextpad=0.3, borderpad=0.4)
        axA.set_title("Harmonized cell type", fontsize=9, fontweight="bold")
        panel_label(fig, axA, "A")
    except Exception as e:  # noqa: BLE001 — keep rendering remaining panels
        print(f"  [WARN] Figure 3 panel A failed: {e}")
        axA.text(0.5, 0.5, f"panel A failed:\n{e}", ha="center", va="center",
                 transform=axA.transAxes, fontsize=7)

    # ---- Panel B: UMAP colored by donor_id ----
    try:
        donors = ["donor_1", "donor_2"]
        d = adata.obs["donor_id"].astype(str).values
        dcolors = np.array([DONOR_COLORS[x] for x in d])
        axB.scatter(umap[:, 0], umap[:, 1], c=dcolors, s=1.2, linewidths=0,
                    rasterized=True)
        axB.set_aspect("equal")
        axB.set_xlabel("UMAP1"); axB.set_ylabel("UMAP2")
        axB.set_xticks([]); axB.set_yticks([])
        for sp in axB.spines.values():
            sp.set_visible(False)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DONOR_COLORS[x],
                       markersize=6, label=x)
            for x in donors
        ]
        axB.legend(handles=handles, loc="upper left", frameon=True,
                   framealpha=0.85, fontsize=7, handletextpad=0.3, borderpad=0.4)
        axB.set_title("Donor (batch)", fontsize=9, fontweight="bold")
        panel_label(fig, axB, "B")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure 3 panel B failed: {e}")
        axB.text(0.5, 0.5, f"panel B failed:\n{e}", ha="center", va="center",
                 transform=axB.transAxes, fontsize=7)

    # ---- Panel C: dot plot of marker genes per Leiden cluster ----
    try:
        tmp = build_dotplot_adata(adata)
        # var index = Ensembl IDs; var['gene_symbols'] = symbols. Passing the
        # SYMBOLS as var_names lets scanpy map them through var['gene_symbols']
        # back onto the Ensembl var index, while tick labels show symbols.
        var_names = {
            grp: list(mapping.values()) for grp, mapping in MARKER_GROUPS.items()
        }
        dp = sc.pl.DotPlot(
            tmp,
            var_names=var_names,
            gene_symbols="gene_symbols",
            groupby="leiden",
            ax=axC,
            standard_scale="var",   # scale each gene across clusters
            cmap="viridis",
        )
        dp.style(dot_edge_lw=0.3).legend(
            colorbar_title="Mean expression in group",
            size_title="Fraction of cells",
            width=0.9,
        )
        dp.make_figure()
        # Tighten tick-label sizes so the 16-cluster rows stay legible in the
        # constrained grid cell (default scanpy "small" = 8.3pt is too large).
        dp.get_axes()["mainplot_ax"].tick_params(labelsize=7.5)
        panel_label(fig, axC, "C")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure 3 panel C failed: {e}")
        axC.text(0.5, 0.5, f"panel C failed:\n{e}", ha="center", va="center",
                 transform=axC.transAxes, fontsize=7)
        panel_label(fig, axC, "C")

    # ---- Panel D: DEG enrichment heatmap per cluster ----
    try:
        scores = pd.read_csv(MARKER_SCORES)
        score_cols = ["LSC", "LPC", "TAC", "PMC_basal", "TDC", "Melanocyte",
                      "Conjunctival"]
        order = ["TAC", "PMC_basal", "TDC", "LPC", "conjunctival", "melanocyte"]
        scores["assigned"] = pd.Categorical(
            scores["assigned"], categories=order, ordered=True
        )
        scores = scores.sort_values(["assigned", "leiden"], kind="stable")
        idx = [
            f"{int(r.leiden)} · {r.assigned}"
            for r in scores.itertuples(index=False)
        ]
        mat = scores[score_cols].to_numpy(dtype=float)
        im = axD.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1,
                        interpolation="nearest")
        axD.set_xticks(range(len(score_cols)))
        axD.set_xticklabels(score_cols, rotation=55, ha="right", fontsize=6)
        axD.set_yticks(range(len(idx)))
        axD.set_yticklabels(idx, fontsize=6.5)
        axD.set_xlabel("Marker set", fontsize=8)
        axD.set_title("DEG enrichment per Leiden cluster", fontsize=9,
                      fontweight="bold")
        for i in range(len(idx) - 1):
            axD.axhline(i + 0.5, color="white", linewidth=0.6)
        for j in range(len(score_cols) - 1):
            axD.axvline(j + 0.5, color="white", linewidth=0.6)
        divider = make_axes_locatable(axD)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Enrichment score", fontsize=7)
        # Tight padding keeps the tick labels inside the canvas
        # (cax is pinned to the panel's right edge).
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        cb.ax.tick_params(labelsize=6.5)
        panel_label(fig, axD, "D")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure 3 panel D failed: {e}")
        axD.text(0.5, 0.5, f"panel D failed:\n{e}", ha="center", va="center",
                 transform=axD.transAxes, fontsize=7)
        panel_label(fig, axD, "D")

    return save_figure(fig, "figure3_gse153515_annotation")


# ---------------------------------------------------------------------------
# Figure S4 — GSE153515 QC (single column, panels A-B)
# ---------------------------------------------------------------------------


def figureS4(adata) -> list[Path]:
    total_counts = compute_total_counts(adata)
    obs = adata.obs[["donor_id", "n_genes"]].copy()
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["total_counts"] = total_counts

    fig = plt.figure(figsize=(SINGLE_COL_IN, 4.3))
    gs = fig.add_gridspec(2, 1, left=0.17, right=0.97, top=0.93, bottom=0.115,
                          hspace=0.42)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # ---- Panel A: n_genes distribution per donor ----
    try:
        sns.violinplot(
            data=obs, x="donor_id", y="n_genes",
            palette=[DONOR_COLORS[d] for d in ["donor_1", "donor_2"]],
            inner="box", linewidth=0.7, saturation=0.9, ax=axA,
        )
        axA.set_xlabel("Donor", fontsize=8)
        axA.set_ylabel("Genes detected per cell", fontsize=8)
        axA.tick_params(labelsize=8)
        panel_label(fig, axA, "A")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure S4 panel A failed: {e}")
        axA.text(0.5, 0.5, f"panel A failed:\n{e}", ha="center", va="center",
                 transform=axA.transAxes, fontsize=7)
        panel_label(fig, axA, "A")

    # ---- Panel B: n_genes vs total_counts ----
    try:
        colors = np.array([DONOR_COLORS[d] for d in obs["donor_id"].values])
        axB.scatter(obs["total_counts"].values, obs["n_genes"].values,
                    c=colors, s=1.5, linewidths=0, alpha=0.5, rasterized=True)
        # Headroom so the outermost tick labels never touch the panel edges.
        axB.set_xlim(0, float(obs["total_counts"].max()) * 1.08)
        axB.set_ylim(0, float(obs["n_genes"].max()) * 1.08)
        axB.set_xlabel("Total counts per cell", fontsize=8)
        axB.set_ylabel("Genes detected per cell", fontsize=8)
        axB.tick_params(labelsize=8)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DONOR_COLORS[d],
                       markersize=5, label=d)
            for d in ["donor_1", "donor_2"]
        ]
        axB.legend(handles=handles, loc="lower right", frameon=True,
                   framealpha=0.85, fontsize=7)
        panel_label(fig, axB, "B")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure S4 panel B failed: {e}")
        axB.text(0.5, 0.5, f"panel B failed:\n{e}", ha="center", va="center",
                 transform=axB.transAxes, fontsize=7)
        panel_label(fig, axB, "B")

    return save_figure(fig, "figureS4_gse153515_qc")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table1() -> Path:
    rows = [
        {
            "dataset": "Training atlas",
            "source": "CELLxGENE 0f7d022a",
            "cells_raw": "~815,000",
            "cells_qc": 99_606,
            "donors": 49,
            "studies": "10 (chen_limbus/cornea/sclera/fetal, lako_GSE155683, "
                       "shi_GSE157474, chakravarti_GSE218123, dickman_GSE186433, "
                       "li_GSE147979)",
            "platform": "10x Genomics",
            "cell_types": "20",
            "role": "training",
            "checksum": "sha256:bc357c7d57977ab8af95cc97a33d257dc5273919f21cb1792d54f97646bb39b6",
        },
        {
            "dataset": "External test (GSE153515)",
            "source": "GEO GSE153515",
            "cells_raw": 17_587,
            "cells_qc": 15_827,
            "donors": "2 (GSM4646295, GSM4646296)",
            "studies": "1 (li_GSE153515 limbal atlas)",
            "platform": "10x Genomics",
            "cell_types": "4 (harmonized: LSC, TAC, basal, differentiated + "
                         "NON_LAYER)",
            "role": "external_test",
            "checksum": "",
        },
    ]
    df = pd.DataFrame(rows)
    out = OUT_DIR / "table1_dataset_summary.csv"
    df.to_csv(out, index=False)
    return out


def table2() -> Path:
    rows = [
        {
            "arm": "HVG/PCA",
            "backbone": "PCA (scaled HVG expression)",
            "trainable": "yes",
            "lambda_cont": "N/A",
            "seeds": 1,
            "epochs": 3,
            "batch_size": 32,
            "precision": "float32",
            "head": "linear 4-class",
            "loss": "cross-entropy",
        },
        {
            "arm": "Frozen Geneformer",
            "backbone": "Geneformer-V2-104M",
            "trainable": "no",
            "lambda_cont": "N/A",
            "seeds": 1,
            "epochs": 3,
            "batch_size": 32,
            "precision": "bf16",
            "head": "linear 4-class",
            "loss": "cross-entropy",
        },
        {
            "arm": "Fine-tune lambda=0",
            "backbone": "Geneformer-V2-104M",
            "trainable": "yes",
            "lambda_cont": 0,
            "seeds": 3,
            "epochs": 3,
            "batch_size": 32,
            "precision": "bf16",
            "head": "linear 31-class",
            "loss": "L_cls",
        },
        {
            "arm": "CAFT lambda=0.5",
            "backbone": "Geneformer-V2-104M",
            "trainable": "yes",
            "lambda_cont": 0.5,
            "seeds": 3,
            "epochs": 3,
            "batch_size": 32,
            "precision": "bf16",
            "head": "linear 31-class",
            "loss": "L_cls + 0.5*L_cont",
        },
    ]
    df = pd.DataFrame(rows)
    out = OUT_DIR / "table2_model_arms.csv"
    df.to_csv(out, index=False)
    return out


def table6() -> Path:
    rows = [
        {
            "component": "Model",
            "detail": "Geneformer-V2-104M (HuggingFace snapshot 04c2b2e8)",
            "weights_checksum": "SHA-256 fff5cba2...",
            "corpus": "Genecorpus-104M (~104M cells, 2,903 datasets)",
            "retrieved": "2026-07-23",
        },
        {
            "component": "Tokenizer",
            "detail": "Geneformer tokenizer (pkl gc104M dicts)",
            "weights_checksum": "",
            "corpus": "Genecorpus-104M",
            "retrieved": "",
        },
        {
            "component": "Training data",
            "detail": "CELLxGENE corpus 0f7d022a (dataset 30127f02)",
            "weights_checksum": "MD5 39f098...",
            "corpus": "corneal/limbal scRNA-seq (49 donors, 10 studies)",
            "retrieved": "2026-07-22",
        },
        {
            "component": "External test",
            "detail": "GSE153515 (17,587 cells; 15,827 after QC)",
            "weights_checksum": "",
            "corpus": "DOIs: 10.1016/j.jtos.2020.12.004, "
                     "10.1167/iovs.62.9.36",
            "retrieved": "",
        },
    ]
    df = pd.DataFrame(rows)
    out = OUT_DIR / "table6_provenance.csv"
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    generated: list[Path] = []

    print("Loading annotated h5ad ...")
    adata = load_annotated()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes; "
          f"leiden clusters: {adata.obs['leiden'].nunique()}")

    print("\n[Figure 3] External test data annotation ...")
    generated += figure3(adata)

    print("\n[Figure S4] GSE153515 QC ...")
    generated += figureS4(adata)

    print("\n[Tables]")
    generated.append(table1())
    generated.append(table2())
    generated.append(table6())

    print("\n=== Summary of generated files ===")
    for p in generated:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(PROJECT_ROOT)}  ({size_kb:.1f} kB)")
    print(f"\nDone. {len(generated)} files written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
