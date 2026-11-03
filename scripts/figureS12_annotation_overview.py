"""Figure S12 - annotation overview of the external test set (GSE153515).

``core_claim``
    The 15,827 QC-passing external-test cells resolve into four ordered
    differentiation classes plus a small non-layer compartment, and both
    donors are interleaved across the same embedding rather than separated,
    so the class structure is not a donor artefact.

``hero_panel``
    Panel A, the harmonized cell-type map, carries the annotation claim;
    panel B shows the donor mixing that supports it.

``stats_shown``
    none. Both panels are categorical embeddings; the class sizes underlying
    the claim are asserted against the locked configuration but not printed
    on the canvas.

``source_data``
    ``data/processed/gse153515_annotated.h5ad`` (backed read; ``obs`` columns
    ``cell_type_harmonized`` and ``donor_id`` plus ``obsm/X_umap``),
    ``results/gse153515_annotation/marker_scores.csv`` (cluster-count
    cross-check) and the locked configuration
    ``config/external_test_gse153515.yml``.

The cluster-level annotation detail (marker-set dot plot and cluster-to-class
heatmap) is drawn once, in Figure S8; it is deliberately not repeated here.

One figure, one producer: this script is the only writer of
``figures/paper1/figureS12_annotation_overview.{pdf,png}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import anndata as ad  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import paper1_figure_data as D  # noqa: E402
import paper1_figure_style as S  # noqa: E402

OUT_NAME = "figureS12_annotation_overview"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 82.0)

# The non-layer compartment is not one of the four ordered cell types in the
# shared palette, so it is drawn in a neutral reference grey (never an arm
# colour).  The four ordered classes use CELLTYPE_COLORS verbatim.
NON_LAYER_COLOR = "#BDBDBD"
CELL_PALETTE = dict(S.CELLTYPE_COLORS)
CELL_PALETTE["NON_LAYER"] = NON_LAYER_COLOR
CELL_ORDER = ("LSC", "TAC", "basal", "differentiated", "NON_LAYER")

# Donors: third category, purple family (matches Figure S4).
DONOR_COLORS = {
    "donor_1": "#2D1B69",
    "donor_2": "#7B6BC4",
}
DONORS = ("donor_1", "donor_2")

POINT_SIZE = 0.7


def _load_config() -> tuple[dict, Path]:
    path = S.PROJECT_ROOT / "config" / "external_test_gse153515.yml"
    if not path.exists():
        raise FileNotFoundError(f"locked config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def _umap_panel(fig, cell, coordinates, values, palette, legend_title, labels):
    """Point cloud plus a legend strip to its right (never over the cloud)."""
    sub = cell.subgridspec(1, 2, width_ratios=[1.0, 0.44], wspace=0.02)
    ax = fig.add_subplot(sub[0])
    legend_ax = fig.add_subplot(sub[1])
    legend_ax.axis("off")

    values = np.asarray(values)
    categories = list(labels)
    observed = set(values)
    if observed != set(categories):
        raise ValueError(
            f"plotted categories {sorted(observed)} != palette {sorted(categories)}"
        )
    draw_order = sorted(categories, key=lambda name: -(values == name).sum())
    for name in draw_order:
        mask = values == name
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=POINT_SIZE,
            color=palette[name],
            linewidths=0,
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4,
               markerfacecolor=palette[name], markeredgecolor="none", label=name)
        for name in categories
    ]
    legend_ax.text(0.0, 0.60, legend_title, fontsize=7, fontweight="bold",
                   va="bottom", ha="left", transform=legend_ax.transAxes)
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.58),
        frameon=False,
        handletextpad=0.2,
        labelspacing=0.3,
        borderpad=0.0,
    )
    return ax


def main() -> int:
    """Render the figure and append its manifest entry."""
    S.set_style()
    config, config_path = _load_config()
    data_path = D.resolve_data_path("gse153515_annotated.h5ad")
    marker_path = D.RESULTS / "gse153515_annotation" / "marker_scores.csv"
    markers = D.load_csv(marker_path)

    expected_shape = config["external_test"]["annotated_shape"]
    adata = ad.read_h5ad(data_path, backed="r")
    try:
        if int(adata.n_obs) != int(expected_shape[0]):
            raise AssertionError(
                f"backed h5ad has {adata.n_obs} cells; config declares "
                f"{expected_shape[0]}"
            )
        obs = adata.obs.loc[:, ["cell_type_harmonized", "donor_id",
                                "is_epithelial", "leiden"]].copy()
        umap = np.asarray(adata.obsm["X_umap"])
    finally:
        adata.file.close()

    cell_types = np.asarray(obs["cell_type_harmonized"].astype(str))
    donors = np.asarray(obs["donor_id"].astype(str))

    n_epithelial = int(np.asarray(obs["is_epithelial"]).astype(bool).sum())
    S.check_value("GSE153515 epithelial cells", n_epithelial,
                  D.MANUSCRIPT_VALUES["gse153515"]["n_cells"])
    declared = config["annotation"]["harmonized_distribution"]
    for name, expected in declared.items():
        observed = int((cell_types == name).sum())
        if observed != int(expected):
            raise AssertionError(
                f"{name}: observed {observed} cells != config {expected}"
            )
    if len(cell_types) != sum(int(value) for value in declared.values()):
        raise AssertionError("harmonized distribution does not account for all cells")

    n_clusters = int(markers["leiden"].nunique())
    observed_clusters = int(obs["leiden"].astype(str).nunique())
    if n_clusters != int(config["annotation"]["n_clusters"]):
        raise AssertionError(
            f"marker table has {n_clusters} clusters; config declares "
            f"{config['annotation']['n_clusters']}"
        )
    if observed_clusters != n_clusters:
        raise AssertionError("cluster counts differ between h5ad and marker table")

    fig = plt.figure(figsize=S.figsize(*CANVAS_MM))
    grid = fig.add_gridspec(1, 2, wspace=0.10)
    ax_a = _umap_panel(fig, grid[0], umap, cell_types, CELL_PALETTE,
                       "Cell type", CELL_ORDER)
    ax_b = _umap_panel(fig, grid[1], umap, donors, DONOR_COLORS,
                       "Donor", DONORS)
    S.panel_label(ax_a, "A")
    S.panel_label(ax_b, "B")

    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={
            "gse153515_annotated.h5ad": data_path,
            "marker_scores.csv": marker_path,
            "external_test_gse153515.yml": config_path,
        },
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print(f"[{OUT_NAME}] {len(cell_types)} cells, {len(CELL_ORDER)} classes; "
          "cluster detail delegated to Figure S8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
