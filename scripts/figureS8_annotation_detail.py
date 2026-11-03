"""Figure S8 - external-test annotation detail (GSE153515).

``core_claim``
    The sixteen Leiden clusters are not a lexicographic list but an ordered
    set of biologically coherent groups: each cluster carries one dominant
    marker set, and every cluster maps cleanly onto one harmonized
    differentiation class.

``hero_panel``
    Panel B, the cluster-to-class assignment heatmap, is the hero: it shows
    that the class call is unambiguous for every cluster.

``stats_shown``
    none. Panel A shows the cluster-level marker-set score (colour) and the
    number of cells in the cluster (dot area); panel B shows the row-normalized
    fraction of cells per cluster assigned to each harmonized class. No test
    or interval is claimed.

``source_data``
    ``data/processed/gse153515_annotated.h5ad`` (backed read; ``obs`` columns
    ``leiden`` and ``cell_type_harmonized`` only) and
    ``results/gse153515_annotation/marker_scores.csv`` (cluster-level marker
    scores and cluster sizes).

One figure, one producer: this script is the only writer of
``figures/paper1/figureS8_annotation_detail.{pdf,png}``.
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
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402

import paper1_figure_data as D  # noqa: E402
import paper1_figure_style as S  # noqa: E402

OUT_NAME = "figureS8_annotation_detail"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 104.0)

# Marker-set columns shipped in marker_scores.csv, in the order drawn.
MARKER_SETS = (
    "LSC",
    "LPC",
    "TAC",
    "PMC_basal",
    "TDC",
    "Melanocyte",
    "Conjunctival",
    "Immune",
)
MARKER_LABELS = {
    "LSC": "LSC",
    "LPC": "LPC",
    "TAC": "TAC",
    "PMC_basal": "PMC/basal",
    "TDC": "TDC",
    "Melanocyte": "Melanocyte",
    "Conjunctival": "Conjunctival",
    "Immune": "Immune",
}
# Harmonized classes, ordered from the least to the most differentiated; the
# non-layer reference class is kept last and neutral.
CLASS_ORDER = ("LSC", "TAC", "basal", "differentiated", "NON_LAYER")

DOT_MAX = 58.0
DOT_MIN = 3.0
GRID_COLOR = "#FFFFFF"


def _load_annotation() -> tuple[object, object, Path, Path]:
    """Return the backed AnnData, marker table, and both input paths."""
    data_path = D.resolve_data_path("gse153515_annotated.h5ad")
    marker_path = D.RESULTS / "gse153515_annotation" / "marker_scores.csv"
    markers = D.load_csv(marker_path)

    missing = [name for name in (*MARKER_SETS, "leiden", "n_cells",
                                "assigned") if name not in markers.columns]
    if missing:
        raise KeyError(f"{marker_path} missing columns: {missing}")

    adata = ad.read_h5ad(data_path, backed="r")
    return adata, markers, data_path, marker_path


def _cluster_order(markers) -> list[int]:
    """Clusters in numeric order (never lexicographic)."""
    clusters = sorted(int(value) for value in markers["leiden"].unique())
    if clusters != list(range(clusters[0], clusters[0] + len(clusters))):
        raise AssertionError(f"Leiden clusters are not contiguous: {clusters}")
    return clusters


def _validate(markers, obs, clusters: list[int]) -> None:
    """Cross-check the marker table against the annotation."""
    observed = sorted(int(value) for value in obs["leiden"].astype(int).unique())
    if observed != clusters:
        raise AssertionError(
            f"marker table clusters {clusters} != h5ad clusters {observed}"
        )
    sizes = obs["leiden"].astype(int).value_counts().sort_index().to_numpy()
    table_sizes = markers.set_index(markers["leiden"].astype(int)).loc[
        clusters, "n_cells"
    ].to_numpy()
    if not np.array_equal(sizes, table_sizes.astype(int)):
        raise AssertionError("cluster sizes differ between h5ad and marker table")

    # Every cluster must have a class assignment from the harmonized labels.
    cross = obs.groupby(obs["leiden"].astype(int), observed=True)[
        "cell_type_harmonized"
    ].nunique()
    if int(cross.max()) != 1:
        raise AssertionError("a cluster contains more than one harmonized class")


def _class_matrix(obs, clusters: list[int]) -> np.ndarray:
    """Row-normalized cluster x harmonized-class fraction matrix."""
    labels = obs["cell_type_harmonized"].astype(str)
    clusters_obs = obs["leiden"].astype(int)
    matrix = np.zeros((len(clusters), len(CLASS_ORDER)), dtype=float)
    for row, cluster in enumerate(clusters):
        mask = (clusters_obs == cluster).to_numpy()
        total = int(mask.sum())
        if total == 0:
            raise AssertionError(f"cluster {cluster} has no cells")
        for column, name in enumerate(CLASS_ORDER):
            matrix[row, column] = float((labels[mask] == name).sum()) / total
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise AssertionError("class fractions do not sum to one per cluster")
    return matrix


def _dot_panel(fig, cell, markers, clusters: list[int]):
    """Panel A: cluster x marker-set dot plot with an in-panel key gutter."""
    sub = cell.subgridspec(1, 2, width_ratios=[1.0, 0.30], wspace=0.08)
    ax = fig.add_subplot(sub[0])
    key_ax = fig.add_subplot(sub[1])
    key_ax.axis("off")

    table = markers.set_index(markers["leiden"].astype(int)).loc[clusters]
    scores = table.loc[:, list(MARKER_SETS)].to_numpy(dtype=float)
    sizes = table["n_cells"].to_numpy(dtype=float)
    if np.any(sizes <= 0):
        raise AssertionError("non-positive cluster size in marker table")

    area = DOT_MIN + (DOT_MAX - DOT_MIN) * (sizes - sizes.min()) / (
        sizes.max() - sizes.min()
    )
    x, y = np.meshgrid(np.arange(len(MARKER_SETS)), np.arange(len(clusters)))
    norm = Normalize(vmin=0.0, vmax=float(scores.max()))
    ax.scatter(
        x.ravel(),
        y.ravel(),
        s=np.repeat(area, len(MARKER_SETS)),
        c=scores.ravel(),
        cmap=ListedColormap(S.SEQUENTIAL),
        norm=norm,
        linewidths=0.0,
        zorder=3,
    )
    ax.set_xticks(range(len(MARKER_SETS)))
    ax.set_xticklabels([MARKER_LABELS[name] for name in MARKER_SETS],
                       rotation=90, ha="center", va="top")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([str(cluster) for cluster in clusters])
    ax.set_xlim(-0.6, len(MARKER_SETS) - 0.4)
    ax.set_ylim(len(clusters) - 0.5, -0.5)
    ax.set_xlabel("Marker gene set")
    ax.set_ylabel("Leiden cluster")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)

    colour_axis = key_ax.inset_axes([0.02, 0.58, 0.30, 0.34])
    colourbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=ListedColormap(S.SEQUENTIAL)),
        cax=colour_axis,
        orientation="vertical",
    )
    colourbar.ax.tick_params(labelsize=7, length=2, pad=1)
    colourbar.set_label("Marker-set score", fontsize=7, labelpad=2)

    key_ax.text(0.0, 0.48, "Cells in cluster", fontsize=7, ha="left",
                va="top", transform=key_ax.transAxes)
    example_sizes = (sizes.min(), float(np.median(sizes)), sizes.max())
    key_positions = (0.30, 0.22, 0.12)
    key_areas = (
        DOT_MIN,
        DOT_MIN + (DOT_MAX - DOT_MIN) * 0.5,
        DOT_MAX,
    )
    for value, ypos, area_value in zip(example_sizes, key_positions, key_areas):
        key_ax.scatter(
            [0.14], [ypos], s=area_value, color="#7A7A7A", linewidths=0.0,
            transform=key_ax.transAxes, clip_on=False,
        )
        key_ax.text(
            0.30, ypos, f"{int(round(value)):,}", fontsize=7, ha="left",
            va="center", transform=key_ax.transAxes,
        )
    return ax


def _class_panel(fig, cell, matrix: np.ndarray, clusters: list[int]):
    """Panel B: cluster-to-class assignment heatmap, every row labelled."""
    ax = fig.add_subplot(cell)
    image = ax.imshow(
        matrix,
        cmap=ListedColormap(S.SEQUENTIAL),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=90, ha="center", va="top")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([str(cluster) for cluster in clusters])
    ax.set_xlabel("Harmonized class")
    ax.set_ylabel("Leiden cluster")

    colour_axis = ax.inset_axes([1.03, 0.0, 0.045, 1.0])
    colourbar = fig.colorbar(image, cax=colour_axis, orientation="vertical")
    colourbar.ax.tick_params(labelsize=7, length=2, pad=1)
    colourbar.set_label("Fraction of cells", fontsize=7, labelpad=2)
    return ax


def main() -> int:
    """Render the figure and append its manifest entry."""
    S.set_style()
    adata, markers, data_path, marker_path = _load_annotation()
    try:
        obs = adata.obs.loc[:, ["leiden", "cell_type_harmonized"]].copy()
        clusters = _cluster_order(markers)
        _validate(markers, obs, clusters)
        matrix = _class_matrix(obs, clusters)
    finally:
        adata.file.close()

    fig = plt.figure(figsize=S.figsize(*CANVAS_MM))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.42)
    ax_a = _dot_panel(fig, grid[0], markers, clusters)
    ax_b = _class_panel(fig, grid[1], matrix, clusters)
    S.panel_label(ax_a, "A", x=-0.20)
    S.panel_label(ax_b, "B", x=-0.20)

    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={
            "gse153515_annotated.h5ad": data_path,
            "marker_scores.csv": marker_path,
        },
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print(f"[{OUT_NAME}] {len(clusters)} clusters, numeric order; every row labelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
