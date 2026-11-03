#!/usr/bin/env python
"""Figure 4 - Same-corpus holdout: fine-tuning raises per-class F1 and annotation.

core_claim
    On the in-corpus external holdout (GSE153515, 14,775 epithelial cells from two
    donors) Geneformer adaptation raises donor-level accuracy and per-class F1,
    while the per-class readout of the rarest ordered stage (TAC) remains at zero
    for the CE-naive arm.
titling_note
    No panel in this figure plots rho, so the figure no longer claims to show a
    gradient collapse in its title. The collapse is a Figure 2 statement, where
    rho is plotted per layer; this figure is the annotation half of the pair.
hero_panel
    D - per-class F1 by arm; it localizes where the accuracy comes from and shows
    that the rare TAC class is not recovered by any arm.
stats_shown
    Panel C: one confusion matrix (CE-naive, seed 42) row-normalized to fractions.
    Panel D: per-class F1; adapted arms are means over seeds 42/123/456 with +/-1
    sample SD (ddof=1); reference arms (frozen, HVG/PCA) are single runs and carry
    no interval. Support (n) per class is read from the shipped per-class table.
source_data
    data/processed/gse153515_annotated.h5ad (obs and obsm/X_umap only);
    artifacts/paper1_tier_b_eval/results_0.0_{42,123,456}.json;
    artifacts/paper1_ordinal_eval/results_0.0_{42,123,456}.json;
    artifacts/paper1_ordinal_eval/results_0.5_{42,123,456}.json;
    artifacts/paper1_tier_b_eval/results_frozen_gf.json;
    artifacts/paper1_tier_b_eval/results_hvg_pca.json;
    figures/paper1/table_t4_per_class_metrics.csv.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import anndata as ad  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import paper1_figure_data as D  # noqa: E402
import paper1_figure_style as S  # noqa: E402

# Arm order drawn in panel D (and therefore the legend order).  Keys are the
# canonical vocabulary in paper1_figure_style.ARMS.
PANEL_D_ARMS = ["frozen", "ce_naive", "ordinal", "ordinal_cont", "hvg_pca"]

# Donor encodings.  The shared module reserves ARM_COLORS for arms and
# CELLTYPE_COLORS for cell types, so donors are encoded by marker fill rather
# than by a third hue: both donors use the document ink #1A1A1A, donor 1
# filled and donor 2 open.  Hue therefore stays reserved for cell type, and no
# arm hex is reused: the frozen arm's #4D4D4D is spent on the frozen bar in
# panel D and nowhere else.
DONOR_INK = "#1A1A1A"
DONOR_STYLES = {
    "donor_1": {"facecolors": DONOR_INK, "edgecolors": "none", "linewidths": 0.0},
    "donor_2": {"facecolors": "none", "edgecolors": DONOR_INK, "linewidths": 0.4},
}

# Bar outline.  A white 0.5 pt stroke leaves no edge on the white canvas, so
# the five bars of a group merged; the neutral ink below is the annotation ink
# already used for the error bars, not an arm colour.
BAR_EDGE_COLOR = "#333333"
BAR_EDGE_LW = 0.4

SEEDS = (42, 123, 456)


def _per_class(path):
    """(per_class dict, provenance dict) from a tier-B style result JSON."""
    doc = D.load_json(path)
    return doc["per_class"], doc["provenance"]


def _arm_sources():
    """Arm key -> (list of per-class records, seeded flag, source paths)."""
    sources = {}
    for key, seeded, paths in [
        ("frozen", False, [D.tier_b("frozen_gf")]),
        ("ce_naive", True, [D.tier_b(f"0.0_{seed}") for seed in SEEDS]),
        ("ordinal", True, [D.artifact_json("paper1_ordinal_eval", f"results_0.0_{seed}")
                           for seed in SEEDS]),
        ("ordinal_cont", True, [D.artifact_json("paper1_ordinal_eval", f"results_0.5_{seed}")
                                for seed in SEEDS]),
        ("hvg_pca", False, [D.tier_b("hvg_pca")]),
    ]:
        records = [_per_class(path)[0] for path in paths]
        sources[key] = {"records": records, "seeded": seeded, "paths": paths}
    return sources


def _umap_panel(fig, cell, coordinates, values, palette, legend_title, labels=None,
                marker_styles=None):
    """Draw one UMAP cell: point cloud plus a legend taken outside the cloud.

    ``marker_styles`` optionally maps a category to scatter keyword values
    (``facecolors``/``edgecolors``/``linewidths``) so that a category can be
    encoded by fill instead of hue; unmapped categories fall back to a filled
    marker in their palette colour.
    """
    sub = cell.subgridspec(1, 2, width_ratios=[1.0, 0.48], wspace=0.02)
    ax = fig.add_subplot(sub[0])
    ax_legend = fig.add_subplot(sub[1])
    ax_legend.axis("off")

    coordinates = np.asarray(coordinates)
    values = np.asarray(values)
    categories = list(labels) if labels is not None else sorted(set(values))
    observed = set(values)
    if observed != set(categories):
        raise ValueError(
            f"unexpected plotted categories: observed {sorted(observed)}, "
            f"palette {sorted(categories)}"
        )

    def style_for(name):
        if marker_styles and name in marker_styles:
            return marker_styles[name]
        return {"facecolors": palette[name], "edgecolors": "none", "linewidths": 0.0}

    # Largest class first so the rare classes are drawn on top.
    draw_order = sorted(categories, key=lambda name: -(values == name).sum())
    for name in draw_order:
        mask = values == name
        style = style_for(name)
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=0.6,
            marker="o",
            facecolors=style["facecolors"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            rasterized=True,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = []
    for name in categories:
        style = style_for(name)
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markersize=4,
                markerfacecolor=style["facecolors"],
                markeredgecolor=style["edgecolors"],
                markeredgewidth=style["linewidths"],
                label=name,
            )
        )
    ax_legend.text(0.0, 0.58, legend_title, fontsize=S.TYPE["tick"],
                   fontweight="bold", va="bottom", ha="left",
                   transform=ax_legend.transAxes)
    ax_legend.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.56),
        frameon=False,
        handletextpad=0.2,
        labelspacing=0.25,
        borderpad=0.0,
    )
    return ax


def _confusion_panel(ax, best_key, best_path):
    """Row-normalized confusion matrix for the CE-naive reference checkpoint."""
    doc = D.load_json(best_path)
    matrix = np.asarray(doc["confusion"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError(f"confusion matrix row with zero cells in {best_path}")
    normalized = matrix / row_sums

    classes = list(doc["provenance"]["classes"])
    image = ax.imshow(
        normalized,
        cmap=ListedColormap(S.SEQUENTIAL),
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        # Fill the allocated panel so panels C and D share one row height.
        aspect="auto",
    )
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel("Predicted layer")
    ax.set_ylabel("True layer")
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = normalized[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=S.TYPE["tick"],
                    color="white" if value > 0.55 else "#222222")
    ax.set_title(S.ARMS[best_key[0]], fontsize=S.TYPE["title"])
    # A 0.5 pt keyline closes the matrix on the colourbar side.
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_linewidth(0.5)
    cax = ax.inset_axes([1.06, 0.0, 0.045, 1.0])
    colorbar = ax.figure.colorbar(image, cax=cax, orientation="vertical")
    colorbar.ax.tick_params(labelsize=S.TYPE["tick"], length=2, pad=1)
    colorbar.set_label("Row-normalized fraction", fontsize=S.TYPE["tick"],
                       labelpad=2)


def _per_class_panel(ax, sources, support, zero_arm="ce_naive"):
    """Grouped per-class F1 bars, one group per class, one bar per arm."""
    classes = list(support)
    x = np.arange(len(classes))
    width = 0.8 / len(PANEL_D_ARMS)
    handles = []

    for index, key in enumerate(PANEL_D_ARMS):
        source = sources[key]
        means, errors = [], []
        for name in classes:
            values = [float(record[name]["f1"]) for record in source["records"]]
            mean = statistics.fmean(values)
            if len(values) > 1:
                errors.append(statistics.stdev(values))
            else:
                errors.append(0.0)
            means.append(mean)
        offset = (index - (len(PANEL_D_ARMS) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            color=S.ARM_COLORS[key],
            label=S.ARMS[key],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LW,
            yerr=errors if source["seeded"] else None,
            capsize=1.6,
            error_kw={"elinewidth": 0.6, "ecolor": "#333333"},
        )
        handles.append(
            Patch(facecolor=S.ARM_COLORS[key], edgecolor=BAR_EDGE_COLOR,
                  linewidth=BAR_EDGE_LW, label=S.ARMS[key])
        )

    zero_index = classes.index("TAC")
    zero_offset = (PANEL_D_ARMS.index(zero_arm)
                   - (len(PANEL_D_ARMS) - 1) / 2) * width
    callout_x = zero_index + zero_offset
    ax.annotate(
        "F1 = 0",
        xy=(callout_x, 0.005),
        xycoords="data",
        xytext=(callout_x, 0.075),
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#333333",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "#555555"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0.0, 1.10)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return handles


def _add_panel_letters(fig, panels):
    """Place each panel letter at the left ink edge shared by its column.

    ``panels`` maps letter -> axes.  Letters in the same column (A/C and B/D)
    take the same figure x, so the two columns read as two aligned strips; the
    letter sits just above its axes and never inside the plotting area.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px, height_px = fig.bbox.width, fig.bbox.height
    ink_left = {}
    for letter, ax in panels.items():
        column = "left" if letter in ("A", "C") else "right"
        ink_left.setdefault(column, []).append(
            ax.get_tightbbox(renderer).x0
        )
    column_x = {column: min(values) for column, values in ink_left.items()}

    for letter, ax in panels.items():
        column = "left" if letter in ("A", "C") else "right"
        axes_box = ax.get_window_extent(renderer)
        text = fig.text(
            column_x[column] / width_px,
            (axes_box.y1 + 2.0 * fig.dpi / 72.0) / height_px,
            letter,
            fontsize=S.TYPE["letter"],
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
        text.set_in_layout(False)


def main() -> None:
    S.set_style()
    inputs = {}

    external_path = D.resolve_data_path("gse153515_annotated.h5ad")
    inputs["gse153515_annotated.h5ad"] = external_path
    table_path = D.fig_table("table_t4_per_class_metrics")
    inputs["table_t4_per_class_metrics.csv"] = table_path
    table = D.load_csv(table_path)

    adata = ad.read_h5ad(external_path, backed="r")
    try:
        obs = adata.obs
        provenance = D.load_json(D.tier_b("0.0_42"))["provenance"]
        n_total = int(provenance["n_eval_cells"])
        S.check_value("GSE153515 external cells", n_total,
                      D.MANUSCRIPT_VALUES["gse153515"]["n_cells"])
        keep = np.asarray(obs["is_epithelial"]).astype(bool)
        external_obs = obs[keep]
        umap = np.asarray(adata.obsm["X_umap"])[keep]
        cell_types = np.asarray(external_obs["cell_type_harmonized"].astype(str))
        donors = np.asarray(external_obs["donor_id"].astype(str))
        if len(external_obs) != n_total:
            raise AssertionError(
                f"epithelial subset {len(external_obs)} != reported {n_total}"
            )

        # Class support: the shipped per-class table and the annotation agree.
        support = {}
        for name in ("LSC", "TAC", "basal", "differentiated"):
            table_row = table[table["class"] == name]
            if len(table_row) != 1:
                raise AssertionError(f"per-class table row for {name!r} missing")
            support[name] = int(float(table_row["support"].iloc[0]))
            observed = int((cell_types == name).sum())
            if support[name] != observed:
                raise AssertionError(
                    f"support mismatch for {name!r}: table {support[name]} "
                    f"vs annotation {observed}"
                )
        S.check_value("TAC fraction of external test",
                      support["TAC"] / n_total * 100.0, 1.2, tol=0.05)
        if sum(support.values()) != n_total:
            raise AssertionError("class supports do not sum to the external test size")

        sources = _arm_sources()
        for key, source in sources.items():
            for path in source["paths"]:
                inputs[f"{Path(path).parent.name}/{Path(path).name}"] = path
        # The per-class table must reproduce the CE-naive seed mean.
        for name in support:
            values = [float(record[name]["f1"]) for record in sources["ce_naive"]["records"]]
            row = table[table["class"] == name]
            S.check_value(f"CE-naive {name} F1", statistics.fmean(values),
                          float(row["f1"].iloc[0]))

        fig = plt.figure(figsize=S.figsize(178.0, 150.0))
        # rect is (left, bottom, width, height) for the constrained engine; the
        # reserved top strip keeps the panel letters clear of the trim.
        fig.get_layout_engine().set(rect=(0.028, 0.018, 0.957, 0.940))
        # One column ratio for both rows, so the two panels in a column start
        # at the same x and the row heights are set by the row, not the aspect.
        grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25],
                                height_ratios=[0.85, 1.0],
                                hspace=0.30, wspace=0.26)

        ax_a = _umap_panel(fig, grid[0, 0], umap, cell_types,
                           dict(S.CELLTYPE_COLORS), "Cell type",
                           labels=list(S.CELLTYPE_COLORS))
        ax_b = _umap_panel(fig, grid[0, 1], umap, donors,
                           dict.fromkeys(DONOR_STYLES, DONOR_INK), "Donor",
                           labels=list(DONOR_STYLES),
                           marker_styles=DONOR_STYLES)

        ax_c = fig.add_subplot(grid[1, 0])
        best_path = D.tier_b("0.0_42")
        _confusion_panel(ax_c, ("ce_naive", provenance["seed"]), best_path)

        ax_d = fig.add_subplot(grid[1, 1])
        handles = _per_class_panel(ax_d, sources, support)
        ax_d.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.23), ncol=3, frameon=False,
                    columnspacing=0.7, handlelength=1.0, labelspacing=0.35,
                    handletextpad=0.35)

        _add_panel_letters(fig, {"A": ax_a, "C": ax_c, "B": ax_b, "D": ax_d})

        S.save_figure(fig, "figure4_external_eval", inputs=inputs)
        plt.close(fig)
    finally:
        adata.file.close()


if __name__ == "__main__":
    main()
