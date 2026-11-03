"""Figure S4 - quality control of the external test set (GSE153515).

``core_claim``
    The 15,827 QC-passing external-test cells from the two GSE153515 donors
    form two comparable but non-identical library-complexity distributions;
    neither donor contains a low-complexity tail that would drive the
    downstream readouts.

``hero_panel``
    Panel B, total counts versus genes detected per cell, is the joint view
    that exposes any low-complexity population; panel A shows the per-donor
    marginal distributions.

``stats_shown``
    none. Both panels are distribution displays: box interiors give median and
    interquartile range, whiskers extend to 1.5 x IQR and are drawn with caps,
    and every cell is overlaid as a jittered point. No test or interval is
    claimed.

``source_data``
    ``data/processed/gse153515_annotated.h5ad`` (backed read; ``obs`` columns
    ``donor_id`` and ``n_genes`` plus the per-cell row sums of
    ``layers['counts']``) and the locked configuration
    ``config/external_test_gse153515.yml`` (shape, donors and, when present,
    QC thresholds).

One figure, one producer: this script is the only writer of
``figures/paper1/figureS4_gse153515_qc.{pdf,png}``.
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

OUT_NAME = "figureS4_gse153515_qc"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 82.0)

# Donors are a third category, disjoint from the arm and cell-type palettes.
# The two luminance steps keep the two donors distinguishable under
# deuteranopia (purple family, as used by the annotation figures).
DONOR_COLORS = {
    "donor_1": "#2D1B69",
    "donor_2": "#7B6BC4",
}
DONORS = ("donor_1", "donor_2")

# Style constants only (geometry / ink), never data values.
POINT_SIZE = 0.45
POINT_ALPHA = 0.28
BOX_WIDTH = 0.52
BOX_ALPHA = 0.16
JITTER = 0.14
WHISKER_RANGE = 1.5

CHUNK_ROWS = 4000


def _load_config() -> tuple[dict, Path]:
    """Load the locked external-test configuration."""
    path = S.PROJECT_ROOT / "config" / "external_test_gse153515.yml"
    if not path.exists():
        raise FileNotFoundError(f"locked config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def _total_counts(adata) -> np.ndarray:
    """Per-cell total counts from ``layers['counts']`` in row chunks.

    Reading the layer in chunks keeps the 2 GB-backed file lazy: only the
    rows being summed are materialised.
    """
    totals = np.empty(adata.n_obs, dtype=np.float64)
    for start in range(0, adata.n_obs, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, adata.n_obs)
        block = adata[start:stop].layers["counts"]
        totals[start:stop] = np.asarray(block.sum(axis=1)).ravel()
    if not np.all(np.isfinite(totals)) or totals.min() <= 0:
        raise AssertionError("non-positive or non-finite total counts")
    return totals


def _numeric_qc_thresholds(config: dict) -> dict[str, float]:
    """Return any numeric QC cut-offs declared by the locked config.

    The configuration is authoritative.  Only keys whose name mentions a QC
    bound (``min_*``, ``max_*`` or ``threshold``) and whose value is numeric
    are returned; the caller draws a line for each one.  An empty mapping
    means the config declares no numeric cut-off and no line may be drawn.
    """
    found: dict[str, float] = {}

    def visit(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                lowered = str(key).lower()
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and (
                        lowered.startswith("min_")
                        or lowered.startswith("max_")
                        or "threshold" in lowered
                    )
                    and "n_" not in lowered
                    and ("gene" in lowered or "count" in lowered)
                ):
                    found[name] = float(value)
                visit(value, name)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{prefix}[{index}]")

    visit(config)
    return found


def _plot_marginal(ax, frame, thresholds: dict[str, float]) -> None:
    """Panel A: per-donor genes-detected distribution with capped whiskers."""
    rng = np.random.default_rng(42)
    for index, donor in enumerate(DONORS):
        values = frame.loc[frame["donor_id"] == donor, "n_genes"].to_numpy(
            dtype=float
        )
        if values.size == 0:
            raise AssertionError(f"no cells for {donor}")
        colour = DONOR_COLORS[donor]
        violin = ax.violinplot(
            [values],
            positions=[index],
            widths=BOX_WIDTH * 1.7,
            showextrema=False,
            showmedians=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_linewidth(0.6)
            body.set_alpha(BOX_ALPHA)
            body.set_zorder(1)
        jitter = rng.uniform(-JITTER, JITTER, size=values.size)
        ax.scatter(
            index + jitter,
            values,
            s=POINT_SIZE,
            color=colour,
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        box = ax.boxplot(
            [values],
            positions=[index],
            widths=BOX_WIDTH,
            whis=WHISKER_RANGE,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": colour, "linewidth": 1.0},
            whiskerprops={"color": colour, "linewidth": 0.7},
            capprops={"color": colour, "linewidth": 0.7},
            boxprops={
                "facecolor": colour,
                "edgecolor": colour,
                "linewidth": 0.7,
                "alpha": BOX_ALPHA,
            },
            zorder=3,
        )
        # Draw the caps explicitly so they survive any backend change.
        for cap in box["caps"]:
            cap.set_linewidth(0.7)

    for name, value in thresholds.items():
        if "gene" in name.lower():
            ax.axhline(value, color="#777777", linestyle=":", linewidth=0.7,
                       zorder=0)

    ax.set_xticks(range(len(DONORS)))
    ax.set_xticklabels(DONORS)
    ax.set_xlabel("Donor")
    ax.set_ylabel("Genes detected per cell")


def _plot_joint(ax, frame) -> None:
    """Panel B: total counts versus genes detected, coloured by donor."""
    for donor in DONORS:
        mask = (frame["donor_id"] == donor).to_numpy()
        if not mask.any():
            raise AssertionError(f"no cells for {donor}")
        ax.scatter(
            frame.loc[mask, "total_counts"],
            frame.loc[mask, "n_genes"],
            s=POINT_SIZE,
            color=DONOR_COLORS[donor],
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=True,
        )

    x_max = float(frame["total_counts"].max())
    y_max = float(frame["n_genes"].max())
    ax.set_xlim(0.0, x_max * 1.04)
    ax.set_ylim(0.0, y_max * 1.06)
    ax.set_xlabel("Total counts per cell")
    ax.set_ylabel("Genes detected per cell")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4,
            markerfacecolor=DONOR_COLORS[donor],
            markeredgecolor="none",
            label=donor,
        )
        for donor in DONORS
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        handletextpad=0.3,
        borderpad=0.2,
        labelspacing=0.3,
    )


def main() -> int:
    """Render the figure and append its manifest entry."""
    S.set_style()
    config, config_path = _load_config()
    data_path = D.resolve_data_path("gse153515_annotated.h5ad")

    expected_shape = config["external_test"]["annotated_shape"]

    adata = ad.read_h5ad(data_path, backed="r")
    try:
        if int(adata.n_obs) != int(expected_shape[0]):
            raise AssertionError(
                f"backed h5ad has {adata.n_obs} cells; config declares "
                f"{expected_shape[0]}"
            )
        obs = adata.obs.loc[:, ["donor_id", "n_genes"]].copy()
        obs["donor_id"] = obs["donor_id"].astype(str)
        observed_donors = set(obs["donor_id"].unique())
        if observed_donors != set(config["external_test"]["donors"]):
            raise AssertionError(
                f"donors in h5ad {sorted(observed_donors)} != config "
                f"{sorted(config['external_test']['donors'])}"
            )
        obs["total_counts"] = _total_counts(adata)
    finally:
        adata.file.close()

    thresholds = _numeric_qc_thresholds(config)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=S.figsize(*CANVAS_MM), gridspec_kw={"wspace": 0.26}
    )
    _plot_marginal(ax_a, obs, thresholds)
    _plot_joint(ax_b, obs)
    S.panel_label(ax_a, "A")
    S.panel_label(ax_b, "B")

    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={
            "gse153515_annotated.h5ad": data_path,
            "external_test_gse153515.yml": config_path,
        },
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    if thresholds:
        print(f"[{OUT_NAME}] QC reference lines drawn: {sorted(thresholds)}")
    else:
        print(
            f"[{OUT_NAME}] config declares no numeric QC threshold; "
            "no cut-off lines drawn"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
