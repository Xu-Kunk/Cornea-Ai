"""Figure 2 — layerwise differentiation-gradient Spearman rho.

``core_claim``
    A frozen Geneformer encoder keeps a stable differentiation gradient across
    all twelve transformer layers, whereas fine-tuning without a continuity
    constraint collapses that gradient in the late layers (the adapted mean is
    near zero at layer 12).

``hero_panel``
    The single axes: the frozen curve versus the CE-naive seed cloud.

``stats_shown``
    Each CE-naive seed is drawn individually; the thick line is the seed mean
    and the band is +/- 1 population SD across the three seeds (ddof = 0,
    n = 3). No hypothesis test is shown.

``source_data``
    ``figures/paper1/figure2_layerwise_rho.csv`` (4 models x 12 layers) and
    ``artifacts/paper1_layerwise_rho/summary.json`` (sanity anchors).

One figure, one producer: this script is the only writer of
``figures/paper1/figure2_layerwise_rho.pdf`` and ``.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paper1_figure_style as S  # noqa: E402, N812
import paper1_figure_data as D  # noqa: E402, N812
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT_NAME = "figure2_layerwise_rho"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 100.0)

# Spearman rho of two independent rankings, i.e. the null reference. This is a
# definitional anchor, not a measured value.
NULL_RHO = 0.0

# Style constants (geometry/ink only; no data values). Weights and marker
# sizes come from the locked style scale.
LINE_FROZEN = S.LW_REFERENCE   # the frozen reference fades relative to the claim
LINE_MEAN = S.LW_MAIN          # the CE-naive mean is the claim
LINE_SEED = S.LW_SEED
MARKER_SEED = S.MARKER_PT
BAND_ALPHA = 0.12
BAND_ZORDER = 0.0              # band sits below the grid (axisbelow -> grid 0.5)
PAD_Y_FRACTION = 0.05
PAD_X_FRACTION = 0.05

# Seeds are separated by linestyle *and* marker, so colour is never the only
# channel. Position in the ordering read from the data selects the style.
SEED_STYLES = (("dotted", "o"), ("dashed", "s"), ("dashdot", "^"))


def load_layerwise():
    """Return the layerwise table and the artifact summary.

    Returns
    -------
    tuple[pandas.DataFrame, dict, pathlib.Path, pathlib.Path]
        Table, parsed summary document, and both input paths.

    Raises
    ------
    AssertionError
        If the table does not hold four models over twelve contiguous layers
        or if the artifact disagrees with a table cell.
    """
    table_path = D.fig_table(OUT_NAME)
    summary_path = D.artifact_json("paper1_layerwise_rho", "summary")
    table = D.load_csv(table_path)
    summary = D.load_json(summary_path)

    layers = np.sort(table["layer"].unique())
    if len(layers) != 12:
        raise AssertionError(f"expected 12 layers, found {len(layers)}")
    if list(layers) != list(range(int(layers[0]), int(layers[0]) + len(layers))):
        raise AssertionError(f"layers are not contiguous: {layers.tolist()}")

    frozen_rows = table[table["group"] == "frozen"]
    naive_rows = table[table["group"] == "naive"]
    if frozen_rows["model"].nunique() != 1:
        raise AssertionError("expected exactly one frozen model")
    if naive_rows["seed"].nunique() != len(SEED_STYLES):
        raise AssertionError(
            f"expected {len(SEED_STYLES)} CE-naive seeds, "
            f"found {naive_rows['seed'].nunique()}"
        )
    if len(table) != (1 + naive_rows["seed"].nunique()) * len(layers):
        raise AssertionError("unexpected number of model-by-layer rows")

    # Artifact cross-check: the summary stores embedding_output + 12 layers,
    # so table layer n maps to summary index n.
    artifact_models = summary["models"]
    frozen_model = str(frozen_rows["model"].iloc[0])
    for _, row in frozen_rows.iterrows():
        stored = float(artifact_models[frozen_model][int(row["layer"])])
        if not np.isclose(stored, float(row["rho"]), rtol=0.0, atol=1e-12):
            raise AssertionError(f"summary/table mismatch at layer {row['layer']}")
    for _, row in naive_rows.iterrows():
        key = f"lam0.0_seed{int(row['seed'])}"
        stored = float(artifact_models[key][int(row["layer"])])
        if not np.isclose(stored, float(row["rho"]), rtol=0.0, atol=1e-12):
            raise AssertionError(f"summary/table mismatch at {key} layer {row['layer']}")

    return table, summary, table_path, summary_path


def check_manuscript_values(table, summary):
    """Assert the layer-12 values against the manuscript and the artifact.

    Parameters
    ----------
    table : pandas.DataFrame
        Layerwise table.
    summary : dict
        Parsed artifact summary.

    Raises
    ------
    AssertionError
        If a plotted layer-12 value differs from the manuscript statement.
    """
    frozen_rows = table[table["group"] == "frozen"].sort_values("layer")
    frozen_final = float(frozen_rows["rho"].iloc[-1])
    S.check_value("frozen layer-12 rho", frozen_final, D.MANUSCRIPT_VALUES["frozen"]["rho"])

    naive_rows = table[table["group"] == "naive"]
    final = (
        naive_rows[naive_rows["layer"] == naive_rows["layer"].max()]
        .sort_values("seed")["rho"]
        .to_numpy(dtype=float)
    )
    for value, expected in zip(final, D.MANUSCRIPT_VALUES["ce_naive"]["rho_seeds"]):
        S.check_value("CE-naive layer-12 rho", value, expected)

    # The artifact's own sanity gate must agree with what is drawn.
    frozen_model = str(table[table["group"] == "frozen"]["model"].iloc[0])
    gate = summary["sanity_gate"]
    if not bool(gate[frozen_model]["passed"]):
        raise AssertionError("frozen sanity gate did not pass")
    S.check_value(
        "frozen sanity gate layer-12 rho",
        float(gate[frozen_model]["layer12_rho"]),
        frozen_final,
        tol=1e-12,
    )


def plot_layerwise(table):
    """Draw the single layerwise axes.

    Parameters
    ----------
    table : pandas.DataFrame
        Layerwise table with one frozen model and three CE-naive seeds.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes, ready to save.
    """
    frozen_rows = table[table["group"] == "frozen"].sort_values("layer")
    naive_rows = table[table["group"] == "naive"]
    pivot = naive_rows.pivot(index="layer", columns="seed", values="rho")
    layers = np.asarray(pivot.index, dtype=float)
    seeds = sorted(int(seed) for seed in pivot.columns)
    values = pivot[seeds].to_numpy(dtype=float)
    mean_vals = values.mean(axis=1)
    sd_vals = values.std(axis=1, ddof=0)

    color = S.ARM_COLORS["ce_naive"]
    frozen_color = S.ARM_COLORS["frozen"]

    fig, ax = plt.subplots(figsize=S.figsize(*CANVAS_MM))

    ax.fill_between(
        layers,
        mean_vals - sd_vals,
        mean_vals + sd_vals,
        color=color,
        alpha=BAND_ALPHA,
        linewidth=0.0,
        zorder=BAND_ZORDER,
    )
    for style_index, seed in enumerate(seeds):
        linestyle, marker = SEED_STYLES[style_index]
        ax.plot(
            layers,
            pivot[seed].to_numpy(dtype=float),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=MARKER_SEED,
            markeredgewidth=0.0,
            linewidth=LINE_SEED,
            zorder=2,
        )
    ax.plot(
        layers,
        mean_vals,
        color=color,
        linestyle="-",
        linewidth=LINE_MEAN,
        zorder=3,
    )
    ax.plot(
        layers,
        frozen_rows["rho"].to_numpy(dtype=float),
        color=frozen_color,
        linestyle="-",
        linewidth=LINE_FROZEN,
        marker="D",
        markersize=MARKER_SEED,
        markeredgewidth=0.0,
        zorder=4,
    )
    ax.axhline(
        NULL_RHO,
        color=S.RULE_COLOR,
        linestyle="--",
        linewidth=S.LW_REFERENCE,
        zorder=1,
    )

    y_low = min(float(np.min(mean_vals - sd_vals)), float(frozen_rows["rho"].min()))
    y_high = max(float(np.max(mean_vals + sd_vals)), float(frozen_rows["rho"].max()))
    pad_y = (y_high - y_low) * PAD_Y_FRACTION
    ax.set_ylim(y_low - pad_y, y_high + pad_y)
    pad_x = (layers.max() - layers.min()) * PAD_X_FRACTION
    ax.set_xlim(layers.min() - pad_x, layers.max() + pad_x)

    ax.set_xticks(layers)
    ax.set_xticklabels([f"{int(layer)}" for layer in layers])
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Differentiation-gradient Spearman \u03c1")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)

    handles = [
        Line2D(
            [0],
            [0],
            color=frozen_color,
            linestyle="-",
            linewidth=LINE_FROZEN,
            marker="D",
            markersize=MARKER_SEED,
        ),
        Line2D([0], [0], color=color, linestyle="-", linewidth=LINE_MEAN),
        Patch(facecolor=color, alpha=BAND_ALPHA, edgecolor="none"),
    ]
    # One line only: the seed list and the n/ddof bookkeeping live in the
    # caption, so the band is the only statistics entry in the legend.
    labels = [
        S.ARMS["frozen"],
        "CE-naive mean",
        "mean \u00b1 1 SD across seeds",
    ]
    for style_index, seed in enumerate(seeds):
        linestyle, marker = SEED_STYLES[style_index]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=MARKER_SEED,
                markeredgewidth=0.0,
                linewidth=LINE_SEED,
            )
        )
        labels.append(f"seed {seed}")

    fig.legend(
        handles,
        labels,
        loc="outside right center",
        frameon=False,
        handlelength=2.8,
        labelspacing=0.55,
        borderaxespad=0.6,
    )
    return fig, ax


def add_panel_letter(fig, ax) -> None:
    """Place the 10 pt bold ``A`` outside the top-left with a >= 2 mm margin."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lefts = [ax.yaxis.label.get_window_extent(renderer).x0]
    lefts += [
        label.get_window_extent(renderer).x0
        for label in ax.get_yticklabels()
        if label.get_text()
    ]
    x_px = max(min(lefts), S.LETTER_TOP_PX / 300.0 * fig.dpi)
    x = float(fig.transFigure.inverted().transform((x_px, 0.0))[0])
    height_in = float(fig.get_size_inches()[1])
    y = 1.0 - (S.LETTER_TOP_PX / 300.0) / height_in
    fig.text(
        x,
        y,
        "A",
        fontsize=S.TYPE["letter"],
        fontweight="bold",
        ha="left",
        va="top",
        color="black",
    )


def main() -> int:
    """Render the figure and write the manifest entry.

    Returns
    -------
    int
        Process exit code.
    """
    S.set_style()
    table, summary, table_path, summary_path = load_layerwise()
    check_manuscript_values(table, summary)

    fig, ax = plot_layerwise(table)
    add_panel_letter(fig, ax)
    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={
            "figure_table": table_path,
            "layerwise_summary": summary_path,
        },
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
