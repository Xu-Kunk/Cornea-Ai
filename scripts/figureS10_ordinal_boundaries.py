"""Figure S10 — ordinal (CORN) head decision boundaries.

``core_claim``
    The ordinal head learns ordered, monotone decision boundaries: each
    threshold logit shifts monotonically across the differentiation stages and
    per-cell cumulative probabilities are consistent far more often than an
    unordered-head permutation null.

``hero_panel``
    Panel A: the three threshold logits, one facet each, across the four stages.

``stats_shown``
    Panel A: violin densities with the per-stage median. Panel B: box plots with
    median and 1.5 x IQR whiskers, outliers not drawn. Panel C: observed
    proportion of monotone-consistent cells against the permutation null
    (n_draws and seed read from the JSON summary).

``source_data``
    ``artifacts/paper1_corn_boundary/percell_lam0.0_42.npz`` (per-cell threshold
    logits and posteriors) and ``results/paper1_corn_boundary/
    percell_lam0.0_{42,123,456}.json`` (monotone-consistency summary and null).

One figure, one producer: this script is the only writer of
``figures/paper1/figureS10_ordinal_boundaries.{pdf,png}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paper1_figure_data as D  # noqa: E402, N812
import paper1_figure_style as S  # noqa: E402, N812
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

OUT_NAME = "figureS10_ordinal_boundaries"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 96.0)

STAGES = ("LSC", "TAC", "basal", "differentiated")
CLASSES = STAGES
MAIN_SEED = 42
THRESHOLD_LABELS = ("P(y \u2265 1)", "P(y \u2265 2)", "P(y \u2265 3)")

# Geometry and ink only; no data values. Each threshold gets its own
# linestyle+marker so the three facets remain separable without colour.
THRESHOLD_STYLES = (("-", "o"), ("--", "s"), (":", "D"))
BOX_OFFSET = 0.21
# One slot per class: every box is 0.6 of its slot, so box width never implies
# a sample size and adjacent boxes keep a gap for their medians and caps.
BOX_WIDTH = 0.6 * BOX_OFFSET
VIOLIN_WIDTH = 0.62
SEPARATOR_COLOR = "#D9D9D9"
# A posterior spread narrower than this collapses to a line on the axis and
# reads as a tick mark, so those medians are marked explicitly instead.
MIN_BOX_IQR = 0.004
# Panel letters: one shared top slot measured from the trim, so the three
# letters share a baseline instead of drifting with their axes heights. The
# top band is reserved in the layout so a letter never sits on the trim.
LETTER_TOP_MM = 2.0
LETTER_OFFSET_MM = 6.5
AXES_TOP_FRACTION = 0.95
TITLE_PAD_PT = 2.0


def load_data():
    """Read the per-cell archive and the three seed summaries.

    Returns
    -------
    tuple[dict, list[pathlib.Path]]
        Arrays plus parsed summaries, and the input paths.
    """
    npz_path = D.ARTIFACTS / "paper1_corn_boundary" / f"percell_lam0.0_{MAIN_SEED}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"per-cell CORN archive not found: {npz_path.resolve()}")
    archive = D.load_npz(npz_path)

    summaries = {}
    paths = [npz_path]
    for seed in (42, 123, 456):
        path = D.RESULTS / "paper1_corn_boundary" / f"percell_lam0.0_{seed}.json"
        summaries[seed] = D.load_json(path)
        paths.append(path)

    for key in ("logits", "probs", "thr_probs", "layer"):
        if key not in archive.files:
            raise AssertionError(f"{npz_path.name} lacks array {key!r}")
    logits = np.asarray(archive["logits"], dtype=float)
    probs = np.asarray(archive["probs"], dtype=float)
    thr_probs = np.asarray(archive["thr_probs"], dtype=float)
    layer = np.asarray(archive["layer"], dtype=int)

    counts = {stage: int((layer == index).sum()) for index, stage in enumerate(STAGES)}
    empty = [stage for stage, count in counts.items() if count == 0]
    if empty:
        raise AssertionError(f"no cells for stage(s): {empty}")

    summary = summaries[MAIN_SEED]
    if int(summary["model"]["seed"]) != MAIN_SEED:
        raise AssertionError("summary seed does not match the per-cell archive")
    if int(summary["external_test"]["n_cells"]) != int(layer.shape[0]):
        raise AssertionError(
            "summary n_cells does not match the per-cell archive "
            f"({summary['external_test']['n_cells']} vs {layer.shape[0]})"
        )

    data = {
        "logits": logits,
        "probs": probs,
        "thr_probs": thr_probs,
        "layer": layer,
        "counts": counts,
        "summary": summary,
        "summaries": summaries,
    }
    return data, paths


def check_manuscript(data) -> None:
    """Assert plotted proportions against the archived summary.

    Raises
    ------
    AssertionError
        If a plotted proportion differs from the archived JSON.
    """
    thr_probs = data["thr_probs"]
    consistent = (thr_probs[:, 0] >= thr_probs[:, 1]) & (thr_probs[:, 1] >= thr_probs[:, 2])
    S.check_value(
        "monotone consistency (overall)",
        float(consistent.mean()),
        float(data["summary"]["monotone_consistency"]["overall"]),
        tol=1e-9,
    )
    for index, stage in enumerate(STAGES):
        S.check_value(
            f"monotone consistency ({stage})",
            float(consistent[data["layer"] == index].mean()),
            float(
                data["summary"]["monotone_consistency"]["per_stage"][stage]["monotone_consistent"]
            ),
            tol=1e-9,
        )

    # The other two seeds are read (and hashed into the manifest) as
    # cross-checks on the plotted seed-42 summary.
    for seed, summary in data["summaries"].items():
        if int(summary["external_test"]["n_cells"]) != int(data["layer"].shape[0]):
            raise AssertionError(f"seed {seed}: n_cells differs from the archive")
        if "permutation_null" not in summary or "monotone_consistency" not in summary:
            raise AssertionError(f"seed {seed}: summary lacks the null or consistency")

    for array_name in ("logits", "probs", "thr_probs"):
        if not np.all(np.isfinite(data[array_name])):
            raise AssertionError(f"{array_name}: non-finite values")
    if np.any(data["probs"] < -1e-9) or np.any(data["probs"] > 1.0 + 1e-9):
        raise AssertionError("posteriors outside [0, 1]")


def plot_panel_a(axes, data) -> None:
    """Threshold logits per stage: one violin facet per threshold."""
    logits = data["logits"]
    layer = data["layer"]
    positions = np.arange(len(STAGES), dtype=float)
    green = S.ARM_COLORS["ordinal"]
    bodies_low = []
    bodies_high = []

    for facet, (ax, label, (linestyle, marker)) in enumerate(
        zip(axes, THRESHOLD_LABELS, THRESHOLD_STYLES)
    ):
        for index in range(len(STAGES)):
            values = logits[layer == index, facet]
            parts = ax.violinplot(
                [values],
                positions=[positions[index]],
                widths=VIOLIN_WIDTH,
                showextrema=False,
                showmedians=False,
                bw_method=0.4,
            )
            body = parts["bodies"][0]
            body.set_facecolor(green)
            body.set_alpha(0.30)
            body.set_edgecolor(green)
            body.set_linewidth(0.5)
            vertices = body.get_paths()[0].vertices[:, 1]
            bodies_low.append(float(vertices.min()))
            bodies_high.append(float(vertices.max()))
            median = float(np.median(values))
            ax.plot(
                [positions[index] - VIOLIN_WIDTH * 0.34, positions[index] + VIOLIN_WIDTH * 0.34],
                [median, median],
                color="#1A1A1A",
                linestyle=linestyle,
                lw=0.9,
                zorder=5,
            )
            ax.plot(
                positions[index],
                median,
                linestyle="none",
                marker=marker,
                markersize=2.8,
                markerfacecolor="#1A1A1A",
                markeredgecolor="white",
                markeredgewidth=0.3,
                zorder=6,
            )
        ax.axhline(0.0, color=S.RULE_COLOR, linestyle=":", lw=S.RULE_LW, zorder=1)
        ax.set_xticks(positions)
        ax.set_title(label, loc="left", fontsize=S.TYPE["title"], pad=TITLE_PAD_PT)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if facet == len(axes) - 1:
            ax.set_xticklabels(STAGES, rotation=30, ha="right")
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="x", length=2.0)

    low = min(bodies_low)
    high = max(bodies_high)
    pad = (high - low) * 0.06
    for ax in axes:
        ax.set_ylim(low - pad, high + pad)
    axes[1].set_ylabel("Threshold logit")


def plot_panel_b(ax, data) -> None:
    """Class posteriors per stage: one box per stage-by-class cell."""
    probs = data["probs"]
    layer = data["layer"]
    stage_positions = np.arange(len(STAGES), dtype=float)
    box_positions = [
        stage_positions[stage] + (class_index - 1.5) * BOX_OFFSET
        for stage in range(len(STAGES))
        for class_index in range(len(CLASSES))
    ]

    for class_index, cls in enumerate(CLASSES):
        values = [probs[layer == stage, class_index] for stage in range(len(STAGES))]
        positions = [
            stage_positions[stage] + (class_index - 1.5) * BOX_OFFSET
            for stage in range(len(STAGES))
        ]
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=BOX_WIDTH,
            whis=1.5,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#1A1A1A", "lw": 0.8},
            whiskerprops={"color": "#4D4D4D", "lw": S.LW_REFERENCE},
            capprops={"color": "#4D4D4D", "lw": S.LW_REFERENCE},
            boxprops={"lw": 0.5, "edgecolor": "#333333"},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(S.CELLTYPE_COLORS[cls])
            patch.set_alpha(0.9)
        # A near-zero posterior has an IQR of a few thousandths, so its box is
        # a 1-2 px line on the axis and reads as a tick mark. Mark that median
        # explicitly rather than leaving an invisible box.
        for stage_values, position in zip(values, positions):
            low, median, high = np.percentile(stage_values, [25, 50, 75])
            if float(high - low) < MIN_BOX_IQR:
                ax.plot(
                    position,
                    float(median),
                    linestyle="none",
                    marker="o",
                    markersize=S.MARKER_PT,
                    markerfacecolor=S.CELLTYPE_COLORS[cls],
                    markeredgecolor="#1A1A1A",
                    markeredgewidth=0.4,
                    zorder=6,
                )

    # A tick at every box centre (minor) plus a labelled tick per stage.
    ax.set_xticks(box_positions, minor=True)
    ax.tick_params(axis="x", which="minor", length=2.2, width=0.6)
    ax.set_xticks(stage_positions)
    ax.set_xticklabels(STAGES, rotation=30, ha="right")
    for boundary in stage_positions[:-1] + 0.5:
        ax.axvline(boundary, color=SEPARATOR_COLOR, lw=S.RULE_LW, zorder=0)

    ax.set_xlim(-0.62, len(STAGES) - 0.38)
    low = float(probs.min()) - 0.04
    high = min(1.04, float(probs.max()) + 0.04)
    ax.set_ylim(low, high)
    ax.set_ylabel("P(y = k)")
    ax.set_title("Class posteriors by stage", fontsize=S.TYPE["title"], pad=TITLE_PAD_PT)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
    ax.legend(
        handles=[
            Patch(facecolor=S.CELLTYPE_COLORS[cls], edgecolor="#333333", lw=0.5) for cls in CLASSES
        ],
        labels=list(CLASSES),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.245),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
    )


def plot_panel_c(ax, data) -> None:
    """Monotone-consistency proportion against the permutation null."""
    thr_probs = data["thr_probs"]
    layer = data["layer"]
    consistent = (thr_probs[:, 0] >= thr_probs[:, 1]) & (thr_probs[:, 1] >= thr_probs[:, 2])
    observed = [float(consistent[layer == index].mean()) for index in range(len(STAGES))]
    observed.append(float(consistent.mean()))

    null = float(data["summary"]["permutation_null"]["expected_consistency"])
    x = np.arange(len(observed), dtype=float)
    width = 0.36
    green = S.ARM_COLORS["ordinal"]

    ax.bar(
        x - width / 2,
        observed,
        width,
        color=green,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="observed",
    )
    # The null is a reference, not a second claim: structural ink, light hatch.
    ax.bar(
        x + width / 2,
        [null] * len(x),
        width,
        color=S.GRID_COLOR,
        hatch="////",
        edgecolor=S.RULE_COLOR,
        linewidth=S.RULE_LW,
        zorder=3,
        label="permutation null",
    )
    ax.axhline(null, color=S.RULE_COLOR, linestyle="--", lw=S.RULE_LW, zorder=2)
    for xi, value in zip(x, observed):
        ax.text(
            xi - width / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=S.TYPE["tick"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(STAGES) + ["Overall"], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0.0, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Fraction P(y\u22651) \u2265 P(y\u22652) \u2265 P(y\u22653)")
    ax.set_title("Monotone consistency vs null", fontsize=S.TYPE["title"], pad=TITLE_PAD_PT)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
    ax.legend(
        loc="upper left",
        frameon=False,
        borderpad=0.4,
        handlelength=1.6,
    )


def assert_no_clipping(fig) -> None:
    """Raise if any text artist leaves the canvas (a clipped label).

    Tick labels are excluded: the axis locator may generate ticks outside the
    view limits, which matplotlib never draws.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tick_labels = set()
    for axis in fig.axes:
        for ax_axis in (axis.xaxis, axis.yaxis):
            for tick in ax_axis.get_major_ticks() + ax_axis.get_minor_ticks():
                tick_labels.add(id(tick.label1))
                tick_labels.add(id(tick.label2))
    for artist in fig.findobj(matplotlib.text.Text):
        if id(artist) in tick_labels:
            continue
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            raise AssertionError(f"text {artist.get_text()!r} is clipped: {box} vs canvas {canvas}")


def _place_panel_letters(fig, panels) -> None:
    """Put every panel letter in one top slot, so all three share a baseline.

    ``S.panel_label`` offsets from each axes' own height, which drops the letter
    on a short facet below the letters on tall panels. Here every letter is
    anchored to the canvas at ``LETTER_TOP_MM`` and only its x follows the axes,
    so the three letters sit on one baseline and one margin.
    """
    fig.canvas.draw()  # constrained layout must be resolved before reading positions
    width_mm = fig.get_size_inches()[0] * S.MM_PER_INCH
    for ax, letter in panels:
        x_mm = ax.get_position().x0 * width_mm - LETTER_OFFSET_MM
        S.panel_letter_mm(fig, letter, x_mm, LETTER_TOP_MM)


def build(data):
    """Assemble the figure: A as three stacked threshold facets, B and C full height."""
    fig = plt.figure(figsize=S.figsize(*CANVAS_MM))
    # Reserve the top band that holds the panel letters.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, AXES_TOP_FRACTION))
    grid = fig.add_gridspec(3, 3, width_ratios=[0.98, 1.10, 0.92], height_ratios=[1, 1, 1])
    facet_axes = [fig.add_subplot(grid[row, 0], sharex=None, sharey=None) for row in range(3)]
    for ax in facet_axes[1:]:
        ax.sharey(facet_axes[0])
    ax_b = fig.add_subplot(grid[:, 1])
    ax_c = fig.add_subplot(grid[:, 2])

    plot_panel_a(facet_axes, data)
    plot_panel_b(ax_b, data)
    plot_panel_c(ax_c, data)

    _place_panel_letters(
        fig,
        ((facet_axes[0], "A"), (ax_b, "B"), (ax_c, "C")),
    )
    return fig


def main() -> int:
    """Render the figure, verify it, and write the manifest entry."""
    S.set_style()
    data, paths = load_data()
    check_manuscript(data)
    fig = build(data)
    assert_no_clipping(fig)
    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={
            "percell_npz": paths[0],
            "summary_seed42": paths[1],
            "summary_seed123": paths[2],
            "summary_seed456": paths[3],
        },
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print(f"  n cells = {data['layer'].shape[0]}, stage counts = {data['counts']}")
    print(f"  null = {float(data['summary']['permutation_null']['expected_consistency']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
