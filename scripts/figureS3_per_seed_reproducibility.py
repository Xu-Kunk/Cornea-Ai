"""Figure S3 — per-seed reproducibility of the four continuity/endpoint metrics.

``core_claim``
    The CE-naive arm reproduces its endpoint and continuity metrics across
    seeds 42/123/456: every metric is stable within the arm, and the
    differentiation-gradient rho stays near zero at every seed.

``hero_panel``
    Panel B (differentiation-gradient rho), where the CE-naive seeds sit near
    zero, against the definitional rho = 0 rule.

``stats_shown``
    Every seed is drawn individually; the thick grey bar is the arithmetic
    mean of the three seeds. No interval, error bar, test or correction is
    drawn. The per-seed values are the plotted statistic.

``source_data``
    ``figures/paper1/table_t7_seed_reproducibility.csv`` (CE-naive rows:
    ``macro_f1``, ``LAA``, ``marker_gradient_rho``, ``ECE``) and
    ``artifacts/paper1_tier_b_eval/results_0.0_{42,123,456}.json`` (the
    archive the CE-naive table rows are cross-checked against).

One figure, one producer: this script is the only writer of
``figures/paper1/figureS3_per_seed_reproducibility.{pdf,png}``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paper1_figure_data as D  # noqa: E402, N812
import paper1_figure_style as S  # noqa: E402, N812
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

OUT_NAME = "figureS3_per_seed_reproducibility"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 128.0)

# Style constants only (geometry and ink); every plotted number is read below.
SEED_MARKERS = ("o", "s", "^")
SEED_SPREAD = 0.16
# The mean bar spans the outer seed centres so it does not stop short of the
# markers it summarises.
MEAN_HALF_WIDTH = SEED_SPREAD
MARKER_SIZE = 2.88  # 12 px at 300 dpi
MEAN_LW = 1.6
MEAN_COLOR = "#4D4D4D"
PAD_Y_FRACTION = 0.14
NULL_RHO = 0.0  # definitional: Spearman rho of two independent rankings

# Metric key -> (column in table_t7, JSON secondary-endpoint field, axis label).
METRICS = (
    ("macro_f1", None, "Donor-level macro-F1"),
    (
        "marker_gradient_rho",
        "marker_gradient_rank_correlation",
        "Differentiation-gradient \u03c1 (Spearman)",
    ),
    ("LAA", "layer_adjacency_accuracy", "Layer adjacency accuracy"),
    ("ECE", "expected_calibration_error", "Expected calibration error"),
)

ARMS_SHOWN = ("ce_naive",)
SEEDS = (42, 123, 456)


def _tick_decimals(ticks: np.ndarray) -> int:
    """Decimals that render one tick step exactly, never fewer than one.

    The four panels span three orders of magnitude (a 0.001 LAA step next to a
    0.1 rho step); one rule keeps the displayed precision tied to the step
    instead of to the panel, so a reader can compare the grids.
    """
    steps = np.diff(np.sort(np.asarray(ticks, dtype=float)))
    steps = steps[steps > 0]
    if steps.size == 0:
        return 1
    step = float(np.min(steps))
    return max(1, int(math.ceil(-math.log10(step) - 1e-9)))


def _format_tick(value: float, decimals: int) -> str:
    """Fixed-precision label with a leading zero and a U+2212 minus sign."""
    text = f"{value:.{decimals}f}"
    if text.startswith("-"):
        text = "\u2212" + text[1:]
    return text


def _apply_tick_precision(ax) -> None:
    """Rewrite the y tick labels at the step's own precision."""
    ticks = np.asarray(ax.get_yticks(), dtype=float)
    decimals = _tick_decimals(ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos, d=decimals: _format_tick(value, d))
    )


def load_values():
    """Read both arms' per-seed metrics and record their provenance.

    Returns
    -------
    dict
        ``{"values": {arm: {metric: dict[seed -> float]}}, "sources": {...}}``.

    Raises
    ------
    AssertionError
        If a seed or metric is missing from either source.
    """
    table_path = D.fig_table("table_t7_seed_reproducibility")
    table = D.load_csv(table_path)
    if len(table["arm"].unique()) != 1:
        raise AssertionError("table_t7 must hold exactly one arm")
    if sorted(int(seed) for seed in table["seed"]) != list(SEEDS):
        raise AssertionError(f"table_t7 seeds != {list(SEEDS)}")

    values: dict[str, dict[str, dict[int, float]]] = {arm: {} for arm in ARMS_SHOWN}
    sources: dict[str, tuple[Path, str]] = {}

    for metric, _field, _label in METRICS:
        values["ce_naive"][metric] = {
            int(row["seed"]): float(row[metric]) for _, row in table.iterrows()
        }
    sources["ce_naive"] = (
        table_path,
        "table_t7_seed_reproducibility.csv:columns "
        + ", ".join(metric for metric, _f, _l in METRICS),
    )

    json_paths = {}
    # The CE-naive rows of table_t7 must reproduce the tier-B archive they were
    # exported from; a drift between the two is a data defect, not a replot.
    for seed in SEEDS:
        path = D.tier_b(f"0.0_{seed}")
        document = D.load_json(path)
        S.check_value(
            f"table_t7 vs tier-B CE-naive macro-F1 (seed {seed})",
            values["ce_naive"]["macro_f1"][seed],
            float(document["primary_endpoint"]["macro_f1"]),
        )
        S.check_value(
            f"table_t7 vs tier-B CE-naive gradient rho (seed {seed})",
            values["ce_naive"]["marker_gradient_rho"][seed],
            float(document["secondary_endpoints"]["marker_gradient_rank_correlation"]),
        )
        json_paths[f"0.0_{seed}"] = path

    for arm in ARMS_SHOWN:
        for metric, _field, _label in METRICS:
            seeds = values[arm].get(metric, {})
            if sorted(seeds) != list(SEEDS):
                raise AssertionError(f"{arm}/{metric}: missing seed(s)")
            if not all(np.isfinite(list(seeds.values()))):
                raise AssertionError(f"{arm}/{metric}: non-finite value")
    return {"values": values, "sources": sources, "json_paths": json_paths}


def check_manuscript(values):
    """Assert the CE-naive gradient values the manuscript states.

    Parameters
    ----------
    values : dict
        Output of :func:`load_values`.

    Raises
    ------
    AssertionError
        If a plotted seed value differs from ``MANUSCRIPT_VALUES``.
    """
    plotted = values["values"]["ce_naive"]["marker_gradient_rho"]
    for seed, expected in zip(SEEDS, D.MANUSCRIPT_VALUES["ce_naive"]["rho_seeds"]):
        S.check_value(f"CE-naive seed-{seed} gradient rho", plotted[seed], expected)


def plot(values):
    """Draw the 2x2 per-metric seed panels.

    Parameters
    ----------
    values : dict
        Output of :func:`load_values`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure ready for :func:`paper1_figure_style.save_figure`.
    """
    fig, axes = plt.subplots(2, 2, figsize=S.figsize(*CANVAS_MM))
    for panel_index, (metric, _field, label) in enumerate(METRICS):
        ax = axes.flat[panel_index]
        plotted: list[float] = []
        for arm_index, arm in enumerate(ARMS_SHOWN):
            colour = S.ARM_COLORS[arm]
            seeds = values["values"][arm][metric]
            for seed_index, seed in enumerate(SEEDS):
                offset = (seed_index - (len(SEEDS) - 1) / 2) * SEED_SPREAD
                ax.plot(
                    arm_index + offset,
                    seeds[seed],
                    linestyle="none",
                    marker=SEED_MARKERS[seed_index],
                    markersize=MARKER_SIZE,
                    markerfacecolor=colour,
                    markeredgecolor=colour,
                    markeredgewidth=0.4,
                    zorder=3,
                )
                plotted.append(seeds[seed])
            mean_value = float(np.mean(list(seeds.values())))
            ax.plot(
                [arm_index - MEAN_HALF_WIDTH, arm_index + MEAN_HALF_WIDTH],
                [mean_value, mean_value],
                color=MEAN_COLOR,
                linewidth=MEAN_LW,
                solid_capstyle="butt",
                zorder=4,
            )
            plotted.append(mean_value)

        if metric == "marker_gradient_rho":
            ax.axhline(
                NULL_RHO,
                color=S.RULE_COLOR,
                linestyle="--",
                linewidth=S.RULE_LW,
                zorder=1,
            )

        low, high = float(np.min(plotted)), float(np.max(plotted))
        pad = (high - low) * PAD_Y_FRACTION
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(-0.5, len(ARMS_SHOWN) - 0.5)
        ax.set_xticks(range(len(ARMS_SHOWN)))
        ax.set_xticklabels([S.ARMS[arm] for arm in ARMS_SHOWN])
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
        _apply_tick_precision(ax)
        if metric == "marker_gradient_rho":
            # The zero rule and the 0.0 gridline would draw the same ink twice;
            # the rule alone marks zero.
            for tick, location in zip(ax.yaxis.get_major_ticks(), ax.get_yticks()):
                if abs(float(location) - NULL_RHO) < 1e-12:
                    tick.gridline.set_visible(False)
        S.panel_label(ax, "ABCD"[panel_index])

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor="#555555",
            markeredgecolor="#555555",
        )
        for marker in SEED_MARKERS
    ] + [Line2D([0], [0], color=MEAN_COLOR, linewidth=MEAN_LW)]
    labels = [f"seed {seed}" for seed in SEEDS] + ["mean"]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        handlelength=1.6,
    )
    return fig


def _visible_texts(fig):
    """Yield only the text artists that the renderer actually draws."""
    for ax in fig.axes:
        yield ax.title
        yield ax.xaxis.label
        yield ax.yaxis.label
        x_lo, x_hi = ax.get_xlim()
        for tick in ax.get_xticklabels():
            if x_lo - 1e-12 <= tick.get_position()[0] <= x_hi + 1e-12:
                yield tick
        y_lo, y_hi = ax.get_ylim()
        for tick in ax.get_yticklabels():
            if y_lo - 1e-12 <= tick.get_position()[1] <= y_hi + 1e-12:
                yield tick
    yield from fig.texts
    for legend in fig.legends:
        yield from legend.get_texts()


def assert_no_clipping(fig):
    """Raise if any drawn text artist leaves the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px, height_px = fig.get_size_inches() * fig.dpi
    canvas = matplotlib.transforms.Bbox.from_bounds(0.0, 0.0, width_px, height_px)
    for artist in _visible_texts(fig):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            raise AssertionError(f"text {artist.get_text()!r} is clipped: {box} vs {canvas}")


def main() -> int:
    """Render the figure and write the manifest entry."""
    S.set_style()
    values = load_values()
    check_manuscript(values)
    fig = plot(values)
    assert_no_clipping(fig)

    inputs = {"table_t7_seed_reproducibility": D.fig_table("table_t7_seed_reproducibility")}
    for name, path in values["json_paths"].items():
        inputs[f"results_{name}"] = path
    pdf_path, png_path = S.save_figure(fig, OUT_NAME, outdir=str(S.FIGURE_DIR), inputs=inputs)
    plt.close(fig)

    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print("plotted value -> file:field")
    for arm in ARMS_SHOWN:
        path, field = values["sources"][arm]
        for metric, _field, _label in METRICS:
            seeds = values["values"][arm][metric]
            shown = ", ".join(f"{seed}={seeds[seed]:.6f}" for seed in SEEDS)
            print(f"  {S.ARMS[arm]} / {metric}: {shown}  <- {path}:{field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
