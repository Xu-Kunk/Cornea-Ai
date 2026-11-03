"""Figure S7 — layerwise differentiation-gradient rho for every seed.

``core_claim``
    Seed-to-seed variation does not rescue the late-layer collapse: all three
    CE-naive seeds fall to a near-zero layer-12 gradient, while the frozen
    encoder holds the highest late-layer values.

``hero_panel``
    The single axes: twelve transformer layers, frozen reference against the
    three adapted seed trajectories.

``stats_shown``
    Each seed is drawn individually (linestyle and marker encode seed identity,
    colour encodes arm). The thick line is the arithmetic mean over the three
    seeds of that arm. No band, interval, test or correction is drawn.

``source_data``
    ``artifacts/paper1_layerwise_rho/layerwise_rho_{frozen_gf,
    naive_s42,naive_s123,naive_s456}.json``,
    fields ``layer_labels`` and ``marker_gradient_rho``.

One figure, one producer: this script is the only writer of
``figures/paper1/figureS7_layerwise_rho_per_seed.{pdf,png}``.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

OUT_NAME = "figureS7_layerwise_rho_per_seed"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 96.0)

# Style constants only (geometry and ink); every plotted number is read below.
SEED_STYLES = ((":", "o"), ("--", "s"), ("-.", "^"))
LINE_SEED = S.LW_SEED
LINE_MEAN = 1.7
LINE_FROZEN = 1.9
MARKER_SIZE = 2.8
NULL_RHO = 0.0  # definitional: Spearman rho of two independent rankings
# Tight range around the data (-0.042 to 0.858): the old 6 % pad spent 85 px of
# labelled negative axis on no ink.
Y_LIMITS = (-0.05, 0.90)
LAYER_PREFIX = "transformer_layer_"
FROZEN_MODEL = "frozen_gf"

FAMILIES = {
    "naive": "ce_naive",
}


def load_values():
    """Read every layerwise seed file and record its provenance.

    Returns
    -------
    dict
        ``{"families": {arm: {seed: np.ndarray}}, "frozen": np.ndarray,
        "layers": np.ndarray, "sources": {...}}``.

    Raises
    ------
    AssertionError
        If the directory does not hold exactly one frozen file and three seeds
        per adapted family, or if the layer grid is not contiguous.
    """
    directory = D.ARTIFACTS / "paper1_layerwise_rho"
    paths = sorted(directory.glob("layerwise_rho_*.json"))
    if not paths:
        raise FileNotFoundError(f"no layerwise rho files under {directory.resolve()}")

    families: dict[str, dict[int, np.ndarray]] = {
        "ce_naive": {},
    }
    frozen = None
    layers: np.ndarray | None = None
    sources: dict[str, tuple[Path, str]] = {}

    for path in paths:
        document = D.load_json(path)
        if document["model"] == FROZEN_MODEL:
            arm = "frozen"
            seed = None
        elif document["arm"] == "caft":
            # Retired arm: its layerwise files stay on disk but are no longer
            # part of this figure (the series was removed, not relabelled).
            continue
        else:
            try:
                arm = FAMILIES[document["arm"]]
            except KeyError as exc:
                raise AssertionError(f"{path.name}: unknown arm {document['arm']!r}") from exc
            seed = int(document["seed"])

        labels = document["layer_labels"]
        index = [i for i, label in enumerate(labels) if str(label).startswith(LAYER_PREFIX)]
        if len(index) != 12:
            raise AssertionError(f"{path.name}: expected 12 transformer layers, found {len(index)}")
        if layers is None:
            layers = np.arange(1, len(index) + 1)
        elif len(index) != len(layers):
            raise AssertionError(f"{path.name}: layer count differs from the first file")

        values = document["marker_gradient_rho"]
        if len(values) <= max(index):
            raise AssertionError(f"{path.name}: rho array shorter than the layer index")
        rho = np.asarray([values[i] for i in index], dtype=float)
        if not np.all(np.isfinite(rho)):
            raise AssertionError(f"{path.name}: non-finite layerwise rho")

        if arm == "frozen":
            if frozen is not None:
                raise AssertionError("more than one frozen layerwise file")
            frozen = rho
        else:
            if seed in families[arm]:
                raise AssertionError(f"duplicate seed {seed} for {arm}")
            families[arm][seed] = rho
        sources["frozen" if seed is None else f"{arm}_{seed}"] = (
            path,
            "layer_labels[" + LAYER_PREFIX + "1..12] -> marker_gradient_rho",
        )

    if frozen is None:
        raise AssertionError("no frozen reference file")
    for arm, seeds in families.items():
        if sorted(seeds) != [42, 123, 456]:
            raise AssertionError(f"{arm}: expected seeds [42, 123, 456], found {sorted(seeds)}")
    return {
        "families": families,
        "frozen": frozen,
        "layers": layers,
        "sources": sources,
    }


def check_manuscript(values):
    """Assert the layer-12 values the figure set states.

    Raises
    ------
    AssertionError
        If a plotted seed value differs from ``MANUSCRIPT_VALUES``.
    """
    S.check_value(
        "frozen layer-12 rho",
        float(values["frozen"][-1]),
        D.MANUSCRIPT_VALUES["frozen"]["rho"],
    )
    ce = values["families"]["ce_naive"]
    for seed, expected in zip(sorted(ce), D.MANUSCRIPT_VALUES["ce_naive"]["rho_seeds"]):
        S.check_value(f"CE-naive seed-{seed} layer-12 rho", float(ce[seed][-1]), expected)


def plot(values):
    """Draw the seed trajectories, the arm means and the frozen reference."""
    layers = values["layers"]
    seed_ids = sorted(values["families"]["ce_naive"])
    fig, ax = plt.subplots(figsize=S.figsize(*CANVAS_MM))

    for arm, seeds in values["families"].items():
        colour = S.ARM_COLORS[arm]
        matrix = np.vstack([seeds[seed] for seed in sorted(seeds)])
        for seed_index, seed in enumerate(sorted(seeds)):
            linestyle, marker = SEED_STYLES[seed_index]
            ax.plot(
                layers,
                seeds[seed],
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markersize=MARKER_SIZE,
                markeredgewidth=0.0,
                linewidth=LINE_SEED,
                zorder=2,
            )
        ax.plot(
            layers,
            matrix.mean(axis=0),
            color=colour,
            linestyle="-",
            linewidth=LINE_MEAN,
            zorder=3,
        )

    ax.plot(
        layers,
        values["frozen"],
        color=S.ARM_COLORS["frozen"],
        linestyle="-",
        marker="D",
        markersize=MARKER_SIZE,
        markeredgewidth=0.0,
        linewidth=LINE_FROZEN,
        zorder=4,
    )
    ax.axhline(
        NULL_RHO,
        color=S.RULE_COLOR,
        linestyle=":",
        linewidth=S.RULE_LW,
        zorder=1,
    )

    ax.set_ylim(*Y_LIMITS)
    ax.set_xlim(layers[0] - 0.4, layers[-1] + 0.4)
    ax.set_xticks(layers)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Differentiation-gradient Spearman \u03c1")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
    S.panel_label(ax, "A", x=0.0, y=1.01)

    handles = [
        Line2D(
            [0],
            [0],
            color=S.ARM_COLORS["frozen"],
            linestyle="-",
            marker="D",
            markersize=MARKER_SIZE,
            linewidth=LINE_FROZEN,
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="none",
            marker=SEED_STYLES[0][1],
            markersize=MARKER_SIZE,
            markeredgewidth=0.0,
        ),
        Line2D([0], [0], color=S.ARM_COLORS["ce_naive"], linewidth=LINE_MEAN),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="none",
            marker=SEED_STYLES[1][1],
            markersize=MARKER_SIZE,
            markeredgewidth=0.0,
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="none",
            marker=SEED_STYLES[2][1],
            markersize=MARKER_SIZE,
            markeredgewidth=0.0,
        ),
    ]
    labels = [
        S.ARMS["frozen"],
        f"seed {seed_ids[0]}",
        f"{S.ARMS['ce_naive']} seed mean",
        f"seed {seed_ids[1]}",
        f"seed {seed_ids[2]}",
    ]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        handlelength=2.4,
        labelspacing=0.5,
        title="seed shapes",
        title_fontsize=S.TYPE["tick"],
    )
    # than the 2 mm minimum; widen the vertical border pad to exactly 2 mm.
    fig.set_constrained_layout_pads(h_pad=2 / S.MM_PER_INCH)
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

    inputs = {key: path for key, (path, _field) in values["sources"].items()}
    pdf_path, png_path = S.save_figure(fig, OUT_NAME, outdir=str(S.FIGURE_DIR), inputs=inputs)
    plt.close(fig)

    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print("plotted value -> file:field")
    for key, (path, field) in values["sources"].items():
        print(f"  {key}: 12 layers  <- {path}:{field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
