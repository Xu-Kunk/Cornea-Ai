"""Figure S9 — sensitivity of the primary endpoint to the rare TAC class.

``core_claim``
    Excluding the 171 TAC cells raises donor-level macro-F1 by ~0.21 for every
    adapted run but by only ~0.15 for the frozen reference, so the arm ranking
    of the primary endpoint is not an artefact of the class the adapted runs
    never recover.

``hero_panel``
    Panel A: the paired four-class and TAC-excluded endpoints per run, with
    their donor-level bootstrap intervals.

``stats_shown``
    Both endpoints are recomputed here with the locked evaluator
    (:func:`corneal_fm.evaluation.donor_level_macro_f1`, n_boot = 1000,
    bootstrap seed 42) and carry that function's asymmetric 95% CI bounds for
    each run individually. Panel B difference (TAC-excluded minus four-class) is
    computed per seed and only then averaged; the arithmetic mean of the three
    seed differences is drawn as a tick, and no interval is drawn for it
    because a difference of two separately bootstrapped endpoints has no stored
    interval. Frozen is a single reference run. Panel C shows pooled cell-level
    per-class F1 for the four-class schema without interval.

``source_data``
    ``artifacts/paper1_tier_b_eval/percell_{naive_s42,naive_s123,naive_s456,
    frozen_gf}.npz`` (arrays
    ``y_true_ext``, ``y_pred_ext``, ``donor_ext``) and the matching
    ``percell_*.npz.meta.json`` sanity gates. The three-class schema is the
    locked four-class schema minus TAC, evaluated by the same function.

One figure, one producer: this script is the only writer of
``figures/paper1/figureS9_tac_sensitivity.{pdf,png}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paper1_figure_data as D  # noqa: E402, N812
import paper1_figure_style as S  # noqa: E402, N812
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from corneal_fm.evaluation import (  # noqa: E402
    BOOT_SEED,
    DEFAULT_CLASSES,
    N_BOOT,
    donor_level_macro_f1,
)

OUT_NAME = "figureS9_tac_sensitivity"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 120.0)

# Style constants only (geometry and ink); every plotted number is read below.
EXCLUDED_CLASS = "TAC"
CLASSES_4 = list(DEFAULT_CLASSES)
CLASSES_3 = [name for name in CLASSES_4 if name != EXCLUDED_CLASS]

# canonical arm key -> per-cell archive prefix. Frozen appears once.
ARM_PREFIX = {
    "ce_naive": "naive_s",
    "frozen": "frozen_gf",
}
ARMS_SHOWN = ("ce_naive", "frozen")
SEEDS = (42, 123, 456)

MARKER_SIZE = S.MARKER_PT
MEAN_HALF_HEIGHT = 0.16
PAD_FRACTION = 0.06
CAPSIZE = 1.6
BAR_WIDTH = 0.8
BAR_EDGE = "#444444"
# Intervals and the leader between the two endpoints are supporting ink: one
# weight and one alpha for both, so neither can be read as a data mark.
INTERVAL_ALPHA = 0.45
# A per-class F1 below this renders as a zero-height rectangle: draw the zero
# explicitly instead of leaving an invisible bar.
ZERO_F1 = 0.005
ZERO_HEADROOM = 0.035


def _archive(arm: str, seed: int | None) -> str:
    """Return the per-cell archive name for an arm (and seed)."""
    prefix = ARM_PREFIX[arm]
    return prefix if seed is None else f"{prefix}{seed}"


def load_values():
    """Recompute both endpoints per run and record the provenance.

    Returns
    -------
    dict
        ``{"runs": [ {...}, ... ], "sources": {...}}`` with the keys
        ``arm``, ``seed``, ``label``, ``four_class``, ``three_class``
        (each ``macro_f1`` / ``ci_lower`` / ``ci_upper`` / ``per_class``),
        ``n_cells`` and ``n_donors``.

    Raises
    ------
    AssertionError
        If a per-cell archive disagrees with the locked sanity gate, with the
        manuscript cell count, or with the stated donor count.
    """
    runs: list[dict] = []
    sources: dict[str, tuple[Path, str]] = {}

    for arm in ARMS_SHOWN:
        seeds = SEEDS if ARM_PREFIX[arm] != ARM_PREFIX["frozen"] else (None,)
        for seed in seeds:
            name = _archive(arm, seed)
            path = D.percell(name)
            with np.load(path, allow_pickle=True) as archive:
                y_true = np.asarray(archive["y_true_ext"])
                y_pred = np.asarray(archive["y_pred_ext"])
                donors = np.asarray(archive["donor_ext"]).astype(str)

            S.check_value(
                f"{name} external cell count",
                float(len(y_true)),
                float(D.MANUSCRIPT_VALUES["gse153515"]["n_cells"]),
            )
            S.check_value(
                f"{name} external donor count",
                float(len(np.unique(donors))),
                float(D.MANUSCRIPT_VALUES["gse153515"]["n_donors"]),
            )

            four = donor_level_macro_f1(
                y_true, y_pred, donors, CLASSES_4, n_boot=N_BOOT, seed=BOOT_SEED
            )
            keep = y_true != EXCLUDED_CLASS
            if int(keep.sum()) == len(y_true):
                raise AssertionError(f"{name}: no {EXCLUDED_CLASS} cells to exclude")
            three = donor_level_macro_f1(
                y_true[keep],
                y_pred[keep],
                donors[keep],
                CLASSES_3,
                n_boot=N_BOOT,
                seed=BOOT_SEED,
            )

            meta_path = Path(str(path) + ".meta.json")
            gate = D.load_json(meta_path)["sanity_gate_external"]
            S.check_value(
                f"{name} locked sanity gate",
                float(four["macro_f1"]),
                float(gate["macro_f1"]),
                tol=1e-9,
            )

            label = S.ARMS[arm] if seed is None else f"{S.ARMS[arm]} {seed}"
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "label": label,
                    "four_class": four,
                    "three_class": three,
                    "n_cells": int(len(y_true)),
                    "n_excluded": int((~keep).sum()),
                }
            )
            sources[f"percell_{name}"] = (
                path,
                "y_true_ext / y_pred_ext / donor_ext",
            )
            sources[f"meta_{name}"] = (meta_path, "sanity_gate_external.macro_f1")

    return {"runs": runs, "sources": sources}


def plot(runs):
    """Draw the paired-endpoint, difference and per-class panels."""
    fig = plt.figure(figsize=S.figsize(*CANVAS_MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(grid[1, :])

    y_positions = -np.arange(len(runs), dtype=float)
    four_values: list[float] = []
    three_values: list[float] = []
    delta_values: list[float] = []

    for run, y in zip(runs, y_positions):
        colour = S.ARM_COLORS[run["arm"]]
        for endpoint, marker, filled, store in (
            (run["four_class"], "o", True, four_values),
            (run["three_class"], "s", False, three_values),
        ):
            value = float(endpoint["macro_f1"])
            low = float(endpoint["ci_lower"])
            high = float(endpoint["ci_upper"])
            if not low <= value <= high:
                raise AssertionError(f"{run['label']}: value outside its own CI")
            ax_a.errorbar(
                [value],
                [y],
                xerr=np.asarray([[value - low], [high - value]]),
                fmt="none",
                ecolor=colour,
                elinewidth=S.LW_SECONDARY,
                alpha=INTERVAL_ALPHA,
                capsize=CAPSIZE,
                zorder=2,
            )
            ax_a.plot(
                value,
                y,
                linestyle="none",
                marker=marker,
                markersize=MARKER_SIZE,
                markerfacecolor=colour if filled else "white",
                markeredgecolor=colour,
                markeredgewidth=0.8,
                zorder=3,
            )
            store.extend([low, high])

        ax_a.plot(
            [float(run["four_class"]["macro_f1"]), float(run["three_class"]["macro_f1"])],
            [y, y],
            color=colour,
            linewidth=S.LW_SECONDARY,
            alpha=INTERVAL_ALPHA,
            zorder=1,
        )

        delta = float(run["three_class"]["macro_f1"]) - float(run["four_class"]["macro_f1"])
        ax_b.plot(
            delta,
            y,
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=colour,
            markeredgecolor=colour,
            markeredgewidth=0.4,
            zorder=3,
        )
        delta_values.append(delta)

    for arm in ("ce_naive",):
        arm_deltas = [d for run, d in zip(runs, delta_values) if run["arm"] == arm]
        if len(arm_deltas) != len(SEEDS):
            raise AssertionError(f"{arm}: expected {len(SEEDS)} seed differences")
        mean_delta = float(np.mean(arm_deltas))
        mean_row = float(
            np.mean([position for run, position in zip(runs, y_positions) if run["arm"] == arm])
        )
        ax_b.plot(
            [mean_delta, mean_delta],
            [mean_row - MEAN_HALF_HEIGHT, mean_row + MEAN_HALF_HEIGHT],
            color=S.ARM_COLORS["frozen"],
            linewidth=S.LW_MAIN,
            solid_capstyle="butt",
            zorder=4,
        )
        delta_values.append(mean_delta)

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([run["label"] for run in runs])
    ax_a.set_ylim(y_positions[-1] - 0.7, y_positions[0] + 0.7)
    low = min(four_values + three_values)
    high = max(four_values + three_values)
    pad = (high - low) * PAD_FRACTION
    ax_a.set_xlim(low - pad, high + pad)
    ax_a.set_xlabel("Donor-level macro-F1", fontsize=S.TYPE["axis"])
    ax_a.set_axisbelow(True)
    ax_a.xaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
    S.panel_label(ax_a, "A")

    low = min(delta_values)
    high = max(delta_values)
    pad = (high - low) * PAD_FRACTION
    ax_b.set_xlim(low - pad, high + pad)
    ax_b.set_xlabel(
        "Difference in macro-F1 (three-class \u2212 four-class)", fontsize=S.TYPE["axis"]
    )
    ax_b.set_axisbelow(True)
    ax_b.xaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
    # Panel B shares panel A's rows: repeat the row labels so every mark in B
    # (including the two seed-mean ticks) sits on an identifiable row.
    ax_b.tick_params(left=False, labelleft=True)
    S.panel_label(ax_b, "B")

    x = np.arange(len(runs), dtype=float)
    width = BAR_WIDTH / len(CLASSES_4)
    zero_marks: list[tuple[float, float, str]] = []
    for class_index, class_name in enumerate(CLASSES_4):
        offset = (class_index - (len(CLASSES_4) - 1) / 2) * width
        values = [float(run["four_class"]["per_class"][class_name]["f1"]) for run in runs]
        colour = S.CELLTYPE_COLORS[class_name]
        ax_c.bar(
            x + offset,
            values,
            width,
            color=colour,
            edgecolor=BAR_EDGE,
            linewidth=0.4,
            label=class_name,
        )
        # A zero-height rectangle draws no ink, so a class the model never
        # predicts would silently vanish from the panel while staying in the
        # legend. Mark each such bar explicitly: a baseline tick plus a point
        # at the bar's own value.
        for position, value in zip(x + offset, values):
            if value < ZERO_F1:
                zero_marks.append((float(position), float(value), colour))
    # A rule at zero gives the zero-height bars a baseline to sit on.
    ax_c.axhline(0.0, color=S.RULE_COLOR, linewidth=S.RULE_LW, zorder=1)
    for position, value, colour in zero_marks:
        ax_c.plot(
            [position - width / 2, position + width / 2],
            [0.0, 0.0],
            color=BAR_EDGE,
            linewidth=S.LW_REFERENCE,
            solid_capstyle="butt",
            zorder=4,
        )
        ax_c.plot(
            position,
            value,
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=colour,
            markeredgecolor=BAR_EDGE,
            markeredgewidth=0.4,
            zorder=5,
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [
            S.ARMS[run["arm"]] if run["seed"] is None else f"{S.ARMS[run['arm']]}\n{run['seed']}"
            for run in runs
        ]
    )
    # Small headroom under zero so the zero marks are not clipped by the spine.
    ax_c.set_ylim(-ZERO_HEADROOM, 1.0)
    ax_c.set_ylabel("Per-class F1 (cell level)", fontsize=S.TYPE["axis"])
    ax_c.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_c.set_axisbelow(True)
    ax_c.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
    S.panel_label(ax_c, "C")

    endpoint_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=S.SHAPE_KEY_COLOR,
            markeredgecolor=S.SHAPE_KEY_COLOR,
            label="four-class (with TAC)",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=S.SHAPE_KEY_COLOR,
            markeredgewidth=0.8,
            label="three-class (TAC excluded)",
        ),
    ]
    handles = (
        endpoint_handles
        + [
            # The plotted seed mean is a short vertical tick, so the key is the
            # same tick rather than a horizontal dash.
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="|",
                markersize=2 * MARKER_SIZE,
                markeredgewidth=S.LW_MAIN,
                color=S.ARM_COLORS["frozen"],
            ),
        ]
        + [
            Patch(
                facecolor=S.CELLTYPE_COLORS[class_name],
                edgecolor=BAR_EDGE,
                linewidth=0.4,
                label=class_name,
            )
            for class_name in CLASSES_4
        ]
    )
    handles[2].set_label("seed mean difference")
    # Column-major legend fill: interleave so the first row reads
    # endpoint / endpoint / mean / first class and the second row the classes.
    order = [0, 4, 1, 5, 2, 6, 3]
    handles = [handles[index] for index in order]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        handlelength=1.4,
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
    fig = plot(values["runs"])
    assert_no_clipping(fig)

    inputs = {key: path for key, (path, _field) in values["sources"].items()}
    pdf_path, png_path = S.save_figure(fig, OUT_NAME, outdir=str(S.FIGURE_DIR), inputs=inputs)
    plt.close(fig)

    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print("plotted value -> file:field")
    for run in values["runs"]:
        four = run["four_class"]
        three = run["three_class"]
        delta = float(three["macro_f1"]) - float(four["macro_f1"])
        print(
            f"  {run['label']}: four-class={four['macro_f1']:.6f} "
            f"[{four['ci_lower']:.6f}, {four['ci_upper']:.6f}] "
            f"TAC-excluded={three['macro_f1']:.6f} "
            f"[{three['ci_lower']:.6f}, {three['ci_upper']:.6f}] "
            f"delta={delta:+.6f}  <- percell_*.npz:y_true_ext/y_pred_ext/donor_ext "
            f"recomputed with corneal_fm.evaluation.donor_level_macro_f1"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
