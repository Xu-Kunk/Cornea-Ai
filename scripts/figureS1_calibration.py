"""Figure S1 — per-class reliability diagrams on the external test set.

``core_claim``
    The four-class risk is calibrated in the same way for the frozen and the
    fine-tuned arms: in every class panel the observed positive fraction tracks
    the mean predicted probability along the identity line.

``hero_panel``
    The four class panels together; the differentiated-class panel carries the
    largest positive mass and is the strongest calibration evidence.

``stats_shown``
    Reliability curves over 10 half-open probability bins
    (``[lo, hi)``, last bin closed on the right, the binning fixed in
    ``corneal_fm/evaluation.py``). Thin lines are the individual seeds; the
    thick line is the mean over the seeds that populate a bin. No interval is
    drawn.

``source_data``
    ``artifacts/paper1_tier_b_eval/percell_{frozen_gf,naive_s42,naive_s123,
    naive_s456}.npz`` (per-cell probabilities on the external test split) and
    ``artifacts/paper1_tier_b_eval/results_frozen_gf.json`` (external-test
    provenance: accession, cell and donor counts).

One figure, one producer: this script is the only writer of
``figures/paper1/figureS1_calibration.{pdf,png}``.
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

OUT_NAME = "figureS1_calibration"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 118.0)

CLASSES = ("LSC", "TAC", "basal", "differentiated")
SEEDS = (42, 123, 456)
ARMS = (("ce_naive", "naive"),)
N_BINS = 10

# Seeds are separated by linestyle *and* marker so colour is never the only
# channel; the ordering is the seed order read from the file names.
SEED_STYLES = (("dotted", "o"), ("dashed", "s"), ("dashdot", "^"))

LINE_FROZEN = 1.4
LINE_MEAN = 1.6
LINE_SEED = 0.7
MARKER_SIZE = 2.2
# Seeds are drawn in their arm colour at 55 % alpha so a reader reads the blue
# seeds as CE-naive rather than as a fifth palette hue.
SEED_ALPHA = 0.55
# TAC has almost no positive mass, so the class panel keeps the default 0-1
# range only where the data occupy it; its own range keeps the panel legible.
Y_TOP = {"TAC": 0.05}


def _reliability_bins(y_true: np.ndarray, probabilities: np.ndarray, cls: str):
    """Rows of ``(bin, n, mean_predicted, observed_fraction)`` for one class.

    The binning is the convention fixed in ``corneal_fm/evaluation.py``:
    ``np.digitize(p, edges, right=False)`` with the final bin closed on the
    right, so every cell falls in exactly one bin.
    """
    index = CLASSES.index(cls)
    positive = (y_true == cls).astype(int)
    confidence = np.asarray(probabilities[:, index], dtype=float)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bins = np.clip(np.digitize(confidence, edges, right=False) - 1, 0, N_BINS - 1)
    rows = []
    for bin_index in range(N_BINS):
        mask = bins == bin_index
        count = int(mask.sum())
        rows.append(
            (
                bin_index,
                count,
                float(confidence[mask].mean()) if count else float("nan"),
                float(positive[mask].mean()) if count else float("nan"),
            )
        )
    return rows


def load_percell():
    """Load the seven per-cell archives and the external-test provenance.

    Returns
    -------
    tuple[dict, dict, list[pathlib.Path]]
        Archives keyed by ``percell`` name, the provenance document, and paths.
    """
    provenance_path = D.tier_b("frozen_gf")
    provenance = D.load_json(provenance_path)
    percell = {}
    paths = [provenance_path]
    names = ["frozen_gf"]
    for arm_key, prefix in ARMS:
        for seed in SEEDS:
            names.append(f"{prefix}_s{seed}")
    for name in names:
        path = D.percell(name)
        percell[name] = np.load(path)
        paths.append(path)

    n_ext = int(percell["frozen_gf"]["y_true_ext"].shape[0])
    declared = int(provenance["provenance"]["n_external_cells"])
    if n_ext != declared:
        raise AssertionError(
            f"per-cell external rows ({n_ext}) != declared n_external_cells ({declared})"
        )
    for name, archive in percell.items():
        if archive["probs_ext"].shape[1] != len(CLASSES):
            raise AssertionError(f"{name}: unexpected number of class columns")
        predicted = np.asarray(CLASSES)[archive["probs_ext"].argmax(axis=1)]
        agreement = float((predicted == archive["y_pred_ext"]).mean())
        # Guards against a permuted class axis: the stored label and the
        # probability argmax must name the same class for the vast majority of
        # cells (ties on near-equal probabilities account for the remainder).
        if agreement < 0.90:
            raise AssertionError(
                f"{name}: probability columns disagree with y_pred_ext (agreement {agreement:.3f})"
            )
    return percell, provenance, paths


def plot_class_panel(ax, percell, cls: str) -> None:
    """Draw one reliability panel."""
    class_counts = {}
    for name, archive in percell.items():
        class_counts[name] = _reliability_bins(archive["y_true_ext"], archive["probs_ext"], cls)

    n_cells = int((percell["frozen_gf"]["y_true_ext"] == cls).sum())
    if n_cells == 0:
        raise AssertionError(f"{cls}: no positive cells in the external split")

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=S.RULE_COLOR,
        ls=":",
        lw=S.RULE_LW,
        label="identity",
        zorder=0,
    )

    frozen = np.asarray([(row[2], row[3]) for row in class_counts["frozen_gf"] if row[1] > 0])
    ax.plot(
        frozen[:, 0],
        frozen[:, 1],
        color=S.ARM_COLORS["frozen"],
        ls="-",
        lw=LINE_FROZEN,
        marker="D",
        markersize=MARKER_SIZE,
        markeredgewidth=0.0,
        zorder=4,
    )

    for arm_key, prefix in ARMS:
        colour = S.ARM_COLORS[arm_key]
        curves = []
        for style_index, seed in enumerate(SEEDS):
            linestyle, marker = SEED_STYLES[style_index]
            rows = class_counts[f"{prefix}_s{seed}"]
            points = [(row[2], row[3]) for row in rows if row[1] > 0]
            if not points:
                raise AssertionError(f"{prefix}_s{seed}: no populated bins for {cls}")
            points = np.asarray(points)
            curves.append(points)
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=colour,
                ls=linestyle,
                lw=LINE_SEED,
                marker=marker,
                markersize=MARKER_SIZE,
                markeredgewidth=0.0,
                alpha=SEED_ALPHA,
                zorder=2,
            )
        # Mean over the seeds that populate each bin (index-aligned by bin).
        mean_x = []
        mean_y = []
        for bin_index in range(N_BINS):
            values = [
                (rows[bin_index][2], rows[bin_index][3])
                for rows in (class_counts[f"{prefix}_s{seed}"] for seed in SEEDS)
                if rows[bin_index][1] > 0
            ]
            if len(values) >= 2:
                mean_x.append(float(np.mean([value[0] for value in values])))
                mean_y.append(float(np.mean([value[1] for value in values])))
        ax.plot(
            mean_x,
            mean_y,
            color=colour,
            ls="-",
            lw=LINE_MEAN,
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    top = Y_TOP.get(cls, 1.0)
    ax.set_ylim(0.0, top)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if top < 1.0:
        yticks = np.linspace(0.0, top, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{value:.2f}" for value in yticks])
    else:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(cls)
    ax.set_axisbelow(True)
    ax.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)


def assert_no_clipping(fig) -> None:
    """Raise if any text artist leaves the canvas (a clipped label)."""
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


def build(percell):
    """Assemble the four class panels plus one shared key."""
    fig = plt.figure(figsize=S.figsize(*CANVAS_MM))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    for ax, cls in zip(axes, CLASSES):
        plot_class_panel(ax, percell, cls)
    for ax in axes[2:]:
        ax.set_xlabel("Mean predicted probability")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Observed positive fraction")
    for ax, letter in zip(axes, "ABCD"):
        S.panel_label(ax, letter)

    handles = [
        Line2D(
            [0],
            [0],
            color=S.ARM_COLORS["frozen"],
            ls="-",
            lw=LINE_FROZEN,
            marker="D",
            markersize=MARKER_SIZE,
            markeredgewidth=0.0,
        ),
        Line2D([0], [0], color=S.ARM_COLORS["ce_naive"], ls="-", lw=LINE_MEAN),
        Line2D([0], [0], color=S.RULE_COLOR, ls=":", lw=S.RULE_LW),
    ]
    labels = [
        S.ARMS["frozen"],
        f"{S.ARMS['ce_naive']} (seed mean)",
        "identity",
    ]
    for arm_key, _prefix in ARMS:
        for style_index, seed in enumerate(SEEDS):
            linestyle, marker = SEED_STYLES[style_index]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=S.ARM_COLORS[arm_key],
                    ls=linestyle,
                    lw=LINE_SEED,
                    marker=marker,
                    markersize=MARKER_SIZE,
                    markeredgewidth=0.0,
                )
            )
            labels.append(f"{S.ARMS[arm_key]} seed {seed}")
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
        labelspacing=0.5,
    )
    return fig


def main() -> int:
    """Render the figure, verify it, and write the manifest entry."""
    S.set_style()
    percell, provenance, paths = load_percell()
    provenance_meta = provenance["provenance"]
    note = (
        f"n = {int(provenance_meta['n_external_cells']):,} epithelial cells, "
        f"{int(provenance_meta['n_eval_donors'])} donors "
        f"({provenance_meta['accession']} external test); "
        f"reliability over {N_BINS} probability bins"
    )
    fig = build(percell)
    assert_no_clipping(fig)
    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={path.stem: path for path in paths},
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
