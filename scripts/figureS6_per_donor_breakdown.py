"""Figure S6 — per-donor breakdown of the primary endpoint, paired by seed.

``core_claim``
    The CE-naive arm's endpoint advantage is donor-consistent: for every seed
    and both epochs, donor_1 and donor_2 move together, so the arm ranking does
    not depend on which of the two held-out donors is scored.

``hero_panel``
    The single panel (CE-naive): the three seed lines keep their donor ordering
    at both epochs.

``stats_shown``
    Each per-donor donor-level macro-F1 is drawn as an individual point; seeds
    are joined within a donor pair, epoch 3 is solid and epoch 2 dashed. No
    interval, test or correction is drawn. Every seed of the arm and epoch is
    plotted; a missing value raises instead of being skipped.

``source_data``
    ``artifacts/paper1_tier_b_eval/results_0.0_{42,123,456}.json``
    (epoch 3) and ``results_epoch2_0.0_{42,123,456}.json`` (epoch 2),
    field ``primary_endpoint.per_donor.{donor_1,donor_2}``.

One figure, one producer: this script is the only writer of
``figures/paper1/figureS6_per_donor_breakdown.{pdf,png}``.
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

OUT_NAME = "figureS6_per_donor_breakdown"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 82.0)

# Style constants only (geometry and ink); every plotted number is read below.
SEED_MARKERS = ("o", "s", "^")
SEED_JITTER = 0.07
EPOCH_JITTER = 0.02
MARKER_SIZE = 2.88  # 12 px at 300 dpi
WHITE_EDGE = 0.5
LINE_WIDTH = 1.0
# Explicit range tight to the data (0.598-0.681) instead of a 14 % pad, which
# left ~90 px of dead space at this canvas height.
Y_LIMITS = (0.594, 0.688)

ARMS_SHOWN = ("ce_naive",)
SEEDS = (42, 123, 456)
# epoch -> (solid/dashed, tier-B name prefix)
EPOCHS = {3: ("-", ""), 2: ("--", "epoch2_")}


def load_values():
    """Read every per-donor value and record its provenance.

    Returns
    -------
    dict
        ``{"values": {arm: {epoch: {seed: {donor: float}}}}, "sources": ...,
        "donors": [...]}``.

    Raises
    ------
    AssertionError
        If a file, seed, epoch or donor record is missing, or if the per-donor
        mean does not reproduce the file's own macro-F1.
    """
    values: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
    sources: dict[str, tuple[Path, str]] = {}
    donor_ids: tuple[str, ...] | None = None

    for arm in ARMS_SHOWN:
        values[arm] = {}
        for epoch, (_linestyle, prefix) in EPOCHS.items():
            values[arm][epoch] = {}
            for seed in SEEDS:
                path = D.tier_b(f"{prefix}{_lambda_for(arm)}_{seed}")
                document = D.load_json(path)
                primary = document["primary_endpoint"]
                per_donor = primary["per_donor"]
                donors = tuple(per_donor)
                if len(donors) != 2:
                    raise AssertionError(f"{path.name}: expected two donors, found {donors}")
                if donor_ids is None:
                    donor_ids = donors
                elif donors != donor_ids:
                    raise AssertionError(f"{path.name}: donor keys {donors} != {donor_ids}")
                record = {donor: float(per_donor[donor]) for donor in donors}
                if not all(np.isfinite(list(record.values()))):
                    raise AssertionError(f"{path.name}: non-finite per-donor value")
                # A per-donor record that no longer averages to the file's own
                # primary endpoint is a mis-mapping, not a plottable value.
                S.check_value(
                    f"{path.name} donor mean vs macro_f1",
                    float(np.mean(list(record.values()))),
                    float(primary["macro_f1"]),
                    tol=1e-12,
                )
                values[arm][epoch][seed] = record
                sources[f"{arm}_epoch{epoch}_seed{seed}"] = (
                    path,
                    "primary_endpoint.per_donor.{" + ",".join(donors) + "}",
                )

    if donor_ids is None:
        raise AssertionError("no donors read")
    return {"values": values, "sources": sources, "donors": list(donor_ids)}


def _lambda_for(arm: str) -> str:
    """Map the canonical arm key to its tier-B results file prefix."""
    if arm == "ce_naive":
        return "0.0"
    if arm == "caft_intended":
        return "0.5"
    raise KeyError(f"no tier-B lambda for arm {arm!r}")


def plot(values):
    """Draw the per-donor paired seed lines for the arm."""
    donors = values["donors"]
    x = np.arange(len(donors), dtype=float)
    fig, axes = plt.subplots(
        1, len(ARMS_SHOWN), figsize=S.figsize(*CANVAS_MM), sharey=False
    )
    axes = np.atleast_1d(axes)

    for panel_index, arm in enumerate(ARMS_SHOWN):
        ax = axes[panel_index]
        colour = S.ARM_COLORS[arm]
        for epoch, (linestyle, _prefix) in EPOCHS.items():
            epoch_offset = EPOCH_JITTER if epoch == 2 else -EPOCH_JITTER
            for seed_index, seed in enumerate(SEEDS):
                record = values["values"][arm][epoch][seed]
                y = [record[donor] for donor in donors]
                seed_offset = (seed_index - (len(SEEDS) - 1) / 2) * SEED_JITTER
                ax.plot(
                    x + seed_offset + epoch_offset,
                    y,
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH,
                    color=colour,
                    marker=SEED_MARKERS[seed_index],
                    markersize=MARKER_SIZE,
                    markeredgecolor="white" if epoch == 3 else colour,
                    markerfacecolor=colour if epoch == 3 else "white",
                    markeredgewidth=WHITE_EDGE if epoch == 3 else 0.7,
                    zorder=3 if epoch == 3 else 2,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([donor.replace("_", " ") for donor in donors])
        ax.set_xlim(-0.35, len(donors) - 1 + 0.35)
        ax.set_ylim(*Y_LIMITS)
        ax.set_title(S.ARMS[arm], fontsize=S.TYPE["title"], fontweight="regular", loc="center")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
        S.panel_label(ax, "AB"[panel_index])

    axes[0].set_ylabel("Per-donor macro-F1")

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
    ] + [
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=LINE_WIDTH,
            linestyle=linestyle,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor="white" if linestyle == "--" else "#555555",
            markeredgecolor="#555555",
            markeredgewidth=WHITE_EDGE,
        )
        for linestyle, _prefix in EPOCHS.values()
    ]
    labels = [f"seed {seed}" for seed in SEEDS] + [f"epoch {epoch}" for epoch in EPOCHS]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        handlelength=1.8,
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
    fig = plot(values)
    assert_no_clipping(fig)

    inputs = {}
    for key, (path, _field) in values["sources"].items():
        inputs[key] = path
    pdf_path, png_path = S.save_figure(fig, OUT_NAME, outdir=str(S.FIGURE_DIR), inputs=inputs)
    plt.close(fig)

    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print("plotted value -> file:field")
    for arm in ARMS_SHOWN:
        for epoch in EPOCHS:
            for seed in SEEDS:
                path, field = values["sources"][f"{arm}_epoch{epoch}_seed{seed}"]
                record = values["values"][arm][epoch][seed]
                shown = ", ".join(f"{d}={record[d]:.6f}" for d in values["donors"])
                print(f"  {S.ARMS[arm]} / epoch {epoch} / seed {seed}: {shown}  <- {path}:{field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
