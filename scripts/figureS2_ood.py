"""Figure S2 — out-of-distribution confidence and per-donor calibration.

``core_claim``
    Fine-tuned arms separate external from internal cells in confidence, while
    the frozen arm over-commits on both splits; calibration itself remains
    donor-dependent and is not rescued by adaptation.

``hero_panel``
    Panel A: the frozen arm's max-probability density, which is piled near 1.0
    on the external split and still heavy on the internal holdout.

``stats_shown``
    Panels A-B: Gaussian kernel density of the per-cell maximum predicted
    probability, external versus internal, pooled over the seeds of an arm (the
    seed count is stated in the panel title). Panel C: expected calibration
    error (10 half-open bins, the convention fixed in
    ``corneal_fm/evaluation.py``) per external-test donor. Panel D: mean of the
    per-cell maximum predicted probability per external-test donor. No
    reference line is drawn in any panel: the data contain no
    calibrated-fraction reference, and a literal threshold would be an
    invented value.

``source_data``
    ``artifacts/paper1_tier_b_eval/percell_{frozen_gf,naive_s42,naive_s123,
    naive_s456}.npz``,
    ``artifacts/paper1_tier_b_eval/results_frozen_gf.json`` (provenance) and
    ``figures/paper1/figureS2_ood_perdonor.csv`` (panel-E values, column
    ``mean_confidence``; cross-checked cell-by-cell against the per-cell
    archives, which are the stronger source).

One figure, one producer: this script is the only writer of
``figures/paper1/figureS2_ood.{pdf,png}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paper1_figure_data as D  # noqa: E402, N812
import paper1_figure_style as S  # noqa: E402, N812
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from corneal_fm.evaluation import expected_calibration_error  # noqa: E402

OUT_NAME = "figureS2_ood"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 169.7)
PERDONOR_CSV = S.FIGURE_DIR / "figureS2_ood_perdonor.csv"

CLASSES = ("LSC", "TAC", "basal", "differentiated")
SEEDS = (42, 123, 456)
ARMS = (("frozen", "frozen_gf"), ("ce_naive", "naive"))
SPLITS = (("ext", "external test"), ("int", "internal holdout"))
DONOR_HATCH = ("", "////")
EDGE_COLOR = "#333333"

LINE_WIDTH = 1.3
KDE_POINTS = 512
# One shared density limit across the three arm panels so the curves are
# directly comparable and no peak is clipped.
OOD_Y_MAX = 15.0
# Panel E is a probability, so it is drawn on the full zero-to-one range: a
# truncated axis would exaggerate the donor gap between near-ceiling arms.
CONFIDENCE_Y_MAX = 1.0
# Paired donor bands, shared by panels D and E. 1.5 px of white between the two
# adjacent donor bars at 300 dpi: the two bars keep a 0.38 data-unit pitch, but
# each is drawn 0.3744 wide so a hairline of the page shows through between
# them.
DONOR_PITCH = 0.38
DONOR_WIDTH = 0.3744


def load_percell():
    """Load the seven per-cell archives and the external-test provenance.

    Returns
    -------
    tuple[dict, dict, list[pathlib.Path]]
        Archives keyed by name, provenance document, and input paths.
    """
    provenance_path = D.tier_b("frozen_gf")
    provenance = D.load_json(provenance_path)
    percell = {}
    paths = [provenance_path]
    for _arm_key, prefix in ARMS:
        for name in arm_members(prefix):
            path = D.percell(name)
            percell[name] = np.load(path)
            paths.append(path)
    for name, archive in percell.items():
        if archive["probs_ext"].shape[1] != len(CLASSES):
            raise AssertionError(f"{name}: unexpected number of class columns")
    return percell, provenance, paths


def arm_members(prefix: str) -> list[str]:
    """Per-cell archive names belonging to one arm."""
    return ["frozen_gf"] if prefix == "frozen_gf" else [f"{prefix}_s{seed}" for seed in SEEDS]


def model_slots() -> list[tuple[str, str]]:
    """``(arm key, archive name)`` in display order, shared by panels D and E."""
    slots = []
    for arm_key, prefix in ARMS:
        for name in arm_members(prefix):
            slots.append((arm_key, name))
    return slots


def load_perdonor():
    """Read the per-donor table written for this figure.

    Returns
    -------
    pandas.DataFrame
        One row per (model, external donor) with ``ece`` and
        ``mean_confidence`` columns.
    """
    frame = D.load_csv(PERDONOR_CSV)
    required = {"model", "donor", "ece", "mean_confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"{PERDONOR_CSV.name} lacks column(s): {sorted(missing)}")
    return frame


def max_probabilities(archive, split: str) -> np.ndarray:
    """Maximum predicted probability per cell for one split."""
    return np.asarray(archive[f"probs_{split}"], dtype=float).max(axis=1)


def plot_kde_panel(ax, percell, arm_key: str, prefix: str) -> None:
    """Density of the maximum predicted probability: external vs internal."""
    colour = S.ARM_COLORS[arm_key]
    grid = np.linspace(0.0, 1.0, KDE_POINTS)
    for split, label in SPLITS:
        pooled = np.concatenate(
            [max_probabilities(percell[name], split) for name in arm_members(prefix)]
        )
        if pooled.size < 2:
            raise AssertionError(f"{prefix}: too few cells for a density on {split}")
        density = gaussian_kde(pooled)(grid)
        ax.plot(
            grid,
            density,
            color=colour,
            ls="-" if split == "ext" else "--",
            lw=LINE_WIDTH,
            zorder=3 if split == "ext" else 2,
            label=label,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, OOD_Y_MAX)
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(S.ARMS[arm_key], fontsize=S.TYPE["title"], fontweight="regular", loc="center")
    ax.set_axisbelow(True)
    ax.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)


def donor_ece(percell, donor: str) -> dict[str, float]:
    """Expected calibration error per arm member for one external donor."""
    ece = {}
    for _arm_key, prefix in ARMS:
        for name in arm_members(prefix):
            archive = percell[name]
            mask = archive["donor_ext"] == donor
            y_index = np.array(
                [CLASSES.index(label) for label in archive["y_true_ext"][mask]],
                dtype=int,
            )
            ece[name] = float(
                expected_calibration_error(
                    y_index, np.asarray(archive["probs_ext"][mask], dtype=float)
                )
            )
    return ece


def donor_mean_confidence(percell, donor: str) -> dict[str, float]:
    """Mean of the per-cell maximum predicted probability, per arm member.

    This is the quantity panel E plots. It is recomputed from the per-cell
    archives so the CSV used for the drawing can be checked against its source.
    """
    confidence = {}
    for _arm_key, prefix in ARMS:
        for name in arm_members(prefix):
            archive = percell[name]
            mask = archive["donor_ext"] == donor
            confidence[name] = float(max_probabilities(archive, "ext")[mask].mean())
    return confidence


def plot_donor_panel(ax, percell, donors) -> None:
    """Per-donor expected calibration error, arm colour plus donor hatch."""
    models = model_slots()
    x = np.arange(len(models), dtype=float)
    pitch = DONOR_PITCH
    width = DONOR_WIDTH
    values = {donor: donor_ece(percell, donor) for donor in donors}

    for donor_index, donor in enumerate(donors):
        offset = (donor_index - 0.5) * pitch
        ax.bar(
            x + offset,
            [values[donor][name] for _arm_key, name in models],
            width,
            color=[S.ARM_COLORS[arm_key] for arm_key, _name in models],
            hatch=DONOR_HATCH[donor_index],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
            zorder=3,
            label=f"donor {donor.split('_')[-1]}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_display_name(name) for _arm_key, name in models], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0.0, float(max(max(v.values()) for v in values.values())) * 1.35)
    ax.set_ylabel("Expected calibration error")
    ax.set_title(
        "Per-donor calibration (external test)",
        fontsize=S.TYPE["title"],
        fontweight="regular",
        loc="center",
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)
    ax.legend(
        handles=[
            Patch(facecolor="#808080", edgecolor="#333333", lw=0.5),
            Patch(facecolor="none", hatch=DONOR_HATCH[1], edgecolor="#333333", lw=0.5),
        ],
        labels=[f"donor {donor.split('_')[-1]}" for donor in donors],
        loc="upper left",
        frameon=False,
        borderpad=0.4,
        handlelength=1.6,
        ncol=len(donors),
    )


def plot_confidence_panel(ax, perdonor, donors) -> None:
    """Per-donor mean maximum predicted probability.

    Donor identity uses panel D's encoding exactly: the arm hue carries the
    model and the fill carries the donor (solid versus hatch), so no second hue
    is introduced and the two panels share one key.
    """
    models = model_slots()
    lookup = {
        (str(row.model), str(row.donor)): float(row.mean_confidence)
        for row in perdonor.itertuples(index=False)
    }
    absent = [
        (name, donor)
        for _arm_key, name in models
        for donor in donors
        if (name, donor) not in lookup
    ]
    if absent:
        raise AssertionError(f"{PERDONOR_CSV.name} lacks per-donor values for {absent}")

    x = np.arange(len(models), dtype=float)
    for donor_index, donor in enumerate(donors):
        offset = (donor_index - 0.5) * DONOR_PITCH
        ax.bar(
            x + offset,
            [lookup[(name, donor)] for _arm_key, name in models],
            DONOR_WIDTH,
            color=[S.ARM_COLORS[arm_key] for arm_key, _name in models],
            hatch=DONOR_HATCH[donor_index],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
            zorder=3,
            label=f"donor {donor.split('_')[-1]}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_display_name(name) for _arm_key, name in models], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0.0, CONFIDENCE_Y_MAX)
    ax.set_ylabel("Mean max probability")
    ax.set_title(
        "Per-donor mean confidence (external test)",
        fontsize=S.TYPE["title"],
        fontweight="regular",
        loc="center",
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)


def _display_name(name: str) -> str:
    """Display label for one per-cell archive name."""
    if name == "frozen_gf":
        return S.ARMS["frozen"]
    arm, seed = name.rsplit("_s", 1)
    if arm != "naive":
        raise AssertionError(f"unexpected adapted arm prefix {arm!r} in {name!r}")
    arm_key = "ce_naive"
    return f"{S.ARMS[arm_key]} seed {seed}"


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


def build(percell, perdonor, donors):
    """Assemble the three density panels, the per-donor ECE and confidence panels."""
    fig = plt.figure(figsize=S.figsize(*CANVAS_MM))
    grid = fig.add_gridspec(3, len(ARMS), height_ratios=[1.0, 0.85, 0.85])
    kde_axes = [fig.add_subplot(grid[0, 0])]
    for column in range(1, len(ARMS)):
        kde_axes.append(fig.add_subplot(grid[0, column], sharey=kde_axes[0]))
    for ax, (arm_key, prefix) in zip(kde_axes, ARMS):
        plot_kde_panel(ax, percell, arm_key, prefix)
    ax_d = fig.add_subplot(grid[1, :])
    plot_donor_panel(ax_d, percell, donors)
    ax_e = fig.add_subplot(grid[2, :])
    plot_confidence_panel(ax_e, perdonor, donors)

    kde_axes[0].legend(
        loc="upper left",
        frameon=False,
        borderpad=0.4,
        handlelength=1.8,
    )
    for ax, letter in zip(kde_axes, "AB"):
        S.panel_label(ax, letter)
    # The two full-width rows follow the density panels, so their letters stay
    # contiguous with the reduced panel count.
    S.panel_label(ax_d, "C", x=0.0)
    S.panel_label(ax_e, "D", x=0.0)
    return fig


def main() -> int:
    """Render the figure, verify it, and write the manifest entry."""
    S.set_style()
    percell, provenance, paths = load_percell()
    perdonor = load_perdonor()
    donors = np.unique(percell["frozen_gf"]["donor_ext"])
    if donors.size < 2:
        raise AssertionError("expected at least two external test donors")
    for name, archive in percell.items():
        if not np.array_equal(donors, np.unique(archive["donor_ext"])):
            raise AssertionError(f"{name}: external donor set differs")

    # Panel E draws the CSV, so the CSV must agree with the per-cell archives it
    # was derived from; a rounded or stale table fails here rather than silently
    # changing a published bar height.
    for donor in donors:
        computed = donor_mean_confidence(percell, donor)
        for _arm_key, name in model_slots():
            row = perdonor[(perdonor["model"] == name) & (perdonor["donor"] == donor)]
            if len(row) != 1:
                raise AssertionError(f"{PERDONOR_CSV.name}: {len(row)} rows for {name}/{donor}")
            S.check_value(
                f"mean confidence {name}/{donor}",
                float(row["mean_confidence"].iloc[0]),
                computed[name],
                tol=1e-9,
            )

    meta = provenance["provenance"]
    n_internal = int(percell["frozen_gf"]["probs_int"].shape[0])
    note = (
        f"external: n = {int(meta['n_external_cells']):,} cells, "
        f"{int(meta['n_eval_donors'])} donors ({meta['accession']}); "
        f"internal holdout: n = {n_internal:,} epithelial cells "
        "(each run's own prepared test split)"
    )
    fig = build(percell, perdonor, donors)
    assert_no_clipping(fig)
    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={path.stem: path for path in paths} | {"perdonor_csv": PERDONOR_CSV},
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    for donor in donors:
        ece = donor_ece(percell, donor)
        shown = ", ".join(f"{name}={value:.4f}" for name, value in ece.items())
        print(f"  ECE ({donor}): {shown}")
        confidence = donor_mean_confidence(percell, donor)
        shown = ", ".join(f"{name}={value:.4f}" for name, value in confidence.items())
        print(f"  mean max probability ({donor}): {shown}")
    print(f"  {note}")
    return 0
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    for donor in donors:
        ece = donor_ece(percell, donor)
        shown = ", ".join(f"{name}={value:.4f}" for name, value in ece.items())
        print(f"  ECE ({donor}): {shown}")
    print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
