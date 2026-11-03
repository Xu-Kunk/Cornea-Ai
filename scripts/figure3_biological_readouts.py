"""Figure 3 — gene-program readouts across representations.

``core_claim``
    The frozen encoder and the ordinal-head arms keep the cornea, epidermis and
    cycling gene programs aligned with the differentiation axis, whereas plain
    cross-entropy fine-tuning does not.

``hero_panel``
    Panel C: basal-program Spearman rho with cell-bootstrap intervals in two
    independent epidermis cohorts.

``stats_shown``
    Panel A: adapted bars are means over seeds 42/123/456 with +/- 1 population
    SD (ddof = 0); frozen is a single reference with a 95% cell-bootstrap
    interval (500 resamples, axis fixed). Panel C: 95% cell-bootstrap intervals
    (500 resamples) per archived checkpoint. Panels B and D: no interval.

``source_data``
    ``artifacts/archive/paper1_bio_validation/bio_validation_results.json``
    (canonical field source), cross-checked against
    ``figure_data_programs.csv`` and ``figure_data_deciles.json``.

One figure, one producer: this script is the only writer of
``figures/paper1/figure3_biological_readouts.pdf`` and ``.png``.
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
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

OUT_NAME = "figure3_biological_readouts"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 150.0)

ARCHIVE = D.ARTIFACTS / "archive" / "paper1_bio_validation"
CANONICAL_JSON = ARCHIVE / "bio_validation_results.json"
PROGRAM_CSV = ARCHIVE / "figure_data_programs.csv"
DECILES_JSON = ARCHIVE / "figure_data_deciles.json"

SEEDS = (42, 123, 456)
CI_LEVEL_PCT = 95  # level of the archived cell-bootstrap interval
NULL_RHO = 0.0  # Spearman rho of independent rankings

# Data key -> canonical arm key in ``S.ARMS``.
PANEL_A_ARMS = (
    ("frozen", "frozen"),
    ("naive", "ce_naive"),
    ("ordinal0", "ordinal"),
    ("ordinal5", "ordinal_cont"),
)
PANEL_B_ARMS = (
    ("frozen", "frozen"),
    ("naive_42", "ce_naive"),
    ("ordinal0_42", "ordinal"),
)
# Hatch means "null/secondary" (the permutation-null convention in Figure S11),
# so only the zero-continuity ablation carries it; the full continuity arm is
# identified by an open marker instead.
PANEL_A_HATCH = {"ordinal"}
PANEL_A_ORDER = {
    arm_key: index for index, (_, arm_key) in enumerate(PANEL_A_ARMS)
}

# Panel C rows: data key -> canonical arm key, or an explicit display label for
# ablation arms that are not part of the shared arm vocabulary.
PANEL_C_ARMS = (
    ("frozen", "frozen", None),
    ("ce_naive", "ce_naive", None),
    ("corns_ordinal", "ordinal", None),
)
PANEL_C_DATASETS = (
    # Display labels follow the caption's accession wording: the series is the
    # GSE147424 epidermis set, so the key "wang2020" never reaches the artwork.
    ("wang2020", "GSE147424", "#333333", "o"),
    ("gse202352", "GSE202352", "#8C8C8C", "s"),
)

PROGRAMS = (("program_stemness", "Stemness"), ("program_differentiation", "Differentiation"))

# Panel D: dataset key inside the canonical JSON and its group heading.
PANEL_D_GROUPS = (
    ("cornea", "Cornea (GSE153515)"),
    ("wang2020", "Epidermis (GSE147424)"),
)
D_MARKER_COLOR = "#333333"
D_STEM_COLOR = "#D0D0D0"
D_STEM_LW = 1.2
D_CAP_HALF = 0.18  # lollipop end-cap half-height, row units (geometry only)

# Style constants (geometry/ink only; no data values).
BAR_WIDTH = 0.17
BAR_GAP = 0.55
BAR_EDGE_COLOR = "#4D4D4D"
BAR_EDGE_LW = 0.4
ERROR_LW = 0.9
ERROR_CAPSIZE = 2.2
POINT_MARKERSIZE = 4.0
# Panel C interval geometry. Four of the six archived 95% cell-bootstrap
# intervals are narrower than the 4 pt marker used elsewhere in the figure, so
# at 4 pt the interval was invisible: the marker covered it. Panel C therefore
# uses a 2.2 pt marker with a 0.5 pt interval line and 1.4 pt caps, which makes
# the cap pair wider than the marker and leaves the interval legible.
C_MARKER_POINT = 2.2
C_LEGEND_MARKER_POINT = 3.2
C_RANGE_LW = 0.5
C_CAPSIZE = 1.4
C_MARKER_EDGE_LW = 0.25
RANGE_LW = 0.9
PAD_FRACTION = 0.06
ROW_STEP = 1.0
GROUP_GAP = 1.2
D_LABEL_OFFSET_FRACTION = 0.055
D_TEXT_CLEARANCE_MM = 4.0
# One zero reference everywhere: S.RULE_COLOR, 0.6 pt, dashed, behind the data.
REF_LW = S.RULE_LW
REF_COLOR = S.RULE_COLOR
REF_STYLE = "--"
REF_ZORDER = 1
# Legend anchor: 6 px inset from the axes top-left corner (6 px at 300 dpi).
LEGEND_INSET_PT = 6.0 / 300.0 * 72.0


def load_payload():
    """Load and cross-check the three archived inputs.

    Returns
    -------
    tuple[dict, pandas.DataFrame, dict, dict[str, pathlib.Path]]
        Canonical JSON, program table, decile table and every input path.
    """
    payload = D.load_json(CANONICAL_JSON)
    program_table = D.load_csv(PROGRAM_CSV)
    deciles = D.load_json(DECILES_JSON)
    paths = {
        "bio_validation_results": CANONICAL_JSON,
        "figure_data_programs": PROGRAM_CSV,
        "figure_data_deciles": DECILES_JSON,
    }
    return payload, program_table, deciles, paths


def _program_row(table, tissue, arm, program):
    """Return the single archived CSV row for a tissue/arm/program key."""
    match = table[
        (table["tissue"] == tissue)
        & (table["arm"] == arm)
        & (table["program"] == program)
    ]
    if len(match) != 1:
        raise AssertionError(f"expected one program row for {tissue}/{arm}/{program}")
    return match.iloc[0]


def build_records(payload, program_table, deciles):
    """Extract every plotted value and verify it against its archive.

    Parameters
    ----------
    payload : dict
        Canonical ``bio_validation_results.json``.
    program_table : pandas.DataFrame
        Archived program table used for cross-checking.
    deciles : dict
        Archived decile table used for cross-checking.

    Returns
    -------
    dict
        Panel-keyed record collections.

    Raises
    ------
    AssertionError
        If a derived seed mean/SD disagrees with its per-seed values or a
        companion archive differs from the canonical record.
    """
    n_boot = int(payload["n_boot"])

    panel_a = []
    for data_arm, arm_key in PANEL_A_ARMS:
        for program, _ in PROGRAMS:
            if data_arm == "frozen":
                block = payload["cornea"]["arms"][data_arm][program]
                record = {
                    "arm": arm_key,
                    "program": program,
                    "value": float(block["rho"]),
                    "low": float(block["ci"]["ci_lower"]),
                    "high": float(block["ci"]["ci_upper"]),
                    "kind": "ci",
                }
                row = _program_row(
                    program_table,
                    "cornea",
                    "frozen",
                    program.removeprefix("program_"),
                )
                np.testing.assert_allclose(
                    [record["value"], record["low"], record["high"]],
                    [float(row["rho"]), float(row["ci_lower"]), float(row["ci_upper"])],
                    rtol=0.0,
                    atol=1e-12,
                )
            else:
                aggregate = payload["cornea_seed_aggregate"][data_arm][program]
                per_seed = np.array(
                    [
                        float(payload["cornea"]["arms"][f"{data_arm}_{seed}"][program]["rho"])
                        for seed in SEEDS
                    ]
                )
                value = float(aggregate["mean"])
                sd = float(aggregate["sd"])
                np.testing.assert_allclose(value, per_seed.mean(), rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(
                    sd, per_seed.std(ddof=0), rtol=0.0, atol=1e-12
                )
                for seed, stored in zip(SEEDS, per_seed):
                    row = _program_row(
                        program_table,
                        "cornea",
                        f"{data_arm}_{seed}",
                        program.removeprefix("program_"),
                    )
                    np.testing.assert_allclose(
                        stored, float(row["rho"]), rtol=0.0, atol=1e-12
                    )
                record = {
                    "arm": arm_key,
                    "program": program,
                    "value": value,
                    "low": value - sd,
                    "high": value + sd,
                    "kind": "sd",
                }
            if not record["low"] <= record["value"] <= record["high"]:
                raise AssertionError(f"interval does not bracket {data_arm}/{program}")
            panel_a.append(record)

    panel_b = []
    for data_arm, arm_key in PANEL_B_ARMS:
        profile = payload["cornea"]["arms"][data_arm]["program_stemness"][
            "pt_decile_profile"
        ]
        archived = deciles["cornea_deciles"][data_arm]
        for position, (value, stored) in enumerate(zip(profile, archived), start=1):
            np.testing.assert_allclose(value, stored, rtol=0.0, atol=1e-12)
            panel_b.append({"arm": arm_key, "decile": position, "value": float(value)})

    panel_c = []
    for dataset, _, _, _ in PANEL_C_DATASETS:
        tissue = "epidermis_wang2020" if dataset == "wang2020" else "epidermis_gse202352"
        for data_arm, arm_key, explicit_label in PANEL_C_ARMS:
            block = payload["epidermis"][dataset]["arms"][data_arm]["program_basal"]
            row = _program_row(program_table, tissue, data_arm, "basal")
            values = [
                float(block["rho"]),
                float(block["ci"]["ci_lower"]),
                float(block["ci"]["ci_upper"]),
            ]
            np.testing.assert_allclose(
                values,
                [float(row["rho"]), float(row["ci_lower"]), float(row["ci_upper"])],
                rtol=0.0,
                atol=1e-12,
            )
            if not values[1] <= values[0] <= values[2]:
                raise AssertionError(f"C interval: {dataset}/{data_arm}")
            label = S.ARMS[arm_key] if arm_key is not None else explicit_label
            panel_c.append(
                {
                    "dataset": dataset,
                    "arm": data_arm,
                    "label": label,
                    "value": values[0],
                    "low": values[1],
                    "high": values[2],
                }
            )

    panel_d = []
    for dataset, heading in PANEL_D_GROUPS:
        if dataset == "cornea":
            block = payload["cornea"]["expression_compartment"]["cycling"]
            stored = deciles["compartment_cornea"]["cycling"]
        else:
            block = payload["epidermis"][dataset]["expression_compartment"]["cycling"]
            stored = deciles["compartment_epidermis"]["cycling"]
        for class_name, value in block.items():
            np.testing.assert_allclose(
                value, stored[class_name], rtol=0.0, atol=1e-12
            )
            panel_d.append(
                {
                    "group": dataset,
                    "heading": heading,
                    "class": class_name,
                    "value": float(value),
                }
            )

    if n_boot <= 0:
        raise AssertionError("archived bootstrap count must be positive")
    return {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d}


def _pad_limits(values, fraction=PAD_FRACTION):
    """Return data limits with a proportional margin on both sides."""
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    return low - span * fraction, high + span * fraction


def draw_panel_a(ax, records):
    """Grouped bars for the cornea programs with arm-specific whiskers."""
    arms = [arm_key for _, arm_key in PANEL_A_ARMS]
    centers = np.arange(len(PROGRAMS), dtype=float) * (1.0 + BAR_GAP)
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * BAR_WIDTH

    for center, (program, title) in zip(centers, PROGRAMS):
        for offset, (_, arm_key) in zip(offsets, PANEL_A_ARMS):
            record = next(
                row
                for row in records
                if row["arm"] == arm_key and row["program"] == program
            )
            error = np.array(
                [
                    [record["value"] - record["low"]],
                    [record["high"] - record["value"]],
                ]
            )
            ax.bar(
                center + offset,
                record["value"],
                BAR_WIDTH,
                color=S.ARM_COLORS[arm_key],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_LW,
                hatch="//" if arm_key in PANEL_A_HATCH else "",
                zorder=3,
            )
            ax.errorbar(
                center + offset,
                record["value"],
                yerr=error,
                fmt="none",
                ecolor="#1A1A1A",
                elinewidth=ERROR_LW,
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LW,
                zorder=4,
            )
            if arm_key == "ordinal_cont":
                # Open marker: the continuity arm is a method, not a null, so
                # it must not use the hatch reserved for secondary/null arms.
                ax.plot(
                    center + offset,
                    record["value"],
                    marker="o",
                    markersize=POINT_MARKERSIZE,
                    markerfacecolor="none",
                    markeredgecolor=BAR_EDGE_COLOR,
                    markeredgewidth=0.8,
                    linestyle="none",
                    zorder=5,
                )

    ax.axhline(
        NULL_RHO,
        color=REF_COLOR,
        linestyle=REF_STYLE,
        linewidth=REF_LW,
        zorder=REF_ZORDER,
    )
    ax.set_xticks(centers)
    ax.set_xticklabels([title for _, title in PROGRAMS])
    ax.tick_params(axis="x", length=0)

    values = [record["low"] for record in records] + [
        record["high"] for record in records
    ]
    low, high = _pad_limits(values)
    ax.set_ylim(low, high)
    half_span = (len(arms) - 1) / 2 * BAR_WIDTH + BAR_WIDTH
    side_pad = BAR_GAP * 0.35
    ax.set_xlim(centers[0] - half_span - side_pad, centers[-1] + half_span + side_pad)

    ax.set_ylabel("Spearman \u03c1 with\npseudotime axis")
    ax.set_title("Cornea (GSE153515)")

    handles = [
        Patch(
            facecolor=S.ARM_COLORS[arm_key],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LW,
            hatch="//" if arm_key in PANEL_A_HATCH else "",
            label=S.ARMS[arm_key],
        )
        for _, arm_key in PANEL_A_ARMS
    ]
    # Overlay the open marker on the continuity arm's swatch so every entry is
    # a patch and the non-null arm still reads as open.
    handles[PANEL_A_ORDER["ordinal_cont"]] = (
        handles[PANEL_A_ORDER["ordinal_cont"]],
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=POINT_MARKERSIZE,
            markerfacecolor="none",
            markeredgecolor=BAR_EDGE_COLOR,
            markeredgewidth=0.8,
        ),
    )
    ax.legend(
        handles=handles,
        labels=[S.ARMS[arm_key] for _, arm_key in PANEL_A_ARMS],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1.8,
        handleheight=0.9,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=LEGEND_INSET_PT,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    ax.text(
        0.01,
        0.60,
        (
            f"Whiskers: frozen {CI_LEVEL_PCT}% CI;\n"
            f"adapted \u00b1 1 SD (n = {len(SEEDS)} seeds)"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#333333",
    )


def draw_panel_b(ax, records):
    """Stemness decile profiles for one archived checkpoint per arm."""
    styles = ("-", "--", ":")
    markers = ("o", "s", "^")
    handles = []
    for (data_index, (_, arm_key)) in enumerate(PANEL_B_ARMS):
        series = sorted(
            (row for row in records if row["arm"] == arm_key),
            key=lambda row: row["decile"],
        )
        x = [row["decile"] for row in series]
        y = [row["value"] for row in series]
        # The frozen and ordinal-head curves nearly coincide over deciles 8-10,
        # so the ordinal markers are nudged up by 2 px (at 300 dpi) while the
        # line keeps the exact archived value; both marks stay locatable.
        marker_offset = None
        if arm_key == "ordinal":
            marker_offset = ScaledTranslation(
                0.0, 2.0 / 300.0, ax.figure.dpi_scale_trans
            )
        ax.plot(
            x,
            y,
            color=S.ARM_COLORS[arm_key],
            linestyle=styles[data_index],
            marker="" if marker_offset is not None else markers[data_index],
            markersize=S.MARKER_PT,
            markeredgewidth=0.0,
            linewidth=S.LW_MAIN,
            zorder=3,
        )
        if marker_offset is not None:
            ax.plot(
                x,
                y,
                color=S.ARM_COLORS[arm_key],
                linestyle="none",
                marker=markers[data_index],
                markersize=S.MARKER_PT,
                markeredgewidth=0.0,
                transform=ax.transData + marker_offset,
                zorder=4,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color=S.ARM_COLORS[arm_key],
                linestyle=styles[data_index],
                marker=markers[data_index],
                markersize=S.MARKER_PT,
                markeredgewidth=0.0,
                linewidth=S.LW_MAIN,
                label=S.ARMS[arm_key],
            )
        )

    all_values = [row["value"] for row in records]
    all_deciles = [row["decile"] for row in records]
    ax.set_xlim(min(all_deciles) - 0.5, max(all_deciles) + 0.5)
    ax.set_ylim(*_pad_limits(all_values))
    ax.set_xticks(all_deciles)
    ax.set_xlabel("Pseudotime decile")
    ax.set_ylabel("Stemness program score")
    ax.set_title("Stemness program along pseudotime")
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        handlelength=2.0,
        labelspacing=0.35,
        borderaxespad=LEGEND_INSET_PT,
        frameon=False,
    )
    _reserve_legend_headroom(ax, legend, max(all_values))


def _reserve_legend_headroom(ax, legend, data_max: float) -> None:
    """Raise the y-limit until an upper-left legend clears the topmost data.

    Panel B's peak sits under the upper-left corner, so the requested anchor
    only works once the axis is given enough headroom (the plotted values and
    the axis meaning are unchanged).
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    # Fraction of the axes height hidden above the legend's lower edge,
    # including the anchor inset above the legend box itself.
    blocked = (axes_box.y1 - legend_box.y0) / axes_box.height
    if blocked >= 1.0:
        return
    y_low, y_high = ax.get_ylim()
    pad_data = 6.0 / 300.0 * fig.dpi * (y_high - y_low) / axes_box.height
    needed = (data_max + pad_data - blocked * y_low) / (1.0 - blocked)
    if needed > y_high:
        ax.set_ylim(y_low, needed)


def draw_panel_c(ax, records):
    """Basal-program rho with archived cell-bootstrap intervals, by dataset."""
    row_of = {
        data_arm: len(PANEL_C_ARMS) - 1 - index
        for index, (data_arm, _, _) in enumerate(PANEL_C_ARMS)
    }
    offsets = {dataset: (index - 0.5) * 0.32 for index, (dataset, _, _, _) in enumerate(PANEL_C_DATASETS)}
    handles = []

    for dataset, label, color, marker in PANEL_C_DATASETS:
        for data_arm, _, _ in PANEL_C_ARMS:
            record = next(
                row
                for row in records
                if row["dataset"] == dataset and row["arm"] == data_arm
            )
            ypos = row_of[data_arm] + offsets[dataset]
            ax.errorbar(
                record["value"],
                ypos,
                xerr=np.array(
                    [
                        [record["value"] - record["low"]],
                        [record["high"] - record["value"]],
                    ]
                ),
                fmt=marker,
                markersize=C_MARKER_POINT,
                color=color,
                markeredgecolor="white",
                markeredgewidth=C_MARKER_EDGE_LW,
                ecolor=color,
                elinewidth=C_RANGE_LW,
                capsize=C_CAPSIZE,
                capthick=C_RANGE_LW,
                zorder=3,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=marker,
                markersize=C_LEGEND_MARKER_POINT,
                color=color,
                markeredgecolor="white",
                markeredgewidth=C_MARKER_EDGE_LW,
                label=f"{label} ({CI_LEVEL_PCT}% CI)",
            )
        )

    ax.axvline(
        NULL_RHO,
        color=REF_COLOR,
        linestyle=REF_STYLE,
        linewidth=REF_LW,
        zorder=REF_ZORDER,
    )
    lows = [row["low"] for row in records]
    highs = [row["high"] for row in records]
    ax.set_xlim(*_pad_limits(lows + highs, fraction=0.08))
    # The legend now lives at the top, so the bottom no longer needs the old
    # legend band; keep a symmetrical half-row margin instead.
    ax.set_ylim(-0.55, len(PANEL_C_ARMS) - 1 + 0.55)
    ax.set_yticks([row_of[data_arm] for data_arm, _, _ in PANEL_C_ARMS])
    ax.set_yticklabels(
        [
            S.ARMS[arm_key] if arm_key is not None else explicit_label
            for _, arm_key, explicit_label in PANEL_C_ARMS
        ]
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Basal program Spearman \u03c1")
    ax.set_title("Epidermis \u2014 basal program")
    # Category ticks: 1.92 pt = 8 px at 300 dpi, the shared tick length.
    ax.tick_params(axis="y", length=1.92)
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=LEGEND_INSET_PT,
        frameon=False,
    )


def _fit_value_labels(fig, ax, texts):
    """Grow the x-limits until the value labels clear the left/right spines."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    pad_px = D_TEXT_CLEARANCE_MM * fig.dpi / S.MM_PER_INCH
    for _ in range(3):
        boxes = [text.get_window_extent(renderer) for text in texts]
        left_need = ax_box.x0 + pad_px - min(box.x0 for box in boxes)
        right_need = max(box.x1 for box in boxes) - (ax_box.x1 - pad_px)
        if left_need <= 0 and right_need <= 0:
            return
        x0, x1 = ax.get_xlim()
        per_px = (x1 - x0) / (ax_box.x1 - ax_box.x0)
        ax.set_xlim(x0 - max(left_need, 0.0) * per_px, x1 + max(right_need, 0.0) * per_px)
        fig.canvas.draw()


def draw_panel_d(fig, ax, records):
    """Cycling-program class means as neutral lollipops with padded labels."""
    positions = []
    headings = []
    cursor = 0.0
    for group_key, heading in PANEL_D_GROUPS:
        group_records = [record for record in records if record["group"] == group_key]
        if not group_records:
            raise AssertionError(f"panel D has no records for {group_key}")
        headings.append((heading, cursor + ROW_STEP * 0.85))
        for _ in group_records:
            positions.append(cursor)
            cursor -= ROW_STEP
        cursor -= GROUP_GAP

    values = [record["value"] for record in records]
    span = max(values) - min(values)
    offset = span * D_LABEL_OFFSET_FRACTION
    labels = []
    for record, ypos in zip(records, positions):
        ax.plot(
            [NULL_RHO, record["value"]],
            [ypos, ypos],
            color=D_STEM_COLOR,
            linewidth=D_STEM_LW,
            solid_capstyle="butt",
            zorder=2,
        )
        # Every row ends in the same cap, so the long TAC connector no longer
        # reads as the one uncapped rule in the panel.
        ax.plot(
            [record["value"], record["value"]],
            [ypos - D_CAP_HALF, ypos + D_CAP_HALF],
            color=D_STEM_COLOR,
            linewidth=D_STEM_LW,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            record["value"],
            ypos,
            marker="o",
            markersize=4.0,
            color=D_MARKER_COLOR,
            zorder=3,
        )
        positive = record["value"] >= NULL_RHO
        labels.append(
            ax.text(
                record["value"] + (offset if positive else -offset),
                ypos,
                f"{record['value']:.3f}",
                ha="left" if positive else "right",
                va="center",
                fontsize=7.0,
                color="#333333",
                zorder=4,
            )
        )

    ax.axvline(
        NULL_RHO,
        color=REF_COLOR,
        linestyle=REF_STYLE,
        linewidth=REF_LW,
        zorder=REF_ZORDER,
    )
    ax.set_xlim(*_pad_limits(values, fraction=0.08))
    ax.set_ylim(min(positions) - ROW_STEP, max(positions) + ROW_STEP * 1.5)
    ax.set_yticks(positions)
    ax.set_yticklabels([record["class"] for record in records])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("Cycling program score\n(mean of per-gene z-scores)")
    ax.set_title("Cycling score by expression class")

    _fit_value_labels(fig, ax, labels)
    for heading, ypos in headings:
        # A white backing box breaks the dashed zero line behind the group
        # headings instead of striking them through.
        ax.text(
            ax.get_xlim()[0],
            ypos,
            heading,
            ha="left",
            va="top",
            fontsize=S.TYPE["axis"],
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
            zorder=5,
        )


def compose_figure(records):
    """Compose the 2 x 2 panel grid on a fixed 178 mm canvas.

    Parameters
    ----------
    records : dict
        Panel-keyed records.

    Returns
    -------
    matplotlib.figure.Figure
        Assembled figure.
    """
    fig = plt.figure(figsize=S.figsize(*CANVAS_MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.0), height_ratios=(1.0, 1.0))
    axes = {
        "A": fig.add_subplot(grid[0, 0]),
        "B": fig.add_subplot(grid[0, 1]),
        "C": fig.add_subplot(grid[1, 0]),
        "D": fig.add_subplot(grid[1, 1]),
    }

    draw_panel_a(axes["A"], records["A"])
    draw_panel_b(axes["B"], records["B"])
    draw_panel_c(axes["C"], records["C"])
    draw_panel_d(fig, axes["D"], records["D"])

    _place_panel_letters(fig, axes)
    return fig


def _column_left_edge(fig, axes, keys) -> float:
    """Figure-fraction x of a column's leftmost ink (y title / tick labels)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lefts = []
    for key in keys:
        ax = axes[key]
        for artist in (ax.yaxis.label, *ax.get_yticklabels()):
            if artist.get_text():
                lefts.append(artist.get_window_extent(renderer).x0)
    x_px = max(min(lefts), S.LETTER_TOP_PX / 300.0 * fig.dpi)
    return float(fig.transFigure.inverted().transform((x_px, 0.0))[0])


def _place_panel_letters(fig, axes) -> None:
    """Anchor each column's letters to that column's leftmost ink.

    Both letters in a column share one x (the column's bbox left edge), so the
    letter never floats on its own axes-relative guess.
    """
    for keys, letters in (("AC", ("A", "C")), ("BD", ("B", "D"))):
        x_fig = _column_left_edge(fig, axes, keys)
        for letter in letters:
            ax = axes[letter]
            position = ax.get_position()
            x_axes = (x_fig - position.x0) / position.width
            S.panel_label(ax, letter, x=x_axes, y=1.02)


def main() -> int:
    """Render the figure and write the manifest entry.

    Returns
    -------
    int
        Process exit code.
    """
    S.set_style()
    payload, program_table, deciles, paths = load_payload()
    records = build_records(payload, program_table, deciles)

    S.check_value(
        "cornea cells",
        float(payload["cornea"]["arms"]["frozen"]["n_cells"]),
        float(D.MANUSCRIPT_VALUES["gse153515"]["n_cells"]),
    )

    fig = compose_figure(records)
    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs=paths,
    )
    plt.close(fig)
    print(
        f"[{OUT_NAME}] wrote {pdf_path} and {png_path} "
        f"({sum(len(v) for v in records.values())} plotted records)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
