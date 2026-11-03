"""Figure S11 - trajectory continuity across the adapted arms.

``core_claim``
    Diffusion pseudotime inferred from the external-test embedding preserves
    the frozen representation's differentiation ordering strongly for the
    frozen encoder and only weakly for CE-naive fine-tuning.

``hero_panel``
    Panel A, the per-arm Spearman rho against the fixed frozen reference with
    cell-bootstrap 95% CIs, carries the claim.

``stats_shown``
    Panel A: Spearman rho between scanpy DPT pseudotime and the frozen
    layer-centroid PCA reference, with percentile cell-bootstrap 95% CIs
    (``n_boot`` = 1000, seed 42; 14,775 cells). Panel B: median inferred
    pseudotime per frozen-reference decile (q = 10), min-max scaled within
    each arm; the band spans the three seeds. No hypothesis test is shown.

``source_data``
    ``artifacts/paper1_tier_b_eval/percell_{frozen_gf,naive_s42,naive_s123,
    naive_s456}.npz`` (``emb_ext`` and ``y_true_ext``); computation reused
    verbatim from ``scripts/analyze_trajectory_continuity.py``. The previously
    produced ``tables/paper1/table_trajectory_continuity.csv`` is read as a
    reproducibility cross-check.

One figure, one producer: this script is the only writer of
``figures/paper1/figureS11_trajectory_continuity.{pdf,png}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import analyze_trajectory_continuity as T  # noqa: E402
import paper1_figure_data as D  # noqa: E402
import paper1_figure_style as S  # noqa: E402

OUT_NAME = "figureS11_trajectory_continuity"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 92.0)

# Analysis parameters: identical to the locked run that produced the shipped
# table (scripts/analyze_trajectory_continuity.py defaults).
PREPROCESS = "pca"
N_PCS = 30
N_NEIGHBORS = 30
N_DCS = 10
N_BOOT = 1000
BOOT_SEED = 42
Q_DECILES = 10

ARM_SPECS = (
    ("frozen", "frozen_gf", None),
    ("ce_naive", "naive_s42", 42),
    ("ce_naive", "naive_s123", 123),
    ("ce_naive", "naive_s456", 456),
)
SEED_MARKERS = {42: "o", 123: "s", 456: "^"}

# Values printed in the SI caption (Figure S11), used as assertions only.
CAPTION_VALUES = {
    "frozen_rho": 0.850,
    "frozen_ci_lower": 0.842,
    "frozen_ci_upper": 0.858,
    "naive_rho_min": 0.323,
    "naive_rho_max": 0.348,
}

NULL_RHO = 0.0
LINE_WIDTH = 0.7
MEAN_WIDTH = 1.5
MARKER_SIZE = 4.0
BAND_ALPHA = 0.14
POINT_OFFSET = 0.11
GRID_COLOR = "#E6E6E6"
REFERENCE_COLOR = "#333333"

# Panel letters: a fixed slot measured from the top-left trim, so a letter can
# never be pushed off the page by the height of the axes under constrained
# layout. The layout reserves the top band (``AXES_TOP_FRACTION``) for them, and
# each letter's x is set clear of its own panel's y tick labels (the margin slot
# used by the single-column SI figures is occupied at the top here).
LETTER_TOP_MM = 2.0
LETTER_GAP_MM = 4.0
AXES_TOP_FRACTION = 0.90


def _load_archives():
    """Load the seven per-cell embeddings and their true labels."""
    archives = {}
    paths = {}
    for _arm_key, name, _seed in ARM_SPECS:
        path = D.percell(name)
        with np.load(path) as archive:
            archives[name] = (
                np.asarray(archive["emb_ext"]),
                np.asarray(archive["y_true_ext"], dtype=str),
            )
        paths[name] = path
    return archives, paths


def _compute():
    """Reuse the trajectory-continuity implementation verbatim."""
    archives, paths = _load_archives()
    frozen_embedding, frozen_labels = archives["frozen_gf"]
    reference, _axis, reference_rho = T.reference_pseudotime(
        frozen_embedding, frozen_labels
    )
    if not np.isclose(reference_rho, D.MANUSCRIPT_VALUES["frozen"]["rho"],
                      rtol=0.0, atol=5e-4):
        raise AssertionError(
            f"frozen reference rho {reference_rho} does not reproduce the "
            f"locked {D.MANUSCRIPT_VALUES['frozen']['rho']}"
        )
    root_index = T.select_root(
        frozen_embedding, frozen_labels, reference, "frozen_ref_min"
    )

    rows = []
    lines = {}
    for arm_key, name, seed in ARM_SPECS:
        embedding, _labels = archives[name]
        inferred, method, _info = T.run_trajectory_inference(
            embedding, root_index, PREPROCESS, N_PCS, N_NEIGHBORS, N_DCS
        )
        rho, ci_lower, ci_upper = T.bootstrap_spearman_ci(
            inferred, reference, N_BOOT, BOOT_SEED
        )
        stratified = T.stratified_consistency(reference, inferred, Q_DECILES)
        if not np.isfinite(stratified):
            raise AssertionError(f"{name}: stratified consistency is undefined")

        valid = np.isfinite(reference) & np.isfinite(inferred)
        bins = pd.qcut(reference[valid], q=Q_DECILES, labels=False,
                       duplicates="drop")
        median_line = (
            pd.Series(inferred[valid]).groupby(bins).median().to_numpy()
        )
        span = float(median_line.max() - median_line.min())
        if span <= 0:
            raise AssertionError(f"{name}: flat stratified curve")
        lines[name] = (median_line - median_line.min()) / span

        rows.append(
            {
                "arm": arm_key,
                "name": name,
                "seed": seed,
                "method": method,
                "rho": float(rho),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "stratified": float(stratified),
            }
        )
    return pd.DataFrame(rows), lines, paths


def _check_against_table(results: pd.DataFrame) -> tuple[Path | None, float]:
    """Compare recomputed rho/CI with the shipped analysis table."""
    table_path = D.ROOT / "tables" / "paper1" / "table_trajectory_continuity.csv"
    if not table_path.exists():
        return None, float("nan")
    table = D.load_csv(table_path)
    worst = 0.0
    key_map = {
        "frozen_gf": "frozen_sNone",
        "naive_s42": "naive_s42",
        "naive_s123": "naive_s123",
        "naive_s456": "naive_s456",
        "caft_s42": "caft_s42",
        "caft_s123": "caft_s123",
        "caft_s456": "caft_s456",
    }
    for row in results.itertuples(index=False):
        key = key_map[row.name]
        stored = table.loc[table["key"] == key]
        if len(stored) != 1:
            raise AssertionError(f"{table_path}: no unique row for {key}")
        for column, field in (
            ("rho_vs_frozen_ref", "rho"),
            ("ci_lower", "ci_lower"),
            ("ci_upper", "ci_upper"),
        ):
            difference = abs(
                float(stored[column].iloc[0]) - float(getattr(row, field))
            )
            worst = max(worst, difference)
    if worst > 5e-4:
        raise AssertionError(
            f"recomputed trajectory metrics differ from {table_path} by "
            f"{worst:.2e} (tolerance 5e-4)"
        )
    return table_path, worst


def _check_caption(results: pd.DataFrame) -> None:
    """Assert the caption's stated rho and CI values against the data."""
    frozen = results[results["arm"] == "frozen"].iloc[0]
    S.check_value("frozen rho", frozen["rho"], CAPTION_VALUES["frozen_rho"])
    S.check_value("frozen CI lower", frozen["ci_lower"],
                  CAPTION_VALUES["frozen_ci_lower"])
    S.check_value("frozen CI upper", frozen["ci_upper"],
                  CAPTION_VALUES["frozen_ci_upper"])

    naive = results[results["arm"] == "ce_naive"]["rho"]
    S.check_value("CE-naive rho min", naive.min(), CAPTION_VALUES["naive_rho_min"])
    S.check_value("CE-naive rho max", naive.max(), CAPTION_VALUES["naive_rho_max"])


def _panel_a(ax, results: pd.DataFrame) -> None:
    """Per-arm rho with cell-bootstrap 95% CI."""
    group_positions = {"frozen": 0.0, "ce_naive": 1.0}
    for arm_key, group in results.groupby("arm", sort=False):
        colour = S.ARM_COLORS[arm_key]
        seeded = group[group["seed"].notna()]
        seeds = seeded["seed"].astype(int).tolist()
        if not seeds:
            row = group.iloc[0]
            ax.errorbar(
                group_positions[arm_key],
                row["rho"],
                yerr=[[row["rho"] - row["ci_lower"]],
                      [row["ci_upper"] - row["rho"]]],
                fmt="D",
                markersize=MARKER_SIZE,
                color=colour,
                ecolor=colour,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.8,
                zorder=3,
            )
            continue
        for index, row in enumerate(seeded.itertuples(index=False)):
            offset = (index - (len(seeds) - 1) / 2) * POINT_OFFSET
            marker = SEED_MARKERS[int(row.seed)]
            ax.errorbar(
                group_positions[arm_key] + offset,
                row.rho,
                yerr=[[row.rho - row.ci_lower], [row.ci_upper - row.rho]],
                fmt=marker,
                markersize=MARKER_SIZE,
                color=colour,
                ecolor=colour,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.8,
                zorder=3,
            )
    ax.axhline(NULL_RHO, color=S.ARM_COLORS["hvg_pca"], linestyle=":",
               linewidth=0.7, zorder=0)
    ax.set_xticks(list(group_positions.values()))
    ax.set_xticklabels([S.ARMS[key] for key in group_positions])
    ax.set_xlim(-0.45, len(group_positions) - 1 + 0.45)
    low = min(0.0, float(results["ci_lower"].min()))
    high = float(results["ci_upper"].max())
    pad = (high - low) * 0.06
    ax.set_ylim(low - pad, high + pad)
    ax.set_ylabel("Spearman \u03c1 (DPT vs frozen reference)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)


def _panel_b(ax, results: pd.DataFrame, lines) -> None:
    """Stratified consistency: median inferred pseudotime per decile."""
    for arm_key, group in results.groupby("arm", sort=False):
        curves = np.vstack([lines[name] for name in group["name"]])
        x = np.arange(curves.shape[1])
        if arm_key == "frozen":
            ax.plot(x, curves[0], color=REFERENCE_COLOR, linewidth=MEAN_WIDTH,
                    marker="D", markersize=MARKER_SIZE, markeredgewidth=0.0,
                    zorder=4)
            continue
        colour = S.ARM_COLORS[arm_key]
        ax.fill_between(
            x,
            curves.min(axis=0),
            curves.max(axis=0),
            color=colour,
            alpha=BAND_ALPHA,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(x, curves.mean(axis=0), color=colour, linewidth=MEAN_WIDTH,
                marker="o", markersize=MARKER_SIZE, markeredgewidth=0.0,
                zorder=3)
    ax.set_xlabel("Frozen-reference pseudotime decile")
    ax.set_ylabel("Median inferred pseudotime (scaled)")
    ax.set_xticks(range(0, Q_DECILES, max(1, Q_DECILES // 5)))
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5)


def _place_panel_letters(fig, panels) -> None:
    """Put each panel letter in one canvas-anchored top slot.

    ``S.panel_label`` offsets the letter from its own axes, so a letter on a
    tall axes (the previous layout put the axes top at 0.93 of the canvas) lands
    above the trim and is clipped by the page. Here each letter is anchored to
    the canvas at ``LETTER_TOP_MM`` from the top trim, and its x is placed clear
    of that panel's y tick labels (both panels carry them, so the margin slot
    used by the single-column SI figures is not free at the top).
    """
    fig.canvas.draw()  # constrained layout must be resolved before reading positions
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    width_mm = fig.get_size_inches()[0] * S.MM_PER_INCH
    for ax, letter in panels:
        tick_lefts = [
            label.get_window_extent(renderer=renderer).x0
            for label in ax.get_yticklabels()
            if label.get_text()
        ]
        if tick_lefts:
            x_mm = min(tick_lefts) / dpi * S.MM_PER_INCH - LETTER_GAP_MM
        else:
            x_mm = ax.get_position().x0 * width_mm
        x_mm = max(x_mm, 1.0)
        S.panel_letter_mm(fig, letter, x_mm, LETTER_TOP_MM)


def _assert_letters_inside(fig) -> None:
    """Fail if a panel letter falls outside the page or onto a tick label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    letters = {"A", "B"}
    for artist in fig.texts:
        if artist.get_text() not in letters:
            continue
        box = artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            raise AssertionError(
                f"panel letter {artist.get_text()!r} is outside the canvas: "
                f"{box} vs {canvas}"
            )
        for ax in fig.axes:
            for label in (*ax.get_yticklabels(), *ax.get_xticklabels()):
                if not label.get_text():
                    continue
                other = label.get_window_extent(renderer=renderer)
                if box.overlaps(other):
                    raise AssertionError(
                        f"panel letter {artist.get_text()!r} overlaps the tick "
                        f"label {label.get_text()!r}: {box} vs {other}"
                    )


def main() -> int:
    """Render the figure and append its manifest entry."""
    S.set_style()
    results, lines, archive_paths = _compute()
    table_path, worst = _check_against_table(results)
    _check_caption(results)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=S.figsize(*CANVAS_MM), gridspec_kw={"wspace": 0.28}
    )
    _panel_a(ax_a, results)
    _panel_b(ax_b, results, lines)
    _place_panel_letters(fig, ((ax_a, "A"), (ax_b, "B")))

    handles = [
        Line2D([0], [0], color=REFERENCE_COLOR, linewidth=MEAN_WIDTH,
               marker="D", markersize=MARKER_SIZE, markeredgewidth=0.0,
               label=S.ARMS["frozen"]),
        Line2D([0], [0], color=S.ARM_COLORS["ce_naive"], linewidth=MEAN_WIDTH,
               marker="o", markersize=MARKER_SIZE, markeredgewidth=0.0,
               label=S.ARMS["ce_naive"]),
    ]
    handles += [
        Line2D([0], [0], color="#555555", linestyle="none", marker=marker,
               markersize=MARKER_SIZE, label=f"seed {seed}")
        for seed, marker in SEED_MARKERS.items()
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
    # Kept exactly as before the arm was dropped: the key still occupies two
    # rows, so the axes height is unchanged apart from the reserved top band
    # that holds the panel letters.
    fig.get_layout_engine().set(rect=(0.02, 0.14, 0.99, AXES_TOP_FRACTION))
    _assert_letters_inside(fig)

    inputs = {f"percell_{name}.npz": path for name, path in archive_paths.items()}
    if table_path is not None:
        inputs["table_trajectory_continuity.csv"] = table_path
    pdf_path, png_path = S.save_figure(
        fig, OUT_NAME, outdir=str(S.FIGURE_DIR), inputs=inputs
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    if table_path is not None:
        print(f"[{OUT_NAME}] recomputation agrees with {table_path.name} "
              f"to {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
