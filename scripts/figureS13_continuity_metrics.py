"""Figure S13 - continuity metrics under task adaptation.

``core_claim``
    Adapting with an ordinal head preserves the differentiation gradient
    (rho ~ 0.83) while plain CE adaptation collapses it (rho ~ 0), at the cost
    of a higher calibration error; the continuity term added on top of the
    ordinal head keeps the gradient with a smaller calibration penalty.

``hero_panel``
    Panel B, the differentiation-gradient rho, carries the claim; panels A
    (LAA) and C (ECE) are the collateral readouts.

``stats_shown``
    Every bar is the three-seed mean with +/- 1 sample SD (ddof = 1) over
    seeds 42/123/456 on the 14,775 external-test epithelial cells. CE-naive
    ECE uses the audited corrected half-open binning
    (``artifacts/paper1_pipeline_audit/ece_corrected.json``); the ordinal-arm
    ECE comes from the ordinal-evaluation JSONs. No hypothesis test is shown.

``source_data``
    ``artifacts/paper1_tier_b_eval/results_0.0_{42,123,456}.json`` (CE-naive
    LAA / rho / ECE), ``artifacts/paper1_ordinal_eval/results_0.0_{42,123,456}
    .json`` and ``.../results_0.5_{42,123,456}.json`` (ordinal head and
    ordinal + continuity), ``artifacts/paper1_pipeline_audit/ece_corrected
    .json`` and ``artifacts/paper1_layerwise_rho/summary.json`` (the latter
    only as a layer-resolved cross-check of the CE-naive rho).

One figure, one producer: this script is the only writer of
``figures/paper1/figureS13_continuity_metrics.{pdf,png}``.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import paper1_figure_data as D  # noqa: E402
import paper1_figure_style as S  # noqa: E402

OUT_NAME = "figureS13_continuity_metrics"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 76.0)

SEEDS = (42, 123, 456)
# Canonical arm keys in draw order.
ARM_KEYS = ("ce_naive", "ordinal", "ordinal_cont")

# (source directory, file stem prefix) per arm.
ARM_SOURCES = {
    "ce_naive": ("paper1_tier_b_eval", "results_0.0"),
    "ordinal": ("paper1_ordinal_eval", "results_0.0"),
    "ordinal_cont": ("paper1_ordinal_eval", "results_0.5"),
}
METRIC_PANELS = (
    ("laa", "Layer adjacency accuracy", "Layer adjacency accuracy"),
    ("rho", "Differentiation-gradient Spearman \u03c1",
     "Differentiation gradient (\u03c1)"),
    ("ece", "Expected calibration error", "Calibration error (ECE)"),
)

BAR_WIDTH = 0.62
CAPSIZE = 2.0


def _load_arm_metrics():
    """Per-arm, per-seed LAA / rho / ECE and the audited CE-naive ECE."""
    results = {key: {} for key in ARM_KEYS}
    paths = {}
    ece_path = D.artifact_json("paper1_pipeline_audit", "ece_corrected")
    corrected = D.load_json(ece_path)
    corrected_ece = {}
    for entry in (corrected.get("ece_corrected") or {}).values():
        if entry.get("seed") is None:
            continue
        corrected_ece[(str(entry["arm"]).lower(), int(entry["seed"]))] = float(
            entry["new_ece_fixed"]
        )
    if not corrected_ece:
        raise KeyError(f"{ece_path}: no corrected ECE entries")

    for key in ARM_KEYS:
        directory, prefix = ARM_SOURCES[key]
        for seed in SEEDS:
            path = D.artifact_json(directory, f"{prefix}_{seed}")
            document = D.load_json(path)
            secondary = document["secondary_endpoints"]
            ece = (
                corrected_ece[("naive", seed)]
                if key == "ce_naive"
                else float(secondary["expected_calibration_error"])
            )
            results[key][seed] = {
                "laa": float(secondary["layer_adjacency_accuracy"]),
                "rho": float(secondary["marker_gradient_rank_correlation"]),
                "ece": ece,
            }
            paths[f"{directory}/{Path(path).name}"] = path
    paths["paper1_pipeline_audit/ece_corrected.json"] = ece_path
    return results, paths


def _check_layerwise(results, paths):
    """Cross-check the CE-naive rho against the layer-resolved summary.

    ``artifacts/paper1_layerwise_rho/summary.json`` holds only layer-resolved
    rho (plus the sanity gate); it contains no LAA or ECE, so it cannot be the
    sole source for this figure.  It is used here to confirm that the
    layer-12 CE-naive value equals the single-value rho drawn in panel B.
    """
    summary_path = D.artifact_json("paper1_layerwise_rho", "summary")
    summary = D.load_json(summary_path)
    gate = summary["sanity_gate"]
    S.check_value(
        "frozen layer-12 rho",
        float(gate["frozen_gf"]["layer12_rho"]),
        D.MANUSCRIPT_VALUES["frozen"]["rho"],
        tol=1e-4,
    )
    for seed in SEEDS:
        S.check_value(
            f"CE-naive seed {seed} layer-12 rho",
            float(gate[f"lam0.0_seed{seed}"]["layer12_rho"]),
            results["ce_naive"][seed]["rho"],
            tol=1e-4,
        )
    paths["paper1_layerwise_rho/summary.json"] = summary_path


def _check_manuscript(results) -> None:
    """Assert the manuscript-stated rho values."""
    for value, expected in zip(
        sorted(results["ce_naive"][seed]["rho"] for seed in SEEDS),
        sorted(D.MANUSCRIPT_VALUES["ce_naive"]["rho_seeds"]),
    ):
        S.check_value("CE-naive rho", value, expected)
    for value, expected in zip(
        sorted(results["ordinal"][seed]["rho"] for seed in SEEDS),
        sorted(D.MANUSCRIPT_VALUES["ordinal"]["rho_seeds"]),
    ):
        S.check_value("ordinal-head rho", value, expected)
    S.check_value(
        "ordinal-head mean rho",
        statistics.fmean(results["ordinal"][seed]["rho"] for seed in SEEDS),
        D.MANUSCRIPT_VALUES["ordinal"]["rho"],
    )
    S.check_value(
        "ordinal + continuity mean rho",
        statistics.fmean(results["ordinal_cont"][seed]["rho"] for seed in SEEDS),
        D.MANUSCRIPT_VALUES["ordinal_cont"]["rho"],
    )


def _panel(ax, results, metric):
    for index, key in enumerate(ARM_KEYS):
        values = [results[key][seed][metric] for seed in SEEDS]
        mean = statistics.fmean(values)
        error = statistics.stdev(values) if len(values) > 1 else 0.0
        ax.bar(
            index,
            mean,
            BAR_WIDTH,
            color=S.ARM_COLORS[key],
            edgecolor="white",
            linewidth=0.5,
            yerr=error,
            capsize=CAPSIZE,
            error_kw={"elinewidth": 0.7, "ecolor": "#333333"},
            zorder=3,
        )
    ax.set_xticks(range(len(ARM_KEYS)))
    ax.set_xticklabels([S.ARMS[key] for key in ARM_KEYS], rotation=18,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(ARM_KEYS) - 0.4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.5)


def _limits(results, metric):
    lows = []
    highs = []
    for key in ARM_KEYS:
        values = [results[key][seed][metric] for seed in SEEDS]
        mean = statistics.fmean(values)
        error = statistics.stdev(values) if len(values) > 1 else 0.0
        lows.append(mean - error)
        highs.append(mean + error)
    return min(lows), max(highs)


def main() -> int:
    """Render the figure and append its manifest entry."""
    S.set_style()
    results, paths = _load_arm_metrics()
    _check_layerwise(results, paths)
    _check_manuscript(results)

    fig, axes = plt.subplots(
        1, len(METRIC_PANELS), figsize=S.figsize(*CANVAS_MM),
        gridspec_kw={"wspace": 0.34},
    )
    for ax, (metric, ylabel, title) in zip(axes, METRIC_PANELS):
        _panel(ax, results, metric)
        low, high = _limits(results, metric)
        if metric == "laa":
            # Accuracy is bounded in [0, 1]; do not truncate near the ceiling.
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        elif metric == "rho":
            ax.axhline(0.0, color=S.ARM_COLORS["hvg_pca"], linestyle=":",
                       linewidth=0.7, zorder=0)
            ax.set_ylim(-0.2, 1.0)
            ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            ax.set_ylim(0.0, high * 1.15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    for ax, letter in zip(axes, "ABC"):
        S.panel_label(ax, letter)

    pdf_path, png_path = S.save_figure(
        fig, OUT_NAME, outdir=str(S.FIGURE_DIR), inputs=paths
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print(
        f"[{OUT_NAME}] layerwise summary holds rho only; LAA/ECE read from the "
        "tier-B and ordinal evaluation JSONs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
