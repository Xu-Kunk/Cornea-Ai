#!/usr/bin/env python
"""Figure 5 - The gradient collapse depends on the evaluation domain.

core_claim
    For Geneformer arms the frozen-to-CE-naive loss of the differentiation
    gradient is a same-corpus effect: it is catastrophic on GSE153515, mild on
    the post-cutoff GSE299152, and reversed on cross-tissue epidermis.
hero_panel
    The single panel (former A and B merged): the two-panel legacy layout drew
    the same arm x domain means twice, once as bars and once as lines.
stats_shown
    Same-corpus adapted arms: mean with the full per-seed min-max range over
    seeds 42/123/456. Out-of-corpus adapted arms: mean +/-1 sample SD (ddof=1,
    n = 3). Frozen (all domains) and every cross-tissue arm are single runs and
    therefore carry no interval. Individual seeds are drawn as points where
    n = 3. No test is performed in this panel. Ordinal + continuity has no
    cross-tissue evaluation and therefore no bar in that group.
source_data
    Same-corpus: artifacts/paper1_tier_b_eval/results_frozen_gf.json,
    results_0.0_{42,123,456}.json (frozen, CE-naive);
    artifacts/paper1_ordinal_eval/results_0.0_{42,123,456}.json (ordinal head)
    and results_0.5_{42,123,456}.json (ordinal + continuity).
    Out-of-corpus: artifacts/paper1_tier_b_eval_gse299152/results_frozen_gf.json,
    results_0.0_{42,123,456}.json, results_ordinal_gf_0.0_{42,123,456}.json,
    results_ordinal_gf_0.5_{42,123,456}.json.
    Cross-tissue: artifacts/archive/paper1_epidermis_eval/
    epidermis_rho_results.json (field results[].rho_L12_ordinal) and
    embeddings.npz (evaluated cell count). Every plotted number is
    secondary_endpoints.marker_gradient_rank_correlation of the named JSON,
    except the cross-tissue rows, which use rho_L12_ordinal.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import paper1_figure_data as D  # noqa: E402
import paper1_figure_style as S  # noqa: E402

DOMAINS = [
    ("same", "Same-corpus\n(GSE153515)"),
    ("out", "Out-of-corpus\n(GSE299152)"),
    ("cross", "Cross-tissue epidermis\n(GSE147424)"),
]
ARMS_SHOWN = ["frozen", "ce_naive", "ordinal", "ordinal_cont"]
SEEDS = (42, 123, 456)

# Bar outline. A white stroke at this bar width leaves no edge on the white
# canvas, so adjacent bars merged into one block; the neutral ink below is the
# annotation ink already used in this figure, not an arm colour.
BAR_EDGE_COLOR = "#333333"
BAR_EDGE_LW = 0.4

# Values the figure set states for the arms plotted here but not carried by
# paper1_figure_data.MANUSCRIPT_VALUES (paper1.tex, Table "tab:domains" and
# Results "The severe Geneformer loss ... did not extend uniformly").
MANUSCRIPT_DOMAIN_VALUES = {
    "out": {
        "frozen": 0.707,
        "ce_naive_mean": 0.641,
        "ce_naive_sd": 0.030,
        "ordinal_mean": 0.350,
        "ordinal_sd": 0.099,
        "ordinal_cont_mean": 0.539,
        "ordinal_cont_sd": 0.007,
        "n_cells": 21374,
    },
    "cross": {
        "frozen": 0.239,
        "ce_naive": 0.455,
        "ordinal": 0.500,
        "n_cells": 18355,
    },}


def _rho(path):
    """Differentiation-gradient rho from a tier-B style evaluation JSON."""
    return float(D.load_json(path)["secondary_endpoints"]
                 ["marker_gradient_rank_correlation"])


def _provenance(path):
    return D.load_json(path)["provenance"]


def _epidermis_rows(path):
    """checkpoint -> rho_L12_ordinal from the archived epidermis evaluation."""
    doc = D.load_json(path)
    return {row["checkpoint"]: float(row["rho_L12_ordinal"])
            for row in doc["results"]}


def _load_domains():
    """Return per-domain arm samples, cell counts and the inputs actually used."""
    inputs = {}
    values = {}

    frozen_path = D.tier_b("frozen_gf")
    inputs["same/frozen"] = frozen_path
    values["same"] = {
        "frozen": [_rho(frozen_path)],
        "ce_naive": [_rho(D.tier_b(f"0.0_{seed}")) for seed in SEEDS],
        "ordinal": [_rho(D.artifact_json("paper1_ordinal_eval",
                                         f"results_0.0_{seed}"))
                    for seed in SEEDS],
        "ordinal_cont": [_rho(D.artifact_json(
            "paper1_ordinal_eval", f"results_0.5_{seed}")) for seed in SEEDS],
        "n_cells": int(_provenance(D.tier_b("0.0_42"))["n_eval_cells"]),
    }
    inputs["same/ce_naive_seeds"] = [D.tier_b(f"0.0_{seed}") for seed in SEEDS]
    inputs["same/ordinal_seeds"] = [
        D.artifact_json("paper1_ordinal_eval", f"results_0.0_{seed}")
        for seed in SEEDS
    ]
    inputs["same/ordinal_cont_seeds"] = [
        D.artifact_json("paper1_ordinal_eval", f"results_0.5_{seed}")
        for seed in SEEDS
    ]

    out_dir = ("paper1_tier_b_eval_gse299152",)
    out_frozen = D.artifact_json(*out_dir, "results_frozen_gf")
    inputs["out/frozen"] = out_frozen
    values["out"] = {
        "frozen": [_rho(out_frozen)],
        "ce_naive": [_rho(D.artifact_json(*out_dir, f"results_0.0_{seed}"))
                     for seed in SEEDS],
        "ordinal": [_rho(D.artifact_json(
            *out_dir, f"results_ordinal_gf_0.0_{seed}")) for seed in SEEDS],
        "ordinal_cont": [_rho(D.artifact_json(
            *out_dir, f"results_ordinal_gf_0.5_{seed}")) for seed in SEEDS],
        "n_cells": int(_provenance(D.artifact_json(*out_dir, "results_0.0_42"))
                       ["n_eval_cells"]),
    }
    inputs["out/ce_naive_seeds"] = [
        D.artifact_json(*out_dir, f"results_0.0_{seed}") for seed in SEEDS
    ]
    inputs["out/ordinal_seeds"] = [
        D.artifact_json(*out_dir, f"results_ordinal_gf_0.0_{seed}")
        for seed in SEEDS
    ]
    inputs["out/ordinal_cont_seeds"] = [
        D.artifact_json(*out_dir, f"results_ordinal_gf_0.5_{seed}")
        for seed in SEEDS
    ]

    epidermis_path = D.artifact_json("archive", "paper1_epidermis_eval",
                                     "epidermis_rho_results")
    embeddings_path = (D.ARTIFACTS / "archive" / "paper1_epidermis_eval"
                       / "embeddings.npz")
    rows = _epidermis_rows(epidermis_path)
    embeddings = D.load_npz(embeddings_path)
    try:
        n_epidermis = int(embeddings["frozen"].shape[0])
    finally:
        embeddings.close()
    inputs["cross/epidermis_rho"] = epidermis_path
    inputs["cross/epidermis_embeddings"] = embeddings_path
    values["cross"] = {
        "frozen": [rows["frozen"]],
        "ce_naive": [rows["ce_naive"]],
        "ordinal": [rows["corns_ordinal"]],
        "n_cells": n_epidermis,
    }
    return values, inputs


def _check(values):
    """Assert every plotted value against the figure set's stated number."""
    frozen = D.MANUSCRIPT_VALUES["frozen"]["rho"]
    S.check_value("frozen same-corpus rho", values["same"]["frozen"][0], frozen)
    S.check_value("frozen same-corpus n", values["same"]["n_cells"],
                  D.MANUSCRIPT_VALUES["gse153515"]["n_cells"])

    for seed_value, expected in zip(
        sorted(values["same"]["ce_naive"]),
        sorted(D.MANUSCRIPT_VALUES["ce_naive"]["rho_seeds"]),
    ):
        S.check_value("CE-naive same-corpus seed rho", seed_value, expected)
    # MANUSCRIPT_VALUES lists the ordinal seeds in a different seed order than
    # 42/123/456, so compare the multisets rather than the positions.
    for seed_value, expected in zip(
        sorted(values["same"]["ordinal"]),
        sorted(D.MANUSCRIPT_VALUES["ordinal"]["rho_seeds"]),
    ):
        S.check_value("ordinal same-corpus seed rho", seed_value, expected)
    S.check_value("ordinal same-corpus mean rho",
                  statistics.fmean(values["same"]["ordinal"]),
                  D.MANUSCRIPT_VALUES["ordinal"]["rho"])
    S.check_value("ordinal + continuity same-corpus mean rho",
                  statistics.fmean(values["same"]["ordinal_cont"]),
                  D.MANUSCRIPT_VALUES["ordinal_cont"]["rho"])

    expected_out = MANUSCRIPT_DOMAIN_VALUES["out"]
    S.check_value("frozen out-of-corpus rho", values["out"]["frozen"][0],
                  expected_out["frozen"])
    S.check_value("CE-naive out-of-corpus mean rho",
                  statistics.fmean(values["out"]["ce_naive"]),
                  expected_out["ce_naive_mean"])
    S.check_value("CE-naive out-of-corpus SD",
                  statistics.stdev(values["out"]["ce_naive"]),
                  expected_out["ce_naive_sd"])
    S.check_value("ordinal out-of-corpus mean rho",
                  statistics.fmean(values["out"]["ordinal"]),
                  expected_out["ordinal_mean"])
    S.check_value("ordinal out-of-corpus SD",
                  statistics.stdev(values["out"]["ordinal"]),
                  expected_out["ordinal_sd"])
    S.check_value("ordinal + continuity out-of-corpus mean rho",
                  statistics.fmean(values["out"]["ordinal_cont"]),
                  expected_out["ordinal_cont_mean"])
    S.check_value("ordinal + continuity out-of-corpus SD",
                  statistics.stdev(values["out"]["ordinal_cont"]),
                  expected_out["ordinal_cont_sd"])
    S.check_value("out-of-corpus n", values["out"]["n_cells"],
                  expected_out["n_cells"])

    expected_cross = MANUSCRIPT_DOMAIN_VALUES["cross"]
    for arm in ("frozen", "ce_naive", "ordinal"):
        S.check_value(f"{arm} cross-tissue rho", values["cross"][arm][0],
                      expected_cross[arm])
    S.check_value("cross-tissue n", values["cross"]["n_cells"],
                  expected_cross["n_cells"])


def _interval(samples, kind):
    """Mean and (lower, upper) whisker lengths for the stated error convention."""
    mean = statistics.fmean(samples)
    if kind == "single":
        return mean, None
    if kind == "range":
        return mean, (mean - min(samples), max(samples) - mean)
    if kind == "sd":
        sd = statistics.stdev(samples)
        return mean, (sd, sd)
    raise ValueError(f"unknown interval convention {kind!r}")


def _flatten_inputs(inputs):
    """Expand per-seed input lists into one manifest entry per file."""
    flat = {}
    for key, value in inputs.items():
        if isinstance(value, list):
            for index, item in enumerate(value):
                flat[f"{key}_{index}"] = item
        else:
            flat[key] = value
    return flat


def main() -> None:
    S.set_style()
    values, inputs = _load_domains()
    _check(values)

    fig = plt.figure(figsize=S.figsize(178.0, 84.0))
    # Reserved top strip for the panel letter (rect is left, bottom, width,
    # height for the constrained engine).
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.945))
    ax = fig.add_subplot()

    x = np.arange(len(DOMAINS))
    width = 0.19
    offsets = {arm: (index - (len(ARMS_SHOWN) - 1) / 2) * width
               for index, arm in enumerate(ARMS_SHOWN)}
    conventions = {"same": "range", "out": "sd", "cross": "single"}

    for domain_index, (domain_key, _) in enumerate(DOMAINS):
        for arm in ARMS_SHOWN:
            # The offset grid is arm-indexed, so an arm that was not evaluated
            # still keeps its slot: the remaining bar centres stay on the same
            # x in every group instead of the cluster shifting left.
            position = domain_index + offsets[arm]
            if arm not in values[domain_key]:
                ax.text(
                    position,
                    0.005,
                    "n.e.",
                    ha="center",
                    va="bottom",
                    fontsize=S.TYPE["tick"],
                    color="#333333",
                )
                continue
            samples = values[domain_key][arm]
            kind = "single" if len(samples) == 1 else conventions[domain_key]
            mean, whiskers = _interval(samples, kind)
            ax.bar(
                position,
                mean,
                width,
                color=S.ARM_COLORS[arm],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_LW,
                yerr=None if whiskers is None else [[whiskers[0]],
                                                    [whiskers[1]]],
                capsize=2.0,
                error_kw={"elinewidth": 0.7, "ecolor": "#333333",
                          "zorder": 2.5},
                zorder=2,
            )
            if len(samples) > 1:
                jitter = np.linspace(-0.035, 0.035, len(samples))
                ax.scatter(
                    position + jitter,
                    samples,
                    s=S.MARKER_PT ** 2,
                    facecolor="white",
                    edgecolor="#333333",
                    linewidth=0.5,
                    zorder=3,
                )

    ax.axhline(0.0, color="#555555", linewidth=S.RULE_LW, linestyle=":",
               zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DOMAINS])
    ax.set_ylabel("Differentiation-gradient Spearman \u03c1")
    # The lowest same-corpus CE-naive seed is -0.042; -0.055 keeps that marker
    # inside the axis while removing the empty negative band of the old -0.12.
    ax.set_ylim(-0.055, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.5, len(DOMAINS) - 0.5)

    # The same-corpus CE-naive mean is -0.008: the bar itself is invisible at
    # this scale, so label its value horizontally, clear of the bar's whisker
    # and the seed marker drawn at the top of that whisker.
    ce_samples = values["same"]["ce_naive"]
    ce_value = statistics.fmean(ce_samples)
    ax.annotate(
        f"\u2212{abs(ce_value):.3f}" if ce_value < 0 else f"{ce_value:.3f}",
        xy=(offsets["ce_naive"], max([ce_value, *ce_samples])),
        xytext=(0.0, 7.0),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=S.TYPE["tick"],
        color="#222222",
    )

    handles = [
        Patch(facecolor=S.ARM_COLORS[arm], edgecolor=BAR_EDGE_COLOR,
              linewidth=BAR_EDGE_LW, label=S.ARMS[arm])
        for arm in ARMS_SHOWN
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, frameon=False, columnspacing=2.2, handlelength=1.2,
              handletextpad=0.4)
    S.panel_label(ax, "A", x=-0.02, y=1.06)

    S.save_figure(fig, "figure5_domain_boundary",
                  inputs=_flatten_inputs(inputs))
    plt.close(fig)


if __name__ == "__main__":
    main()
