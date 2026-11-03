"""Figure 6 — cross-architecture differentiation-gradient spectrum.

``core_claim``
    Classification fine-tuning collapses the frozen differentiation gradient in
    every evaluated architecture, its seed-to-seed determinism is
    architecture-dependent, and an ordinal head restores the gradient.

``hero_panel``
    Panel A: the frozen -> naive -> ordinal sequence, drawn per architecture on
    one shared rho axis.

``stats_shown``
    Panel A: individual runs are drawn as points and the run mean as a tick.
    Panel B: one point per seed at its archived validation-rho peak, joined to
    the epoch at which that run's checkpoint was selected on validation
    macro-F1. No interval, test or interpolated epoch is drawn, and no frozen
    reference appears in panel B: the archived frozen value is a test-donor
    rho, so it is not comparable with a validation rho.
    Panel C: CellPLM cross-entropy rho and macro-F1 at five training budgets,
    with the ordinal+LoRA 15-epoch and convergence runs, on a log epoch axis.
    The two readouts are drawn on two stacked axes that share that epoch axis,
    not on one frame with two y scales: the macro-F1 knee and the rho knee sit
    roughly a factor of four apart in time. Means and sample SDs are taken over
    the three archived seeds; each CE seed is also drawn at its own
    ``epochs_actual``. The vertical guide marks the first budget whose mean
    macro-F1 has reached the plateau, computed from the archived numbers.

``source_data``
    ``artifacts/paper1_vanzyl/run6/convergence/checkpoints/naive_seed{42,123,
    456,789,2024}/args.json`` (panel B: ``results.best_val_rho``,
    ``results.best_val_rho_epoch``, ``results.best_val_f1``,
    ``results.best_val_f1_epoch``),
    ``artifacts/paper1_vanzyl/convergence/final_test_eval.json`` (scGPT-human
    frozen / best-val / final-epoch),
    ``artifacts/paper1_vanzyl/run6_ext/eval/naive_seed{789,2024}.json``
    (final-epoch seeds 789/2024),
    ``artifacts/paper1_vanzyl/run6/eval/ordinal_seed*.json`` (scGPT ordinal),
    ``artifacts/paper1_vanzyl/corn_lora/summary.json`` (scGPT ordinal+LoRA),
    ``artifacts/paper1_vanzyl/gf_official_1epoch/eval_armb_seed*.json`` (GF
    one-epoch) and ``gf_official_1epoch_summary.json`` (GF three-epoch),
    ``results/paper1_ordinal_eval/results_0.0_*.json`` (GF ordinal head),
    the archived CellPLM evaluation JSONs (frozen / CE / ordinal) and
    ``artifacts/paper1_vanzyl/cellplm_lora/seed42/seed42.json`` (CellPLM LoRA).

    Panel C: ``results/paper1_vanzyl/cellplm_e{3,15,30,60}/seed*/seed*.json``
    (CE budgets), ``results/paper1_vanzyl/cellplm/seed*/seed*.json`` (CE
    convergence), ``results/paper1_vanzyl/cellplm_corn_lora/seed*/seed*.json``
    and ``cellplm_corn_lora_conv/seed*/seed*.json`` (ordinal+LoRA) and
    ``results/paper1_vanzyl/cellplm/corn/seed*/seed*.json`` (ordinal head).

Metric conventions are those of the source files: the Geneformer rows use the
raw Spearman rho stated by the manuscript, the scGPT-human and CellPLM rows use
the L2-normed rho that those artifacts report as primary. Both are Spearman rho
of the same stage-order endpoint.

Panel C carries no second y axis: macro-F1 is drawn in the document ink on its
own stacked axis, so the muted CE blue that used to veer towards the ordinal +
continuity sky blue (#56B4E9, Figures 1, 3, 4 and 5) is gone.

One figure, one producer: this script is the only writer of
``figures/paper1/figure6_cross_architecture_spectrum.{pdf,png}``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import paper1_figure_data as D  # noqa: E402, N812
import paper1_figure_style as S  # noqa: E402, N812
from matplotlib.lines import Line2D  # noqa: E402

OUT_NAME = "figure6_cross_architecture_spectrum"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 112.0)

# Style constants only (geometry, ink); no data values.
MARKER_SIZE = S.MARKER_PT
MEAN_HALF_HEIGHT = 0.30
MEAN_LINE_WIDTH = 1.2
ROW_SPACING = 1.0
POINT_JITTER = 0.20
SEED_MARKERS = ("o", "s", "^", "D", "X")

# Validation-rho peak epochs the manuscript states (paper1 Table S7, column
# "Val rho peak (epoch)", rows 42/123/456). Asserted, never plotted as literals.
MANUSCRIPT_PEAK_EPOCHS = {42: 3, 123: 3, 456: 6}

ROOT = Path(__file__).resolve().parents[1]
VANZYL = ROOT / "artifacts" / "paper1_vanzyl"
# The CellPLM evaluation JSONs were produced on the remote node and are archived
# locally only inside the revision evidence mirror. Read-only.
CELLPLM_MIRROR = (
    ROOT
    / "docs"
    / "manuscript"
    / "patterns"
    / "archive"
    / "revision_2026-10-16"
    / "evidence"
    / "remote_json"
    / "home"
    / "summer"
    / "cellplm_vanzyl"
    / "artifacts"
    / "paper1_vanzyl"
    / "cellplm"
)


# ---------------------------------------------------------------------------
# Panel C: CellPLM cross-entropy dose-response
# ---------------------------------------------------------------------------
# One directory per training budget; ``None`` marks convergence, where the x
# value is each seed's own archived ``epochs_actual`` rather than a preset.
CELLPLM_RESULTS = ROOT / "results" / "paper1_vanzyl"
DOSE_SEEDS = (42, 123, 456)
CE_BUDGETS = (
    ("ce_3", "cellplm_e3", 3),
    ("ce_15", "cellplm_e15", 15),
    ("ce_30", "cellplm_e30", 30),
    ("ce_60", "cellplm_e60", 60),
    ("ce_convergence", "cellplm", None),
)
LORA_BUDGETS = (
    ("lora_15", "cellplm_corn_lora", 15),
    ("lora_convergence", "cellplm_corn_lora_conv", None),
)
ORDINAL_BUDGET = ("ordinal_convergence", "cellplm/corn", None)
# Frozen CellPLM reference on the test donor (the archived frozen baseline of
# the seed-42 projection); the manuscript states this value.
PANEL_C_FROZEN_RHO = 0.8652
# Epochs that carry a tick; the log spacing makes the plain labels sufficient.
PANEL_C_XTICKS = (3, 15, 30, 60, 150, 450)
PANEL_C_XLIM = (2.3, 620.0)
# The 60-epoch CE budget has mean rho 0.122 +/- 0.291 (sample SD), so the lower
# whisker reaches -0.169; the floor leaves headroom above it. The ceiling
# clears the ordinal + LoRA error-bar cap (0.925 + 0.003).
PANEL_C_RHO_LIM = (-0.25, 0.95)
# Panel C is drawn as two stacked axes that share the log-epoch x axis, not as
# one axis with two independently scaled y axes: a macro-F1 knee near epoch 15
# and a rho collapse near epoch 60 are roughly a factor of four apart in time,
# and a twin axis makes unrelated knees look coupled. Macro-F1 therefore gets
# its own panel, its own scale and a neutral ink, and the ordinal + continuity
# sky blue stays reserved for the rho series that is actually blue.
PANEL_C_F1_COLOR = "#1A1A1A"
PANEL_C_F1_PAD = 0.14  # fraction of the F1 span added above and below
# Tick sets for the two stacked axes. Both stay inside the computed limits, so
# the reader sees the plateau of the task score and the collapse of rho on the
# same grid spacing.
PANEL_C_F1_TICKS = (0.80, 0.85, 0.90, 0.95)
PANEL_C_RHO_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8)
# A budget counts as plateaued when its three-seed mean macro-F1 sits within
# the largest per-seed SD of the top budgets; that epoch is drawn as a shared
# structural guide in both stacked panels.
PANEL_C_PLATEAU_BAND = 0.02
# Per-seed marks are nudged left of the budget mean so a seed glyph never lands
# on the mean's whisker cap; the mean x itself is unchanged.
SEED_X_OFFSET = 0.90


def _first_existing(candidates, description):
    """Return the first existing path, else raise listing every candidate."""
    for path in candidates:
        if Path(path).is_file():
            return Path(path)
    listing = "\n".join(f"  - {Path(p)}" for p in candidates)
    raise FileNotFoundError(f"{description} not found; tried:\n{listing}")


def cellplm_json(*parts):
    """CellPLM evaluation JSON: local artifact first, evidence mirror second."""
    return _first_existing(
        [VANZYL / "cellplm" / Path(*parts), CELLPLM_MIRROR / Path(*parts)],
        f"CellPLM evaluation JSON {'/'.join(parts)}",
    )


def _sorted_seed_values(values: dict, seeds) -> list[float]:
    return [float(values[f"s{seed}"]) for seed in seeds]


def load_values() -> dict:
    """Read every plotted value and record ``(value, file:field)`` provenance.

    Returns
    -------
    dict
        ``{"rows": [...], "panel_b": {...}, "sources": {label: (path, field)}}``.
    """
    sources: dict[str, tuple[Path, str]] = {}
    rows: list[dict] = []

    def add_row(label, arm, values, path, field, seeds=None):
        if not values:
            raise AssertionError(f"{label}: no values")
        rows.append(
            {
                "label": label,
                "display": re.sub(r"\s*\(n = \d+\)$", "", label),
                "arm": arm,
                "values": np.asarray(values, dtype=float),
                "seeds": None if seeds is None else list(seeds),
            }
        )
        if path is not None:
            sources[label] = (Path(path), field)

    # ---- Geneformer (same-corpus GSE153515) --------------------------------
    gf_frozen = float(D.MANUSCRIPT_VALUES["frozen"]["rho"])
    add_row(
        f"GF {S.ARMS['frozen']} (n = 1)",
        "frozen",
        [gf_frozen],
        None,
        "paper1_figure_data.MANUSCRIPT_VALUES['frozen']['rho']",
    )

    one_epoch = VANZYL / "gf_official_1epoch"
    gf_one = []
    for seed in (42, 123, 456):
        path = one_epoch / f"eval_armb_seed{seed}.json"
        gf_one.append(float(D.load_json(path)["rho_raw"]))
    add_row(
        f"GF {S.ARMS['ce_naive']}, 1 epoch (n = 3)",
        "ce_naive",
        gf_one,
        one_epoch / "eval_armb_seed42.json",
        "eval_armb_seed{42,123,456}.json:rho_raw (mean of three)",
        seeds=(42, 123, 456),
    )

    gf_summary_path = one_epoch / "gf_official_1epoch_summary.json"
    gf_summary = D.load_json(gf_summary_path)
    gf_three = [
        float(gf_summary["per_seed"][str(seed)]["reference_3epoch"]["rho_raw"])
        for seed in (42, 123, 456)
    ]
    add_row(
        f"GF {S.ARMS['ce_naive']}, 3 epochs (n = 3)",
        "ce_naive",
        gf_three,
        gf_summary_path,
        "per_seed.{42,123,456}.reference_3epoch.rho_raw",
        seeds=(42, 123, 456),
    )

    gf_ordinal = []
    for seed in (42, 123, 456):
        path = D.result_json("paper1_ordinal_eval", f"results_0.0_{seed}")
        doc = D.load_json(path)
        gf_ordinal.append(float(doc["secondary_endpoints"]["marker_gradient_rank_correlation"]))
    add_row(
        f"GF {S.ARMS['ordinal']} (n = 3)",
        "ordinal",
        gf_ordinal,
        D.result_json("paper1_ordinal_eval", "results_0.0_42"),
        "results_0.0_{42,123,456}.json:secondary_endpoints.marker_gradient_rank_correlation",
        seeds=(42, 123, 456),
    )

    # ---- scGPT-human (same-corpus van Zyl, test donor Hu0216) -------------
    conv_path = VANZYL / "convergence" / "final_test_eval.json"
    conv = D.load_json(conv_path)
    scgpt_frozen = float(conv["frozen_baseline"]["rho"]["rho_normed"])
    add_row(
        f"scGPT {S.ARMS['frozen']} (n = 1)",
        "frozen",
        [scgpt_frozen],
        conv_path,
        "frozen_baseline.rho.rho_normed",
    )

    best_val = conv["aggregate"]["best_val"]["rho_normed"]["per_seed"]
    add_row(
        f"scGPT {S.ARMS['ce_naive']}, best-val (n = 3)",
        "ce_naive",
        _sorted_seed_values(best_val, (42, 123, 456)),
        conv_path,
        "aggregate.best_val.rho_normed.per_seed",
        seeds=(42, 123, 456),
    )

    final = conv["aggregate"]["final_epoch"]["rho_normed"]["per_seed"]
    scgpt_final = _sorted_seed_values(final, (42, 123, 456))
    ext_paths = {}
    for seed in (789, 2024):
        path = VANZYL / "run6_ext" / "eval" / f"naive_seed{seed}.json"
        ext_paths[seed] = path
        scgpt_final.append(float(D.load_json(path)["rho_normed"]))
    add_row(
        f"scGPT {S.ARMS['ce_naive']}, final epoch (n = 5)",
        "ce_naive",
        scgpt_final,
        conv_path,
        "aggregate.final_epoch.rho_normed.per_seed (42/123/456) + "
        "run6_ext/eval/naive_seed{789,2024}.json:rho_normed",
        seeds=(42, 123, 456, 789, 2024),
    )

    scgpt_ordinal = []
    for seed in (42, 123, 456):
        path = VANZYL / "run6" / "eval" / f"ordinal_seed{seed}.json"
        scgpt_ordinal.append(float(D.load_json(path)["rho_normed"]))
    add_row(
        f"scGPT {S.ARMS['ordinal']} (n = 3)",
        "ordinal",
        scgpt_ordinal,
        VANZYL / "run6" / "eval" / "ordinal_seed42.json",
        "ordinal_seed{42,123,456}.json:rho_normed",
        seeds=(42, 123, 456),
    )

    lora_path = VANZYL / "corn_lora" / "summary.json"
    lora = D.load_json(lora_path)
    per_seed = lora["aggregate"]["best_val"]["rho_normed"]["per_seed"]
    add_row(
        f"scGPT {S.ARMS['ordinal_lora']} (n = 3)",
        "ordinal_lora",
        _sorted_seed_values(per_seed, (42, 123, 456)),
        lora_path,
        "aggregate.best_val.rho_normed.per_seed",
        seeds=(42, 123, 456),
    )

    # ---- CellPLM (van Zyl, test donor Hu0216) -----------------------------
    cp_frozen_path = cellplm_json("seed42", "seed42.json")
    cp_frozen = float(D.load_json(cp_frozen_path)["frozen_baseline"]["rho_normed"])
    add_row(
        f"CellPLM {S.ARMS['frozen']} (n = 1)",
        "frozen",
        [cp_frozen],
        cp_frozen_path,
        "frozen_baseline.rho_normed (seed 42 reference)",
    )

    cp_ce_path = cellplm_json("summary_vanzyl_cellplm.json")
    cp_ce = D.load_json(cp_ce_path)["per_seed"]
    add_row(
        f"CellPLM {S.ARMS['ce_naive']} (n = 3)",
        "ce_naive",
        [float(cp_ce[str(seed)]["rho_normed"]) for seed in (42, 123, 456)],
        cp_ce_path,
        "per_seed.{42,123,456}.rho_normed",
        seeds=(42, 123, 456),
    )

    cp_ord_path = cellplm_json("corn", "summary_vanzyl_cellplm_corn.json")
    cp_ord = D.load_json(cp_ord_path)["per_seed"]
    add_row(
        f"CellPLM {S.ARMS['ordinal']} (n = 3)",
        "ordinal",
        [float(cp_ord[str(seed)]["rho_normed"]) for seed in (42, 123, 456)],
        cp_ord_path,
        "per_seed.{42,123,456}.rho_normed",
        seeds=(42, 123, 456),
    )

    cp_lora_path = VANZYL / "cellplm_lora" / "seed42" / "seed42.json"
    cp_lora = float(D.load_json(cp_lora_path)["eval"]["rho_normed"])
    add_row(
        f"CellPLM {S.ARMS['lora']} (n = 1)",
        "lora",
        [cp_lora],
        cp_lora_path,
        "eval.rho_normed",
    )

    if sources.get(next(iter(sources)))[0] is None:  # pragma: no cover
        raise AssertionError("provenance table missing")

    # Provenance for the panel-A scGPT final-epoch seeds 789/2024.
    sources["panel_a_scgpt_final_789"] = (
        VANZYL / "run6_ext" / "eval" / "naive_seed789.json",
        "naive_seed{789,2024}.json:rho_normed",
    )

    # ---- panel B: archived per-seed validation-rho peaks -------------------
    peak_seeds = (42, 123, 456, 789, 2024)
    peaks = {}
    total_epochs = None
    for seed in peak_seeds:
        path = VANZYL / "run6" / "convergence" / "checkpoints" / f"naive_seed{seed}" / "args.json"
        document = D.load_json(path)
        results = document["results"]
        epochs = int(document["epochs"])
        if total_epochs is None:
            total_epochs = epochs
        elif epochs != total_epochs:
            raise AssertionError(f"seed {seed}: training budget {epochs} != {total_epochs}")
        record = {
            "rho": float(results["best_val_rho"]),
            "rho_epoch": int(results["best_val_rho_epoch"]),
            "f1": float(results["best_val_f1"]),
            "f1_epoch": int(results["best_val_f1_epoch"]),
        }
        for field in ("rho_epoch", "f1_epoch"):
            if not 1 <= record[field] <= epochs:
                raise AssertionError(f"seed {seed}: {field}={record[field]} outside 1..{epochs}")
        if not -1.0 <= record["rho"] <= 1.0:
            raise AssertionError(f"seed {seed}: validation rho outside [-1, 1]")
        peaks[seed] = record
        sources[f"validation_peak_seed{seed}"] = (
            path,
            "results.best_val_rho / results.best_val_rho_epoch / "
            "results.best_val_f1 / results.best_val_f1_epoch",
        )

    panel_b = {
        "seeds": peak_seeds,
        "peaks": peaks,
        "epochs": total_epochs,
    }
    return {"rows": rows, "panel_b": panel_b, "sources": sources}


def _dose_runs(directory, budget, label, sources):
    """Read one budget's three seed JSONs and return ``(epochs, rho, f1)``."""
    runs = []
    for seed in DOSE_SEEDS:
        path = CELLPLM_RESULTS / directory / f"seed{seed}" / f"seed{seed}.json"
        # No mirror, no fallback: a missing budget file is a hard error.
        document = D.load_json(path)
        if int(document["seed"]) != seed:
            raise AssertionError(f"{path}: seed field {document['seed']!r} != {seed}")
        epochs = int(document["training"]["epochs_actual"])
        if budget is not None and epochs != budget:
            raise AssertionError(f"{path}: epochs_actual {epochs} != budget {budget}")
        evaluation = document["eval"]
        for field in ("rho_normed", "macro_f1"):
            if field not in evaluation:
                raise KeyError(f"{path}: eval.{field} is missing")
        rho = float(evaluation["rho_normed"])
        f1 = float(evaluation["macro_f1"])
        if not -1.0 <= rho <= 1.0:
            raise AssertionError(f"{path}: rho_normed {rho} outside [-1, 1]")
        if not 0.0 <= f1 <= 1.0:
            raise AssertionError(f"{path}: macro_f1 {f1} outside [0, 1]")
        runs.append((epochs, rho, f1))
        sources[f"{label}_seed{seed}"] = (
            path,
            "eval.rho_normed / eval.macro_f1 / training.epochs_actual",
        )
    if len(runs) != len(DOSE_SEEDS):
        raise AssertionError(f"{label}: expected {len(DOSE_SEEDS)} seeds, got {len(runs)}")
    return runs


def _dose_series(budgets, sources):
    return [
        {"label": label, "budget": budget, "runs": _dose_runs(run_dir, budget, label, sources)}
        for label, run_dir, budget in budgets
    ]


def load_dose_response() -> dict:
    """Read the CellPLM dose-response runs for panel C.

    Every budget carries exactly three seeds (42/123/456) and every run must
    expose ``eval.rho_normed``, ``eval.macro_f1`` and
    ``training.epochs_actual``.

    Returns
    -------
    dict
        ``{"ce": [...], "ordinal_lora": [...], "ordinal": [...],
        "frozen_rho": float, "sources": {label: (path, field)}}``, where each
        budget entry is ``{"label", "budget", "runs": [(epochs, rho, f1), ...]}``.
    """
    sources: dict[str, tuple[Path, str]] = {}
    dose = {
        "ce": _dose_series(CE_BUDGETS, sources),
        "ordinal_lora": _dose_series(LORA_BUDGETS, sources),
        # Ordinal-only convergence is the level the LoRA arm has to recover;
        # loaded so the manuscript can cite it from the same failure-fast path.
        "ordinal": _dose_series((ORDINAL_BUDGET,), sources),
    }

    # The frozen reference is the seed-42 frozen baseline, which the manuscript
    # states to four decimals; assert the constant against its archive.
    anchor = D.load_json(CELLPLM_RESULTS / "cellplm_e3" / "seed42" / "seed42.json")
    frozen_rho = float(anchor["frozen_baseline"]["rho_normed"])
    S.check_value("CellPLM frozen rho (panel C reference)", frozen_rho, PANEL_C_FROZEN_RHO)
    dose["frozen_rho"] = frozen_rho
    dose["sources"] = sources
    return dose


def check_manuscript(rows, panel_b) -> None:
    """Assert every value that the manuscript states.

    Raises
    ------
    AssertionError
        If a plotted value differs from ``MANUSCRIPT_VALUES``.
    """
    by_label = {row["label"]: row for row in rows}

    def row(key):
        return by_label[key]["values"]

    S.check_value(
        "GF frozen rho",
        row(f"GF {S.ARMS['frozen']} (n = 1)")[0],
        D.MANUSCRIPT_VALUES["frozen"]["rho"],
    )

    # GF three-epoch per seed: the manuscript lists the three CE-naive values
    # without a per-seed label, so compare as a set.
    gf_three = np.sort(row(f"GF {S.ARMS['ce_naive']}, 3 epochs (n = 3)"))
    expected_three = np.sort(np.asarray(D.MANUSCRIPT_VALUES["ce_naive"]["rho_seeds"]))
    for value, expected in zip(gf_three, expected_three):
        S.check_value("GF CE-naive 3-epoch per-seed rho", value, expected)

    gf_ordinal = row(f"GF {S.ARMS['ordinal']} (n = 3)")
    S.check_value(
        "GF ordinal mean rho", float(np.mean(gf_ordinal)), D.MANUSCRIPT_VALUES["ordinal"]["rho"]
    )
    expected_ordinal = np.sort(np.asarray(D.MANUSCRIPT_VALUES["ordinal"]["rho_seeds"]))
    for value, expected in zip(np.sort(gf_ordinal), expected_ordinal):
        S.check_value("GF ordinal per-seed rho", value, expected)

    S.check_value(
        "scGPT frozen rho",
        row(f"scGPT {S.ARMS['frozen']} (n = 1)")[0],
        D.MANUSCRIPT_VALUES["scgpt_human"]["frozen_rho"],
    )

    # Panel B: the three manuscript seeds must peak at the stated epochs.
    for seed, expected_epoch in MANUSCRIPT_PEAK_EPOCHS.items():
        S.check_value(
            f"scGPT validation-rho peak epoch (seed {seed})",
            float(panel_b["peaks"][seed]["rho_epoch"]),
            float(expected_epoch),
        )
    if len(panel_b["peaks"]) != 5:
        raise AssertionError("panel B expects five seeds")

    lora_row = row(f"scGPT {S.ARMS['ordinal_lora']} (n = 3)")
    S.check_value(
        "scGPT ordinal+LoRA mean rho",
        float(np.mean(lora_row)),
        D.MANUSCRIPT_VALUES["scgpt"]["ordinal_lora_rho_mean"],
    )

    ce_row = row(f"CellPLM {S.ARMS['ce_naive']} (n = 3)")
    S.check_value(
        "CellPLM CE mean rho", float(np.mean(ce_row)), D.MANUSCRIPT_VALUES["cellplm"]["ce_rho_mean"]
    )
    # MANUSCRIPT_VALUES stores the sample SD (n - 1) of the three seeds.
    S.check_value(
        "CellPLM CE rho SD",
        float(np.std(ce_row, ddof=1)),
        D.MANUSCRIPT_VALUES["cellplm"]["ce_rho_sd"],
        tol=1e-3,
    )

    cp_ordinal = row(f"CellPLM {S.ARMS['ordinal']} (n = 3)")
    S.check_value(
        "CellPLM ordinal mean rho",
        float(np.mean(cp_ordinal)),
        D.MANUSCRIPT_VALUES["cellplm"]["ordinal_rho_mean"],
    )
    S.check_value(
        "CellPLM ordinal rho SD",
        float(np.std(cp_ordinal, ddof=1)),
        D.MANUSCRIPT_VALUES["cellplm"]["ordinal_rho_sd"],
        tol=1e-3,
    )

    for row_values in rows:
        values = row_values["values"]
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{row_values['label']}: non-finite rho")
        if np.any(np.abs(values) > 1.0):
            raise AssertionError(f"{row_values['label']}: rho outside [-1, 1]")


def plot_panel_a(ax, rows) -> None:
    """Draw the per-run dot plot: one row per architecture-by-arm cell."""
    y_positions = -np.arange(len(rows), dtype=float) * ROW_SPACING
    all_values = np.concatenate([row["values"] for row in rows])

    # Architecture groups: two 0.4 pt rules mark the bounds of the fine-tuned
    # block.  A filled band behind the data is not used anywhere in the set.
    architecture_order: list[str] = []
    for row in rows:
        architecture = row["label"].split(" ")[0]
        if not architecture_order or architecture_order[-1] != architecture:
            if architecture in architecture_order:
                raise AssertionError(f"architecture {architecture} is not contiguous")
            architecture_order.append(architecture)
    band_edges = []
    start = 0
    for index in range(1, len(rows) + 1):
        if (
            index == len(rows)
            or rows[index]["label"].split(" ")[0] != rows[start]["label"].split(" ")[0]
        ):
            band_edges.append((start, index - 1))
            start = index
    for _, last in band_edges[:-1]:
        ax.axhline(y_positions[last] - 0.5, color=S.GRID_COLOR,
                   lw=S.GRID_LW, zorder=1)

    ax.axvline(0.0, color=S.RULE_COLOR, ls=":", lw=S.RULE_LW, zorder=1)

    for row, y in zip(rows, y_positions):
        values = row["values"]
        colour = S.ARM_COLORS[row["arm"]]
        count = len(values)
        offsets = np.linspace(-POINT_JITTER, POINT_JITTER, count) if count > 1 else [0.0]
        if count > 1:
            ax.plot(
                [values.mean(), values.mean()],
                [y - MEAN_HALF_HEIGHT, y + MEAN_HALF_HEIGHT],
                color="#222222",
                lw=MEAN_LINE_WIDTH,
                solid_capstyle="butt",
                zorder=4,
            )
        ax.plot(
            values,
            y + np.asarray(offsets),
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=5,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["display"] for row in rows])
    # Rows are labelled directly; the tick marks would only duplicate them.
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y_positions[-1] - 0.75, y_positions[0] + 0.75)

    low = float(min(all_values.min(), 0.0))
    high = float(max(all_values.max(), 0.0))
    pad = (high - low) * 0.04
    ax.set_xlim(low - pad, high + pad)
    # The default locator returns one tick beyond the view (a "1.0" that no
    # value reaches); with the twin axis gone the axes now run to the canvas
    # edge, and that label stuck out. Keep only the ticks the view contains.
    ax.set_xticks(
        [tick for tick in ax.get_xticks() if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]]
    )
    ax.set_xlabel("Differentiation-gradient Spearman \u03c1")
    ax.set_title("Frozen, fine-tuned and ordinal arms across three architectures")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=S.ARM_COLORS["frozen"],
            markeredgecolor="white",
        ),
        Line2D([0], [0], color="#222222", lw=MEAN_LINE_WIDTH),
        Line2D([0], [0], color=S.RULE_COLOR, ls=":", lw=S.RULE_LW),
    ]
    ax.legend(
        handles,
        ["individual run", "run mean", "\u03c1 = 0"],
        # Upper centre keeps the key clear of the dotted rho = 0 line at the
        # low end of the axis and of the two tall rows at the high end.
        loc="upper center",
        frameon=False,
        borderpad=0.4,
        handlelength=1.6,
    )


def plot_panel_b(ax, panel_b) -> None:
    """Per-seed validation-rho peak (x, y) and the checkpoint-selection epoch.

    One point per seed: x is the epoch of the per-seed maximum validation rho,
    y is that maximum. A dotted connector reaches the epoch at which the
    checkpoint was selected by validation macro-F1, marked with an open ring.
    Where the two epochs coincide (seed 2024) the single filled peak marker
    already sits at the selection epoch, so no ring is drawn on top of it: the
    old ring-plus-dot bullseye had no distinct meaning.
    No trajectory or interpolated epoch is drawn. No frozen reference is drawn:
    the archived frozen value is a test-donor rho, not a validation rho.
    """
    colour = S.ARM_COLORS["ce_naive"]
    peaks = panel_b["peaks"]
    seeds = list(panel_b["seeds"])
    values = np.asarray([peaks[seed]["rho"] for seed in seeds], dtype=float)

    for seed in seeds:
        record = peaks[seed]
        peak_epoch = record["rho_epoch"]
        select_epoch = record["f1_epoch"]
        value = record["rho"]
        if select_epoch != peak_epoch:
            ax.plot(
                [peak_epoch, select_epoch],
                [value, value],
                color=colour,
                ls=":",
                lw=0.8,
                zorder=2,
            )
            # open ring = epoch selected on validation macro-F1
            ax.plot(
                select_epoch,
                value,
                linestyle="none",
                marker="o",
                markersize=S.MARKER_PT,
                markerfacecolor="white",
                markeredgecolor=colour,
                markeredgewidth=0.7,
                zorder=4,
            )
        # filled dot = epoch of the validation-rho peak (also the selection
        # epoch for seed 2024, where the two coincide)
        ax.plot(
            peak_epoch,
            value,
            linestyle="none",
            marker="o",
            markersize=S.MARKER_PT,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=5,
        )
        ax.annotate(
            f"{seed}",
            xy=(peak_epoch, value),
            xytext=(0, 6.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=S.TYPE["tick"],
            color="#333333",
            zorder=6,
            annotation_clip=False,
        )

    # The axis spans only the epochs that carry a mark; the 15-epoch training
    # budget is stated in the caption, not encoded in the axis range.
    axis_max = max(max(r["rho_epoch"], r["f1_epoch"]) for r in peaks.values())
    if axis_max > int(panel_b["epochs"]):
        raise AssertionError("plotted epoch exceeds the training budget")
    ax.set_xticks(np.arange(1, axis_max + 1))
    ax.set_xlim(0.5, axis_max + 0.5)
    low = float(values.min())
    high = float(values.max())
    pad = (high - low) * 0.22
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Epoch of peak validation ρ")
    ax.set_ylabel("Peak validation ρ")
    ax.set_title("Validation-ρ peak and selection epoch")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=S.MARKER_PT,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.4,
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=S.MARKER_PT,
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=0.7,
        ),
        Line2D([0], [0], color=colour, ls=":", lw=0.8),
    ]
    ax.legend(
        handles,
        [
            "validation-ρ peak",
            "epoch selected on macro-F1",
            "same seed",
        ],
        loc="lower right",
        frameon=False,
        borderpad=0.4,
        handlelength=1.6,
    )


def _dose_summary(budget):
    """Mean epoch, mean rho, sample SD of rho and mean macro-F1 for a budget."""
    runs = np.asarray(budget["runs"], dtype=float)
    return (
        float(runs[:, 0].mean()),
        float(runs[:, 1].mean()),
        float(runs[:, 1].std(ddof=1)),
        float(runs[:, 2].mean()),
    )


def _dose_f1_sd(budget):
    """Sample SD of macro-F1 over the archived seeds of one budget."""
    runs = np.asarray(budget["runs"], dtype=float)
    return float(runs[:, 2].std(ddof=1))


def _f1_plateau_epoch(dose) -> float:
    """First budget epoch whose mean macro-F1 has reached the plateau.

    The plateau band is the largest per-seed SD among the budgets whose mean
    macro-F1 sits within ``PANEL_C_PLATEAU_BAND`` of the series maximum, so the
    guide is a property of the archived numbers rather than a chosen constant.
    """
    means = np.asarray([_dose_summary(budget)[3] for budget in dose["ce"]])
    epochs = np.asarray([_dose_summary(budget)[0] for budget in dose["ce"]])
    peak = float(means.max())
    top = means >= peak - PANEL_C_PLATEAU_BAND
    band = max([_dose_f1_sd(budget) for budget, keep in zip(dose["ce"], top) if keep])
    reached = epochs[means >= peak - band]
    if not reached.size:
        raise AssertionError("no budget reaches the macro-F1 plateau band")
    return float(reached.min())


def _panel_c_axes(ax, dose) -> None:
    """Shared log-epoch x axis for the two stacked panel-C axes."""
    ax.set_xscale("log")
    ax.set_xlim(*PANEL_C_XLIM)
    ax.set_xticks(PANEL_C_XTICKS)
    # A log axis would otherwise fill the gaps with 2/3/5 minor ticks that carry
    # no label and only add ink.
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)


def plot_panel_c_f1(ax, dose) -> None:
    """Macro-F1 against training length, on the shared log epoch axis.

    The cross-entropy arm's task score is drawn as its three-seed mean with a
    sample-SD error bar, in the document ink rather than in an arm colour: on a
    scale of its own it is a task readout, not a representation arm, and the CE
    blue belongs to the rho series in the panel below. The vertical guide marks
    the first budget whose mean macro-F1 has reached the plateau, so the reader
    can compare the task knee with the rho knee directly beneath it.
    """
    summaries = [_dose_summary(budget) for budget in dose["ce"]]
    epochs = np.asarray([summary[0] for summary in summaries])
    f1 = np.asarray([summary[3] for summary in summaries])
    f1_sd = np.asarray([_dose_f1_sd(budget) for budget in dose["ce"]])

    _panel_c_axes(ax, dose)
    # sharex would hide these; setting them explicitly re-enabled them, and the
    # epoch labels appeared twice once the panels were stacked.
    ax.tick_params(axis="x", labelbottom=False)

    plateau = _f1_plateau_epoch(dose)
    ax.axvline(
        plateau,
        color=S.RULE_COLOR,
        linestyle=(0, (4, 2)),
        lw=S.LW_REFERENCE,
        zorder=1,
    )

    ax.errorbar(
        epochs,
        f1,
        yerr=f1_sd,
        color=PANEL_C_F1_COLOR,
        lw=S.LW_MAIN,
        marker="o",
        markersize=S.MARKER_PT,
        markerfacecolor=PANEL_C_F1_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.35,
        capsize=1.8,
        elinewidth=0.7,
        capthick=0.7,
        zorder=5,
    )

    low = float((f1 - f1_sd).min())
    high = float((f1 + f1_sd).max())
    pad = (high - low) * PANEL_C_F1_PAD
    ax.set_ylim(low - pad, high + pad)
    ax.set_ylabel("Macro-F1")
    # The measured statement, in one line that fits the widened panel width.
    ax.set_title("Macro-F1 saturates before \u03c1 collapses")
    ax.set_yticks(PANEL_C_F1_TICKS)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)


def plot_panel_c_rho(ax, dose) -> None:
    """Differentiation-gradient rho against training length, same epoch axis.

    The CE-naive arm is drawn as a budget mean with a sample-SD error bar and
    every seed overlaid as a filled dot at its own ``epochs_actual``. The
    ordinal+LoRA arm is drawn at its 15-epoch budget and at convergence. The
    frozen baseline is a structural dashed reference, annotated only. Every
    series is named by a direct label rather than a key, since at this size a
    legend would have to cover a mark or an error bar.
    """
    rho_colour = S.ARM_COLORS["ce_naive"]
    lora_colour = S.ARM_COLORS["ordinal_lora"]

    _panel_c_axes(ax, dose)
    # Epoch labels belong to this axis only; the twin tick set above is shared.
    ax.set_xticklabels([str(tick) for tick in PANEL_C_XTICKS])
    ax.set_ylim(*PANEL_C_RHO_LIM)
    ax.set_yticks(PANEL_C_RHO_TICKS)

    # The same structural guide as the macro-F1 panel, so the two knees can be
    # read against one epoch.
    ax.axvline(
        _f1_plateau_epoch(dose),
        color=S.RULE_COLOR,
        linestyle=(0, (4, 2)),
        lw=S.LW_REFERENCE,
        zorder=1,
    )

    # Frozen reference first, so every plotted arm sits above it.
    ax.axhline(
        dose["frozen_rho"],
        color=S.RULE_COLOR,
        linestyle=(0, (4, 2)),
        lw=S.LW_REFERENCE,
        zorder=1,
    )
    # Named under its own rule, in the band between the rule and the CE curve,
    # which is empty from epoch 3 to epoch 15.
    ax.text(
        3.2,
        dose["frozen_rho"] - 0.02,
        "frozen",
        ha="left",
        va="top",
        fontsize=S.TYPE["tick"],
        color="#333333",
        zorder=6,
    )

    ce = [_dose_summary(budget) for budget in dose["ce"]]
    ce_x = np.asarray([summary[0] for summary in ce])
    ce_rho = np.asarray([summary[1] for summary in ce])
    ce_sd = np.asarray([summary[2] for summary in ce])

    ax.errorbar(
        ce_x,
        ce_rho,
        yerr=ce_sd,
        color=rho_colour,
        lw=S.LW_MAIN,
        marker="o",
        markersize=S.MARKER_PT,
        markerfacecolor=rho_colour,
        markeredgecolor="white",
        markeredgewidth=0.35,
        capsize=1.8,
        elinewidth=0.7,
        capthick=0.7,
        zorder=5,
    )
    for budget in dose["ce"]:
        for epochs, rho, _f1 in budget["runs"]:
            # Filled, offset dots: hollow rings already mean "epoch selected on
            # macro-F1" in panel B, and at the budget x the ring merged with the
            # error-bar cap into a figure-of-eight.
            ax.plot(
                epochs * SEED_X_OFFSET,
                rho,
                linestyle="none",
                marker="o",
                markersize=1.8,
                markerfacecolor=rho_colour,
                markeredgecolor=rho_colour,
                markeredgewidth=0.2,
                zorder=4,
            )

    lora = [_dose_summary(budget) for budget in dose["ordinal_lora"]]
    ax.errorbar(
        np.asarray([summary[0] for summary in lora]),
        np.asarray([summary[1] for summary in lora]),
        yerr=np.asarray([summary[2] for summary in lora]),
        color=lora_colour,
        lw=S.LW_MAIN,
        marker="o",
        markersize=S.MARKER_PT,
        markerfacecolor=lora_colour,
        markeredgecolor="white",
        markeredgewidth=0.35,
        capsize=1.8,
        elinewidth=0.7,
        capthick=0.7,
        zorder=5,
    )

    ax.set_xlabel("Training epochs (log scale)")
    ax.set_ylabel("Differentiation-gradient \u03c1")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S.GRID_COLOR, linestyle="-", linewidth=S.GRID_LW)

    # Direct labels: at this size the panel has no rectangle large enough for a
    # four-entry legend that does not cover a mark or an error bar, so each
    # series is named where it has clear space.
    ax.text(
        150.0,
        0.34,
        "CE-naive",
        ha="left",
        va="bottom",
        fontsize=S.TYPE["tick"],
        color=rho_colour,
        zorder=6,
    )
    ax.text(
        560.0,
        0.78,
        "ordinal + LoRA",
        ha="right",
        va="top",
        fontsize=S.TYPE["tick"],
        color=lora_colour,
        zorder=6,
    )

def assert_no_clipping(fig) -> None:
    """Raise if any text artist leaves the canvas (a clipped label)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for artist in fig.findobj(matplotlib.text.Text):
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


def build(values):
    """Assemble the figure."""
    fig = plt.figure(figsize=S.figsize(*CANVAS_MM))
    # 2 % top strip keeps the panel letters inside the 2 mm trim rule
    # (rect is left, bottom, width, height for the constrained engine).
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.98))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.0], height_ratios=[1.0, 0.78])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    # Panel C is a stacked pair sharing one log-epoch x axis: macro-F1 above,
    # rho below. Two independently scaled y axes on one frame made a task knee
    # near epoch 15 and a gradient collapse near epoch 60 look coupled.
    cell_c = grid[1, 1].subgridspec(2, 1, height_ratios=[0.95, 1.45], hspace=0.13)
    ax_c_f1 = fig.add_subplot(cell_c[0])
    ax_c_rho = fig.add_subplot(cell_c[1], sharex=ax_c_f1)
    plot_panel_a(ax_a, values["rows"])
    plot_panel_b(ax_b, values["panel_b"])
    plot_panel_c_f1(ax_c_f1, values["dose"])
    plot_panel_c_rho(ax_c_rho, values["dose"])
    # y = 1.038 keeps the letter top ~2 mm inside the trim on this canvas;
    # at the old 1.06 the letters touched the page edge.
    S.panel_label(ax_a, "A", y=1.038)
    S.panel_label(ax_b, "B", y=1.038)
    # Same axis-relative slot as A and B; the letter rides the upper stacked
    # axis of panel C so it clears the panel title, not the rho axis title.
    S.panel_label(ax_c_f1, "C", y=1.13)
    return fig


def main() -> int:
    """Render the figure, verify it, and write the manifest entry."""
    S.set_style()
    values = load_values()
    check_manuscript(values["rows"], values["panel_b"])
    values["dose"] = load_dose_response()
    values["sources"].update(values["dose"]["sources"])
    fig = build(values)
    assert_no_clipping(fig)

    inputs = {}
    for path, _field in values["sources"].values():
        # One JSON can back several rows; hash it once. The five convergence
        # checkpoints each use "args.json", so key those by their run directory;
        # the per-seed run JSONs are all named after their seed, so key those by
        # the run directory too.
        if path.stem == "args":
            key = f"args_{path.parent.name}"
        elif path.stem == path.parent.name:
            key = f"{path.parent.parent.name}_{path.stem}"
        else:
            key = path.stem
        inputs.setdefault(key, path)
    pdf_path, png_path = S.save_figure(fig, OUT_NAME, outdir=str(S.FIGURE_DIR), inputs=inputs)
    plt.close(fig)

    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print("plotted value -> file:field")
    for row in values["rows"]:
        if row["label"] in values["sources"]:
            path, field = values["sources"][row["label"]]
            try:
                shown = str(path.relative_to(ROOT))
            except ValueError:
                shown = str(path)
            print(
                f"  {row['label']}: {', '.join(f'{v:.4f}' for v in row['values'])}"
                f"  <- {shown}:{field}"
            )
        else:
            print(
                f"  {row['label']}: {', '.join(f'{v:.4f}' for v in row['values'])}"
                "  <- paper1_figure_data.MANUSCRIPT_VALUES['frozen']['rho']"
            )
    print("panel B (validation-rho peak / selection epoch)")
    for seed in values["panel_b"]["seeds"]:
        record = values["panel_b"]["peaks"][seed]
        path, field = values["sources"][f"validation_peak_seed{seed}"]
        print(
            f"  seed {seed}: rho={record['rho']:.4f} at epoch {record['rho_epoch']};"
            f" checkpoint selected on F1 at epoch {record['f1_epoch']} "
            f"(f1={record['f1']:.4f})  <- {path.relative_to(ROOT)}:{field}"
        )
    print("panel C (CellPLM dose-response; mean +/- sample SD over three seeds)")
    print(f"  frozen reference rho={values['dose']['frozen_rho']:.4f}")
    for key, title in (
        ("ce", "CE-naive"),
        ("ordinal_lora", "ordinal + LoRA"),
        ("ordinal", "ordinal head (convergence reference)"),
    ):
        for budget in values["dose"][key]:
            epochs, rho, rho_sd, f1 = _dose_summary(budget)
            f1_sd = float(np.asarray(budget["runs"], dtype=float)[:, 2].std(ddof=1))
            path = values["dose"]["sources"][f"{budget['label']}_seed42"][0]
            print(
                f"  {title} @ {budget['label']}: epochs={epochs:.2f} "
                f"rho={rho:.4f} (sd {rho_sd:.4f}) "
                f"f1={f1:.4f} (sd {f1_sd:.4f})"
                f"  <- {path.relative_to(ROOT)} (seed 42; seeds 123/456 sibling)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
