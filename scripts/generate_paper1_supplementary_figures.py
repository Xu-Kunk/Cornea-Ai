#!/usr/bin/env python
"""Generate the Paper1 (Doc S1) supplementary figures S3, S5, S6, S7, S8.

Style follows scripts/generate_paper1_results_figures.py (Okabe-Ito palette,
panel labels, 300-dpi vector PDF + PNG, provenance metadata, per-figure CSV
export of every plotted value for the data-availability statement).

Outputs (figures/paper1/):
  figureS3_per_seed_reproducibility.pdf/.png/.csv
      2x2 per-metric panels (macro-F1, differentiation-gradient rho, LAA,
      ECE).  For each arm (naive / CAFT) the three seed values are drawn as
      points plus the 3-seed mean with min-max range.
      Source: figures/paper1/table_t7_seed_reproducibility.csv
  figureS5_training_dynamics.pdf/.png/.csv
      1x2 panels (naive / CAFT) of train-loss curves per seed vs training
      step, with epoch boundaries derived from each run's log_history
      (steps/epoch = max step / num_train_epochs = 2490).
      Source: save/caft_gf_lam*/model/checkpoint-7473/trainer_state.json (6 runs)
  figureS6_per_donor_breakdown.pdf/.png/.csv
      1x2 panels (naive / CAFT); per-donor donor_1/donor_2 donor-level
      macro-F1 paired by seed; epoch 3 solid, epoch 2 dashed.
      Source: artifacts/paper1_tier_b_eval/results_{lambda}_{seed}.json and
      results_epoch2_{lambda}_{seed}.json
  figureS7_layerwise_rho_per_seed.pdf/.png/.csv
      single panel: layerwise differentiation-gradient Spearman rho
      (x = transformer layers 1-12) per seed for the 6 fine-tuned models
      (naive blue-green family / CAFT pink family) with the frozen
      Geneformer as a thick dark reference line (seed-to-seed layerwise
      variation).
      Source: artifacts/paper1_layerwise_rho/layerwise_rho_*.json (7 files)
  figureS8_internal_external_inversion.pdf/.png/.csv
      internal (train-split) vs external (GSE153515) donor-level macro-F1
      paired lines for HVG/PCA and frozen Geneformer, showing the ranking
      inversion (internal 0.7787 / 0.8834 -> external 0.6192 / 0.4421), with
      the six fine-tuned arms shown at their external values.
      Sources: external values + 95% CIs from
      artifacts/paper1_tier_b_eval/results_{hvg_pca,frozen_gf,lambda_seed}.json;
      internal baseline values from INTERNAL_BASELINES in
      scripts/run_tier_b_baselines.py (the tier-B evaluation JSONs contain no
      "internal" key - verified).

Notes
-----
* Every numeric value plotted is read from the source files listed above.
  The internal training-split baselines (HVG/PCA 0.7787, frozen Geneformer
  0.8834) are NOT stored in the tier-B evaluation JSONs, so they are taken
  from the documented constant INTERNAL_BASELINES in
  scripts/run_tier_b_baselines.py (line ~129) and labeled "internal".
* Okabe-Ito colorblind-safe palette; Liberation Sans fallback; vector PDF
  (pdf.fonttype=42) + 300-dpi PNG; provenance metadata in the PDF Title.
* Reference: Li D-Q, Kim S, et al. Ocul Surf. 2021;20:20-32.
  doi:10.1016/j.jtos.2020.12.004 (PMID: 33388438) - GSE153515 limbal atlas.

Usage
-----
    python3 scripts/generate_paper1_supplementary_figures.py
    python3 scripts/generate_paper1_supplementary_figures.py --only S5
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures" / "paper1"
DEFAULT_T7_CSV = DEFAULT_OUT_DIR / "table_t7_seed_reproducibility.csv"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "save"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
DEFAULT_LAYERWISE_DIR = PROJECT_ROOT / "artifacts" / "paper1_layerwise_rho"

# Okabe-Ito colorblind-safe palette
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_BLUEGREEN = "#009E73"
OI_ORANGE = "#E69F00"
OI_SKYBLUE = "#56B4E9"
OI_PINK = "#CC79A7"
OI_YELLOW = "#F0E442"
OI_GRAY = "#999999"

ARM_LAMBDAS = [0.0, 0.5]
ARM_LABELS = {0.0: "Fine-tune \u03bb=0", 0.5: "CAFT \u03bb=0.5"}
ARM_COLORS = {0.0: OI_BLUE, 0.5: OI_VERMILLION}
SEEDS = [42, 123, 456]

# Seed hue families (same as Figure 7 in the main results script).
_NAIVE_SEED_COLORS = {42: OI_BLUEGREEN, 123: "#5EC4AB", 456: "#005A40"}
_CAFT_SEED_COLORS = {42: OI_PINK, 123: "#E0A6C9", 456: "#9C4E73"}
_FROZEN_RHO_COLOR = "#333333"

# Internal training-split baselines (donor-level macro-F1).  Source:
# INTERNAL_BASELINES = {"HVG/PCA": 0.7787, "Frozen Geneformer": 0.8834} in
# scripts/run_tier_b_baselines.py (line ~129); documented in
# docs/changes/2026-08-02-paper1-tier-b-eval-results.md and
# figures/paper1/table_t3_primary_endpoint.csv.  Not present in the
# tier-B external-evaluation JSONs.
INTERNAL_BASELINES = {"HVG/PCA": 0.7787, "Frozen Geneformer": 0.8834}

EXTERNAL_BASELINE_FILES = {
    "HVG/PCA": "results_hvg_pca.json",
    "Frozen Geneformer": "results_frozen_gf.json",
}

RESULT_FILE_RE = re.compile(r"^results_(\d+(?:\.\d+)?)_(\d+)\.json$")
EPOCH2_FILE_RE = re.compile(r"^results_epoch2_(\d+(?:\.\d+)?)_(\d+)\.json$")
TRAIN_DIR_RE = re.compile(r"^caft_gf_lam(\d+(?:\.\d+)?)_seed(\d+)_")

# ---------------------------------------------------------------------------
# Styling (identical contract to the main results-figure script)
# ---------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#000000",
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    """Place a bold uppercase panel label (12 pt) at the top-left of `ax`."""
    pos = ax.get_position()
    fig.text(
        pos.x0 - 0.004,
        pos.y1 + 0.008,
        label,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def figure_metadata(title: str) -> dict[str, str]:
    """Provenance metadata embedded in the PDF Title field."""
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "Title": title,
        "Author": "Corneal-FM (Paper1, Ophthalmology Science)",
        "Subject": "Doc S1 supplementary figures, Paper1 "
                   f"(generated {generated} UTC)",
        "Keywords": "CAFT; Geneformer; donor-level macro-F1; GSE153515",
        "Creator": "scripts/generate_paper1_supplementary_figures.py",
    }


def save_figure(
    fig: plt.Figure, name: str, out_dir: Path, metadata: dict[str, str] | None = None
) -> list[Path]:
    """Save vector PDF (with provenance metadata) + 300-dpi PNG."""
    out_pdf = out_dir / f"{name}.pdf"
    out_png = out_dir / f"{name}.png"
    if metadata:  # PNG backend does not support a metadata kwarg
        fig.savefig(out_pdf, dpi=300, metadata=metadata)
    else:
        fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return [out_pdf, out_png]


def no_data_note(ax: plt.Axes, msg: str) -> None:
    ax.text(
        0.5,
        0.5,
        msg,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color=OI_GRAY,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _numeric(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Figure S3 - per-seed reproducibility (4 metrics x 2 arms x 3 seeds)
# ---------------------------------------------------------------------------


def _seed_color(lam: float, seed: int) -> str:
    return _NAIVE_SEED_COLORS[seed] if lam == 0.0 else _CAFT_SEED_COLORS[seed]


def figure_s3(table_t7_csv: Path, out_dir: Path) -> list[Path]:
    """Per-seed reproducibility of the four continuity/endpoint metrics."""
    if not table_t7_csv.exists():
        print(f"[Figure S3] WARNING: {table_t7_csv} missing; skipping")
        return []
    df = pd.read_csv(table_t7_csv)
    metrics = [
        ("macro_f1", "macro-F1 (donor level)"),
        ("marker_gradient_rho", "Differentiation gradient \u03c1 (Spearman)"),
        ("LAA", "LAA (layer adjacency)"),
        ("ECE", "ECE (calibration)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    csv_rows: list[dict[str, Any]] = []
    seed_offsets = [-0.08, 0.0, 0.08]
    for panel_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes.flat[panel_idx]
        for arm_idx, lam in enumerate(ARM_LAMBDAS):
            arm_label = ARM_LABELS[lam]
            sub = df[df["arm"] == arm_label]
            vals = [_numeric(v) for v in sub[metric].tolist()]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            x = float(arm_idx)
            for off, seed in zip(seed_offsets, SEEDS):
                seed_val = _numeric(
                    sub.loc[sub["seed"] == seed, metric].iloc[0]
                ) if (sub["seed"] == seed).any() else None
                if seed_val is None:
                    continue
                ax.plot(
                    x + off, seed_val, "o", ms=4, mec="none",
                    color=_seed_color(lam, seed), zorder=3,
                )
                csv_rows.append({
                    "arm": arm_label, "metric": metric, "seed": seed,
                    "value": seed_val, "stat": "seed",
                })
            mean_val = float(np.mean(vals))
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            ax.errorbar(
                x, mean_val, yerr=np.array([[mean_val - lo], [hi - mean_val]]),
                fmt="D", ms=5, color=ARM_COLORS[lam], capsize=3,
                lw=1.2, zorder=4,
            )
            csv_rows.append({"arm": arm_label, "metric": metric, "seed": None,
                             "value": mean_val, "stat": "mean"})
            csv_rows.append({"arm": arm_label, "metric": metric, "seed": None,
                             "value": lo, "stat": "min"})
            csv_rows.append({"arm": arm_label, "metric": metric, "seed": None,
                             "value": hi, "stat": "max"})
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([ARM_LABELS[0.0], ARM_LABELS[0.5]])
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.45, 1.45)
        panel_label(fig, ax, "ABCD"[panel_idx])

    handles = [
        plt.Line2D([0], [0], marker="o", ls="none", ms=4, mec="none",
                   color=OI_BLUEGREEN, label="naive seeds 42/123/456"),
        plt.Line2D([0], [0], marker="o", ls="none", ms=4, mec="none",
                   color=OI_PINK, label="CAFT seeds 42/123/456"),
        plt.Line2D([0], [0], marker="D", ls="none", ms=5,
                   color=OI_BLUE, label="naive mean \u00b1 range"),
        plt.Line2D([0], [0], marker="D", ls="none", ms=5,
                   color=OI_VERMILLION, label="CAFT mean \u00b1 range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(
            out_dir / "figureS3_per_seed_reproducibility.csv",
            index=False, float_format="%.17g",
        )
        print(f"[Figure S3] {len(csv_rows)} plotted value(s) -> "
              "figureS3_per_seed_reproducibility.csv")
    print(f"[Figure S3] per-seed reproducibility: {len(ARM_LAMBDAS)} arm(s) x "
          f"{len(metrics)} metric(s), {len(SEEDS)} seed(s)")
    return save_figure(
        fig, "figureS3_per_seed_reproducibility", out_dir,
        figure_metadata("Figure S3 - Per-seed reproducibility (Doc S1)"),
    )


# ---------------------------------------------------------------------------
# Figure S5 - training dynamics (train loss vs step, per seed x lambda)
# ---------------------------------------------------------------------------


def _find_training_runs(train_dir: Path) -> list[dict[str, Any]]:
    """Locate the six final runs (checkpoint-7473) for lambda x seed."""
    runs: list[dict[str, Any]] = []
    for state_file in sorted(train_dir.glob("caft_gf_lam*/model/checkpoint-7473/trainer_state.json")):
        m = TRAIN_DIR_RE.match(state_file.parent.parent.parent.name)
        if not m:
            continue
        lam = float(m.group(1))
        seed = int(m.group(2))
        if lam not in ARM_LAMBDAS or seed not in SEEDS:
            continue
        runs.append({"lambda": lam, "seed": seed, "path": state_file})
    # dedupe on (lambda, seed) keeping the last
    seen: set[tuple[float, int]] = set()
    unique: list[dict[str, Any]] = []
    for run in runs:
        key = (run["lambda"], run["seed"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(run)
    return unique


def figure_s5(train_dir: Path, out_dir: Path) -> list[Path]:
    """Train-loss curves per seed for lambda=0 (naive) and lambda=0.5 (CAFT)."""
    runs = _find_training_runs(train_dir)
    if not runs:
        print("[Figure S5] WARNING: no checkpoint-7473 trainer_state.json "
              "files found; skipping")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    csv_rows: list[dict[str, Any]] = []
    steps_per_epoch: dict[tuple[float, int], float] = {}
    for run in runs:
        lam = run["lambda"]
        seed = run["seed"]
        state = json.loads(run["path"].read_text())
        entries = [e for e in state.get("log_history", []) if "loss" in e]
        if not entries:
            continue
        max_step = float(max(e["step"] for e in entries))
        n_epochs = float(state.get("num_train_epochs") or 1.0)
        steps_per_epoch[(lam, seed)] = max_step / n_epochs
        steps = [e["step"] for e in entries]
        losses = [float(e["loss"]) for e in entries]
        epochs = [e.get("epoch", np.nan) for e in entries]
        panel_idx = ARM_LAMBDAS.index(lam)
        ax = axes[panel_idx]
        ax.plot(steps, losses, lw=1.2, ls="-", color=_seed_color(lam, seed),
                label=f"seed {seed}", zorder=3)
        for step, epoch, loss in zip(steps, epochs, losses):
            csv_rows.append({"run": run["path"].parent.parent.parent.name,
                             "lambda": lam, "seed": seed, "step": step,
                             "epoch": epoch, "loss": loss})

    for panel_idx, lam in enumerate(ARM_LAMBDAS):
        ax = axes[panel_idx]
        specs = [(lam, s) for s in SEEDS if (lam, s) in steps_per_epoch]
        if specs:
            sp = steps_per_epoch[specs[0]]
            for boundary in (sp, 2 * sp):
                ax.axvline(boundary, color=OI_GRAY, ls="--", lw=0.7, zorder=1)
            ax.text(sp, ax.get_ylim()[1] * 0.98, "end epoch 1",
                    ha="center", va="top", fontsize=6.5, color=OI_GRAY)
            ax.text(2 * sp, ax.get_ylim()[1] * 0.98, "end epoch 2",
                    ha="center", va="top", fontsize=6.5, color=OI_GRAY)
        ax.set_title(ARM_LABELS[lam], fontsize=9)
        ax.set_xlabel("Training step")
        if panel_idx == 0:
            ax.set_ylabel("Train loss")
        ax.legend(loc="upper right", fontsize=6.5)
        panel_label(fig, ax, "AB"[panel_idx])
    fig.tight_layout()

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_dir / "figureS5_training_dynamics.csv",
                                      index=False, float_format="%.17g")
        print(f"[Figure S5] {len(csv_rows)} training-step record(s) -> "
              "figureS5_training_dynamics.csv")
    sp = steps_per_epoch.get((0.0, 42), 2490.0)
    print(f"[Figure S5] training dynamics: {len(runs)} run(s), "
          f"{sp:.0f} steps/epoch")
    return save_figure(
        fig, "figureS5_training_dynamics", out_dir,
        figure_metadata("Figure S5 - Training dynamics (Doc S1)"),
    )


# ---------------------------------------------------------------------------
# Figure S6 - per-donor breakdown (donor_1 / donor_2 paired by seed)
# ---------------------------------------------------------------------------


def _load_per_donor(results_dir: Path, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in sorted(results_dir.glob("*.json")):
        m = pattern.match(f.name)
        if not m:
            continue
        lam = float(m.group(1))
        seed = int(m.group(2))
        if lam not in ARM_LAMBDAS or seed not in SEEDS:
            continue
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        primary = raw.get("primary_endpoint") or {}
        per_donor = primary.get("per_donor") or {}
        rows.append({
            "lambda": lam, "seed": seed, "file": f,
            "donor_1": _numeric(per_donor.get("donor_1")),
            "donor_2": _numeric(per_donor.get("donor_2")),
            "macro_f1": _numeric(primary.get("macro_f1")),
        })
    return rows


def figure_s6(results_dir: Path, out_dir: Path) -> list[Path]:
    """Per-donor donor-level macro-F1 paired by seed (epoch 3 vs epoch 2)."""
    epoch3 = _load_per_donor(results_dir, RESULT_FILE_RE)
    epoch2 = _load_per_donor(results_dir, EPOCH2_FILE_RE)
    if not epoch3:
        print("[Figure S6] WARNING: no per-donor results JSONs found; skipping")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    csv_rows: list[dict[str, Any]] = []
    for panel_idx, lam in enumerate(ARM_LAMBDAS):
        ax = axes[panel_idx]
        plotted: list[float] = []
        for seed in SEEDS:
            r3 = next((r for r in epoch3 if r["lambda"] == lam and r["seed"] == seed), None)
            r2 = next((r for r in epoch2 if r["lambda"] == lam and r["seed"] == seed), None)
            color = _seed_color(lam, seed)
            if r3 and r3["donor_1"] is not None and r3["donor_2"] is not None:
                ax.plot([0, 1], [r3["donor_1"], r3["donor_2"]], ls="-",
                        color=color, lw=1.2, marker="o", ms=4, zorder=3)
                for donor, val in (("donor_1", r3["donor_1"]), ("donor_2", r3["donor_2"])):
                    csv_rows.append({"arm": ARM_LABELS[lam], "seed": seed,
                                     "epoch": 3, "donor": donor,
                                     "macro_f1": val})
                    plotted.append(val)
            if r2 and r2["donor_1"] is not None and r2["donor_2"] is not None:
                ax.plot([0, 1], [r2["donor_1"], r2["donor_2"]], ls="--",
                        color=color, lw=0.9, marker="s", ms=3, zorder=2)
                for donor, val in (("donor_1", r2["donor_1"]), ("donor_2", r2["donor_2"])):
                    csv_rows.append({"arm": ARM_LABELS[lam], "seed": seed,
                                     "epoch": 2, "donor": donor,
                                     "macro_f1": val})
                    plotted.append(val)
        if plotted:
            lo = min(plotted) - 0.05
            hi = max(plotted) + 0.05
            ax.set_ylim(max(0.0, lo), hi)
        ax.set_title(ARM_LABELS[lam], fontsize=9)
        if panel_idx == 0:
            ax.set_ylabel("Per-donor macro-F1")
        ax.legend(handles=[
            plt.Line2D([0], [0], ls="-", color=OI_GRAY, label="epoch 3"),
            plt.Line2D([0], [0], ls="--", color=OI_GRAY, label="epoch 2"),
        ], loc="lower left", fontsize=6.5)
        panel_label(fig, ax, "AB"[panel_idx])
    fig.tight_layout()

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out_dir / "figureS6_per_donor_breakdown.csv",
                                      index=False, float_format="%.17g")
        print(f"[Figure S6] {len(csv_rows)} per-donor value(s) -> "
              "figureS6_per_donor_breakdown.csv")
    print(f"[Figure S6] per-donor breakdown: epoch 3 {len(epoch3)} run(s), "
          f"epoch 2 {len(epoch2)} run(s)")
    return save_figure(
        fig, "figureS6_per_donor_breakdown", out_dir,
        figure_metadata("Figure S6 - Per-donor breakdown (Doc S1)"),
    )


# ---------------------------------------------------------------------------
# Figure S7 - layerwise differentiation-gradient rho per seed
# ---------------------------------------------------------------------------


def figure_s7(layerwise_dir: Path, out_dir: Path) -> list[Path]:
    """Layerwise rho per seed (6 fine-tuned) + frozen reference line."""
    files = sorted(layerwise_dir.glob("layerwise_rho_*.json"))
    if not files:
        print("[Figure S7] WARNING: no layerwise_rho_*.json files found; skipping")
        return []
    fig, ax = plt.subplots(figsize=(7, 3.5))
    csv_rows: list[dict[str, Any]] = []
    drawn: list[tuple[str, str]] = []
    all_rho: list[float] = []
    for f in files:
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model = raw.get("model", f.stem)
        lam = raw.get("lambda")
        seed = raw.get("seed")
        labels = raw.get("layer_labels") or []
        layers = [i for i, lab in enumerate(labels)
                  if str(lab).startswith("transformer_layer_")]
        vals = raw.get("marker_gradient_rho")
        if not isinstance(vals, list) or len(vals) <= max(layers, default=0):
            print(f"[Figure S7] WARNING: bad rho array in {f.name}; skipping")
            continue
        xs = list(layers)
        ys = [_numeric(vals[i]) for i in layers]
        ys = [np.nan if v is None else v for v in ys]
        if all(np.isnan(v) for v in ys):
            continue
        for layer, rho in zip(xs, ys):
            if not np.isnan(rho):
                csv_rows.append({"model": model, "arm": raw.get("arm"),
                                 "lambda": lam, "seed": seed,
                                 "layer": layer, "rho": rho})
                all_rho.append(float(rho))
        if model == "frozen_gf":
            color = _FROZEN_RHO_COLOR
            lw, ls, label = 2.4, "-", "frozen GF"
        elif lam == 0.0:
            color = _NAIVE_SEED_COLORS.get(int(seed), OI_BLUEGREEN)
            lw, ls, label = 1.2, "-", f"naive seed {seed}"
        else:
            color = _CAFT_SEED_COLORS.get(int(seed), OI_PINK)
            lw, ls, label = 1.2, "-", f"CAFT seed {seed}"
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, label=label, zorder=3)
        drawn.append((model, label))
    if not drawn:
        print("[Figure S7] WARNING: no plottable layerwise rho; skipping")
        plt.close(fig)
        return []

    ax.axhline(0.0, color=OI_GRAY, ls=":", lw=0.7, zorder=1)
    frozen_peak = next(
        (c["rho"] for c in csv_rows if c["model"] == "frozen_gf" and c["layer"] == 12),
        None,
    )
    if frozen_peak is not None:
        ax.annotate(
            f"frozen L12 peak: {frozen_peak:.3f}",
            xy=(12, frozen_peak), xytext=(9.6, 0.90),
            fontsize=6.5, color="#555555",
            arrowprops=dict(arrowstyle="->", lw=0.6),
        )
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(list(range(1, 13)))
    lo = min(all_rho) - 0.20
    hi = max(all_rho) + 0.10
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Differentiation gradient \u03c1 (Spearman)")
    ax.set_title("Layerwise differentiation-gradient Spearman \u03c1",
                 fontsize=9)
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(
            out_dir / "figureS7_layerwise_rho_per_seed.csv",
            index=False, float_format="%.17g",
        )
        print(f"[Figure S7] {len(csv_rows)} layerwise rho value(s) -> "
              "figureS7_layerwise_rho_per_seed.csv")
    print(f"[Figure S7] layerwise rho per seed: {len(drawn)} model line(s), "
          f"layers 1-12")
    return save_figure(
        fig, "figureS7_layerwise_rho_per_seed", out_dir,
        figure_metadata("Figure S7 - Layerwise rho per seed (Doc S1)"),
    )


# ---------------------------------------------------------------------------
# Figure S8 - internal vs external ranking inversion
# ---------------------------------------------------------------------------


def figure_s8(results_dir: Path, out_dir: Path) -> list[Path]:
    """Internal (train-split) vs external (GSE153515) donor-level macro-F1."""
    if not results_dir.exists():
        print("[Figure S8] WARNING: results dir missing; skipping")
        return []
    # External values (from JSON, with bootstrap 95% CIs).
    models: list[dict[str, Any]] = []
    for name, fname in EXTERNAL_BASELINE_FILES.items():
        path = results_dir / fname
        if not path.exists():
            print(f"[Figure S8] WARNING: {fname} missing; skipping baseline")
            continue
        raw = json.loads(path.read_text())
        primary = raw.get("primary_endpoint") or {}
        models.append({
            "label": name, "color": OI_GRAY if name == "HVG/PCA" else _FROZEN_RHO_COLOR,
            "external": _numeric(primary.get("macro_f1")),
            "ci_lower": _numeric(primary.get("ci_lower")),
            "ci_upper": _numeric(primary.get("ci_upper")),
            "internal": INTERNAL_BASELINES.get(name),
            "seed": None, "lambda": None,
        })
    for f in sorted(results_dir.glob("results_*.json")):
        m = RESULT_FILE_RE.match(f.name)
        if not m:
            continue
        lam = float(m.group(1))
        seed = int(m.group(2))
        if lam not in ARM_LAMBDAS or seed not in SEEDS:
            continue
        raw = json.loads(f.read_text())
        primary = raw.get("primary_endpoint") or {}
        models.append({
            "label": ARM_LABELS[lam],
            "color": _seed_color(lam, seed),
            "external": _numeric(primary.get("macro_f1")),
            "ci_lower": _numeric(primary.get("ci_lower")),
            "ci_upper": _numeric(primary.get("ci_upper")),
            "internal": None,  # FT arms have no internal values in the JSONs
            "seed": seed, "lambda": lam,
        })

    fig, ax = plt.subplots(figsize=(7, 4))
    csv_rows: list[dict[str, Any]] = []
    for model in models:
        ext = model["external"]
        if ext is None:
            continue
        int_val = model["internal"]
        csv_rows.append({
            "model": model["label"], "setting": "internal",
            "macro_f1": int_val,
            "ci_lower": None, "ci_upper": None,
        })
        csv_rows.append({
            "model": model["label"], "setting": "external",
            "macro_f1": ext,
            "ci_lower": model["ci_lower"], "ci_upper": model["ci_upper"],
        })
        if int_val is not None:
            ax.plot([0, 1], [int_val, ext], ls="-", color=model["color"],
                    lw=2.2 if model["label"] == "Frozen Geneformer" else 1.2,
                    marker="o", ms=5, zorder=3)
            ax.annotate(f"{int_val:.4f}", xy=(0, int_val), xytext=(-0.28, int_val),
                        fontsize=6.5, color=model["color"], ha="center", va="center")
            ax.annotate(f"{ext:.4f}", xy=(1, ext), xytext=(1.28, ext),
                        fontsize=6.5, color=model["color"], ha="center", va="center")
            if model["ci_lower"] is not None and model["ci_upper"] is not None:
                ax.errorbar([1], [ext], yerr=np.array([[
                    ext - model["ci_lower"], model["ci_upper"] - ext]]).T,
                    fmt="none", color=model["color"], capsize=3, lw=1.0, zorder=2)
        else:
            # FT arm: external only (internal not available).
            ax.plot([1], [ext], "o", ms=4.5, mec="none", color=model["color"],
                    zorder=3, alpha=0.85)
    # 3-seed mean markers per FT arm.
    for lam in ARM_LAMBDAS:
        vals = [m["external"] for m in models
                if m["lambda"] == lam and m["external"] is not None]
        if vals:
            ax.plot([1], [float(np.mean(vals))], "D", ms=6,
                    color=ARM_COLORS[lam], zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Internal\n(train split)", "External\n(GSE153515)"])
    ax.set_ylabel("Donor-level macro-F1")
    ax.set_xlim(-0.45, 1.65)
    handles = [
        plt.Line2D([0], [0], ls="-", color=OI_GRAY, lw=1.2,
                   label="HVG/PCA (internal \u2192 external)"),
        plt.Line2D([0], [0], ls="-", color=_FROZEN_RHO_COLOR, lw=2.2,
                   label="Frozen Geneformer (internal \u2192 external)"),
        plt.Line2D([0], [0], marker="o", ls="none", ms=4.5, mec="none",
                   color=OI_BLUEGREEN, label="Fine-tune \u03bb=0 seeds (ext)"),
        plt.Line2D([0], [0], marker="o", ls="none", ms=4.5, mec="none",
                   color=OI_PINK, label="CAFT \u03bb=0.5 seeds (ext)"),
        plt.Line2D([0], [0], marker="D", ls="none", ms=6,
                   color=OI_GRAY, label="3-seed mean (ext)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5)
    fig.tight_layout()

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(
            out_dir / "figureS8_internal_external_inversion.csv",
            index=False, float_format="%.17g",
        )
        print(f"[Figure S8] {len(csv_rows)} model x setting value(s) -> "
              "figureS8_internal_external_inversion.csv")
    print(f"[Figure S8] internal/external inversion: {len(models)} model(s); "
          "internal values only for the 2 reference baselines "
          "(INTERNAL_BASELINES in scripts/run_tier_b_baselines.py)")
    return save_figure(
        fig, "figureS8_internal_external_inversion", out_dir,
        figure_metadata("Figure S8 - Internal vs external ranking (Doc S1)"),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Paper1 (Doc S1) supplementary figures S3/S5/S6/S7/S8."
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--train_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--layerwise_dir", default=str(DEFAULT_LAYERWISE_DIR))
    parser.add_argument("--table_t7_csv", default=str(DEFAULT_T7_CSV))
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of S3,S5,S6,S7,S8")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    builders: dict[str, tuple[Any, str]] = {
        "S3": (lambda: figure_s3(Path(args.table_t7_csv), out_dir),
               "Per-seed reproducibility"),
        "S5": (lambda: figure_s5(Path(args.train_dir), out_dir),
               "Training dynamics"),
        "S6": (lambda: figure_s6(Path(args.results_dir), out_dir),
               "Per-donor breakdown"),
        "S7": (lambda: figure_s7(Path(args.layerwise_dir), out_dir),
               "Layerwise rho per seed"),
        "S8": (lambda: figure_s8(Path(args.results_dir), out_dir),
               "Internal vs external inversion"),
    }
    requested = [s.strip().upper() for s in (args.only or "S3,S5,S6,S7,S8").split(",")]
    generated: list[Path] = []
    for key in ("S3", "S5", "S6", "S7", "S8"):
        if key not in requested:
            continue
        print(f"\n[Figure {key}] {builders[key][1]} ...")
        try:
            generated.extend(builders[key][0]())
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[Figure {key}] ERROR: {exc}")
    print(f"\nDone. {len(generated)} file(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
