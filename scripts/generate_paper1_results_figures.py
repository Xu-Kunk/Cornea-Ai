#!/usr/bin/env python
# LIBRARY ONLY (2026-10-19).  This module is retained because
# `scripts/analyze_trajectory_continuity.py` and `scripts/sens_umap_fig.py`
# import its locked palette and save helpers.  Its figure-producing entry
# point is SUPERSEDED: every manuscript figure now has a dedicated
# `scripts/figure*.py` producer.  Do not run this file to regenerate
# manuscript figures; doing so overwrites the rebuilt artwork with the old
# canvases and re-introduces one-figure-multiple-producers.

"""Generate publication-quality RESULTS figures and tables for the Paper1
manuscript (Ophthalmology Science / Elsevier style).

Reads the Tier-B external-test evaluation outputs produced by
``scripts/run_tier_b_eval.py``:

    artifacts/paper1_tier_b_eval/results_{lambda}_{seed}.json
    artifacts/paper1_tier_b_eval/summary.csv

and renders the results figures for the manuscript:

Outputs (figures/paper1/):
  figure5_external_eval.pdf/.png         — external test overview (2x2, 7x7 in):
      A  UMAP colored by predicted cell type (best model; falls back to the
         ground-truth harmonized cell type when per-cell predictions are not
         yet stored — the eval pipeline does not persist per-cell labels)
      B  UMAP colored by donor
      C  Row-normalized 4x4 confusion-matrix heatmap (best model)
      D  Per-class F1 grouped bar chart with error bars across seeds
  figure5_donor_macro_f1.pdf/.png        — primary endpoint (1x2, 7x3.5 in):
      A  Donor-level macro-F1 per arm with bootstrap 95% CI error bars; the
         internal HVG/PCA (0.7787) and frozen-Geneformer (0.8834) baselines are
         drawn as dashed reference lines (labeled *internal*); external test
         baselines HVG/PCA (0.6192) and frozen-Geneformer (0.4421) are drawn
         as dash-dot reference lines (labeled *ext*)
      B  Per-donor F1 strip plot (individual donor scores per arm x seed)
  figure7_continuity_metrics.pdf/.png    — continuity analysis (1x3, 10x3.5 in):
      A  Layer adjacency accuracy (LAA)   B  differentiation gradient Spearman rho
      C  expected calibration error (ECE)
  figure2_layerwise_rho.pdf/.png         — layerwise differentiation-gradient Spearman rho
      (single panel, 7x3.5 in): x = transformer layers 1-12; frozen-Geneformer
      as a thick dark solid line, naive fine-tune (lambda=0) as three seed
      lines in the blue hue family (the CAFT lambda=0.5 family is retired
      from the paper); key points annotated (naive L3 peak, L12 collapse,
      frozen L12 peak 0.806)
  figure2_layerwise_rho.csv              — all 4 models x 12 transformer-layer
      rho values (long format), for the data-availability statement
  table_t3_primary_endpoint.csv          — Arm, macro-F1 (95% CI), per-donor F1
  table_t4_per_class_metrics.csv         — Arm, Class, P/R/F1, Support
  table_t5_secondary_endpoints.csv       — Arm, LAA, rho, ECE (mean across seeds)
  table_t7_seed_reproducibility.csv      — Arm, Seed, macro-F1, LAA, rho, ECE

Notes
-----
* Okabe-Ito colorblind-safe palette; Arial (TrueType, msttcorefonts); Liberation Sans kept as a
  metric-equivalent fallback only.
* Figures are saved as vector PDF (editable text, ``pdf.fonttype=42``) plus
  300-dpi PNG.  Each PDF carries provenance metadata in its Title field.
* Missing data is handled gracefully: arms without results JSON are skipped,
  and panels with no data render a placeholder note instead of crashing.
* Reference: Li D-Q, Kim S, et al. Ocul Surf. 2021;20:20-32.
  doi:10.1016/j.jtos.2020.12.004  (PMID: 33388438)  — GSE153515 limbal atlas.
  IOVS 2021;62(9):36. doi:10.1167/iovs.62.9.36  (PMID: 34297801).

Usage
-----
    /home/dq/venv-ds/bin/python3 scripts/generate_paper1_results_figures.py
    /home/dq/venv-ds/bin/python3 scripts/generate_paper1_results_figures.py \
        --results_dir artifacts/paper1_tier_b_eval --output_dir figures/paper1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures" / "paper1"
DEFAULT_EXTERNAL_H5AD = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
DEFAULT_LAYERWISE_RHO_JSON = (
    PROJECT_ROOT / "artifacts" / "paper1_layerwise_rho" / "summary.json"
)
# v6: ordinal-head arms live in a separate evaluation directory (same schema
# as the tier-B JSONs: results_{lambda}_{seed}.json).  Corrected ECE
# (half-open [lo,hi) binning, audited) for the archived CE arms is read from
# artifacts/paper1_pipeline_audit/ece_corrected.json.
DEFAULT_ORDINAL_DIR = PROJECT_ROOT / "artifacts" / "paper1_ordinal_eval"
DEFAULT_ECE_CORRECTED_JSON = (
    PROJECT_ROOT / "artifacts" / "paper1_pipeline_audit" / "ece_corrected.json"
)

# Okabe-Ito colorblind-safe palette
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_BLUEGREEN = "#009E73"
OI_ORANGE = "#E69F00"
OI_SKYBLUE = "#56B4E9"
OI_PINK = "#CC79A7"
OI_YELLOW = "#F0E442"
OI_GRAY = "#999999"

EVAL_CLASSES = ["LSC", "TAC", "basal", "differentiated"]

CELLTYPE_COLORS = {
    "LSC": OI_BLUE,
    "TAC": OI_VERMILLION,
    "basal": OI_BLUEGREEN,
    "differentiated": OI_ORANGE,
    "NON_LAYER": OI_GRAY,
}
DONOR_COLORS = {"donor_1": OI_BLUE, "donor_2": OI_ORANGE}

# Fine-tuned arms (lambda -> display name / bar color).  The CAFT arm
# (lambda=0.5) is retired from the paper: its continuity term never executed
# (dead-code defect -> trained pure cross-entropy), so only the naive
# lambda=0 arm remains in this registry.  Ordinal-head arms (ordinal-naive /
# ordinal-CAFT) are tracked separately in ORDINAL_ARM_LABELS below.
ARM_LAMBDAS = [0.0]
ARM_LABELS = {0.0: "Fine-tune \u03bb=0"}
ARM_COLORS = {0.0: OI_BLUE}

# ---------------------------------------------------------------------------
# v6 unified arm -> color mapping (shared by every Paper 1 figure):
#   frozen = dark gray  |  naive = blue  |  ordinal(-naive) = green  |
#   ordinal-CAFT = orange (ordinal family second slot)  |  AA-FT /
#   soft-Spearman / flow = remaining Okabe-Ito (sky / pink / dark-yellow).
#   CAFT (vermillion) is retired from the paper: its continuity term never
#   executed (dead-code defect -> trained pure cross-entropy).
# ---------------------------------------------------------------------------
ARM_COLOR_FROZEN = "#333333"
ARM_COLOR_NAIVE = OI_BLUE            # #0072B2
ARM_COLOR_ORDINAL_NAIVE = OI_BLUEGREEN   # #009E73
ARM_COLOR_ORDINAL_CAFT = OI_ORANGE       # #E69F00
ARM_COLOR_AAFT = OI_SKYBLUE          # #56B4E9
ARM_COLOR_SS = OI_PINK               # #CC79A7
ARM_COLOR_FLOW = "#C5A500"           # darkened Okabe-Ito yellow (readable on white)

# Ordinal-head arms (CORN; separate evaluation dir).
ORDINAL_ARM_LABELS = {0.0: "ordinal-naive", 0.5: "ordinal-CAFT"}
ORDINAL_ARM_COLORS = {0.0: ARM_COLOR_ORDINAL_NAIVE, 0.5: ARM_COLOR_ORDINAL_CAFT}

# Internal baselines (training-split values, NOT external test). Drawn as
# dashed reference lines when no external-eval JSON exists for them.
# v6 palette: frozen = #333333, HVG/PCA = #999999 (identical to the unified
# arm mapping used by every other Paper 1 figure).
INTERNAL_BASELINES = [
    {
        "key": "HVG/PCA",
        "label": "HVG/PCA + logreg (internal)",
        "macro_f1": 0.7787,
        "color": OI_GRAY,
    },
    {
        "key": "Frozen GF",
        "label": "scFM frozen Geneformer (internal)",
        "macro_f1": 0.8834,
        "color": ARM_COLOR_FROZEN,
    },
]

# External test baselines (donor-level macro-F1 on GSE153515, produced by
# scripts/run_tier_b_baselines.py). Loaded from results_dir at runtime; the
# entries below define filename, label, and color only.
EXTERNAL_BASELINE_FILES = {
    "HVG/PCA": "results_hvg_pca.json",
    "Frozen GF": "results_frozen_gf.json",
}
EXTERNAL_BASELINE_LABELS = {
    "HVG/PCA": "HVG/PCA (ext)",
    "Frozen GF": "Frozen GF (ext)",
}
EXTERNAL_BASELINE_COLORS = {
    "HVG/PCA": OI_GRAY,
    "Frozen GF": ARM_COLOR_FROZEN,
}

RESULT_FILE_RE = re.compile(r"^results_(\d+(?:\.\d+)?)_(\d+)\.json$")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # embed TrueType (editable text in vector PDF)
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#000000",
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    """Place a bold uppercase panel label (8 pt, v8fix hierarchy) at the
    top-left of `ax`.

    The letter is anchored just above the panel title's rendered top edge so
    it can never collide with the (possibly long) title or the top y-tick
    labels.  Falls back to a fixed offset when the title is empty or the
    canvas cannot be drawn yet.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tb = ax.title.get_window_extent(renderer)
        if tb.height > 0:
            y_disp = tb.y1 + 7  # 7 px gap above the title
            x_disp = ax.get_position().x0 * fig.bbox.width - 0.02 * fig.bbox.width
            xf, yf = fig.transFigure.inverted().transform((x_disp, y_disp))
            fig.text(xf, yf, label, fontsize=8, fontweight="bold",
                     va="bottom", ha="left")
            return
    except Exception:  # noqa: BLE001 - fall back to a fixed offset
        pass
    pos = ax.get_position()
    fig.text(
        pos.x0 - 0.06,
        pos.y1 + 0.03,
        label,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def corner_panel_label(fig: plt.Figure, ax: plt.Axes, label: str,
                       size: float = 8.0) -> None:
    """8 pt bold panel tag inside the axes top-left corner (0.1 in from the
    left edge, 0.15 in from the top) — v8fix / Cell Press convention for
    single-panel figures."""
    fig.canvas.draw()
    bb = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_in = bb.width * fig_w
    ax_h_in = bb.height * fig_h
    ax.text(0.1 / ax_w_in, 1.0 - 0.15 / ax_h_in, label,
            transform=ax.transAxes, fontsize=size, fontweight="bold",
            ha="left", va="top")


def figure_metadata(title: str) -> dict[str, str]:
    """Provenance metadata embedded in the PDF Title field (uns-like)."""
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "Title": title,
        "Author": "Corneal-FM (Paper1, Ophthalmology Science)",
        "Subject": "Tier-B external-test evaluation, GSE153515 "
                   f"(generated {generated} UTC)",
        "Keywords": "Geneformer; donor-level macro-F1; GSE153515",
        "Creator": "scripts/generate_paper1_results_figures.py",
    }


def save_figure(
    fig: plt.Figure, name: str, out_dir: Path, metadata: dict[str, str] | None = None
) -> list[Path]:
    """Save vector PDF (with provenance metadata) + 300-dpi PNG."""
    out_pdf = out_dir / f"{name}.pdf"
    out_png = out_dir / f"{name}.png"
    if metadata:  # PNG backend does not support a metadata kwarg
        fig.savefig(out_pdf, dpi=300, metadata=metadata, facecolor="white")
    else:
        fig.savefig(out_pdf, dpi=300, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    return [out_pdf, out_png]


def no_data_note(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color=OI_GRAY, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Data loading (defensive: supports the JSON written by run_tier_b_eval.py and
# the stylized {"donor_level_macro_f1": {"mean": ...}} layout described in the
# figure specification)
# ---------------------------------------------------------------------------


def _normalize_primary(pe: Any) -> dict[str, Any]:
    if not isinstance(pe, dict):
        return {}
    nested = pe.get("donor_level_macro_f1")
    if isinstance(nested, dict):
        return {
            "macro_f1": nested.get("mean", pe.get("macro_f1")),
            "ci_lower": nested.get("ci_lower", pe.get("ci_lower")),
            "ci_upper": nested.get("ci_upper", pe.get("ci_upper")),
            "per_donor": nested.get("per_donor", pe.get("per_donor")),
        }
    return {k: pe.get(k) for k in ("macro_f1", "ci_lower", "ci_upper", "per_donor")}


def _per_class_dict(per_class: Any) -> dict[str, dict[str, float]]:
    if isinstance(per_class, dict):
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in per_class.items()}
    if isinstance(per_class, list):
        out: dict[str, dict[str, float]] = {}
        for d in per_class:
            if isinstance(d, dict) and d.get("class"):
                out[str(d["class"])] = d
        return out
    return {}


def _confusion_array(confusion: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(confusion, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size > 0:
            return arr
    except (TypeError, ValueError):
        pass
    return None


def _normalize_record(raw: dict[str, Any], lam: float, seed: int) -> dict[str, Any]:
    primary = _normalize_primary(raw.get("primary_endpoint"))
    se = raw.get("secondary_endpoints") or {}
    return {
        "lambda": lam,
        "seed": seed,
        "macro_f1": primary.get("macro_f1"),
        "ci_lower": primary.get("ci_lower"),
        "ci_upper": primary.get("ci_upper"),
        "per_donor": primary.get("per_donor") or {},
        "per_class": _per_class_dict(raw.get("per_class")),
        "confusion": _confusion_array(raw.get("confusion")),
        "secondary": {
            "laa": se.get("layer_adjacency_accuracy"),
            "rho": se.get("marker_gradient_rank_correlation"),
            "ece": se.get("expected_calibration_error"),
        },
        "provenance": raw.get("provenance") or {},
    }


def _load_results(results_dir: Path) -> dict[tuple[float, int], dict[str, Any]]:
    """Load every results_{lam}_{seed}.json under results_dir."""
    records: dict[tuple[float, int], dict[str, Any]] = {}
    for p in sorted(results_dir.glob("results_*.json")):
        m = RESULT_FILE_RE.match(p.name)
        if not m:
            continue
        lam, seed = float(m.group(1)), int(m.group(2))
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [WARN] could not read {p.name}: {exc}; skipping.")
            continue
        rec = _normalize_record(raw, lam, seed)
        rec["_path"] = p
        records[(lam, seed)] = rec
    return records


def _load_external_baselines(results_dir: Path) -> list[dict[str, Any]]:
    """Load external test baseline macro-F1 from results_hvg_pca.json and
    results_frozen_gf.json (same schema as the per-arm results JSON).

    Returns a list of dicts with keys: key, label, macro_f1, color. Missing or
    unreadable files are skipped with a warning (graceful degradation).
    """
    baselines: list[dict[str, Any]] = []
    for key, fname in EXTERNAL_BASELINE_FILES.items():
        p = results_dir / fname
        if not p.is_file():
            print(f"  [WARN] external baseline {fname} not found; skipping")
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [WARN] could not read {fname}: {exc}; skipping")
            continue
        primary = _normalize_primary(raw.get("primary_endpoint"))
        macro_f1 = _numeric(primary.get("macro_f1"))
        if macro_f1 is None:
            print(f"  [WARN] {fname} has no donor-level macro-F1; skipping")
            continue
        baselines.append(
            {
                "key": key,
                "label": EXTERNAL_BASELINE_LABELS[key],
                "macro_f1": macro_f1,
                "color": EXTERNAL_BASELINE_COLORS[key],
            }
        )
    return baselines


def _numeric(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _load_arm_frame(
    results_dir: Path, records: dict[tuple[float, int], dict[str, Any]]
) -> pd.DataFrame:
    """Per-seed frame: arm lambda, seed, primary + secondary metrics.

    Rows present in summary.csv but missing from the JSON results are merged
    in (they carry metric columns but no per-class / per-donor detail).
    """
    rows: list[dict[str, Any]] = []
    for (lam, seed), rec in sorted(records.items()):
        rows.append(
            {
                "lambda": lam,
                "seed": seed,
                "macro_f1": _numeric(rec["macro_f1"]),
                "ci_lower": _numeric(rec["ci_lower"]),
                "ci_upper": _numeric(rec["ci_upper"]),
                "laa": _numeric(rec["secondary"]["laa"]),
                "rho": _numeric(rec["secondary"]["rho"]),
                "ece": _numeric(rec["secondary"]["ece"]),
                "per_donor": rec["per_donor"],
                "per_class": rec["per_class"],
            }
        )

    summary_csv = results_dir / "summary.csv"
    if summary_csv.is_file():
        try:
            sdf = pd.read_csv(summary_csv)
        except (OSError, pd.errors.ParserError) as exc:
            print(f"  [WARN] could not read {summary_csv.name}: {exc}; ignoring.")
            sdf = pd.DataFrame()
        for _, r in sdf.iterrows():
            lam = _numeric(r.get("lambda"))
            seed = int(r.get("seed")) if _numeric(r.get("seed")) is not None else None
            if (
                lam is None
                or seed is None
                or (lam, seed) in records
                or lam == 0.5  # retired CAFT rows: continuity term never ran
            ):
                continue
            rows.append(
                {
                    "lambda": lam,
                    "seed": seed,
                    "macro_f1": _numeric(r.get("macro_f1")),
                    "ci_lower": _numeric(r.get("ci_lower")),
                    "ci_upper": _numeric(r.get("ci_upper")),
                    "laa": _numeric(r.get("LAA")),
                    "rho": _numeric(r.get("marker_gradient_rho")),
                    "ece": _numeric(r.get("ECE")),
                    "per_donor": {},
                    "per_class": {},
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["arm_label"] = df["lambda"].map(ARM_LABELS)
    return df.sort_values(["lambda", "seed"]).reset_index(drop=True)


def _best_key(records: dict[tuple[float, int], dict[str, Any]]) -> tuple[float, int] | None:
    """Best-model checkpoint: lowest remaining lambda (0.0 after the CAFT
    retirement), then the lowest seed for that lambda."""
    if not records:
        return None
    for lam in ARM_LAMBDAS:
        seeds = sorted(s for (k, s) in records if k == lam)
        if seeds:
            return (lam, seeds[0])
    return sorted(records.keys())[0]


def _stats(values: list[float | None]) -> tuple[float | None, float | None, int]:
    """NaN-safe (mean, std, n); std is None when n < 2."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None, None, 0
    mean = float(arr.mean())
    std = float(arr.std(ddof=0)) if arr.size > 1 else None
    return mean, std, int(arr.size)


# ---------------------------------------------------------------------------
# Figure 4 — external test evaluation overview (2x2)
# ---------------------------------------------------------------------------


def _resolve_prediction_column(adata: Any, best_lam: float | None) -> str | None:
    """Find an obs column holding per-cell model predictions, if one exists.

    The Tier-B eval pipeline currently does not persist per-cell predictions;
    this hook lets a future pipeline (e.g. predictions_{lam}_{seed}.csv saved
    into obs) light up the "predicted cell type" panel without code changes.
    """
    candidates = [c for c in adata.obs.columns if "pred" in c.lower()]
    if not candidates:
        return None
    if best_lam is not None:
        needle = f"lam{best_lam}".replace("lambda", "lam")
        for c in candidates:
            if needle in c.lower():
                return c
    return candidates[0]


def _category_colors(values: np.ndarray, known: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    palette = [OI_BLUE, OI_VERMILLION, OI_BLUEGREEN, OI_ORANGE,
               OI_SKYBLUE, OI_PINK, OI_GRAY]
    cats = sorted({str(v) for v in values})
    cmap = {c: known.get(c, palette[i % len(palette)]) for i, c in enumerate(cats)}
    out = np.array([cmap[str(v)] for v in values])
    return out, cats


def _draw_umap(
    fig: plt.Figure,
    ax: plt.Axes,
    umap: np.ndarray,
    colors: np.ndarray,
    cats: list[str],
    cat_colors: dict[str, str],
    title: str,
    label: str,
    legend_order: list[str] | None = None,
) -> None:
    ax.scatter(umap[:, 0], umap[:, 1], c=colors, s=1.2, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Locked manuscript cell order LSC -> TAC -> basal -> differentiated
    # (remaining categories, e.g. NON_LAYER, after them).
    order = legend_order or cats
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_colors[c],
                   markersize=6, label=c)
        for c in order
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.85,
              fontsize=7, handletextpad=0.3, borderpad=0.4)
    ax.set_title(title, fontsize=9, fontweight="bold")
    panel_label(fig, ax, label)


def _load_reference_per_class(
    results_dir: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-class metrics for the two reference arms (HVG/PCA, frozen GF) from
    their tier-B baseline JSONs (single runs, no seed dimension)."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key, fname in EXTERNAL_BASELINE_FILES.items():
        p = results_dir / fname
        if not p.is_file():
            print(f"  [WARN] reference per-class {fname} not found; skipping")
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [WARN] could not read {fname}: {exc}; skipping")
            continue
        pc = _per_class_dict(raw.get("per_class"))
        if pc:
            out[key] = pc
    return out


def figure4(
    external_h5ad: Path,
    df: pd.DataFrame,
    records: dict[tuple[float, int], dict[str, Any]],
    best_key: tuple[float, int] | None,
    out_dir: Path,
    ordinal_records: dict[tuple[float, int], dict[str, Any]] | None = None,
    reference_per_class: dict[str, dict[str, dict[str, float]]] | None = None,
) -> list[Path]:
    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(
        2, 2, left=0.075, right=0.97, top=0.93, bottom=0.075,
        wspace=0.22, hspace=0.28,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    best_lam = best_key[0] if best_key else None

    # ---- Panels A & B: UMAP (predicted cell type / donor) ----
    import anndata as ad

    adata = None
    try:
        adata = ad.read_h5ad(str(external_h5ad), backed="r")
        umap = np.asarray(adata.obsm["X_umap"])
        donor = np.asarray(adata.obs["donor_id"].astype(str))

        # Panel A: cell-type labels (external test).  The Tier-B eval pipeline
        # does not persist per-cell predictions, so this panel shows the
        # ground-truth harmonized labels (v6 caption wording).
        pred_col = _resolve_prediction_column(adata, best_lam)
        if pred_col is not None:
            labels = np.asarray(adata.obs[pred_col].astype(str))
        else:
            labels = np.asarray(adata.obs["cell_type_harmonized"].astype(str))
        colors, cats = _category_colors(labels, CELLTYPE_COLORS)
        cat_colors = dict(zip(cats, [CELLTYPE_COLORS.get(c, OI_GRAY) for c in cats]))
        title_a = "Cell-type labels (external test)"
        # Locked legend order: LSC -> TAC -> basal -> differentiated (rest after).
        legend_order = (["LSC", "TAC", "basal", "differentiated"]
                        + [c for c in cats if c not in ("LSC", "TAC", "basal",
                                                        "differentiated")])
        _draw_umap(fig, ax_a, umap, colors, cats, cat_colors, title_a, "A",
                   legend_order=legend_order)

        # Panel B: donor.
        dcolors, donors = _category_colors(donor, DONOR_COLORS)
        donor_cmap = dict(zip(donors, [DONOR_COLORS.get(d, OI_GRAY) for d in donors]))
        _draw_umap(fig, ax_b, umap, dcolors, donors, donor_cmap, "Donor (batch)", "B")
    except Exception as e:  # noqa: BLE001 — keep rendering remaining panels
        print(f"  [WARN] Figure 4 panels A/B failed: {e}")
        no_data_note(ax_a, "UMAP unavailable")
        no_data_note(ax_b, "UMAP unavailable")
        panel_label(fig, ax_a, "A")
        panel_label(fig, ax_b, "B")
    finally:
        if adata is not None:
            adata.file.close()

    # ---- Panel C: row-normalized confusion matrix (best model) ----
    try:
        if best_key is not None:
            conf = records[best_key]["confusion"]
            if conf is not None and conf.shape == (4, 4):
                mat = conf.astype(float)
                row_sums = mat.sum(axis=1, keepdims=True)
                norm = np.divide(mat, row_sums, out=np.zeros_like(mat),
                                 where=row_sums > 0)
                best_label = ARM_LABELS.get(best_lam, "")
                sns.heatmap(
                    norm, ax=ax_c, annot=True, fmt=".2f", cmap="Blues",
                    vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                    xticklabels=EVAL_CLASSES, yticklabels=EVAL_CLASSES,
                    annot_kws={"fontsize": 7.5},
                    cbar_kws={"label": "Fraction (row-normalized)",
                              "shrink": 0.85},
                )
                ax_c.set_xlabel("Predicted layer", fontsize=8)
                ax_c.set_ylabel("True layer", fontsize=8)
                ax_c.tick_params(axis="x", rotation=0)
                cb = ax_c.collections[0].colorbar
                cb.ax.tick_params(labelsize=7, length=2, pad=1)
                cb.set_label("Fraction (row-normalized)", fontsize=7)
                ax_c.set_title(f"Confusion matrix — {best_label} "
                              f"(seed {best_key[1]})", fontsize=9,
                              fontweight="bold")
            else:
                no_data_note(ax_c, "Confusion matrix unavailable")
        else:
            no_data_note(ax_c, "Confusion matrix unavailable")
        panel_label(fig, ax_c, "C")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure 4 panel C failed: {e}")
        no_data_note(ax_c, f"panel C failed:\n{e}")
        panel_label(fig, ax_c, "C")

    # ---- Panel D: per-class F1, grouped by arm, error bars across seeds ----
    try:
        classes = EVAL_CLASSES
        x = np.arange(len(classes))
        ordinal_records = ordinal_records or {}
        reference_per_class = reference_per_class or {}

        # (key, label, color, per-class records or None).  CAFT (lambda=0.5)
        # is retired from the paper (dead-code defect), so only the naive CE
        # arm is drawn from the tier-B records.
        pc_sources: list[dict[str, Any]] = []
        for lam, label, color in [
            (0.0, "naive", ARM_COLOR_NAIVE),
        ]:
            recs = [rec for (k, _), rec in sorted(records.items()) if k == lam]
            if recs:
                pc_sources.append(
                    {"label": label, "color": color,
                     "records": recs, "seeded": True}
                )
        for lam, label, color, hatch in [
            (0.0, "ordinal-naive", ARM_COLOR_ORDINAL_NAIVE, "\\\\"),
            (0.5, "ordinal-CAFT", ARM_COLOR_ORDINAL_CAFT, None),
        ]:
            recs = [rec for (k, _), rec in sorted(ordinal_records.items())
                    if k == lam]
            if recs:
                pc_sources.append(
                    {"label": label, "color": color, "hatch": hatch,
                     "records": recs, "seeded": True}
                )
        for key, label, color in [
            ("HVG/PCA", "HVG/PCA", OI_GRAY),
            ("Frozen GF", "frozen GF", ARM_COLOR_FROZEN),
        ]:
            pc = reference_per_class.get(key)
            if pc:
                pc_sources.append(
                    {"label": label, "color": color,
                     "records": [{"per_class": pc}], "seeded": False}
                )

        width = 0.8 / max(1, len(pc_sources))
        n_plotted = 0
        class_support: dict[str, int] = {}
        class_max: dict[str, float] = {c: 0.0 for c in classes}
        for i, src in enumerate(pc_sources):
            means, stds = [], []
            for cls in classes:
                vals: list[float | None] = []
                for rec in src["records"]:
                    pc = rec.get("per_class") or {}
                    entry = pc.get(cls)
                    if isinstance(entry, dict):
                        v = _numeric(entry.get("f1"))
                        if v is not None:
                            vals.append(v)
                        # Same locked test set across arms -> class support is
                        # identical for every arm; read it from any record.
                        if cls not in class_support:
                            n_s = _numeric(entry.get("support"))
                            if n_s is not None:
                                class_support[cls] = int(n_s)
                m, s, _ = _stats(vals)
                means.append(m if m is not None else 0.0)
                stds.append(s if s is not None else 0.0)
                class_max[cls] = max(class_max[cls], means[-1])
            offset = (i - (len(pc_sources) - 1) / 2) * width
            ax_d.bar(x + offset, means, width, color=src["color"],
                     label=src["label"], yerr=stds, capsize=2.0,
                     hatch=src.get("hatch"),
                     error_kw={"elinewidth": 0.7}, linewidth=0.6,
                     edgecolor="#ffffff")
            n_plotted += 1
        print(f"  [self-check] Figure 5 panel D: {n_plotted} arm bar group(s) "
              f"drawn: {[s['label'] for s in pc_sources]}")
        # QA-locked per-class support (external test, n = 14,775 epithelial
        # cells): LSC 3,423 / TAC 171 / basal 6,698 / differentiated 4,483.
        for c, fallback in {"LSC": 3423, "TAC": 171, "basal": 6698,
                            "differentiated": 4483}.items():
            class_support.setdefault(c, fallback)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(classes)
        ax_d.set_ylabel("Per-class F1")
        ax_d.set_ylim(0, 1.08)
        ax_d.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if n_plotted:
            # Per-class support (n) annotations above each class group.
            for ci, cls in enumerate(classes):
                if cls == "TAC":
                    continue  # n is annotated inside the F1 = 0 label below
                ax_d.text(ci, class_max[cls] + 0.035,
                          f"n = {class_support[cls]:,}", ha="center",
                          va="bottom", fontsize=6.5, color="#555555")
            # TAC column: explicit zero-F1 label (bars are ~0 in every arm;
            # n = 171 = 1.2% of the test set).  Four-class axes unchanged.
            n_tac = class_support["TAC"]
            total = sum(class_support.values())
            ax_d.text(1, 0.100, "F1 = 0", ha="center", va="bottom",
                      fontsize=7, fontweight="bold", color="#333333")
            ax_d.text(1, 0.036, f"(n = {n_tac:,}, {n_tac / total * 100:.1f}%)",
                      ha="center", va="bottom", fontsize=6.5, color="#555555")
            # Legend below the panel (single row set; the panel top is fully
            # occupied by the per-class support annotations).  The TAC F1 = 0
            # note lives in the caption (tex), not in the figure.
            ax_d.legend(loc="upper center", frameon=False, fontsize=6.5,
                        ncol=3, bbox_to_anchor=(0.5, -0.05),
                        labelspacing=0.35, handletextpad=0.4,
                        columnspacing=0.7)
        else:
            no_data_note(ax_d, "No per-class results")
        ax_d.set_title("Per-class F1 by arm (mean \u00b1 SD across seeds)",
                       fontsize=9, fontweight="bold")
        panel_label(fig, ax_d, "D")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure 4 panel D failed: {e}")
        no_data_note(ax_d, f"panel D failed:\n{e}")
        panel_label(fig, ax_d, "D")

    return save_figure(
        fig, "figure5_external_eval", out_dir,
        figure_metadata("Figure 4 - External test evaluation overview (GSE153515)"),
    )


# ---------------------------------------------------------------------------
# Figure 5 — donor-level macro-F1 comparison (1x2)
# ---------------------------------------------------------------------------


def figure5(
    df: pd.DataFrame,
    out_dir: Path,
    external_baselines: list[dict[str, Any]] | None = None,
) -> list[Path]:
    fig = plt.figure(figsize=(6.5, 3.25))
    gs = fig.add_gridspec(
        1, 2, left=0.085, right=0.975, top=0.85, bottom=0.20, wspace=0.38,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    arms_with_data = [lam for lam in ARM_LAMBDAS if not df[df["lambda"] == lam].empty]

    # ---- Panel A: macro-F1 per arm + internal/external reference lines ----
    try:
        bar_x, bar_m, bar_err, bar_labels = [], [], [], []
        for i, lam in enumerate(arms_with_data):
            sub = df[df["lambda"] == lam]
            m, _, n = _stats(sub["macro_f1"].tolist())
            if m is None:
                continue
            lo = sub["ci_lower"].dropna().mean()
            up = sub["ci_upper"].dropna().mean()
            err_low = (m - lo) if pd.notna(lo) else 0.0
            err_up = (up - m) if pd.notna(up) else 0.0
            bar_x.append(i)
            bar_m.append(m)
            bar_err.append([[err_low], [err_up]])
            bar_labels.append(ARM_LABELS[lam])
            ax_a.bar(i, m, 0.55, color=ARM_COLORS[lam], label=ARM_LABELS[lam],
                    yerr=[[err_low], [err_up]], capsize=3, linewidth=0.5,
                    edgecolor="#ffffff", error_kw={"elinewidth": 0.8})
            ax_a.text(i, m + max(err_up, 0.01) + 0.01, f"{m:.3f}",
                     ha="center", va="bottom", fontsize=7)

        # Internal baselines as dashed reference lines (labeled internal).
        for bl in INTERNAL_BASELINES:
            ax_a.axhline(bl["macro_f1"], color=bl["color"], ls="--", lw=1.2,
                        zorder=0)
            ax_a.text(ax_a.get_xlim()[1], bl["macro_f1"],
                     f" {bl['macro_f1']:.3f}  ", ha="right", va="bottom",
                     fontsize=6.5, color=bl["color"])

        # External test baselines as dash-dot reference lines (labeled ext).
        ext_baselines = external_baselines or []
        for bl in ext_baselines:
            ax_a.axhline(bl["macro_f1"], color=bl["color"], ls="-.",
                        lw=1.2, zorder=0)
            ax_a.text(ax_a.get_xlim()[1], bl["macro_f1"],
                     f" {bl['macro_f1']:.3f}  ", ha="right", va="bottom",
                     fontsize=6.5, color=bl["color"])

        if bar_x:
            ax_a.set_xticks(bar_x)
            ax_a.set_xticklabels(bar_labels)
            all_vals = (bar_m
                        + [bl["macro_f1"] for bl in INTERNAL_BASELINES]
                        + [bl["macro_f1"] for bl in ext_baselines])
            lo = max(0.0, min(all_vals) - 0.08)
            hi = min(1.08, max(all_vals) + 0.06)
            ax_a.set_ylim(lo, hi)
            ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            ax_a.set_xlim(-0.5, 0.5)
            all_vals = ([bl["macro_f1"] for bl in INTERNAL_BASELINES]
                        + [bl["macro_f1"] for bl in ext_baselines])
            ax_a.set_ylim(max(0.0, min(all_vals) - 0.08),
                         min(1.0, max(all_vals) + 0.06))
        ax_a.set_ylabel("Donor-level macro-F1")
        ax_a.set_title("Macro-F1 per arm (95% CI)", fontsize=9, fontweight="bold")
        ref_handles = [
            plt.Line2D([0], [0], color=bl["color"], ls="--", lw=1.2,
                       label=bl["label"])
            for bl in INTERNAL_BASELINES
        ] + [
            plt.Line2D([0], [0], color=bl["color"], ls="-.", lw=1.2,
                       label=bl["label"])
            for bl in ext_baselines
        ]
        ax_a.legend(handles=ref_handles, loc="lower left", fontsize=6.5,
                   frameon=False)
        panel_label(fig, ax_a, "A")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure 5 panel A failed: {e}")
        no_data_note(ax_a, f"panel A failed:\n{e}")
        panel_label(fig, ax_a, "A")

    # ---- Panel B: per-donor F1 strip plot ----
    try:
        points: list[dict[str, Any]] = []
        for lam in arms_with_data:
            sub = df[df["lambda"] == lam]
            for _, r in sub.iterrows():
                pd_ = r["per_donor"]
                if not isinstance(pd_, dict):
                    continue
                for donor, f1v in pd_.items():
                    f = _numeric(f1v)
                    if f is None:
                        continue
                    points.append({"arm": lam, "donor": str(donor), "f1": f})
        if points:
            pdf_ = pd.DataFrame(points)
            rng = np.random.default_rng(0)
            donors = sorted(pdf_["donor"].unique())
            donor_colors = {d: DONOR_COLORS.get(d, OI_GRAY) for d in donors}
            for i, lam in enumerate(arms_with_data):
                sub = pdf_[pdf_["arm"] == lam]
                xs = rng.uniform(-0.14, 0.14, size=len(sub)) + i
                ax_b.scatter(
                    xs, sub["f1"].values,
                    c=[donor_colors[d] for d in sub["donor"].values],
                    s=14, alpha=0.9, linewidths=0.3, edgecolor="#ffffff",
                    zorder=3,
                )
                for d in donors:
                    dv = sub[sub["donor"] == d]["f1"].values
                    if dv.size:
                        ax_b.hlines(dv.mean(), i - 0.3, i + 0.3,
                                   color=donor_colors[d], lw=1.2, zorder=4)
            ax_b.set_xticks(range(len(arms_with_data)))
            ax_b.set_xticklabels([ARM_LABELS[lam] for lam in arms_with_data])
            ax_b.set_ylim(0, 1.08)
            ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            handles = [
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=donor_colors[d], markersize=6,
                           label=d)
                for d in donors
            ]
            ax_b.legend(handles=handles, loc="lower left", fontsize=7,
                       frameon=False)
        else:
            no_data_note(ax_b, "No per-donor results")
        ax_b.set_ylabel("Donor-level F1")
        ax_b.set_title("Per-donor F1 by arm", fontsize=9, fontweight="bold")
        panel_label(fig, ax_b, "B")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Figure 5 panel B failed: {e}")
        no_data_note(ax_b, f"panel B failed:\n{e}")
        panel_label(fig, ax_b, "B")

    return save_figure(
        fig, "figure5_donor_macro_f1", out_dir,
        figure_metadata("Figure 5 - Donor-level macro-F1 comparison"),
    )


# ---------------------------------------------------------------------------
# Figure 6 — continuity metrics under task adaptation (1x3)
# ---------------------------------------------------------------------------


def _bar_with_stats(ax: plt.Axes, x: float, values: list[float | None],
                    color: str, label: str, hatch: str | None = None) -> None:
    m, s, _ = _stats(values)
    if m is None:
        return
    yerr = [[s or 0.0]] if s is not None else None
    ax.bar(x, m, 0.5, color=color, label=label, yerr=yerr, capsize=3,
           linewidth=0.6, edgecolor="#ffffff", hatch=hatch,
           error_kw={"elinewidth": 0.8})
    ax.text(x, (m or 0.0) + (s or 0.0) + 0.01, f"{m:.3f}", ha="center",
            va="bottom", fontsize=6.5)


def _ylim(values: list[float | None], floor_zero: bool) -> tuple[float, float]:
    v = [x for x in values if x is not None]
    if not v:
        return (0.0, 1.0)
    lo, hi = min(v), max(v)
    span = hi - lo
    # minimum pad keeps 7-pt value labels inside narrow-range panels (e.g. LAA
    # all ~0.974-0.977).
    pad = max(span * 0.2, 0.02)
    lo2 = max(0.0, lo - pad) if floor_zero else lo - pad
    return (lo2, hi + pad)


def _load_continuity_frame(results_dir: Path, label_map: dict[float, str]
                           ) -> pd.DataFrame:
    """Per-seed continuity metrics (LAA / rho / ECE) for a results dir whose
    results_{lambda}_{seed}.json files share the tier-B schema."""
    rows: list[dict[str, Any]] = []
    records = _load_results(results_dir)
    for (lam, seed), rec in sorted(records.items()):
        rows.append(
            {
                "arm": label_map.get(lam, f"\u03bb={lam}"),
                "lambda": lam,
                "seed": seed,
                "laa": _numeric(rec["secondary"]["laa"]),
                "rho": _numeric(rec["secondary"]["rho"]),
                "ece": _numeric(rec["secondary"]["ece"]),
            }
        )
    return pd.DataFrame(rows)


def _load_corrected_ece(path: Path) -> dict[tuple[str, int], float]:
    """arm/seed -> corrected ECE from ece_corrected.json (half-open binning)."""
    if not path.is_file():
        print(f"  [WARN] corrected-ECE file {path} not found; "
              "using archived JSON ECE")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [WARN] could not read {path}: {exc}; using archived ECE")
        return {}
    out: dict[tuple[str, int], float] = {}
    for entry in (raw.get("ece_corrected") or {}).values():
        arm = (entry.get("arm") or "").lower()
        seed = entry.get("seed")
        val = entry.get("new_ece_fixed")
        if arm and seed is not None and val is not None:
            out[(arm, int(seed))] = float(val)
    return out


def figure6(ce_df: pd.DataFrame, ordinal_df: pd.DataFrame,
            ece_corrected: dict[tuple[str, int], float],
            out_dir: Path) -> list[Path]:
    """Continuity metrics under task adaptation (v6): LAA / rho / ECE for the
    archived CE arm (naive) and the two ordinal-head arms (ordinal-naive,
    ordinal-CAFT).  The CAFT arm is retired from the paper (dead-code defect:
    its continuity term never executed -> trained pure cross-entropy), so it
    no longer appears here.  ECE for the archived CE arm uses the audited
    corrected values (half-open binning); ordinal ECE comes from the ordinal
    evaluation JSONs (already corrected convention)."""
    metrics = [
        ("laa", "LAA (layer adjacency)", ARM_COLOR_NAIVE, True),
        ("rho", "Differentiation gradient \u03c1 (Spearman)",
         ARM_COLOR_ORDINAL_NAIVE, False),
        # color slot is informational (bars take their color from the arm
        # entry); gray replaced the retired CAFT vermillion.
        ("ece", "Expected calibration error (ECE)", OI_GRAY, True),
    ]
    arms: list[dict[str, Any]] = [
        {
            "key": "naive", "label": "naive (CE)", "color": ARM_COLOR_NAIVE,
            "sub": ce_df[ce_df["lambda"] == 0.0], "hatch": None,
            "ece_src": "corrected",
        },
        {
            "key": "ord_naive", "label": "ordinal-naive",
            "color": ARM_COLOR_ORDINAL_NAIVE,
            "sub": ordinal_df[ordinal_df["lambda"] == 0.0], "hatch": "\\\\",
            "ece_src": "json",
        },
        {
            "key": "lora", "label": "LoRA",
            "color": OI_SKYBLUE,
            "sub": pd.DataFrame(), "hatch": None,
            "ece_src": "json",
            "manual_rho": 0.834,
        },
    ]

    fig = plt.figure(figsize=(6.5, 2.275))
    gs = fig.add_gridspec(
        1, 3, left=0.06, right=0.985, top=0.80, bottom=0.30, wspace=0.25,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax, (col, title, _color, floor_zero) in zip(axes, metrics):
        try:
            vals: list[float | None] = []
            for i, arm in enumerate(arms):
                sub = arm["sub"]
                # Handle manual arms (LoRA, CORN+LoRA) with no sub DataFrame
                if sub.empty and "manual_rho" in arm and col == "rho":
                    arm_vals = [arm["manual_rho"]]
                    _bar_with_stats(ax, i, arm_vals, arm["color"], arm["label"],
                                    hatch=arm["hatch"])
                    vals.extend(arm_vals)
                    continue
                if sub.empty:
                    continue
                if col == "ece" and arm["ece_src"] == "corrected":
                    arm_vals = [
                        ece_corrected.get((arm["key"], int(s)))
                        for s in sub["seed"].tolist()
                    ]
                    arm_vals = [v for v in arm_vals if v is not None]
                else:
                    arm_vals = sub[col].dropna().tolist()
                _bar_with_stats(ax, i, arm_vals, arm["color"], arm["label"],
                                hatch=arm["hatch"])
                vals.extend(arm_vals)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([a["label"] for a in arms], fontsize=6.5,
                               rotation=0)
            ax.set_xlim(-0.8, 2.4)
            if col == "rho":
                # Cap ticks to a nice grid so AutoLocator does not extend
                # tick labels (e.g. -0.25/1.25) past the canvas top/bottom.
                ax.set_ylim(-0.25, 1.05)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            elif col == "ece":
                ax.set_ylim(0.0, 0.22)
                ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
            else:
                ax.set_ylim(*_ylim(vals, floor_zero))
            ax.set_title(title, fontsize=8, fontweight="bold")
            panel_label(fig, ax, {"laa": "A", "rho": "B", "ece": "C"}[col])
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] Figure 6 panel ({title}) failed: {e}")
            no_data_note(ax, f"panel failed:\n{e}")
            panel_label(fig, ax, {"laa": "A", "rho": "B", "ece": "C"}[col])
    plotted = [a["label"] for a in arms if not a["sub"].empty or "manual_rho" in a]
    print(f"  [self-check] Figure 7: {len(plotted)} adapted-arm bar(s) drawn: "
          f"{plotted}")
    # Single shared legend, centred below the three panels (arm colour/hatch
    # key; the x tick labels carry the same arm names).
    handles = [
        Patch(facecolor=a["color"], edgecolor="#ffffff", lw=0.6,
              hatch=a["hatch"] or "", label=a["label"])
        for a in arms if not a["sub"].empty or "manual_rho" in a
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.055), ncol=len(handles),
                   fontsize=6.5, frameon=False, columnspacing=1.2,
                   handlelength=1.2, handleheight=0.9)
    fig.text(
        0.5, 0.012,
        "Data are represented as three-seed mean \u00b1 SD (error bars); "
        "ordinal heads preserve \u03c1 (0.833/0.828) vs naive "
        "\u2248 \u22120.008; ordinal values from the ordinal-head "
        "evaluation runs.",
        ha="center", va="bottom", fontsize=6.0, color="#555555",
    )
    return save_figure(
        fig, "figure7_continuity_metrics", out_dir,
        figure_metadata("Continuity metrics under task adaptation "
                        "(LAA, differentiation-gradient rho, ECE)"),
    )


# ---------------------------------------------------------------------------
# Figure 7 — layerwise differentiation-gradient Spearman rho (single panel)
# ---------------------------------------------------------------------------

# Model key -> (group, lambda, seed) for the layerwise-rho summary. Group
# membership drives color family and legend labelling; lambda/seed are None
# for the frozen checkpoint.  The CAFT (lam0.5_*) models are retired from the
# paper (dead-code defect in the continuity term), so only frozen + naive FT
# remain; the layerwise summary data holds no ordinal-arm entries.
_LAYERWISE_MODEL_GROUPS: dict[str, tuple[str, float | None, int | None]] = {
    "frozen_gf": ("frozen", None, None),
    "lam0.0_seed42": ("naive", 0.0, 42),
    "lam0.0_seed123": ("naive", 0.0, 123),
    "lam0.0_seed456": ("naive", 0.0, 456),
}

# Seed -> shade within the group hue family.  v6 unified mapping: naive =
# blue family (Okabe-Ito blue -> sky -> dark navy); the frozen line uses a
# dark gray.  (The CAFT vermillion family was retired with the CAFT arm.)
_NAIVE_SEED_COLORS = {42: "#0072B2", 123: "#56B4E9", 456: "#003C66"}
_FROZEN_RHO_COLOR = "#333333"

_GROUP_LABELS = {
    "frozen": "frozen GF",
    "naive": "naive FT \u03bb=0 (seeds 42/123/456)",
}
_GROUP_HANDLE_COLORS = {
    "frozen": _FROZEN_RHO_COLOR,
    "naive": OI_BLUE,
}


def _load_layerwise_rho(path: Path) -> dict[str, Any] | None:
    """Load the layerwise-rho summary.json; None on any failure (graceful)."""
    if not path.is_file():
        print(f"  [WARN] layerwise rho summary {path} not found; "
              "Figure 7 skipped")
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [WARN] could not read {path}: {exc}; Figure 7 skipped")
        return None
    return raw


def figure7(layerwise_rho: Path, out_dir: Path) -> list[Path]:
    """Layerwise differentiation-gradient Spearman rho per transformer layer.

    Reads the layerwise summary produced by the layerwise-rho pipeline
    (artifacts/paper1_layerwise_rho/summary.json): 4 models x 13 layers
    (frozen GF + 3 naive-FT seeds; the CAFT models are retired from the
    paper).  The embedding_output layer (index 0) is always NaN and is
    dropped, so the x axis runs over transformer layers 1-12.
    """
    raw = _load_layerwise_rho(layerwise_rho)
    if raw is None:
        return []
    layer_labels = raw.get("layer_labels") or []
    models = raw.get("models") or {}
    if not isinstance(models, dict) or not models:
        print("  [WARN] no models matrix in layerwise rho summary; "
              "Figure 7 skipped")
        return []

    # Drop the embedding_output column (index 0, all NaN) -> layers 1..12.
    layers = [
        i for i, lab in enumerate(layer_labels)
        if isinstance(lab, str) and lab.startswith("transformer_layer_")
    ]
    if not layers:
        print("  [WARN] no transformer_layer_* columns found; "
              "Figure 7 skipped")
        return []

    fig = plt.figure(figsize=(6.5, 3.25))
    fig.subplots_adjust(bottom=0.14, left=0.10, right=0.985, top=0.94)
    ax = fig.add_subplot(111)

    csv_rows: list[dict[str, Any]] = []
    drawn_groups: set[str] = set()
    all_rho: list[float] = []
    for key, (group, lam, seed) in _LAYERWISE_MODEL_GROUPS.items():
        vals = models.get(key)
        if not isinstance(vals, list) or len(vals) <= max(layers):
            print(f"  [WARN] model '{key}' has no usable rho vector; skipping")
            continue
        xs = list(layers)
        ys = [_numeric(vals[i]) for i in layers]
        if all(v is None for v in ys):
            print(f"  [WARN] model '{key}' rho values are all NaN; skipping")
            continue
        ys = [np.nan if v is None else v for v in ys]
        if group == "frozen":
            color, lw, ls = _FROZEN_RHO_COLOR, 2.4, "-"
        elif group == "naive":
            color = _NAIVE_SEED_COLORS.get(seed, OI_BLUEGREEN)
            lw, ls = 1.2, "-"
        else:  # no group remains after the CAFT retirement (defensive)
            color, lw, ls = OI_GRAY, 1.2, "-"
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=3)
        drawn_groups.add(group)
        for xi, yi in zip(xs, ys):
            if yi is None:
                continue
            all_rho.append(yi)
            csv_rows.append(
                {
                    "model": key,
                    "group": group,
                    "lambda": lam,
                    "seed": seed,
                    "layer": xi,
                    "rho": yi,
                }
            )

    if not drawn_groups:
        print("  [WARN] no rho curves drawn; Figure 7 skipped")
        plt.close(fig)
        return []

    # Reference line at rho = 0 (boundary above/below random ordering).
    ax.axhline(0.0, color=OI_GRAY, ls=":", lw=0.7, zorder=1)

    # ---- Key-point annotations (observer feedback: concise, move narrative to caption)
    annot_kw = dict(
        fontsize=6.5, color="#555555",
        arrowprops=dict(arrowstyle="->", lw=0.6, color="#555555"),
    )
    ax.annotate("L3 peak", xy=(3, 0.84), xytext=(3.6, 0.93),
                ha="left", **annot_kw)
    ax.annotate("collapse", xy=(12, -0.04), xytext=(9.6, -0.155),
                ha="left", **annot_kw)
    ax.annotate("frozen: 0.806", xy=(12, 0.806), xytext=(10.2, 0.90),
                ha="left", **annot_kw)
    # head-init point marker (minimal: just a dot + label)
    ax.plot(11, 0.78, "o", ms=5, mfc="#009E73", mec="#009E73", mew=0.5, zorder=5)
    ax.text(11.2, 0.78, "head-init", fontsize=6.5, color="#009E73",
            va="center", ha="left")

    # ---- Axes & legend
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(list(range(1, 13)))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    rho_lo, rho_hi = min(all_rho), max(all_rho)
    ax.set_ylim(rho_lo - 0.20, rho_hi + 0.10)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Differentiation gradient \u03c1 (Spearman)")
    ax.set_title("Layerwise differentiation-gradient Spearman \u03c1",
                 fontsize=9, fontweight="bold")
    # v6: thirteen hidden states exist (embedding_output + L1..L12); the
    # embedding output (L0) is a constant CLS vector -> rho undefined (NaN),
    # so the x axis covers transformer layers 1-12.
    ax.text(0.985, 0.015, "L0 (embedding output) excluded \u2014 "
            "constant CLS (NaN)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#555555")

    # v8fix: 8 pt bold panel tag at the axes top-left corner.
    corner_panel_label(fig, ax, "A")

    handles = [
        plt.Line2D([0], [0], color=_GROUP_HANDLE_COLORS[g], ls="-",
                   lw=2.4 if g == "frozen" else 1.4, label=_GROUP_LABELS[g])
        for g in ("frozen", "naive")
        if g in drawn_groups
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False)

    # ---- CSV export (long format: all 4 models x 12 layers)
    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_path = out_dir / "figure2_layerwise_rho.csv"
        csv_df.to_csv(csv_path, index=False, float_format="%.17g")
        print(f"  {len(csv_rows)} layerwise rho values -> {csv_path.name}")

    n_curves = len(csv_rows) // len(layers) if layers else 0
    print(f"  {len(drawn_groups)} model group(s) drawn: {sorted(drawn_groups)}, "
          f"layers 1-{len(layers)}")
    print(f"  [self-check] Figure 2: {n_curves} curve(s), "
          f"{len(drawn_groups)} group(s) ({sorted(drawn_groups)}) "
          f"- no CAFT family")

    return save_figure(
        fig, "figure2_layerwise_rho", out_dir,
        figure_metadata("Figure 7 - Layerwise differentiation-gradient Spearman rho"),
    )


# ---------------------------------------------------------------------------
# Tables (CSV)
# ---------------------------------------------------------------------------


def _donor_mean(sub: pd.DataFrame, donor: str) -> float | None:
    vals = []
    for pd_ in sub["per_donor"].tolist():
        if isinstance(pd_, dict) and donor in pd_:
            v = _numeric(pd_[donor])
            if v is not None:
                vals.append(v)
    m, _, _ = _stats(vals)
    return m


def table_t3(df: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    donors: list[str] = []
    for pd_ in df["per_donor"].tolist():
        if isinstance(pd_, dict):
            for d in pd_:
                if d not in donors:
                    donors.append(str(d))
    for lam in ARM_LAMBDAS:
        sub = df[df["lambda"] == lam]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "arm": ARM_LABELS[lam],
            "source": "external (GSE153515)",
            "macro_f1": round(float(sub["macro_f1"].mean()), 4),
            "ci_lower": round(float(sub["ci_lower"].dropna().mean()), 4),
            "ci_upper": round(float(sub["ci_upper"].dropna().mean()), 4),
            "n_seeds": int(len(sub)),
        }
        for d in donors:
            m = _donor_mean(sub, d)
            row[d] = None if m is None else round(m, 4)
        rows.append(row)
    for bl in INTERNAL_BASELINES:
        rows.append(
            {
                "arm": bl["label"],
                "source": "internal (training split)",
                "macro_f1": bl["macro_f1"],
                "ci_lower": "",
                "ci_upper": "",
                "n_seeds": "",
            }
        )
    out = out_dir / "table_t3_primary_endpoint.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def table_t4(records: dict[tuple[float, int], dict[str, Any]], out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for lam in ARM_LAMBDAS:
        subs = [rec for (k, _), rec in records.items() if k == lam]
        if not subs:
            continue
        for cls in EVAL_CLASSES:
            prec, rec_, f1s, sup = [], [], [], []
            for rec in subs:
                pc = rec["per_class"].get(cls) or {}
                if pc.get("precision") is not None:
                    prec.append(_numeric(pc["precision"]))
                    rec_.append(_numeric(pc["recall"]))
                    f1s.append(_numeric(pc["f1"]))
                s = _numeric(pc.get("support"))
                if s is not None:
                    sup.append(s)
            m_p, _, _ = _stats(prec)
            m_r, _, _ = _stats(rec_)
            m_f, _, _ = _stats(f1s)
            m_s, _, _ = _stats(sup)
            rows.append(
                {
                    "arm": ARM_LABELS[lam],
                    "class": cls,
                    "precision": None if m_p is None else round(m_p, 4),
                    "recall": None if m_r is None else round(m_r, 4),
                    "f1": None if m_f is None else round(m_f, 4),
                    "support": None if m_s is None else round(m_s, 1),
                    "n_seeds": len(subs),
                }
            )
    out = out_dir / "table_t4_per_class_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def table_t5(df: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for lam in ARM_LAMBDAS:
        sub = df[df["lambda"] == lam]
        if sub.empty:
            continue
        m_laa, _, _ = _stats(sub["laa"].dropna().tolist())
        m_rho, _, _ = _stats(sub["rho"].dropna().tolist())
        m_ece, _, _ = _stats(sub["ece"].dropna().tolist())
        rows.append(
            {
                "arm": ARM_LABELS[lam],
                "LAA": None if m_laa is None else round(m_laa, 4),
                "marker_gradient_rho": None if m_rho is None else round(m_rho, 4),
                "ECE": None if m_ece is None else round(m_ece, 4),
                "n_seeds": int(len(sub)),
            }
        )
    out = out_dir / "table_t5_secondary_endpoints.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def table_t7(df: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for lam in ARM_LAMBDAS:
        sub = df[df["lambda"] == lam]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "arm": ARM_LABELS[lam],
                    "seed": int(r["seed"]),
                    "macro_f1": None if pd.isna(r["macro_f1"]) else round(float(r["macro_f1"]), 4),
                    "LAA": None if pd.isna(r["laa"]) else round(float(r["laa"]), 4),
                    "marker_gradient_rho": None if pd.isna(r["rho"]) else round(float(r["rho"]), 4),
                    "ECE": None if pd.isna(r["ece"]) else round(float(r["ece"]), 4),
                }
            )
    out = out_dir / "table_t7_seed_reproducibility.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir", default=str(DEFAULT_RESULTS_DIR),
        help="Directory holding results_{lambda}_{seed}.json + summary.csv "
             f"(default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output_dir", default=str(DEFAULT_OUT_DIR),
        help=f"Figure/table output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--external_h5ad", default=str(DEFAULT_EXTERNAL_H5AD),
        help="Annotated GSE153515 h5ad for UMAP panels A/B "
             f"(default: {DEFAULT_EXTERNAL_H5AD})",
    )
    parser.add_argument(
        "--layerwise_rho", default=str(DEFAULT_LAYERWISE_RHO_JSON),
        help="Layerwise differentiation-gradient rho summary.json "
             f"(default: {DEFAULT_LAYERWISE_RHO_JSON})",
    )
    parser.add_argument(
        "--ordinal_dir", default=str(DEFAULT_ORDINAL_DIR),
        help="Directory holding ordinal-head evaluation results "
             f"(default: {DEFAULT_ORDINAL_DIR})",
    )
    parser.add_argument(
        "--ece_corrected", default=str(DEFAULT_ECE_CORRECTED_JSON),
        help="Corrected ECE json (half-open binning audit) "
             f"(default: {DEFAULT_ECE_CORRECTED_JSON})",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    ordinal_dir = Path(args.ordinal_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"Loading results from {results_dir} ...")
    records = _load_results(results_dir)
    # CAFT (lambda=0.5) is retired from the paper: its continuity term never
    # executed (dead-code defect -> trained pure cross-entropy), so skip those
    # result files entirely.  Ordinal 0.5 (ordinal-CAFT) is a different arm
    # and is kept — it is loaded separately from ordinal_dir below.
    n_caft = sum(1 for (lam, _seed) in records if lam == 0.5)
    records = {k: v for k, v in records.items() if k[0] != 0.5}
    if n_caft:
        print(f"  [self-check] dropped {n_caft} retired CAFT (lambda=0.5) "
              f"result(s)")
    print(f"  {len(records)} results JSON(s) loaded")
    df = _load_arm_frame(results_dir, records)
    print(f"  {len(df)} per-seed arm rows")
    if df.empty:
        print("  [WARN] no evaluation results found; figures will show "
              "reference lines / placeholder notes only.")

    print(f"Loading ordinal-head results from {ordinal_dir} ...")
    ordinal_records = _load_results(ordinal_dir)
    print(f"  {len(ordinal_records)} ordinal results JSON(s) loaded")
    ordinal_df = _load_continuity_frame(ordinal_dir, ORDINAL_ARM_LABELS)
    print(f"  {len(ordinal_df)} ordinal per-seed rows")

    ece_corrected = _load_corrected_ece(Path(args.ece_corrected))
    print(f"  {len(ece_corrected)} corrected-ECE value(s) loaded")

    reference_per_class = _load_reference_per_class(results_dir)
    print(f"  per-class metrics for {len(reference_per_class)} reference arm(s)")

    best_key = _best_key(records)
    print(f"  best-model checkpoint: {best_key}")

    external_baselines = _load_external_baselines(results_dir)
    print(f"  {len(external_baselines)} external baseline(s) loaded")

    generated: list[Path] = []

    print("\n[Figure 4] External test evaluation overview ...")
    generated += figure4(Path(args.external_h5ad), df, records, best_key,
                         out_dir, ordinal_records=ordinal_records,
                         reference_per_class=reference_per_class)

    print("\n[Figure 5] Donor-level macro-F1 comparison ...")
    # obs-1 audit item 15: figure5_donor_macro_f1 is an orphan (user-approved
    # archiving); the figure is archived and not regenerated.
    # generated += figure5(df, out_dir, external_baselines)

    print("\n[Figure 6] Continuity metrics under task adaptation ...")
    generated += figure6(df, ordinal_df, ece_corrected, out_dir)

    print("\n[Figure 7] Layerwise differentiation-gradient rho ...")
    generated += figure7(Path(args.layerwise_rho), out_dir)

    print("\n[Tables]")
    generated.append(table_t3(df, out_dir))
    generated.append(table_t4(records, out_dir))
    generated.append(table_t5(df, out_dir))
    generated.append(table_t7(df, out_dir))

    print("\n=== Summary of generated files ===")
    for p in generated:
        size_kb = p.stat().st_size / 1024
        try:
            display = p.relative_to(PROJECT_ROOT)
        except ValueError:
            display = p
        print(f"  {display}  ({size_kb:.1f} kB)")

    # Self-check: PNG dimensions of the three main-text figures (300 dpi).
    import matplotlib.image as mpimg

    print("\n=== Self-check (main-text figures, 300-dpi PNG) ===")
    expected = {
        "figure2_layerwise_rho": "frozen + naive curves (no CAFT family)",
        "figure5_external_eval": "panel D: 5 arm bars (naive, ordinal-naive, "
                                 "ordinal-CAFT, HVG/PCA, frozen)",
        "figure7_continuity_metrics": "3 adapted-arm bars (naive, "
                                      "ordinal-naive, ordinal-CAFT)",
    }
    for name, note in expected.items():
        png = out_dir / f"{name}.png"
        if not png.is_file():
            print(f"  [FAIL] {name}.png not generated")
            continue
        img = mpimg.imread(png)
        print(f"  {name}.png: {img.shape[1]} x {img.shape[0]} px  ({note})")

    print(f"\nDone. {len(generated)} files written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
