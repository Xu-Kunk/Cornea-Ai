#!/usr/bin/env python
"""Generate publication-quality RESULTS figures and tables for the Paper1
manuscript (Ophthalmology Science / Elsevier style).

Reads the Tier-B external-test evaluation outputs produced by
``scripts/run_tier_b_eval.py``:

    artifacts/paper1_tier_b_eval/results_{lambda}_{seed}.json
    artifacts/paper1_tier_b_eval/summary.csv

and renders the results figures for the manuscript:

Outputs (figures/paper1/):
  figure4_external_test_eval.pdf/.png    — external test overview (2x2, 7x7 in):
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
  figure6_caft_continuity.pdf/.png       — continuity analysis (1x3, 10x3.5 in):
      A  Layer adjacency accuracy (LAA)   B  differentiation gradient Spearman rho
      C  expected calibration error (ECE)
  figure7_layerwise_rho.pdf/.png         — layerwise differentiation-gradient Spearman rho
      (single panel, 7x3.5 in): x = transformer layers 1-12; frozen-Geneformer
      as a thick dark solid line, naive fine-tune (lambda=0) and CAFT
      (lambda=0.5) each as three seed lines in the blue-green / pink hue
      families; key points annotated (naive L3 peak, L12 collapse, frozen
      L12 peak 0.806)
  figure7_layerwise_rho.csv              — all 7 models x 12 transformer-layer
      rho values (long format), for the data-availability statement
  table_t3_primary_endpoint.csv          — Arm, macro-F1 (95% CI), per-donor F1
  table_t4_per_class_metrics.csv         — Arm, Class, P/R/F1, Support
  table_t5_secondary_endpoints.csv       — Arm, LAA, rho, ECE (mean across seeds)
  table_t7_seed_reproducibility.csv      — Arm, Seed, macro-F1, LAA, rho, ECE

Notes
-----
* Okabe-Ito colorblind-safe palette; Liberation Sans (metric-compatible with
  Arial) is used because Arial/Helvetica are not installed on this system.
* Figures are saved as vector PDF (editable text, ``pdf.fonttype=42``) plus
  300-dpi PNG.  Each PDF carries provenance metadata in its Title field.
* Missing data is handled gracefully: arms without results JSON are skipped,
  and panels with no data render a placeholder note instead of crashing.
* Reference: Li D-Q, Kim S, et al. Ocul Surf. 2021;20:20-32.
  doi:10.1016/j.jtos.2020.12.004  (PMID: 33388438)  — GSE153515 limbal atlas.
  IOVS 2021;62(9):36. doi:10.1167/iovs.62.9.36  (PMID: 34297801).

Usage
-----
    python3 scripts/generate_paper1_results_figures.py
    python3 scripts/generate_paper1_results_figures.py \
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

# Fine-tuned CAFT arms (lambda -> display name / bar color).
ARM_LAMBDAS = [0.0, 0.5]
ARM_LABELS = {0.0: "Fine-tune \u03bb=0", 0.5: "CAFT \u03bb=0.5"}
ARM_COLORS = {0.0: OI_BLUE, 0.5: OI_VERMILLION}

# Internal baselines (training-split values, NOT external test). Drawn as
# dashed reference lines when no external-eval JSON exists for them.
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
        "color": OI_SKYBLUE,
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
    "HVG/PCA": OI_BLUEGREEN,
    "Frozen GF": OI_PINK,
}

RESULT_FILE_RE = re.compile(r"^results_(\d+(?:\.\d+)?)_(\d+)\.json$")

# ---------------------------------------------------------------------------
# Styling
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
            "pdf.fonttype": 42,  # embed TrueType (editable text in vector PDF)
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
    """Provenance metadata embedded in the PDF Title field (uns-like)."""
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "Title": title,
        "Author": "Corneal-FM (Paper1, Ophthalmology Science)",
        "Subject": "Tier-B external-test evaluation, GSE153515 "
                   f"(generated {generated} UTC)",
        "Keywords": "CAFT; Geneformer; donor-level macro-F1; GSE153515",
        "Creator": "scripts/generate_paper1_results_figures.py",
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
            if lam is None or seed is None or (lam, seed) in records:
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
    """Best-model checkpoint: CAFT lambda=0.5 (lowest seed), else lambda=0."""
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
) -> None:
    ax.scatter(umap[:, 0], umap[:, 1], c=colors, s=1.2, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_colors[c],
                   markersize=6, label=c)
        for c in cats
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.85,
              fontsize=7, handletextpad=0.3, borderpad=0.4)
    ax.set_title(title, fontsize=9, fontweight="bold")
    panel_label(fig, ax, label)


def figure4(
    external_h5ad: Path,
    df: pd.DataFrame,
    records: dict[tuple[float, int], dict[str, Any]],
    best_key: tuple[float, int] | None,
    out_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(
        2, 2, left=0.05, right=0.97, top=0.955, bottom=0.075,
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

        # Panel A: predicted cell type (falls back to ground truth).
        pred_col = _resolve_prediction_column(adata, best_lam)
        if pred_col is not None:
            labels = np.asarray(adata.obs[pred_col].astype(str))
            src = "Predicted cell type"
        else:
            labels = np.asarray(adata.obs["cell_type_harmonized"].astype(str))
            src = "Cell type (predictions unavailable)"
        colors, cats = _category_colors(labels, CELLTYPE_COLORS)
        cat_colors = dict(zip(cats, [CELLTYPE_COLORS.get(c, OI_GRAY) for c in cats]))
        title_a = f"{src} — {'CAFT \u03bb=0.5' if best_lam == 0.5 else 'model'}"
        _draw_umap(fig, ax_a, umap, colors, cats, cat_colors, title_a, "A")

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
                    cbar_kws={"label": "Fraction (row-normalized)",
                              "shrink": 0.85},
                )
                ax_c.set_xlabel("Predicted layer", fontsize=8)
                ax_c.set_ylabel("True layer", fontsize=8)
                ax_c.tick_params(axis="x", rotation=0)
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
        n_arms = 0
        width = 0.38
        for i, lam in enumerate(ARM_LAMBDAS):
            sub = df[df["lambda"] == lam]
            if sub.empty:
                continue
            means, stds = [], []
            for cls in classes:
                vals: list[float | None] = []
                for pc in sub["per_class"].tolist():
                    if not isinstance(pc, dict):
                        continue
                    entry = pc.get(cls)
                    if isinstance(entry, dict):
                        v = _numeric(entry.get("f1"))
                        if v is not None:
                            vals.append(v)
                m, s, _ = _stats(vals)
                means.append(m if m is not None else 0.0)
                stds.append(s if s is not None else 0.0)
            offset = (i - (sum(1 for k in ARM_LAMBDAS
                               if not df[df["lambda"] == k].empty) - 1) / 2) * width
            ax_d.bar(x + offset, means, width, color=ARM_COLORS[lam],
                    label=ARM_LABELS[lam], yerr=stds, capsize=2.5,
                    error_kw={"elinewidth": 0.8}, linewidth=0.5,
                    edgecolor="#ffffff")
            n_arms += 1
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(classes)
        ax_d.set_ylabel("Per-class F1")
        ax_d.set_ylim(0, 1.05)
        if n_arms:
            ax_d.legend(loc="lower left", frameon=False, fontsize=7)
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
        fig, "figure4_external_test_eval", out_dir,
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
    fig = plt.figure(figsize=(7, 3.5))
    gs = fig.add_gridspec(
        1, 2, left=0.085, right=0.975, top=0.88, bottom=0.20, wspace=0.38,
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
            hi = min(1.0, max(all_vals) + 0.06)
            ax_a.set_ylim(lo, hi)
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
            ax_b.set_ylim(0, 1.05)
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
# Figure 6 — CAFT continuity analysis (1x3)
# ---------------------------------------------------------------------------


def _bar_with_stats(ax: plt.Axes, lam: float, values: list[float | None],
                    color: str, label: str) -> None:
    m, s, _ = _stats(values)
    if m is None:
        return
    yerr = [[s or 0.0]] if s is not None else None
    ax.bar(lam, m, 0.5, color=color, label=label, yerr=yerr, capsize=3,
           linewidth=0.5, edgecolor="#ffffff", error_kw={"elinewidth": 0.8})
    ax.text(lam, (m or 0.0) + (s or 0.0) + 0.01, f"{m:.3f}", ha="center",
            va="bottom", fontsize=7)


def _ylim(values: list[float | None], floor_zero: bool) -> tuple[float, float]:
    v = [x for x in values if x is not None]
    if not v:
        return (0.0, 1.0)
    lo, hi = min(v), max(v)
    pad = (hi - lo) * 0.2 if hi > lo else 0.05
    lo2 = max(0.0, lo - pad) if floor_zero else lo - pad
    return (lo2, hi + pad)


def figure6(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    metrics = [
        ("laa", "Layer adjacency accuracy (LAA)", OI_BLUE, True),
        ("rho", "Differentiation gradient \u03c1 (Spearman)", OI_BLUEGREEN, False),
        ("ece", "Expected calibration error (ECE)", OI_VERMILLION, True),
    ]
    arms_with_data = [lam for lam in ARM_LAMBDAS if not df[df["lambda"] == lam].empty]

    fig = plt.figure(figsize=(10, 3.5))
    gs = fig.add_gridspec(
        1, 3, left=0.06, right=0.985, top=0.88, bottom=0.20, wspace=0.30,
    )
    generated = []
    for ax, (col, title, color, floor_zero) in zip(
        [fig.add_subplot(gs[i]) for i in range(3)], metrics
    ):
        try:
            vals: list[float | None] = []
            for lam in arms_with_data:
                sub = df[df["lambda"] == lam]
                arm_vals = sub[col].dropna().tolist()
                _bar_with_stats(ax, lam, arm_vals, color, ARM_LABELS[lam])
                vals.extend(arm_vals)
            if arms_with_data:
                ax.set_xticks(arms_with_data)
                ax.set_xticklabels([ARM_LABELS[lam] for lam in arms_with_data])
                ax.set_ylim(*_ylim(vals, floor_zero))
            else:
                no_data_note(ax, "No results")
            ax.set_title(title, fontsize=9, fontweight="bold")
            panel_label(fig, ax, {"laa": "A", "rho": "B", "ece": "C"}[col])
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] Figure 6 panel ({title}) failed: {e}")
            no_data_note(ax, f"panel failed:\n{e}")
            panel_label(fig, ax, {"laa": "A", "rho": "B", "ece": "C"}[col])
    generated = save_figure(
        fig, "figure6_caft_continuity", out_dir,
        figure_metadata("Figure 6 - CAFT continuity analysis"),
    )
    return generated


# ---------------------------------------------------------------------------
# Figure 7 — layerwise differentiation-gradient Spearman rho (single panel)
# ---------------------------------------------------------------------------

# Model key -> (group, lambda, seed) for the layerwise-rho summary. Group
# membership drives color family and legend labelling; lambda/seed are None
# for the frozen checkpoint.
_LAYERWISE_MODEL_GROUPS: dict[str, tuple[str, float | None, int | None]] = {
    "frozen_gf": ("frozen", None, None),
    "lam0.0_seed42": ("naive", 0.0, 42),
    "lam0.0_seed123": ("naive", 0.0, 123),
    "lam0.0_seed456": ("naive", 0.0, 456),
    "lam0.5_seed42": ("CAFT", 0.5, 42),
    "lam0.5_seed123": ("CAFT", 0.5, 123),
    "lam0.5_seed456": ("CAFT", 0.5, 456),
}

# Seed -> shade within the group hue family (blue-green for naive fine-tune,
# pink for CAFT). The frozen line uses a dark gray.
_NAIVE_SEED_COLORS = {42: OI_BLUEGREEN, 123: "#5EC4AB", 456: "#005A40"}
_CAFT_SEED_COLORS = {42: OI_PINK, 123: "#E0A6C9", 456: "#9C4E73"}
_FROZEN_RHO_COLOR = "#333333"

_GROUP_LABELS = {
    "frozen": "frozen GF",
    "naive": "naive FT \u03bb=0 (seeds 42/123/456)",
    "CAFT": "CAFT \u03bb=0.5 (seeds 42/123/456)",
}
_GROUP_HANDLE_COLORS = {
    "frozen": _FROZEN_RHO_COLOR,
    "naive": OI_BLUEGREEN,
    "CAFT": OI_PINK,
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
    (artifacts/paper1_layerwise_rho/summary.json): 7 models x 13 layers.
    The embedding_output layer (index 0) is always NaN and is dropped, so the
    x axis runs over transformer layers 1-12.
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

    fig = plt.figure(figsize=(7, 3.5))
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
        else:  # CAFT
            color = _CAFT_SEED_COLORS.get(seed, OI_PINK)
            lw, ls = 1.2, "-"
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

    # ---- Key-point annotations (naive L3 peak, L12 collapse, frozen L12 peak)
    annot_kw = dict(
        fontsize=6.5, color="#555555",
        arrowprops=dict(arrowstyle="->", lw=0.6, color="#555555"),
    )
    ax.annotate("naive L3 peak", xy=(3, 0.84), xytext=(3.6, 0.93),
                ha="left", **annot_kw)
    ax.annotate("collapse at L12", xy=(12, -0.04), xytext=(9.6, -0.155),
                ha="left", **annot_kw)
    ax.annotate("frozen L12 peak: 0.806", xy=(12, 0.806), xytext=(10.2, 0.90),
                ha="left", **annot_kw)

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

    handles = [
        plt.Line2D([0], [0], color=_GROUP_HANDLE_COLORS[g], ls="-",
                   lw=2.4 if g == "frozen" else 1.4, label=_GROUP_LABELS[g])
        for g in ("frozen", "naive", "CAFT")
        if g in drawn_groups
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False)

    # ---- CSV export (long format: all 7 models x 12 layers)
    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_path = out_dir / "figure7_layerwise_rho.csv"
        csv_df.to_csv(csv_path, index=False, float_format="%.17g")
        print(f"  {len(csv_rows)} layerwise rho values -> {csv_path.name}")

    print(f"  {len(drawn_groups)} model group(s), layers 1-{len(layers)}")

    return save_figure(
        fig, "figure7_layerwise_rho", out_dir,
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
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"Loading results from {results_dir} ...")
    records = _load_results(results_dir)
    print(f"  {len(records)} results JSON(s) loaded")
    df = _load_arm_frame(results_dir, records)
    print(f"  {len(df)} per-seed arm rows")
    if df.empty:
        print("  [WARN] no evaluation results found; figures will show "
              "reference lines / placeholder notes only.")

    best_key = _best_key(records)
    print(f"  best-model checkpoint: {best_key}")

    external_baselines = _load_external_baselines(results_dir)
    print(f"  {len(external_baselines)} external baseline(s) loaded")

    generated: list[Path] = []

    print("\n[Figure 4] External test evaluation overview ...")
    generated += figure4(Path(args.external_h5ad), df, records, best_key, out_dir)

    print("\n[Figure 5] Donor-level macro-F1 comparison ...")
    generated += figure5(df, out_dir, external_baselines)

    print("\n[Figure 6] CAFT continuity analysis ...")
    generated += figure6(df, out_dir)

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
    print(f"\nDone. {len(generated)} files written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
