#!/usr/bin/env python
"""TAC sensitivity analysis for the Paper 1 Tier-B external benchmark (GSE153515).

The external test contains 14,775 limbal epithelial cells across 4 harmonized
classes: basal 6,698 / differentiated 4,483 / LSC 3,423 / TAC 171 (TAC = 1.2%,
an extreme class-imbalance case). Every adapted arm (naive FT lambda=0 and CAFT
lambda=0.5, seeds 42/123/456) achieves TAC F1 = 0.0 with 0 true positives; only
the frozen-Geneformer baseline recovers any TAC cells.

This script quantifies how sensitive the donor-level macro-F1 primary endpoint
is to the presence of the TAC class:

  (a) Full 4-class donor-level macro-F1 (must reproduce the locked
      ``sanity_gate_external.macro_f1`` in each percell_*.meta.json) —
      reused verbatim from :func:`corneal_fm.evaluation.donor_level_macro_f1`
      (n_boot=1000, seed=42).
  (b) TAC-excluded 3-class donor-level macro-F1 (classes {LSC, basal,
      differentiated}) using the SAME function and bootstrap protocol. True
      TAC cells are dropped; predicted TAC cells on remaining cells are
      treated exactly like NON_LAYER predictions in the 4-class analysis
      (out-of-labels -> FN, no FP) — identical sklearn semantics, verified.
  (c) Per-class precision / recall / F1 at BOTH the cell level (pooled over
      all cells) and the donor level (per-donor score averaged across donors),
      explicitly labelled in the outputs.

Outputs:
  tables/paper1/table_tac_sensitivity.csv  — tidy long-format numeric table
  figures/paper1/figureS9_tac_sensitivity.{pdf,png} — 2x2 publication figure
  (300-dpi PNG; rcParams/colour palette reused from
  scripts/generate_paper1_results_figures.py)

Usage:
    <venv>/bin/python3 scripts/analyze_tac_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt

# Style/colour reuse — the exact rcParams + Okabe-Ito palette used by
# scripts/generate_paper1_results_figures.py (imported, not duplicated).
import generate_paper1_results_figures as gfig  # noqa: E402

from corneal_fm.evaluation import donor_level_macro_f1  # noqa: E402
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans",
                                   "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
FIG_OUT_DIR = PROJECT_ROOT / "figures" / "paper1"
TABLE_OUT_DIR = PROJECT_ROOT / "tables" / "paper1"

CLASSES_4 = ["LSC", "TAC", "basal", "differentiated"]
CLASSES_3 = ["LSC", "basal", "differentiated"]  # TAC excluded

N_BOOT = 1000
BOOT_SEED = 42

# arm -> (display name, group, lambda, seed)
ARMS: list[tuple[str, str, float | None, int | None]] = [
    ("naive_s42", "naive", 0.0, 42),
    ("naive_s123", "naive", 0.0, 123),
    ("naive_s456", "naive", 0.0, 456),
    ("caft_s42", "caft", 0.5, 42),
    ("caft_s123", "caft", 0.5, 123),
    ("caft_s456", "caft", 0.5, 456),
    ("frozen_gf", "frozen", None, None),
]

ARM_DISPLAY = {
    "naive_s42": "naive s42",
    "naive_s123": "naive s123",
    "naive_s456": "naive s456",
    "caft_s42": "CAFT s42",
    "caft_s123": "CAFT s123",
    "caft_s456": "CAFT s456",
    "frozen_gf": "Frozen GF",
}

ARM_GROUP_COLORS = {
    "naive": gfig.OI_BLUE,
    "caft": gfig.OI_ORANGE,  # v8fix arm-color lock: CAFT = orange (was vermillion)
    "frozen": gfig.OI_GRAY,
}

DONOR_LABELS = {"donor_1": "GSM4646295", "donor_2": "GSM4646296"}


def _load_arm(arm: str) -> dict:
    """Load per-cell arrays for one arm from artifacts (read-only)."""
    path = ARTIFACT_DIR / f"percell_{arm}.npz"
    with np.load(path, allow_pickle=True) as d:
        return {
            "y_true": np.asarray(d["y_true_ext"]),
            "y_pred": np.asarray(d["y_pred_ext"]),
            "donors": np.asarray(d["donor_ext"]).astype(str),
        }


def _load_meta(arm: str) -> dict:
    path = ARTIFACT_DIR / f"percell_{arm}.npz.meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def donor_level_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    donors: np.ndarray,
    classes: list[str],
) -> dict[str, dict]:
    """Per-class P/R/F1 computed per donor, then averaged across donors.

    For each class and each donor, precision/recall/F1 are computed on that
    donor's cells (labels=[class]); the reported value is the mean across
    donors (donor-level). Support is the pooled cell count of the class.
    """
    from sklearn.metrics import precision_recall_fscore_support

    donor_ids = np.unique(donors)
    out: dict[str, dict] = {}
    for cls in classes:
        prec, rec, f1s = [], [], []
        for d in donor_ids:
            mask = donors == d
            p, r, f, _ = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], labels=[cls], zero_division=0
            )
            prec.append(float(p[0]))
            rec.append(float(r[0]))
            f1s.append(float(f[0]))
        out[cls] = {
            "precision": float(np.mean(prec)),
            "recall": float(np.mean(rec)),
            "f1": float(np.mean(f1s)),
            "support": int((y_true == cls).sum()),
        }
    return out


def analyze_arm(arm: str) -> dict:
    """Run endpoints (a) 4-class with TAC, (b) 3-class without TAC, (c) per-class."""
    data = _load_arm(arm)
    y_true, y_pred, donors = data["y_true"], data["y_pred"], data["donors"]

    # (a) Primary endpoint — full 4 classes, existing implementation reused.
    with_tac = donor_level_macro_f1(
        y_true, y_pred, donors, CLASSES_4, n_boot=N_BOOT, seed=BOOT_SEED
    )

    # (b) TAC-excluded endpoint — drop TRUE TAC cells, same function/protocol.
    keep = y_true != "TAC"
    without_tac = donor_level_macro_f1(
        y_true[keep], y_pred[keep], donors[keep],
        CLASSES_3, n_boot=N_BOOT, seed=BOOT_SEED,
    )

    # (c) Per-class metrics at both levels, for both class schemas.
    cell_4 = with_tac["per_class"]  # pooled cell-level (from donor_level_macro_f1)
    donor_4 = donor_level_per_class(y_true, y_pred, donors, CLASSES_4)
    cell_3 = without_tac["per_class"]  # pooled cell-level (3 classes)
    donor_3 = donor_level_per_class(
        y_true[keep], y_pred[keep], donors[keep], CLASSES_3
    )

    return {
        "arm": arm,
        "display": ARM_DISPLAY[arm],
        "n_cells": int(len(y_true)),
        "n_tac_cells": int((y_true == "TAC").sum()),
        "with_tac": {
            "macro_f1": with_tac["macro_f1"],
            "ci_lower": with_tac["ci_lower"],
            "ci_upper": with_tac["ci_upper"],
            "per_donor": with_tac["per_donor"],
            "per_class_cell": cell_4,
            "per_class_donor": donor_4,
        },
        "without_tac": {
            "macro_f1": without_tac["macro_f1"],
            "ci_lower": without_tac["ci_lower"],
            "ci_upper": without_tac["ci_upper"],
            "per_donor": without_tac["per_donor"],
            "per_class_cell": cell_3,
            "per_class_donor": donor_3,
        },
        "delta_macro_f1": without_tac["macro_f1"] - with_tac["macro_f1"],
    }


def sanity_check(arm: str, res: dict) -> dict:
    """Compare recomputed 4-class macro-F1 against the locked meta gate."""
    meta = _load_meta(arm)
    sg = meta.get("sanity_gate_external") or {}
    expected = float(sg["macro_f1"])
    got = res["with_tac"]["macro_f1"]
    diff = abs(got - expected)
    return {
        "arm": arm,
        "meta_macro_f1": expected,
        "recomputed_macro_f1": got,
        "abs_diff": diff,
        "within_1e-3": diff < 1e-3,
    }


# ---------------------------------------------------------------------------
# Table (tidy long format)
# ---------------------------------------------------------------------------

def build_table(results: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for arm, res in results.items():
        for endpoint in ("with_tac", "without_tac"):
            r = res[endpoint]
            n_donors = len(r["per_donor"])
            # macro-F1 row
            rows.append({
                "arm": res["display"],
                "endpoint": endpoint,
                "classes": "LSC,TAC,basal,differentiated" if endpoint == "with_tac"
                           else "LSC,basal,differentiated",
                "level": "donor",
                "class": "ALL",
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "support": res["n_cells"] if endpoint == "with_tac"
                           else res["n_cells"] - res["n_tac_cells"],
                "macro_f1": r["macro_f1"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "n_donors": n_donors,
            })
            # per-class rows at both levels
            for level, pc in (("cell", r["per_class_cell"]),
                              ("donor", r["per_class_donor"])):
                for cls, m in pc.items():
                    rows.append({
                        "arm": res["display"],
                        "endpoint": endpoint,
                        "classes": "LSC,TAC,basal,differentiated"
                                   if endpoint == "with_tac" else "LSC,basal,differentiated",
                        "level": level,
                        "class": cls,
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                        "support": m["support"],
                        "macro_f1": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "n_donors": n_donors,
                    })
    df = pd.DataFrame(rows)
    # Keep arms in canonical order (pandas categorical order).
    order = [res["display"] for _, res in results.items()]
    df["arm"] = pd.Categorical(df["arm"], categories=order, ordered=True)
    return df.sort_values(["arm", "endpoint", "level", "class"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure (style reused from generate_paper1_results_figures.py)
# ---------------------------------------------------------------------------

def _grouped_bars(
    ax: plt.Axes,
    x_labels: list[str],
    series: list[tuple[str, list[float], str]],
    ylabel: str,
    title: str,
    ci: list[list[tuple[float, float]]] | None = None,
) -> None:
    """Grouped bar chart: one cluster per x label, one bar per series.

    ``ci`` is a parallel list (one entry per series) of ``(lower, upper)``
    bootstrap 95% CI bounds aligned with each x position; drawn as asymmetric
    error bars.
    """
    n = len(x_labels)
    width = 0.8 / len(series)
    x = np.arange(n)
    for i, (name, vals, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        if ci is not None and i < len(ci):
            lo = [v - c[0] for v, c in zip(vals, ci[i])]
            hi = [c[1] - v for v, c in zip(vals, ci[i])]
            ax.bar(x + offset, vals, width, color=color, label=name,
                   yerr=[lo, hi], capsize=3.0, linewidth=0.6,
                   edgecolor="#ffffff", error_kw={"elinewidth": 0.7})
        else:
            ax.bar(x + offset, vals, width, color=color, label=name,
                   linewidth=0.6, edgecolor="#ffffff")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=6.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False)


def make_figure(results: dict[str, dict], out_dir: Path) -> list[Path]:
    arms = list(results.keys())
    x_labels = [ARM_DISPLAY[a] for a in arms]

    # ---- Panel A: donor-level macro-F1 with vs without TAC (bootstrap CI)
    with_vals = [results[a]["with_tac"]["macro_f1"] for a in arms]
    without_vals = [results[a]["without_tac"]["macro_f1"] for a in arms]
    ci_w = [(results[a]["with_tac"]["ci_lower"],
             results[a]["with_tac"]["ci_upper"]) for a in arms]
    ci_wo = [(results[a]["without_tac"]["ci_lower"],
              results[a]["without_tac"]["ci_upper"]) for a in arms]

    # ---- Panel C/D per-class F1, cell-level and donor-level (4 classes)
    cell_f1 = {c: [results[a]["with_tac"]["per_class_cell"][c]["f1"] for a in arms]
               for c in CLASSES_4}
    donor_f1 = {c: [results[a]["with_tac"]["per_class_donor"][c]["f1"] for a in arms]
                for c in CLASSES_4}

    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.985, top=0.93, bottom=0.10,
                          wspace=0.30, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- A: with-TAC vs without-TAC
    _grouped_bars(
        ax_a, x_labels,
        [("with TAC", with_vals, gfig.OI_BLUE),
         ("without TAC", without_vals, gfig.OI_BLUEGREEN)],
        "Donor-level macro-F1",
        "Macro-F1 with vs without TAC (95% CI)",
        ci=[ci_w, ci_wo],
    )
    ax_a.set_ylim(0.0, 1.0)
    for i, v in enumerate(with_vals):
        ax_a.text(i - 0.12, v + 0.012, f"{v:.3f}", fontsize=6.0, ha="center")
    for i, v in enumerate(without_vals):
        ax_a.text(i + 0.12, v + 0.012, f"{v:.3f}", fontsize=6.0, ha="center")
    gfig.panel_label(fig, ax_a, "A")

    # ---- B: TAC-exclusion delta (without − with)
    deltas = [results[a]["delta_macro_f1"] for a in arms]
    colors = [gfig.OI_VERMILLION if d >= 0 else gfig.OI_BLUE for d in deltas]
    ax_b.bar(np.arange(len(arms)), deltas, 0.6, color=colors, linewidth=0.6,
             edgecolor="#ffffff")
    ax_b.axhline(0.0, color=gfig.OI_GRAY, lw=0.8)
    ax_b.set_xticks(np.arange(len(arms)))
    ax_b.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=6.0)
    for i, d in enumerate(deltas):
        ax_b.text(i, d + (0.004 if d >= 0 else -0.008), f"{d:.3f}",
                  fontsize=6.0, ha="center")
    ax_b.set_ylabel("\u0394 macro-F1 (without TAC \u2212 with TAC)")
    ax_b.set_title("Macro-F1 change on TAC exclusion",
                   fontsize=7.5, fontweight="bold")
    gfig.panel_label(fig, ax_b, "B")

    # ---- C: per-class F1, cell level (pooled)
    _grouped_bars(
        ax_c, x_labels,
        [(c, cell_f1[c], gfig.CELLTYPE_COLORS[c]) for c in CLASSES_4],
        "Per-class F1 (cell level)",
        "Per-class F1, cell-level (pooled)",
    )
    ax_c.set_ylim(0.0, 1.0)
    gfig.panel_label(fig, ax_c, "C")

    # ---- D: per-class F1, donor level
    _grouped_bars(
        ax_d, x_labels,
        [(c, donor_f1[c], gfig.CELLTYPE_COLORS[c]) for c in CLASSES_4],
        "Per-class F1 (donor level)",
        "Per-class F1, donor-level (mean across donors)",
    )
    ax_d.set_ylim(0.0, 1.0)
    gfig.panel_label(fig, ax_d, "D")

    return gfig.save_figure(
        fig, "figureS9_tac_sensitivity", out_dir,
        gfig.figure_metadata(
            "Figure - TAC sensitivity analysis (GSE153515 donor-level macro-F1)"
        ),
    )


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def fmt_num(v: float) -> str:
    return f"{v:.4f}"


def print_report(results: dict[str, dict], sanity: list[dict]) -> None:
    line = "=" * 100
    print(line)
    print("TAC SENSITIVITY ANALYSIS — Paper1 Tier-B external test (GSE153515)")
    print("  cells = 14,775 | basal 6,698 | differentiated 4,483 | "
          "LSC 3,423 | TAC 171 (1.2%)")
    print(f"  endpoint = donor-level macro-F1, bootstrap 95% CI "
          f"(n_boot={N_BOOT}, seed={BOOT_SEED})")
    print(line)

    # --- sanity check table
    print("\n[1] SANITY CHECK — recomputed 4-class macro-F1 vs locked meta gate")
    print(f"    {'arm':<12s} {'meta macro-F1':>14s} {'recomputed':>12s} "
          f"{'abs diff':>10s} {'<=1e-3':>8s}")
    for s in sanity:
        print(f"    {s['arm']:<12s} {s['meta_macro_f1']:>14.6f} "
              f"{s['recomputed_macro_f1']:>12.6f} {s['abs_diff']:>10.2e} "
              f"{'OK' if s['within_1e-3'] else 'FAIL':>8s}")
    ok = all(s["within_1e-3"] for s in sanity)
    print(f"    -> all arms within 1e-3 tolerance: {'YES' if ok else 'NO'}")

    # --- macro-F1 endpoint table
    print("\n[2] DONOR-LEVEL MACRO-F1 (with vs without TAC)")
    print(f"    {'arm':<12s} {'with TAC':>10s} {'95% CI':>24s} "
          f"{'without TAC':>12s} {'95% CI':>24s} {'delta':>8s}")
    for arm in ARMS:
        a = arm[0]
        r = results[a]
        w, wo = r["with_tac"], r["without_tac"]
        print(f"    {r['display']:<12s} {w['macro_f1']:>10.4f} "
              f"[{w['ci_lower']:.4f}, {w['ci_upper']:.4f}] "
              f"{wo['macro_f1']:>12.4f} "
              f"[{wo['ci_lower']:.4f}, {wo['ci_upper']:.4f}] "
              f"{r['delta_macro_f1']:>+8.4f}")

    # --- per-class table, cell level
    print("\n[3] PER-CLASS P/R/F1 — CELL LEVEL (pooled across cells)")
    print(f"    {'arm':<12s} {'cls':<14s} {'P':>7s} {'R':>7s} {'F1':>7s} {'sup':>6s}")
    for arm in ARMS:
        a = arm[0]
        r = results[a]
        pc = r["with_tac"]["per_class_cell"]
        for cls in CLASSES_4:
            m = pc[cls]
            print(f"    {r['display']:<12s} {cls:<14s} {m['precision']:>7.3f} "
                  f"{m['recall']:>7.3f} {m['f1']:>7.3f} {m['support']:>6d}")

    # --- per-class table, donor level
    print("\n[4] PER-CLASS P/R/F1 — DONOR LEVEL (per-donor mean)")
    print(f"    {'arm':<12s} {'cls':<14s} {'P':>7s} {'R':>7s} {'F1':>7s} {'sup':>6s}")
    for arm in ARMS:
        a = arm[0]
        r = results[a]
        pc = r["with_tac"]["per_class_donor"]
        for cls in CLASSES_4:
            m = pc[cls]
            print(f"    {r['display']:<12s} {cls:<14s} {m['precision']:>7.3f} "
                  f"{m['recall']:>7.3f} {m['f1']:>7.3f} {m['support']:>6d}")

    # --- per-donor detail for the headline arms
    print("\n[5] PER-DONOR MACRO-F1 DETAIL (with TAC)")
    print(f"    {'arm':<12s} {'donor':<12s} {'accession':<12s} {'F1':>8s} "
          f"{'n cells':>8s}")
    for arm in ARMS:
        a = arm[0]
        r = results[a]
        data = _load_arm(a)
        for d in sorted(r["with_tac"]["per_donor"]):
            acc = DONOR_LABELS.get(d, d)
            n = int((data["donors"] == d).sum())
            print(f"    {r['display']:<12s} {d:<12s} {acc:<12s} "
                  f"{r['with_tac']['per_donor'][d]:>8.4f} {n:>8d}")

    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    sanity: list[dict] = []
    for arm, *_ in ARMS:
        res = analyze_arm(arm)
        results[arm] = res
        sanity.append(sanity_check(arm, res))

    print_report(results, sanity)

    # Table CSV
    df = build_table(results)
    csv_path = TABLE_OUT_DIR / "table_tac_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nTable written: {csv_path.relative_to(PROJECT_ROOT)} "
          f"({csv_path.stat().st_size / 1024:.1f} kB, {len(df)} rows)")

    # Figure
    generated = make_figure(results, FIG_OUT_DIR)
    for p in generated:
        print(f"Figure written: {p.relative_to(PROJECT_ROOT)} "
              f"({p.stat().st_size / 1024:.1f} kB)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
