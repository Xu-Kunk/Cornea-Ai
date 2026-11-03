#!/usr/bin/env python
"""UMAP collapse figure (Paper 1 van Zyl) + sensitivity summary figure.

Reads the UMAP coordinates produced by sens_umap_reduce.py
(umap_collapse/_coords/*.npz) and the sensitivity summary
(sensitivity/sensitivity_summary.json), and draws:

  fig_umap_collapse.{pdf,png} — 3x3 grid (rows: frozen / naive / CORN;
      cols: scGPT / GF / CellPLM), cells colored by 3-tier cell type
      (LSC / basal / differentiated), Okabe-Ito palette, 300 dpi.
  fig_umap_scgpt_bistable.{pdf,png} — 1x4 scGPT panels (frozen,
      naive s42 collapsed, naive s123 preserved, CORN s42 rescued).
  fig_sensitivity_summary.{pdf,png} — per-variant rho across arms.

Style: Arial (Liberation Sans fallback), 300-dpi PNG + vector PDF with
provenance metadata (des-1 / manuscript conventions).

Usage (p5070 or local):
    /home/summer/corneal_scgpt/venv/bin/python scripts/sens_umap_fig.py
        [--sensitivity-dir artifacts/paper1_vanzyl/sensitivity]
        [--umap-dir artifacts/paper1_vanzyl/umap_collapse]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paper1_results_figures import (  # noqa: E402
    OI_BLUE,
    OI_BLUEGREEN,
    OI_ORANGE,
    OI_GRAY,
    figure_metadata,
    save_figure,
    setup_style,
)

CELLTYPE_COLORS = {
    "LSC": OI_BLUE,
    "basal": OI_BLUEGREEN,
    "differentiated": OI_ORANGE,
}
ARCH_LABELS = {"scgpt": "scGPT", "gf": "Geneformer", "cellplm": "CellPLM"}
ARM_LABELS = {"frozen": "frozen", "naive": "naive", "corn": "CORN"}
PANEL_POINTS = 20158


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_coords(umap_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    coord_dir = umap_dir / "_coords"
    if not coord_dir.exists():
        return out
    for p in sorted(coord_dir.glob("*.npz")):
        out[p.stem] = np.load(p)["coord"]
    return out


def _tag_to_meta(tag: str) -> tuple[str, str, int | None]:
    parts = tag.split("_")
    arch = parts[0]
    if "frozen" in parts[1:]:
        return arch, "frozen", None
    arm = parts[1]
    seed_s = parts[-1]
    if seed_s.startswith("seed"):
        seed_s = seed_s[4:]
    seed = int(seed_s) if seed_s.isdigit() else None
    return arch, arm, seed


def _umap_arrow_axes(ax: plt.Axes) -> None:
    """Draw small UMAP1/UMAP2 arrow axes in the bottom-left of a UMAP panel."""
    kwargs = dict(xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="->", color="#555555", lw=0.6),
                  fontsize=5.5, color="#555555")
    ax.annotate("", xy=(0.25, 0.08), xytext=(0.08, 0.08), **kwargs)
    ax.annotate("", xy=(0.08, 0.25), xytext=(0.08, 0.08), **kwargs)
    ax.text(0.27, 0.06, "UMAP1", transform=ax.transAxes, fontsize=5,
            color="#555555", ha="left", va="center")
    ax.text(0.06, 0.27, "UMAP2", transform=ax.transAxes, fontsize=5,
            color="#555555", ha="center", va="bottom", rotation=90)


def draw_umap_grid(coords: dict[str, np.ndarray], y_by_tag: dict[str, np.ndarray],
                   rho_by_tag: dict[str, float], out_dir: Path) -> list[Path]:
    setup_style()
    arches = ["scgpt", "gf", "cellplm"]
    arms = ["frozen", "naive", "corn"]
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.2))
    # Legend markers sized to match the in-figure scatter points
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=3, color=c,
                          label=k) for k, c in CELLTYPE_COLORS.items()]

    for i, arm in enumerate(arms):
        for j, arch in enumerate(arches):
            ax = axes[i, j]
            # prefer seed42; fall back to any seed of this (arch, arm)
            tag = next((t for t in coords
                        if _tag_to_meta(t) == (arch, arm, 42)), None)
            if tag is None:
                tag = next((t for t in coords
                            if _tag_to_meta(t)[:2] == (arch, arm)), None)
            if tag is None:
                ax.set_visible(False)
                continue
            coord = coords[tag]
            y = y_by_tag.get(tag)
            if y is None:
                ax.set_visible(False)
                continue
            for k, c in CELLTYPE_COLORS.items():
                m = y == k
                ax.scatter(coord[m, 0], coord[m, 1], s=1.8, c=c, alpha=0.8,
                           rasterized=True, linewidths=0)
            rho = rho_by_tag.get(tag)
            _, _, seed = _tag_to_meta(tag)
            title = f"{ARCH_LABELS[arch]} · {ARM_LABELS[arm]}"
            if seed is not None:
                title += f" s{seed}"
            if rho is not None:
                title += f"\nρ = {rho:.3f}"
            ax.set_title(title, fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    # Single shared UMAP1/UMAP2 arrow annotation at the bottom-left of the grid
    _umap_arrow_axes(axes[2, 0])
    # Short main title (<=60 chars); donor/cell-count detail moved to caption
    fig.suptitle("Differentiation-gradient collapse and CORN rescue",
                 fontsize=10, fontweight="bold")
    fig.text(0.5, 0.005, "van Zyl test donor, 20,158 cells, colored by cell type.",
             ha="center", va="bottom", fontsize=6.5, color="#555555")
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    meta = figure_metadata("UMAP collapse grid — frozen / naive / CORN x "
                         "scGPT / GF / CellPLM (van Zyl test donor)")
    out_pdf = out_dir / "fig_umap_collapse.pdf"
    out_png = out_dir / "fig_umap_collapse.png"
    fig.savefig(out_pdf, dpi=300, metadata=meta, facecolor="white",
                bbox_inches="tight")
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [out_pdf, out_png]


def draw_scgpt_bistable(coords: dict[str, np.ndarray],
                        y_by_tag: dict[str, np.ndarray],
                        rho_by_tag: dict[str, float],
                        out_dir: Path) -> list[Path]:
    setup_style()
    panels = [("scgpt_frozen", "frozen"), ("scgpt_naive_seed42", "naive s42"),
              ("scgpt_naive_seed123", "naive s123"), ("scgpt_corn_seed42", "CORN s42")]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    coord_extents: list[tuple[float, float, float, float]] = []
    for ax, (tag, label) in zip(axes, panels):
        if tag not in coords:
            ax.set_visible(False)
            continue
        coord = coords[tag]
        y = y_by_tag.get(tag)
        for k, c in CELLTYPE_COLORS.items():
            m = y == k
            ax.scatter(coord[m, 0], coord[m, 1], s=1.8, c=c, alpha=0.8,
                       rasterized=True, linewidths=0)
        coord_extents.append((coord[:, 0].min(), coord[:, 0].max(),
                              coord[:, 1].min(), coord[:, 1].max()))
        rho = rho_by_tag.get(tag)
        t = label if rho is None else f"{label}\nρ = {rho:.3f}"
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    # Equalize all panels to a shared data range so the four boxes are identical
    if coord_extents:
        x_min = min(e[0] for e in coord_extents)
        x_max = max(e[1] for e in coord_extents)
        y_min = min(e[2] for e in coord_extents)
        y_max = max(e[3] for e in coord_extents)
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        for ax in axes:
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("scGPT bistability: seed-dependent collapse (s42) vs "
                 "preservation (s123) and CORN rescue", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    meta = figure_metadata("scGPT bistable UMAP — frozen / naive s42 / naive s123 / "
                         "CORN s42 (van Zyl test donor)")
    out_pdf = out_dir / "fig_umap_scgpt_bistable.pdf"
    out_png = out_dir / "fig_umap_scgpt_bistable.png"
    fig.savefig(out_pdf, dpi=300, metadata=meta, facecolor="white",
                bbox_inches="tight")
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [out_pdf, out_png]


def draw_sensitivity_summary(summary: dict, out_dir: Path) -> list[Path]:
    """Per-variant rho: x = variant, lines = arms (frozen/naive/CORN)."""
    setup_style()
    verdicts = summary.get("verdicts", {})
    variant_names = sorted(v for v in verdicts if not v.startswith("_"))
    if not variant_names:
        print("  [WARN] no verdicts in summary; sensitivity figure skipped")
        return []
    # group variants by dimension
    dims = {"A (SVD axis)": [v for v in variant_names if v.startswith("A_")],
            "M (marker panel)": [v for v in variant_names if v.startswith("M_")],
            "R (rank)": [v for v in variant_names if v.startswith("R_")]}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    arm_colors = {"frozen": OI_GRAY, "naive": OI_BLUE, "corn": OI_BLUEGREEN}
    for ax, (dname, vs) in zip(axes, dims.items()):
        if not vs:
            ax.set_visible(False)
            continue
        xs = np.arange(len(vs))
        for arm, color in arm_colors.items():
            vals = []
            for v in vs:
                rows = list(verdicts[v].values())
                vv = [d[f"rho_{arm}"] for d in rows]
                vals.append(float(np.mean(vv)) if vv else np.nan)
            ax.plot(xs, vals, marker="o", ms=3.5, lw=1.4, c=color,
                    label=arm)
        ax.axhline(0, color="#bbbbbb", lw=0.6, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([v.split("_", 1)[1] for v in vs], rotation=45,
                           ha="right", fontsize=6.5)
        ax.set_title(dname, fontsize=8.5)
        ax.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Spearman ρ (mean over arms/seeds)", fontsize=8)
    fig.legend(handles=[plt.Line2D([0], [0], c=c, lw=1.8, label=a)
                        for a, c in arm_colors.items()],
               loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sensitivity: differentiation-gradient ρ across SVD-axis, "
                 "marker-panel and rank-assignment variants", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return save_figure(
        fig, "fig_sensitivity_summary", out_dir,
        figure_metadata("Sensitivity summary — rho across variant dimensions "
                        "(van Zyl test donor)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensitivity-dir", type=Path,
                    default=ROOT / "artifacts" / "paper1_vanzyl" / "sensitivity")
    ap.add_argument("--umap-dir", type=Path,
                    default=ROOT / "artifacts" / "paper1_vanzyl" / "umap_collapse")
    args = ap.parse_args()

    summary_path = args.sensitivity_dir / "sensitivity_summary.json"
    if not summary_path.is_file():
        print(f"ERROR: {summary_path} not found (run sens_umap_reduce first)")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coords = load_coords(args.umap_dir)
    if not coords:
        print(f"ERROR: no UMAP coords under {args.umap_dir / '_coords'}")
        return 1
    print(f"coords: {sorted(coords)}")

    # y labels: read from per-arm variant ordinal (same order as coords)
    per_arm = summary.get("per_arm_variants", {})
    y_by_tag: dict[str, np.ndarray] = {}
    for tag, rec in per_arm.items():
        if tag.startswith("_"):
            continue
        ords = rec.get("ordinal", [])
        if ords:
            y_by_tag[tag] = np.array(
                [["LSC", "basal", "differentiated"][int(o)] for o in ords])
    rho_by_tag = {tag: float(v) for tag, v in
                  summary.get("locked_metric_rho", {}).items()}

    out_dir = args.umap_dir
    paths = draw_umap_grid(coords, y_by_tag, rho_by_tag, out_dir)
    print(f"umap grid -> {[str(p) for p in paths]}")
    paths = draw_scgpt_bistable(coords, y_by_tag, rho_by_tag, out_dir)
    print(f"bistable -> {[str(p) for p in paths]}")
    paths = draw_sensitivity_summary(summary, args.sensitivity_dir)
    print(f"sensitivity -> {[str(p) for p in paths]}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
