#!/usr/bin/env python
"""Figure: three-domain boundary map of differentiation-gradient collapse (Paper 1, v6).

Panel A — Spearman rho (differentiation-gradient) by evaluation domain and arm,
with 3-seed scatter where available and the ordinal range for the out-of-corpus arm.
Panel B — corpus-position schematic: fine-tuning effect (frozen -> CE-naive) across
the three domains, annotated with the collapse/mild/improvement pattern.

Data sources (verified):
  GSE153515 same-corpus: frozen 0.8064; CE-naive mean -0.0082 (seeds -0.0424/-0.0360/+0.0539);
    ordinal 0.8330 (seeds 0.8341/0.8351/0.8299)
  GSE299152 out-of-corpus: frozen 0.7066; CE-naive 0.6407 (0.6167/0.6307/0.6746);
    ordinal-naive 0.3501 (0.2610/0.4566/0.3325); ordinal-CAFT 0.5394 (0.5310/0.5449/0.5421)
  Epidermis cross-tissue: frozen 0.2395; CE-naive 0.4551; ordinal 0.4997
  (artifacts/paper1_tier_b_eval*, paper1_ordinal_eval, paper1_epidermis_eval)
Output: figures/paper1/figure_domain_boundary.pdf/.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# sys.path.insert(0, "<abs path to bio-viz-config/scripts>")  # dev-only styling helpers; not required by requirements-core.txt
from bio_viz_config import bio_figure, bio_save, figure_qa, set_journal_style

# Okabe-Ito (colorblind-safe)
BLUE = "#0072B2"       # frozen
VERM = "#D55E00"       # CE-naive
GREEN = "#009E73"      # ordinal
GREY = "#8C8C8C"

DOMAINS = ["Same-corpus\n(GSE153515)", "Out-of-corpus\n(GSE299152)", "Cross-tissue\n(epidermis)"]

# arm -> per-domain rho (mean), with seed lists / ranges for scatter
FROZEN = [0.8064, 0.7066, 0.2395]
CE = [(-0.0082, [-0.0424, -0.0360, 0.0539]),
      (0.6407, [0.6167, 0.6307, 0.6746]),
      (0.4551, None)]
ORD = [(0.8330, [0.8341, 0.8351, 0.8299]),
       (None, None, (0.3501, 0.5394)),  # range for out-of-corpus (two sub-arms)
       (0.4997, None)]
ORD_SEEDS_OUT = {"ord_naive": [0.2610, 0.4566, 0.3325], "ord_caft": [0.5310, 0.5449, 0.5421]}


def panel_a(ax):
    x = np.arange(3)
    w = 0.24
    ax.axhline(0.0, color="k", lw=0.8, ls=":", zorder=1)
    # bars: frozen
    b1 = ax.bar(x - w, FROZEN, w, color=BLUE, label="Frozen", zorder=3)
    # CE-naive (mean bars + seed dots)
    ce_means = [c[0] for c in CE]
    b2 = ax.bar(x, ce_means, w, color=VERM, label="CE-naive fine-tuning", zorder=3)
    for i, (m, seeds) in enumerate(CE):
        if seeds:
            ax.scatter([i] * 3, seeds, s=12, color=VERM, alpha=0.6, zorder=4,
                       edgecolor="white", linewidth=0.4)
    # ordinal (bar or range)
    ord_mid = [ORD[0][0], 0.5 * (0.3501 + 0.5394), ORD[2][0]]
    b3 = ax.bar(x + w, ord_mid, w, color=GREEN, label="Ordinal head (CORN)", zorder=3)
    ax.errorbar(1, ord_mid[1], yerr=[[0.5394 - ord_mid[1]], [ord_mid[1] - 0.3501]],
                fmt="none", ecolor=GREEN, elinewidth=1.4, capsize=3, zorder=4)
    for i, entry in enumerate(ORD):
        seeds = entry[1]
        if i == 1:
            ax.scatter([1 + w] * 3, ORD_SEEDS_OUT["ord_naive"], s=12, color=GREEN, alpha=0.5,
                       edgecolor="white", linewidth=0.4, zorder=4)
            ax.scatter([1 + w] * 3, ORD_SEEDS_OUT["ord_caft"], s=12, color=GREEN, alpha=0.9,
                       edgecolor="white", linewidth=0.4, zorder=4)
        elif seeds:
            ax.scatter([i + w] * 3, seeds, s=12, color=GREEN, alpha=0.6,
                       edgecolor="white", linewidth=0.4, zorder=4)
    # collapse annotations
    ax.annotate("", xy=(0, CE[0][0]), xytext=(0, FROZEN[0]),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.1))
    ax.text(0 + 0.13, 0.38, r"$\Delta\rho$ = $-0.81$", fontsize=7, color="0.25")
    ax.text(0, -0.16, "catastrophic\ncollapse", fontsize=7, ha="center", color=VERM)
    ax.text(1, 0.80, "mild", fontsize=7, ha="center", color=VERM)
    ax.text(2, 0.33, "improve", fontsize=7, ha="center", color=VERM)
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, fontsize=8)
    ax.set_ylabel(r"Differentiation-gradient Spearman $\rho$", fontsize=9)
    ax.set_ylim(-0.30, 1.02)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
    ax.set_title("A", loc="left", fontsize=11, fontweight="bold")


def panel_b(ax):
    xs = [0, 1, 2]
    # corpus-position map: frozen rho trajectory
    ax.plot(xs, FROZEN, "-o", color=BLUE, lw=1.6, ms=4, label="Frozen")
    ax.plot(xs, [CE[i][0] for i in range(3)], "-s", color=VERM, lw=1.6, ms=4,
            label="CE-naive")
    ax.plot([0, 1, 2], [ORD[0][0], ord_mid := 0.5 * (0.3501 + 0.5394), ORD[2][0]],
            "-^", color=GREEN, lw=1.6, ms=4, label="Ordinal (CORN)")
    # shade the "catastrophic" zone on the left
    ax.axvspan(-0.4, 0.4, color=VERM, alpha=0.06)
    ax.text(0, 1.06, "overwrite of an\nencoded axis", fontsize=7, ha="center", color=VERM)
    ax.text(1, 1.06, "partial loss", fontsize=7, ha="center", color="0.3")
    ax.text(2, 1.06, "axis learned\nde novo", fontsize=7, ha="center", color=GREEN)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Same-corpus", "Out-of-corpus", "Cross-tissue"], fontsize=8)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=9)
    ax.set_ylim(-0.15, 1.16)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    ax.set_title("B", loc="left", fontsize=11, fontweight="bold")


def main() -> None:
    set_journal_style("cell")
    fig, (axa, axb) = bio_figure(nrows=1, ncols=2, figsize=(7.2, 3.0), wspace=0.35)
    panel_a(axa)
    panel_b(axb)
    outdir = Path(__file__).resolve().parents[1] / "figures" / "paper1"
    bio_save(fig, "figure_domain_boundary", outdir=outdir)
    fig_qa = figure_qa(fig, journal="cell")
    print("QA:", fig_qa if isinstance(fig_qa, str) else "passed")
    print("saved figure_domain_boundary.pdf/.png")


if __name__ == "__main__":
    main()
