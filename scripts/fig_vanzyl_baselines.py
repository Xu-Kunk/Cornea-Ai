#!/usr/bin/env python
"""E2 baseline-vs-scFM rho comparison figure (publication draft; des-1 polish later).

Reads artifacts/paper1_vanzyl/baselines/baselines_rho.json + the run6 eval
summary + convergence eval, plots rho_normed (L2-normed) with per-seed points
for the seed-spread arms, frozen dashed reference line at 0.7887.
Outputs fig_baselines_rho.{png,pdf} @ 300 dpi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ROOT / "artifacts" / "paper1_vanzyl" / "baselines" / "baselines_rho.json"
RUN6 = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval" / "summary_vanzyl_6run.json"
CONV = ROOT / "artifacts" / "paper1_vanzyl" / "convergence" / "final_test_eval.json"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "baselines"

COLORS = {
    "frozen": "#4C72B0",
    "naive": "#DD8452",
    "ordinal": "#55A868",
    "hvg_pca": "#C44E52",
    "harmony": "#8172B3",
    "scvi": "#937860",
}


def load() -> tuple[dict, dict, dict]:
    d = json.loads(BASELINES.read_text(encoding="utf-8"))
    run6 = json.loads(RUN6.read_text(encoding="utf-8"))
    conv = json.loads(CONV.read_text(encoding="utf-8"))
    return d, run6, conv


def main() -> int:
    d, run6, conv = load()
    frozen_rho = d["reference_arms"]["frozen"]["rho_normed"]

    rows: list[tuple[str, float, list[float], str, str]] = []
    # baselines (point estimate only, but keep per-seed empty)
    for name, b in d["baselines"].items():
        label = {
            "hvg_pca_nh2000_npc50": "HVG/PCA (2000, 50)",
            "hvg_pca_nh5000_npc50": "HVG/PCA (5000, 50)",
            "hvg_pca_nh2000_npc100": "HVG/PCA (2000, 100)",
            "hvg_pca_nh5000_npc100": "HVG/PCA (5000, 100)",
            "harmony_nh2000_npc50": "Harmony (2000, 50)",
            "scvi_nlatent128": "scVI (128-d)",
        }.get(name, name)
        ckey = "harmony" if name.startswith("harmony") else (
            "scvi" if name.startswith("scvi") else "hvg_pca")
        rows.append((label, b["rho"]["rho_normed"], [], ckey, name))
    # scFM arms
    rows.append(("frozen scFM (test donor)", frozen_rho, [],
                 "frozen", "frozen"))
    n_6 = run6["aggregate"]["naive"]["rho_normed"]
    n_per = list(n_6["per_seed"].values())
    rows.append(("naive ft 3-epoch (6-run)", n_6["mean"], n_per, "naive", "naive_6run"))
    o_6 = run6["aggregate"]["ordinal"]["rho_normed"]
    o_per = list(o_6["per_seed"].values())
    rows.append(("ordinal ft (CORN)", o_6["mean"], o_per, "ordinal", "ordinal_6run"))
    bv = conv["aggregate"]["best_val"]["rho_normed"]
    fe = conv["aggregate"]["final_epoch"]["rho_normed"]
    rows.append(("naive ft 15-epoch best-val", bv["mean"],
                 list(bv["per_seed"].values()), "naive", "naive_bv"))
    rows.append(("naive ft 15-epoch final", fe["mean"],
                 list(fe["per_seed"].values()), "naive", "naive_fe"))

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=300)
    y_pos = list(range(len(rows)))[::-1]
    for (label, rho, seeds, kind, _key), y in zip(rows, y_pos, strict=True):
        color = COLORS[kind]
        if seeds:
            ax.scatter(seeds, [y] * len(seeds), s=22, alpha=0.55,
                       color=color, zorder=3, edgecolor="none")
        marker = "D" if kind in ("hvg_pca", "harmony", "scvi") else (
            "o" if kind == "frozen" else "s")
        ms = 9 if kind in ("hvg_pca", "harmony", "scvi") else 10
        ax.scatter([rho], [y], s=ms ** 2 * 2.2, marker=marker, color=color,
                   zorder=4, edgecolor="white", linewidth=1.1)
        ax.text(0.003, y + 0.02, f"{rho:.3f}", fontsize=8.5, va="bottom",
                color="0.25")
    ax.axvline(frozen_rho, color=COLORS["frozen"], ls="--", lw=1.2, alpha=0.85)
    ax.axvline(0.5, color="0.6", ls=":", lw=1.0)
    ax.text(frozen_rho, len(rows) - 0.35, f" frozen scFM ρ = {frozen_rho:.3f}",
            fontsize=9, color=COLORS["frozen"], ha="left")
    ax.text(0.5, -0.6, "ρ = 0.5 (gradient-holding threshold)", fontsize=8,
            color="0.5", ha="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(-0.25, 1.05)
    ax.set_xlabel("ρ (marker-gradient Spearman, L2-normed embeddings, "
                  "test donor Hu0216)", fontsize=10)
    ax.set_title("E2: statistical baselines vs scFM arms — differentiation "
                 "gradient ρ", fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_baselines_rho.png", dpi=300)
    fig.savefig(OUT / "fig_baselines_rho.pdf")
    print(f"wrote {OUT / 'fig_baselines_rho.png'} / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
