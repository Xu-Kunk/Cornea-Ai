#!/usr/bin/env python
"""Cluster 0 forensics + TAC absence investigation (READ-ONLY).

Evidence collection for the pre-G1 decision on cluster 0 (13,584 cells).
Statistics are restricted to annotation counts (obs) and expression evidence
(counts layer: %exp = fraction of cells with count > 0; mean expr = mean of
log1p(counts)). No rho / ordinal-axis / model-embedding metrics. No label or
data modification, no re-clustering.

TAC proliferation genes (MKI67/BIRC5/RRM2/TOP2A/PLK1/TK1) were removed from
the QC gene set by the locked min_cells=3% filter, so their %exp is measured
from the raw pre-QC partition matrices and aligned to the 80,817 QC cells via
unique `observation_joinid`.

Outputs (printed tables + json + 2 figures + report skeleton):
    cluster0_forensics.json
    umap_cluster0.png/pdf        (300 dpi)
    marker_violin.png/pdf        (300 dpi)
    report.md                    (A/B/C numbers; evidence chain & recommendation
                                  appended by the orchestrator on the local side)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H5AD = Path("/home/summer/corneal_scgpt/data/vanzyl_annotated.h5ad")
OUT = Path("/home/summer/corneal_scgpt/artifacts/paper1_vanzyl/cluster0_forensics")

EPI_MARKERS = ["KRT14", "KRT15", "KRT19", "KRT5", "KRT12", "KRT3",
               "COL17A1", "ITGB1"]
C0_DEG = ["CHRM3", "DGKI", "CARMN", "MYH11"]
STROMAL_MARKERS = ["DCN", "COL1A1", "COL3A1", "MYOC", "FN1", "VIM", "ACTA2"]
LSC_PANEL = ["ABCG2", "ITGA9", "TSPAN7", "SOX17"]
TAC_GENES = ["MKI67", "BIRC5", "RRM2", "TOP2A", "PLK1", "TK1"]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def resolve_symbols(adata: ad.AnnData, symbols: list[str]) -> dict[str, str]:
    """Map gene symbols (var['feature_name']) to var_names (Ensembl IDs)."""
    sym2vn: dict[str, str] = {}
    for vn, fn in zip(adata.var_names.astype(str),
                      adata.var["feature_name"].astype(str)):
        sym2vn.setdefault(fn, vn)
    out: dict[str, str] = {}
    for s in symbols:
        if s in sym2vn:
            out[s] = sym2vn[s]
    return out


def pct_mean_for_set(csr, rows: np.ndarray, pos_list: list[int],
                     sym: str) -> tuple[float, float]:
    """%exp + mean log1p expr for a symbol given its column position(s)."""
    vals_pct: list[float] = []
    vals_mean: list[float] = []
    for pos in pos_list:
        idx = np.flatnonzero(rows)
        col = csr[idx, :][:, pos]  # sparse (n_sel, 1)
        n = idx.size
        pct = 100.0 * float(col.nnz) / n
        mean = float(np.log1p(col.data).mean()) if col.nnz else 0.0
        vals_pct.append(pct)
        vals_mean.append(mean)
    if not vals_pct:
        return np.nan, np.nan
    return max(vals_pct), max(vals_mean)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"Reading {H5AD} ...")
    adata = ad.read_h5ad(H5AD)
    log(f"Loaded: {adata.shape[0]} x {adata.shape[1]}, "
        f"X dtype={getattr(adata.X, 'dtype', type(adata.X))}")

    obs = adata.obs
    counts = adata.layers["counts"].tocsr()
    vn2pos = {str(vn): int(i) for i, vn in enumerate(adata.var_names)}
    adata.X = None  # release scaled X; only counts layer needed
    umap = adata.obsm["X_umap"]
    n_all = adata.n_obs
    log(f"counts layer: {counts.shape}, nnz={counts.nnz}")

    clusters = obs["leiden"].astype(str)
    c0 = (clusters == "0").to_numpy()
    n_c0 = int(c0.sum())
    log(f"cluster 0: {n_c0} cells")

    report: list[str] = []
    def add(s: str = "") -> None:
        report.append(s)

    # ------------------------------------------------------------------
    # A. Cluster 0 source profile
    # ------------------------------------------------------------------
    add("# Cluster 0 Forensics + TAC Absence — Evidence Report")
    add("")
    add(f"- Data: `vanzyl_annotated.h5ad` (Stage 1, p5070), "
        f"{n_all} cells x {counts.shape[1]} genes after QC.")
    add(f"- Cluster 0: {n_c0} cells ({100 * n_c0 / n_all:.2f}% of all cells).")
    add("- Statistics: annotation counts + counts-layer expression only "
        "(%exp, mean log1p). Read-only.")
    add("")

    json_out: dict = {"n_all_cells": int(n_all),
                      "cluster0": {"n_cells": int(n_c0)}}

    add("## A. Cluster 0 source profile")
    add("")
    for col in ["partition", "donor"]:
        c0_counts = obs.loc[c0, col].astype(str).value_counts()
        all_counts = obs[col].astype(str).value_counts()
        add(f"### A.{'1' if col == 'partition' else '2'}. {col} composition "
            f"(cluster 0 vs all cells)")
        add("")
        add("| value | c0 n | c0 % | all n | all % | fold vs all |")
        add("|---|---|---|---|---|---|")
        for v in all_counts.index:
            c0_n = int(c0_counts.get(v, 0))
            c0_p = 100.0 * c0_n / n_c0
            a_p = 100.0 * int(all_counts[v]) / n_all
            fold = (c0_p / a_p) if a_p > 0 else np.nan
            add(f"| {v} | {c0_n} | {c0_p:.1f} | {int(all_counts[v])} | "
                f"{a_p:.1f} | {fold:.2f}x |")
        add("")
        json_out["cluster0"][col] = {
            str(k): int(v) for k, v in c0_counts.items()
        }
        json_out["cluster0"][f"{col}_pct"] = {
            str(k): round(100.0 * v / n_c0, 2) for k, v in c0_counts.items()
        }

    # sample composition (top 10)
    c0_samp = obs.loc[c0, "sample"].astype(str).value_counts()
    all_samp = obs["sample"].astype(str).value_counts()
    add("### A.3. sample composition (cluster 0, top 10)")
    add("")
    add("| sample | c0 n | c0 % | all n | all % | fold vs all |")
    add("|---|---|---|---|---|---|")
    for v in c0_samp.head(10).index:
        a_n = int(all_samp.get(v, 0))
        c0_n = int(c0_samp[v])
        a_p = 100.0 * a_n / n_all
        c0_p = 100.0 * c0_n / n_c0
        fold = (c0_p / a_p) if a_p > 0 else np.nan
        add(f"| {v} | {c0_n} | {c0_p:.1f} | {a_n} | {a_p:.1f} | "
            f"{fold:.2f}x |")
    add(f"\ncluster 0 spans {c0_samp.size} samples "
        f"(all data spans {all_samp.size} samples).")
    add("")
    json_out["cluster0"]["sample_top10"] = {
        str(k): int(v) for k, v in c0_samp.head(10).items()
    }

    # author_cell_type cross-tab
    for col in ["author_cell_type", "cell_type"]:
        c0_ct = obs.loc[c0, col].astype(str).value_counts()
        all_ct = obs[col].astype(str).value_counts()
        add(f"### A.4. {col} composition (cluster 0 vs all cells)")
        add("")
        add("| value | c0 n | c0 % | all n | all % | fold vs all |")
        add("|---|---|---|---|---|---|")
        for v in all_ct.index:
            c0_n = int(c0_ct.get(v, 0))
            c0_p = 100.0 * c0_n / n_c0
            a_p = 100.0 * int(all_ct[v]) / n_all
            fold = (c0_p / a_p) if a_p > 0 else np.nan
            add(f"| {v} | {c0_n} | {c0_p:.1f} | {int(all_ct[v])} | "
                f"{a_p:.1f} | {fold:.2f}x |")
        add("")
        json_out["cluster0"][col] = {
            str(k): int(v) for k, v in c0_ct.items()
        }

    # cell_type_raw x leiden cross-tab (all clusters)
    add("### A.5. cell_type_raw (DEG enrichment) x leiden cluster — full table")
    add("")
    ct_cl = pd.crosstab(obs["cell_type_raw"].astype(str),
                        clusters.astype(str))
    add("```")
    add(ct_cl.to_string())
    add("```")
    add("")
    json_out["cell_type_raw_x_leiden"] = {
        str(k): {str(c): int(v) for c, v in row.items()}
        for k, row in ct_cl.iterrows()
    }

    # ------------------------------------------------------------------
    # B. Marker expression: cluster 0 vs established basal/LSC/diff
    # ------------------------------------------------------------------
    add("## B. Marker expression (counts layer; %exp = count>0, "
        "mean = mean log1p(counts))")
    add("")
    add("Groups: **cluster0** (13,584) | **basal** = cell_type_raw "
        "PMC_basal (25,332) | **LSC_other** = cell_type_raw LSC excl. "
        "cluster 0 (clusters 17+20, 1,312) | **diff** = cell_type_raw TDC "
        "(14,472) | **NON_LAYER** (4,593).")
    add("")

    raw = obs["cell_type_raw"].astype(str)
    basal = (raw == "PMC_basal").to_numpy()
    lsc_other = ((raw == "LSC") & ~c0).to_numpy()
    diff = (raw == "TDC").to_numpy()
    nonlayer = (obs["cell_class_4"].astype(str) == "NON_LAYER").to_numpy()

    group_sets = {
        "cluster0": c0,
        "basal": basal,
        "LSC_other": lsc_other,
        "diff": diff,
        "NON_LAYER": nonlayer,
    }
    group_n = {k: int(v.sum()) for k, v in group_sets.items()}
    json_out["groups_n"] = group_n

    all_markers = (EPI_MARKERS + C0_DEG + STROMAL_MARKERS + LSC_PANEL
                   + TAC_GENES)
    resolved = resolve_symbols(adata, all_markers)
    missing = sorted(set(all_markers) - set(resolved))
    if missing:
        add(f"NOTE: markers not found in 13,250 retained genes: {missing}")
        add("")

    def group_stats(sym: str) -> dict:
        pos_list = [vn2pos[resolved[sym]]] if sym in resolved else []
        out = {}
        for gname, gmask in group_sets.items():
            p, m = pct_mean_for_set(counts, gmask, pos_list, sym)
            out[gname] = {"pct": round(p, 2), "mean": round(m, 4)}
        return out

    add("### B.1. Epithelial markers")
    add("")
    add("| gene | group | %exp | mean log1p |")
    add("|---|---|---|---|")
    for sym in EPI_MARKERS:
        st = group_stats(sym)
        json_out.setdefault("markers", {})[sym] = st
        for g in ["cluster0", "basal", "LSC_other", "diff"]:
            add(f"| {sym} | {g} | {st[g]['pct']} | {st[g]['mean']} |")
    add("")

    add("### B.2. Cluster 0 top DEGs (CHRM3, DGKI, CARMN, MYH11)")
    add("")
    add("| gene | group | %exp | mean log1p |")
    add("|---|---|---|---|")
    for sym in C0_DEG:
        st = group_stats(sym)
        json_out["markers"][sym] = st
        for g in ["cluster0", "basal", "LSC_other", "diff", "NON_LAYER"]:
            add(f"| {sym} | {g} | {st[g]['pct']} | {st[g]['mean']} |")
    add("")

    add("### B.3. Stromal / TM markers")
    add("")
    add("| gene | group | %exp | mean log1p |")
    add("|---|---|---|---|")
    for sym in STROMAL_MARKERS:
        st = group_stats(sym)
        json_out["markers"][sym] = st
        for g in ["cluster0", "basal", "LSC_other", "diff"]:
            add(f"| {sym} | {g} | {st[g]['pct']} | {st[g]['mean']} |")
    add("")

    add("### B.4. LSC panel (score_LSC genes: ABCG2, ITGA9, TSPAN7, SOX17)")
    add("")
    add("| gene | group | %exp | mean log1p |")
    add("|---|---|---|---|")
    for sym in LSC_PANEL:
        st = group_stats(sym)
        json_out["markers"][sym] = st
        for g in ["cluster0", "basal", "LSC_other", "diff"]:
            add(f"| {sym} | {g} | {st[g]['pct']} | {st[g]['mean']} |")
    add("")

    # ------------------------------------------------------------------
    # C. TAC absence investigation
    # ------------------------------------------------------------------
    add("## C. TAC / cycling-program investigation")
    add("")
    add("NOTE: all 6 TAC proliferation genes (MKI67, BIRC5, RRM2, TOP2A, "
        "PLK1, TK1) were REMOVED from the QC gene set (13,250 genes) by the "
        "locked min_cells=3% gene filter (rare-expression markers; the "
        "Stage 1 `scoring_note` pre-announced exactly this). They therefore "
        "do not exist in `layers['counts']`. %exp below is computed from the "
        "raw pre-QC partition matrices (59,413 genes) and aligned to the "
        "80,817 QC-passing cells via unique `observation_joinid`.")
    add("")
    log("Loading raw partition files for TAC gene detection (pre-QC set)...")
    raw_files = [
        Path("/home/summer/corneal_scgpt/data/vanzyl_central_cornea.h5ad"),
        Path("/home/summer/corneal_scgpt/data/vanzyl_tm_wedge.h5ad"),
    ]
    jid_lists: list[np.ndarray] = []
    det_lists: list[np.ndarray] = []
    for rp in raw_files:
        ra = ad.read_h5ad(rp, backed="r")
        fn = ra.var["feature_name"].astype(str)
        raw_pos = {g: int(np.flatnonzero(fn == g)[0]) for g in TAC_GENES}
        cols = [raw_pos[g] for g in TAC_GENES]
        xcol = ra[:, cols].X  # sparse (n, 6); toarray() for dense bool
        det = np.asarray((xcol > 0).toarray())
        det_lists.append(det)
        jid_lists.append(ra.obs["observation_joinid"].astype(str).to_numpy())
        ra.file.close()
    jids_all = np.concatenate(jid_lists)
    det_all = np.vstack(det_lists)  # 89,794 x 6 bool (merge order)
    del jid_lists, det_lists
    jpos = {j: i for i, j in enumerate(jids_all)}
    out_jids = obs["observation_joinid"].astype(str).to_numpy()
    missing_jid = sum(1 for j in out_jids if j not in jpos)
    if missing_jid:
        log(f"FATAL: {missing_jid} output cells unmatched by "
            "observation_joinid")
        sys.exit(1)
    det_out = det_all[[jpos[j] for j in out_jids]]  # aligned to obs order
    n_tac_any_out = det_out.sum(axis=1)
    log(f"TAC detection aligned: {det_out.shape[0]} QC cells, "
        f"{int((n_tac_any_out >= 1).sum())} express >=1 TAC gene")

    add("### C.1. TAC 6-gene %exp per Leiden cluster "
        "(raw pre-QC counts>0, aligned)")
    add("")
    add("| cluster | n | " + " | ".join(TAC_GENES) + " | any_of_6 |")
    add("|---|---|" + "---|" * (len(TAC_GENES) + 1))
    per_cluster_pct: dict[str, dict] = {}
    cl_sorted = sorted(set(clusters), key=int)
    for cl in cl_sorted:
        m = (clusters == cl).to_numpy()
        n = int(m.sum())
        sub = det_out[m, :]
        pct = 100.0 * sub.mean(axis=0)
        any6 = 100.0 * float((sub.sum(axis=1) >= 1).sum()) / n
        per_cluster_pct[cl] = {
            g: round(float(p), 2)
            for g, p in zip(TAC_GENES, np.asarray(pct).ravel())
        }
        per_cluster_pct[cl]["n"] = n
        per_cluster_pct[cl]["any_of_6_pct"] = round(any6, 2)
        add(f"| {cl} | {n} | "
            + " | ".join(f"{per_cluster_pct[cl][g]}" for g in TAC_GENES)
            + f" | {any6:.2f} |")
    add("")
    json_out["tac_per_cluster_pct"] = per_cluster_pct

    co_n = int((n_tac_any_out >= 2).sum())
    co_any = int((n_tac_any_out >= 1).sum())
    add("### C.2. Co-expression of >=2 TAC genes (any pair, QC cells)")
    add(f"- QC cells (80,817): expressing >=2 of 6 TAC genes: **{co_n}** "
        f"({100 * co_n / n_all:.3f}%); expressing >=1: {co_any} "
        f"({100 * co_any / n_all:.3f}%).")
    co_n_pre = int((det_all.sum(axis=1) >= 2).sum())
    add(f"- Context, all 89,794 pre-QC cells: >=2 TAC genes: {co_n_pre} "
        f"({100 * co_n_pre / det_all.shape[0]:.3f}%).")
    add("")
    json_out["tac_coexp_ge2"] = {
        "n_qc": int(co_n),
        "frac_qc": round(co_n / n_all, 5),
        "n_pre_qc": int(co_n_pre),
        "frac_pre_qc": round(co_n_pre / det_all.shape[0], 5),
    }

    st = obs["score_TAC"]
    quants = st.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,
                          1.0]).to_dict()
    add("### C.3. score_TAC distribution (obs; scored pre-QC on full gene set)")
    add("")
    add("| quantile | value |")
    add("|---|---|")
    for q, v in quants.items():
        add(f"| {100 * q:.0f}% | {v:.4f} |")
    add("")
    json_out["score_tac_quantiles"] = {
        str(100 * q): round(float(v), 4) for q, v in quants.items()
    }

    for thr_name, thr in [("gt95pct", quants[0.95]),
                          ("gt99pct", quants[0.99])]:
        high = (st > thr).to_numpy()
        hn = int(high.sum())
        high_by_cl = clusters[high].value_counts()
        add(f"### C.4. score_TAC > {thr_name} ({thr:.3f}): {hn} cells "
            f"({100 * hn / n_all:.3f}%) — cluster composition")
        add("")
        add("| cluster | n | % of high-score cells | cell_type_raw |")
        add("|---|---|---|---|")
        for cl in cl_sorted:
            n = int(high_by_cl.get(cl, 0))
            if n == 0:
                continue
            ct = obs.loc[high, "cell_type_raw"].astype(str)
            ct_of_cl = ct[clusters[high] == cl].value_counts()
            add(f"| {cl} | {n} | {100 * n / hn:.1f} | "
                f"{'; '.join(f'{k}({v})' for k, v in ct_of_cl.items())} |")
        add("")
        json_out[f"score_tac_{thr_name}"] = {
            "threshold": round(float(thr), 4),
            "n_cells": int(hn),
            "frac": round(hn / n_all, 5),
            "by_cluster": {str(k): int(v) for k, v in high_by_cl.items()},
        }

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Liberation Sans", "Arial", "DejaVu Sans"):
        if fam in available:
            plt.rcParams["font.family"] = fam
            log(f"font: {fam}")
            break

    # Fig 1: UMAP cluster 0 vs rest + partition inset pie
    fig, ax = plt.subplots(figsize=(8.5, 7))
    not_c0 = ~c0
    ax.scatter(umap[not_c0, 0], umap[not_c0, 1], s=1.2, c="#B0B0B0",
               alpha=0.55, label=f"other clusters (n={int(not_c0.sum())})",
               rasterized=True)
    ax.scatter(umap[c0, 0], umap[c0, 1], s=2.2, c="#D62728", alpha=0.8,
               label=f"cluster 0 (n={n_c0})", rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Cluster 0 (13,584 cells) — Stage 1 van Zyl annotation",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    axins = ax.inset_axes([0.70, 0.02, 0.28, 0.28])
    part_counts = obs.loc[c0, "partition"].astype(str).value_counts()
    labels = [f"{k}\n{100 * v / n_c0:.1f}%" for k, v in part_counts.items()]
    axins.pie(part_counts.values, labels=labels, startangle=90,
              colors=["#1F77B4", "#FF7F0E"], textprops={"fontsize": 8},
              wedgeprops={"edgecolor": "white"})
    axins.set_title("cluster 0 partition", fontsize=8)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"umap_cluster0.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    log("saved umap_cluster0")

    # Fig 2: split violins — cluster0 vs basal/LSC_other/diff, 8 genes
    violin_genes = ["KRT14", "KRT15", "KRT12", "KRT3",
                    "CHRM3", "MYH11", "DCN", "ACTA2"]
    gorder = ["cluster0", "basal", "LSC_other", "diff"]
    gcolors = ["#D62728", "#2CA02C", "#1F77B4", "#9467BD"]
    fig, axes = plt.subplots(1, len(violin_genes),
                             figsize=(22, 5), sharey=True)
    for i, sym in enumerate(violin_genes):
        ax = axes[i]
        if sym not in resolved:
            ax.set_title(f"{sym}\n(absent)", fontsize=10)
            ax.set_xticks([])
            continue
        vn = vn2pos[resolved[sym]]
        col = counts[:, vn]
        vdata = []
        for gname in gorder:
            rows = np.flatnonzero(group_sets[gname])
            vals = np.log1p(col[rows].toarray().ravel())
            vdata.append(vals)
        parts = ax.violinplot(vdata, positions=[1, 2, 3, 4],
                              showmedians=True, widths=0.85)
        for pc, colr in zip(parts["bodies"], gcolors):
            pc.set_facecolor(colr)
            pc.set_alpha(0.55)
            pc.set_edgecolor(colr)
        for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(0.8)
        ax.set_title(sym, fontsize=10)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["c0", "basal", "LSC", "diff"], fontsize=8)
        if i == 0:
            ax.set_ylabel("log1p(counts)", fontsize=9)
    fig.suptitle("Marker expression — cluster 0 vs established epithelial "
                 "reference groups (counts layer, log1p)", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"marker_violin.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    log("saved marker_violin")

    # report skeleton
    add("## Evidence chain")
    add("")
    add("(filled on the local side by the orchestrator)")
    add("")
    add("## Recommendation (tendency, not final decision)")
    add("")
    add("(filled on the local side by the orchestrator)")
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    (OUT / "cluster0_forensics.json").write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False)
    )
    log(f"Wrote {OUT / 'report.md'} and {OUT / 'cluster0_forensics.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
