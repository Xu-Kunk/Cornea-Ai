#!/usr/bin/env python3
"""GSE299152 human-only external test set preparation (raw counts only).

Steps: load 3 human donors (Donor2046/2206/3748) from filtered_feature_bc_matrix.h5,
per-donor QC (cells x genes, mean counts, mito%), concatenate with donor obs,
marker-axis check (TP63/KRT15/KRT14 vs KRT12/KRT3), write QC_report.md + raw h5ad.
No integration / Harmony / clustering / annotation / normalization.
"""
import os
import sys
import numpy as np
import scanpy as sc
import anndata as ad

OUT = "/home/dq/Codex/Ai/data/gse299152"
DONORS = ["Donor2046", "Donor2206", "Donor3748"]
MITO_PREFIX = "MT-"

# Marker genes: basal/LSC pole vs differentiated pole
BASAL = ["TP63", "KRT15", "KRT14"]
DIFF = ["KRT12", "KRT3"]

def per_donor_stats(adata: ad.AnnData, donor: str) -> dict:
    X = adata.X  # scipy sparse
    total_counts = np.asarray(X.sum(axis=1)).ravel()
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    mean_counts = float(total_counts.mean())
    # mitochondrial fraction
    mito_genes = [g for g in adata.var_names if str(g).startswith(MITO_PREFIX)]
    if mito_genes:
        mito_idx = adata.var_names.isin(mito_genes)
        mito_counts = np.asarray(X[:, mito_idx].sum(axis=1)).ravel()
        mito_pct = float(100.0 * np.sum(mito_counts) / np.sum(total_counts))
    else:
        mito_pct = float("nan")
    stats = {
        "donor": donor,
        "cells": n_cells,
        "genes": n_genes,
        "mean_counts": round(mean_counts, 1),
        "total_counts": int(total_counts.sum()),
        "mito_genes_detected": len(mito_genes),
        "mito_pct": round(mito_pct, 2),
    }
    # marker expression: mean over cells (raw counts), per donor
    marker_expr = {}
    for gene in BASAL + DIFF:
        if gene in adata.var_names:
            gidx = list(adata.var_names).index(gene)
            marker_expr[gene] = float(np.asarray(X[:, gidx].todense()).mean()) if hasattr(X[:, gidx], "todense") else float(X[:, gidx].mean())
        else:
            marker_expr[gene] = None
    stats["marker_expr"] = marker_expr
    return stats


def main():
    adatas, stats_list = [], []
    for donor in DONORS:
        path = os.path.join(OUT, f"GSE299152_{donor}_filtered_feature_bc_matrix.h5")
        if not os.path.exists(path):
            print(f"MISSING FILE: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"[load] {donor}: {path}", flush=True)
        adata = sc.read_10x_h5(path, gex_only=True)  # RNA only; excludes ADT
        adata.var_names_make_unique()
        adata.obs["donor"] = donor
        st = per_donor_stats(adata, donor)
        stats_list.append(st)
        adatas.append(adata)
        print(f"  -> {donor}: {adata.n_obs} cells x {adata.n_vars} genes, "
              f"mean counts={st['mean_counts']}, mito%={st['mito_pct']}", flush=True)

    # Concatenate
    print("[concat] concatenating 3 donors...", flush=True)
    combined = ad.concat(adatas, join="outer", index_unique="-", merge="same")
    combined.obs_names_make_unique()
    print(f"  -> combined: {combined.n_obs} cells x {combined.n_vars} genes", flush=True)

    # Recompute marker expression on the concatenated object (gene union may differ)
    def gene_expr(adata, gene):
        if gene not in adata.var_names:
            return None
        gidx = list(adata.var_names).index(gene)
        col = adata.X[:, gidx]
        if hasattr(col, "todense"):
            return float(np.asarray(col.todense()).mean())
        return float(col.mean())

    # per-donor marker means on combined object (consistent gene space)
    per_donor_marker = {}
    for donor in DONORS:
        sub = combined[combined.obs["donor"] == donor]
        per_donor_marker[donor] = {g: gene_expr(sub, g) for g in BASAL + DIFF}

    # Overall marker means
    marker_overall = {g: gene_expr(combined, g) for g in BASAL + DIFF}

    # Axis resolvability: basal pole present (TP63/KRT15/KRT14 expressed), diff pole present (KRT12/KRT3)
    def pole_present(expr_map):
        vals = [v for v in expr_map.values() if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    basal_mean = pole_present({g: marker_overall[g] for g in BASAL})
    diff_mean = pole_present({g: marker_overall[g] for g in DIFF})
    resolvable = (basal_mean > 0.05) and (diff_mean > 0.05)
    # per-donor check
    per_donor_ok = True
    for donor in DONORS:
        b = pole_present({g: per_donor_marker[donor][g] for g in BASAL})
        d = pole_present({g: per_donor_marker[donor][g] for g in DIFF})
        if not (b > 0.05 and d > 0.05):
            per_donor_ok = False

    verdict = (
        "RESOLVABLE: both poles expressed (basal/LSC mean {:.2f} vs differentiated mean {:.2f}); "
        "the LSC/TAC/basal/differentiated ordinal axis is detectable on raw counts.".format(basal_mean, diff_mean)
        if resolvable and per_donor_ok else
        "LIMITED: one or both marker poles below detection threshold; axis may not be resolvable."
    )

    # Write QC report
    report = []
    report.append("# GSE299152 — Human External Test Set QC Report")
    report.append("")
    report.append("Dataset: Single-cell mapping of the glycocalyx in the adult human and mouse ocular surface (PNAS Nexus 2025)")
    report.append("Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE299152")
    report.append("Files: NCBI GEO FTP suppl (filtered_feature_bc_matrix.h5, 10x Genomics)")
    report.append("")
    report.append("## GSM -> Donor mapping (GEO record)")
    report.append("")
    report.append("| GSM | Sample | Included |")
    report.append("|-----|--------|----------|")
    report.append("| GSM9034550 | Donor3748_mRNA | YES |")
    report.append("| GSM9034552 | Donor2046_mRNA | YES |")
    report.append("| GSM9034554 | Donor2206_mRNA | YES |")
    report.append("| GSM9034551 / 9034553 / 9034555 | Donor ADT (protein) | no (ADT) |")
    report.append("| GSM9034556-90345561 | Mice1/4/Conj1 (mRNA+ADT) | no (mouse) |")
    report.append("")
    report.append("## Per-donor QC (RNA only, raw counts)")
    report.append("")
    report.append("| Donor | Cells | Genes | Mean counts/cell | Total counts | Mito genes found | Mito % |")
    report.append("|-------|-------|-------|------------------|--------------|------------------|--------|")
    for st in stats_list:
        report.append("| {} | {} | {} | {:.1f} | {} | {} | {} |".format(
            st["donor"], st["cells"], st["genes"], st["mean_counts"],
            st["total_counts"], st["mito_genes_detected"], st["mito_pct"]))
    report.append("")
    report.append("## Concatenated object")
    report.append("")
    report.append("- Shape: {} cells x {} genes (expected ~26,628 x ~36,610)".format(combined.n_obs, combined.n_vars))
    report.append("- obs column: `donor` (Donor2046 / Donor2206 / Donor3748)")
    report.append("- Raw counts only; no normalization, integration, Harmony, clustering or annotation (later spec).")
    report.append("- Mouse samples excluded.")
    report.append("")
    report.append("## Marker-axis check (mean raw expression per donor)")
    report.append("")
    report.append("| Donor | TP63 | KRT15 | KRT14 | KRT12 | KRT3 |")
    report.append("|-------|------|-------|-------|-------|------|")
    for donor in DONORS:
        m = per_donor_marker[donor]
        report.append("| {} | {} | {} | {} | {} | {} |".format(
            donor,
            "%.3f" % m["TP63"] if m["TP63"] is not None else "n.d.",
            "%.3f" % m["KRT15"] if m["KRT15"] is not None else "n.d.",
            "%.3f" % m["KRT14"] if m["KRT14"] is not None else "n.d.",
            "%.3f" % m["KRT12"] if m["KRT12"] is not None else "n.d.",
            "%.3f" % m["KRT3"] if m["KRT3"] is not None else "n.d."))
    report.append("| **Overall** | {} | {} | {} | {} | {} |".format(
        "%.3f" % marker_overall["TP63"] if marker_overall["TP63"] is not None else "n.d.",
        "%.3f" % marker_overall["KRT15"] if marker_overall["KRT15"] is not None else "n.d.",
        "%.3f" % marker_overall["KRT14"] if marker_overall["KRT14"] is not None else "n.d.",
        "%.3f" % marker_overall["KRT12"] if marker_overall["KRT12"] is not None else "n.d.",
        "%.3f" % marker_overall["KRT3"] if marker_overall["KRT3"] is not None else "n.d."))
    report.append("")
    report.append("## Verdict")
    report.append("")
    report.append(verdict)
    report.append("")

    report_path = os.path.join(OUT, "QC_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(report))
    print(f"[write] {report_path}", flush=True)

    # Save concatenated raw object
    h5ad_path = os.path.join(OUT, "gse299152_human_raw.h5ad")
    print(f"[write] {h5ad_path}", flush=True)
    combined.write_h5ad(h5ad_path)

    print("\n===== SUMMARY =====", flush=True)
    print(f"Combined: {combined.n_obs} cells x {combined.n_vars} genes", flush=True)
    for st in stats_list:
        print(f"  {st['donor']}: {st['cells']} cells, {st['genes']} genes, "
              f"mean counts {st['mean_counts']}, mito% {st['mito_pct']}", flush=True)
    print("Marker overall means:", {k: (round(v, 4) if v is not None else None) for k, v in marker_overall.items()}, flush=True)
    print("VERDICT:", verdict, flush=True)


if __name__ == "__main__":
    main()
