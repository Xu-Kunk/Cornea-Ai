#!/usr/bin/env python
"""Assemble GSE202352 (Wiedemann 2023, human skin 10x) + marker_axis stage.

Mirrors the locked GSE147424 pipeline (scripts/epidermis_tpoac_eval.py
stage marker_axis) on a NEW cross-tissue control dataset:

  - Reads all 12 samples with sc.read_10x_mtx (skips macOS `._` junk), and
    concatenates into one AnnData (cells x genes, var = gene symbols),
    storing per-cell donor/site/compartment metadata parsed from the
    file names.
  - Applies the EXACT marker_axis logic: marker-positive filter
    pos = (basal_m > 0) | (diff_m > 0); s-score = mean(z(log1p diff)) -
    mean(z(log1p basal)) z-scored within marker-positive cells; quartile
    binning -> ordinal_idx 0..3 -> ordinal_layer names.
  - Saves processed h5ad (marker-positive cells only) with obs columns
    marker_score / ordinal_layer / ordinal_idx / cell_type /
    author_cell_type / tissue / disease / donor_id / site / compartment /
    n_counts and var = gene symbols.

File-name schema (GSM61118XX_Sample_721_SubjectN_<site>[_epi|_dermis]):
  donor_id    = SubjectN
  site        = hip | palm | sole
  compartment = epi | dermis | unknown (Subject1-2 samples carry no suffix)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "epidermis" / "gse202352" / "extracted"
OUT_DIR = ROOT / "artifacts" / "paper1_epidermis_eval_gse202352"
OUT_H5AD = OUT_DIR / "gse202352_processed.h5ad"
STATS_JSON = OUT_DIR / "process_stats.json"

BASAL_MARKERS = ["KRT5", "KRT14", "TP63"]
DIFF_MARKERS = ["KRT1", "KRT10", "IVL", "FLG"]
LAYER_ORDER = ["basal", "suprabasal", "spinous", "differentiated"]
ORDER_MAP = {c: i for i, c in enumerate(LAYER_ORDER)}

# GSM6111848 (Subject3 hip_dermis) is corrupt at the GEO source: only ~10% of
# its matrix.mtx.gz gzip stream decompresses ("invalid block type" at offset
# 10.1%); the md5 matches the file re-downloaded directly from GEO FTP, and no
# alternate deposit exists. Excluding it (11/12 samples) keeps the pipeline
# logic identical; coverage still includes all 4 subjects, all 3 sites and
# both compartments.
EXCLUDED = {"GSM6111848": "corrupt at GEO source (matrix.mtx.gz)"}


def parse_sample_name(name: str) -> dict:
    """Parse GSM6111844_Sample_721_Subject1_hip[_epi|_dermis] -> metadata."""
    base = name.removesuffix(".filtered_feature_bc_matrix.tar.gz")
    parts = base.split("_")
    assert parts[0].startswith("GSM"), name
    meta = {
        "gsm": parts[0],
        "sample_id": parts[2],
        "donor_id": parts[3],          # SubjectN
        "site": parts[4],              # hip | palm | sole
        "compartment": parts[5] if len(parts) > 5 else "unknown",
    }
    assert meta["donor_id"].startswith("Subject"), name
    assert meta["site"] in ("hip", "palm", "sole"), name
    assert meta["compartment"] in ("epi", "dermis", "unknown"), name
    return meta


def load_all_samples() -> sc.AnnData:
    """Read the samples (skip `._` junk and corrupt GSM6111848) and concatenate."""
    sample_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    sample_dirs = [p for p in sample_dirs if p.name not in EXCLUDED]
    print(f"loading {len(sample_dirs)}/{len(EXCLUDED) + len(sample_dirs)} samples "
          f"(excluded: {sorted(EXCLUDED)})", flush=True)
    adatas = []
    for d in sample_dirs:
        # exactly one Cell Ranger dir per sample (no `._` AppleDouble junk)
        cr_dirs = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith("._")]
        assert len(cr_dirs) == 1, (d, [p.name for p in cr_dirs])
        cr_dir = cr_dirs[0]
        a = sc.read_10x_mtx(cr_dir, var_names="gene_symbols", make_unique=False)
        meta = parse_sample_name(f"{d.name}_{cr_dir.name}.tar.gz")
        a.obs["gsm"] = meta["gsm"]
        a.obs["sample_id"] = meta["sample_id"]
        a.obs["donor_id"] = meta["donor_id"]
        a.obs["site"] = meta["site"]
        a.obs["compartment"] = meta["compartment"]
        print(f"  {d.name}: {a.n_obs} cells x {a.n_vars} genes -> {meta}", flush=True)
        adatas.append(a)
    # concat requires a unique var index -> dedupe per sample first
    # (symbol sets are identical across all 12 samples)
    adatas = [dedupe_var_names(a) for a in adatas]
    adata = sc.concat(adatas, join="outer", fill_value=0)
    print(f"concatenated: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    return adata


def dedupe_var_names(adata: sc.AnnData) -> sc.AnnData:
    """Keep the highest-expressed copy of duplicated gene symbols.

    GSE202352's filtered_feature_bc_matrix annotation carries 91 duplicated
    symbols (e.g. U1, PNRC2: multiple ENSG -> one symbol). The locked Wang
    dataset had unique symbols, so this is a GSE202352-only cleanup:
    duplicated var_names would break h5ad writing and the Geneformer token
    mapping. The 7 epidermal marker genes are unaffected.
    """
    n_dup = adata.n_vars - len(set(adata.var_names))
    if n_dup == 0:
        return adata
    tot = np.asarray(adata.X.sum(axis=0)).ravel()
    order = np.lexsort((tot, adata.var_names))  # symbol, then counts asc
    keep = []
    seen = set()
    for i in order:
        sym = adata.var_names[i]
        if sym not in seen:  # last occurrence per symbol = max counts
            seen.add(sym)
            keep.append(i)
    keep = np.asarray(keep)
    adata = adata[:, keep].copy()
    print(f"  deduped {n_dup} duplicated gene symbols -> {adata.n_vars} genes", flush=True)
    return adata


def marker_axis(adata: sc.AnnData) -> tuple[dict, sc.AnnData]:
    """Exact marker_axis logic from epidermis_tpoac_eval.py."""
    x = adata.X
    if sp.issparse(x):
        x = x.toarray()
    x = np.nan_to_num(x, nan=0.0)  # dropout -> 0 (10x has no NaN; keep parity)
    var = adata.var_names
    idx = {g: i for i, g in enumerate(var)}
    log1p = np.log1p(x)

    def group_mean(genes):
        return np.mean([log1p[:, idx[g]] for g in genes if g in idx], axis=0)

    basal_m = group_mean(BASAL_MARKERS)
    diff_m = group_mean(DIFF_MARKERS)
    pos = (basal_m > 0) | (diff_m > 0)

    def group_zscore(genes, mask):
        m = group_mean(genes)
        mv = m[mask]
        return (m - mv.mean()) / (mv.std() + 1e-9)

    basal_z = group_zscore(BASAL_MARKERS, pos)
    diff_z = group_zscore(DIFF_MARKERS, pos)
    s = np.full(len(pos), np.nan)
    s[pos] = diff_z[pos] - basal_z[pos]  # high = differentiated
    s_ret = s[pos]
    q = np.quantile(s_ret, [0.25, 0.5, 0.75])
    ordinal = np.full(len(pos), -1, dtype=int)
    ordinal[pos] = np.digitize(s_ret, q).clip(0, 3)  # 0..3

    stats = {
        "n_samples": int(len(set(adata.obs["gsm"]))),
        "excluded_samples": EXCLUDED,
        "n_cells_raw": int(len(pos)),
        "n_cells_marker_positive": int(pos.sum()),
        "frac_marker_positive": float(pos.mean()),
        "n_genes": int(adata.n_vars),
        "s_q25_q50_q75_retained": [float(x) for x in q],
        "frac_strong_basal_retained": float((s_ret < -1.0).mean()),
        "frac_strong_diff_retained": float((s_ret > 1.0).mean()),
        "ordinal_counts_retained": np.bincount(ordinal[pos], minlength=4).tolist(),
        "ordinal_counts_by_layer": {
            c: int((ordinal[pos] == i).sum()) for i, c in enumerate(LAYER_ORDER)
        },
        "coverage": {
            "donor": adata.obs["donor_id"].value_counts().to_dict(),
            "site": adata.obs["site"].value_counts().to_dict(),
            "compartment": adata.obs["compartment"].value_counts().to_dict(),
        },
    }
    # save processed h5ad (marker-positive cells only), obs per task spec
    obs = adata.obs[pos].copy()
    obs["marker_score"] = s[pos]
    obs["ordinal_layer"] = [LAYER_ORDER[k] for k in ordinal[pos]]
    obs["ordinal_idx"] = ordinal[pos].astype(int)
    obs["cell_type"] = obs["ordinal_layer"].astype(str)
    obs["author_cell_type"] = obs["ordinal_layer"].astype(str)
    obs["tissue"] = "skin"
    obs["disease"] = "healthy"
    obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()[pos]
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    a2 = sp.csr_matrix(x[pos])
    import anndata as ad

    a2 = ad.AnnData(X=sp.csr_matrix(x[pos]), obs=obs, var=adata.var)
    a2.write_h5ad(OUT_H5AD)
    stats["processed_h5ad"] = str(OUT_H5AD)
    stats["processed_cells"] = int(a2.n_obs)
    return stats, a2


def main() -> None:
    print("loading GSE202352 samples ...", flush=True)
    adata = load_all_samples()
    print("running marker_axis ...", flush=True)
    stats, a2 = marker_axis(adata)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    print(
        "per-layer counts: "
        + ", ".join(f"{c}={int((a2.obs['ordinal_layer'] == c).sum())}" for c in LAYER_ORDER),
        flush=True,
    )


if __name__ == "__main__":
    main()
