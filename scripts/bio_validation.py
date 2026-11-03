#!/usr/bin/env python
"""CNS-anchor biological validation for Paper 1 — marker-program monotonicity
along the model's differentiation axis.

Question: does the differentiation axis measured in each arm's embedding
(order the literature's published biology) — and is that biology preserved by
the ordinal recipe and erased by naive fine-tuning?

Two tissues, two Cell-published anchors:
  * Cornea (GSE153515): Cotsarelis et al. 1989 Cell — limbal stem cells are
    slow-cycling cells in the limbus; quiescent LSC -> proliferative TA
    compartment -> corneal differentiation. Programs: stemness
    (TP63/KRT15/KRT19, decreasing), corneal differentiation (KRT12/KRT3,
    increasing), cycling (PCNA/MCM2/MKI67/TOP2A, peak at TAC).
  * Epidermis (GSE147424): Fuchs & Green 1980 Cell — keratin switching during
    terminal differentiation. Programs: basal (KRT5/KRT14/COL17A1,
    decreasing), differentiation (KRT1/KRT10, increasing), cycling
    (MKI67/PCNA, decreasing from the proliferative basal compartment).

Independent (non-annotation) readouts are flagged: TP63, PCNA, MCM2 (cornea);
COL17A1, MKI67, PCNA (epidermis) are absent from the locked annotation marker
sets / the epidermis marker-axis definition.

Axis: centroid-PCA pseudotime, the paper's locked metric
(corneal_fm.evaluation.marker_gradient_correlation internals).

Outputs:
  artifacts/paper1_bio_validation/bio_validation_results.json
  artifacts/paper1_bio_validation/figure_data_deciles.csv

Usage (CPU):
    /home/dq/venv-ds/bin/python3 scripts/bio_validation.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy.stats as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_bio_validation"

# --- locked class orders -----------------------------------------------------
CORNEAL_LAYERS = ["LSC", "TAC", "basal", "differentiated"]
EPIDERMAL_LAYERS = ["basal", "suprabasal", "spinous", "differentiated"]

# --- cornea gene programs (symbol -> ENSG; ENSG in the GSE153515 matrices) ---
CORNEAL_GENES: dict[str, dict[str, str]] = {
    "stemness": {"TP63": "ENSG00000073282",
                 "KRT15": "ENSG00000171346",
                 "KRT19": "ENSG00000171345",
                 "NFIB": "ENSG00000147862",
                 "CDH3": "ENSG00000062038"},
    "differentiation": {"KRT12": "ENSG00000187242",
                        "KRT3": "ENSG00000186480"},
    "cycling": {"PCNA": "ENSG00000132646",
                "MCM2": "ENSG00000073111",
                "MKI67": "ENSG00000148773",
                "TOP2A": "ENSG00000131747"},
}
# absent from the locked annotation marker sets
CORNEAL_INDEPENDENT = ["TP63", "PCNA", "MCM2", "NFIB", "CDH3"]

# --- epidermis gene programs (symbols in the epidermis matrix) ---------------
EPIDERMAL_GENES: dict[str, list[str]] = {
    "basal": ["KRT5", "KRT14", "COL17A1"],
    "differentiation": ["KRT1", "KRT10"],
    "cycling": ["MKI67", "PCNA"],
}
# absent from the epidermis marker-axis definition (KRT1/KRT10/IVL/FLG/KRT5/
# KRT14/TP63)
EPIDERMAL_INDEPENDENT = ["COL17A1", "MKI67", "PCNA"]

N_BOOT = 500
N_NULL = 500
BOOT_SEED = 42

# checkpoint label -> per-cell embedding npz (GSE153515, layer-12 CLS)
CORNEAL_ARMS: dict[str, dict] = {
    "frozen": {
        "npz": "paper1_tier_b_eval/percell_frozen_gf.npz",
        "key": "emb_ext", "seed": None},
    "naive_42": {"npz": "paper1_tier_b_eval/percell_naive_s42.npz",
                 "key": "emb_ext", "seed": 42},
    "naive_123": {"npz": "paper1_tier_b_eval/percell_naive_s123.npz",
                  "key": "emb_ext", "seed": 123},
    "naive_456": {"npz": "paper1_tier_b_eval/percell_naive_s456.npz",
                  "key": "emb_ext", "seed": 456},
    "caft_42": {"npz": "paper1_tier_b_eval/percell_caft_s42.npz",
                "key": "emb_ext", "seed": 42},
    "caft_123": {"npz": "paper1_tier_b_eval/percell_caft_s123.npz",
                 "key": "emb_ext", "seed": 123},
    "caft_456": {"npz": "paper1_tier_b_eval/percell_caft_s456.npz",
                 "key": "emb_ext", "seed": 456},
    "ordinal0_42": {"npz": "paper1_bio_validation/ordinal_emb_lam0.0_seed42.npz",
                    "key": "emb", "seed": 42},
    "ordinal0_123": {"npz": "paper1_bio_validation/ordinal_emb_lam0.0_seed123.npz",
                     "key": "emb", "seed": 123},
    "ordinal0_456": {"npz": "paper1_bio_validation/ordinal_emb_lam0.0_seed456.npz",
                     "key": "emb", "seed": 456},
    "ordinal5_42": {"npz": "paper1_bio_validation/ordinal_emb_lam0.5_seed42.npz",
                    "key": "emb", "seed": 42},
    "ordinal5_123": {"npz": "paper1_bio_validation/ordinal_emb_lam0.5_seed123.npz",
                     "key": "emb", "seed": 123},
    "ordinal5_456": {"npz": "paper1_bio_validation/ordinal_emb_lam0.5_seed456.npz",
                     "key": "emb", "seed": 456},
}
SEED_GROUPS = {
    "naive": ["naive_42", "naive_123", "naive_456"],
    "caft": ["caft_42", "caft_123", "caft_456"],
    "ordinal0": ["ordinal0_42", "ordinal0_123", "ordinal0_456"],
    "ordinal5": ["ordinal5_42", "ordinal5_123", "ordinal5_456"],
}

EPIDERMAL_ARMS = ["frozen", "ce_naive", "corns_ordinal", "caft",
                  "aa_ft_newaxis", "ss_newaxis"]


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def axis_pseudotime(
    emb: np.ndarray,
    labels: np.ndarray,
    layer_order: list[str],
) -> np.ndarray:
    """Centroid-PCA pseudotime (paper's locked metric internals)."""
    order_map = {c: i for i, c in enumerate(layer_order)}
    valid = np.array([lbl in order_map for lbl in labels])
    lab = labels[valid]
    e = emb[valid]
    ordinals = np.array([order_map[x] for x in lab], dtype=float)
    present = [c for c in layer_order if (ordinals == order_map[c]).any()]
    centroids = np.vstack([e[ordinals == order_map[c]].mean(axis=0)
                           for c in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[0]
    first = e[ordinals == order_map[present[0]]].mean(axis=0)
    last = e[ordinals == order_map[present[-1]]].mean(axis=0)
    if (last - first) @ axis < 0:
        axis = -axis
    pt = np.full(len(emb), np.nan)
    pt[valid] = e @ axis
    return pt


def program_score(expr: np.ndarray, gene_idx: list[int]) -> np.ndarray:
    """Mean of z-scored log1p expression across program genes."""
    sub = np.log1p(expr[:, gene_idx])
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    z = (sub - sub.mean(axis=0)) / (sub.std(axis=0) + 1e-12)
    return z.mean(axis=1)


def _bootstrap_ci(pt: np.ndarray, prog: np.ndarray, n_boot: int,
                  seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(pt)
    p, s = pt[ok], prog[ok]
    rhos = np.empty(n_boot)
    n = len(p)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[i] = st.spearmanr(p[idx], s[idx]).statistic
    return {"ci_lower": float(np.percentile(rhos, 2.5)),
            "ci_upper": float(np.percentile(rhos, 97.5))}


def _permutation_null(pt: np.ndarray, prog: np.ndarray, n_null: int,
                      seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(pt)
    p, s = pt[ok], prog[ok]
    obs = st.spearmanr(p, s).statistic
    null = np.empty(n_null)
    n = len(s)
    for i in range(n_null):
        null[i] = st.spearmanr(p, s[rng.permutation(n)]).statistic
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return {"observed": float(obs),
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "z": float(z),
            "p_empirical": float((np.abs(null) >= abs(obs)).mean())}


def analyze_arm(
    emb: np.ndarray,
    labels: np.ndarray,
    expr: np.ndarray,
    gene_lookup: dict[str, str],
    var_names: np.ndarray,
    programs: dict[str, list[str]],
    layer_order: list[str],
    independent: list[str],
) -> dict:
    """gene_lookup maps gene symbol -> identifier in var_names (ENSG or symbol)."""
    pt = axis_pseudotime(emb, labels, layer_order)
    present = {}
    for sym, gene_id in gene_lookup.items():
        hit = np.where(var_names == gene_id)[0]
        if len(hit):
            present[sym] = int(hit[0])
    out: dict = {"n_cells": int(emb.shape[0]), "genes_present": present}
    for prog, syms in programs.items():
        idx = [present[s] for s in syms if s in present]
        if not idx:
            continue
        prog_expr = expr[:, idx]
        ok = ~np.isnan(pt)
        ps = program_score(prog_expr, list(range(len(idx))))
        rho = st.spearmanr(pt[ok], ps[ok]).statistic
        entry = {
            "rho": float(rho),
            "ci": _bootstrap_ci(pt, ps, N_BOOT, BOOT_SEED),
            "null": _permutation_null(pt, ps, N_NULL, BOOT_SEED),
            "genes": [s for s in syms if s in present],
            "independent": all(s in independent for s in syms if s in present),
        }
        # pseudotime decile profile (arm-dependent): mean program score per
        # decile of the arm's own pseudotime; peak decile of the cycling
        # program localizes the proliferative TA compartment along the axis.
        ok_pt = ~np.isnan(pt)
        deciles = np.quantile(pt[ok_pt], np.arange(0, 1.01, 0.1))
        dec_means = []
        for d in range(10):
            mask = (pt[ok_pt] >= deciles[d]) & (pt[ok_pt] <= deciles[d + 1])
            dec_means.append(float(ps[ok_pt][mask].mean()) if mask.sum() else np.nan)
        entry["pt_decile_profile"] = dec_means
        out[f"program_{prog}"] = entry
    # per-gene monotonicity
    for sym, i in present.items():
        ok = ~np.isnan(pt)
        col = expr[:, i]
        if hasattr(col, "toarray"):
            col = col.toarray().flatten()
        rho = st.spearmanr(pt[ok], np.log1p(col[ok])).statistic
        out[f"gene_{sym}"] = {
            "rho": float(rho),
            "independent": sym in independent,
        }
    return out


def load_corneal_data():
    import anndata as ad

    # expression from the RAW matrix (retains MKI67 etc. lost in QC filtering),
    # aligned to the 14,775 epithelial eval cells by barcode
    ann = ad.read_h5ad(str(PROJECT_ROOT / "data/gse153515_annotated.h5ad"),
                       backed="r")
    raw = ad.read_h5ad(str(PROJECT_ROOT / "data/gse153515_limbal_stem.h5ad"),
                       backed="r")
    labels = np.asarray(ann.obs["cell_type_harmonized"].astype(str))
    keep = np.where(labels != "NON_LAYER")[0]
    eval_barcodes = np.asarray(ann.obs_names[keep], dtype=str)
    eval_labels = labels[keep]
    raw_bc = np.asarray(raw.obs_names, dtype=str)
    loc = {b: i for i, b in enumerate(raw_bc)}
    raw_idx = np.array([loc[b] for b in eval_barcodes])
    expr = raw.X[raw_idx].toarray() if hasattr(raw.X, "toarray") else np.asarray(raw.X[raw_idx])
    # raw var_names are ENSG ids
    return eval_labels, expr, np.asarray(raw.var_names, dtype=str)

def load_epidermal_data(name: str = "wang2020"):
    import anndata as ad

    # the processed epidermis object (ordinal_layer labels, symbol var_names)
    if name == "wang2020":
        h5 = PROJECT_ROOT / "artifacts/paper1_epidermis_eval/wang2020_processed.h5ad"
    elif name == "gse202352":
        h5 = PROJECT_ROOT / "artifacts/paper1_epidermis_eval_gse202352/gse202352_processed.h5ad"
    else:
        raise ValueError(f"unknown epidermis dataset {name}")
    a = ad.read_h5ad(str(h5), backed="r")
    expr = a.X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.nan_to_num(expr, nan=0.0)
    labels = np.asarray(a.obs["ordinal_layer"].astype(str))
    return labels, expr, np.asarray(a.var_names, dtype=str)


def expression_compartment(
    expr: np.ndarray,
    labels: np.ndarray,
    gene_lookup: dict[str, str],
    var_names: np.ndarray,
    programs: dict[str, list[str]],
    layer_order: list[str],
) -> dict:
    """Arm-invariant: mean program score per annotated class (data structure)."""
    present = {}
    for sym, gene_id in gene_lookup.items():
        hit = np.where(var_names == gene_id)[0]
        if len(hit):
            present[sym] = int(hit[0])
    out: dict = {}
    for prog, syms in programs.items():
        idx = [present[s] for s in syms if s in present]
        if not idx:
            continue
        ps = program_score(expr, idx)
        out[prog] = {
            c: float(ps[labels == c].mean())
            for c in layer_order if (labels == c).any()
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"timestamp_utc": datetime.now(UTC).isoformat(),
                     "n_boot": N_BOOT, "n_null": N_NULL}

    # ---- cornea -------------------------------------------------------------
    print(f"{_ts()} loading corneal expression (raw matrix, barcode-aligned)")
    cor_labels, cor_expr, cor_var = load_corneal_data()
    # cornea var_names are ENSG ids: look up programs by their ENSG identifier
    cor_lookup: dict[str, str] = {}
    for prog_syms in CORNEAL_GENES.values():
        cor_lookup.update(prog_syms)
    cor_results: dict[str, dict] = {}
    for arm, spec in CORNEAL_ARMS.items():
        p = PROJECT_ROOT / "artifacts" / spec["npz"]
        if not p.exists():
            print(f"{_ts()} SKIP cornea arm {arm}: {p.name} missing")
            continue
        d = np.load(p)
        emb = d[spec["key"]]
        y = d["labels"] if "labels" in d.files else d["y_true_ext"]
        # alignment guard: eval labels must match the annotated harmonization
        if not np.array_equal(np.asarray(y), cor_labels):
            raise RuntimeError(f"label misalignment in {arm}")
        cor_results[arm] = analyze_arm(
            emb, cor_labels, cor_expr, cor_lookup, cor_var,
            {k: list(v.keys()) for k, v in CORNEAL_GENES.items()},
            CORNEAL_LAYERS, CORNEAL_INDEPENDENT)
        print(f"{_ts()} cornea {arm}: "
              f"{ {k.split('_')[1]: round(v['rho'], 3) for k, v in cor_results[arm].items() if k.startswith('program_')} }")
    results["cornea"] = {"layer_order": CORNEAL_LAYERS,
                         "arms": cor_results,
                         "expression_compartment": expression_compartment(
                             cor_expr, cor_labels, cor_lookup, cor_var,
                             {k: list(v.keys()) for k, v in CORNEAL_GENES.items()},
                             CORNEAL_LAYERS),
                         "programs": {k: list(v.keys()) for k, v in CORNEAL_GENES.items()},
                         "independent": CORNEAL_INDEPENDENT}

    # ---- epidermis (both cross-tissue datasets) ------------------------------
    results["epidermis"] = {}
    for epid_name, epid_dir in (("wang2020", "paper1_epidermis_eval"),
                                ("gse202352", "paper1_epidermis_eval_gse202352")):
        print(f"{_ts()} loading epidermal expression ({epid_name})")
        ep_labels, ep_expr, ep_var = load_epidermal_data(epid_name)
        npz = PROJECT_ROOT / "artifacts" / epid_dir / "embeddings.npz"
        if not npz.exists():
            print(f"{_ts()} SKIP epidermis {epid_name}: embeddings.npz missing")
            continue
        emb_all = np.load(str(npz))
        ep_results: dict[str, dict] = {}
        for arm in EPIDERMAL_ARMS:
            if arm not in emb_all.files:
                continue
            ep_results[arm] = analyze_arm(
                emb_all[arm], ep_labels, ep_expr,
                {g: g for p in EPIDERMAL_GENES.values() for g in p},
                ep_var, EPIDERMAL_GENES, EPIDERMAL_LAYERS, EPIDERMAL_INDEPENDENT)
            print(f"{_ts()} epidermis[{epid_name}] {arm}: "
                  f"{ {k.split('_')[1]: round(v['rho'], 3) for k, v in ep_results[arm].items() if k.startswith('program_')} }")
        results["epidermis"][epid_name] = {
            "layer_order": EPIDERMAL_LAYERS,
            "arms": ep_results,
            "expression_compartment": expression_compartment(
                ep_expr, ep_labels,
                {g: g for p in EPIDERMAL_GENES.values() for g in p},
                ep_var, EPIDERMAL_GENES, EPIDERMAL_LAYERS),
            "programs": EPIDERMAL_GENES,
            "independent": EPIDERMAL_INDEPENDENT,
        }

    # ---- 3-seed aggregation --------------------------------------------------
    agg: dict = {}
    for group, arms in SEED_GROUPS.items():
        rhos = {}
        for arm in arms:
            if arm not in cor_results:
                continue
            for k, v in cor_results[arm].items():
                if k.startswith("program_"):
                    rhos.setdefault(k, []).append(v["rho"])
        agg[group] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                          "values": [round(x, 4) for x in v]}
                      for k, v in rhos.items()}
    results["cornea_seed_aggregate"] = agg

    out_json = OUT_DIR / "bio_validation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{_ts()} wrote {out_json}")

    # ---- figure data export (CSV for the figure script) ---------------------
    rows = []
    for tissue in ("cornea", "epidermis"):
        tissue_res = results[tissue]
        if tissue == "epidermis":
            for epid_name, epid_res in tissue_res.items():
                for arm, arm_res in epid_res["arms"].items():
                    for k, v in arm_res.items():
                        if not k.startswith("program_"):
                            continue
                        rows.append({
                            "tissue": f"epidermis_{epid_name}",
                            "arm": arm, "program": k.split("_", 1)[1],
                            "rho": v["rho"], "ci_lower": v["ci"]["ci_lower"],
                            "ci_upper": v["ci"]["ci_upper"], "null_z": v["null"]["z"],
                            "layer_order": ",".join(epid_res["layer_order"]),
                        })
        else:
            layer_order = tissue_res["layer_order"]
            for arm, arm_res in tissue_res["arms"].items():
                for k, v in arm_res.items():
                    if not k.startswith("program_"):
                        continue
                    rows.append({
                        "tissue": tissue, "arm": arm, "program": k.split("_", 1)[1],
                        "rho": v["rho"], "ci_lower": v["ci"]["ci_lower"],
                        "ci_upper": v["ci"]["ci_upper"], "null_z": v["null"]["z"],
                        "layer_order": ",".join(layer_order),
                    })
    import csv

    csv_path = OUT_DIR / "figure_data_programs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{_ts()} wrote {csv_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
