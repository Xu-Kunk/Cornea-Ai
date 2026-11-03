"""CS_before: Continuity Score baseline computation (pre-fine-tune).

Purpose: Compute the Continuity Score (CS) baseline for the corneal
differentiation axis BEFORE any model fine-tuning, in PCA embedding space.

This establishes the GPR denominator: GPR = CS_after / CS_before.

Two stratification methods:
  Method A — Marker-gene expression gradients (KRT14/KRT12/TP63)
  Method B — Existing cell_type annotations from CELLxGENE

Output:
  - data/cs_before_results.json  (numeric results)
  - figures/cs_before_*.png      (publication-quality plots)

Environment compatibility:
  - Windows (local dev, paths via PROJECT_ROOT detection)
  - Linux ARM (ASUS <compute-node>, paths via CORNEAL_FM_DATA env var or relative)
  - Requires: scanpy, anndata, scikit-learn, scipy, matplotlib (CPU only)

Usage:
  python scripts/compute_cs_before.py
  python scripts/compute_cs_before.py --test_set  # use test set (fast)
  CORNEAL_FM_DATA=/mnt/data python scripts/compute_cs_before.py  # <compute-node>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # headless-safe (<compute-node> has no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config — environment-aware path resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Allow override via env var for <compute-node> deployment
DATA_DIR = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
FIG_DIR = Path(os.environ.get("CORNEAL_FM_FIGURES", str(PROJECT_ROOT / "figures")))
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_TOP_GENES = 3000

# Canonical corneal epithelial differentiation markers
MARKERS = {
    "stem": ["TP63", "KRT15", "KRT14"],          # LSC (limbal stem cells)
    "TAC": ["KRT14", "KRT5"],                    # Transit amplifying
    "basal": ["KRT14", "KRT5"],                  # Basal (shared with TAC)
    "wing": ["KRT3", "KRT4", "KRT12"],           # Wing cells
    "squamous": ["KRT12", "KRT3", "IVL"],        # Superficial squamous
}

# Differentiation axis (ordered): LSC → TAC → basal → wing → squamous
AXIS_LABELS = ["LSC", "TAC", "Basal", "Wing", "Squamous"]


# ---------------------------------------------------------------------------
# Part 1: Data Inspection
# ---------------------------------------------------------------------------

def inspect_data(adata: ad.AnnData) -> None:
    """Print available observations and markers for script validation."""
    print("=" * 64)
    print("  DATA INSPECTION")
    print("=" * 64)
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars:,}")
    print(f"\n  obs columns: {list(adata.obs.columns)}")
    print(f"  obsm keys:   {list(adata.obsm.keys()) if adata.obsm else '(none)'}")

    # Cell type categories
    if "cell_type" in adata.obs:
        ct = adata.obs["cell_type"]
        print(f"\n  cell_type categories ({ct.nunique()}):")
        for name, count in ct.value_counts().items():
            print(f"    {name:<45} {count:>6}")

    # Check marker availability (handle Ensembl ID var_names + feature_name column)
    print(f"\n  Marker gene availability:")
    all_markers = sorted(set(g for genes in MARKERS.values() for g in genes))
    if "feature_name" in adata.var.columns:
        available_symbols = set(adata.var["feature_name"].values)
        available = [g for g in all_markers if g in available_symbols]
        missing = [g for g in all_markers if g not in available_symbols]
        print(f"    Lookup via var['feature_name'] (var_names are Ensembl IDs)")
    else:
        available = [g for g in all_markers if g in adata.var_names]
        missing = [g for g in all_markers if g not in adata.var_names]
    print(f"    Available: {available}")
    if missing:
        print(f"    ⚠️ Missing:  {missing}")

    # Check for layer-specific annotations
    layer_like_cols = [c for c in adata.obs.columns
                       if any(k in c.lower() for k in ["layer", "strat", "zone", "region"])]
    if layer_like_cols:
        print(f"\n  🔍 Layer-like annotation columns found: {layer_like_cols}")
        for col in layer_like_cols:
            print(f"    {col}: {adata.obs[col].value_counts().to_dict()}")

    print("=" * 64)


# ---------------------------------------------------------------------------
# Part 2: Stratification Methods
# ---------------------------------------------------------------------------

def stratify_by_markers(
    adata: ad.AnnData,
    symbol_to_idx: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Method A: Assign cells to differentiation layers by marker expression.

    Uses normalized expression of canonical markers to score each cell
    along the LSC→Squamous axis.

    NOTE: Caller must ensure adata.X is already log1p-normalized.
    """
    print("\n--- Method A: Marker-gene stratification ---")

    def gene_score(genes: list[str], label: str) -> np.ndarray:
        """Mean expression of available genes in the list.
        
        Handles: direct var_names match, feature_name column, or
        pre-built symbol_to_idx positional index mapping.
        """
        # Strategy 1: var_names are already gene symbols
        if any(g in adata.var_names for g in genes):
            avail = [g for g in genes if g in adata.var_names]
            expr = adata[:, avail].X
            if issparse(expr):
                expr = expr.toarray()
            return np.asarray(expr, dtype=np.float32).mean(axis=1)

        # Strategy 2: pre-built positional index mapping (stable across .raw transforms)
        if symbol_to_idx:
            idxs = [symbol_to_idx[g] for g in genes if g in symbol_to_idx]
            if idxs:
                expr = adata.X[:, idxs]
                if issparse(expr):
                    expr = expr.toarray()
                found = [g for g in genes if g in symbol_to_idx]
                print(f"  {label:<10} (via positional index: {found})")
                return np.asarray(expr, dtype=np.float32).mean(axis=1)

        # Strategy 3: feature_name column in var
        if "feature_name" in adata.var.columns:
            snames = dict(zip(adata.var["feature_name"], adata.var_names))
            ids = [snames[g] for g in genes if g in snames]
            if ids:
                expr = adata[:, ids].X
                if issparse(expr):
                    expr = expr.toarray()
                return np.asarray(expr, dtype=np.float32).mean(axis=1)

        print(f"  ⚠️ No genes found for {label}")
        return np.zeros(adata.n_obs)

    # Score each differentiation level
    scores = {}
    for layer, genes in MARKERS.items():
        scores[layer] = gene_score(genes, layer)
        print(f"  {layer:<10} score: mean={scores[layer].mean():.4f}, "
              f"std={scores[layer].std():.4f} (genes: "
              f"{[g for g in genes if g in adata.var_names or ('feature_name' in adata.var.columns and g in set(adata.var['feature_name'].values)) if 'feature_name' in adata.var.columns]})")

    # Composite axis score: high stem, low squamous
    stem_score = scores["stem"]
    squamous_score = scores["squamous"]

    # Only assign layers to cells with detectable marker expression
    # (non-epithelial cells will have score ≈ 0 for all markers)
    has_markers = (stem_score + squamous_score) > 0.01
    n_with = has_markers.sum()
    n_without = (~has_markers).sum()
    print(f"\n  Cells with detectable marker expression: {n_with} "
          f"({100*n_with/len(has_markers):.1f}%)")
    print(f"  Cells without markers (non-epithelial): {n_without} → excluded")

    # Differentiation index: 0 = stem, 1 = squamous (only for cells with markers)
    diff_index = np.zeros(adata.n_obs)
    diff_index[has_markers] = squamous_score[has_markers] / (
        stem_score[has_markers] + squamous_score[has_markers] + 1e-8
    )

    # Quantile-based 5-layer assignment (only on cells with markers)
    if n_with > 10:
        quantiles = np.percentile(diff_index[has_markers], [10, 30, 50, 70, 90])
        layer_assignment = np.full(adata.n_obs, -1, dtype=int)  # -1 = non-epithelial
        layer_assignment[has_markers] = np.clip(
            np.digitize(diff_index[has_markers], quantiles), 0, 4
        )
    else:
        layer_assignment = np.full(adata.n_obs, -1, dtype=int)

    layer_names = np.array([AXIS_LABELS[i] if i >= 0 else "non-epi"
                            for i in layer_assignment])

    df = pd.DataFrame({
        "diff_index": diff_index,
        "stem_score": stem_score,
        "squamous_score": squamous_score,
        "layer": layer_names,
        "layer_idx": layer_assignment,
    }, index=adata.obs_names)

    print(f"\n  Layer distribution (Method A):")
    for i, name in enumerate(AXIS_LABELS):
        n = (layer_assignment == i).sum()
        print(f"    {name:<10}: {n:>6} cells ({100*n/len(layer_assignment):.1f}%)")

    return df


def stratify_by_annotation(adata: ad.AnnData) -> pd.DataFrame:
    """Method B: Use author_cell_type annotations from CELLxGENE.

    Maps the Human Ocular Surface Atlas cell type hierarchy to the
    5-layer corneal epithelial differentiation axis:

      Layer 0 (Stem):   LPC (limbal progenitor cells)
      Layer 1 (TAC):    Limbus_Epi-C1, Limbus_Epi-C2, Fetal_Epi/LSC/LPC
      Layer 2 (Basal):  Epi-TDC (transitively differentiating, KRT14+)
      Layer 3 (Wing):   Epi-TDC/PMC, Unknown Epithelium (late transitional)
      Layer 4 (Squamous): Epi-PMC (post-mitotic, terminally differentiated)

    Non-epithelial cell types are excluded (layer_idx = -1).

    Reference: Maiti et al. 2022 (Cell Stem Cell); CELLxGENE CAP annotation.
    """
    print("\n--- Method B: author_cell_type annotation stratification ---")

    if "author_cell_type" not in adata.obs.columns:
        print("  ⚠️ author_cell_type not found in obs. Skipping Method B.")
        return pd.DataFrame({"layer": [], "layer_idx": []}, index=adata.obs_names)

    act = adata.obs["author_cell_type"]
    categories = sorted(act.unique())
    print(f"  author_cell_type categories ({len(categories)}):")
    for c in categories:
        print(f"    - {c} ({(act == c).sum()} cells)")

    # Mapping: author_cell_type → differentiation layer index (0-4)
    LAYER_MAPPING = {
        # Layer 0: Quiescent stem cells
        "LPC": 0,
        # Layer 1: Activated limbal progenitors / TAC
        "Limbus_Epi-C1": 1,
        "Limbus_Epi-C2": 1,
        "Fetal_Epi/LSC/LPC": 1,
        # Layer 2: Transitively differentiating (basal/early wing, KRT14+)
        "Epi-TDC": 2,
        # Layer 3: Late transitional (wing → squamous transition)
        "Epi-TDC/PMC": 3,
        "Unknown Epithelium": 3,
        # Layer 4: Post-mitotic terminally differentiated (squamous)
        "Epi-PMC": 4,
    }

    layer_arr = np.full(adata.n_obs, -1, dtype=int)
    for cat, layer_idx in LAYER_MAPPING.items():
        mask = (act == cat).values
        layer_arr[mask] = layer_idx

    n_epi = (layer_arr >= 0).sum()
    n_non = (layer_arr < 0).sum()
    print(f"\n  Mapped: {n_epi} epithelial cells, {n_non} non-epithelial excluded")
    for i, name in enumerate(AXIS_LABELS):
        n = (layer_arr == i).sum()
        print(f"    {name:<10} (layer {i}): {n:>6} cells ({100*n/max(n_epi,1):.1f}% of epithelial)")

    df = pd.DataFrame({
        "author_cell_type": act.values,
        "layer_idx": layer_arr,
        "layer": [AXIS_LABELS[i] if i >= 0 else "non-epi" for i in layer_arr],
    }, index=adata.obs_names)

    return df


# ---------------------------------------------------------------------------
# Part 3: Continuity Score Computation
# ---------------------------------------------------------------------------

def compute_continuity_score(
    embeddings: np.ndarray,
    layer_indices: np.ndarray,
    axis_labels: list[str] = AXIS_LABELS,
) -> dict:
    """Compute Continuity Score in a given embedding space.

    CS = mean cosine相似度 between adjacent layer centroids.

    Returns:
        dict with CS, per-pair similarities, LAA.
    """
    n_layers = len(axis_labels)
    centroids = {}

    for i, label in enumerate(axis_labels):
        mask = layer_indices == i
        if mask.sum() == 0:
            print(f"  ⚠️ Layer {label} has 0 cells — skipping")
            continue
        centroids[i] = embeddings[mask].mean(axis=0)

    # Adjacent-layer cosine similarities
    pair_sims = {}
    for i in range(n_layers - 1):
        if i in centroids and (i + 1) in centroids:
            sim = cosine_similarity(
                centroids[i].reshape(1, -1),
                centroids[i + 1].reshape(1, -1),
            )[0, 0]
            pair_name = f"{axis_labels[i]}↔{axis_labels[i+1]}"
            pair_sims[pair_name] = float(sim)

    if pair_sims:
        cs = float(np.mean(list(pair_sims.values())))
    else:
        cs = 0.0

    # Layer Adjacency Accuracy (LAA):
    # For each pair of adjacent layers, check if their centroid distance
    # is less than the distance between random non-adjacent layer pairs.
    layer_ids = sorted(centroids.keys())
    n_random_pairs = 100
    rng = np.random.default_rng(SEED)

    adjacent_dists = []
    for i in range(len(layer_ids) - 1):
        a, b = layer_ids[i], layer_ids[i + 1]
        d = cos_dist(centroids[a], centroids[b])
        adjacent_dists.append(d)

    if len(layer_ids) >= 3:
        non_adjacent_dists = []
        for _ in range(n_random_pairs):
            a, b = rng.choice(layer_ids, size=2, replace=False)
            if abs(a - b) > 1:
                d = cos_dist(centroids[a], centroids[b])
                non_adjacent_dists.append(d)

        if non_adjacent_dists:
            mean_adj = np.mean(adjacent_dists)
            mean_non_adj = np.mean(non_adjacent_dists)
            laa = float(mean_adj < mean_non_adj)  # binary: 1 if adjacent closer
            laa_ratio = float(mean_adj / mean_non_adj) if mean_non_adj > 0 else 1.0
        else:
            laa = -1  # cannot compute
            laa_ratio = -1.0
    else:
        laa = -1
        laa_ratio = -1.0

    return {
        "continuity_score": cs,
        "pair_similarities": pair_sims,
        "layer_adjacency_accuracy": laa,
        "laa_distance_ratio": laa_ratio,
        "n_cells_per_layer": {
            axis_labels[i]: int((layer_indices == i).sum())
            for i in range(n_layers)
        },
    }


# ---------------------------------------------------------------------------
# Part 4: Visualization
# ---------------------------------------------------------------------------

def plot_cs_results(
    results_a: dict,
    results_b: dict | None,
    embedding_type: str = "PCA",
) -> None:
    """Generate publication-quality CS visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel A: Pair similarities heatmap-style bar chart
    ax = axes[0]
    if results_a is not None:
        pairs_a = results_a["pair_similarities"]
        labels = list(pairs_a.keys())
        values = list(pairs_a.values())

        bars = ax.barh(labels, values, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Cosine Similarity (adjacent layer centroids)")
        ax.set_title(f"Continuity Score ({embedding_type}) — Method A: Marker gradient",
                     fontsize=11, fontweight="bold")
        ax.axvline(results_a["continuity_score"], color="red", linestyle="--",
                   label=f'CS = {results_a["continuity_score"]:.4f}')
        ax.legend()
        ax.set_xlim(0, 1)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Method A: No cells with\ndetectable markers", ha="center",
                va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(f"Method A: Marker gradient (no data)", fontsize=11)

    # Panel B: Layer centroid positions in first 2 PCs (if available)
    ax = axes[1]
    if results_b is not None:
        pairs_b = results_b["pair_similarities"]
        labels_b = list(pairs_b.keys())
        values_b = list(pairs_b.values())
        ax.barh(labels_b, values_b, color="#55A868", alpha=0.85,
                edgecolor="black", linewidth=0.5, label="Method B: Annotation")
        ax.axvline(results_b["continuity_score"], color="darkred", linestyle=":",
                   label=f'CS_B = {results_b["continuity_score"]:.4f}')
        ax.legend()
    ax.set_xlabel("Cosine Similarity")
    ax.set_title(f"Method B: cell_type annotation", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)

    fig.savefig(FIG_DIR / f"cs_before_{embedding_type.lower()}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"cs_before_{embedding_type.lower()}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  📊 Figure saved: figures/cs_before_{embedding_type.lower()}.png/.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute CS_before baseline")
    parser.add_argument("--test_set", action="store_true",
                        help="Use test set (20k) instead of train set (100k)")
    parser.add_argument("--donor_split", action="store_true",
                        help="Use donor-disjoint split files")
    parser.add_argument("--n_pcs", type=int, default=50,
                        help="Number of PCs for embedding (default: 50)")
    args = parser.parse_args()

    # Load data
    if args.donor_split:
        data_file = DATA_DIR / ("corneal_20k_donor_split.h5ad" if args.test_set
                                else "corneal_100k_donor_split.h5ad")
    else:
        data_file = DATA_DIR / ("corneal_20k_test.h5ad" if args.test_set
                                else "corneal_100k_subsample.h5ad")
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print(f"   Platform: {platform.system()} {platform.machine()}")
        print(f"   DATA_DIR: {DATA_DIR}")
        sys.exit(1)

    set_name = "test (20k)" if args.test_set else "train (100k)"
    print(f"\n📁 Loading {set_name}: {data_file.name} ...")
    print(f"   Platform: {platform.system()} {platform.machine()}")
    adata = sc.read_h5ad(str(data_file))
    print(f"   Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"   Sparse: {issparse(adata.X)}")
    print(f"   .raw:   {adata.raw is not None}")
    print(f"   layers: {list(adata.layers.keys()) if adata.layers else '(none)'}")

    # Build stable gene symbol → var index mapping BEFORE any transformation
    _symbol_to_var_idx = {}
    if "feature_name" in adata.var.columns:
        _symbol_to_var_idx = dict(zip(
            adata.var["feature_name"].values,
            range(adata.n_vars)  # positional indices, stable across copies
        ))
        print(f"   Gene symbol mapping: {len(_symbol_to_var_idx)} symbols via feature_name")
    else:
        # Donor-split files may lose var columns during sc.concat + write.
        # Recover feature_name from the original source files.
        src_file = DATA_DIR / "corneal_100k_subsample.h5ad"
        if src_file.exists():
            print(f"   ⚠️ feature_name missing, recovering from {src_file.name} var ...")
            src = sc.read_h5ad(str(src_file), backed="r")
            if "feature_name" in src.var.columns and src.n_vars == adata.n_vars:
                _symbol_to_var_idx = dict(zip(
                    src.var["feature_name"].values,
                    range(src.n_vars)
                ))
                print(f"   Gene symbol mapping: {len(_symbol_to_var_idx)} symbols recovered")
            src.file.close()
        if not _symbol_to_var_idx:
            print(f"   ⚠️ No gene symbol mapping available — Method A will be skipped")

    # Part 1: Inspect (before any modification)
    inspect_data(adata)

    # ---- Normalize ONCE for all downstream ----
    print("\n--- Normalizing (once) ---")
    if adata.raw is not None:
        # Use raw counts for normalization if available
        adata_norm = adata.raw.to_adata()
        adata_norm.obs = adata.obs.copy()
        # Preserve var columns (feature_name etc.) from original adata
        for col in adata.var.columns:
            if col not in adata_norm.var.columns:
                adata_norm.var[col] = adata.var[col].values
    else:
        adata_norm = adata.copy()

    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    print("   Normalized + log1p done.")

    # Part 2A: Marker stratification (on normalized data)
    df_markers = stratify_by_markers(adata_norm, symbol_to_idx=_symbol_to_var_idx)

    # Part 2B: Annotation stratification (doesn't need expression)
    df_annot = stratify_by_annotation(adata_norm)

    # Part 3: PCA embedding
    print(f"\n--- Computing PCA embedding ({args.n_pcs} PCs) ---")
    sc.pp.highly_variable_genes(adata_norm, n_top_genes=N_TOP_GENES)
    adata_hvg = adata_norm[:, adata_norm.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=args.n_pcs, random_state=SEED)
    pca_emb = adata_hvg.obsm["X_pca"]
    print(f"   PCA embedding: {pca_emb.shape}")

    # Compute CS — Method A (markers), only cells with detected markers
    epi_mask_a = df_markers["layer_idx"].values >= 0
    if epi_mask_a.sum() > 0:
        print(f"\n=== CS_before (Method A: Marker gradient, PCA) === "
              f"[{epi_mask_a.sum()} cells with markers]")
        results_a = compute_continuity_score(
            pca_emb[epi_mask_a],
            df_markers.loc[epi_mask_a, "layer_idx"].values,
        )
        print(f"   CS = {results_a['continuity_score']:.4f}")
        print(f"   Pair similarities: {results_a['pair_similarities']}")
        print(f"   LAA = {results_a['layer_adjacency_accuracy']}")
        print(f"   LAA ratio = {results_a['laa_distance_ratio']:.4f}")
    else:
        print("\n⚠️ No cells with detectable markers (Method A) — skipping")
        results_a = None

    # Compute CS — Method B (annotations), only epithelial cells
    epi_mask = df_annot["layer_idx"].values >= 0
    if epi_mask.sum() > 0:
        print(f"\n=== CS_before (Method B: Annotation, PCA) === "
              f"[{epi_mask.sum()} epithelial cells]")
        results_b = compute_continuity_score(
            pca_emb[epi_mask],
            df_annot.loc[epi_mask, "layer_idx"].values,
        )
        print(f"   CS = {results_b['continuity_score']:.4f}")
        print(f"   Pair similarities: {results_b['pair_similarities']}")
    else:
        print("\n⚠️ No epithelial cells matched in Method B — skipping")
        results_b = None

    # Part 4: Visualization
    plot_cs_results(results_a, results_b, embedding_type="PCA")

    # Save results
    output = {
        "dataset": str(data_file.name),
        "embedding": "PCA",
        "n_pcs": args.n_pcs,
        "method_A_markers": results_a,
        "method_B_annotation": results_b,
        "seed": SEED,
    }
    out_path = DATA_DIR / "cs_before_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  💾 Results saved: {out_path}")

    # Summary
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Dataset:     {set_name}")
    print(f"  Embedding:   PCA ({args.n_pcs} PCs)")
    print(f"  Donor split: {args.donor_split}")
    if results_a:
        print(f"  CS (Method A, markers):    {results_a['continuity_score']:.4f}")
    else:
        print(f"  CS (Method A, markers):    N/A (no markers detected)")
    if results_b:
        print(f"  CS (Method B, annotation): {results_b['continuity_score']:.4f}")
    if results_a:
        print(f"  LAA (Method A):            {results_a['layer_adjacency_accuracy']}")
    print(f"\n  These values are the GPR denominator.")
    print(f"  After fine-tuning: GPR = CS_after / CS_before")
    print(f"  GPR < 1.0 = continuity blind spot (fine-tune degrades)")
    print(f"  GPR > 1.0 = CAFT preserves/enhances continuity")
    print("=" * 64)


if __name__ == "__main__":
    main()
