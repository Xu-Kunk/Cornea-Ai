#!/usr/bin/env python
"""Per-cell trajectory-continuity analysis across the 7 Paper 1 model arms.

Tests the earlier framing that "CAFT preserves the differentiation trajectory
while naive fine-tuning destroys it", by measuring each arm's inferred
trajectory against the frozen reference ordering. **The measurement does not
support that framing.** Measured on the external test set: frozen rho = 0.850
(95% CI 0.842-0.858), CE-naive rho = 0.323-0.348, CAFT-intended rho =
0.372-0.428. CAFT-intended sits only marginally above CE-naive with overlapping
per-seed intervals, and both are far below frozen. The manuscript therefore
states "CAFT is intermediate" and makes no trajectory-preservation claim; do not
restate the original framing in any downstream text or figure caption.

For each of the 7 arms (frozen Geneformer-V2-104M, naive FT lambda=0 and CAFT
lambda=0.5 x seeds {42,123,456}) we run UNSUPERVISED trajectory inference on
the external-test per-cell embedding (emb_ext, 14,775 cells) and compare the
inferred pseudotime against the *frozen-defined* reference pseudotime.

Reference pseudotime — the layer-centroid PCA projection used by
``corneal_fm.evaluation.marker_gradient_correlation`` (the rho metric locked in
the Paper 1 artifacts): the first singular vector of the 4 per-layer centroids
(LSC / TAC / basal / differentiated), oriented so the projection increases
LSC -> differentiated; reference pseudotime = emb @ axis (marker-free; labels
used only to place the layer centroids).

IMPORTANT DESIGN DECISION (honest reporting): the reference is computed ONCE on
the frozen (pretrained) embedding — the representation that actually defines the
differentiation gradient in Paper 1 (rho vs ordinal = 0.806, sanity-locked).
Computing the same construction per arm is *self-referential* and
non-discriminative: in the naive/CAFT arms the arm-specific axis is itself noise
(locked rho ~ 0.05-0.23; its minimum end even lands on a basal cell), so
"DPT vs same-arm reference" degenerates into "DPT vs each arm's own dominant
variance direction" and is trivially high everywhere.  Comparing every arm's
inferred trajectory against the fixed frozen ordering is the meaningful test of
"does this arm preserve the pretrained differentiation trajectory".  The
per-arm reference rho is still reported as a secondary self-consistency metric.

Inferred pseudotime — primary method: scanpy DPT (diffusion pseudotime) on a
PCA-reduced kNN graph, root = the frozen-reference minimum end (a single LSC
cell, identical across arms).  If scanpy is unavailable or DPT fails, a
numpy/scipy fallback is used: kNN graph + Gaussian diffusion kernel ->
Dijkstra shortest-path pseudotime from the same root; the CSV records the
method that produced each value.

Metrics per arm:
  * rho_vs_frozen_ref  Spearman(inferred, frozen reference) [+ bootstrap 95% CI,
                       n_boot=1000 seed 42, matching Paper 1 conventions]
  * rho_vs_arm_ref     secondary: Spearman(inferred, same-arm reference)
  * strat_rho          stratified consistency: cells binned into deciles of the
                       frozen reference; Spearman(median inferred per bin,
                       bin index) — is the frozen trajectory ordering still
                       monotone in the arm's inferred trajectory?
  * rho_reference_vs_ordinal  sanity: Spearman(arm reference, layer ordinal)
                       must reproduce the locked meta.json rho (fail-warn on
                       |diff| > 1e-4).

Reads only  artifacts/paper1_tier_b_eval/percell_*.npz (read-only).
Writes only:
  figures/paper1/figure3_trajectory_continuity.{pdf,png}
  tables/paper1/table_trajectory_continuity.csv

Usage:
    <venv>/bin/python3 scripts/analyze_trajectory_continuity.py
    ... --preprocess l2pca      # per-cell L2 normalise before PCA
    ... --root lsc_medoid       # root = medoid of LSC cells (frozen space)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the locked rcParams / Okabe-Ito palette / save helpers of the results
# figure pipeline verbatim (style unification requirement).
from generate_paper1_results_figures import (  # noqa: E402,F401
    OI_BLUE,
    OI_BLUEGREEN,
    OI_GRAY,
    OI_ORANGE,
    OI_PINK,
    OI_SKYBLUE,
    OI_VERMILLION,
    figure_metadata,
    panel_label,
    save_figure,
    setup_style,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants (Paper 1 locked contract)
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
FIG_DIR = PROJECT_ROOT / "figures" / "paper1"
TABLE_DIR = PROJECT_ROOT / "tables" / "paper1"

LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]
SEEDS = [42, 123, 456]
N_EXT = 14775

# Seed -> shade within the group hue family (v6 unified mapping): naive =
# blue family, CAFT = orange family (v8fix lock), frozen = dark gray —
# identical to figure2_layerwise_rho conventions.
_NAIVE_SEED_COLORS = {42: "#0072B2", 123: "#56B4E9", 456: "#003C66"}
_CAFT_SEED_COLORS = {42: "#E69F00", 123: "#F2C14E", 456: "#A06A00"}
_FROZEN_RHO_COLOR = "#333333"

ARMS: list[dict[str, Any]] = (
    [{"arm": "frozen", "file": "percell_frozen_gf.npz", "lam": None, "seed": None}]
    + [
        {
            "arm": "naive" if lam == 0.0 else "caft",
            "file": f"percell_{'naive' if lam == 0.0 else 'caft'}_s{seed}.npz",
            "lam": lam,
            "seed": seed,
        }
        for lam in (0.0, 0.5)
        for seed in SEEDS
    ]
)

SANITY_TOL = 1e-4


# ---------------------------------------------------------------------------
# Reference pseudotime — EXACT re-construction of marker_gradient_correlation
# (corneal_fm/evaluation.py:461).  The public function only returns the
# Spearman rho, so the projection axis is rebuilt here with the same steps:
#   1) keep cells whose label is in LAYER_ORDER_4
#   2) per-layer centroids -> center -> SVD -> first singular vector = axis
#   3) orient axis so projection increases from first to last layer
#   4) reference pseudotime = emb @ axis
# ---------------------------------------------------------------------------

def reference_pseudotime(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (pseudotime, axis, rho_vs_ordinal) mirroring marker_gradient_correlation.

    ``pseudotime`` is defined for every input cell; cells outside
    LAYER_ORDER_4 receive NaN (none exist in the external set, kept for parity).
    ``rho_vs_ordinal`` is Spearman(pseudotime[valid], ordinal[valid]) — the
    locked Paper 1 marker-gradient rho, used as a sanity gate.
    """
    labels = np.asarray(labels)
    emb = np.asarray(embeddings, dtype=float)
    valid = np.array([lbl in LAYER_ORDER_4 for lbl in labels])
    lab = labels[valid]
    e = emb[valid]

    order_map = {c: i for i, c in enumerate(LAYER_ORDER_4)}
    ordinals = np.array([order_map[x] for x in lab], dtype=float)
    present = [c for c in LAYER_ORDER_4 if (ordinals == order_map[c]).any()]

    pseudotime = np.full(len(emb), np.nan)
    if len(present) < 2:
        return pseudotime, np.full(emb.shape[1], np.nan), float("nan")

    centroids = np.vstack([e[ordinals == order_map[c]].mean(axis=0) for c in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[0]  # principal axis (unit vector)
    # SVD sign is arbitrary; orient so pseudotime increases LSC -> differentiated.
    first_centroid = e[ordinals == order_map[present[0]]].mean(axis=0)
    last_centroid = e[ordinals == order_map[present[-1]]].mean(axis=0)
    if (last_centroid - first_centroid) @ axis < 0:
        axis = -axis

    proj = e @ axis
    pseudotime[valid] = proj
    rho, _ = spearmanr(proj, ordinals)
    return pseudotime, axis, float(rho)


def select_root(
    emb: np.ndarray, labels: np.ndarray, ref_pt: np.ndarray, mode: str
) -> int:
    """Root-cell index selection.

    frozen_ref_min: argmin of the frozen reference pseudotime (an LSC-end cell).
    lsc_medoid:     medoid (nearest-to-centroid) of LSC-labeled cells — a
                    robust LSC anchor independent of the (possibly noisy)
                    reference axis.
    arm_ref_min:    argmin of the (per-arm) reference pseudotime.
    """
    if mode == "arm_ref_min":
        return int(np.argmin(np.where(np.isfinite(ref_pt), ref_pt, np.inf)))
    labels = np.asarray(labels)
    if mode == "lsc_medoid":
        emb_f = np.asarray(emb, dtype=float)
        lsc = emb_f[labels == "LSC"]
        if len(lsc) == 0:
            raise ValueError("no LSC cells for lsc_medoid root")
        centre = lsc.mean(axis=0)
        d = np.sum((lsc - centre) ** 2, axis=1)
        return int(np.where(labels == "LSC")[0][int(np.argmin(d))])
    # frozen_ref_min (default)
    return int(np.argmin(np.where(np.isfinite(ref_pt), ref_pt, np.inf)))


# ---------------------------------------------------------------------------
# Trajectory inference — scanpy DPT (primary) + numpy/scipy fallback
# ---------------------------------------------------------------------------

def _pca_embedding(emb: np.ndarray, n_comps: int) -> np.ndarray:
    """PCA reduction of the embedding (project convention: N_PC=30)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(n_comps, emb.shape[1]), random_state=0)
    return pca.fit_transform(emb)


def _fallback_diffusion_pseudotime(
    emb: np.ndarray, root_idx: int, n_neighbors: int
) -> np.ndarray:
    """kNN graph + Gaussian diffusion kernel -> Dijkstra shortest-path time.

    numpy/scipy-only fallback when scanpy / DPT is unavailable:
      1. kNN graph (k = n_neighbors, Euclidean)
      2. Gaussian affinity A_ij = exp(-d_ij^2 / sigma^2), sigma = mean distance
         to the k-th neighbour (graph-scale, as in scanpy's neighbors); made
         symmetric
      3. pseudotime = Dijkstra distance from the root in the
         weight = -log(A_ij) (>= 0) graph  -> monotone from root, graph-aware.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(emb)
    dist, idx = nn.kneighbors(emb)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self
    sigma = float(np.mean(dist[:, -1]))  # mean k-th-neighbour distance
    aff = np.exp(-(dist ** 2) / (sigma ** 2))
    n = emb.shape[0]
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = idx.ravel()
    vals = aff.ravel()
    g = csr_matrix((vals, (rows, cols)), shape=(n, n))
    g = (g + g.T).tocsr()  # symmetrize
    g.data[g.data <= 1e-12] = 1e-12
    w = csr_matrix((-np.log(g.data), g.indices, g.indptr), shape=g.shape)
    return dijkstra(w, directed=False, indices=root_idx)


def run_trajectory_inference(
    emb_ext: np.ndarray, root_idx: int, preprocess: str, n_pcs: int,
    n_neighbors: int, n_dcs: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Infer an unsupervised pseudotime for the arm embedding.

    Returns (pseudotime, method, info).  ``preprocess`` in
    {"pca", "l2pca", "raw"}: pca = PCA(n_pcs) on the raw embedding;
    l2pca = per-cell L2 normalisation first; raw = neighbours on the raw
    embedding with no reduction.
    """
    emb = np.asarray(emb_ext, dtype=np.float32)
    if preprocess == "l2pca":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)

    info: dict[str, Any] = {"preprocess": preprocess, "n_cells": emb.shape[0]}

    # --- Primary: scanpy DPT -------------------------------------------
    try:
        import scanpy as sc

        adata = sc.AnnData(X=emb)
        if preprocess in ("pca", "l2pca"):
            sc.pp.pca(adata, n_comps=n_pcs, random_state=0)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs,
                            random_state=0)
        else:  # raw
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=0)
        adata.uns["iroot"] = int(root_idx)
        sc.tl.dpt(adata, n_dcs=n_dcs)
        pt = np.asarray(adata.obs["dpt_pseudotime"].to_numpy(), dtype=float)
        info["method"] = "scanpy_dpt"
        info["n_nan_dpt"] = int(np.isnan(pt).sum())
        if np.isnan(pt).sum() <= 0.02 * pt.size:
            return pt, "scanpy_dpt", info
        print(f"    WARN: DPT produced {info['n_nan_dpt']} NaN cells; "
              f"falling back to diffusion-kernel pseudotime")
    except Exception as exc:  # noqa: BLE001
        print(f"    WARN: scanpy DPT unavailable/failed ({exc}); "
              f"using diffusion-kernel fallback")

    # --- Fallback: kNN + diffusion kernel + shortest path ----------------
    emb_use = emb if preprocess == "raw" else _pca_embedding(emb, n_pcs)
    pt = _fallback_diffusion_pseudotime(emb_use, int(root_idx), n_neighbors)
    info["method"] = "knn_diffusion_fallback"
    info["n_nan_dpt"] = int(np.isnan(pt).sum())
    return pt, "knn_diffusion_fallback", info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int,
                          seed: int) -> tuple[float, float, float]:
    """Spearman rho + percentile bootstrap 95% CI (cells resampled with
    replacement), matching the Paper 1 bootstrap convention (n_boot=1000,
    seed 42).  Returns (rho, ci_lower, ci_upper)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rng = np.random.default_rng(seed)
    rho, _ = spearmanr(x, y)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        r, _ = spearmanr(x[idx], y[idx])
        boot[i] = r
    return float(rho), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stratified_consistency(ref_pt: np.ndarray, inf_pt: np.ndarray, q: int) -> float:
    """Spearman(median inferred pseudotime per reference decile, decile index).

    Tests whether the inferred trajectory still orders the reference-pseudotime
    quantiles monotonically.  NaN-safe; returns NaN if <2 deciles have cells."""
    ok = np.isfinite(ref_pt) & np.isfinite(inf_pt)
    r, i = ref_pt[ok], inf_pt[ok]
    if len(r) < 10:
        return float("nan")
    try:
        bins = pd.qcut(r, q=q, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    df = pd.DataFrame({"b": bins, "i": i}).dropna()
    med = df.groupby("b")["i"].median()
    if len(med) < 2:
        return float("nan")
    rho, _ = spearmanr(np.arange(len(med)), med.to_numpy())
    return float(rho)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def render_figure(results: pd.DataFrame, strat_lines: dict[str, np.ndarray | None],
                  q: int, out_dir: Path) -> list[Path]:
    """figure3_trajectory_continuity: A) inferred-vs-frozen-reference rho per arm
    with bootstrap 95% CI; B) stratified consistency (median inferred
    pseudotime per frozen-reference decile, min-max scaled per arm)."""
    fig = plt.figure(figsize=(6.5, 3.25))

    def _color(row: pd.Series) -> str:
        if row["arm"] == "frozen":
            return _FROZEN_RHO_COLOR
        if row["arm"] == "naive":
            return _NAIVE_SEED_COLORS.get(int(row["seed"]), OI_BLUEGREEN)
        return _CAFT_SEED_COLORS.get(int(row["seed"]), OI_PINK)

    # ---- Panel A: per-arm rho + CI (vs frozen reference) ---------------
    ax = fig.add_subplot(1, 2, 1)
    for pos, row in results.iterrows():
        color = _color(row)
        ax.errorbar(
            pos, row["rho_vs_frozen_ref"],
            yerr=[[row["rho_vs_frozen_ref"] - row["ci_lower"]],
                  [row["ci_upper"] - row["rho_vs_frozen_ref"]]],
            fmt="o", ms=6, color=color, ecolor=color, elinewidth=1.4,
            capsize=3, zorder=3,
        )
    ax.axhline(0.0, color=OI_GRAY, ls=":", lw=0.7, zorder=1)
    ax.set_xticks(list(results.index))
    ax.set_xticklabels(
        [r["arm"] if r["arm"] == "frozen" else f"{'CAFT' if r['arm'] == 'caft' else 'naive'}\ns{int(r['seed'])}"
         for _, r in results.iterrows()],
        fontsize=6.5,
    )
    ax.set_ylabel("Spearman \u03c1 (DPT vs frozen reference pseudotime)")
    ax.set_title("Trajectory continuity vs frozen reference", fontsize=9,
                 fontweight="bold", loc="left")
    lo = min(0.0, results["ci_lower"].min())
    hi = results["ci_upper"].max()
    ax.set_ylim(lo - 0.15, max(hi + 0.15, 1.05))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    panel_label(fig, ax, "A")

    # ---- Panel B: stratified consistency --------------------------------
    # v8fix (obs-1): per-seed curves merged into family envelope + mean line;
    # the 3-entry legend is parked in the upper-left corner with a white box
    # (was a 7-entry all-fine-tuned legend that crowded the panel).
    ax = fig.add_subplot(1, 2, 2)
    fam_curves: dict[str, list[np.ndarray]] = {"naive": [], "caft": []}
    frozen_med: np.ndarray | None = None
    for _, row in results.iterrows():
        color = _color(row)
        med = strat_lines.get(row["key"])
        if med is None or len(med) == 0:
            continue
        y = med - med.min()
        span = med.max() - med.min()
        y = y / span if span > 0 else np.zeros_like(y)
        if row["arm"] == "frozen":
            frozen_med = y
            ax.plot(np.arange(len(y)), y, marker="o", ms=3.5, lw=1.8,
                    color=color, zorder=3)
        else:
            fam = "naive" if row["arm"] == "naive" else "caft"
            fam_curves[fam].append(y)
            ax.plot(np.arange(len(y)), y, marker="o", ms=2.5, lw=0.8,
                    color=color, alpha=0.35, zorder=2)
    for fam, color in (("naive", OI_BLUE), ("caft", OI_ORANGE)):
        curves = fam_curves[fam]
        if not curves:
            continue
        mat = np.vstack(curves)
        mean_y = mat.mean(axis=0)
        lo = mat.min(axis=0)
        hi = mat.max(axis=0)
        xs = np.arange(len(mean_y))
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0, zorder=1)
        ax.plot(xs, mean_y, marker="o", ms=3.5, lw=1.6, color=color, zorder=3)
    ax.set_xlabel("Frozen reference pseudotime decile (0 = LSC end)")
    ax.set_ylabel("Median inferred pseudotime (scaled)")
    ax.set_title("Stratified consistency", fontsize=9, fontweight="bold",
                 loc="left")
    ax.set_xticks(range(0, q, max(1, q // 5)))
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    handles = [
        plt.Line2D([0], [0], color=_FROZEN_RHO_COLOR, lw=1.8, label="frozen GF"),
    ]
    if fam_curves["naive"]:
        handles.append(plt.Line2D([0], [0], color=OI_BLUE, lw=1.6,
                                  label="naive (mean \u00b1 seed range)"))
    if fam_curves["caft"]:
        handles.append(plt.Line2D([0], [0], color=OI_ORANGE, lw=1.6,
                                  label="CAFT (mean \u00b1 seed range)"))
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", frameon=True,
              framealpha=0.9, borderpad=0.4, handlelength=1.4)
    panel_label(fig, ax, "B")
    fig.text(
        0.5, 0.005,
        "Error bars: bootstrap 95% CI (n = 14,775 external-test epithelial "
        "cells); DPT vs frozen layer-centroid PCA pseudotime reference.",
        ha="center", va="bottom", fontsize=6.3, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))

    return save_figure(
        fig, "figure3_trajectory_continuity", out_dir,
        figure_metadata("Figure 8 - Trajectory continuity across Paper 1 arms "
                        "(DPT vs frozen layer-centroid PCA reference)"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    parser.add_argument("--table-dir", default=str(TABLE_DIR))
    parser.add_argument("--preprocess", choices=["pca", "l2pca", "raw"],
                        default="pca",
                        help="embedding preprocessing for trajectory inference "
                             "(default pca = PCA-30 on raw embedding)")
    parser.add_argument("--root", choices=["frozen_ref_min", "lsc_medoid",
                                           "arm_ref_min"],
                        default="frozen_ref_min",
                        help="root-cell convention (default: frozen-reference "
                             "minimum end, a single LSC cell for all arms)")
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--n-dcs", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q", type=int, default=10, help="reference deciles")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    table_dir = Path(args.table_dir)
    setup_style()
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # ---- frozen (reference) arm loaded first -----------------------------
    frozen_cfg = next(a for a in ARMS if a["arm"] == "frozen")
    with np.load(data_dir / frozen_cfg["file"]) as d:
        frozen_emb = d["emb_ext"]
        frozen_labels = np.asarray(d["y_true_ext"], dtype=str)
    assert frozen_emb.shape == (N_EXT, 768), frozen_emb.shape

    frozen_ref, frozen_axis, frozen_rho_ord = reference_pseudotime(
        frozen_emb, frozen_labels
    )
    frozen_meta = data_dir / (frozen_cfg["file"] + ".meta.json")
    if frozen_meta.is_file():
        locked_frozen = float(
            json.load(open(frozen_meta))["sanity_gate_external"]["rho"]
        )
        diff = abs(frozen_rho_ord - locked_frozen)
        print(f"  SANITY frozen reference: rho={frozen_rho_ord:.6f} "
              f"locked={locked_frozen:.6f} diff={diff:.2e} -> "
              f"{'PASS' if diff <= SANITY_TOL else 'FAIL'}")

    root_idx = select_root(frozen_emb, frozen_labels, frozen_ref, args.root)
    root_label = str(frozen_labels[root_idx])
    print(f"  Root cell: index {root_idx} label={root_label} "
          f"({args.root} convention)")

    rows: list[dict[str, Any]] = []
    strat_lines: dict[str, np.ndarray | None] = {}

    for arm in ARMS:
        npz_path = data_dir / arm["file"]
        with np.load(npz_path) as d:
            emb_ext = d["emb_ext"]
            y_true_ext = np.asarray(d["y_true_ext"], dtype=str)
        assert emb_ext.shape == (N_EXT, 768), emb_ext.shape

        # --- per-arm reference (secondary self-consistency) ---------------
        arm_ref, _, rho_vs_ord = reference_pseudotime(emb_ext, y_true_ext)
        meta_path = npz_path.with_suffix(npz_path.suffix + ".meta.json")
        locked = None
        if meta_path.is_file():
            locked = float(json.load(open(meta_path))["sanity_gate_external"]["rho"])
        if locked is not None and np.isfinite(rho_vs_ord):
            diff = abs(rho_vs_ord - locked)
            print(f"  SANITY {arm['arm']} s{arm['seed']}: ref-rho={rho_vs_ord:.6f} "
                  f"locked={locked:.6f} diff={diff:.2e} -> "
                  f"{'PASS' if diff <= SANITY_TOL else 'FAIL'}")

        # --- unsupervised trajectory inference (fixed root) ---------------
        inf_pt, method, info = run_trajectory_inference(
            emb_ext, root_idx, args.preprocess, args.n_pcs,
            args.n_neighbors, args.n_dcs,
        )

        # --- metrics ------------------------------------------------------
        rho, ci_lo, ci_hi = bootstrap_spearman_ci(inf_pt, frozen_ref,
                                                  args.n_boot, args.seed)
        rho_arm, _, _ = bootstrap_spearman_ci(inf_pt, arm_ref,
                                              args.n_boot, args.seed)
        strat = stratified_consistency(frozen_ref, inf_pt, args.q)

        ok = np.isfinite(frozen_ref) & np.isfinite(inf_pt)
        med_line: np.ndarray | None = None
        if ok.sum() >= 10:
            bins = pd.qcut(frozen_ref[ok], q=args.q, labels=False,
                           duplicates="drop")
            med_line = pd.Series(inf_pt[ok]).groupby(bins).median().to_numpy()
        key = f"{arm['arm']}_s{arm['seed']}"
        strat_lines[key] = med_line

        rows.append({
            "arm": arm["arm"],
            "lambda": arm["lam"],
            "seed": arm["seed"],
            "key": key,
            "n_cells": N_EXT,
            "root_cell_label": root_label,
            "root_convention": args.root,
            "method": method,
            "preprocess": args.preprocess,
            "n_pcs": args.n_pcs if args.preprocess != "raw" else np.nan,
            "n_neighbors": args.n_neighbors,
            "n_dcs": args.n_dcs if method == "scanpy_dpt" else np.nan,
            "n_inferred_valid": int(np.isfinite(inf_pt).sum()),
            "rho_vs_frozen_ref": rho,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "strat_rho_vs_frozen": strat,
            "rho_vs_arm_ref": rho_arm,
            "rho_reference_vs_ordinal": rho_vs_ord,
            "locked_rho": locked,
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        })
        seed_txt = "-" if arm["seed"] is None else f"s{arm['seed']}"
        print(f"  {arm['arm']:6s} {seed_txt:<4s} vs_frozen={rho: .4f} "
              f"[{ci_lo:.4f},{ci_hi:.4f}] strat={strat: .4f} "
              f"vs_arm={rho_arm: .4f}")

    results = pd.DataFrame(rows)
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "table_trajectory_continuity.csv"
    results.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved {csv_path}")

    fig_dir.mkdir(parents=True, exist_ok=True)
    outs = render_figure(results, strat_lines, args.q, fig_dir)
    print(f"  Saved {[str(p) for p in outs]}")

    # --- hypothesis check -------------------------------------------------
    frozen = results.loc[results["arm"] == "frozen", "rho_vs_frozen_ref"]
    naive = results.loc[results["arm"] == "naive", "rho_vs_frozen_ref"]
    caft = results.loc[results["arm"] == "caft", "rho_vs_frozen_ref"]
    if len(frozen) and len(naive) and len(caft):
        f, n, c = float(frozen.iloc[0]), float(naive.mean()), float(caft.mean())
        print(f"\n  Hypothesis: frozen({f:.4f}) > naive({n:.4f}) "
              f"{'HOLDS' if f > n else 'REVERSED'}; "
              f"CAFT({c:.4f}) between frozen and naive "
              f"{'HOLDS' if min(f, n) <= c <= max(f, n) else 'REVERSED'}")
        if not (f > n):
            print("    !! frozen <= naive: re-run with --preprocess l2pca "
                  "and/or --root lsc_medoid, then re-check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
