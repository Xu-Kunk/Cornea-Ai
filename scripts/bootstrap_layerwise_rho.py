#!/usr/bin/env python
"""Bootstrap 95% CI for the Figure 7 layerwise differentiation-gradient rho.

Context
-------
Figure 7 plots the marker-gradient Spearman rho at every transformer layer
(1-12) for the 7 Paper-1 models (frozen Geneformer-V2-104M + 6 CAFT
fine-tunes: lambda in {0.0, 0.5} x seeds {42, 123, 456}). The point estimates
live in ``artifacts/paper1_layerwise_rho/summary.json`` (produced by
``scripts/measure_layerwise_rho.py``, sanity-gated against the Tier-B eval
JSONs to 1e-6).

Situation B (current): the layerwise pipeline does NOT persist per-cell
per-layer embeddings (no ``--cache`` option; only 13 rho values per model are
saved). The only per-cell cache available is the FINAL-layer CLS embedding
``emb_ext`` in ``artifacts/paper1_tier_b_eval/percell_*.npz`` (14,775 external
epithelial cells per model). We verified bit-exact reproduction of the L12
rho from ``emb_ext`` (diff = 0.0 for all 7 models), so a cell-level bootstrap
for layer 12 is exact and requires NO GPU.

What this script does
---------------------
1. Loads the L12 per-cell CLS embeddings + harmonized labels per model.
2. SANITY GATE (fail-closed): the full-data rho recomputed from the cached
   embeddings must equal the ``summary.json`` L12 rho within 1e-6 (mirrors
   ``measure_layerwise_rho.py``); otherwise abort with exit code 2.
3. Cell-level bootstrap: n_boot=1000 resamples with replacement (same RNG
   stream, ``np.random.default_rng(seed)``, for every model -> resamples are
   cell-paired across arms), recomputing the marker-gradient rho end-to-end
   (centroids -> PCA pseudotime axis -> Spearman) on each resample, exactly as
   ``corneal_fm.evaluation.marker_gradient_correlation`` does. The centroid
   step is vectorized (einsum over a precomputed per-cell one-hot mask) and
   bit-identical to the reference metric on full data.
4. Percentile-method 95% CI (2.5 / 97.5 percentiles of the bootstrap
   distribution).

Outputs (figures/paper1/)
    figure2_layerwise_rho_bootstrap.csv  long-format CI table (per model,
        per bootstrapped layer): rho, ci_lower, ci_upper, ci_width, boot_se,
        n_boot, n_valid, boot_seed, ci_method, n_cells
    figure2_layerwise_rho.csv            updated with ci_lower/ci_upper
        columns (NaN for layers without bootstrap data)
    figure2_layerwise_rho.{pdf,png}      redrawn with 95% CI error bars at
        the bootstrapped layer(s); identical style/colors/annotations to the
        original figure (reuses rcParams/colors/save_figure from
        generate_paper1_results_figures.py)

Per-layer extension (situation A)
---------------------------------
If per-cell per-layer embeddings become available (e.g. by adding a --cache
option to measure_layerwise_rho.py that dumps ``(13, n_cells, 768)`` CLS
tensors), pass ``--layer-embeddings-dir DIR`` where DIR contains
``<model_key>.npz`` files with keys ``embeddings`` (n_layers, n_cells, dim)
and ``labels`` (n_cells,). The script then bootstraps any layer subset
(``--layers 1,2,...,12``) CPU-only. Without that directory only layer 12 can
be bootstrapped (from the cached final-layer embeddings) and requesting any
other layer fails with a clear message.

Methodology notes
-----------------
* Bootstrap unit = cell (14,775 external epithelial cells, NON_LAYER excluded
  inside the metric).
* The PCA pseudotime axis is recomputed inside every resample (centroid
  geometry is itself part of the estimand), so the CI reflects sampling
  variability of the whole marker-gradient procedure.
* Same RNG stream per model -> the resampled cell indices are identical across
  arms (paired bootstrap), which sharpens cross-arm comparisons.

Usage:
    /home/dq/venv-ds/bin/python3 scripts/bootstrap_layerwise_rho.py \
        [--n-boot 1000] [--seed 42] [--layers 12] [--workers 7]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --- environment (headless-safe, no GPU) ------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore")

# Reuse the manuscript figure styling / colors / save helpers unchanged.
from generate_paper1_results_figures import (  # noqa: E402
    _CAFT_SEED_COLORS,
    _FROZEN_RHO_COLOR,
    _GROUP_HANDLE_COLORS,
    _GROUP_LABELS,
    _LAYERWISE_MODEL_GROUPS,
    _NAIVE_SEED_COLORS,
    OI_BLUEGREEN,
    OI_GRAY,
    OI_PINK,
    figure_metadata,
    save_figure,
    setup_style,
)
from corneal_fm.evaluation import (  # noqa: E402
    DEFAULT_LAYER_ORDER,
    marker_gradient_correlation,
)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

LAYERWISE_JSON = PROJECT_ROOT / "artifacts" / "paper1_layerwise_rho" / "summary.json"
PERCELL_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
FIGURES_DIR = PROJECT_ROOT / "figures" / "paper1"

LAYER_ORDER = list(DEFAULT_LAYER_ORDER)
ORDER_MAP = {c: i for i, c in enumerate(LAYER_ORDER)}
N_LAYERS = 13  # indices 0..12 (embedding_output + 12 transformer layers)

SANITY_TOL = 1e-6
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 42
DEFAULT_LAYERS = "12"  # only layer 12 is cached in situation B


def percell_file_for(key: str) -> str:
    """Map a summary.json model key to its Tier-B per-cell npz (L12 CLS)."""
    group, lam, seed = _LAYERWISE_MODEL_GROUPS[key]
    if lam is None:
        return "percell_frozen_gf.npz"
    arm = "naive" if lam == 0.0 else "caft"
    return f"percell_{arm}_s{seed}.npz"


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Metric / bootstrap core (bit-parity with marker_gradient_correlation)
# ---------------------------------------------------------------------------

def _point_rho(emb: np.ndarray, labels: np.ndarray) -> float:
    """Reference metric on the full cell set (identical to the layerwise pipeline)."""
    return marker_gradient_correlation(emb, labels, layer_order=LAYER_ORDER)


def _prepare(emb: np.ndarray, labels: np.ndarray):
    """Vectorized pre-computation (labels never change across resamples)."""
    e0 = np.asarray(emb, dtype=np.float64)
    valid = np.isin(labels, LAYER_ORDER)
    e0 = e0[valid]
    ordinals = np.array([ORDER_MAP[x] for x in labels[valid]], dtype=np.float64)
    k = len(LAYER_ORDER)
    onehot_all = ordinals[:, None] == np.arange(k)  # (n, k) bool, cached
    return e0, ordinals, onehot_all


def _bootstrap_rhos(
    e0: np.ndarray, ordinals: np.ndarray, onehot_all: np.ndarray,
    rng: np.random.Generator, n_boot: int,
) -> np.ndarray:
    """n_boot cell-level resamples of the marker-gradient rho.

    Bit-equivalent to marker_gradient_correlation on each resample: centroids
    via einsum over the (gathered) one-hot membership mask, PCA pseudotime
    axis from the centroid SVD (oriented LSC -> differentiated), Spearman of
    pseudotime vs layer ordinal. The resampled embedding rows are gathered
    once per iteration (``e0[idx]``) so that each membership bit is paired
    with the *resampled* cell; the per-cell pseudotime is computed as
    ``(e0 @ axis)[idx]`` (linearity).
    """
    n = e0.shape[0]
    k = len(LAYER_ORDER)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        o = ordinals[idx]
        ohb = onehot_all[idx]
        e_res = e0[idx]  # resampled rows; MUST pair with ohb
        ctr = np.einsum("ni,nd->id", ohb, e_res, optimize=True) / ohb.sum(0)[:, None]
        c = ctr - ctr.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        axis = vt[0]
        if (ctr[k - 1] - ctr[0]) @ axis < 0:
            axis = -axis
        p = (e0 @ axis)[idx]
        r = float(spearmanr(p, o)[0])
        out[b] = r
    return out


def _bootstrap_layer(
    emb: np.ndarray, labels: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    """Point estimate + percentile 95% CI for one layer (cell-level bootstrap)."""
    point = _point_rho(emb, labels)
    e0, ordinals, onehot_all = _prepare(emb, labels)
    rng = np.random.default_rng(seed)  # same stream for every model (paired)
    boots = _bootstrap_rhos(e0, ordinals, onehot_all, rng, n_boot)
    valid = boots[~np.isnan(boots)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {
        "point": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "boot_se": float(np.std(valid, ddof=1)),
        "n_valid": int(len(valid)),
    }


# ---------------------------------------------------------------------------
# Per-model worker (picklable, used by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    key = payload["key"]
    layers = payload["layers"]
    n_boot = payload["n_boot"]
    seed = payload["seed"]
    layer_dir = payload.get("layer_dir")
    summary = payload["summary"]

    rows: list[dict[str, Any]] = []
    if layer_dir:
        npz_path = Path(layer_dir) / f"{key}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"per-layer embeddings for '{key}' not found: {npz_path}"
            )
        d = np.load(npz_path, allow_pickle=True)
        all_emb = np.asarray(d["embeddings"], dtype=np.float64)  # (n_layers, n, d)
        labels = np.asarray(d["labels"])
        if all_emb.ndim != 3:
            raise ValueError(f"{npz_path}: 'embeddings' must be 3-D "
                             f"(n_layers, n_cells, dim), got {all_emb.shape}")
        for layer_i in layers:
            if layer_i >= all_emb.shape[0]:
                raise ValueError(f"{key}: layer {layer_i} outside cached "
                                 f"{all_emb.shape[0]} layers")
            rows.append(_row_for_layer(key, layer_i, all_emb[layer_i], labels,
                                       n_boot, seed, summary))
    else:
        # Situation B: only layer 12 (final-layer CLS) is cached.
        for layer_i in layers:
            if layer_i != 12:
                raise RuntimeError(
                    f"layer {layer_i} requested but only layer 12 per-cell "
                    f"embeddings are cached (percell_*.npz). Re-run with "
                    f"--layers 12, or provide --layer-embeddings-dir with "
                    f"(13, n, 768) per-layer CLS tensors."
                )
        npz_path = PERCELL_DIR / percell_file_for(key)
        if not npz_path.is_file():
            raise FileNotFoundError(f"per-cell cache not found: {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        emb = np.asarray(d["emb_ext"], dtype=np.float64)
        labels = np.asarray(d["y_true_ext"])
        rows.append(_row_for_layer(key, 12, emb, labels, n_boot, seed, summary))
    return rows


def _row_for_layer(
    key: str, layer_i: int, emb: np.ndarray, labels: np.ndarray,
    n_boot: int, seed: int, summary: dict[str, Any],
) -> dict[str, Any]:
    group, lam, sd = _LAYERWISE_MODEL_GROUPS[key]
    res = _bootstrap_layer(emb, labels, n_boot, seed)

    # Fail-closed sanity gate against the layerwise pipeline's rho.
    ref = summary["models"][key][layer_i]
    diff = abs(res["point"] - ref)
    if diff > SANITY_TOL:
        raise RuntimeError(
            f"SANITY GATE FAILED {key} layer {layer_i}: recomputed "
            f"rho={res['point']:.8f} vs summary.json={ref:.8f} "
            f"diff={diff:.2e} (> {SANITY_TOL:.0e}); refusing to continue."
        )
    return {
        "model": key,
        "group": group,
        "lambda": lam,
        "seed": sd,
        "layer": int(layer_i),
        "rho": res["point"],
        "ci_lower": res["ci_lower"],
        "ci_upper": res["ci_upper"],
        "ci_width": res["ci_upper"] - res["ci_lower"],
        "boot_se": res["boot_se"],
        "n_boot": int(n_boot),
        "n_valid": res["n_valid"],
        "boot_seed": int(seed),
        "ci_method": "percentile_2.5_97.5",
        "n_cells": int(labels.shape[0]),
    }


# ---------------------------------------------------------------------------
# Figure 7 redraw with CI error bars (identical styling to the original)
# ---------------------------------------------------------------------------

def _draw_figure7_with_ci(
    summary: dict[str, Any], ci_by_model: dict[str, dict[int, tuple[float, float]]],
    out_dir: Path,
) -> list[Path]:
    """Mirror figure7() from generate_paper1_results_figures.py, adding 95% CI
    error bars at the bootstrapped layer(s). Style/colors/annotations kept
    identical; rcParams from setup_style() and Okabe-Ito palette reused.
    """
    setup_style()
    layer_labels = summary.get("layer_labels") or []
    models = summary.get("models") or {}
    layers = [
        i for i, lab in enumerate(layer_labels)
        if isinstance(lab, str) and lab.startswith("transformer_layer_")
    ]
    if not layers:
        print("  [WARN] no transformer_layer_* columns; Figure 7 redraw skipped")
        return []

    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(111)

    csv_rows: list[dict[str, Any]] = []
    drawn_groups: set[str] = set()
    all_rho: list[float] = []
    for key, (group, lam, seed) in _LAYERWISE_MODEL_GROUPS.items():
        vals = models.get(key)
        if not isinstance(vals, list) or len(vals) <= max(layers):
            continue
        xs = list(layers)
        ys = []
        for i in xs:
            v = vals[i]
            ys.append(np.nan if v is None else float(v))
        if all(np.isnan(v) for v in ys):
            continue
        if group == "frozen":
            color, lw, ls = _FROZEN_RHO_COLOR, 2.4, "-"
        elif group == "naive":
            color = _NAIVE_SEED_COLORS.get(seed, OI_BLUEGREEN)
            lw, ls = 1.2, "-"
        else:  # CAFT
            color = _CAFT_SEED_COLORS.get(seed, OI_PINK)
            lw, ls = 1.2, "-"
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=3)
        drawn_groups.add(group)
        for xi, yi in zip(xs, ys):
            if np.isnan(yi):
                continue
            all_rho.append(yi)
            row = {
                "model": key, "group": group, "lambda": lam, "seed": seed,
                "layer": xi, "rho": yi,
            }
            ci = ci_by_model.get(key, {}).get(xi)
            if ci is not None:
                row["ci_lower"], row["ci_upper"] = ci
                ax.errorbar(
                    [xi], [yi], yerr=[[yi - ci[0]], [ci[1] - yi]],
                    fmt="none", ecolor=color, elinewidth=1.1,
                    capsize=3.0, capthick=1.1, zorder=4,
                )
                all_rho.extend(ci)
            csv_rows.append(row)

    if not drawn_groups:
        print("  [WARN] no rho curves drawn; Figure 7 redraw skipped")
        plt.close(fig)
        return []

    ax.axhline(0.0, color=OI_GRAY, ls=":", lw=0.7, zorder=1)

    annot_kw = dict(
        fontsize=6.5, color="#555555",
        arrowprops=dict(arrowstyle="->", lw=0.6, color="#555555"),
    )
    ax.annotate("naive L3 peak", xy=(3, 0.84), xytext=(3.6, 0.93),
                ha="left", **annot_kw)
    ax.annotate("collapse at L12", xy=(12, -0.04), xytext=(9.6, -0.155),
                ha="left", **annot_kw)
    ax.annotate("frozen L12 peak: 0.806", xy=(12, 0.806), xytext=(10.2, 0.90),
                ha="left", **annot_kw)

    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(list(range(1, 13)))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    rho_lo, rho_hi = min(all_rho), max(all_rho)
    ax.set_ylim(rho_lo - 0.20, rho_hi + 0.10)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Differentiation gradient \u03c1 (Spearman)")
    ax.set_title("Layerwise differentiation-gradient Spearman \u03c1",
                 fontsize=9, fontweight="bold")

    handles = [
        plt.Line2D([0], [0], color=_GROUP_HANDLE_COLORS[g], ls="-",
                   lw=2.4 if g == "frozen" else 1.4, label=_GROUP_LABELS[g])
        for g in ("frozen", "naive", "CAFT")
        if g in drawn_groups
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False)

    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_path = out_dir / "figure2_layerwise_rho.csv"
        csv_df.to_csv(csv_path, index=False, float_format="%.17g")
        print(f"  {len(csv_rows)} layerwise rho values -> {csv_path.name}")

    print(f"  {len(drawn_groups)} model group(s), layers 1-{len(layers)}")
    return save_figure(
        fig, "figure2_layerwise_rho", out_dir,
        figure_metadata("Figure 7 - Layerwise differentiation-gradient Spearman rho"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--layers", default=DEFAULT_LAYERS,
        help="Comma-separated layer indices to bootstrap (default '12'; "
             "only 12 works without --layer-embeddings-dir)",
    )
    parser.add_argument(
        "--layer-embeddings-dir", default=None,
        help="Dir with <model_key>.npz {embeddings: (n_layers, n, d), "
             "labels: (n,)} for per-layer bootstrap (situation A)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(7, os.cpu_count() or 1),
        help="Parallel workers across models (default min(7, nproc))",
    )
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the Figure 7 redraw (CI table only)")
    args = parser.parse_args()

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    if not layers:
        print(f"{_ts()}  ERROR: no layers requested")
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not LAYERWISE_JSON.is_file():
        print(f"{_ts()}  ERROR: {LAYERWISE_JSON} not found")
        return 1
    summary = json.loads(LAYERWISE_JSON.read_text(encoding="utf-8"))
    models = summary.get("models") or {}
    keys = [k for k in _LAYERWISE_MODEL_GROUPS if k in models]
    print(f"{_ts()}  Models: {keys}")
    print(f"{_ts()}  Layers: {layers} | n_boot={args.n_boot} seed={args.seed} "
          f"workers={args.workers}")

    payloads = [
        {
            "key": key, "layers": layers, "n_boot": args.n_boot,
            "seed": args.seed, "layer_dir": args.layer_embeddings_dir,
            "summary": summary,
        }
        for key in keys
    ]

    all_rows: list[dict[str, Any]] = []
    t0 = time.time()
    if args.workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, p): p["key"] for p in payloads}
            for fut in as_completed(futs):
                key = futs[fut]
                rows = fut.result()  # raises on gate failure / missing cache
                all_rows.extend(rows)
                print(f"{_ts()}  {key}: {len(rows)} layer(s) "
                      f"({time.time() - t0:.0f}s)")
    else:
        for p in payloads:
            rows = _worker(p)
            all_rows.extend(rows)
            print(f"{_ts()}  {p['key']}: {len(rows)} layer(s) "
                  f"({time.time() - t0:.0f}s)")

    if not all_rows:
        print(f"{_ts()}  ERROR: no bootstrap rows produced")
        return 1

    boot_df = pd.DataFrame(all_rows)
    boot_csv = out_dir / "figure2_layerwise_rho_bootstrap.csv"
    boot_df.to_csv(boot_csv, index=False, float_format="%.17g")
    print(f"{_ts()}  {len(boot_df)} bootstrap rows -> {boot_csv.name}")

    # --- Merge CI columns into the existing figure2 CSV (non-bootstrapped
    # --- layers keep NaN), preserving original values bit-exactly.
    fig2_csv = out_dir / "figure2_layerwise_rho.csv"
    if fig2_csv.is_file():
        fig2_df = pd.read_csv(fig2_csv)
        if "ci_lower" not in fig2_df.columns:
            fig2_df["ci_lower"] = np.nan
        if "ci_upper" not in fig2_df.columns:
            fig2_df["ci_upper"] = np.nan
        for _, r in boot_df.iterrows():
            m = (
                (fig2_df["model"] == r["model"])
                &                 (fig2_df["layer"] == r["layer"])
            )
            fig2_df.loc[m, "ci_lower"] = r["ci_lower"]
            fig2_df.loc[m, "ci_upper"] = r["ci_upper"]
        fig2_df.to_csv(fig2_csv, index=False, float_format="%.17g")
        print(f"{_ts()}  Updated {fig2_csv.name} with ci_lower/ci_upper "
              f"({int(boot_df['layer'].nunique())} bootstrapped layer(s))")
    else:
        print(f"{_ts()}  [WARN] {fig2_csv.name} not found; "
              f"skipping in-place CI merge")

    # --- Redraw Figure 7 with CI error bars -------------------------------
    if not args.no_plot:
        ci_by_model: dict[str, dict[int, tuple[float, float]]] = {}
        for _, r in boot_df.iterrows():
            ci_by_model.setdefault(r["model"], {})[int(r["layer"])] = (
                float(r["ci_lower"]), float(r["ci_upper"]),
            )
        paths = _draw_figure7_with_ci(summary, ci_by_model, out_dir)
        print(f"{_ts()}  Figure 7 redrawn: {[str(p) for p in paths]}")

    # --- Report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Bootstrap 95% CI (percentile, cell-level) — "
          f"n_boot={args.n_boot}, seed={args.seed}")
    print("=" * 78)
    hdr = (f"{'model':<16}{'group':<8}{'layer':>6}{'rho':>10}"
           f"{'ci_lower':>10}{'ci_upper':>10}{'width':>8}{'boot_se':>9}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in boot_df.sort_values(["layer", "model"]).iterrows():
        print(f"{r['model']:<16}{r['group']:<8}{int(r['layer']):>6}"
              f"{r['rho']:>10.4f}{r['ci_lower']:>10.4f}"
              f"{r['ci_upper']:>10.4f}{r['ci_width']:>8.4f}"
              f"{r['boot_se']:>9.4f}")
    print("-" * len(hdr))
    print(f"Total wall time: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
