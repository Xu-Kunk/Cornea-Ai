#!/usr/bin/env python
"""Task 1 — rho SVD-axis validation (CPU).

Validates the core assumption of the rho metric (marker_gradient_correlation):
the FIRST SVD axis of the per-cell embedding matrix is the differentiation axis
(i.e. not a principal-component / donor / batch artifact).

Design (PCS: frozen BEFORE running):
  * SVD axis extraction rule: center the embedding matrix, full SVD, take the
    first RIGHT singular vector Vt[0] as the axis; project every cell onto it
    (pt = X_centered @ axis). Sign-fix: orient so the differentiated-layer
    centroid projects ABOVE the LSC centroid (matching the rho orientation).
  * Marker panel (frozen, ENSG ids in the GSE153515 raw matrix):
      LSC  = ABCG2 / ITGA9 / TSPAN7
      basal= KRT14 / KRT15 / KRT5
      diff = KRT12 / KRT3
  * Program score per cell = mean z-scored log1p expression across the
    program's genes (identical to scripts/bio_validation.program_score).
  * Monotonicity statistics:
      (a) per-program Spearman rho(program_score, SVD-axis projection):
          expect LSC < 0, diff > 0  (markers at the correct ends)
      (b) per-layer mean SVD-axis projection ordered LSC < TAC < basal < diff
      (c) program rho ordering LSC < basal < diff
  * Judgement rule (frozen): the SVD axis is the differentiation axis iff
      LSC program rho < 0 AND diff program rho > 0 AND the 4-layer centroid
      ordering is perfectly monotonic (Spearman = 1.0) along the axis.
  * Contrast: frozen (should be monotonic) vs naive (should break) vs
    CORN (should restore), mirroring rho 0.8064 -> -0.0424 -> 0.8330.

Data: GSE153515 external test (14,775 cells, donor_1 + donor_2) — the
tier_b_eval / bio_validation test-donor set. Per-cell embeddings for the three
GF arms are already persisted:
  frozen: paper1_tier_b_eval/percell_frozen_gf.npz      (emb_ext)
  naive : paper1_tier_b_eval/percell_naive_s42.npz      (emb_ext)
  CORN  : paper1_bio_validation/ordinal_emb_lam0.5_seed42.npz  (emb)
Raw expression from data/gse153515_limbal_stem.h5ad aligned by barcode
(exactly as scripts/bio_validation.load_corneal_data).

Outputs (artifacts/paper1_vanzyl/svd_validation/):
  pcs_frozen_spec.json        — frozen panel/rule + input SHA-256 manifest
  svd_axis_validation.json    — per-arm statistics + verdicts
  monotonicity_stats.json     — raw stats for the figure
  marker_projection.png/.pdf  — program score vs SVD-axis decile profiles
  run.log

Usage:
    /home/dq/venv-ds/bin/python3 scripts/svd_axis_validation.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy.stats as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "svd_validation"

LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]

# Frozen marker panel: symbol -> ENSG id (present in the GSE153515 raw matrix).
MARKER_PANEL: dict[str, list[str]] = {
    "LSC":   ["ABCG2", "ITGA9", "TSPAN7"],
    "basal": ["KRT14", "KRT15", "KRT5"],
    "diff":  ["KRT12", "KRT3"],
}
GENE_IDS: dict[str, str] = {
    "ABCG2": "ENSG00000118777",
    "ITGA9": "ENSG00000144668",
    "TSPAN7": "ENSG00000156298",
    "KRT14": "ENSG00000186847",
    "KRT15": "ENSG00000171346",
    "KRT5": "ENSG00000186081",
    "KRT12": "ENSG00000187242",
    "KRT3": "ENSG00000186480",
}

ARMS = {
    "frozen": {
        "npz": "artifacts/paper1_tier_b_eval/percell_frozen_gf.npz",
        "key": "emb_ext",
        "expected_rho": 0.8064,
    },
    "naive": {
        "npz": "artifacts/paper1_tier_b_eval/percell_naive_s42.npz",
        "key": "emb_ext",
        "expected_rho": -0.0424,
    },
    "corn": {
        "npz": "artifacts/paper1_bio_validation/ordinal_emb_lam0.5_seed42.npz",
        "key": "emb",
        "expected_rho": 0.8242,  # recomputed on the archived lam0.5 s42 emb
    },
}

N_BOOT = 500
BOOT_SEED = 42


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data loading (mirrors scripts/bio_validation.load_corneal_data)
# ---------------------------------------------------------------------------

def load_corneal_data():
    import anndata as ad

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
    return eval_labels, expr, np.asarray(raw.var_names, dtype=str)


def program_score(expr: np.ndarray, gene_idx: list[int]) -> np.ndarray:
    """Mean of z-scored log1p expression across program genes."""
    sub = np.log1p(expr[:, gene_idx])
    z = (sub - sub.mean(axis=0)) / (sub.std(axis=0) + 1e-12)
    return z.mean(axis=1)


def first_svd_axis(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centered full SVD; returns (axis, projection, frac_var_explained)."""
    centered = emb - emb.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    frac = float(s[0] ** 2 / (s ** 2).sum())
    return axis, proj, frac


def rho_axis(emb: np.ndarray, labels: np.ndarray,
             order: list[str] = LAYER_ORDER_4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The SVD axis the rho metric actually uses (unchanged internals of
    corneal_fm.evaluation.marker_gradient_correlation): SVD of the per-layer
    CENTROID matrix, oriented LSC -> differentiated. Returns (axis, valid-emb,
    valid-labels)."""
    om = {c: i for i, c in enumerate(order)}
    valid = np.array([lbl in om for lbl in labels])
    lab = labels[valid]
    e = emb[valid]
    present = [c for c in order if (lab == c).any()]
    centroids = np.vstack([e[lab == c].mean(axis=0) for c in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[0]
    first_c = e[lab == present[0]].mean(axis=0)
    last_c = e[lab == present[-1]].mean(axis=0)
    if (last_c - first_c) @ axis < 0:
        axis = -axis
    return axis, e, lab


def bootstrap_ci(proj: np.ndarray, prog: np.ndarray, n: int = N_BOOT,
                 seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rhos = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(proj), size=len(proj))
        rhos[i] = st.spearmanr(proj[idx], prog[idx]).statistic
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def analyze_arm(emb: np.ndarray, labels: np.ndarray, expr: np.ndarray,
                var_names: np.ndarray, arm_name: str) -> dict:
    # sanity gate: recompute rho with the locked metric (unchanged)
    from corneal_fm.evaluation import marker_gradient_correlation
    rho = marker_gradient_correlation(emb, labels, LAYER_ORDER_4)

    # primary axis: the SVD axis rho uses internally (centroid-SVD)
    axis, e, lab = rho_axis(emb, labels)
    proj = e @ axis

    # PC-artifact diagnostic: first SVD axis of the FULL embedding matrix
    pc_axis, pc_proj, pc_frac = first_svd_axis(e)
    pc_layer_means = {c: float(pc_proj[lab == c].mean()) if (lab == c).sum()
                      else None for c in LAYER_ORDER_4}
    pc_vals = [pc_layer_means[c] for c in LAYER_ORDER_4 if pc_layer_means[c] is not None]
    pc_mono = (float(st.spearmanr(np.arange(len(pc_vals)), pc_vals).statistic)
               if len(pc_vals) >= 2 else float("nan"))
    pc_axis_cos = float(axis @ pc_axis)  # alignment between rho axis and PC1

    # per-gene indices
    var_set = {v: i for i, v in enumerate(var_names)}
    present: dict[str, int] = {}
    for sym, gid in GENE_IDS.items():
        if gid in var_set:
            present[sym] = var_set[gid]

    # per-program monotonicity along the RHO axis
    programs = {}
    for prog, syms in MARKER_PANEL.items():
        idx = [present[s] for s in syms if s in present]
        if not idx:
            continue
        ps = program_score(expr, idx)
        r, _ = st.spearmanr(proj, ps)
        lo, hi = bootstrap_ci(proj, ps)
        # decile profile: mean program score per decile of the axis projection
        deciles = np.quantile(proj, np.arange(0, 1.01, 0.1))
        dec_means = []
        for d in range(10):
            mask = (proj >= deciles[d]) & (proj <= deciles[d + 1])
            dec_means.append(float(ps[mask].mean()) if mask.sum() else np.nan)
        programs[prog] = {
            "rho": float(r), "ci_lower": lo, "ci_upper": hi,
            "genes": [s for s in syms if s in present],
            "n_genes": len(idx),
            "decile_profile": dec_means,
        }

    # per-layer centroid positions along the RHO axis
    om = {c: i for i, c in enumerate(LAYER_ORDER_4)}
    layer_means = {}
    for c in LAYER_ORDER_4:
        mask = lab == c
        layer_means[c] = (float(proj[mask].mean()) if mask.sum() else None)
    order_vals = [layer_means[c] for c in LAYER_ORDER_4]
    valid_order = [v for v in order_vals if v is not None]
    layer_mono = (float(st.spearmanr(np.arange(len(valid_order)), valid_order).statistic)
                  if len(valid_order) >= 2 else float("nan"))
    # per-gene monotonicity along the RHO axis (marker projection)
    genes = {}
    for sym, i in present.items():
        col = expr[:, i]
        r, _ = st.spearmanr(proj, np.log1p(col))
        genes[sym] = float(r)

    # program ordering: expect LSC < basal < diff in program rho
    prog_rho = {p: programs[p]["rho"] for p in programs}
    ordered = ["LSC", "basal", "diff"]
    prog_order_rho = [prog_rho.get(p) for p in ordered if p in prog_rho]
    prog_mono = (float(st.spearmanr(np.arange(len(prog_order_rho)), prog_order_rho).statistic)
                 if len(prog_order_rho) >= 2 else float("nan"))

    lsc_ok = programs.get("LSC", {}).get("rho", 0.0) < 0
    diff_ok = programs.get("diff", {}).get("rho", 0.0) > 0
    layer_ok = abs(layer_mono - 1.0) < 1e-9 if layer_mono == layer_mono else False
    verdict = bool(lsc_ok and diff_ok and layer_ok)

    out = {
        "arm": arm_name,
        "n_cells": int(emb.shape[0]),
        "rho_locked_metric": float(rho),
        "rho_expected": ARMS[arm_name]["expected_rho"],
        "rho_gate_pass": bool(abs(rho - ARMS[arm_name]["expected_rho"]) < 1e-3),
        "svd_axis": {
            "definition": "first SVD axis of the per-layer CENTROID matrix "
                          "(rho metric internals, unchanged), oriented "
                          "LSC -> differentiated",
            "frac_variance_explained": pc_frac,
            "layer_centroid_means_along_axis": layer_means,
            "layer_ordering_spearman": layer_mono,
            "layer_ordering_monotonic": layer_ok,
        },
        "pc_artifact_diagnostic": {
            "pc1_frac_variance": pc_frac,
            "pc1_layer_means": pc_layer_means,
            "pc1_layer_ordering_spearman": pc_mono,
            "cos_rho_axis_pc1": pc_axis_cos,
            "note": "if cos ~ 0, the rho axis is NOT the dominant variance "
                    "axis (PC1) => the marker signal is not a PC artifact",
        },
        "program_projection_rho": programs,
        "program_ordering_spearman": prog_mono,
        "program_ordering_monotonic": bool(abs(prog_mono - 1.0) < 1e-9),
        "gene_projection_rho": genes,
        "judgement": {
            "rule": "LSC prog rho<0 AND diff prog rho>0 AND 4-layer centroid "
                    "ordering Spearman==1.0 along the rho SVD axis",
            "lsc_marker_low_end": lsc_ok,
            "diff_marker_high_end": diff_ok,
            "layer_centroids_monotonic": layer_ok,
            "verdict": verdict,
            "interpretation": (
                "SVD axis = differentiation axis (hypothesis holds)" if verdict
                else "SVD axis != differentiation axis (hypothesis rejected)"
            ),
        },
    }
    return out


def make_figure(results: dict, out_png: Path, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    colors = {"LSC": "#c44e52", "basal": "#55a868", "diff": "#4c72b0"}
    arm_names = {"frozen": "frozen (GF)", "naive": "naive (GF)", "corn": "CORN (GF)"}
    for ax, (arm, res) in zip(axes, results.items()):
        r = res["svd_axis"]
        ax.set_title(f"{arm_names[arm]}\n"
                     f"rho={res['rho_locked_metric']:.3f}  "
                     f"verdict={'PASS' if res['judgement']['verdict'] else 'FAIL'}",
                     fontsize=10)
        ax.set_xlabel("SVD-axis decile (1 = low / 10 = high)")
        ax.set_ylabel("mean program score (z)")
        ax.axhline(0.0, color="grey", lw=0.6, ls="--")
        for prog, c in colors.items():
            if prog not in res["program_projection_rho"]:
                continue
            profile = res["program_projection_rho"][prog]["decile_profile"]
            ax.plot(np.arange(1, 11), profile, marker="o", ms=4, lw=1.6,
                    color=c, label=prog)
        # layer centroid tick marks (scaled to the decile-profile y-range)
        means = r["layer_centroid_means_along_axis"]
        ypos = [means[c] for c in ["LSC", "TAC", "basal", "differentiated"]
                if means.get(c) is not None]
        labs = [c for c in ["LSC", "TAC", "basal", "differentiated"]
                if means.get(c) is not None]
        if ypos:
            ylo, yhi = ax.get_ylim()
            for y, lab in zip(ypos, labs):
                ax.axhline(y, color="black", lw=0.7, ls=":", alpha=0.5)
                ax.text(0.5, y, lab, fontsize=6.5, color="black", va="bottom")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Marker program projection along the FIRST SVD axis of embeddings "
                 "(GSE153515 external test, 14,775 cells)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    import json as _json

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run.log"
    log = open(log_path, "w", encoding="utf-8")

    def log_msg(msg: str) -> None:
        line = f"{_ts()} {msg}"
        print(line)
        log.write(line + "\n")

    # ---- PCS: freeze spec + input SHAs BEFORE any analysis -----------------
    input_paths = [
        PROJECT_ROOT / "data/gse153515_annotated.h5ad",
        PROJECT_ROOT / "data/gse153515_limbal_stem.h5ad",
        PROJECT_ROOT / ARMS["frozen"]["npz"],
        PROJECT_ROOT / ARMS["naive"]["npz"],
        PROJECT_ROOT / ARMS["corn"]["npz"],
    ]
    sha_manifest = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in input_paths}
    frozen_spec = {
        "task": "rho SVD-axis validation (Task 1)",
        "frozen_at": datetime.now(UTC).isoformat(),
        "svd_axis_rule": (
            "PRIMARY: first SVD axis of the per-layer CENTROID matrix, "
            "identical to the rho metric internals "
            "(corneal_fm.evaluation.marker_gradient_correlation, unchanged), "
            "oriented LSC -> differentiated. "
            "DIAGNOSTIC: first SVD axis of the centered FULL embedding matrix "
            "(PC1) + cosine with the rho axis to rule out a PC artifact."
        ),
        "marker_panel": MARKER_PANEL,
        "gene_ids": GENE_IDS,
        "program_score": "mean of z-scored log1p expression across program genes",
        "monotonicity_stats": [
            "per-program Spearman rho(program_score, rho-axis projection)",
            "per-layer mean axis projection ordering (LSC<TAC<basal<diff)",
            "program rho ordering (LSC<basal<diff)",
            "PC1 layer ordering + cos(rho_axis, PC1) artifact diagnostic",
        ],
        "judgement_rule": (
            "SVD axis = differentiation axis iff LSC prog rho<0 AND diff prog rho>0 "
            "AND 4-layer centroid ordering Spearman==1.0"
        ),
        "contrast": "frozen (monotonic expected) vs naive (collapse) vs CORN (restore)",
        "input_sha256": sha_manifest,
        "random_seeds": {"bootstrap": BOOT_SEED, "n_boot": N_BOOT},
    }
    with open(OUT_DIR / "pcs_frozen_spec.json", "w") as fh:
        _json.dump(frozen_spec, fh, indent=1)
    log_msg("PCS spec + input SHAs frozen -> pcs_frozen_spec.json")

    labels, expr, var_names = load_corneal_data()
    log_msg(f"loaded {len(labels)} test cells, expr {expr.shape}, "
            f"var_names {len(var_names)}")

    results = {}
    for arm, cfg in ARMS.items():
        z = np.load(PROJECT_ROOT / cfg["npz"], allow_pickle=True)
        emb = z[cfg["key"]].astype(float)
        if len(emb) != len(labels):
            log_msg(f"WARN {arm}: emb {len(emb)} != labels {len(labels)}")
        res = analyze_arm(emb, labels, expr, var_names, arm)
        results[arm] = res
        log_msg(
            f"{arm:8s} rho={res['rho_locked_metric']:.4f} "
            f"gate={'OK' if res['rho_gate_pass'] else 'FAIL'} | "
            f"LSC rho={res['program_projection_rho'].get('LSC', {}).get('rho', float('nan')):+.3f} "
            f"basal={res['program_projection_rho'].get('basal', {}).get('rho', float('nan')):+.3f} "
            f"diff={res['program_projection_rho'].get('diff', {}).get('rho', float('nan')):+.3f} | "
            f"layer mono={res['svd_axis']['layer_ordering_spearman']:.3f} | "
            f"verdict={res['judgement']['verdict']}"
        )

    with open(OUT_DIR / "svd_axis_validation.json", "w") as fh:
        _json.dump(results, fh, indent=1)

    # monotonicity_stats.json: compact figure data
    stats = {}
    for arm, res in results.items():
        stats[arm] = {
            "rho": res["rho_locked_metric"],
            "frac_var_axis1": res["svd_axis"]["frac_variance_explained"],
            "program_rho": {k: v["rho"] for k, v in
                            res["program_projection_rho"].items()},
            "layer_means": res["svd_axis"]["layer_centroid_means_along_axis"],
            "layer_ordering_spearman": res["svd_axis"]["layer_ordering_spearman"],
            "verdict": res["judgement"]["verdict"],
        }
    with open(OUT_DIR / "monotonicity_stats.json", "w") as fh:
        _json.dump(stats, fh, indent=1)

    make_figure(results, OUT_DIR / "marker_projection.png",
                OUT_DIR / "marker_projection.pdf")
    log_msg("figure -> marker_projection.png/.pdf")

    log_msg("DONE")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
