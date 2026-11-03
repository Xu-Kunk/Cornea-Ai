#!/usr/bin/env python
"""INDEPENDENT recomputation of Paper1 Tier-B metrics from percell npz arrays.

Does NOT import corneal_fm.evaluation (or any project metric code) — every
metric is reimplemented from first principles here. Compares against:
  - percell_*.npz.meta.json sanity_gate_external
  - artifacts/paper1_tier_b_eval/results_*.json (epoch-3 primary/secondary)
  - artifacts/paper1_layerwise_rho/layerwise_rho_*.json (L12 rho cross-check)

CPU-only. Writes artifacts/paper1_pipeline_audit/metric_recompute.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/dq/Codex/Ai")
EVAL_DIR = ROOT / "artifacts" / "paper1_tier_b_eval"
RHO_DIR = ROOT / "artifacts" / "paper1_layerwise_rho"
OUT = ROOT / "artifacts" / "paper1_pipeline_audit"

CLASSES = ["LSC", "TAC", "basal", "differentiated"]
N_BOOT = 1000
SEED = 42
N_BINS = 10


# ---------------------------------------------------------------------------
# Independent metric implementations
# ---------------------------------------------------------------------------

def donor_macro_f1_ind(y_true, y_pred, donors, classes):
    """Per-donor macro-F1 (sklearn f1 per donor, macro over classes), mean over donors."""
    donor_ids = np.unique(donors)
    per_donor = {}
    for d in donor_ids:
        m = donors == d
        per_donor[str(d)] = float(f1_score(y_true[m], y_pred[m], labels=classes,
                                           average="macro", zero_division=0))
    macro = float(np.mean(list(per_donor.values())))
    return macro, per_donor


def bootstrap_ci_ind(per_donor_vals, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap CI of the mean over donor ids (independent impl)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(per_donor_vals.values()), dtype=float)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means[b] = arr[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_class_ind(y_true, y_pred, classes):
    """Per-class P/R/F1/support + confusion, independent implementation."""
    n = len(classes)
    conf = np.zeros((n, n), dtype=np.int64)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            conf[idx[t], idx[p]] += 1
    per = {}
    for i, c in enumerate(classes):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        sup = int(conf[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per[c] = {"precision": float(prec), "recall": float(rec),
                  "f1": float(f1), "support": sup}
    return per, conf


def laa_ind(labels, emb, layer_order):
    """LAA: fraction of cells whose NN (k=2) label is within |d|<=1 in layer_order."""
    valid = np.array([l in layer_order for l in labels])
    if valid.sum() < 2:
        return 0.0
    lab = labels[valid]
    emb_v = np.asarray(emb, dtype=float)[valid]
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(emb_v)
    _, idx = nn.kneighbors(emb_v)
    order = {c: i for i, c in enumerate(layer_order)}
    li = np.array([order[x] for x in lab])
    lj = np.array([order[x] for x in lab[idx[:, 1]]])
    return float((np.abs(li - lj) <= 1).mean())


def marker_rho_ind(emb, labels, layer_order):
    """rho: PCA axis of per-layer centroids -> pseudotime -> Spearman vs ordinal."""
    emb = np.asarray(emb, dtype=float)
    valid = np.array([l in layer_order for l in labels])
    lab = labels[valid]
    e = emb[valid]
    if len(lab) == 0:
        return float("nan")
    order = {c: i for i, c in enumerate(layer_order)}
    ords = np.array([order[x] for x in lab], dtype=float)
    present = [c for c in layer_order if (ords == order[c]).any()]
    if len(present) < 2:
        return float("nan")
    cents = np.vstack([e[ords == order[c]].mean(0) for c in present])
    cc = cents - cents.mean(0)
    # PCA via eigen-decomposition of the covariance (independent of np.linalg.svd path)
    cov = cc.T @ cc / max(len(present) - 1, 1)
    w, V = np.linalg.eigh(cov)
    axis = V[:, np.argmax(w)]
    first = e[ords == order[present[0]]].mean(0)
    last = e[ords == order[present[-1]]].mean(0)
    if (last - first) @ axis < 0:
        axis = -axis
    pt = e @ axis
    rho, _ = spearmanr(pt, ords)
    return float(rho)


def ece_ind(y_true_idx, y_prob, n_bins=N_BINS, mode="project"):
    """ECE, independent implementation. mode='project' mirrors evaluation.py
    (conf > lo for bin 0 => conf=0 excluded); mode='std' uses half-open [lo,hi)
    with the last bin closed."""
    y_prob = np.asarray(y_prob, dtype=float)
    pred = y_prob.argmax(1)
    conf = y_prob.max(1)
    acc = pred == y_true_idx
    n = len(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if mode == "project":
            mask = conf > lo if b == 0 else (conf > lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def load_npz(name):
    z = np.load(EVAL_DIR / name)
    return {k: z[k] for k in z.files}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    npz_names = [
        "percell_naive_s42.npz", "percell_naive_s123.npz", "percell_naive_s456.npz",
        "percell_caft_s42.npz", "percell_caft_s123.npz", "percell_caft_s456.npz",
        "percell_frozen_gf.npz",
    ]
    for name in npz_names:
        meta = json.loads((EVAL_DIR / name.replace(".npz", ".npz.meta.json")).read_text())
        data = load_npz(name)
        y_true = data["y_true_ext"].astype(str)
        y_pred = data["y_pred_ext"].astype(str)
        probs = data["probs_ext"].astype(float)
        emb = data["emb_ext"]
        donors = data["donor_ext"].astype(str)

        macro, per_donor = donor_macro_f1_ind(y_true, y_pred, donors, CLASSES)
        lo, hi = bootstrap_ci_ind(per_donor)
        per_class, conf = per_class_ind(y_true, y_pred, CLASSES)
        t0 = time.time()
        laa = laa_ind(y_true, emb, CLASSES)
        t_laa = time.time() - t0
        rho = marker_rho_ind(emb, y_true, CLASSES)
        idx_map = {c: i for i, c in enumerate(CLASSES)}
        y_idx = np.array([idx_map[l] for l in y_true], dtype=int)
        ece_proj = ece_ind(y_idx, probs, mode="project")
        ece_std = ece_ind(y_idx, probs, mode="std")

        row = {
            "model": meta["model"], "lam": meta["lambda"], "seed": meta["seed"],
            "independent": {
                "macro_f1": macro, "ci_lower": lo, "ci_upper": hi,
                "per_donor": per_donor, "per_class": per_class,
                "confusion": conf.tolist(), "laa": laa, "rho": rho,
                "ece": ece_proj, "ece_project_binning": ece_proj, "ece_std_binning": ece_std,
                "laa_seconds": round(t_laa, 1),
            },
            "meta_gate": meta.get("sanity_gate_external"),
        }
        # compare against meta gate
        g = meta.get("sanity_gate_external") or {}
        diffs = {}
        if g:
            for f in ("macro_f1", "ci_lower", "ci_upper", "laa", "rho", "ece"):
                diffs[f] = float(row["independent"][f]) - float(g[f])
        row["diff_vs_meta"] = diffs
        # compare against epoch-3 results JSON
        if meta["lambda"] is None:
            rj = json.loads((EVAL_DIR / "results_frozen_gf.json").read_text())
        else:
            rj = json.loads((EVAL_DIR / f"results_{meta['lambda']}_{meta['seed']}.json").read_text())
        ref = {
            "macro_f1": rj["primary_endpoint"]["macro_f1"],
            "ci_lower": rj["primary_endpoint"]["ci_lower"],
            "ci_upper": rj["primary_endpoint"]["ci_upper"],
            "rho": rj["secondary_endpoints"]["marker_gradient_rank_correlation"],
            "laa": rj["secondary_endpoints"]["layer_adjacency_accuracy"],
            "ece": rj["secondary_endpoints"]["expected_calibration_error"],
        }
        row["diff_vs_results_json"] = {f: float(row["independent"][f]) - float(ref[f]) for f in ref}
        report[name] = row
        print(f"[{name}] macro={macro:.6f} ci=({lo:.6f},{hi:.6f}) laa={laa:.6f} "
              f"rho={rho:.6f} ece(proj)={ece_proj:.6f} ece(std)={ece_std:.6f} "
              f"(laa {t_laa:.0f}s)")

        # L12 rho cross-check vs layerwise json
        arm = "frozen_gf" if meta["lambda"] is None else (
            "naive" if meta["lambda"] == 0.0 else "caft")
        if meta["lambda"] is None:
            lw_path = RHO_DIR / "layerwise_rho_frozen_gf.json"
        else:
            lw_path = RHO_DIR / f"layerwise_rho_{arm}_s{meta['seed']}.json"
        if lw_path.is_file():
            lw = json.loads(lw_path.read_text())
            row["layerwise_L12_rho"] = lw["marker_gradient_rho"][-1]
            row["rho_diff_vs_layerwise_L12"] = rho - lw["marker_gradient_rho"][-1]

    # also verify layerwise rho L1..L11 from the layerwise JSONs (no recompute possible
    # on CPU for hidden states; only the final-layer emb exists in npz). Report the
    # L1 rho and L0-nan behaviour for the record.
    summary = json.loads((RHO_DIR / "summary.json").read_text())
    report["layerwise_summary"] = {
        "n_models": summary["n_models"],
        "sanity_gate_all_passed": summary["sanity_gate_all_passed"],
        "sanity_gate": summary["sanity_gate"],
        "L0_nan_for_all": all(np.isnan(v[0]) for v in summary["models"].values()),
        "L1_by_model": {k: v[1] for k, v in summary["models"].items()},
    }

    (OUT / "metric_recompute.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nwrote", OUT / "metric_recompute.json")


if __name__ == "__main__":
    main()
