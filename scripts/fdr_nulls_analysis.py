#!/usr/bin/env python
"""Tasks 2+3: null controls for rho_L12 + BH-FDR corrected-significance table.

Task 3 (nulls): label-shuffle and feature-shuffle null distributions of
marker_gradient_rank_correlation on the GSE153515 external test (frozen /
CE-naive / CAFT cached embeddings; ordinal reported against the same null —
the null is a property of the test set + metric, not the model). Reports
z-scores + percentiles for the observed rhos.

Task 2 (FDR): comparison grid = {frozen, CE-naive, CAFT(=L_cont), CORN-ordinal,
AA-FT 0.5/1.0, SS 0.5/1.0, flow 0.5} x {rho_L12, macro-F1, LAA, ECE}.
p-values: rho and LAA vs empirical label-shuffle nulls (one-sided); macro-F1
arm-vs-CE-naive via paired donor bootstrap (2 donors) or 3-seed sign-flip
permutation. Benjamini-Hochberg FDR at q=0.05 across the whole family.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/dq/Codex/Ai")
sys.path.insert(0, str(ROOT))

from corneal_fm.evaluation import (  # noqa: E402
    layer_adjacency_accuracy,
    marker_gradient_correlation,
)

LO = ["LSC", "TAC", "basal", "differentiated"]
N_LABEL_SHUFFLE = 500
N_FEATURE_SHUFFLE = 300
N_F1_BOOT = 2000
RNG = np.random.default_rng(42)


def load_ext(name: str, norm: bool):
    z = np.load(ROOT / "artifacts/paper1_tier_b_eval" / f"percell_{name}.npz")
    emb = z["emb_ext"].astype(float)
    y = z["y_true_ext"].astype(str)
    probs = z["probs_ext"]
    if norm:
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb, y, probs


# ---------------------------------------------------------------------------
# Task 3 — null distributions
# ---------------------------------------------------------------------------

def label_shuffle_null(emb, y, n=N_LABEL_SHUFFLE):
    rng = np.random.default_rng(RNG.integers(1e9))
    out = np.empty(n)
    y_arr = np.asarray(y)
    for i in range(n):
        perm = rng.permutation(len(y_arr))
        out[i] = marker_gradient_correlation(emb, y_arr[perm], layer_order=LO)
    return out


def feature_shuffle_null(emb, n=N_FEATURE_SHUFFLE):
    """Per-dimension permutation across cells.

    A GLOBAL feature permutation leaves the geometry metric invariant (the
    per-layer centroids and the projection are permuted identically), so the
    meaningful feature null breaks the inter-dimension geometry by permuting
    each column's values across cells independently.
    """
    rng = np.random.default_rng(RNG.integers(1e9))
    n_cells, d = emb.shape
    y = np.load(ROOT / "artifacts/paper1_tier_b_eval/percell_frozen_gf.npz")["y_true_ext"].astype(str)
    out = np.empty(n)
    for i in range(n):
        permuted = np.empty_like(emb)
        for j in range(d):
            permuted[:, j] = emb[rng.permutation(n_cells), j]
        out[i] = marker_gradient_correlation(permuted, y, layer_order=LO)
    return out


def laa_label_shuffle_null(emb, y, n=200):
    rng = np.random.default_rng(RNG.integers(1e9))
    out = np.empty(n)
    y_arr = np.asarray(y)
    for i in range(n):
        perm = rng.permutation(len(y_arr))
        out[i] = layer_adjacency_accuracy(y_arr[perm], emb, layer_order=LO)
    return out


def z_pct(obs, null):
    z = (obs - null.mean()) / (null.std() + 1e-12)
    pct = float((null <= obs).mean())
    p_one = (1 + int((null >= obs).sum())) / (len(null) + 1)
    return z, pct, p_one


def run_nulls():
    frozen_emb, frozen_y, _ = load_ext("frozen_gf", norm=True)
    naive_emb, naive_y, _ = load_ext("naive_s42", norm=False)
    caft_emb, caft_y, _ = load_ext("caft_s42", norm=False)

    null_lab_frozen = label_shuffle_null(frozen_emb, frozen_y)
    null_lab_naive = label_shuffle_null(naive_emb, naive_y)
    null_lab_caft = label_shuffle_null(caft_emb, caft_y)
    null_feat_frozen = feature_shuffle_null(frozen_emb)
    null_feat_naive = feature_shuffle_null(naive_emb)
    null_feat_caft = feature_shuffle_null(caft_emb)

    obs = {
        "frozen": (0.8064, frozen_emb, frozen_y),
        "ce_naive": (-0.0424, naive_emb, naive_y),
        "caft": (0.1913, caft_emb, caft_y),
        "corns_ordinal": (0.8330, frozen_emb, frozen_y),  # ordinal rho vs the test-set null
    }
    rows = []
    for name, (rho, emb, y) in obs.items():
        lab_null = {"frozen": null_lab_frozen, "ce_naive": null_lab_naive,
                    "caft": null_lab_caft, "corns_ordinal": null_lab_frozen}[name]
        feat_null = {"frozen": null_feat_frozen, "ce_naive": null_feat_naive,
                     "caft": null_feat_caft, "corns_ordinal": null_feat_frozen}[name]
        zl, pctl, pl = z_pct(rho, lab_null)
        zf, pctf, pf = z_pct(rho, feat_null)
        rows.append({
            "model": name, "rho_obs": rho,
            "label_null_mean": float(lab_null.mean()), "label_null_sd": float(lab_null.std()),
            "label_z": float(zl), "label_pctile": pctl, "label_p": pl,
            "feat_null_mean": float(feat_null.mean()), "feat_null_sd": float(feat_null.std()),
            "feat_z": float(zf), "feat_pctile": pctf, "feat_p": pf,
        })
        print(f"{name:14s} rho={rho:+.4f} | label-null {lab_null.mean():+.3f}±{lab_null.std():.3f} "
              f"z={zl:+.1f} pct={pctl*100:.1f}% p={pl:.4f} | feat-null {feat_null.mean():+.3f}±{feat_null.std():.3f} "
              f"z={zf:+.1f} pct={pctf*100:.1f}% p={pf:.4f}")

    # LAA null (frozen cells)
    null_laa = laa_label_shuffle_null(frozen_emb, frozen_y)
    print(f"\nLAA label-null (frozen cells): {null_laa.mean():.3f}±{null_laa.std():.3f}")

    out = {
        "external_test": "GSE153515", "n_label_shuffle": N_LABEL_SHUFFLE,
        "n_feature_shuffle": N_FEATURE_SHUFFLE,
        "label_null_frozen": null_lab_frozen.tolist(), "label_null_naive": null_lab_naive.tolist(),
        "label_null_caft": null_lab_caft.tolist(),
        "feature_null_frozen": null_feat_frozen.tolist(), "feature_null_naive": null_feat_naive.tolist(),
        "feature_null_caft": null_feat_caft.tolist(),
        "laa_label_null_frozen": null_laa.tolist(),
        "results": rows,
    }
    (ROOT / "artifacts/paper1_fdr_nulls").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "artifacts/paper1_fdr_nulls/null_controls.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved artifacts/paper1_fdr_nulls/null_controls.json")
    return out


# ---------------------------------------------------------------------------
# Task 2 — comparison grid + BH-FDR
# ---------------------------------------------------------------------------

def _results(path: str) -> dict:
    d = json.load(open(ROOT / path))
    se = d.get("secondary_endpoints", {})
    pe = d.get("primary_endpoint", {})
    return {
        "rho": se.get("marker_gradient_rank_correlation"),
        "f1": pe.get("macro_f1"),
        "per_donor": (pe.get("per_donor") or {}),
        "laa": se.get("layer_adjacency_accuracy"),
        "ece": se.get("expected_calibration_error"),
    }


def f1_p_two_sided(arm_donors: dict, ref_donors: dict) -> float:
    """Paired donor bootstrap of the F1 difference (2 donors)."""
    keys = [k for k in arm_donors if k in ref_donors]
    d = np.array([arm_donors[k] - ref_donors[k] for k in keys], dtype=float)
    rng = np.random.default_rng(7)
    means = np.empty(N_F1_BOOT)
    for b in range(N_F1_BOOT):
        idx = rng.integers(0, len(d), size=len(d))
        means[b] = d[idx].mean()
    p = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return max(p, 1.0 / N_F1_BOOT)


def f1_p_seed_perm(arm_seeds: list[float], ref_seeds: list[float]) -> float:
    """Exact sign-flip permutation over 3 paired seeds (2^3=8 arrangements)."""
    d = np.array(arm_seeds) - np.array(ref_seeds)
    obs = abs(d.mean())
    cnt = 0
    for bits in range(1 << len(d)):
        flip = np.array([1.0 if (bits >> j) & 1 else -1.0 for j in range(len(d))])
        cnt += abs((flip * d).mean()) >= obs - 1e-12
    return cnt / (1 << len(d))


def bh_fdr(pvals: list[float], q: float = 0.05):
    """Benjamini-Hochberg corrected q-values (step-up, applied in p-sorted
    order, then mapped back to the caller's order)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    q_sorted = sorted_p * m / np.arange(1, m + 1)
    q_stepup = np.minimum.accumulate(q_sorted[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = q_stepup
    return adj


def run_fdr(nulls: dict) -> None:
    lab_null = np.asarray(nulls["label_null_frozen"])
    laa_null = np.asarray(nulls["laa_label_null_frozen"])

    # 3-seed means (D5 results)
    naive_rho = np.mean([-0.0424, 0.0539, -0.0360])
    naive_f1s = [0.6542, 0.6515, 0.6420]
    caft_rho = np.mean([0.1913, 0.2298, 0.2095])
    caft_f1s = [0.6330, 0.6559, 0.6464]
    ord_rho = np.mean([0.8341, 0.8299, 0.8351])
    ord_f1s = [0.6481, 0.6459, 0.6529]

    models = {
        "frozen": {"rho": 0.8064, "f1": 0.4421, "laa": 0.9665, "ece": 0.5967,
                   "per_donor": {"donor_1": 0.5093, "donor_2": 0.3749}},
        "ce_naive": {"rho": naive_rho, "f1": np.mean(naive_f1s), "laa": 0.9768, "ece": 0.1008,
                     "seeds": naive_f1s},
        "caft": {"rho": caft_rho, "f1": np.mean(caft_f1s), "laa": 0.9773, "ece": 0.1084,
                 "seeds": caft_f1s},
        "corns_ordinal": {"rho": ord_rho, "f1": np.mean(ord_f1s), "laa": 0.9739, "ece": 0.1251,
                          "seeds": ord_f1s},
        "aa_ft_0.5": _results("artifacts/paper1_method_arms/results_axis_ft_0.5_42.json"),
        "aa_ft_1.0": _results("artifacts/paper1_method_arms/results_axis_ft_1.0_42.json"),
        "ss_0.5": _results("artifacts/paper1_method_arms/results_soft_spearman_0.5_42.json"),
        "ss_1.0": _results("artifacts/paper1_method_arms/results_soft_spearman_1.0_42.json"),
        "flow_0.5": _results("artifacts/paper1_method_arms/results_flow_0.5_42.json"),
    }
    naive_donors = models["ce_naive"]["donors"] if "donors" in models["ce_naive"] else \
        _results("artifacts/paper1_tier_b_eval/results_0.0_42.json")["per_donor"]

    tests = []
    for name, m in models.items():
        # rho vs label-shuffle null (one-sided H1: rho > 0)
        rho = float(m["rho"])
        p_rho = (1 + int((lab_null >= rho).sum())) / (len(lab_null) + 1)
        tests.append({"comparison": f"{name} rho_L12 > 0", "p": p_rho, "obs": rho})
        # LAA vs chance null (one-sided H1: LAA > chance)
        laa = float(m["laa"])
        p_laa = (1 + int((laa_null >= laa).sum())) / (len(laa_null) + 1)
        tests.append({"comparison": f"{name} LAA > chance", "p": p_laa, "obs": laa})
        # macro-F1 vs CE-naive (two-sided)
        if name == "ce_naive":
            continue
        if "seeds" in m:
            p_f1 = f1_p_seed_perm(m["seeds"], naive_f1s)
        else:
            p_f1 = f1_p_two_sided(m["per_donor"], naive_donors)
        tests.append({"comparison": f"{name} F1 = CE-naive", "p": p_f1, "obs": float(m["f1"])})

    pvals = [t["p"] for t in tests]
    adj = bh_fdr(pvals)
    for t, qv in zip(tests, adj):
        t["q"] = float(qv)
        t["significant_0.05"] = bool(qv < 0.05)

    print(f"\n=== BH-FDR table (family of {len(tests)} comparisons, q=0.05) ===")
    print(f"{'comparison':32s} {'obs':>8s} {'p':>8s} {'q':>8s} {'sig':>5s}")
    for t in sorted(tests, key=lambda x: x["q"]):
        print(f"{t['comparison']:32s} {t['obs']:8.4f} {t['p']:8.4f} {t['q']:8.4f} "
              f"{'*' if t['significant_0.05'] else ''}")

    out = {"family_size": len(tests), "q_threshold": 0.05, "tests": tests}
    with open(ROOT / "artifacts/paper1_fdr_nulls/bh_fdr_table.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved artifacts/paper1_fdr_nulls/bh_fdr_table.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    nulls = None
    if mode in ("nulls", "all"):
        nulls = run_nulls()
    if mode in ("fdr", "all"):
        if nulls is None:
            nulls = json.load(open(ROOT / "artifacts/paper1_fdr_nulls/null_controls.json"))
        run_fdr(nulls)
