#!/usr/bin/env python
"""Audit the reported frozen -> fine-tuned change in differentiation-order rho.

The figure set reports a *difference* between the frozen and fine-tuned
differentiation-order Spearman rho, but the archived statistics only test
positive association. This script adds:

1. A gate that reproduces the archived rho / layer-adjacency-accuracy (LAA)
   values from the per-cell npz arrays (no retraining, no GPU).
2. A paired, donor-stratified cell bootstrap for Delta-rho = rho_frozen -
   rho_adapted, per seed and for the three-seed mean. The stage-centroid axis
   is REFIT ON THE RESAMPLED CELLS inside every replicate (primary). A
   fixed full-sample-axis variant is also reported as a secondary check.
3. The same paired bootstrap for Delta-LAA = LAA_frozen - LAA_adapted.
4. A permutation null for LAA (stage labels permuted, class counts preserved),
   plus the analytic i.i.d. expectation.
5. Optional k-NN variants of LAA to show the metric is not an artefact of a
   single neighbour.

Contrasts: frozen vs naive (seeds 42/123/456) is primary; frozen vs CAFT
(seeds 42/123/456) is secondary. The ordinal (CORN) arms have archived
per-cell probabilities but no archived embeddings, so Delta-rho for
ordinal - naive is NOT computable from the archived arrays.

The metric definitions are imported from ``corneal_fm.evaluation`` and are
re-implemented here only in a numerically identical, faster form for the
bootstrap loops (equivalence is asserted at run time). No existing result
file is modified; outputs are written to
``results/paper1_vanzyl/claim_audit/``.

Usage:
    .venv/bin/python scripts/audit_claim_delta_rho.py [--n-boot 2000]
        [--n-perm 2000] [--seed 20261021] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Official metric implementations (imported, never modified).
from corneal_fm.evaluation import (  # noqa: E402
    layer_adjacency_accuracy,
    marker_gradient_correlation,
)

LAYER_ORDER = ["LSC", "TAC", "basal", "differentiated"]
_ORD = {c: i for i, c in enumerate(LAYER_ORDER)}
PERCELL_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
OUT_DIR = PROJECT_ROOT / "results" / "paper1_vanzyl" / "claim_audit"

FROZEN = "frozen_gf"
NAIVE_ARMS = ["naive_s42", "naive_s123", "naive_s456"]
CAFT_ARMS = ["caft_s42", "caft_s123", "caft_s456"]
ADAPTED_ARMS = NAIVE_ARMS + CAFT_ARMS
ALL_ARMS = [FROZEN] + ADAPTED_ARMS

GATE_TOL = 0.002
CANDIDATE_K = 256  # sufficient true-NN candidates for exact bootstrap search

EXPECTED_GATE = {
    "frozen_rho": 0.8064,
    "frozen_laa": 0.9665,
    "naive_rho_s42": -0.0424,
    "naive_rho_s123": -0.0360,
    "naive_rho_s456": 0.0539,
    "naive_rho_mean": -0.0082,
    "naive_laa_mean": 0.9768,
}


def _load_arm(arm: str) -> dict:
    path = PERCELL_DIR / f"percell_{arm}.npz"
    with np.load(path, allow_pickle=True) as data:
        return {
            "emb": np.asarray(data["emb_ext"], dtype=np.float64),
            "y": np.asarray(data["y_true_ext"], dtype=str),
            "donor": np.asarray(data["donor_ext"], dtype=str),
            "path": str(path),
        }


def _ordinals_of(labels: np.ndarray) -> np.ndarray:
    return np.array([_ORD[x] for x in labels], dtype=np.int64)


def _axis_full(emb: np.ndarray, ordinals: np.ndarray) -> np.ndarray:
    """Stage-centroid principal axis, oriented LSC -> differentiated.

    Mirrors the axis construction inside
    ``corneal_fm.evaluation.marker_gradient_correlation``.
    """
    present = [i for i in range(len(LAYER_ORDER)) if (ordinals == i).any()]
    centroids = np.vstack([emb[ordinals == i].mean(axis=0) for i in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[0]
    first = emb[ordinals == present[0]].mean(axis=0)
    last = emb[ordinals == present[-1]].mean(axis=0)
    if (last - first) @ axis < 0:
        axis = -axis
    return axis


def _rho_refit(emb: np.ndarray, ordinals: np.ndarray) -> float:
    """Spearman rho with the centroid axis refit on ``emb`` itself.

    Numerically identical to ``marker_gradient_correlation``; duplicated only
    to avoid rebuilding Python objects inside the bootstrap loop. Equivalence
    is asserted in ``main``.
    """
    present = [i for i in range(len(LAYER_ORDER)) if (ordinals == i).any()]
    if len(present) < 2:
        return float("nan")
    axis = _axis_full(emb, ordinals)
    rho, _ = spearmanr(emb @ axis, ordinals)
    return float(rho)


def _rho_fixed(emb: np.ndarray, ordinals: np.ndarray, axis: np.ndarray) -> float:
    """Spearman rho using a fixed (full-sample) stage-centroid axis."""
    rho, _ = spearmanr(emb @ axis, ordinals)
    return float(rho)


def _build_candidates(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True K nearest neighbours (self excluded) for every cell.

    Candidate lists make each bootstrap replicate an O(n*K) exact search for
    the nearest *drawn* cell instead of an O(n^2) full recomputation. With
    K=256 the probability that no candidate is drawn in a 63%-draw bootstrap
    is (0.37)**256 ~ 1e-110, so the result is exact for practical purposes;
    this is verified against sklearn on random replicates.
    """
    nn = NearestNeighbors(n_neighbors=CANDIDATE_K + 1, metric="euclidean").fit(emb)
    dist, idx = nn.kneighbors(emb)
    return idx[:, 1:].astype(np.int32), dist[:, 1:].astype(np.float32)


def _prep_arm(emb: np.ndarray, ordinals: np.ndarray, axis_fixed: np.ndarray) -> dict:
    """Per-arm arrays reused across bootstrap replicates."""
    class_idx = [np.flatnonzero(ordinals == c) for c in range(len(LAYER_ORDER))]
    return {
        "emb": emb,
        "class_idx": class_idx,
        "emb_class": [emb[ci] for ci in class_idx],
        "pt_fixed": emb @ axis_fixed,
    }


def _rho_refit_fast(
    mult: np.ndarray, idxrep: np.ndarray, prep: dict, ord_rank: np.ndarray
) -> float:
    """Paired-bootstrap rho with the stage-centroid axis refit in-replicate.

    Equivalent to ``_rho_refit(emb[idxrep], ordinals[idxrep])`` but computes
    class centroids from cell multiplicities and projects with a single
    matrix-vector product, avoiding repeated 768-d gathers.
    """
    counts = np.array([mult[ci].sum() for ci in prep["class_idx"]])
    present = [c for c in range(len(LAYER_ORDER)) if counts[c] > 0]
    centroids = np.vstack(
        [(mult[prep["class_idx"][c]] @ prep["emb_class"][c]) / counts[c] for c in present]
    )
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[0]
    if (centroids[-1] - centroids[0]) @ axis < 0:
        axis = -axis
    pseudotime = (prep["emb"] @ axis)[idxrep]
    return float(np.corrcoef(rankdata(pseudotime), ord_rank)[0, 1])


def _rho_fixed_fast(idxrep: np.ndarray, prep: dict, ord_rank: np.ndarray) -> float:
    """Paired-bootstrap rho with the full-sample axis held fixed."""
    pseudotime = prep["pt_fixed"][idxrep]
    return float(np.corrcoef(rankdata(pseudotime), ord_rank)[0, 1])


def _laa_kth_fast(
    mult: np.ndarray, cand_idx: np.ndarray, ordinals: np.ndarray, k: int = 1
) -> float:
    """k-th nearest drawn-neighbour LAA, scanning candidates column-wise.

    Equivalent to ``_laa_from_multiplicity_vectorized``; stops once every
    cell has k drawn candidates, so it touches only a handful of columns.
    """
    n = len(mult)
    drawn = mult > 0
    rows = np.arange(n)
    seen = np.zeros(n, dtype=np.int64)
    kth = np.full(n, -1, dtype=np.int64)
    for j in range(cand_idx.shape[1]):
        cj = cand_idx[:, j]
        valid = drawn[cj] & (cj != rows)
        if not valid.any():
            continue
        seen[valid] += 1
        newly = valid & (seen == k)
        if newly.any():
            kth[newly] = cj[newly]
        if (kth >= 0).all():
            break
    kth = np.where(kth >= 0, kth, rows)
    adj = (np.abs(ordinals - ordinals[kth]) <= 1).astype(np.float64)
    adj[mult >= 2] = 1.0
    return float((mult * adj).sum() / mult.sum())


def _laa_from_multiplicity_vectorized(
    mult: np.ndarray,
    cand_idx: np.ndarray,
    ordinals: np.ndarray,
    k: int = 1,
) -> float:
    """LAA on a bootstrap multiset described by cell multiplicities.

    k-th nearest drawn neighbour only (k=1 is the paper's LAA). Self is
    excluded; duplicates are retained (a cell drawn >= 2 times has a
    distance-0 duplicate neighbour, hence a trivially adjacent neighbour).
    """
    n = len(mult)
    drawn = mult > 0
    rows = np.arange(n)
    valid = drawn[cand_idx] & (cand_idx != rows[:, None])
    counts = np.cumsum(valid, axis=1)
    pos = np.argmax(counts >= k, axis=1)
    has = counts[:, -1] >= k
    kth = cand_idx[rows, pos]
    kth = np.where(has, kth, rows)
    adj = (np.abs(ordinals - ordinals[kth]) <= 1).astype(np.float64)
    adj[mult >= 2] = 1.0
    return float((mult * adj).sum() / mult.sum())


def _laa_fraction_of_k(
    mult: np.ndarray,
    cand_idx: np.ndarray,
    ordinals: np.ndarray,
    k: int,
) -> float:
    """Mean over cells of the fraction of k nearest drawn neighbours adjacent."""
    n = len(mult)
    drawn = mult > 0
    rows = np.arange(n)
    valid = drawn[cand_idx[:, :k]] & (cand_idx[:, :k] != rows[:, None])
    total = valid.sum(axis=1)
    adj = (np.abs(ordinals[:, None] - ordinals[cand_idx[:, :k]]) <= 1) & valid
    adj_sum = adj.sum(axis=1)
    frac = np.where(total > 0, adj_sum / np.maximum(total, 1), 1.0)
    return float((mult * frac).sum() / mult.sum())


def _laa_subset_fast(
    mask: np.ndarray, cand_idx: np.ndarray, ordinals: np.ndarray, k: int = 1
) -> float:
    """k-NN LAA on the unique selected cells (no duplicate neighbours).

    Used for the duplicate-free robustness check: the with-replacement
    bootstrap multiset is collapsed to its unique cells before the metric is
    evaluated, so distance-zero duplicate neighbours cannot inflate LAA.
    """
    n = len(mask)
    rows = np.arange(n)
    seen = np.zeros(n, dtype=np.int64)
    kth = np.full(n, -1, dtype=np.int64)
    for j in range(cand_idx.shape[1]):
        cj = cand_idx[:, j]
        valid = mask[cj] & (cj != rows)
        if not valid.any():
            continue
        seen[valid] += 1
        newly = valid & (seen == k)
        if newly.any():
            kth[newly] = cj[newly]
        if not (mask & (kth < 0)).any():
            break
    kth = np.where(kth >= 0, kth, rows)
    adj = (np.abs(ordinals - ordinals[kth]) <= 1).astype(np.float64)
    sel = np.flatnonzero(mask)
    return float(adj[sel].mean())


def _analytic_expectation(counts: np.ndarray) -> dict:
    """i.i.d. and finite-population expectations of adjacent-label probability."""
    n = int(counts.sum())
    p = counts / n
    # Specified cross-check: sum_i p_i (p_{i-1} + p_i + p_{i+1}).
    iid = float(
        sum(p[i] * sum(p[j] for j in (i - 1, i, i + 1) if 0 <= j < len(p)) for i in range(len(p)))
    )
    # Finite-population version: the neighbour of a cell is never that cell.
    finite = 0.0
    for i in range(len(p)):
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(p):
                if i == j:
                    finite += counts[i] * (counts[i] - 1) / (n * (n - 1))
                else:
                    finite += counts[i] * counts[j] / (n * (n - 1))
    return {"iid": iid, "finite_population": float(finite)}


def _draw_mult(structure: list[tuple[str, np.ndarray]], rng: np.random.Generator, n: int):
    mult = np.zeros(n, dtype=np.int64)
    for _, idx in structure:
        pick = rng.choice(idx, size=len(idx), replace=True)
        np.add.at(mult, pick, 1)
    return mult


def _summary_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "ci95_percentile": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "frac_le_zero": float(np.mean(values <= 0.0)),
    }


def _loo_contrast(arms: dict[str, dict], donor: str, adapted: list[str]) -> dict:
    """Leave-one-donor-out rho / LAA and the frozen-vs-adapted differences."""
    mask = arms[FROZEN]["donor"] == donor
    frozen_rho = marker_gradient_correlation(
        arms[FROZEN]["emb"][mask], arms[FROZEN]["y"][mask], layer_order=LAYER_ORDER
    )
    frozen_laa = layer_adjacency_accuracy(
        arms[FROZEN]["y"][mask], arms[FROZEN]["emb"][mask], layer_order=LAYER_ORDER
    )
    per_seed = {}
    for arm in adapted:
        rho_value = marker_gradient_correlation(
            arms[arm]["emb"][mask], arms[arm]["y"][mask], layer_order=LAYER_ORDER
        )
        laa_value = layer_adjacency_accuracy(
            arms[arm]["y"][mask], arms[arm]["emb"][mask], layer_order=LAYER_ORDER
        )
        per_seed[arm] = {
            "rho": rho_value,
            "laa": laa_value,
            "delta_rho": frozen_rho - rho_value,
            "delta_laa": frozen_laa - laa_value,
        }
    mean_rho = float(np.mean([per_seed[a]["rho"] for a in adapted]))
    mean_laa = float(np.mean([per_seed[a]["laa"] for a in adapted]))
    return {
        "n_cells": int(mask.sum()),
        "frozen_rho": frozen_rho,
        "frozen_laa": frozen_laa,
        "per_seed": per_seed,
        "pooled_three_seed_mean": {
            "rho": mean_rho,
            "laa": mean_laa,
            "delta_rho": frozen_rho - mean_rho,
            "delta_laa": frozen_laa - mean_laa,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--quick", action="store_true", help="small run for smoke test")
    args = parser.parse_args()
    if args.quick:
        args.n_boot = min(args.n_boot, 50)
        args.n_perm = min(args.n_perm, 50)

    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] per-cell arrays")
    arms = {arm: _load_arm(arm) for arm in ALL_ARMS}
    n = len(arms[FROZEN]["y"])
    for arm in ALL_ARMS:
        assert len(arms[arm]["y"]) == n
        assert np.array_equal(arms[arm]["y"], arms[FROZEN]["y"]), arm
        assert np.array_equal(arms[arm]["donor"], arms[FROZEN]["donor"]), arm

    y = arms[FROZEN]["y"]
    donor = arms[FROZEN]["donor"]
    donors = sorted(np.unique(donor))
    donor_sizes = {d: int(np.sum(donor == d)) for d in donors}
    structure = [(d, np.flatnonzero(donor == d)) for d in donors]
    ordinals = _ordinals_of(y)
    class_counts = {c: int(np.sum(y == c)) for c in LAYER_ORDER}

    # --- Gate: official metrics from the npz arrays ----------------------
    print("[gate] reproducing archived rho / LAA")
    point = {}
    for arm in ALL_ARMS:
        point[arm] = {
            "rho": marker_gradient_correlation(
                arms[arm]["emb"], arms[arm]["y"], layer_order=LAYER_ORDER
            ),
            "laa": layer_adjacency_accuracy(
                arms[arm]["y"], arms[arm]["emb"], layer_order=LAYER_ORDER
            ),
        }
    naive_rho_mean = float(np.mean([point[a]["rho"] for a in NAIVE_ARMS]))
    naive_laa_mean = float(np.mean([point[a]["laa"] for a in NAIVE_ARMS]))
    caft_rho_mean = float(np.mean([point[a]["rho"] for a in CAFT_ARMS]))
    caft_laa_mean = float(np.mean([point[a]["laa"] for a in CAFT_ARMS]))
    gate_checks = {
        "frozen_rho": (point[FROZEN]["rho"], EXPECTED_GATE["frozen_rho"]),
        "frozen_laa": (point[FROZEN]["laa"], EXPECTED_GATE["frozen_laa"]),
        "naive_rho_s42": (point["naive_s42"]["rho"], EXPECTED_GATE["naive_rho_s42"]),
        "naive_rho_s123": (point["naive_s123"]["rho"], EXPECTED_GATE["naive_rho_s123"]),
        "naive_rho_s456": (point["naive_s456"]["rho"], EXPECTED_GATE["naive_rho_s456"]),
        "naive_rho_mean": (naive_rho_mean, EXPECTED_GATE["naive_rho_mean"]),
        "naive_laa_mean": (naive_laa_mean, EXPECTED_GATE["naive_laa_mean"]),
    }
    gate = {
        "tolerance": GATE_TOL,
        "checks": {
            k: {
                "reproduced": float(v[0]),
                "expected": float(v[1]),
                "abs_diff": float(abs(v[0] - v[1])),
                "passed": bool(abs(v[0] - v[1]) <= GATE_TOL),
            }
            for k, v in gate_checks.items()
        },
    }
    gate["passed"] = bool(all(c["passed"] for c in gate["checks"].values()))
    print(f"[gate] passed={gate['passed']}")
    for k, c in gate["checks"].items():
        print(
            f"       {k}: {c['reproduced']:.6f} vs {c['expected']:.6f} (diff {c['abs_diff']:.2e})"
        )
    if not gate["passed"]:
        print("[gate] FAILED - refusing to compute downstream statistics")
        (OUT_DIR / "delta_rho_laa.json").write_text(
            json.dumps(
                {
                    "meta": {"generated_utc": datetime.now(UTC).isoformat(timespec="seconds")},
                    "gate": gate,
                },
                indent=2,
            )
        )
        return 2

    # --- Fixed full-sample axes + fast-path equivalence ------------------
    print("[verify] fast metric paths vs official functions")
    cand = {arm: _build_candidates(arms[arm]["emb"]) for arm in ALL_ARMS}
    axes = {arm: _axis_full(arms[arm]["emb"], ordinals) for arm in ALL_ARMS}
    preps = {arm: _prep_arm(arms[arm]["emb"], ordinals, axes[arm]) for arm in ALL_ARMS}
    ones = np.ones(n, dtype=np.int64)
    fast_checks = {}
    for arm in ALL_ARMS:
        r_refit = _rho_refit(arms[arm]["emb"], ordinals)
        r_fixed = _rho_fixed(arms[arm]["emb"], ordinals, axes[arm])
        l_fast = _laa_from_multiplicity_vectorized(ones, cand[arm][0], ordinals, k=1)
        fast_checks[arm] = {
            "rho_refit_vs_official": float(abs(r_refit - point[arm]["rho"])),
            "rho_fixed_vs_official": float(abs(r_fixed - point[arm]["rho"])),
            "laa_fast_vs_official": float(abs(l_fast - point[arm]["laa"])),
        }
        assert fast_checks[arm]["rho_refit_vs_official"] < 1e-8, fast_checks[arm]
        assert fast_checks[arm]["rho_fixed_vs_official"] < 1e-8, fast_checks[arm]
        assert fast_checks[arm]["laa_fast_vs_official"] < 1e-12, fast_checks[arm]
    print(
        f"[verify] max diffs: rho_refit="
        f"{max(v['rho_refit_vs_official'] for v in fast_checks.values()):.2e} "
        f"rho_fixed={max(v['rho_fixed_vs_official'] for v in fast_checks.values()):.2e} "
        f"laa={max(v['laa_fast_vs_official'] for v in fast_checks.values()):.2e}"
    )

    # Verify the replicate-level fast paths against the reference
    # implementations on random multiplicities (never in the reported path).
    check_rng = np.random.default_rng(args.seed + 2)
    replicate_diffs = {"rho_refit": 0.0, "rho_fixed": 0.0, "laa": 0.0}
    for _ in range(5):
        m_check = _draw_mult(structure, check_rng, n)
        ir_check = np.repeat(np.arange(n), m_check)
        ord_rank_check = rankdata(ordinals[ir_check])
        mask_check = m_check > 0
        uniq_check = np.flatnonzero(mask_check)
        for arm in ALL_ARMS:
            ref_refit = _rho_refit(arms[arm]["emb"][ir_check], ordinals[ir_check])
            ref_fixed = _rho_fixed(arms[arm]["emb"][ir_check], ordinals[ir_check], axes[arm])
            ref_laa = _laa_from_multiplicity_vectorized(m_check, cand[arm][0], ordinals, k=1)
            ref_laa_subset = layer_adjacency_accuracy(
                y[uniq_check], arms[arm]["emb"][uniq_check], layer_order=LAYER_ORDER
            )
            replicate_diffs["rho_refit"] = max(
                replicate_diffs["rho_refit"],
                abs(ref_refit - _rho_refit_fast(m_check, ir_check, preps[arm], ord_rank_check)),
            )
            replicate_diffs["rho_fixed"] = max(
                replicate_diffs["rho_fixed"],
                abs(ref_fixed - _rho_fixed_fast(ir_check, preps[arm], ord_rank_check)),
            )
            replicate_diffs["laa"] = max(
                replicate_diffs["laa"],
                abs(ref_laa - _laa_kth_fast(m_check, cand[arm][0], ordinals, k=1)),
            )
            replicate_diffs["laa_subset"] = max(
                replicate_diffs.get("laa_subset", 0.0),
                abs(ref_laa_subset - _laa_subset_fast(mask_check, cand[arm][0], ordinals)),
            )
    for key, val in replicate_diffs.items():
        assert val < 1e-10, (key, val)
    print(f"[verify] replicate-level max diffs: {replicate_diffs}")

    # --- Paired donor-stratified cell bootstrap --------------------------
    print(f"[bootstrap] {args.n_boot} replicates, seed={args.seed}")
    rng = np.random.default_rng(args.seed)
    n_boot = args.n_boot
    boot = {
        arm: {
            "rho_refit": np.empty(n_boot),
            "rho_fixed": np.empty(n_boot),
            "laa": np.empty(n_boot),
            "laa_dedup": np.empty(n_boot),
        }
        for arm in ALL_ARMS
    }
    t_boot = time.time()
    for b in range(n_boot):
        mult = _draw_mult(structure, rng, n)
        idxrep = np.repeat(np.arange(n), mult)
        ord_rank = rankdata(ordinals[idxrep])
        mask = mult > 0
        for arm in ALL_ARMS:
            # Axis REFIT on the resampled cells for this arm (primary).
            boot[arm]["rho_refit"][b] = _rho_refit_fast(mult, idxrep, preps[arm], ord_rank)
            # Fixed full-sample axis for this arm (secondary).
            boot[arm]["rho_fixed"][b] = _rho_fixed_fast(idxrep, preps[arm], ord_rank)
            # k-NN LAA with duplicates retained (as resampled) ...
            boot[arm]["laa"][b] = _laa_kth_fast(mult, cand[arm][0], ordinals, k=1)
            # ... and with the multiset collapsed to its unique cells.
            boot[arm]["laa_dedup"][b] = _laa_subset_fast(mask, cand[arm][0], ordinals, k=1)
        if (b + 1) % 200 == 0:
            print(f"          {b + 1}/{n_boot} ({time.time() - t_boot:.0f}s)")
    boot_s = time.time() - t_boot

    def paired_block(metric_key: str, adapted: list[str], metric: str) -> dict:
        out = {}
        for arm in adapted:
            delta = boot[FROZEN][metric_key] - boot[arm][metric_key]
            point_value = point[FROZEN][metric] - point[arm][metric]
            stats = _summary_stats(delta)
            stats["point_estimate"] = float(point_value)
            stats["point_estimate_inside_ci"] = bool(
                stats["ci95_percentile"][0] <= point_value <= stats["ci95_percentile"][1]
            )
            out[arm] = stats
        pooled_delta = boot[FROZEN][metric_key] - np.mean(
            [boot[a][metric_key] for a in adapted], axis=0
        )
        pooled_point = point[FROZEN][metric] - float(np.mean([point[a][metric] for a in adapted]))
        stats = _summary_stats(pooled_delta)
        stats["point_estimate"] = float(pooled_point)
        stats["point_estimate_inside_ci"] = bool(
            stats["ci95_percentile"][0] <= pooled_point <= stats["ci95_percentile"][1]
        )
        out["pooled_three_seed_mean"] = stats
        return out

    delta_rho = {
        "primary_naive_refit_axis": paired_block("rho_refit", NAIVE_ARMS, "rho"),
        "secondary_naive_fixed_full_sample_axis": paired_block("rho_fixed", NAIVE_ARMS, "rho"),
        "secondary_caft_refit_axis": paired_block("rho_refit", CAFT_ARMS, "rho"),
        "secondary_caft_fixed_full_sample_axis": paired_block("rho_fixed", CAFT_ARMS, "rho"),
    }
    delta_laa = {
        "primary_naive": paired_block("laa", NAIVE_ARMS, "laa"),
        "secondary_caft": paired_block("laa", CAFT_ARMS, "laa"),
    }
    delta_laa_dedup = {
        "primary_naive": paired_block("laa_dedup", NAIVE_ARMS, "laa"),
        "secondary_caft": paired_block("laa_dedup", CAFT_ARMS, "laa"),
    }
    loo = {
        "naive": {d: _loo_contrast(arms, d, NAIVE_ARMS) for d in donors},
        "caft": {d: _loo_contrast(arms, d, CAFT_ARMS) for d in donors},
    }

    # --- LAA permutation null --------------------------------------------
    print(f"[null] {args.n_perm} label permutations per arm")
    rng_perm = np.random.default_rng(args.seed + 1)
    null = {}
    for arm in ALL_ARMS:
        nn1 = cand[arm][0][:, 0]
        vals = np.empty(args.n_perm)
        for p in range(args.n_perm):
            perm = rng_perm.permutation(ordinals)
            vals[p] = float(np.mean(np.abs(perm - perm[nn1]) <= 1))
        observed = point[arm]["laa"]
        null[arm] = {
            "observed": float(observed),
            "null_mean": float(vals.mean()),
            "null_sd": float(vals.std(ddof=1)),
            "null_p95": float(np.percentile(vals, 95)),
            "null_max": float(vals.max()),
            "observed_minus_null_mean": float(observed - vals.mean()),
            "observed_minus_null_p95": float(observed - np.percentile(vals, 95)),
            "upper_tail_p": float((np.sum(vals >= observed) + 1) / (args.n_perm + 1)),
        }
    analytic = _analytic_expectation(np.array([class_counts[c] for c in LAYER_ORDER]))

    # --- Optional k-NN LAA variants (full data, descriptive) -------------
    print("[knn] k = 1, 5, 10")
    knn_variants = {}
    for arm in ALL_ARMS:
        cidx = cand[arm][0]
        entry = {}
        for k in (1, 5, 10):
            entry[f"kth_neighbour_k{k}"] = _laa_from_multiplicity_vectorized(
                ones, cidx, ordinals, k=k
            )
            entry[f"fraction_of_k_k{k}"] = _laa_fraction_of_k(ones, cidx, ordinals, k=k)
        knn_variants[arm] = entry

    wall = time.time() - t_start

    payload = {
        "meta": {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "script": "scripts/audit_claim_delta_rho.py",
            "purpose": "paired Delta-rho / Delta-LAA audit of the frozen -> fine-tuned claim",
            "n_cells": int(n),
            "n_donors": len(donors),
            "donor_sizes": donor_sizes,
            "class_counts": class_counts,
            "layer_order": LAYER_ORDER,
            "n_boot": n_boot,
            "n_perm": args.n_perm,
            "bootstrap_seed": args.seed,
            "permutation_seed": args.seed + 1,
            "resampling_unit": (
                "cell, stratified by donor: cells drawn with replacement within each "
                "donor, donor sizes preserved. This is a stratified CELL bootstrap, "
                "not a donor-level bootstrap (only two donors)."
            ),
            "axis_handling": {
                "primary": (
                    "stage centroids refit on the resampled cells inside every "
                    "replicate, per arm, then cells projected onto that axis"
                ),
                "secondary": (
                    "stage-centroid axis estimated once on the full sample, per arm, "
                    "and held fixed across replicates"
                ),
            },
            "candidate_neighbours_K": CANDIDATE_K,
            "wall_clock_seconds": float(wall),
            "bootstrap_seconds": float(boot_s),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "input_files": {arm: arms[arm]["path"] for arm in ALL_ARMS},
            "fast_path_checks": fast_checks,
            "replicate_fast_path_checks_max_abs_diff": replicate_diffs,
            "not_computable": {
                "delta_rho_ordinal_minus_naive": (
                    "NOT computable from archived arrays: the ordinal (CORN) arms "
                    "store per-cell logits/probabilities (artifacts/paper1_corn_boundary/"
                    "percell_lam0.0_*.npz) but no CLS embeddings, so rho cannot be "
                    "recomputed without re-running inference."
                )
            },
        },
        "gate": gate,
        "point_estimates": {
            **point,
            "naive_pooled_three_seed_mean": {"rho": naive_rho_mean, "laa": naive_laa_mean},
            "caft_pooled_three_seed_mean": {"rho": caft_rho_mean, "laa": caft_laa_mean},
        },
        "paired_delta_rho": {
            "definition": "rho_frozen - rho_adapted, evaluated on the same resampled cells",
            "contrasts": delta_rho,
            "leave_one_donor_out": loo,
        },
        "paired_delta_laa": {
            "definition": "LAA_frozen - LAA_adapted, evaluated on the same resampled cells",
            "contrasts": delta_laa,
            "diagnostic": (
                "k-NN LAA is not Frechet-differentiable under a with-replacement "
                "cell bootstrap: resampled duplicates are distance-0 neighbours and "
                "are trivially adjacent, which inflates LAA in every replicate and "
                "shrinks/offsets the difference distribution. point_estimate_inside_ci "
                "records whether this artefact breaks the percentile interval; see "
                "supplementary_dedup_estimator for the duplicate-free interval."
            ),
            "supplementary_dedup_estimator": {
                "definition": (
                    "same paired resamples collapsed to their unique cells before "
                    "evaluating LAA (duplicates removed, so no distance-0 neighbours)"
                ),
                "contrasts": delta_laa_dedup,
            },
        },
        "laa_null": {
            "method": (
                "stage labels permuted across cells with class counts preserved; "
                "the nearest-neighbour graph is held fixed, so each permutation "
                "re-labels the same edges"
            ),
            "analytic_expectation": analytic,
            "analytic_expectation_note": (
                "iid = sum_i p_i (p_{i-1}+p_i+p_{i+1}) with p_i the observed class "
                "proportions (the requested cross-check); finite_population replaces "
                "self-pairs with n_i(n_i-1)/(n(n-1))"
            ),
            "per_arm": null,
        },
        "laa_knn_variants": {
            "note": (
                "kth_neighbour_kK: fraction of cells whose K-th nearest neighbour "
                "(self excluded) is adjacent; K=1 is the paper's LAA. "
                "fraction_of_k_kK: mean over cells of the fraction of the K nearest "
                "neighbours that are adjacent."
            ),
            "per_arm": knn_variants,
        },
    }

    out_json = OUT_DIR / "delta_rho_laa.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out_json} in {wall:.1f}s")
    (OUT_DIR / "_payload.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
