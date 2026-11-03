#!/usr/bin/env python
"""Donor-stratified paired bootstrap of Δrho between two arms.

The differentiation-gradient metric is
``corneal_fm.evaluation.marker_gradient_correlation``: Spearman rho between the
CLS-embedding projection on the per-layer-centroid principal axis (oriented
LSC -> differentiated) and the 4-stage ordinal. Its axis is estimated from the
evaluated cells, so a paired comparison must re-fit the axis inside every
bootstrap replicate, for both arms, on the SAME resampled cells.

Design (as specified 2026-10-21):
  * resampling is stratified by donor: within each donor, draw
    ``n_donor`` cells with replacement, then concatenate the donors in a fixed
    order (donor_1 then donor_2 for GSE153515);
  * one index vector per replicate is applied to BOTH arms -> paired;
  * ``n_boot = 2000``, ``seed = 42`` (same for every pair, documented here);
  * every replicate calls the locked metric function unchanged (no fast-path
    reimplementation), so the estimator is exactly the one used in the Tier-B
    records.

Arm sources on GSE153515 (14,775 epithelial cells, 2 donors, identical cell
order — verified element-wise against ``y_true_ext`` of the Tier-B per-cell
files):

    frozen       artifacts/paper1_tier_b_eval/percell_frozen_gf.npz           (emb_ext)
    naive        artifacts/paper1_tier_b_eval/percell_naive_s{seed}.npz      (emb_ext)
    ordinal      artifacts/archive/paper1_bio_validation/ordinal_emb_lam0.0_seed{seed}.npz (emb)
    epi4_ce      results/paper1_vanzyl/epi4_ce/seed{seed}/percell_epi4_{seed}.npz (emb_ext)

Usage:
    python scripts/paired_drho_bootstrap.py \
        --pairs frozen-naive,frozen-ordinal,naive-ordinal \
        --out results/paper1_vanzyl/epi4_ce/paired_drho_reference_arms.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

SEEDS = (42, 123, 456)
DEFAULT_N_BOOT = 2000
DEFAULT_BOOT_SEED = 42
TIER_B = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
ORDINAL_EMB = PROJECT_ROOT / "artifacts" / "archive" / "paper1_bio_validation"
EPI4_DIR = PROJECT_ROOT / "results" / "paper1_vanzyl" / "epi4_ce"


def load_arm(arm: str, seed: int | None = None, variant: str | None = None) -> dict:
    """CLS embeddings + labels + donor ids for one arm's external evaluation.

    ``variant`` selects the epi4-CE sub-directory (``drop_nonepi`` /
    ``mask_nonepi``) for ``arm="epi4_ce"``.
    """
    if arm == "frozen":
        doc = np.load(TIER_B / "percell_frozen_gf.npz", allow_pickle=True)
        return {
            "emb": np.asarray(doc["emb_ext"], dtype=float),
            "labels": np.asarray(doc["y_true_ext"]).astype(str),
            "donors": np.asarray(doc["donor_ext"]).astype(str),
            "source": str((TIER_B / "percell_frozen_gf.npz").relative_to(PROJECT_ROOT)),
        }
    if arm == "naive":
        path = TIER_B / f"percell_naive_s{seed}.npz"
        doc = np.load(path, allow_pickle=True)
        return {
            "emb": np.asarray(doc["emb_ext"], dtype=float),
            "labels": np.asarray(doc["y_true_ext"]).astype(str),
            "donors": np.asarray(doc["donor_ext"]).astype(str),
            "source": str(path.relative_to(PROJECT_ROOT)),
        }
    if arm == "ordinal":
        path = ORDINAL_EMB / f"ordinal_emb_lam0.0_seed{seed}.npz"
        doc = np.load(path, allow_pickle=True)
        # Donor ids / labels come from the Tier-B per-cell file; the stored
        # labels are asserted element-wise identical first (never reordered).
        ref = np.load(TIER_B / f"percell_naive_s{seed}.npz", allow_pickle=True)
        labels = np.asarray(ref["y_true_ext"]).astype(str)
        stored = np.asarray(doc["labels"]).astype(str)
        if not np.array_equal(stored, labels):
            raise ValueError(
                f"ordinal embeddings for seed {seed} are not in the Tier-B cell "
                f"order (labels differ element-wise); refusing to pair"
            )
        return {
            "emb": np.asarray(doc["emb"], dtype=float),
            "labels": labels,
            "donors": np.asarray(ref["donor_ext"]).astype(str),
            "source": str(path.relative_to(PROJECT_ROOT)),
        }
    if arm in ("drop_nonepi", "mask_nonepi"):
        variant, arm = arm, "epi4_ce"
    if arm == "epi4_ce":
        root = EPI4_DIR / variant if variant else EPI4_DIR
        path = root / f"seed{seed}" / f"percell_epi4_{seed}.npz"
        if not path.is_file() and variant:  # pre-relocation flat layout
            path = EPI4_DIR / f"seed{seed}" / f"percell_epi4_{seed}.npz"
        doc = np.load(path, allow_pickle=True)
        return {
            "emb": np.asarray(doc["emb_ext"], dtype=float),
            "labels": np.asarray(doc["y_true_ext"]).astype(str),
            "donors": np.asarray(doc["donor_ext"]).astype(str),
            "source": str(path.relative_to(PROJECT_ROOT)),
        }
    raise ValueError(f"unknown arm {arm!r}")


def check_pairing(a: dict, b: dict) -> bool:
    """Element-wise label identity and donor identity (same cell order)."""
    return (
        a["emb"].shape == b["emb"].shape
        and np.array_equal(a["labels"], b["labels"])
        and np.array_equal(a["donors"], b["donors"])
    )


def donor_strata(donors: np.ndarray) -> list[np.ndarray]:
    """Indices of each donor, in sorted donor-name order."""
    return [np.flatnonzero(donors == d) for d in np.unique(donors)]


def paired_delta(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    labels: np.ndarray,
    donors: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
    return_replicates: bool = False,
):
    """Point rho per arm + donor-stratified paired bootstrap of the difference.

    One index vector per replicate (drawn within donor strata) is applied to
    both arms; the metric re-fits each arm's centroid axis inside the replicate.

    With ``return_replicates=True`` the replicate array is returned alongside the
    summary dict, so a caller can pool replicates across seeds and record that
    pooled interval explicitly (see :func:`run_pair`).
    """
    rho_a = float(marker_gradient_correlation(emb_a, labels))
    rho_b = float(marker_gradient_correlation(emb_b, labels))
    strata = donor_strata(donors)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        parts = [rng.integers(0, len(idx), len(idx)) for idx in strata]
        idx = np.concatenate([s[p] for s, p in zip(strata, parts)])
        # donors are concatenated in a fixed (sorted) order, so the labels and
        # donor array of the replicate follow the same construction
        deltas[i] = marker_gradient_correlation(
            emb_a[idx], labels[idx]
        ) - marker_gradient_correlation(emb_b[idx], labels[idx])
    result = {
        "rho_a": rho_a,
        "rho_b": rho_b,
        "delta": rho_a - rho_b,
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
        "frac_boot_positive": float((deltas > 0).mean()),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "resampling": "donor-stratified, cells with replacement within donor",
        "ci_method": "per-seed percentile of that seed's own replicates",
    }
    if return_replicates:
        return result, deltas
    return result


def run_pair(arm_a: str, arm_b: str, seeds=SEEDS, n_boot: int = DEFAULT_N_BOOT,
             seed: int = DEFAULT_BOOT_SEED) -> dict:
    """Per-seed Δrho for a pair (frozen has no seed -> shared across seeds).

    Each seed's replicates are also pooled across seeds and summarised by their
    own percentile interval, stored under ``pooled`` so an averaged interval in
    the manuscript can be recomputed from this file.
    """
    per_seed = {}
    skipped = []
    replicates = {}
    for s in seeds:
        ea = load_arm(arm_a, None if arm_a == "frozen" else s)
        eb = load_arm(arm_b, None if arm_b == "frozen" else s)
        if not check_pairing(ea, eb):
            skipped.append({"seed": s, "reason": "labels/donors/shape differ"})
            continue
        res, deltas = paired_delta(ea["emb"], eb["emb"], ea["labels"],
                                   ea["donors"], n_boot=n_boot, seed=seed,
                                   return_replicates=True)
        res["n_cells"] = int(ea["emb"].shape[0])
        res["source_a"] = ea["source"]
        res["source_b"] = eb["source"]
        per_seed[str(s)] = res
        replicates[str(s)] = deltas
    ds = [v["delta"] for v in per_seed.values()]
    out = {
        "pair": f"{arm_a} - {arm_b}",
        "sign_convention": "delta = rho(arm_a) - rho(arm_b)",
        "per_seed": per_seed,
        "skipped": skipped,
        "n_boot": n_boot,
        "boot_seed": seed,
    }
    if ds:
        out["mean_delta"] = float(np.mean(ds))
        out["sd_delta"] = float(np.std(ds, ddof=1)) if len(ds) > 1 else None
        out["sd_kind"] = "sample SD (ddof=1) over seeds"
        # Averaged interval: the arithmetic mean of the per-seed bootstrap
        # bounds. This is the convention the manuscript quotes for the
        # frozen - naive contrast (mean 0.8145, bounds 0.7369/0.8942, printed
        # as 0.737-0.894); it is stored explicitly so an averaged interval in
        # the text can be recomputed from this file by hand.
        lowers = [v["ci_lower"] for v in per_seed.values()]
        uppers = [v["ci_upper"] for v in per_seed.values()]
        out["averaged_ci95"] = {
            "ci95": [float(np.mean(lowers)), float(np.mean(uppers))],
            "method": "arithmetic mean of the per-seed bootstrap bounds "
                      "(ci_lower and ci_upper of each seed)",
            "n_seeds": len(lowers),
            "n_boot_per_seed": int(n_boot),
            "note": "an interval for the three-seed average under the "
                    "convention used for the frozen - naive contrast; it is "
                    "not a coverage-calibrated interval for the mean",
        }
        if len(ds) > 1:
            from scipy import stats as _stats
            mean_d = float(np.mean(ds))
            sd_d = float(np.std(ds, ddof=1))
            se_d = sd_d / np.sqrt(len(ds))
            df_d = len(ds) - 1
            tcrit = float(_stats.t.ppf(0.975, df_d))
            out["across_seed"] = {
                "mean_delta": mean_d,
                "sd_delta": sd_d,
                "se_delta": float(se_d),
                "df": df_d,
                "t_critical": tcrit,
                "ci95": [mean_d - tcrit * se_d, mean_d + tcrit * se_d],
                "method": "two-sided 95% t-interval on the seed means "
                          "(t = mean of per-seed deltas, SE = sample SD over "
                          "seeds / sqrt(n_seeds))",
            }
    if replicates:
        pooled = np.concatenate([replicates[key] for key in sorted(replicates)])
        out["pooled"] = {
            "mean_delta": float(np.mean(pooled)),
            "ci95": [float(np.percentile(pooled, 2.5)),
                     float(np.percentile(pooled, 97.5))],
            "method": "percentile of the pooled donor-stratified paired "
                      "replicates: the per-seed replicate arrays are "
                      "concatenated and the 2.5th/97.5th percentiles taken "
                      "once over the union",
            "resampling": "one index vector per replicate, cells drawn with "
                          "replacement within donor and applied to both arms",
            "n_boot_per_seed": int(n_boot),
            "n_seeds_pooled": len(replicates),
            "n_replicates": int(pooled.size),
            "bound_convention": "2.5th and 97.5th percentiles of the pooled "
                                "replicate distribution",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="frozen-naive,frozen-ordinal,naive-ordinal",
                    help="Comma-separated ARM_A-ARM_B pairs (delta = rho_A - rho_B).")
    ap.add_argument("--out", default=str(EPI4_DIR / "paired_drho_reference_arms.json"))
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_BOOT_SEED)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                    help="Comma-separated model seeds to pair.")
    args = ap.parse_args()
    model_seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    result = {
        "metric": "corneal_fm.evaluation.marker_gradient_correlation "
                  "(per-layer-centroid principal axis, LSC -> differentiated)",
        "external_test": "GSE153515, 14,775 epithelial cells, 2 donors",
        "pairing_verified": "labels and donor arrays element-wise identical "
                            "across arms (see per-pair n_cells and skipped)",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pairs": [],
    }
    for token in args.pairs.split(","):
        a, b = [t.strip() for t in token.split("-")]
        print(f"[pair] {a} - {b} ...", flush=True)
        result["pairs"].append(
            run_pair(a, b, seeds=model_seeds, n_boot=args.n_boot, seed=args.seed)
        )
        print(f"[pair] {a} - {b} done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    for p in result["pairs"]:
        for s, row in p["per_seed"].items():
            print(f"  {p['pair']:22s} seed {s}: Δρ={row['delta']:+.4f} "
                  f"[{row['ci_lower']:+.4f},{row['ci_upper']:+.4f}]")
        if "mean_delta" in p:
            print(f"  {p['pair']:22s} mean   : Δρ={p['mean_delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
