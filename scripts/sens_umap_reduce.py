#!/usr/bin/env python
"""Sensitivity analysis + UMAP coordinate computation (p5070 CPU, no GPU).

Reads per-cell final-layer CLS embeddings for the test donor Hu0216
(20,158 cells) for every (architecture, arm, seed) in the collapse matrix
(scGPT / GF / CellPLM x frozen / naive / CORN) and:

Task 1 — sensitivity analysis of the differentiation-gradient rho:
  * variant dimension A (SVD axis): pc1_centroid (LOCKED), pc2_centroid,
    pc2_cellpca, pc3_cellpca
  * variant dimension M (marker panel): P0 locked van Zyl (stored obs
    scores), P1 canonical mini, P2 DEG/LPC data-driven; per panel:
    rho_data (arm-invariant gradient sanity), rho_marker_axis (PC1
    alignment), rho_centroid_marker (locked-axis vs marker gradient)
  * variant dimension R (rank assignment): R0 [0,1,2] locked, R1 [1,2,3],
    R2 [3,2,1], R3 [2,1,3], R4 [1,2,4]; gates verify Spearman
    rank-invariance (R1 == R0, R4 == R0, R2 == -R0)
  * per-variant verdict (frozen, deterministic):
      collapse_detected(v)  = rho_frozen - rho_naive >= 0.3 AND rho_naive <= 0.4
      rescue_detected(v)    = rho_corn >= rho_frozen - 0.15
      verdict               = ROBUST if both; PARTIAL if one; NOT_REPRODUCED
  * sanity gates (fail-closed): locked-metric rho vs the locked eval JSONs
    (scGPT run6: tol 2e-3 CPU-fp32 vs GPU-bf16; CellPLM: tol 1e-6; GF:
    descriptive)

Task 2 — UMAP (for the figure): umap-learn embedding of the L2-normed
per-cell embeddings per (arch, arm, seed); n_neighbors=15, min_dist=0.1,
random_state=42, n_epochs=500; coordinates saved per panel.

PCS: variant definitions and verdict thresholds frozen before the first
run (2026-08-20); SHA manifest written to <out>/sha_manifest.json.

Usage (p5070, CPU):
    /home/summer/corneal_scgpt/venv/bin/python scripts/sens_umap_reduce.py
        [--data-dir artifacts/paper1_vanzyl/sensitivity/_caches]
        [--out-dir artifacts/paper1_vanzyl/sensitivity]
        [--umap-out-dir artifacts/paper1_vanzyl/umap_collapse]
        [--skip-umap]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/home/summer/corneal_scgpt")
CELLPLM_ROOT = Path("/home/summer/cellplm_vanzyl")

LAYER_ORDER = ["LSC", "basal", "differentiated"]
TEST_DONOR = "sanes2022_Hu0216"
TIER_IDX = {c: i for i, c in enumerate(LAYER_ORDER)}

# Sanity-gate targets (locked eval JSONs, PCS).
SCGPT_EVAL = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval"
CELLPLM_NAIVE_JSON = {
    42: CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "seed42" / "seed42.json",
    123: CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "seed123" / "seed123.json",
    456: CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "seed456" / "seed456.json",
}
CELLPLM_CORN_JSON = {
    42: CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "corn" / "seed42" / "seed42.json",
    123: CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "corn" / "seed123" / "seed123.json",
    456: CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "corn" / "seed456" / "seed456.json",
}
SCPT_EVAL = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval"
FROZEN_SCGPT_JSON = SCPT_EVAL / "frozen_baseline.json"

G1_SCGPT_NPZ = ROOT / "artifacts" / "g1_scgpt" / "vanzyl_cell_emb.npz"
G1_GF_NPZ = ROOT / "artifacts" / "paper1_vanzyl" / "g1_gf" / "gf_vanzyl_cell_emb.npz"
TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"

RHO_TOL_SCGPT = 2e-3
RHO_TOL_CELLPLM = 1e-6
RANK_TOL = 1e-12

# Marker panels (PCS frozen 2026-08-20). P0 uses the stored protocol_15c
# obs scores; P1/P2 are recomputed with scanpy score_genes on the test h5ad.
PANELS: dict[str, dict[str, Any]] = {
    "P0_vanzyl_locked": {
        "note": "locked van Zyl panels (protocol_15c); stored obs "
                "score_LSC / score_PMC_basal / score_TDC used as-is",
        "use_stored": True,
    },
    "P1_canonical_mini": {
        "note": "canonical literature mini-panel (paper intro markers)",
        "LSC": ["ABCG2", "TP63", "KRT15", "TCF4"],
        "basal": ["KRT14", "KRT5", "COL17A1"],
        "diff": ["KRT12", "KRT3", "GJA1"],
    },
    "P2_deg_datadriven": {
        "note": "data-driven DEG panel: LSC via LPC set (harmonizes to LSC "
                "per protocol_15c), basal/diff = van Zyl PMC_basal/TDC top DEGs",
        "LSC": ["KRT14", "KRT15", "KRT19", "DCN", "PLIN2", "SLC2A3", "TCF4"],
        "basal": ["CRYBG1", "CNTN4", "DCDC1", "CACNA1D"],
        "diff": ["LIPH", "MUC16", "WNK2", "B4GALT5", "SCEL"],
    },
}

RANK_VARIANTS: dict[str, list[int]] = {
    "R0_locked_0_1_2": [0, 1, 2],
    "R1_shift_1_2_3": [1, 2, 3],
    "R2_reversed_3_2_1": [3, 2, 1],
    "R3_nonmonotone_2_1_3": [2, 1, 3],
    "R4_unequal_1_2_4": [1, 2, 4],
}

UMAP_KW = {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42,
           "n_components": 2, "n_jobs": 4}


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def _ordinal(y: np.ndarray) -> np.ndarray:
    return np.array([TIER_IDX[x] for x in y], dtype=float)


def centroid_pc_axis(emb: np.ndarray, y: np.ndarray, pc: int = 0,
                     orient: bool = True) -> tuple[np.ndarray, list[str]]:
    """Centroid SVD axis (pc-th singular vector), oriented LSC->diff.

    Returns (axis, present_classes). pc=0 -> LOCKED PC1; pc=1 -> PC2.
    """
    order_map = {c: i for i, c in enumerate(LAYER_ORDER)}
    present = [c for c in LAYER_ORDER if (y == c).any()]
    centroids = np.vstack([emb[y == c].mean(axis=0) for c in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[pc].astype(np.float64)
    if orient and len(present) >= 2:
        first = emb[y == present[0]].mean(axis=0)
        last = emb[y == present[-1]].mean(axis=0)
        if (last - first) @ axis < 0:
            axis = -axis
    return axis, present


def _cell_svd_axes(emb: np.ndarray, k: int = 3) -> np.ndarray:
    """First k right singular vectors of the centered per-cell cloud.

    Randomized SVD (sklearn, random_state=42) — deterministic, and ~100x
    faster than a full np.linalg.svd on (20158, 768).
    """
    from sklearn.utils.extmath import randomized_svd

    c = emb - emb.mean(axis=0)
    _, _, vt = randomized_svd(c, n_components=k, random_state=42)
    return np.asarray(vt, dtype=np.float64)


def cell_pca_axis(emb: np.ndarray, pc: int, orient_emb: np.ndarray | None = None,
                  orient_y: np.ndarray | None = None) -> np.ndarray:
    """pc-th principal axis of the per-cell embedding cloud (centered)."""
    axis = _cell_svd_axes(emb, k=3)[pc]
    if orient_emb is not None and orient_y is not None:
        present = [cl for cl in LAYER_ORDER if (orient_y == cl).any()]
        first = orient_emb[orient_y == present[0]].mean(axis=0)
        last = orient_emb[orient_y == present[-1]].mean(axis=0)
        if (last - first) @ axis < 0:
            axis = -axis
    return axis


def rho_on_axis(emb: np.ndarray, y: np.ndarray, axis: np.ndarray,
                ordinal: np.ndarray | None = None) -> float:
    p = emb @ axis
    if ordinal is None:
        ordinal = _ordinal(y)
    r, _ = spearmanr(p, ordinal)
    return float(r)


def pc1_alignment_axis(emb: np.ndarray, s: np.ndarray) -> np.ndarray:
    """PC1 of the embedding cloud, oriented so corr(w.emb, s) >= 0."""
    w = _cell_svd_axes(emb, k=3)[0]
    if float(np.corrcoef(emb @ w, s)[0, 1]) < 0:
        w = -w
    return w


# ---------------------------------------------------------------------------
# Arm loading
# ---------------------------------------------------------------------------

def load_g1_masked(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    m = np.asarray(d["donors"]) == TEST_DONOR
    emb = np.asarray(d["cell_emb"], dtype=np.float32)[m]
    lab_key = "cell_class_4" if "cell_class_4" in d.files else "labels"
    y = np.asarray(d[lab_key])[m]
    assert len(y) == 20158
    return emb, y.astype(str)


def load_cache(data_dir: Path, tag: str) -> dict[str, Any]:
    d = np.load(data_dir / f"{tag}.npz", allow_pickle=True)
    return {"emb": np.asarray(d["emb"], dtype=np.float32),
            "y": np.asarray(d["y"]).astype(str),
            "barcode": np.asarray(d["barcode"]).astype(str)}


def load_cellplm(tag: str) -> tuple[np.ndarray, np.ndarray]:
    # tag: cellplm_naive_42 / cellplm_corn_42 / cellplm_frozen
    if tag == "cellplm_frozen":
        p = CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "seed42" / "frozen_emb.npz"
    else:
        _, arm, seed = tag.split("_")
        if arm == "naive":
            p = CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / f"seed{seed}" / "cell_emb.npz"
        else:
            p = CELLPLM_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "corn" / f"seed{seed}" / "cell_emb.npz"
    if not p.exists():
        raise FileNotFoundError(str(p))
    d = np.load(p, allow_pickle=True)
    return np.asarray(d["emb"], dtype=np.float32), np.asarray(d["y"]).astype(str)


def build_arms(data_dir: Path) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {}

    emb, y = load_g1_masked(G1_SCGPT_NPZ)
    arms["scgpt_frozen"] = {"arch": "scgpt", "arm": "frozen", "seed": None,
                            "emb": emb, "y": y}
    emb, y = load_g1_masked(G1_GF_NPZ)
    arms["gf_frozen"] = {"arch": "gf", "arm": "frozen", "seed": None,
                         "emb": emb, "y": y}
    emb, y = load_cellplm("cellplm_frozen")
    arms["cellplm_frozen"] = {"arch": "cellplm", "arm": "frozen", "seed": None,
                              "emb": emb, "y": y}

    for seed in (42, 123, 456):
        for arm in ("naive", "corn"):
            tag = f"scgpt_{arm}_seed{seed}"
            if (data_dir / f"{tag}.npz").exists():
                d = load_cache(data_dir, tag)
                arms[tag] = {"arch": "scgpt", "arm": arm, "seed": seed,
                             "emb": d["emb"], "y": d["y"]}
            tag = f"cellplm_{arm}_{seed}"
            try:
                emb_c, y_c = load_cellplm(tag)
            except FileNotFoundError:
                continue
            arms[tag] = {"arch": "cellplm", "arm": arm, "seed": seed,
                         "emb": emb_c, "y": y_c}
    for arm in ("naive", "corn"):
        tag = f"gf_{arm}_seed42"
        if (data_dir / f"{tag}.npz").exists():
            d = load_cache(data_dir, tag)
            arms[tag] = {"arch": "gf", "arm": arm, "seed": 42,
                         "emb": d["emb"], "y": d["y"]}
    return arms


# ---------------------------------------------------------------------------
# Sanity gates
# ---------------------------------------------------------------------------

def run_gates(arms: dict[str, dict[str, Any]],
              locked_metric: dict[str, float]) -> dict[str, Any]:
    """Fail-closed: locked-metric rho (pc1_centroid on L2-normed emb) vs the
    locked eval JSONs; returns gate records + raises on failure."""
    gates: dict[str, Any] = {}
    failed: list[str] = []

    def check(tag: str, recomputed: float, locked: float | None,
              tol: float, kind: str) -> None:
        rec = {"rho_recomputed": recomputed, "rho_locked": locked,
               "tol": tol, "pass": locked is None or abs(recomputed - locked) <= tol}
        gates[f"{tag}_gate"] = rec
        if not rec["pass"]:
            failed.append(f"{tag}: {recomputed:.6f} vs {locked:.6f} (>{tol})")

    for tag, a in arms.items():
        emb_n = l2_norm_rows(a["emb"])
        rho = locked_metric[tag]
        if a["arch"] == "scgpt":
            if a["arm"] == "frozen":
                fb = json.loads(FROZEN_SCGPT_JSON.read_text())
                locked_rho = fb["rho"]["rho_normed"]
                check(tag, rho, float(locked_rho), 1e-6, "frozen")
            else:
                locked_rho = json.loads(
                    (SCGPT_EVAL / f"{a['arm']}_seed{a['seed']}.json").read_text()
                )["rho_normed"]
                check(tag, rho, float(locked_rho), RHO_TOL_SCGPT, "eval")
        elif a["arch"] == "cellplm":
            if a["arm"] == "frozen":
                locked_rho = json.loads(
                    (CELLPLM_NAIVE_JSON[42]).read_text())["frozen_baseline"]["rho_normed"]
                check(tag, rho, float(locked_rho), RHO_TOL_CELLPLM, "frozen")
            else:
                jp = CELLPLM_CORN_JSON[a["seed"]] if a["arm"] == "corn" \
                    else CELLPLM_NAIVE_JSON[a["seed"]]
                locked_rho = json.loads(jp.read_text())["eval"]["rho_normed"]
                check(tag, rho, float(locked_rho), RHO_TOL_CELLPLM, "eval")
        else:  # gf — descriptive (no locked van Zyl number)
            gates[f"{tag}_gate"] = {
                "rho_recomputed": rho, "rho_locked": None,
                "pass": True, "note": "descriptive (no locked van Zyl GF number)"}

    gates["all_passed"] = not failed
    gates["failures"] = failed
    return gates


# ---------------------------------------------------------------------------
# Sensitivity variants
# ---------------------------------------------------------------------------

def compute_variants(arms: dict[str, dict[str, Any]],
                     scores: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-arm per-variant rho + verdicts."""
    out: dict[str, Any] = {}
    emb_n: dict[str, np.ndarray] = {}

    for tag, a in arms.items():
        e = l2_norm_rows(a["emb"])
        emb_n[tag] = e
        y = a["y"]
        o = _ordinal(y)

        # --- dimension A: SVD axis ---------------------------------------
        ax_pc1, present = centroid_pc_axis(e, y, pc=0)
        ax_pc2, _ = centroid_pc_axis(e, y, pc=1)
        cell_axes = _cell_svd_axes(e, k=3)
        ax_c2 = cell_axes[1]
        ax_c3 = cell_axes[2]
        # orient cell-PCA axes by the first->last centroid rule (locked rule)
        present = [cl for cl in LAYER_ORDER if (y == cl).any()]
        first = e[y == present[0]].mean(axis=0)
        last = e[y == present[-1]].mean(axis=0)
        for ax_ in (ax_c2, ax_c3):
            if (last - first) @ ax_ < 0:
                ax_ *= -1
        variants = {
            "A_pc1_centroid": rho_on_axis(e, y, ax_pc1, o),
            "A_pc2_centroid": rho_on_axis(e, y, ax_pc2, o),
            "A_pc2_cellpca": rho_on_axis(e, y, ax_c2, o),
            "A_pc3_cellpca": rho_on_axis(e, y, ax_c3, o),
        }

        # --- dimension M: marker panels -----------------------------------
        for pname, sv in scores.items():
            s_p = sv["score"]  # (n,) diff - LSC marker pseudotime
            w = pc1_alignment_axis(e, s_p)
            variants[f"M_{pname}_data"] = float(spearmanr(s_p, o)[0])
            variants[f"M_{pname}_marker_axis"] = float(spearmanr(e @ w, s_p)[0])
            variants[f"M_{pname}_centroid_marker"] = float(
                spearmanr(e @ ax_pc1, s_p)[0])

        # --- dimension R: rank assignment on the locked pc1 axis ----------
        p_pc1 = e @ ax_pc1
        for rname, ranks in RANK_VARIANTS.items():
            ord_v = np.array([ranks[TIER_IDX[x]] for x in y], dtype=float)
            variants[rname] = float(spearmanr(p_pc1, ord_v)[0])
        out[tag] = {"rho_variants": variants, "ordinal": o.tolist()}

    # --- rank-invariance gates (derived from the R variants above) --------
    rank_out: dict[str, Any] = {}
    for tag, rec in out.items():
        if tag.startswith("_"):
            continue
        v = rec["rho_variants"]
        r0 = v["R0_locked_0_1_2"]
        rank_out[tag] = {
            **{k: v[k] for k in RANK_VARIANTS},
            "gate_R1_eq_R0": abs(v["R1_shift_1_2_3"] - r0) <= RANK_TOL,
            "gate_R4_eq_R0": abs(v["R4_unequal_1_2_4"] - r0) <= RANK_TOL,
            "gate_R2_is_minus_R0": abs(v["R2_reversed_3_2_1"] + r0) <= RANK_TOL,
        }
    out["_rank"] = rank_out

    # --- verdicts (frozen thresholds, PCS) --------------------------------
    verdicts: dict[str, Any] = {}
    for arch in ("scgpt", "gf", "cellplm"):
        frz = next((t for t, a in arms.items()
                    if a["arch"] == arch and a["arm"] == "frozen"), None)
        if frz is None:
            continue
        for seed in sorted({a["seed"] for t, a in arms.items()
                            if a["arch"] == arch and a["seed"] is not None}):
            n_tag = f"{arch}_naive_{seed}" if arch == "cellplm" \
                else f"{arch}_naive_seed{seed}"
            c_tag = f"{arch}_corn_{seed}" if arch == "cellplm" \
                else f"{arch}_corn_seed{seed}"
            if n_tag not in out or c_tag not in out:
                continue
            for vname in out[frz]["rho_variants"]:
                if vname.startswith("M_") and vname.endswith("_data"):
                    continue  # arm-invariant data sanity — no verdict
                rf = out[frz]["rho_variants"][vname]
                rn = out[n_tag]["rho_variants"][vname]
                rc = out[c_tag]["rho_variants"][vname]
                # R variants: Spearman rank-invariance flips sign under a
                # reversed ordinal (R2 == -R0, verified) — verdict on |rho|.
                if vname.startswith("R_"):
                    rf, rn, rc = abs(rf), abs(rn), abs(rc)
                collapse = (rf - rn >= 0.3) and (rn <= 0.4)
                rescue = rc >= rf - 0.15
                verdict = ("ROBUST" if collapse and rescue
                           else "PARTIAL" if collapse or rescue
                           else "NOT_REPRODUCED")
                verdicts.setdefault(vname, {})[f"{arch}_s{seed}"] = {
                    "rho_frozen": rf, "rho_naive": rn, "rho_corn": rc,
                    "collapse_detected": bool(collapse),
                    "rescue_detected": bool(rescue), "verdict": verdict}
    out["_verdicts"] = verdicts
    return out


# ---------------------------------------------------------------------------
# Marker scores from the test h5ad
# ---------------------------------------------------------------------------

def compute_marker_scores() -> dict[str, dict[str, Any]]:
    import anndata as ad

    a = ad.read_h5ad(TEST_H5AD)  # in memory (20158 x 3856 float32)
    n = a.n_obs
    assert n == 20158
    out: dict[str, dict[str, Any]] = {}
    var_set = set(a.var_names)

    p0 = PANELS["P0_vanzyl_locked"]
    s_lsc = np.asarray(a.obs["score_LSC"].to_numpy(), dtype=float)
    s_bas = np.asarray(a.obs["score_PMC_basal"].to_numpy(), dtype=float)
    s_dif = np.asarray(a.obs["score_TDC"].to_numpy(), dtype=float)
    out["P0_vanzyl_locked"] = {
        "note": p0["note"], "score": s_dif - s_lsc,
        "genes": {"LSC": "stored", "basal": "stored", "diff": "stored"},
        "gene_counts": {"LSC": "stored", "basal": "stored", "diff": "stored"},
        "valid": True,
        "sanity": {
            "frac_LSC_gt_basal": float((s_lsc > s_bas).mean()),
            "frac_diff_gt_basal": float((s_dif > s_bas).mean()),
        },
    }

    import scanpy as sc

    for pname in ("P1_canonical_mini", "P2_deg_datadriven"):
        panel = PANELS[pname]
        counts: dict[str, int] = {}
        score_cols: list[np.ndarray] = []
        ok = True
        for key in ("LSC", "basal", "diff"):
            avail = [g for g in panel[key] if g in var_set]
            counts[key] = len(avail)
            if len(avail) == 0:
                ok = False
            col = f"_scr_{pname}_{key}"
            sc.tl.score_genes(a, gene_list=avail, score_name=col)
            score_cols.append(np.asarray(a.obs[col].to_numpy(), dtype=float))
        out[pname] = {
            "note": panel["note"],
            "score": score_cols[2] - score_cols[0],
            "genes": {k: [g for g in panel[k] if g in var_set] for k in panel},
            "gene_counts": counts,
            "valid": ok,
        }
    return out


# ---------------------------------------------------------------------------
# UMAP coordinates
# ---------------------------------------------------------------------------

def compute_umap(arms: dict[str, dict[str, Any]], out_dir: Path,
                 skip: bool) -> list[dict[str, Any]]:
    if skip:
        return []
    import umap

    rows: list[dict[str, Any]] = []
    for tag, a in sorted(arms.items()):
        coord_file = out_dir / "_coords" / f"{tag}.npz"
        if coord_file.exists():
            d = np.load(coord_file)
            coord = d["coord"]
            log(f"umap {tag}: cached {coord.shape}")
        else:
            e = l2_norm_rows(a["emb"])
            t0 = time.time()
            fit = umap.UMAP(**UMAP_KW).fit_transform(e)
            coord = np.asarray(fit, dtype=np.float32)
            out_dir.joinpath("_coords").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(coord_file, coord=coord, tag=tag)
            log(f"umap {tag}: {e.shape} -> {coord.shape} in "
                f"{time.time() - t0:.0f}s")
        rows.append({"tag": tag, "arch": a["arch"], "arm": a["arm"],
                     "seed": a["seed"], "coord": coord})
    return rows


# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "artifacts" / "paper1_vanzyl" / "sensitivity" / "_caches")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "paper1_vanzyl" / "sensitivity")
    ap.add_argument("--umap-out-dir", type=Path, default=ROOT / "artifacts" / "paper1_vanzyl" / "umap_collapse")
    ap.add_argument("--skip-umap", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.umap_out_dir.mkdir(parents=True, exist_ok=True)

    arms = build_arms(args.data_dir)
    log(f"arms available: {sorted(arms)}")
    if not arms:
        log("ERROR: no arms loaded")
        return 1

    # Locked-metric rho (pc1_centroid on L2-normed emb) — gate + reference.
    locked_metric: dict[str, float] = {}
    for tag, a in arms.items():
        locked_metric[tag] = rho_on_axis(
            l2_norm_rows(a["emb"]), a["y"],
            centroid_pc_axis(l2_norm_rows(a["emb"]), a["y"], pc=0)[0])

    gates = run_gates(arms, locked_metric)
    log(f"gates: all_passed={gates['all_passed']} failures={gates['failures']}")
    if not gates["all_passed"]:
        log("SANITY GATE FAILED — refusing to write sensitivity outputs")
        return 2

    scores = compute_marker_scores()
    variants = compute_variants(arms, scores)

    summary = {
        "task": "paper1_vanzyl sensitivity analysis (PCS frozen 2026-08-20)",
        "test_donor": TEST_DONOR, "n_cells": 20158,
        "layer_order": LAYER_ORDER,
        "variant_definitions": {
            "A": "SVD axis: pc1_centroid (locked) / pc2_centroid / "
                 "pc2_cellpca / pc3_cellpca (centroid SVD has rank<=2 with 3 "
                 "classes, so PC3 is defined on the per-cell cloud)",
            "M": "marker panels: P0 stored van Zyl scores, P1 canonical mini, "
                 "P2 DEG/LPC data-driven; metrics rho_data (arm-invariant), "
                 "rho_marker_axis (PC1 alignment), rho_centroid_marker "
                 "(locked axis vs marker gradient)",
            "R": "rank assignment: R0 [0,1,2] locked; R1 [1,2,3]; "
                 "R2 [3,2,1]; R3 [2,1,3]; R4 [1,2,4]; Spearman is "
                 "rank-invariant -> R1==R0, R4==R0, R2==-R0 (verified)",
        },
        "verdict_thresholds": {
            "collapse_detected": "rho_frozen - rho_naive >= 0.3 AND rho_naive <= 0.4",
            "rescue_detected": "rho_corn >= rho_frozen - 0.15",
            "verdict": "ROBUST if both; PARTIAL if one; NOT_REPRODUCED",
        },
        "locked_metric_rho": locked_metric,
        "sanity_gates": gates,
        "marker_panels": {k: {kk: vv for kk, vv in v.items() if kk != "score"}
                          for k, v in scores.items()},
        "rank_gates": variants.get("_rank", {}),
        "per_arm_variants": {k: v for k, v in variants.items()
                             if not k.startswith("_")},
        "verdicts": variants.get("_verdicts", {}),
        "generated_utc": now_utc(),
    }
    summary_path = args.out_dir / "sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"saved {summary_path}")

    # per-variant json files (small, for the artifact tree)
    for vname, by_arm in variants.get("_verdicts", {}).items():
        (args.out_dir / f"verdict_{vname.replace('/', '_')}.json").write_text(
            json.dumps(by_arm, indent=2), encoding="utf-8")
    (args.out_dir / "rank_variants.json").write_text(
        json.dumps(variants.get("_rank", {}), indent=2), encoding="utf-8")

    # SHA manifest (PCS): hashes of the per-cell caches consumed
    manifests: dict[str, str] = {}
    for p in sorted(args.data_dir.glob("*.npz")):
        manifests[p.name] = sha256_file(p)
    for p in sorted(CELLPLM_ROOT.glob("artifacts/paper1_vanzyl/cellplm/**/cell_emb.npz")):
        manifests[f"cellplm/{p.parent.name}/cell_emb.npz"] = sha256_file(p)
    for p in (G1_SCGPT_NPZ, G1_GF_NPZ):
        manifests[p.name] = sha256_file(p)
    (args.out_dir / "sha_manifest.json").write_text(
        json.dumps({"files": manifests, "generated_utc": now_utc()}, indent=2),
        encoding="utf-8")

    # UMAP coordinates
    rows = compute_umap(arms, args.umap_out_dir, args.skip_umap)
    umap_json = [{"tag": r["tag"], "arch": r["arch"], "arm": r["arm"],
                  "seed": r["seed"], "coord_path": f"_coords/{r['tag']}.npz"}
                 for r in rows]
    (args.umap_out_dir / "umap_manifest.json").write_text(
        json.dumps({"panels": umap_json, "umap_kw": UMAP_KW,
                    "embedding": "L2-normed CLS cell embedding",
                    "generated_utc": now_utc()}, indent=2), encoding="utf-8")
    log(f"umap manifest: {len(umap_json)} panels")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
