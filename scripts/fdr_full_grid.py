#!/usr/bin/env python
"""Task 2 — BH-FDR full grid over ALL rho comparisons (CPU).

Submission-completeness statistics: every rho comparison across
  * arms         frozen / naive / CORN / LoRA / CORN+LoRA
  * architectures GF / scGPT / CellPLM
  * datasets     van Zyl(GSE153515) / GSE299152 / GSE186433 / GSE155683
  * tissues      cornea / epidermis
gets a label-shuffle z (stored in artifacts, or computed locally from the
archived per-cell embeddings), a one-sided p = Phi(-z) (H1: rho > 0), and a
Benjamini-Hochberg q over the whole family.

PCS (frozen BEFORE running):
  * p-value rule: one-sided upper-tail from the label-shuffle z, p = Phi(-z)
    (z = (rho_obs - null_mean)/null_sd of the label-shuffle null; H1 rho > 0).
  * FDR method: Benjamini-Hochberg step-up at q=0.05 — the SAME bh_fdr()
    implementation as scripts/fdr_nulls_analysis.py (unchanged).
  * Family membership: one primary row per (arm x architecture x dataset x
    tissue) using the paper's headline/locked rho and ITS OWN label-shuffle z
    (stored where the artifact computed one; computed locally, n=100 seed 42,
    where the rho is reproducible from archived embeddings but z was not
    stored). Sensitivity checkpoints (best-val vs final, per-seed detail) are
    listed in the grid but NOT duplicated into the FDR family (no
    pseudo-replication).
  * Rows without any locally available z (GSE299152 GF arms; CellPLM naive /
    CORN; CORN+LoRA E5 still training on p5070) are listed in the full grid
    with status="z_unavailable_local" and EXCLUDED from the FDR family
    (disclosed in the spec + run.log; they get q=null).
  * Data: only read-only artifact JSONs / npz (input SHA-256 manifest
    frozen before any computation).

Outputs (artifacts/paper1_vanzyl/fdr/):
  pcs_frozen_spec.json    — frozen method + membership + input SHA manifest
  bh_fdr_table.json       — FDR family (comparison, rho, z, p, q, sig)
  full_grid.json          — ALL rho comparisons incl. pending rows
  run.log

Usage:
    <venv>/bin/python3 scripts/fdr_full_grid.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "fdr"

N_NULL_LOCAL = 100
NULL_SEED = 42
Q_THRESHOLD = 0.05

EPI_ORDER = ["basal", "suprabasal", "spinous", "differentiated"]
VZ_ORDER_3 = ["LSC", "basal", "differentiated"]


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_fdr(pvals: list[float], q: float = Q_THRESHOLD):
    """Benjamini-Hochberg step-up — identical to fdr_nulls_analysis.bh_fdr."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    q_sorted = sorted_p * m / np.arange(1, m + 1)
    q_stepup = np.minimum.accumulate(q_sorted[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = q_stepup
    return adj


def p_from_z(z: float) -> float:
    """One-sided upper-tail p from label-shuffle z (H1: rho > 0)."""
    return float(norm.sf(z))


def label_shuffle_z(emb: np.ndarray, labels: np.ndarray, order: list[str],
                    n: int = N_NULL_LOCAL, seed: int = NULL_SEED) -> tuple[float, float, float]:
    """Label-shuffle null of marker_gradient_correlation on this embedding.
    Returns (null_mean, null_sd, z) — z for the OBSERVED rho on the same emb.
    The caller verifies rho_obs == marker_gradient_correlation(emb, labels)."""
    from corneal_fm.evaluation import marker_gradient_correlation
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    obs = marker_gradient_correlation(emb, y, layer_order=order)
    out = np.empty(n)
    for i in range(n):
        perm = rng.permutation(len(y))
        out[i] = marker_gradient_correlation(emb, y[perm], layer_order=order)
    null_mean = float(out.mean())
    null_sd = float(out.std())
    z = (obs - null_mean) / (null_sd + 1e-12)
    return null_mean, null_sd, float(z)


def main() -> int:
    from corneal_fm.evaluation import marker_gradient_correlation

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run.log"
    log = open(log_path, "w", encoding="utf-8")

    def log_msg(msg: str) -> None:
        line = f"{_ts()} {msg}"
        print(line)
        log.write(line + "\n")

    # ------------------------------------------------------------------ PCS
    # Inputs (read-only artifacts) -> SHA manifest frozen BEFORE any compute.
    input_paths = [
        PROJECT_ROOT / "artifacts/paper1_fdr_nulls/null_controls.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/run6/eval/summary_vanzyl_6run.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/convergence/final_test_eval.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/cornea2/summary_cornea2.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/cross_tissue/final_test_eval.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/cross_tissue/epidermis_test_frozen_diag.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/loo/frozen_baseline_loo220.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_summary.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed42.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed123.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed456.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed789.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed2024.json",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed31415.json",
        PROJECT_ROOT / "artifacts/paper1_tier_b_eval_gse299152/summary.csv",
        PROJECT_ROOT / "artifacts/paper1_epidermis_eval/epidermis_rho_results.json",
        PROJECT_ROOT / "artifacts/paper1_epidermis_eval/embeddings.npz",
        PROJECT_ROOT / "artifacts/paper1_epidermis_eval_gse202352/gse202352_rho_results.json",
        PROJECT_ROOT / "artifacts/paper1_epidermis_eval_gse202352/embeddings.npz",
        PROJECT_ROOT / "artifacts/paper1_vanzyl/cellplm/seed42/frozen_emb.npz",
    ]
    sha_manifest = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in input_paths}

    spec = {
        "task": "BH-FDR full grid over all rho comparisons (Task 2)",
        "frozen_at": datetime.now(UTC).isoformat(),
        "p_value_rule": "one-sided upper-tail p = Phi(-z) from label-shuffle z "
                        "(H1: rho > 0); z = (rho_obs - null_mean)/null_sd",
        "fdr_method": "Benjamini-Hochberg step-up, q=0.05, "
                      "identical bh_fdr() to scripts/fdr_nulls_analysis.py",
        "family_membership": "one primary row per (arm x architecture x dataset "
                             "x tissue); headline rho with ITS OWN label-shuffle z; "
                             "sensitivity checkpoints listed in grid only "
                             "(no pseudo-replication); rows without locally "
                             "available z excluded from FDR family (disclosed)",
        "local_null": {"n": N_NULL_LOCAL, "seed": NULL_SEED,
                       "metric": "corneal_fm.evaluation.marker_gradient_correlation "
                                 "(unchanged)"},
        "q_threshold": Q_THRESHOLD,
        "input_sha256": sha_manifest,
    }
    with open(OUT_DIR / "pcs_frozen_spec.json", "w") as fh:
        json.dump(spec, fh, indent=1)
    log_msg("PCS spec + input SHA-256 manifest frozen -> pcs_frozen_spec.json")

    # ------------------------------------------------------------- assemble
    # 1) GF van Zyl (GSE153515 external test, 14,775 cells) — z stored in
    #    null_controls.json (n=500 label-shuffle per arm).
    nc = json.loads((PROJECT_ROOT / "artifacts/paper1_fdr_nulls/null_controls.json").read_text())
    nc_arm = {r["model"]: r for r in nc["results"]}
    gf_vz = [
        ("frozen",        "frozen",   nc_arm["frozen"]["rho_obs"],   nc_arm["frozen"]["label_z"]),
        ("naive",         "naive",    nc_arm["ce_naive"]["rho_obs"], nc_arm["ce_naive"]["label_z"]),
        ("corns_ordinal", "CORN",     nc_arm["corns_ordinal"]["rho_obs"], nc_arm["corns_ordinal"]["label_z"]),
    ]

    # 2) GF official 1-epoch (E3, 6 seeds) — per-seed z stored; family row uses
    #    the mean rho and the mean per-seed z (documented).
    e3_rhos, e3_zs = [], []  # (summary json exists for provenance; per-seed json used)
    for s in [42, 123, 456, 789, 2024, 31415]:
        d = json.loads((PROJECT_ROOT / f"artifacts/paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed{s}.json").read_text())
        e3_rhos.append(d["eval"]["rho_normed"])
        e3_zs.append(d["eval"]["null_z"])
    gf_1ep = ("GF official 1-epoch (E3, 6-seed mean)", float(np.mean(e3_rhos)),
              float(np.mean(e3_zs)))

    # 3) scGPT van Zyl (Hu0216, 20,158 cells) — run6 headline (3-seed means).
    run6 = json.loads((PROJECT_ROOT / "artifacts/paper1_vanzyl/run6/eval/summary_vanzyl_6run.json").read_text())
    agg = run6["aggregate"]
    scgpt_vz = [
        ("frozen", run6["frozen"]["rho"]["rho_normed"], run6["frozen"]["rho"]["null_z"]),
        ("naive",  agg["naive"]["rho_normed"]["mean"],  agg["naive"]["null_z"]["mean"]),
        ("CORN",   agg["ordinal"]["rho_normed"]["mean"], agg["ordinal"]["null_z"]["mean"]),
    ]
    # LoRA (E4, 2-seed final-epoch) — z recorded in lane.log (98.8 / 97.2).
    scgpt_vz.append(("LoRA", 0.8336, 98.0))

    # 4) scGPT GSE186433 / GSE155683 (cornea2) — frozen + naive best-val mean.
    c2 = json.loads((PROJECT_ROOT / "artifacts/paper1_vanzyl/cornea2/summary_cornea2.json").read_text())
    c2_rows = []
    for ds in ["GSE186433", "GSE155683"]:
        d = c2["datasets"][ds]
        bv = [d["seeds"][s]["best_val"] for s in ["s42", "s123"]]
        c2_rows.append((ds, "frozen", d["frozen"], d["frozen_z"]))
        c2_rows.append((ds, "naive", float(np.mean([b["rho_normed"] for b in bv])),
                        float(np.mean([b["null_z"] for b in bv]))))

    # 5) scGPT epidermis cross-tissue (GSE147424) — frozen + naive best-val mean.
    ct = json.loads((PROJECT_ROOT / "artifacts/paper1_vanzyl/cross_tissue/final_test_eval.json").read_text())
    ct_fb = ct["frozen_baseline"]["results"]["epidermis_test"]
    ct_naive_bv = ct["aggregate"]["best_val"]
    scgpt_epi = [
        ("frozen", ct_fb["rho_normed"], ct_fb["null_z"]),
        ("naive",  ct_naive_bv["rho_normed"]["mean"], ct_naive_bv["null_z"]["mean"]),
    ]

    # 6) GF epidermis GSE147424 (v14) + GSE202352 — rho stored, z NOT stored,
    #    but embeddings archived locally -> compute label-shuffle z (gate: rho
    #    must reproduce exactly).
    def load_epi(dataset_dir: str) -> dict:
        z = np.load(PROJECT_ROOT / dataset_dir / "embeddings.npz", allow_pickle=True)
        labels = np.array([EPI_ORDER[k] for k in z["ordinal"]])
        return {"z": z, "labels": labels}

    epi_computed = {}
    epi_extra = {}
    for ds, d in [("GSE147424", "artifacts/paper1_epidermis_eval"),
                  ("GSE202352", "artifacts/paper1_epidermis_eval_gse202352")]:
        res = json.loads((PROJECT_ROOT / d / "epidermis_rho_results.json").read_text()) \
            if ds == "GSE147424" else \
            json.loads((PROJECT_ROOT / d / "gse202352_rho_results.json").read_text())
        data = load_epi(d)
        rows = {}
        extra = []
        for row in res["results"]:
            name = row["checkpoint"]
            rho_stored = row["rho_L12_ordinal"]
            emb = data["z"][name].astype(float)
            rho_re = marker_gradient_correlation(emb, data["labels"], layer_order=EPI_ORDER)
            if abs(rho_re - rho_stored) > 1e-6:
                log_msg(f"GATE FAIL {ds} {name}: stored {rho_stored:.4f} != recompute {rho_re:.4f}")
                sys.exit(2)
            null_m, null_sd, zz = label_shuffle_z(emb, data["labels"], EPI_ORDER)
            arm = {"frozen": "frozen", "ce_naive": "naive",
                   "corns_ordinal": "CORN"}.get(name)
            if arm is not None:
                rows[arm] = {"rho": rho_re, "z": zz, "null_mean": null_m, "null_sd": null_sd}
            else:
                extra.append({"checkpoint": name, "rho": rho_re, "z": zz,
                              "null_mean": null_m, "null_sd": null_sd})
            log_msg(f"{ds} {name:14s} rho={rho_re:.4f} z={zz:+.2f} "
                    f"(null {null_m:+.4f} +/- {null_sd:.4f}, n={N_NULL_LOCAL})")
        epi_computed[ds] = rows
        epi_extra[ds] = extra

    # 7) CellPLM van Zyl frozen — embedding archived (emb+y, 3-tier) -> compute z.
    cp = np.load(PROJECT_ROOT / "artifacts/paper1_vanzyl/cellplm/seed42/frozen_emb.npz",
                 allow_pickle=True)
    cp_emb = cp["emb"].astype(float)
    cp_y = cp["y"].astype(str)
    cp_rho = marker_gradient_correlation(cp_emb, cp_y, layer_order=VZ_ORDER_3)
    cp_emb_n = cp_emb / (np.linalg.norm(cp_emb, axis=1, keepdims=True) + 1e-12)
    cp_rho_n = marker_gradient_correlation(cp_emb_n, cp_y, layer_order=VZ_ORDER_3)
    null_m, null_sd, cp_z = label_shuffle_z(cp_emb_n, cp_y, VZ_ORDER_3)
    log_msg(f"CellPLM frozen vanZyl rho_raw={cp_rho:.4f} rho_normed={cp_rho_n:.4f} "
            f"z={cp_z:+.2f} (null {null_m:+.4f} +/- {null_sd:.4f})")

    # 8) Pending rows (no locally available z):
    #    GSE299152 (GF) rho values from tier_b summary.csv; CellPLM naive/CORN
    #    from lane.log / merge references; CORN+LoRA E5 still training.
    gse299152 = {
        "frozen": 0.706565780997028,
        "naive": 0.6407,   # 3-seed mean 0.6167/0.6307/0.6746
        "CORN": 0.3501,    # ordinal lambda0 3-seed mean (0.2610/0.4566/0.3325)
    }
    pending = [
        {"arm": "frozen", "architecture": "GF", "dataset": "GSE299152", "tissue": "cornea",
         "rho": gse299152["frozen"], "note": "z not stored in tier_b results; "
         "per-cell embeddings not archived locally (GB10-run) -> z unavailable on CPU"},
        {"arm": "naive", "architecture": "GF", "dataset": "GSE299152", "tissue": "cornea",
         "rho": gse299152["naive"], "note": "same as above"},
        {"arm": "CORN", "architecture": "GF", "dataset": "GSE299152", "tissue": "cornea",
         "rho": gse299152["CORN"], "note": "same as above"},
        {"arm": "naive", "architecture": "CellPLM", "dataset": "van Zyl", "tissue": "cornea",
         "rho": -0.007, "note": "embeddings on p5070 (not synced); z unavailable locally"},
        {"arm": "CORN", "architecture": "CellPLM", "dataset": "van Zyl", "tissue": "cornea",
         "rho": 0.918, "note": "embeddings on p5070 (not synced); z unavailable locally"},
        {"arm": "CORN+LoRA", "architecture": "scGPT", "dataset": "van Zyl", "tissue": "cornea",
         "rho": None, "note": "E5 training on p5070 (launched 2026-08-20 23:40, "
         "ETA ~2026-08-21 12:30); results not available at freeze time"},
    ]

    # ------------------------------------------------------- FDR family rows
    # Each: (comparison label, architecture, dataset, tissue, rho, z, source)
    family = []
    for arm, label, rho, zz in gf_vz:
        family.append((f"GF vanZyl {label}", "GF", "van Zyl", "cornea", rho, zz,
                       "null_controls.json (n=500 label-shuffle)"))
    family.append((f"GF vanZyl {gf_1ep[0]}", "GF", "van Zyl", "cornea", gf_1ep[1],
                   gf_1ep[2], "gf_official_1epoch per-seed z mean"))
    for label, rho, zz in scgpt_vz:
        family.append((f"scGPT vanZyl {label}", "scGPT", "van Zyl", "cornea", rho, zz,
                       "run6 eval / E4 lane.log"))
    for ds, arm, rho, zz in c2_rows:
        family.append((f"scGPT {ds} {arm}", "scGPT", ds, "cornea", rho, zz,
                       "summary_cornea2.json"))
    for arm, rho, zz in scgpt_epi:
        family.append((f"scGPT epidermis {arm}", "scGPT", "GSE147424", "epidermis", rho, zz,
                       "cross_tissue final_test_eval.json"))
    for arm, v in epi_computed["GSE147424"].items():
        family.append(("GF epidermis " + arm, "GF", "GSE147424", "epidermis", v["rho"], v["z"],
                       "computed locally (n=100 seed 42, rho gate PASS)"))
    for arm, v in epi_computed["GSE202352"].items():
        family.append(("GF epidermis " + arm, "GF", "GSE202352", "epidermis", v["rho"], v["z"],
                       "computed locally (n=100 seed 42, rho gate PASS)"))
    family.append(("CellPLM vanZyl frozen", "CellPLM", "van Zyl", "cornea", cp_rho_n, cp_z,
                   "computed locally (n=100 seed 42)"))

    # ---- BH-FDR over the family
    pvals = [p_from_z(zz) for *_hdr, rho, zz, src in family]
    qs = bh_fdr(pvals)
    rows_out = []
    for (label, arch, ds, tissue, rho, zz, src), p, q in zip(family, pvals, qs):
        rows_out.append({
            "comparison": label, "architecture": arch, "dataset": ds, "tissue": tissue,
            "rho": round(float(rho), 4), "z": round(float(zz), 2),
            "p": float(p), "q": float(q), "significant_0.05": bool(q < Q_THRESHOLD),
            "z_source": src,
        })
    n_sig = sum(r["significant_0.05"] for r in rows_out)

    # ---- grid: family + sensitivity + pending
    grid = {
        "task": "FDR full grid — all rho comparisons",
        "family_size": len(rows_out),
        "n_significant_0.05": n_sig,
        "q_threshold": Q_THRESHOLD,
        "family": rows_out,
        "sensitivity_checkpoints_not_in_family": [
            {"comparison": "scGPT vanZyl naive convergence best-val",
             "rho": 0.4395, "z": 48.2, "note": "same arm as run6 naive; "
             "listed here to avoid pseudo-replication"},
            {"comparison": "scGPT vanZyl naive convergence final-epoch",
             "rho": 0.5023, "z": 54.4, "note": "same arm as run6 naive"},
            {"comparison": "scGPT vanZyl LoRA best-val",
             "rho": 0.8568, "z": 107.55, "note": "E4 best-val checkpoint"},
            {"comparison": "GF vanZyl naive per-seed detail",
             "rho": -0.0424, "z": -8.53, "note": "s42 3-epoch reference"},
        ],
        "five_negative_arms_outside_task_arms": [
            {"comparison": f"GF epidermis {ds} {c['checkpoint']}",
             "rho": c["rho"], "z": c["z"],
             "note": "not in task arm list (frozen/naive/CORN/LoRA/CORN+LoRA); "
                     "recorded here with computed z, NOT in the FDR family"}
            for ds in ["GSE147424", "GSE202352"] for c in epi_extra[ds]
        ],
        "pending_rows": pending,
    }
    # bh_fdr_table.json = the FDR family only (submission table);
    # full_grid.json   = the complete grid (family + sensitivity + five-negative
    #                    + pending), so "全网格" is preserved in one file.
    bh_table = {
        "task": "BH-FDR table over the rho family (Task 2)",
        "family_size": len(rows_out),
        "n_significant_0.05": n_sig,
        "q_threshold": Q_THRESHOLD,
        "p_value_rule": spec["p_value_rule"],
        "fdr_method": spec["fdr_method"],
        "rows": rows_out,
    }
    (OUT_DIR / "bh_fdr_table.json").write_text(json.dumps(bh_table, indent=1), encoding="utf-8")
    (OUT_DIR / "full_grid.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")

    # ---- console table
    log_msg(f"=== BH-FDR full grid: family={len(rows_out)} significant={n_sig} (q<{Q_THRESHOLD}) ===")
    for r in sorted(rows_out, key=lambda x: x["q"]):
        sig = "*" if r["significant_0.05"] else ""
        log_msg(f"{r['comparison']:28s} rho={r['rho']:+.4f} z={r['z']:+8.2f} "
                f"p={r['p']:.2e} q={r['q']:.2e} {sig}")
    log_msg(f"saved bh_fdr_table.json + full_grid.json ({len(rows_out)} family rows, "
            f"{len(grid['pending_rows'])} pending)")
    log_msg("DONE")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
