#!/usr/bin/env python
"""Protocol 02f — van Zyl LOO split (second test donor, approved supplementary).

Leave-one-donor-out replicate of the PREREG §8 split with test donor
sanes2022_Hu0220 (FROZEN judgment: second-largest 3-tier donor, 16,705
cells; closest in size to the primary test donor Hu0216, 20,158).
Train = the other 4 donors (45,935 cells). Analysis set rule unchanged
(leiden != '0' AND cell_class_4 in {LSC, basal, differentiated}).

Val split mirrors lock_val_split.py: 10% of the train set, stratified by
donor_id x cell_class_4, rng seed 42.

Runs on <compute-node> (needs data/scgpt_vanzyl.h5ad, remote-only).

Outputs:
  artifacts/paper1_vanzyl/loo/scgpt_vanzyl_loo220_{train,test}.h5ad
  artifacts/paper1_vanzyl/loo/val_indices.npz
  artifacts/paper1_vanzyl/loo/split_val_manifest.json
  artifacts/paper1_vanzyl/loo/split_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path("<home>/corneal_scgpt")
PREPPED = ROOT / "data" / "scgpt_vanzyl.h5ad"
ANN = ROOT / "data" / "vanzyl_annotated.h5ad"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "loo"

TEST_DONOR = "sanes2022_Hu0220"  # FROZEN judgment (pre-run)
LAYERS = ["LSC", "basal", "differentiated"]
SPLIT_SEED = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    p = ad.read_h5ad(PREPPED, backed="r")
    obs = p.obs
    leiden = np.asarray(obs["leiden"].astype(str))
    cc4 = np.asarray(obs["cell_class_4"].astype(str))
    donor = np.asarray(obs["donor_id"].astype(str))
    mask = (leiden != "0") & np.isin(cc4, LAYERS)
    log(f"analysis set: {int(mask.sum())}/{p.n_obs}")

    donors = sorted(set(donor.tolist()))
    counts = {d: int(mask[donor == d].sum()) for d in donors}
    log(f"per-donor 3-tier counts: {counts}")
    assert TEST_DONOR in donors
    train_donors = [d for d in donors if d != TEST_DONOR]
    log(f"TEST donor = {TEST_DONOR} ({counts[TEST_DONOR]} cells); "
        f"train = {train_donors}")

    tr_mask = mask & np.isin(donor, train_donors)
    te_mask = mask & (donor == TEST_DONOR)
    n_train = int(tr_mask.sum())
    n_test = int(te_mask.sum())
    log(f"train {n_train} / test {n_test}")
    assert n_train == 62_640 - counts[TEST_DONOR]

    ptr = np.where(tr_mask)[0]
    pte = np.where(te_mask)[0]
    sp_tr = OUT / "scgpt_vanzyl_loo220_train.h5ad"
    sp_te = OUT / "scgpt_vanzyl_loo220_test.h5ad"
    if not (sp_tr.is_file() and sp_te.is_file()):
        a_tr = p[ptr].to_memory().copy()
        a_te = p[pte].to_memory().copy()
        a_tr.write(sp_tr)
        a_te.write(sp_te)
        log(f"wrote train {a_tr.shape} / test {a_te.shape} "
            f"({time.time() - t0:.0f}s)")
    else:
        log("train/test h5ads exist, skipping write")

    # ---- val split (lock_val_split.py logic verbatim) ----
    a_tr = ad.read_h5ad(sp_tr, backed="r")
    vdonor = np.asarray(a_tr.obs["donor_id"].astype(str))
    vcc4 = np.asarray(a_tr.obs["cell_class_4"].astype(str))
    n = a_tr.n_obs
    assert set(vdonor.tolist()) == set(train_donors), \
        "test donor leaked into the LOO train set!"
    strata = sorted(set(zip(vdonor.tolist(), vcc4.tolist())))
    rng = np.random.default_rng(SPLIT_SEED)
    val_idx = np.zeros(0, dtype=np.int64)
    stratum_counts = {}
    for d, c in strata:
        idx = np.where((vdonor == d) & (vcc4 == c))[0]
        perm = rng.permutation(len(idx))
        n_val = max(1, int(round(len(idx) * 0.1)))
        val_idx = np.concatenate([val_idx, idx[perm[:n_val]]])
        stratum_counts[f"{d}|{c}"] = {
            "n_total": int(len(idx)), "n_val": int(n_val),
            "n_train": int(len(idx) - n_val),
        }
    val_idx.sort()
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    n_val = int(val_mask.sum())
    n_tr = n - n_val
    log(f"val={n_val} train={n_tr} (frac={n_val / n:.4f})")
    log(f"val class dist: {dict(Counter(vcc4[val_mask].tolist()))}")
    log(f"val donor dist: {dict(Counter(vdonor[val_mask].tolist()))}")

    np.savez(OUT / "val_indices.npz", val_idx=val_idx.astype(np.int64),
             n_total=n)

    # ---- manifests ----
    files = [PREPPED, sp_tr, sp_te, OUT / "val_indices.npz"]
    hashes = {f.name: sha256(f) for f in files}

    split_manifest = {
        "task": "scGPT LOO (second test donor) arm (approved supplementary)",
        "rule": "test donor = sanes2022_Hu0220 (FROZEN pre-run judgment: "
                "second-largest 3-tier donor); train = other 4",
        "analysis_set_rule": "leiden != '0' AND cell_class_4 in "
                             "{LSC, basal, differentiated} (LOCKED PREREG §7)",
        "layer_order": LAYERS,
        "per_donor_3tier_counts": counts,
        "test_donor": TEST_DONOR,
        "train_donors": train_donors,
        "n_cells_total_3tier": int(mask.sum()),
        "n_train_cells": n_train,
        "n_test_cells": n_test,
        "prepped_panel": {"genes": int(p.shape[1]), "tier": 4253,
                          "note": "shared scgpt_vanzyl.h5ad panel "
                                  "(protocol_02b, PREREG §4 escalation)"},
        "h5ad_sha256": {
            "scgpt_vanzyl.h5ad": hashes["scgpt_vanzyl.h5ad"],
            "scgpt_vanzyl_loo220_train.h5ad":
                hashes["scgpt_vanzyl_loo220_train.h5ad"],
            "scgpt_vanzyl_loo220_test.h5ad":
                hashes["scgpt_vanzyl_loo220_test.h5ad"],
            "val_indices.npz": hashes["val_indices.npz"],
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8")

    val_manifest = {
        "rule": "90/10 stratified by donor_id x cell_class_4 inside the LOO "
                "train set (mirrors lock_val_split.py, PREREG §9); test donor "
                "Hu0220 excluded by construction",
        "source": str(sp_tr),
        "n_cells_total": n,
        "n_train": n_tr,
        "n_val": n_val,
        "val_frac": round(n_val / n, 4),
        "seed": SPLIT_SEED,
        "stratum_counts": stratum_counts,
        "val_class_dist": dict(Counter(vcc4[val_mask].tolist())),
        "val_donor_dist": dict(Counter(vdonor[val_mask].tolist())),
        "sha256": {
            "scgpt_vanzyl_train.h5ad": hashes["scgpt_vanzyl_loo220_train.h5ad"],
            "val_indices.npz": hashes["val_indices.npz"],
        },
        "frozen_baseline_for_collapse": {
            "note": "filled from the G1 npz donor-masked to Hu0220 "
                    "(pre-training); threshold = frozen rho - 0.15",
            "rho_frz_L2normed": None,
            "collapse_threshold": None,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "split_val_manifest.json").write_text(
        json.dumps(val_manifest, indent=2), encoding="utf-8")
    log(f"manifests written; total {time.time() - t0:.0f}s")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
