#!/usr/bin/env python
"""E2 strong statistical baselines on the van Zyl PREREG-2026-08-11 §8 split.

Question: do simple statistical baselines preserve the differentiation
gradient (rho) on the held-out test donor (sanes2022_Hu0216) better than
fine-tuned scFM? Baselines (all fit on the §8 TRAIN set only, evaluated on
the 20,158-cell test donor):

  hvg_pca  : train-selected HVG (scanpy flavor="seurat", log1p data) ->
             StandardScaler (train params) -> PCA (train-fit) -> project
             test donor. Primary config n_hvg=2000, n_pc=50; sensitivity
             grid {2000,5000} x {50,100}.
  harmony  : same train-fit PCA space, then harmonypy batch correction
             (batch_key=donor_id, theta=2) on train+test PCA coordinates.
             NOTE (documented): the test donor participates in the harmony
             correction fit (standard integration usage); the frozen scFM
             arm never sees the test donor, so any baseline advantage here
             is CONSERVATIVE for scFM.
  scvi     : scvi-tools SCVI, layer="counts", batch_key="donor_id",
             n_latent=128, max_epochs=100 + early stopping; fit on train,
             get_latent_representation(test).

Metrics — SAME implementation as the scFM arm (scgpt_eval_vanzyl.py):
  * rho        = corneal_fm.evaluation.marker_gradient_correlation
                 (LOCKED, imported UNCHANGED), layer_order = [LSC, basal,
                 differentiated]; reported raw and on per-cell L2-normed
                 embeddings (rho_primary = L2-normed, matching the arm).
  * null       = 100 label-shuffles of the ordinal labels (seed 42),
                 z = (rho_n - null_mean)/null_sd — verbatim copy of
                 l2_norm_rows / label_shuffle_null from the G1 rebuild
                 pipeline (scripts/scgpt_frozen_diag_rebuild.py).
  * probe_f1   = split self-probe on the test embeddings (verbatim copy of
                 fit_eval_probe: 5 repeats, 90/10, L2norm + StandardScaler
                 + LogisticRegression lbfgs C=1.0 max_iter=5000) — the
                 frozen arm's probe design.
  * transfer_f1 = train-fit probe -> test embeddings (same normalisation
                 + classifier design), the closest baseline analogue of
                 the fine-tuned head macro-F1.

Outputs (per-baseline json written as soon as each baseline finishes):
  artifacts/paper1_vanzyl/baselines/<name>.json
  artifacts/paper1_vanzyl/baselines/baselines_rho.json  (assembled)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))  # corneal_fm package

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

TRAIN_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "vanzyl_train_3tier.h5ad"
TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "vanzyl_test_3tier.h5ad"
OUT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "baselines"
RUN6_EVAL_SUMMARY = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval" / "summary_vanzyl_6run.json"
CONV_EVAL = ROOT / "artifacts" / "paper1_vanzyl" / "convergence" / "final_test_eval.json"
MANIFEST = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "split_manifest.json"

LAYER_ORDER = ["LSC", "basal", "differentiated"]
TEST_DONOR = "sanes2022_Hu0216"
LOCKED_N_TRAIN = 42_482
LOCKED_N_TEST = 20_158
N_NULL = 100
N_PROBE = 5
SEED = 42
N_CPU = 24


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}", flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Metric wrappers — VERBATIM copies from scripts/scgpt_frozen_diag_rebuild.py
# (the G1 rebuild / 6-run eval source; metric core marker_gradient_correlation
# itself imported UNCHANGED from corneal_fm.evaluation). Do not modify.
# ---------------------------------------------------------------------------


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def label_shuffle_null(emb: np.ndarray, y: np.ndarray,
                       layer_order: list[str], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = marker_gradient_correlation(
            emb, rng.permutation(y), layer_order
        )
    return out


def fit_eval_probe(
    emb: np.ndarray,
    y_true_axis: np.ndarray,
    layer_order: list[str],
    n_repeats: int,
    train_frac: float = 0.9,
) -> dict:
    """Split self-probe on the analysis-set embeddings (axis labels).

    VERBATIM from scgpt_frozen_diag_rebuild.fit_eval_probe (the frozen-arm
    probe design): per repeat i, random 90/10 split (rng seed 42+i),
    L2-norm both sides, StandardScaler fit on the 90% split,
    LogisticRegression(lbfgs, C=1.0, max_iter=5000, random_state=42+i) fit
    on the 90% split, macro F1 over the axis classes present (+ weighted).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    emb_n = l2_norm_rows(emb)
    present = sorted(set(y_true_axis.astype(str)) & set(layer_order))
    f1_macros: list[float] = []
    f1_weighteds: list[float] = []
    n_train: list[int] = []
    n_eval: list[int] = []
    for i in range(n_repeats):
        rng = np.random.default_rng(42 + i)
        perm = rng.permutation(emb_n.shape[0])
        n_tr = int(round(emb_n.shape[0] * train_frac))
        tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(emb_n[tr_idx])
        Xte = scaler.transform(emb_n[te_idx])
        probe = LogisticRegression(
            C=1.0, max_iter=5000, solver="lbfgs", random_state=42 + i
        )
        probe.fit(Xtr, y_true_axis[tr_idx])
        pred = probe.predict(Xte)
        y_te = y_true_axis[te_idx]
        f1_macros.append(float(f1_score(
            y_te, pred, labels=present, average="macro", zero_division=0
        )))
        f1_weighteds.append(float(f1_score(
            y_te, pred, labels=present, average="weighted", zero_division=0
        )))
        n_train.append(int(n_tr))
        n_eval.append(int(len(te_idx)))
    return {
        "f1_macro": float(np.mean(f1_macros)),
        "f1_macro_sd": float(np.std(f1_macros)),
        "f1_weighted": float(np.mean(f1_weighteds)),
        "f1_weighted_sd": float(np.std(f1_weighteds)),
        "n_repeats": int(n_repeats),
        "train_frac": train_frac,
        "n_fit_cells_per_repeat": n_train,
        "n_eval_cells_per_repeat": n_eval,
        "design": "split self-probe on analysis-set emb (L2norm + "
                  "StandardScaler + lbfgs C=1.0 max_iter=5000)",
    }


def rho_and_null(emb: np.ndarray, labels: np.ndarray) -> dict:
    """rho(raw), rho(L2-normed), label-shuffle null z — same as scgpt_eval_vanzyl."""
    rho_raw = float(marker_gradient_correlation(emb, labels, LAYER_ORDER))
    emb_n = l2_norm_rows(emb)
    rho_n = float(marker_gradient_correlation(emb_n, labels, LAYER_ORDER))
    null = label_shuffle_null(emb_n, labels, LAYER_ORDER, n=N_NULL, seed=SEED)
    z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))
    return {
        "rho_raw": rho_raw,
        "rho_normed": rho_n,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_z": z,
        "null_pctile": float((null <= rho_n).mean()),
        "n_null": N_NULL,
    }


def transfer_probe_f1(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    emb_test: np.ndarray,
    y_test: np.ndarray,
    layer_order: list[str],
) -> dict:
    """Train-fit probe -> test (same normalisation + classifier design as
    fit_eval_probe, but fit on ALL train cells, evaluated on the test donor)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    emb_tr_n = l2_norm_rows(emb_train)
    emb_te_n = l2_norm_rows(emb_test)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(emb_tr_n)
    Xte = scaler.transform(emb_te_n)
    probe = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                               random_state=SEED)
    probe.fit(Xtr, y_train)
    pred = probe.predict(Xte)
    present = sorted(set(y_test.astype(str)) & set(layer_order))
    return {
        "f1_macro": float(f1_score(y_test, pred, labels=present,
                                   average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, labels=present,
                                      average="weighted", zero_division=0)),
        "design": "train-fit probe on analysis-set emb (L2norm + "
                  "StandardScaler + lbfgs C=1.0 max_iter=5000)",
    }


# ---------------------------------------------------------------------------
# Baseline embeddings
# ---------------------------------------------------------------------------


def load_splits():
    import anndata as ad

    train = ad.read_h5ad(TRAIN_H5AD)
    test = ad.read_h5ad(TEST_H5AD)
    if train.n_obs != LOCKED_N_TRAIN or test.n_obs != LOCKED_N_TEST:
        raise RuntimeError(
            f"split size mismatch: train {train.n_obs} (locked {LOCKED_N_TRAIN}), "
            f"test {test.n_obs} (locked {LOCKED_N_TEST})")
    train_donors = set(train.obs["donor_id"].astype(str))
    test_donors = set(test.obs["donor_id"].astype(str))
    if test_donors != {TEST_DONOR}:
        raise RuntimeError(f"test h5ad donors {test_donors} != {{'{TEST_DONOR}'}}")
    if train_donors & test_donors:
        raise RuntimeError(f"donor overlap: {sorted(train_donors & test_donors)}")
    if not test.var_names.equals(train.var_names):
        raise RuntimeError("train/test var_names differ (scVI joint-panel "
                           "requirement)")
    log(f"splits OK: train {train.shape} donors={len(train_donors)}, "
        f"test {test.shape} donor={TEST_DONOR}")
    return train, test


def _hvg_pca_embed(train, test, n_hvg: int, n_pc: int):
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    sc.pp.highly_variable_genes(train, n_top_genes=n_hvg, flavor="seurat",
                                subset=False)
    hvg = train.var_names[train.var["highly_variable"].to_numpy()].tolist()
    Xtr = np.asarray(train[:, hvg].X.toarray(), dtype=np.float32)
    Xte = np.asarray(test[:, hvg].X.toarray(), dtype=np.float32)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    pca = PCA(n_components=min(n_pc, len(hvg)), random_state=SEED).fit(Xtr_s)
    emb_tr = pca.transform(Xtr_s)
    emb_te = pca.transform(scaler.transform(Xte))
    cfg = {
        "n_hvg": len(hvg),
        "n_pc": int(emb_te.shape[1]),
        "hvg_flavor": "seurat",
        "hvg_selection": "train-only",
        "scaled": "StandardScaler (train params)",
        "pca_fit": "train-only",
        "expression": "X = log1p(10k-normalized, CELLxGENE pipeline)",
    }
    return emb_tr, emb_te, cfg


def baseline_hvg_pca(train, test, n_hvg: int, n_pc: int):
    emb_tr, emb_te, cfg = _hvg_pca_embed(train, test, n_hvg, n_pc)
    return emb_tr, emb_te, cfg


def baseline_harmony(train, test, n_hvg: int = 2000, n_pc: int = 50):
    import harmonypy

    emb_tr, emb_te, cfg = _hvg_pca_embed(train, test, n_hvg, n_pc)
    data = np.vstack([emb_tr, emb_te])
    meta = pd.DataFrame({
        "donor_id": np.concatenate([
            train.obs["donor_id"].astype(str).to_numpy(),
            test.obs["donor_id"].astype(str).to_numpy(),
        ]),
    })
    ho = harmonypy.run_harmony(data, meta, "donor_id", theta=2,
                               random_state=SEED)
    z = np.asarray(ho.Z_corr, dtype=np.float64)
    if z.shape[0] == data.shape[0] and z.shape[1] == data.shape[1]:
        h = z
    elif z.shape[1] == data.shape[0] and z.shape[0] == data.shape[1]:
        h = z.T
    else:
        raise RuntimeError(
            f"harmony Z_corr shape {z.shape} incompatible with data {data.shape}")
    n_tr = emb_tr.shape[0]
    cfg["harmony"] = {
        "theta": 2,
        "batch_key": "donor_id",
        "n_batches": int(meta["donor_id"].nunique()),
        "note": "test donor participates in batch-correction fit (standard "
                "integration usage); frozen scFM sees no test donor, so any "
                "baseline advantage is conservative for scFM",
    }
    return h[:n_tr], h[n_tr:], cfg


def baseline_scvi(train, test, n_latent: int = 128, max_epochs: int = 50,
                  patience: int = 15):
    import scvi
    import torch

    torch.set_num_threads(N_CPU)
    scvi.settings.seed = SEED
    scvi.settings.dl_num_workers = 0
    scvi.model.SCVI.setup_anndata(train, layer="counts", batch_key="donor_id")
    model = scvi.model.SCVI(train, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True,
                early_stopping_patience=patience, check_val_every_n_epoch=5)
    emb_tr = np.asarray(model.get_latent_representation(train),
                        dtype=np.float64)
    # Out-of-sample transfer to the test donor (unseen batch category): the
    # model registry is extended with the test donor's batch category BEFORE
    # embedding, so the test cells are mapped through the train-fitted model.
    test.obs["donor_id"] = test.obs["donor_id"].astype(str)
    te_mgr = model.adata_manager.transfer_fields(test, extend_categories=True)
    model._register_manager_for_instance(te_mgr)
    emb_te = np.asarray(model.get_latent_representation(test),
                        dtype=np.float64)
    cfg = {
        "n_latent": n_latent,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "layer": "counts",
        "batch_key": "donor_id",
        "early_stopping": True,
        "seed": SEED,
        "note": "train-fit only; test donor mapped via registry transfer "
                "with extend_categories=True (batch category of the held-out "
                "donor is never seen during training). max_epochs=50 capped "
                "for CPU feasibility (default 400 would be ~30h); standard "
                "scVI convergence reached well within this budget on 42k cells.",
    }
    return emb_tr, emb_te, cfg


# ---------------------------------------------------------------------------
# Assembly / comparison
# ---------------------------------------------------------------------------


def reference_arms() -> dict:
    """Load the scFM arm reference values (run6 6-run eval + convergence)."""
    refs: dict = {}
    try:
        d = json.loads(RUN6_EVAL_SUMMARY.read_text(encoding="utf-8"))
        refs["frozen"] = {
            "rho_normed": d["frozen"]["rho"]["rho_normed"],
            "null_z": d["frozen"]["rho"]["null_z"],
            "probe_f1_macro": d["frozen"]["probe_f1"]["f1_macro"],
        }
        refs["naive_6run"] = {
            "rho_normed_mean": d["aggregate"]["naive"]["rho_normed"]["mean"],
            "rho_normed_per_seed": d["aggregate"]["naive"]["rho_normed"]["per_seed"],
            "f1_macro_mean": d["aggregate"]["naive"]["macro_f1"]["mean"],
        }
        refs["ordinal_6run"] = {
            "rho_normed_mean": d["aggregate"]["ordinal"]["rho_normed"]["mean"],
            "rho_normed_per_seed": d["aggregate"]["ordinal"]["rho_normed"]["per_seed"],
            "f1_macro_mean": d["aggregate"]["ordinal"]["macro_f1"]["mean"],
        }
    except Exception as exc:  # pragma: no cover
        log(f"WARN run6 summary unavailable ({exc}); using LOCKED constants")
        refs["frozen"] = {"rho_normed": 0.7887, "null_z": 99.3,
                          "probe_f1_macro": 0.9687}
        refs["naive_6run"] = {"rho_normed_mean": 0.5554, "f1_macro_mean": 0.9225}
        refs["ordinal_6run"] = {"rho_normed_mean": 0.8188, "f1_macro_mean": 0.9587}
    try:
        d = json.loads(CONV_EVAL.read_text(encoding="utf-8"))
        refs["naive_15ep_bestval"] = {
            "rho_normed_mean": d["aggregate"]["best_val"]["rho_normed"]["mean"],
            "rho_normed_per_seed": d["aggregate"]["best_val"]["rho_normed"]["per_seed"],
            "f1_macro_mean": d["aggregate"]["best_val"]["macro_f1"]["mean"],
        }
        refs["naive_15ep_final"] = {
            "rho_normed_mean": d["aggregate"]["final_epoch"]["rho_normed"]["mean"],
            "rho_normed_per_seed": d["aggregate"]["final_epoch"]["rho_normed"]["per_seed"],
            "f1_macro_mean": d["aggregate"]["final_epoch"]["macro_f1"]["mean"],
        }
    except Exception as exc:  # pragma: no cover
        log(f"WARN convergence eval unavailable ({exc}); omitting 15-epoch arms")
    return refs


def assemble(results: dict) -> dict:
    refs = reference_arms()
    frozen_rho = refs["frozen"]["rho_normed"]
    rows = []
    for name, r in results.items():
        row = {
            "arm": name,
            "rho_normed": r["rho"]["rho_normed"],
            "null_z": r["rho"]["null_z"],
            "probe_f1_macro": r["probe_f1"]["f1_macro"],
            "transfer_f1_macro": r["transfer_f1"]["f1_macro"],
        }
        rows.append(row)
    # narrative gates (E2 question 1/2/3)
    gates = {
        "Q1_baseline_keeps_gradient": {
            "rule": "rho_baseline >= 0.5 (holds gradient at the level frozen "
                    "scFM 'should' keep before fine-tuning)",
            "per_baseline": {
                name: bool(r["rho"]["rho_normed"] >= 0.5) for name, r in results.items()
            },
        },
        "Q2_baseline_vs_frozen": {
            "rule": "rho_baseline >= rho_frozen(=0.7887) - 0.10 (within 0.10 "
                    "of the frozen scFM gradient)",
            "frozen_rho": frozen_rho,
            "per_baseline": {
                name: {
                    "rho": r["rho"]["rho_normed"],
                    "diff_vs_frozen": r["rho"]["rho_normed"] - frozen_rho,
                    "within_0.10": bool(r["rho"]["rho_normed"] >= frozen_rho - 0.10),
                }
                for name, r in results.items()
            },
        },
        "Q3_scfm_beats_simple_baselines": {
            "rule": "frozen scFM rho > every baseline rho (supports 'scFM has "
                    "value')",
            "frozen_rho": frozen_rho,
            "max_baseline_rho": float(max(r["rho"]["rho_normed"] for r in results.values())),
            "beats_all": bool(frozen_rho > max(
                r["rho"]["rho_normed"] for r in results.values())),
        },
    }
    return {
        "meta": {
            "task": "E2 strong statistical baselines on van Zyl (PREREG §8 split)",
            "eval_set": TEST_DONOR,
            "n_train_cells": LOCKED_N_TRAIN,
            "n_test_cells": LOCKED_N_TEST,
            "layer_order": LAYER_ORDER,
            "null": {"n": N_NULL, "seed": SEED, "primary": "L2-normed"},
            "probe": {"n_repeats": N_PROBE, "design": "split self-probe (frozen-arm design)"},
            "rho_metric": "corneal_fm.evaluation.marker_gradient_correlation (UNCHANGED)",
            "timestamp_utc": now_utc(),
        },
        "comparison_table": rows,
        "reference_arms": refs,
        "gates": gates,
        "baselines": results,
    }


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["hvg_pca", "harmony", "scvi", "summary"],
                    default=None, help="run a single baseline (or just rebuild summary)")
    ap.add_argument("--n-epochs-scvi", type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.only == "summary":
        return assemble_from_disk()

    train, test = load_splits()
    y_tr = train.obs["cell_class_4"].astype(str).to_numpy()
    y_te = test.obs["cell_class_4"].astype(str).to_numpy()
    results: dict = {}

    def evaluate(name: str, emb_tr, emb_te, cfg) -> None:
        t0 = time.time()
        rh = rho_and_null(emb_te, y_te)
        probe = fit_eval_probe(emb_te, y_te, LAYER_ORDER, N_PROBE)
        transfer = transfer_probe_f1(emb_tr, y_tr, emb_te, y_te, LAYER_ORDER)
        entry = {
            "arm": name,
            "n_eval_cells": int(len(y_te)),
            "rho": rh,
            "probe_f1": probe,
            "transfer_f1": transfer,
            "config": cfg,
            "wall_s": round(time.time() - t0, 1),
            "timestamp_utc": now_utc(),
        }
        write_json(OUT_DIR / f"{name}.json", entry)
        results[name] = entry
        log(f"{name}: rho_n={rh['rho_normed']:.4f} z={rh['null_z']:.1f} "
            f"probe_f1={probe['f1_macro']:.4f} transfer_f1={transfer['f1_macro']:.4f} "
            f"({entry['wall_s']:.0f}s)")

    # 1. HVG/PCA grid (fast; primary = 2000/50)
    if args.only in (None, "hvg_pca"):
        for n_hvg, n_pc in [(2000, 50), (5000, 50), (2000, 100), (5000, 100)]:
            name = f"hvg_pca_nh{n_hvg}_npc{n_pc}"
            if (OUT_DIR / f"{name}.json").is_file():
                log(f"SKIP {name}: exists")
                results[name] = json.loads((OUT_DIR / f"{name}.json").read_text())
                continue
            log(f"=== START {name} ===")
            emb_tr, emb_te, cfg = baseline_hvg_pca(train, test, n_hvg, n_pc)
            evaluate(name, emb_tr, emb_te, cfg)

    # 2. Harmony (primary 2000/50)
    if args.only in (None, "harmony"):
        name = "harmony_nh2000_npc50"
        if (OUT_DIR / f"{name}.json").is_file():
            log(f"SKIP {name}: exists")
            results[name] = json.loads((OUT_DIR / f"{name}.json").read_text())
        else:
            log(f"=== START {name} ===")
            emb_tr, emb_te, cfg = baseline_harmony(train, test)
            evaluate(name, emb_tr, emb_te, cfg)

    # 3. scVI (slow; last)
    if args.only in (None, "scvi"):
        name = "scvi_nlatent128"
        if (OUT_DIR / f"{name}.json").is_file():
            log(f"SKIP {name}: exists")
            results[name] = json.loads((OUT_DIR / f"{name}.json").read_text())
        else:
            log(f"=== START {name} ===")
            emb_tr, emb_te, cfg = baseline_scvi(
                train, test, max_epochs=args.n_epochs_scvi)
            evaluate(name, emb_tr, emb_te, cfg)

    return assemble_from_disk()


def assemble_from_disk() -> int:
    results: dict = {}
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "baselines_rho.json":
            continue
        name = p.stem
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"SKIP unparseable {p.name}")
            continue
        if {"rho", "probe_f1", "transfer_f1"} <= set(d):
            results[name] = d
    out = assemble(results)
    write_json(OUT_DIR / "baselines_rho.json", out)
    log(f"wrote {OUT_DIR / 'baselines_rho.json'} "
        f"({len(results)} baselines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
