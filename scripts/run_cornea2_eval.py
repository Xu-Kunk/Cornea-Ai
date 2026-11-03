#!/usr/bin/env python
"""Cornea2 naive-arm eval on the held-out test donor (mirrors
scgpt_eval_vanzyl.py + run6/eval): best-val AND final-epoch checkpoints.

Per seed: cell-level macro-F1 (bootstrap CI n=1000 seed 42) + rho raw/normed
+ label-shuffle z + probe F1 — frozen vs naive comparison uses
<ds>_frozen_baseline.json. Predictions: naive head over cell_type_raw
classes -> majority-map to 3-tier (naive_map from args.json).

Usage
-----
    repo/.venv/bin/python scripts/run_cornea2_eval.py --dataset GSE186433 \
        --seed 42 [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from scgpt_cls_cornea2 import (  # noqa: E402
    load_vocab, build_finetune_model, tokenize_chunked, l2_norm_rows,
    label_shuffle_null, EVAL_CLASSES, N_NULL, NULL_SEED,
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cornea2"
DATASETS = {
    "GSE186433": {
        "test": PROJECT_ROOT / "data" / "cornea2" / "gse186433_test.h5ad",
        "test_donor": "04",
    },
    "GSE155683": {
        "test": PROJECT_ROOT / "data" / "cornea2" / "gse155683_test.h5ad",
        "test_donor": "donor_4",
    },
}
N_PROBE = 5
N_BOOT = 1000
BOOT_SEED = 42


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def cell_bootstrap_f1(y_true, y_pred, n_boot, seed) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    classes = sorted(set(y_true.tolist()))
    n = len(y_true)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = f1_score(y_true[idx], y_pred[idx], labels=classes,
                            average="macro", zero_division=0)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5))


def eval_checkpoint(model, tok, adata, naive_classes, naive_map, device,
                    checkpoint_name: str) -> dict:
    model.eval()
    embs, clss = [], []
    with torch.no_grad():
        for s in range(0, adata.n_obs, 64):
            src = tok["genes"][s : s + 64].to(device)
            values = tok["values"][s : s + 64].to(device)
            pad = tok["mask"][s : s + 64].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(src, values, src_key_padding_mask=pad, CLS=True)
            embs.append(out["cell_emb"].float().cpu().numpy())
            clss.append(out["cls_output"].float().cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    cls = np.concatenate(clss, axis=0).astype(np.float32)
    y_true = adata.obs["cell_class_4"].astype(str).to_numpy()

    pred_author = np.array([naive_classes[int(i)] for i in cls.argmax(1)],
                           dtype=str)
    pred = np.array([naive_map.get(c, "NON_LAYER") for c in pred_author],
                    dtype=str)
    present = sorted(set(y_true.tolist()) & set(EVAL_CLASSES))
    f1 = float(f1_score(y_true, pred, labels=present, average="macro",
                        zero_division=0))
    f1_mean, ci_lo, ci_hi = cell_bootstrap_f1(y_true, pred, N_BOOT, BOOT_SEED)
    rho_raw = float(marker_gradient_correlation(emb, y_true, EVAL_CLASSES))
    emb_n = l2_norm_rows(emb)
    rho_n = float(marker_gradient_correlation(emb_n, y_true, EVAL_CLASSES))
    null = label_shuffle_null(emb_n, y_true, EVAL_CLASSES, n=N_NULL, seed=NULL_SEED)
    z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))

    # probe (split self-probe, same design as frozen)
    emb_nn = l2_norm_rows(emb)
    f1_macros = []
    for i in range(N_PROBE):
        rng = np.random.default_rng(42 + i)
        perm = rng.permutation(emb_nn.shape[0])
        n_tr = int(round(emb_nn.shape[0] * 0.9))
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(emb_nn[perm[:n_tr]])
        Xte = scaler.transform(emb_nn[perm[n_tr:]])
        probe = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                   random_state=42 + i)
        probe.fit(Xtr, y_true[perm[:n_tr]])
        f1_macros.append(float(f1_score(
            y_true[perm[n_tr:]], probe.predict(Xte), labels=present,
            average="macro", zero_division=0)))

    return {
        "checkpoint": checkpoint_name,
        "n_eval_cells": int(adata.n_obs),
        "macro_f1": f1,
        "bootstrap_ci95": [ci_lo, ci_hi],
        "n_boot": N_BOOT,
        "rho_raw": rho_raw,
        "rho_normed": rho_n,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_z": z,
        "null_pctile": float((null <= rho_n).mean()),
        "n_null": N_NULL,
        "probe_f1_macro": float(np.mean(f1_macros)),
        "probe_f1_macro_sd": float(np.std(f1_macros)),
        "pred_dist": {k: int(v) for k, v in
                      zip(*np.unique(pred, return_counts=True))},
        "y_true_dist": {k: int(v) for k, v in
                        zip(*np.unique(y_true, return_counts=True))},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--seed", type=int, choices=[42, 123], required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tag = args.dataset.lower()
    cfg = DATASETS[args.dataset]
    run_dir = OUT_DIR / args.dataset / f"naive_seed{args.seed}"
    args_json = json.loads((run_dir / "args.json").read_text())
    frozen = json.loads((OUT_DIR / f"{tag}_frozen_baseline.json").read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vocab, pargs = load_vocab()
    import anndata as ad

    adata = ad.read_h5ad(cfg["test"])
    keep = np.array([g in vocab for g in adata.var_names])
    gene_ids = np.array([vocab[g] for g in adata.var_names[keep]], dtype=np.int64)
    x_mat = adata[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = tokenize_chunked(x_mat, gene_ids, vocab, pargs)

    naive_classes = args_json["naive_classes"]
    naive_map = args_json["naive_map"]
    n_cls = args_json["n_cls"]

    results = {"dataset": args.dataset, "seed": args.seed,
               "test_donor": cfg["test_donor"],
               "frozen_baseline": {
                   "rho_normed": frozen["rho"]["rho_normed"],
                   "null_z": frozen["rho"]["null_z"],
                   "probe_f1_macro": frozen["probe_f1"]["f1_macro"],
               }}
    for ckpt_name in ["model_best_val.pt", "model_epoch15.pt"]:
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            log(f"  SKIP {ckpt_name}: not found")
            continue
        model = build_finetune_model(vocab, pargs, n_cls)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.to(device)
        t0 = time.time()
        res = eval_checkpoint(model, tok, adata, naive_classes, naive_map,
                              device, ckpt_name)
        res["wall_s"] = round(time.time() - t0, 1)
        key = "best_val" if ckpt_name == "model_best_val.pt" else "final_epoch"
        results[key] = res
        log(f"  {key} ({ckpt_name}): F1={res['macro_f1']:.4f} "
            f"CI=[{res['bootstrap_ci95'][0]:.4f},{res['bootstrap_ci95'][1]:.4f}] "
            f"rho_n={res['rho_normed']:.4f} z={res['null_z']:.1f}")

    if "best_val" in results and "final_epoch" in results:
        bv, fe = results["best_val"], results["final_epoch"]
        results["collapse_reproduced"] = {
            "rho_naive_best_val": bv["rho_normed"],
            "rho_naive_final_epoch": fe["rho_normed"],
            "rho_frozen": frozen["rho"]["rho_normed"],
            "delta_best_val": frozen["rho"]["rho_normed"] - bv["rho_normed"],
            "delta_final_epoch": frozen["rho"]["rho_normed"] - fe["rho_normed"],
            "note": "naive fine-tuning destroys the differentiation gradient "
                    "if delta (frozen - naive) is large positive",
        }
    results["timestamp_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    log(f"saved {run_dir / 'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
