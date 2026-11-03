#!/usr/bin/env python
"""Cornea2 frozen baseline eval (mirrors scgpt_frozen_diag_rebuild.py).

Computes the frozen (no fine-tuning) scGPT metrics on each dataset's
HELD-OUT TEST DONOR:
    rho(L2-normed) + label-shuffle z (n=100) + split self-probe F1 (5 reps)
    + test-donor cell-level bootstrap macro-F1 CI (n=1000, seed 42).

The frozen rho_frz is consumed by scgpt_cls_cornea2.py --frozen-rho for the
collapse-start epoch threshold (rho_frz - 0.15).

Outputs (per dataset):
    artifacts/paper1_vanzyl/cornea2/<ds>_frozen_baseline.json
    artifacts/paper1_vanzyl/cornea2/<ds>_frozen_emb.npz

Usage
-----
    repo/.venv/bin/python scripts/run_cornea2_frozen_eval.py \
        --dataset GSE186433 [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.tokenizer.gene_tokenizer import BuiltinVocab
from scgpt.utils import load_pretrained
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from scgpt_cls_cornea2 import (  # noqa: E402
    PRETRAINED_DIR, load_vocab, build_finetune_model, tokenize_chunked,
    l2_norm_rows, label_shuffle_null, EVAL_CLASSES, N_NULL, NULL_SEED,
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


def fit_eval_probe(emb, y_true_axis, layer_order, n_repeats,
                   train_frac=0.9) -> dict:
    emb_n = l2_norm_rows(emb)
    present = sorted(set(y_true_axis.astype(str)) & set(layer_order))
    f1_macros, f1_weighteds = [], []
    for i in range(n_repeats):
        rng = np.random.default_rng(42 + i)
        perm = rng.permutation(emb_n.shape[0])
        n_tr = int(round(emb_n.shape[0] * train_frac))
        tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(emb_n[tr_idx])
        Xte = scaler.transform(emb_n[te_idx])
        probe = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                   random_state=42 + i)
        probe.fit(Xtr, y_true_axis[tr_idx])
        pred = probe.predict(Xte)
        y_te = y_true_axis[te_idx]
        f1_macros.append(float(f1_score(y_te, pred, labels=present,
                                        average="macro", zero_division=0)))
        f1_weighteds.append(float(f1_score(y_te, pred, labels=present,
                                           average="weighted",
                                           zero_division=0)))
    return {
        "f1_macro": float(np.mean(f1_macros)),
        "f1_macro_sd": float(np.std(f1_macros)),
        "f1_weighted": float(np.mean(f1_weighteds)),
        "f1_weighted_sd": float(np.std(f1_weighteds)),
        "n_repeats": int(n_repeats),
        "design": "split self-probe (L2norm + StandardScaler + lbfgs C=1.0 "
                  "max_iter=5000, seeds 42+i)",
    }


def cell_bootstrap_f1(y_true, y_pred, n_boot, seed) -> tuple[float, float, float]:
    """Cell-level bootstrap macro-F1 CI (single test donor -> cell-level)."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vocab, pargs = load_vocab()
    model = build_finetune_model(vocab, pargs, n_cls=3)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"frozen model params={n_params / 1e6:.2f}M device={device}")

    import anndata as ad

    t0 = time.time()
    adata = ad.read_h5ad(cfg["test"])
    keep = np.array([g in vocab for g in adata.var_names])
    gene_ids = np.array([vocab[g] for g in adata.var_names[keep]], dtype=np.int64)
    x_mat = adata[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = tokenize_chunked(x_mat, gene_ids, vocab, pargs)
    log(f"test {adata.n_obs} cells tokenized ({time.time() - t0:.1f}s)")

    embs = []
    with torch.no_grad():
        for s in range(0, adata.n_obs, 64):
            src = tok["genes"][s : s + 64].to(device)
            values = tok["values"][s : s + 64].to(device)
            pad = tok["mask"][s : s + 64].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(src, values, src_key_padding_mask=pad, CLS=False)
            embs.append(out["cell_emb"].float().cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    y_true = adata.obs["cell_class_4"].astype(str).to_numpy()
    log(f"embeddings {emb.shape} ({time.time() - t0:.1f}s)")

    rho_raw = float(marker_gradient_correlation(emb, y_true, EVAL_CLASSES))
    emb_n = l2_norm_rows(emb)
    rho_n = float(marker_gradient_correlation(emb_n, y_true, EVAL_CLASSES))
    null = label_shuffle_null(emb_n, y_true, EVAL_CLASSES, n=N_NULL, seed=NULL_SEED)
    z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))
    probe = fit_eval_probe(emb, y_true, EVAL_CLASSES, N_PROBE)
    log(f"frozen rho_raw={rho_raw:.4f} rho_normed={rho_n:.4f} z={z:.1f} "
        f"probe_f1={probe['f1_macro']:.4f}")

    # probe classifier -> test-donor cell-level bootstrap F1 (same probe
    # design; predictions from the LAST repeat's classifier)
    emb_nn = l2_norm_rows(emb)
    present = sorted(set(y_true.tolist()) & set(EVAL_CLASSES))
    rng = np.random.default_rng(42 + N_PROBE - 1)
    perm = rng.permutation(emb_nn.shape[0])
    n_tr = int(round(emb_nn.shape[0] * 0.9))
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(emb_nn[perm[:n_tr]])
    Xte = scaler.transform(emb_nn[perm[n_tr:]])
    probe_last = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                    random_state=42 + N_PROBE - 1)
    probe_last.fit(Xtr, y_true[perm[:n_tr]])
    pred = probe_last.predict(Xte)
    f1_mean, ci_lo, ci_hi = cell_bootstrap_f1(
        y_true[perm[n_tr:]], pred, N_BOOT, BOOT_SEED
    )

    result = {
        "dataset": args.dataset,
        "arm": "frozen",
        "n_eval_cells": int(adata.n_obs),
        "test_donor": cfg["test_donor"],
        "rho": {
            "rho_raw": rho_raw,
            "rho_normed": rho_n,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_z": z,
            "null_pctile": float((null <= rho_n).mean()),
            "n_null": N_NULL,
        },
        "probe_f1": probe,
        "probe_bootstrap_f1": {
            "macro_f1": f1_mean,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "design": "probe classifier (last repeat) -> held-out 10% "
                      "test-donor cells, cell-level bootstrap macro-F1",
        },
        "y_true_dist": {k: int(v) for k, v in
                        zip(*np.unique(y_true, return_counts=True))},
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.dataset.lower()
    (OUT_DIR / f"{tag}_frozen_baseline.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    np.savez(OUT_DIR / f"{tag}_frozen_emb.npz", emb=emb, y=y_true)
    log(f"saved {OUT_DIR / f'{tag}_frozen_baseline.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
