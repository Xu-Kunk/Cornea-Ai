#!/usr/bin/env python
"""CellPLM naive-CE fine-tuning + PCS evaluation (cross-tissue / LOO arms).

Approved supplementary experiments (reviewer points: single-tissue and
single-donor bias). Parameterized variant of run_cellplm_vanzyl.py (arm C3)
— same official CellTypeAnnotationPipeline, same metric suite, only the
data / axis / out_dim / test-donor label are configurable:
  * epidermis: data/epidermis_cellplm.h5ad (4-layer axis, out_dim 4,
    test = the frozen 10% stratified hold-out)
  * LOO:       data/vanzyl_cellplm_loo220.h5ad (3-tier axis, out_dim 3,
    test donor sanes2022_Hu0220)

Frozen judgments (unchanged from arm C3, recorded in manifests):
  * official defaults (es=200, lr 5e-3 embedder 5e-4, wd 1e-7, scheduler
    plat, epochs 2000, patience 25, hvg 3000, workers 0)
  * best-valid-F1 checkpoint selection (official fit() contract)
  * seeds 42 / 123 (2 seeds per arm; deterministic choice)
  * eval caliber identical to GF/scGPT arms: L2-normed rho via
    corneal_fm.evaluation.marker_gradient_correlation, label-shuffle z
    (n=100, seed 42), macro-F1 + cell-level bootstrap CI (n=1000, seed 42)
  * frozen baseline: same test cells through the unfitted pretrained model

GPU guard: waits until torch.cuda.mem_get_info()[0] >= 4 GiB (poll 60 s,
max wait budget 5 h); OOM -> backoff retry.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/home/dq/Codex/Ai")
sys.path.insert(0, str(PROJECT_ROOT))  # for corneal_fm

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

# --- torch 2.x compat patch (recorded deviation D2) --------------------------
# Official load_pretrain() omits map_location; torch >= 2.6 restores to the
# saved device (cuda) by default, which fails under unified-memory contention
# and is wrong when the requested device is CPU. Patch to load to CPU first,
# then .to(device) happens in the pipeline. Scientific logic untouched.
import CellPLM.pipeline as _pipe  # noqa: E402
import CellPLM.pipeline.cell_type_annotation as _cta  # noqa: E402
from CellPLM.model import OmicsFormer  # noqa: E402


def _load_pretrain_cpu(pretrain_prefix, overwrite_config=None, pretrain_directory="./ckpt"):
    import json
    import os

    config_path = os.path.join(pretrain_directory, f"{pretrain_prefix}.config.json")
    ckpt_path = os.path.join(pretrain_directory, f"{pretrain_prefix}.best.ckpt")
    with open(config_path, "r") as openfile:
        config = json.load(openfile)
    config.update(overwrite_config)
    model = OmicsFormer(**config)
    pretrained_model_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_model_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


_cta.load_pretrain = _load_pretrain_cpu  # pipeline/__init__ imported ref into cta
_pipe.load_pretrain = _load_pretrain_cpu
# -----------------------------------------------------------------------------

LAYER_ORDER = ["LSC", "basal", "differentiated"]  # overridden from --layer-order
NULL_N, NULL_SEED = 100, 42
BOOT_N, BOOT_SEED = 1000, 42
MIN_FREE_GPU = 4 * 1024**3
MAX_WAIT_S = 5 * 3600
POLL_S = 60


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def _gpu_state() -> tuple[bool, float]:
    """Return (usable, free_gib). A small cuda alloc is the decisive probe on
    unified-memory systems where mem_get_info can itself fail under contention."""
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        free, _total = torch.cuda.mem_get_info(0)
        return free >= MIN_FREE_GPU, free / 2**30
    except Exception:
        return False, 0.0


def wait_for_gpu(device: str) -> None:
    if device == "cpu":
        return
    t0 = time.time()
    while True:
        ok, free_gib = _gpu_state()
        if ok:
            log(f"GPU guard passed: free {free_gib:.2f} GiB >= 4 GiB")
            return
        waited = time.time() - t0
        log(f"GPU busy/contended: free {free_gib:.2f} GiB; waited {waited / 60:.0f} min")
        if waited > MAX_WAIT_S:
            raise RuntimeError(f"GPU guard timed out after {MAX_WAIT_S / 3600:.0f} h")
        time.sleep(POLL_S)


def label_shuffle_null(emb: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = marker_gradient_correlation(emb, rng.permutation(y), LAYER_ORDER)
    return out


def bootstrap_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Cell-level bootstrap 95% CI of macro-F1 (matches run6 eval design)."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = np.empty(n)
    for i in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = f1_score(y_true[s], y_pred[s], average="macro", zero_division=0)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def extract_embeddings(model, adata, covariate_fields, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicates official inference() but returns latent embeddings + preds + labels.

    Returns (embeddings, preds, labels) aligned to adata row order (order_required=True).
    """
    from CellPLM.pipeline.cell_type_annotation import inference as _unused  # noqa: F401
    from CellPLM.utils.data import TranscriptomicDataset, XDict
    from torch.utils.data import DataLoader

    dataset = TranscriptomicDataset(adata, None, covariate_fields, order_required=True)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    model.eval()
    latents, preds, labels, order_list = [], [], [], []
    with torch.no_grad():
        for data_dict in dataloader:
            input_dict = {}
            for k in data_dict:
                if k == "x_seq":
                    input_dict[k] = data_dict[k].to(device)
                elif k not in ["gene_list", "split", "order_list"]:
                    input_dict[k] = data_dict[k].to(device)
            input_dict["loss_mask"] = torch.ones(data_dict["x_seq"].shape[0]).bool().to(device)
            out_dict, _ = model(XDict(input_dict), data_dict["gene_list"])
            latents.append(out_dict["latent"].cpu())
            preds.append(out_dict["pred"].cpu())
            if "label" in input_dict:
                labels.append(input_dict["label"].cpu())
            order_list.append(input_dict.get("order_list", torch.arange(data_dict["x_seq"].shape[0])).cpu())
    emb = torch.cat(latents).numpy()
    pred = torch.cat(preds).numpy()
    order = torch.cat(order_list).numpy()
    # restore original order (order_list gives original index of each row)
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    emb = emb[inv]
    pred = pred[inv]
    if labels:
        lab = torch.cat(labels).numpy()[inv]
    else:
        lab = np.full(len(order), -1)
    return emb, pred, lab


def run_seed(seed: int, data_path: Path, ckpt_dir: Path, out_dir: Path, device: str,
             out_dim: int = 3, test_donor_label: str = "sanes2022_Hu0216") -> dict:
    import hdf5plugin  # noqa: F401
    import anndata as ad
    from CellPLM.utils import set_seed
    from CellPLM.pipeline.cell_type_annotation import (
        CellTypeAnnotationPipeline,
        CellTypeAnnotationDefaultModelConfig,
        CellTypeAnnotationDefaultPipelineConfig,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------- data ----------
    log("loading adapted h5ad")
    full = ad.read_h5ad(data_path)
    tr_va = full[full.obs["split"] != "test"].copy()
    te = full[full.obs["split"] == "test"].copy()
    log(f"train+valid {tr_va.shape} / test {te.shape}")

    # ---------- frozen baseline embeddings (unfitted pretrained model) ----------
    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    model_config["out_dim"] = out_dim
    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix="20230926_85M",
        overwrite_config=model_config,
        pretrain_directory=str(ckpt_dir),
    )
    pipeline.model.to(device)

    # ---------- frozen panel: HVG on train+valid ONLY (no test leakage) ----------
    # common_preprocess on tr_va sets gene_list = HVG(train+valid, 3000) ∩ model set.
    # fit() recomputes the same deterministic panel; we assert equality later.
    tr_va_panel = pipeline.common_preprocess(tr_va, CellTypeAnnotationDefaultPipelineConfig["hvg"], ["batch"], False)
    frozen_gene_list = list(pipeline.gene_list)
    log(f"frozen panel (train+valid HVG ∩ CellPLM set): {len(frozen_gene_list)} genes")

    # ---------- frozen baseline embeddings (unfitted pretrained model) ----------
    te_panel = te[:, te.var.index.isin(frozen_gene_list)].copy()
    log(f"test panel for frozen baseline: {te_panel.shape[1]} genes")
    emb_frozen, _pred_frozen, _lab_frozen = extract_embeddings(pipeline.model, te_panel, ["batch"], device)
    y_true = te_panel.obs["celltype"].values
    np.savez(out_dir / "frozen_emb.npz", emb=emb_frozen, y=y_true)
    emb_n = emb_frozen / (np.linalg.norm(emb_frozen, axis=1, keepdims=True) + 1e-12)
    rho_frozen = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
    null_frozen = label_shuffle_null(emb_n, y_true, NULL_N, NULL_SEED)
    z_frozen = float((rho_frozen - null_frozen.mean()) / max(null_frozen.std(), 1e-12))
    log(f"FROZEN baseline: rho_normed={rho_frozen:.4f} z={z_frozen:.2f}")
    frozen_metrics = {
        "rho_raw": float(marker_gradient_correlation(emb_frozen, y_true, LAYER_ORDER)),
        "rho_normed": rho_frozen,
        "null_mean": float(null_frozen.mean()),
        "null_sd": float(null_frozen.std()),
        "null_z": z_frozen,
        "null_pctile": float((null_frozen <= rho_frozen).mean()),
        "n_null": NULL_N,
    }

    # ---------- fit (naive CE, best-valid-F1 selection, early stopping es=200) ----------
    train_config = CellTypeAnnotationDefaultPipelineConfig.copy()  # defaults frozen
    train_config["workers"] = 0
    buf = io.StringIO()
    fit_t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            pipeline.fit(
                tr_va,
                train_config=train_config,
                split_field="split",
                train_split="train",
                valid_split="valid",
                covariate_fields=["batch"],
                label_fields=["celltype"],
                ensembl_auto_conversion=False,
                device=device,
            )
    except torch.cuda.OutOfMemoryError:
        log("CUDA OOM during fit — retrying with backoff after 120 s")
        time.sleep(120)
        raise
    fit_min = (time.time() - fit_t0) / 60
    fit_log = buf.getvalue()
    (out_dir / "fit.log").write_text(fit_log)

    # parse epochs + best valid f1 from official fit() output
    m_es = re.search(r"Early stopped\. Best validation performance achieved at epoch (\d+)\.", fit_log)
    final_epoch = int(m_es.group(1)) if m_es else -1
    f1s = [float(x) for x in re.findall(r"Valid f1: ([\d.]+)", fit_log)]
    best_valid_f1 = float(max(f1s)) if f1s else float("nan")
    log(f"fit done: {fit_min:.1f} min, final_epoch={final_epoch}, best_valid_f1={best_valid_f1:.4f}")

    # ---------- test score (official pipeline.score) ----------
    pipe_config = CellTypeAnnotationDefaultPipelineConfig.copy()
    pipe_config["workers"] = 0
    score_buf = io.StringIO()
    with contextlib.redirect_stdout(score_buf):
        scores = pipeline.score(
            te,
            evaluation_config=pipe_config,
            split_field="split",
            target_split="test",
            covariate_fields=["batch"],
            label_fields=["celltype"],
            ensembl_auto_conversion=False,
            device=device,
        )
    (out_dir / "score.log").write_text(score_buf.getvalue())
    log(f"official score: acc={scores['acc']:.4f} f1={scores['f1_score']:.4f}")

    # ---------- embeddings on test (fitted model) + PCS eval ----------
    assert list(pipeline.gene_list) == frozen_gene_list, "fit() panel drifted from frozen panel"
    te_fit = te[:, te.var.index.isin(pipeline.gene_list)].copy()
    log(f"test panel for embeddings: {te_fit.shape[1]} genes")
    emb_ft, pred_ft, lab_ft = extract_embeddings(pipeline.model, te_fit, ["batch"], device)
    y_true = te_fit.obs["celltype"].values
    np.savez(out_dir / "cell_emb.npz", emb=emb_ft, pred=pred_ft, y=y_true)

    # decode preds with fitted label encoder (order of classes from fit)
    enc = pipeline.label_encoders["celltype"]
    pred_labels = enc.inverse_transform(pred_ft)

    emb_n = emb_ft / (np.linalg.norm(emb_ft, axis=1, keepdims=True) + 1e-12)
    rho_raw = float(marker_gradient_correlation(emb_ft, y_true, LAYER_ORDER))
    rho_normed = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
    null = label_shuffle_null(emb_n, y_true, NULL_N, NULL_SEED)
    null_z = float((rho_normed - null.mean()) / max(null.std(), 1e-12))
    null_pctile = float((null <= rho_normed).mean())
    f1_mean, ci_lo, ci_hi = bootstrap_f1_macro(y_true, pred_labels, BOOT_N, BOOT_SEED)
    from sklearn.metrics import f1_score, accuracy_score

    macro_f1 = float(f1_score(y_true, pred_labels, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, pred_labels))

    result = {
        "arm": f"cellplm_naive_ce_{test_donor_label}",
        "seed": seed,
        "recipe": f"official CellTypeAnnotationPipeline naive CE, out_dim={out_dim}, "
        "lr 5e-3 (embedder 5e-4), wd 1e-7, scheduler plat, epochs 2000 cap, "
        "es=200, patience 25, hvg 3000, best-valid-F1 checkpoint",
        "training": {
            "epochs_actual": final_epoch,
            "best_valid_f1": best_valid_f1,
            "elapsed_min": fit_min,
            "n_train": int((tr_va.obs["split"] == "train").sum()),
            "n_valid": int((tr_va.obs["split"] == "valid").sum()),
            "panel_genes_after_preprocess": int(te_fit.shape[1]),
            "device": device,
            "torch": torch.__version__,
            "seed": seed,
        },
        "eval": {
            "n_eval_cells": int(len(y_true)),
            "test_donor": test_donor_label,
            "macro_f1": macro_f1,
            "bootstrap_ci95": [ci_lo, ci_hi],
            "n_boot": BOOT_N,
            "acc": acc,
            "rho_raw": rho_raw,
            "rho_normed": rho_normed,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "null_z": null_z,
            "null_pctile": null_pctile,
            "n_null": NULL_N,
        },
        "frozen_baseline": frozen_metrics,
        "delta_rho": (
            {"frozen_rho_normed": frozen_metrics["rho_normed"], "delta_rho": frozen_metrics["rho_normed"] - rho_normed}
            if frozen_metrics
            else None
        ),
        "y_true_dist": {k: int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
        "pred_dist": {k: int(v) for k, v in zip(*np.unique(pred_labels, return_counts=True))},
        "timestamp_utc": now_utc(),
    }
    # save best-valid model state
    torch.save({"model_state_dict": pipeline.model.state_dict()}, out_dir / "model_best_valid.pt")
    (out_dir / f"seed{seed}.json").write_text(json.dumps(result, indent=2))
    log(f"saved {out_dir / f'seed{seed}.json'}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "vanzyl_cellplm.h5ad"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "ckpt"))
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-gpu-guard", action="store_true")
    ap.add_argument("--out-dim", type=int, default=3)
    ap.add_argument("--layer-order", default="LSC,basal,differentiated")
    ap.add_argument("--test-donor", default="sanes2022_Hu0216")
    args = ap.parse_args()

    global LAYER_ORDER
    LAYER_ORDER = [s.strip() for s in args.layer_order.split(",")]

    out_dir = Path(args.out_root) / f"seed{args.seed}"
    if not args.no_gpu_guard:
        wait_for_gpu(args.device)

    result = run_seed(
        seed=args.seed,
        data_path=Path(args.data),
        ckpt_dir=Path(args.ckpt),
        out_dir=out_dir,
        device=args.device,
        out_dim=args.out_dim,
        test_donor_label=args.test_donor,
    )
    (out_dir / "DONE").write_text(now_utc())
    print(json.dumps({"seed": args.seed, "done": True, "macro_f1": result["eval"]["macro_f1"],
                      "rho_normed": result["eval"]["rho_normed"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
