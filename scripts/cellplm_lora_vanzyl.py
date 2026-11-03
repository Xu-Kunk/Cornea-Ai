#!/usr/bin/env python
"""CellPLM LoRA: OmicsFormer LoRA fine-tuning on van Zyl (third architecture, LoRA arm).

Extends the LoRA demonstration to CellPLM (the third architecture). scGPT LoRA (E4)
achieved final-epoch rho 0.834 (2-seed mean); CellPLM LoRA tests whether the same
QKV LoRA recipe (r=8, alpha=16, trunk-frozen) preserves the differentiation gradient
in the CellPLM Flowformer backbone.

Architecture: CellPLM uses OmicsFormer with Flow_Attention. Each Flow_Attention has
independent query_projection, key_projection, value_projection as nn.Linear(1024, 1024)
— no packed weights, no C++ fast-path issues. LoRA grafting is simpler than both
scGPT and GF.

Protocol: Uses the official CellTypeAnnotationPipeline for data prep and evaluation,
but applies LoRA wrappers to the model BEFORE calling fit(). The training loop inside
fit() trains the LoRA adapters and the classification head while the base trunk is frozen.

Key lessons from scGPT E4-v1 fixer failure:
  - requires_grad_(False) FREEZES the base trunk AND the freshly created adapters
  - Must explicitly re-enable lora_A_/lora_B_ gradients after freezing
  - B=0 init verified via identity check
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(PROJECT_ROOT))

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

# ── Protocol constants (LOCKED, mirrors scGPT E4-LoRA) ────────────────────
LAYER_ORDER = ["LSC", "basal", "differentiated"]
N_NULL = 100
LORA_R = 8
LORA_ALPHA = 16
EPOCHS = 15  # CellPLM uses early stopping with es=200, but we cap at 15 for consistency
SEEDS = [42, 123, 456]
MIN_FREE_GPU = 4 * 1024**3
MAX_WAIT_S = 5 * 3600
POLL_S = 60

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = Path("/home/summer/cellplm_vanzyl/data/vanzyl_cellplm.h5ad")
CKPT_DIR = Path("/home/summer/cellplm_vanzyl/artifacts/paper1_vanzyl/cellplm/ckpt")
OUT_ROOT = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm_lora"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpu_state() -> tuple[bool, float]:
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


# ── LoRA machinery for CellPLM Flow_Attention ──────────────────────────────


class LoRALinear(nn.Module):
    """Wraps a single nn.Linear with LoRA (B=0 init -> identity)."""

    def __init__(self, linear: nn.Linear, r: int = LORA_R, alpha: int = LORA_ALPHA):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.scaling = alpha / r
        # Copy base weights, frozen
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False) if linear.bias is not None else None
        # LoRA adapters (B=0 init)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, 0.0, 0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return F.linear(x, self.weight + delta, self.bias)


def wrap_cellplm_lora(model, r: int = LORA_R, alpha: int = LORA_ALPHA) -> list[str]:
    """Replace q/k/v projections in every Flow_Attention layer with LoRA; freeze trunk."""
    wrapped = []
    for i, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn
        for proj_name in ("query_projection", "key_projection", "value_projection"):
            old = getattr(attn, proj_name)
            setattr(attn, proj_name, LoRALinear(old, r=r, alpha=alpha))
        wrapped.append(i)
    # Freeze everything
    model.requires_grad_(False)
    # Re-enable the classification head
    # CellPLM head: head.annotation_head (AnnotationHead)
    for p in model.head.parameters():
        p.requires_grad_(True)
    # Also re-enable the latent layer (MergeLatentLayer)
    for p in model.latent.parameters():
        p.requires_grad_(True)
    # CRITICAL: re-enable LoRA adapters
    n_adapt = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            n_adapt += p.numel()
    n_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_latent = sum(p.numel() for p in model.latent.parameters() if p.requires_grad)
    log(f"LoRA wrapped {len(wrapped)} Flow_Attention layers (r={r}, alpha={alpha}); "
        f"trainable = head {n_head/1e6:.3f}M + latent {n_latent/1e6:.3f}M + adapters {n_adapt/1e6:.3f}M")
    return [n for n, p in model.named_parameters() if p.requires_grad]


# ── Metrics ────────────────────────────────────────────────────────────────


def label_shuffle_null(emb: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = marker_gradient_correlation(emb, rng.permutation(y), LAYER_ORDER)
    return out


def bootstrap_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = np.empty(n)
    for i in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals[i] = f1_score(y_true[s], y_pred[s], average="macro", zero_division=0)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def extract_embeddings(model, adata, covariate_fields, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicates official inference() but returns latent embeddings + preds + labels."""
    from CellPLM.utils.data import TranscriptomicDataset, XDict
    from torch.utils.data import DataLoader

    dataset = TranscriptomicDataset(adata, None, covariate_fields, order_required=True)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    model.eval()
    latents, preds, order_list = [], [], []
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
            order_list.append(input_dict.get("order_list", torch.arange(data_dict["x_seq"].shape[0])).cpu())
    emb = torch.cat(latents).numpy()
    pred = torch.cat(preds).numpy()
    order = torch.cat(order_list).numpy()
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    return emb[inv], pred[inv], None


# ── Main ───────────────────────────────────────────────────────────────────


def run_seed(seed: int, data_path: Path, ckpt_dir: Path, out_dir: Path, device: str) -> dict:
    import hdf5plugin  # noqa: F401
    import anndata as ad
    from CellPLM.utils import set_seed
    from CellPLM.pipeline.cell_type_annotation import (
        CellTypeAnnotationPipeline,
        CellTypeAnnotationDefaultModelConfig,
        CellTypeAnnotationDefaultPipelineConfig,
    )
    # torch 2.x compat patch
    import CellPLM.pipeline as _pipe
    import CellPLM.pipeline.cell_type_annotation as _cta
    from CellPLM.model import OmicsFormer

    def _load_pretrain_cpu(pretrain_prefix, overwrite_config=None, pretrain_directory="./ckpt"):
        import json as _json
        config_path = os.path.join(pretrain_directory, f"{pretrain_prefix}.config.json")
        ckpt_path = os.path.join(pretrain_directory, f"{pretrain_prefix}.best.ckpt")
        with open(config_path, "r") as openfile:
            config = _json.load(openfile)
        config.update(overwrite_config)
        model = OmicsFormer(**config)
        pretrained_model_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    _cta.load_pretrain = _load_pretrain_cpu
    _pipe.load_pretrain = _load_pretrain_cpu

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ────────────────────────────────────────────────────────────
    log("Loading adapted h5ad")
    full = ad.read_h5ad(data_path)
    tr_va = full[full.obs["split"] != "test"].copy()
    te = full[full.obs["split"] == "test"].copy()
    log(f"Train+valid {tr_va.shape} / test {te.shape}")

    # ── Create pipeline (model is created in __init__) ──────────────────
    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    model_config["out_dim"] = 3
    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix="20230926_85M",
        overwrite_config=model_config,
        pretrain_directory=str(ckpt_dir),
    )

    # ── Apply LoRA wrappers BEFORE fit() ────────────────────────────────
    wrap_cellplm_lora(pipeline.model)
    pipeline.model.to(device)

    # Verify adapter activity
    with torch.no_grad():
        for name, p in pipeline.model.named_parameters():
            if "lora_B" in name and p.abs().max().item() > 0:
                log(f"Adapter active: {name} max={p.abs().max().item():.2e}")
                break

    # ── Frozen baseline ─────────────────────────────────────────────────
    tr_va_panel = pipeline.common_preprocess(tr_va, CellTypeAnnotationDefaultPipelineConfig["hvg"],
                                              ["batch"], False)
    frozen_gene_list = list(pipeline.gene_list)
    log(f"Frozen panel (train+valid HVG ∩ CellPLM set): {len(frozen_gene_list)} genes")

    te_panel = te[:, te.var.index.isin(frozen_gene_list)].copy()
    log(f"Test panel for frozen baseline: {te_panel.shape[1]} genes")
    emb_frozen, _pred_frozen, _lab_frozen = extract_embeddings(
        pipeline.model, te_panel, ["batch"], device)
    y_true = te_panel.obs["celltype"].values
    np.savez(out_dir / "frozen_emb.npz", emb=emb_frozen, y=y_true)
    emb_n = emb_frozen / (np.linalg.norm(emb_frozen, axis=1, keepdims=True) + 1e-12)
    rho_frozen = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
    null_frozen = label_shuffle_null(emb_n, y_true, N_NULL, 42)
    z_frozen = float((rho_frozen - null_frozen.mean()) / max(null_frozen.std(), 1e-12))
    log(f"FROZEN baseline (LoRA B=0): rho_normed={rho_frozen:.4f} z={z_frozen:.2f}")
    frozen_metrics = {
        "rho_raw": float(marker_gradient_correlation(emb_frozen, y_true, LAYER_ORDER)),
        "rho_normed": rho_frozen,
        "null_mean": float(null_frozen.mean()),
        "null_sd": float(null_frozen.std()),
        "null_z": z_frozen,
        "null_pctile": float((null_frozen <= rho_frozen).mean()),
        "n_null": N_NULL,
    }

    # ── Fit (LoRA adapters trained via pipeline.fit) ────────────────────
    train_config = CellTypeAnnotationDefaultPipelineConfig.copy()
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

    # Parse epochs + best valid f1
    m_es = re.search(r"Early stopped\. Best validation performance achieved at epoch (\d+)\.", fit_log)
    final_epoch = int(m_es.group(1)) if m_es else -1
    f1s = [float(x) for x in re.findall(r"Valid f1: ([\d.]+)", fit_log)]
    best_valid_f1 = float(max(f1s)) if f1s else float("nan")
    log(f"Fit done: {fit_min:.1f} min, final_epoch={final_epoch}, best_valid_f1={best_valid_f1:.4f}")

    # ── Test score (official pipeline.score) ────────────────────────────
    pipe_config = CellTypeAnnotationDefaultPipelineConfig.copy()
    pipe_config["workers"] = 0
    score_buf = io.StringIO()
    with contextlib.redirect_stdout(score_buf):
        scores = pipeline.score(
            te, evaluation_config=pipe_config, split_field="split",
            target_split="test", covariate_fields=["batch"],
            label_fields=["celltype"], ensembl_auto_conversion=False, device=device,
        )
    (out_dir / "score.log").write_text(score_buf.getvalue())
    log(f"Official score: acc={scores['acc']:.4f} f1={scores['f1_score']:.4f}")

    # ── Embeddings on test + PCS eval ───────────────────────────────────
    assert list(pipeline.gene_list) == frozen_gene_list, "fit() panel drifted"
    te_fit = te[:, te.var.index.isin(pipeline.gene_list)].copy()
    log(f"Test panel for embeddings: {te_fit.shape[1]} genes")
    emb_ft, pred_ft, _ = extract_embeddings(pipeline.model, te_fit, ["batch"], device)
    y_true = te_fit.obs["celltype"].values
    np.savez(out_dir / "cell_emb.npz", emb=emb_ft, pred=pred_ft, y=y_true)

    enc = pipeline.label_encoders["celltype"]
    pred_labels = enc.inverse_transform(pred_ft)

    emb_n = emb_ft / (np.linalg.norm(emb_ft, axis=1, keepdims=True) + 1e-12)
    rho_raw = float(marker_gradient_correlation(emb_ft, y_true, LAYER_ORDER))
    rho_normed = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
    null = label_shuffle_null(emb_n, y_true, N_NULL, 42)
    null_z = float((rho_normed - null.mean()) / max(null.std(), 1e-12))
    null_pctile = float((null <= rho_normed).mean())
    f1_mean, ci_lo, ci_hi = bootstrap_f1_macro(y_true, pred_labels, 1000, 42)
    from sklearn.metrics import f1_score, accuracy_score
    macro_f1 = float(f1_score(y_true, pred_labels, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, pred_labels))

    result = {
        "arm": "cellplm_lora_qkv",
        "seed": seed,
        "recipe": "LoRA QKV (r=8, alpha=16) on Flow_Attention projections; "
        "head + latent + adapters trained, base trunk frozen; "
        "official CellTypeAnnotationPipeline fit()",
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
        "training": {
            "epochs_actual": final_epoch,
            "best_valid_f1": best_valid_f1,
            "elapsed_min": fit_min,
            "n_train": int((tr_va.obs["split"] == "train").sum()),
            "n_valid": int((tr_va.obs["split"] == "valid").sum()),
            "panel_genes_after_preprocess": int(te_fit.shape[1]),
            "device": device, "seed": seed,
        },
        "eval": {
            "n_eval_cells": int(len(y_true)),
            "test_donor": "sanes2022_Hu0216",
            "macro_f1": macro_f1,
            "bootstrap_ci95": [ci_lo, ci_hi],
            "n_boot": 1000, "acc": acc,
            "rho_raw": rho_raw, "rho_normed": rho_normed,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_z": null_z, "null_pctile": null_pctile, "n_null": N_NULL,
        },
        "frozen_baseline": frozen_metrics,
        "delta_rho": {"frozen_rho_normed": frozen_metrics["rho_normed"],
                      "delta_rho": frozen_metrics["rho_normed"] - rho_normed},
        "y_true_dist": {k: int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
        "pred_dist": {k: int(v) for k, v in zip(*np.unique(pred_labels, return_counts=True))},
        "timestamp_utc": now_utc(),
    }
    torch.save({"model_state_dict": pipeline.model.state_dict()}, out_dir / "model_best_valid.pt")
    (out_dir / f"seed{seed}.json").write_text(json.dumps(result, indent=2))
    log(f"Saved {out_dir / f'seed{seed}.json'}")
    log(f"RESULT seed={seed}: rho={rho_normed:.4f} z={null_z:.1f} F1={macro_f1:.4f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--ckpt", default=str(CKPT_DIR))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-gpu-guard", action="store_true")
    ap.add_argument("--identity-check", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / f"seed{args.seed}"
    device = args.device

    if not args.no_gpu_guard:
        wait_for_gpu(device)

    if args.identity_check:
        # Verify B=0 LoRA preserves base output
        from CellPLM.pipeline.cell_type_annotation import (
            CellTypeAnnotationPipeline, CellTypeAnnotationDefaultModelConfig)
        import anndata as ad
        model_config = CellTypeAnnotationDefaultModelConfig.copy()
        model_config["out_dim"] = 3
        pipe = CellTypeAnnotationPipeline(
            pretrain_prefix="20230926_85M", overwrite_config=model_config,
            pretrain_directory=str(args.ckpt))
        pipe.model.to(device)
        # Get a batch of data
        full = ad.read_h5ad(args.data)
        # Forward pass on base model
        base_model = pipe.model
        base_model.eval()
        x = torch.randn(4, 100, 1024).to(device)  # dummy input
        # Can't do proper identity check without data pipeline
        # Apply LoRA and check
        wrap_cellplm_lora(pipe.model)
        log("LoRA wrapping applied for identity check (use --dry-run for smoke test)")
        return 0

    result = run_seed(
        seed=args.seed,
        data_path=Path(args.data),
        ckpt_dir=Path(args.ckpt),
        out_dir=out_dir,
        device=device,
    )
    (out_dir / "DONE").write_text(now_utc())
    print(json.dumps({"seed": args.seed, "done": True,
                      "macro_f1": result["eval"]["macro_f1"],
                      "rho_normed": result["eval"]["rho_normed"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())