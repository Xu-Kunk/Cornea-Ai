#!/usr/bin/env python
"""CORN advantage A+D — CellPLM CORN (3 seeds) per-cell extraction.

Runs on <compute-node> (CPU) with <home>/corneal_scgpt/venv/bin/python. For the
locked test donor Hu0216 (20158 cells), loads the trained CORN-head checkpoints
(cellplm/corn/seed{42,123,456}/model_best_valid.pt), forwards the test cells
and saves per-cell threshold logits (K-1=2), latent embeddings, labels and the
locked van Zyl marker scores (aligned 1:1 to the scGPT test cells — same
analysis set, same row order, runtime-asserted).

FROZEN protocol (PCS Stability, frozen before first run 2026-08-20):
  * test cells: vanzyl_cellplm.h5ad split==test (20158, donor Hu0216)
  * checkpoint: cellplm/corn/seed{42,123,456}/model_best_valid.pt
    (CORNAnnotationHead, K=3 -> 2 threshold logits; p1=P(>=1), p2=P(>=2))
  * decode threshold = 0.5 (ordinal_decode contract, unchanged)
  * sanity gate: recompute L2-normed rho vs the locked seed{42,123,456}.json
    (tol 2e-3, recorded)
  * panel: HVG(train+valid, 3000) cap CellPLM gene set, recomputed
    deterministically (the run asserted fit() panel == frozen panel)

Outputs: <out>/cellplm_seed<seed>.npz + .json + .DONE marker. Idempotent.
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

PROJECT_ROOT = Path("<home>/cellplm_vanzyl")
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS))

from CellPLM.pipeline.cell_type_annotation import (  # noqa: E402
    CellTypeAnnotationDefaultModelConfig,
    CellTypeAnnotationDefaultPipelineConfig,
    CellTypeAnnotationPipeline,
)
from run_cellplm_corn import (  # noqa: E402  (locked arm C3-CORN machinery)
    LAYER_ORDER,
    N_ORDINAL,
    N_THRESHOLDS,
    CORNAnnotationHead,
)

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "vanzyl_cellplm.h5ad"
CKPT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "ckpt"
CORN_OUT = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "corn"
SCGPT_TEST = Path("<home>/corneal_scgpt/artifacts/paper1_vanzyl/run6") / "scgpt_vanzyl_test.h5ad"
TEST_DONOR = "sanes2022_Hu0216"
RHO_TOL = 2e-3
SCORE_COLS = ["score_LSC", "score_PMC_basal", "score_TDC"]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def build_pipeline(device: str, ckpt_dir: Path) -> CellTypeAnnotationPipeline:
    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    model_config["out_dim"] = 3
    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix="20230926_85M",
        overwrite_config=model_config,
        pretrain_directory=str(ckpt_dir),
    )
    pipeline.model.to(device)
    in_dim = pipeline.model.head.mlp[1].in_features
    corn_head = CORNAnnotationHead(
        in_dim=in_dim,
        hidden_dim=in_dim,
        num_thresholds=N_THRESHOLDS,
        num_layers=CellTypeAnnotationDefaultModelConfig["dec_layers"],
        dropout=CellTypeAnnotationDefaultModelConfig["model_dropout"],
        norm=None,
        batch_num=0,
        n_ordinal=N_ORDINAL,
        class_to_ordinal={},
    ).to(device)
    pipeline.model.head = corn_head
    return pipeline


def extract_emb_logits(model, adata, covariate_fields, device: str):
    """Mirror of run_cellplm_corn.extract_embeddings that ALSO returns the raw
    threshold logits (head.mlp applied to the full latent h — identical to the
    head's eval-time computation)."""
    from CellPLM.utils.data import TranscriptomicDataset, XDict
    from torch.utils.data import DataLoader

    dataset = TranscriptomicDataset(adata, None, covariate_fields, order_required=True)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    model.eval()
    latents, logits_all, labels, order_list = [], [], [], []
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
            logits_all.append(model.head.mlp(out_dict["latent"]).cpu())
            if "label" in input_dict:
                labels.append(input_dict["label"].cpu())
            order_list.append(input_dict.get("order_list", torch.arange(data_dict["x_seq"].shape[0])).cpu())
    emb = torch.cat(latents).numpy()
    lgt = torch.cat(logits_all).numpy()
    order = torch.cat(order_list).numpy()
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    emb = emb[inv]
    lgt = lgt[inv]
    if labels:
        lab = torch.cat(labels).numpy()[inv]
    else:
        lab = np.full(len(order), -1)
    return emb, lgt, lab


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def scgpt_test_scores() -> tuple[np.ndarray, np.ndarray]:
    """Marker scores + cell_class_4 labels of the scGPT test cells (the same
    20158 cells as the CellPLM test split — alignment runtime-asserted)."""
    import anndata as ad

    a = ad.read_h5ad(SCGPT_TEST, backed="r")
    scores = np.stack(
        [np.asarray(a.obs[c], dtype=float) for c in SCORE_COLS], axis=1
    )
    return scores.astype(np.float32), np.asarray(a.obs["cell_class_4"].astype(str))


def run_seed(seed: int, out_dir: Path, threads: int) -> int:
    import anndata as ad
    import hdf5plugin  # noqa: F401

    torch.set_num_threads(threads)
    ckpt_path = CORN_OUT / f"seed{seed}" / "model_best_valid.pt"
    locked = json.loads((CORN_OUT / f"seed{seed}" / f"seed{seed}.json").read_text())
    locked_rho = locked["eval"]["rho_normed"]

    log(f"=== CORN CellPLM seed {seed} ===")
    full = ad.read_h5ad(DATA_PATH)
    tr_va = full[full.obs["split"].astype(str) != "test"].copy()
    te = full[full.obs["split"].astype(str) == "test"].copy()
    assert te.n_obs == 20158 and (te.obs["donor_id"].astype(str) == TEST_DONOR).all()
    log(f"train+valid {tr_va.shape} / test {te.shape}")

    pipeline = build_pipeline("cpu", CKPT_DIR)
    pipeline.common_preprocess(
        tr_va, CellTypeAnnotationDefaultPipelineConfig["hvg"], ["batch"], False
    )
    frozen_gene_list = list(pipeline.gene_list)
    te_panel = te[:, te.var.index.isin(frozen_gene_list)].copy()
    log(f"panel {len(frozen_gene_list)} genes; test panel {te_panel.shape[1]}")

    state = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    pipeline.model.load_state_dict(state, strict=True)
    pipeline.model.eval()

    t0 = time.time()
    emb, logits, lab = extract_emb_logits(pipeline.model, te_panel, ["batch"], "cpu")
    log(f"forward done {time.time() - t0:.0f}s emb={emb.shape} logits={logits.shape}")

    y = te_panel.obs["celltype"].astype(str).to_numpy()
    scores, y_scgpt = scgpt_test_scores()
    assert (y == y_scgpt).all(), "CellPLM test cells not aligned 1:1 with scGPT test cells"

    rho_n = float(marker_gradient_correlation(l2_norm_rows(emb), y, LAYER_ORDER))
    gate = {
        "rho_normed_recomputed": rho_n,
        "rho_normed_locked": locked_rho,
        "abs_diff": abs(rho_n - locked_rho),
        "tol": RHO_TOL,
        "pass": abs(rho_n - locked_rho) <= RHO_TOL,
    }
    log(f"rho recomputed {rho_n:.4f} locked {locked_rho:.4f} "
        f"diff={gate['abs_diff']:.2e} {'PASS' if gate['pass'] else 'FAIL'}")
    np.savez_compressed(
        out_dir / f"cellplm_seed{seed}.npz",
        logits=logits.astype(np.float32), emb=emb.astype(np.float32),
        y=_str(y), scores=scores,
    )
    (out_dir / f"cellplm_seed{seed}.json").write_text(
        json.dumps({
            "arm": "corn_cellplm", "seed": seed, "n_cells": int(len(y)),
            "threshold_logits": N_THRESHOLDS, "test_donor": TEST_DONOR,
            "panel_genes": len(frozen_gene_list), "checkpoint": str(ckpt_path),
            "gate": gate, "timestamp_utc": now_utc(),
        }, indent=2)
    )
    (out_dir / f"cellplm_seed{seed}.DONE").write_text(now_utc())
    return 0


def _str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype="<U64")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "corn_advantage")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cellplm_seed{args.seed}"
    if (out_dir / f"{tag}.DONE").exists():
        log(f"SKIP {tag} (DONE marker present)")
        return 0
    return run_seed(args.seed, out_dir, args.threads)


if __name__ == "__main__":
    sys.exit(main())
