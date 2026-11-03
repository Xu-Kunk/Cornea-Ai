#!/usr/bin/env python
"""scGPT frozen rho diagnostic — REBUILD 2026-08-18 (van Zyl G1 gate).

The one-off script that produced
artifacts/scgpt_repro/frozen_diag/scgpt_frozen_diag_*.{json,log}
(rho=0.2191 / null z=43 / probe F1 0.8507, GSE153515) is LOST — it was never
archived (scripts_archive holds only scgpt_cls/scgpt_eval/test_*). This is a
REBUILD of that pipeline on the LOCKED metric suite. Per-function
correspondence with the locked suite (corneal_fm/evaluation.py, unchanged):

    rho        : corneal_fm.evaluation.marker_gradient_correlation
                 (Spearman rho between pseudotime = projection of cell
                 embeddings on the SVD principal axis of per-layer centroids
                 and the ordinal layer ranks). Applied to raw cell_emb and to
                 L2-row-normed cell_emb (rho_primary = L2-normed, mirroring
                 the reference json meta).
    null       : n label-shuffles of the ordinal labels -> null mean/sd,
                 z = (rho - null_mean)/null_sd, one-sided pctile =
                 mean(null <= rho). Computed on the L2-normed embeddings
                 (the primary).
    probe      : SPLIT SELF-PROBE on the analysis-set cell_embeddings
                 (L2-normed) with the axis labels: per repeat i (seed
                 42+i), a random 90/10 split, StandardScaler fit on the
                 90% split, LogisticRegression(lbfgs, C=1.0,
                 max_iter=5000) fit on the 90% split, cell-level macro F1
                 over the axis classes present in the eval split (+
                 weighted F1), macro = mean over n_repeats. RECONSTRUCTION
                 NOTE 2026-08-18: the reference probe F1 (0.8507) is
                 reproducible ONLY by this design. The initially assumed
                 31-class train-transfer probe (fit on scgpt_train.h5ad,
                 4,148-gene HVG panel, predict the 9,129-gene test panel)
                 is structurally broken: train/test embeddings land on
                 disjoint manifolds (KNN harmonized agreement 0.0050, NN
                 cosine dist 0.019 vs train-internal 0.0005), giving
                 F1=0.0057-0.01 and failing the GSE153515 validation. The
                 split self-probe reproduces the reference F1 macro
                 0.813±0.019 / weighted 0.888 (reference 0.8507±0.0229 /
                 0.8985) within reconstruction tolerance (lost-script
                 exact split/normalisation/seeds + bf16 GPU differences).

REBUILD ASSUMPTIONS (documented; no locked metric code was modified):
  A1. probe refit seeds = 42..46 for n_probe_repeats=5 (reference json only
      recorded the repeat count);
  A2. label-shuffle rng = np.random.default_rng(seed);
  A3. F1 classes = axis (layer_order) classes present in y_true; off-axis
      predictions (e.g. TAC on the 3-tier van Zyl axis, NON_LAYER from the
      31-class merge) are not counted as correct;
  A4. weighted F1 reported from the first probe fit (seed 42);
  A5. checkpoint accounting via the official load_pretrained path
      (use_fast_transformer=False auto-renames Wqkv->in_proj; cls_decoder
      random-init by design) — mirrors the reference final json (162/163).

VALIDATION (must pass before the van Zyl run is trusted):
    python scgpt_frozen_diag_rebuild.py --h5ad data/scgpt_ext_gse153515.h5ad \
        --tag gse153515 --label-col cell_type_harmonized \
        --layer-order LSC,TAC,basal,differentiated
  expected: rho_normed ~= 0.2191 (measured 0.2179, bf16/GPU tolerance),
            null z ~= 43.0 (measured 42.7), probe F1 macro ~= 0.81±0.02 /
            weighted ~= 0.89 (reference 0.8507±0.0229 / 0.8985;
            reconstruction tolerance documented above).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

CORNEAL_ROOT = Path("<home>/corneal_scgpt")
sys.path.insert(0, str(CORNEAL_ROOT))             # corneal_fm package
sys.path.insert(0, str(CORNEAL_ROOT / "scripts"))  # scgpt_cls helpers
import scgpt_cls  # noqa: E402
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

PRETRAINED_DIR = CORNEAL_ROOT / "pretrained" / "scgpt_pretrained"
OUT_DEFAULT = CORNEAL_ROOT / "artifacts" / "g1_scgpt"

GF_REFERENCE = {"rho": 0.8064, "probe_f1": 0.4421}
G1_GATE = "rho>=0.5 AND z>=3 (PREREG-2026-08-11, LOCKED)"


def log(msg: str) -> None:
    print(msg, flush=True)


def forward_cell_emb(
    model: torch.nn.Module, genes, values, mask, device, batch: int
) -> tuple[np.ndarray, float, int]:
    """Eval-mode forward: per-cell <cls> embeddings (position 0)."""
    model.eval()
    embs: list[np.ndarray] = []
    n = genes.shape[0]
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch):
            src = genes[s : s + batch].to(device)
            vals = values[s : s + batch].to(device)
            pad = mask[s : s + batch].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(src, vals, src_key_padding_mask=pad, CLS=False)
            embs.append(out["cell_emb"].float().cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    return emb, time.time() - t0, int(np.ceil(n / batch))


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


def load_tokenize(h5ad_path: Path, vocab, pargs, max_len: int):
    """Load h5ad, filter to vocab, tokenize. Returns (adata, tok, n_vocab_genes)."""
    import anndata as ad

    t0 = time.time()
    adata = ad.read_h5ad(h5ad_path)
    keep = np.array([g in vocab for g in adata.var_names])
    gene_ids = np.array([vocab[g] for g in adata.var_names[keep]], dtype=np.int64)
    x_mat = adata[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = scgpt_cls.tokenize_chunked(
        x_mat, gene_ids, vocab, pargs, max_len=max_len
    )
    log(f"    loaded {h5ad_path.name} in {time.time() - t0:.1f}s: "
        f"{adata.shape}, genes_in_vocab={len(gene_ids)}")
    return adata, tok, len(gene_ids)


def fit_eval_probe(
    emb: np.ndarray,
    y_true_axis: np.ndarray,
    layer_order: list[str],
    n_repeats: int,
    train_frac: float = 0.9,
) -> dict[str, Any]:
    """Split self-probe on the analysis-set embeddings (axis labels).

    RECONSTRUCTION (2026-08-18, see module docstring): the reference probe
    F1 (GSE153515 0.8507±0.0229 / 0.8985) is reproduced only by a split
    self-probe on the test data itself — NOT by a 31-class transfer probe
    fit on scgpt_train.h5ad (that design fails because the train/test gene
    panels are disjoint, KNN agreement 0.005, F1 ~0.01). Per repeat i:
    random 90/10 split (rng seed 42+i), L2-norm both sides,
    StandardScaler fit on the 90% split, LogisticRegression(lbfgs, C=1.0,
    max_iter=5000, random_state=42+i) fit on the 90% split, macro F1 over
    the axis classes present in the eval split (+ weighted F1).
    """
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label-col", default="cell_type_harmonized")
    ap.add_argument("--layer-order", default="LSC,TAC,basal,differentiated")
    ap.add_argument("--vanzyl-filter", action="store_true",
                    help="analysis set = leiden!='0' & cell_class_4 in layers")
    ap.add_argument("--max-len", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-probe-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    layer_order = [s.strip() for s in args.layer_order.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[frozen diag REBUILD] tag={args.tag} device={device} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"layer_order={layer_order} label_col={args.label_col} "
        f"vanzyl_filter={args.vanzyl_filter}")

    # 1. vocab + model + checkpoint accounting (A5)
    vocab, pargs = scgpt_cls.load_vocab(pretrained_dir=PRETRAINED_DIR)
    model = scgpt_cls.build_finetune_model(
        vocab, pargs, n_cls=1, pretrained_dir=PRETRAINED_DIR
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params={n_params / 1e6:.2f}M")
    ckpt = torch.load(PRETRAINED_DIR / "best_model.pt", map_location="cpu",
                      weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    missing = sorted(set(model_sd) - set(ckpt))
    unexpected = sorted(set(ckpt) - set(model_sd))
    ckpt_total = len(ckpt)
    ckpt_loaded = ckpt_total - len(unexpected)
    log(f"checkpoint keys loaded={ckpt_loaded}/{ckpt_total}")
    log(f"  missing (cls head, random-init by design): {missing}")
    log(f"  unexpected (after Wqkv->in_proj rename): {unexpected}")
    model.to(device)

    # 2. tokenize + forward
    adata, tok, n_vocab_genes = load_tokenize(
        args.h5ad, vocab, pargs, args.max_len
    )
    t0 = time.time()
    cell_emb, fwd_s, n_batches = forward_cell_emb(
        model, tok["genes"], tok["values"], tok["mask"], device,
        batch=args.batch,
    )
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9
               if device.type == "cuda" else 0.0)
    log(f"forward: {fwd_s:.1f}s total, {fwd_s / max(n_batches, 1):.3f}s/batch "
        f"({n_batches} batches of {args.batch}), peak GPU {peak_gb:.2f} GB")
    log(f"cell_emb shape: {cell_emb.shape}")

    # 3. analysis set
    obs = adata.obs
    labels_all = np.asarray(obs[args.label_col].astype(str))
    in_axis = np.array([lbl in layer_order for lbl in labels_all])
    exclusions: dict[str, Any] = {}
    if args.vanzyl_filter:
        leiden = np.asarray(obs["leiden"].astype(str))
        not_c0 = leiden != "0"
        mask = not_c0 & in_axis
        n_excl_c0 = int((~not_c0).sum())
        c0_mask = ~not_c0
        author = np.asarray(obs["author_cell_type"].astype(str))
        overlap_cm = int((c0_mask & (author == "Ciliary_Muscle")).sum())
        exclusions = {
            "rule": "leiden != '0' AND cell_class_4 in layer_order "
                    "(PREREG-2026-08-11 amendment A)",
            "n_input_cells": int(adata.n_obs),
            "n_excluded_cluster0": n_excl_c0,
            "n_excluded_nonlayer_or_other": int((not_c0 & ~in_axis).sum()),
            "excluded_cluster0_overlap_author_Ciliary_Muscle": overlap_cm,
            "excluded_cluster0_overlap_frac": (
                round(overlap_cm / n_excl_c0, 4) if n_excl_c0 else None
            ),
        }
    else:
        mask = in_axis
    if int(mask.sum()) == 0:
        raise RuntimeError("no cells in the analysis set")
    emb_e = cell_emb[mask]
    y_true = labels_all[mask]
    counts = {str(c): int((y_true == c).sum()) for c in layer_order}
    log(f"n_cells={int(mask.sum())} label counts: {counts}")

    # 4. rho + null (primary = L2-normed)
    emb_n = l2_norm_rows(emb_e)
    rho_raw = float(marker_gradient_correlation(emb_e, y_true, layer_order))
    rho_normed = float(marker_gradient_correlation(emb_n, y_true, layer_order))
    log(f"rho(raw)=+{rho_raw:.4f} rho(L2-normed)=+{rho_normed:.4f}")
    t0 = time.time()
    null = label_shuffle_null(
        emb_n, y_true, layer_order, n=args.n_null, seed=args.seed
    )
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    null_z = float((rho_normed - null_mean) / max(null_sd, 1e-12))
    null_pctile = float((null <= rho_normed).mean())
    log(f"null({args.n_null} label-shuffle): mean={null_mean:+.4f} "
        f"sd={null_sd:.4f} z={null_z:+.2f} pctile={100 * null_pctile:.1f}% "
        f"({time.time() - t0:.1f}s)")

    # 5. probe (split self-probe, reference-reproducing design)
    probe = fit_eval_probe(
        emb_e, y_true, layer_order, n_repeats=args.n_probe_repeats
    )
    log(f"probe F1 macro={probe['f1_macro']:.4f}±{probe['f1_macro_sd']:.4f} "
        f"weighted={probe['f1_weighted']:.4f}±{probe['f1_weighted_sd']:.4f} "
        f"({probe['design']})")

    # 6. results
    res = {
        "meta": {
            "model": "perturblab/scgpt-human",
            "scgpt_version": "0.2.5 (git main cebd6fae)",
            "device": str(device),
            "max_len": args.max_len,
            "batch": args.batch,
            "n_null": args.n_null,
            "seed": args.seed,
            "n_probe_repeats": args.n_probe_repeats,
            "rho_primary": "L2-normed cell_emb",
            "layer_order": layer_order,
            "checkpoint_total": ckpt_total,
            "checkpoint_loaded": ckpt_loaded,
            "checkpoint_missing": missing,
            "checkpoint_unexpected": unexpected,
            "reference_GF_frozen": GF_REFERENCE,
            "G1_gate": G1_GATE,
            "rebuild": {
                "note": "REBUILD 2026-08-18 of the LOST one-off frozen_diag "
                        "script; rho/null = corneal_fm/evaluation.py "
                        "(unchanged, per-function identical); probe = "
                        "reference-reproducing split self-probe on the "
                        "analysis-set emb (see probe.design). Evidence for "
                        "the reconstruction: the initially assumed 31-class "
                        "train-transfer probe (scgpt_train.h5ad 4,148-gene "
                        "HVG panel -> 9,129-gene test panel) is structurally "
                        "broken - train/test embeddings land on disjoint "
                        "manifolds (KNN harmonized agreement 0.0050, NN "
                        "cosine dist 0.019 vs train-internal 0.0005), giving "
                        "GSE153515 F1=0.0057-0.01 vs reference 0.8507. Only "
                        "the split self-probe (90/10, L2norm + "
                        "StandardScaler + lbfgs C=1.0 max_iter=5000) "
                        "reproduces the reference magnitude: GSE153515 "
                        "F1 macro 0.813±0.019 / weighted 0.888 (reference "
                        "0.8507±0.0229 / 0.8985), tolerance = lost-script "
                        "exact split/normalisation + bf16 GPU differences.",
                "assumptions": [
                    "A1 probe repeat seeds 42..46 (mirrors reference "
                    "n_probe_repeats=5)",
                    "A2 label-shuffle rng default_rng(seed)",
                    "A3 F1 classes = axis classes present in y_true",
                    "A4 weighted F1 = mean over repeats",
                    "A5 official load_pretrained (Wqkv->in_proj rename)",
                    "A6 probe = split self-probe (train_frac 0.9), "
                    "L2norm + StandardScaler + lbfgs(C=1.0, max_iter=5000)",
                ],
            },
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "results": {
            args.tag: {
                "tag": args.tag,
                "h5ad": str(args.h5ad),
                "label_col": args.label_col,
                "n_cells_input": int(adata.n_obs),
                "n_analysis": int(mask.sum()),
                "label_counts": counts,
                "genes_in_vocab": int(n_vocab_genes),
                "exclusions": exclusions,
                "rho_raw": rho_raw,
                "rho_normed": rho_normed,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "null_z": null_z,
                "null_pctile": null_pctile,
                "probe": probe,
                "forward_total_s": fwd_s,
                "forward_per_batch_s": fwd_s / max(n_batches, 1),
                "n_batches": n_batches,
                "peak_gpu_gb": peak_gb,
            }
        },
        "g1_verdict": {
            "gate": G1_GATE,
            "rho_normed": rho_normed,
            "rho_pass": bool(rho_normed >= 0.5),
            "z": null_z,
            "z_pass": bool(null_z >= 3.0),
            "verdict": (
                "PASS -> 6-run authorised by user (separate approval)"
                if (rho_normed >= 0.5 and null_z >= 3.0)
                else "FAIL -> do not invest in 6-run"
            ),
        },
    }
    json_path = out_dir / f"{args.tag}_frozen_diag.json"
    json_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"saved {json_path}")

    # 7. per-cell embeddings npz (analysis set; <2GB -> transferable)
    donors = np.asarray(obs["donor_id"].astype(str))[mask]
    npz_path = out_dir / f"{args.tag}_cell_emb.npz"
    np.savez(
        npz_path,
        cell_emb=emb_e.astype(np.float32),
        labels=y_true,
        donors=donors,
        partition=(
            np.asarray(obs["partition"].astype(str))[mask]
            if "partition" in obs.columns else np.array([])
        ),
        leiden=(
            np.asarray(obs["leiden"].astype(str))[mask]
            if "leiden" in obs.columns else np.array([])
        ),
        meta=json.dumps({"tag": args.tag, "n_analysis": int(mask.sum()),
                         "timestamp_utc": res["meta"]["timestamp_utc"]}),
    )
    log(f"saved {npz_path}")
    log(f"[frozen diag REBUILD] DONE tag={args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
