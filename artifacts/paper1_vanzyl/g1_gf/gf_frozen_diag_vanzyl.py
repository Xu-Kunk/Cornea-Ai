#!/usr/bin/env python
"""Geneformer V2 frozen rho diagnostic — van Zyl G1 gate (control arm).

Mirrors the scGPT frozen-diagnosis metric suite
(artifacts/scgpt_repro/frozen_diag/scgpt_frozen_diag_*.{json,log}) on the
Geneformer-V2-104M frozen arm. The weight-loading / inference / probe-F1
contract follows scripts/run_tier_b_eval.py's FROZEN arm (reference
artifacts/paper1_tier_b_eval/results_frozen_gf.json: rho=0.8064, F1=0.4421 on
GSE153515):

    BertForSequenceClassification.from_pretrained(Geneformer-V2-104M HF
    snapshot, num_labels=4, torch_dtype=bf16)  # fresh untrained 4-class head
    -> CLS embeddings = model.bert(input_ids, attn).last_hidden_state[:,0]
    -> logits = model.classifier(model.dropout(pooler_output))
    -> argmax -> 4 harmonized classes (untrained-head F1 = "probe" of the
    frozen backbone, the run_tier_b_eval frozen-arm口径).

Metric suite (LOCKED, PREREG-2026-08-11, unchanged):
    rho  : corneal_fm.evaluation.marker_gradient_correlation (raw + L2-normed;
           primary = L2-normed), 3-tier ordinal axis [LSC, basal,
           differentiated] (PREREG-2026-08-11 amendment B).
    null : n label-shuffles on the L2-normed embeddings (rng
           default_rng(seed)) -> mean/sd/z = (rho-null_mean)/null_sd, one-sided
           pctile = mean(null <= rho).
    probe: GF frozen-arm口径 = untrained-head F1 (NOT the scGPT train-split LR
           probe — no GF train split exists for van Zyl). Single deterministic
           forward -> f1_macro_sd = None. Cell-level macro/weighted over the
           3-tier axis classes present in y_true + donor-level macro (reference
           parity with results_frozen_gf.json's donor-level primary endpoint).

Analysis set (PREREG-2026-08-11 amendment A): leiden != '0' AND
cell_class_4 in {LSC, basal, differentiated} (80,817 -> 67,233 -> ~62,640).

DEVIATIONS vs the task brief (recorded, not silent):
  D1. The tokenized dataset attrs do NOT include leiden (only input_ids,
      length, cell_type, disease, tissue, donor_id, author_cell_type,
      cell_class_4). leiden (+ partition) are merged from the LOCKED h5ad obs
      (data/vanzyl_annotated.h5ad, SHA-256 fad5e9a7...67f4e8) by row order;
      row-order alignment is asserted via an exact donor_id sequence match
      between dataset attrs and h5ad obs (fails closed otherwise).
  D2. probe F1 sd is None (single deterministic untrained head, run_tier_b_eval
      frozen-arm contract); no refit repeats are possible by design.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

CORNEAL_ROOT = Path("<home>/corneal_scgpt")
sys.path.insert(0, str(CORNEAL_ROOT))  # corneal_fm package

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

DATASET_DIR = CORNEAL_ROOT / "data" / "gf_tokenized_vanzyl.dataset"
H5AD_OBS = CORNEAL_ROOT / "data" / "vanzyl_annotated.h5ad"
OUT_DEFAULT = CORNEAL_ROOT / "artifacts" / "paper1_vanzyl" / "g1_gf"
HF_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M"
)
MODEL_SAFETENSORS_SHA256 = (
    "fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f"
)

LAYER_ORDER = ["LSC", "basal", "differentiated"]  # amendment B (3-tier axis)
LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]  # run_tier_b_eval order
GF_REFERENCE = {"rho": 0.8064, "probe_f1": 0.4421}  # results_frozen_gf.json
G1_GATE = "rho>=0.5 AND z>=3 (PREREG-2026-08-11, LOCKED)"


def log(msg: str) -> None:
    print(msg, flush=True)


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def label_shuffle_null(
    emb: np.ndarray, y: np.ndarray, layer_order: list[str],
    n: int, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = marker_gradient_correlation(emb, rng.permutation(y), layer_order)
    return out


def hf_snapshot() -> Path:
    hits = sorted(_glob.glob(HF_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError(
            f"Geneformer-V2-104M snapshot not found ({HF_SNAP_GLOB})"
        )
    return Path(hits[0])


def collate_right_pad(
    input_ids_list: list, lengths_list: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad input_ids within a batch to the batch max length
    (run_tier_b_eval._collate_batch contract)."""
    max_len = max(len(ids) for ids in input_ids_list)
    padded = np.zeros((len(input_ids_list), max_len), dtype=np.int64)
    for i, ids in enumerate(input_ids_list):
        padded[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
    return (
        torch.from_numpy(padded),
        torch.from_numpy(np.asarray(lengths_list, dtype=np.int64)),
    )


def forward_cell_emb_logits(
    model: torch.nn.Module, dataset, indices: np.ndarray, device, batch: int,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """run_tier_b_eval.run_inference contract: CLS = last_hidden_state[:,0];
    logits = classifier(dropout(pooler_output)); bf16 autocast."""
    model.eval()
    embs: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    n = len(indices)
    n_batches = int(np.ceil(n / batch))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            chunk = dataset[indices[s:e].tolist()]  # analysis-set rows (order kept)
            input_ids, length = collate_right_pad(
                chunk["input_ids"], chunk["length"],
            )
            input_ids = input_ids.to(device)
            length = length.to(device)
            attn_mask = (
                torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                < length.unsqueeze(1)
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                bert_out = model.bert(input_ids, attention_mask=attn_mask)
                emb = bert_out.last_hidden_state[:, 0]  # CLS at position 0
                logits = model.classifier(model.dropout(bert_out.pooler_output))
            embs.append(emb.float().cpu().numpy())
            logits_list.append(logits.float().cpu().numpy())
            if (s // batch + 1) % 200 == 0 or s + batch >= n:
                log(f"    inference batch {s // batch + 1}/{n_batches} "
                    f"(gpu_alloc={torch.cuda.memory_allocated(device) / 1e9:.2f}GB)")
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    lgt = np.concatenate(logits_list, axis=0).astype(np.float32)
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )
    return emb, lgt, time.time() - t0, n_batches, peak_gb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[GF frozen diag vanZyl] device={device} torch={torch.__version__} "
        f"cuda={torch.version.cuda} gpu="
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"layer_order={LAYER_ORDER} batch={args.batch} n_null={args.n_null} "
        f"seed={args.seed}")

    # --- 1. weights: HF snapshot integrity + checkpoint accounting ----------
    snap = hf_snapshot()
    safetensors = snap / "model.safetensors"
    t0 = time.time()
    sha = __import__("hashlib").sha256(safetensors.read_bytes()).hexdigest()
    sha_ok = sha == MODEL_SAFETENSORS_SHA256
    log(f"snapshot: {snap}")
    log(f"model.safetensors sha256={sha} expected={MODEL_SAFETENSORS_SHA256} "
        f"match={sha_ok} ({time.time() - t0:.1f}s)")
    if not sha_ok:
        log("FATAL: snapshot integrity check failed")
        return 2

    import transformers
    from safetensors.torch import load_file
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained(
        str(snap), num_labels=4, torch_dtype=torch.bfloat16,
    )
    ckpt = load_file(str(safetensors))  # safetensors binary format
    model_sd = model.state_dict()
    missing = sorted(set(model_sd) - set(ckpt))
    unexpected = sorted(set(ckpt) - set(model_sd))
    ckpt_total = len(ckpt)
    ckpt_loaded = ckpt_total - len(unexpected)
    log(f"checkpoint keys loaded={ckpt_loaded}/{ckpt_total}")
    log(f"  missing (fresh 4-class head, run_tier_b_eval frozen-arm contract): "
        f"{missing}")
    log(f"  unexpected (MLM head from BertForMaskedLM ckpt): {unexpected}")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params={n_params / 1e6:.2f}M")
    model.to(device)

    # --- 2. input: tokenized dataset + locked h5ad obs (row-aligned) --------
    from datasets import load_from_disk

    import anndata as ad

    t0 = time.time()
    ds = load_from_disk(str(DATASET_DIR))
    log(f"dataset: {len(ds)} cells, columns {ds.column_names} "
        f"({time.time() - t0:.1f}s)")
    obs_adata = ad.read_h5ad(str(H5AD_OBS), backed="r")
    obs = obs_adata.obs
    n_obs = obs_adata.n_obs
    for col in ("leiden", "cell_class_4", "donor_id", "partition",
                "author_cell_type"):
        if col not in obs.columns:
            raise RuntimeError(f"h5ad obs missing required column {col!r}")
    donor_attr = np.asarray(ds["donor_id"], dtype=str)
    donor_h5ad = np.asarray(obs["donor_id"].astype(str).to_numpy(), dtype=str)
    align_ok = bool(np.array_equal(donor_attr, donor_h5ad))
    log(f"h5ad obs: {n_obs} rows; donor_id alignment dataset-vs-h5ad: "
        f"{align_ok}")
    if not align_ok:
        raise RuntimeError("dataset <-> h5ad row alignment FAILED (donor_id "
                           "sequence mismatch)")

    # --- 3. analysis set (PREREG amendment A) --------------------------------
    labels_all = np.asarray(obs["cell_class_4"].astype(str))
    leiden_all = np.asarray(obs["leiden"].astype(str))
    donors_all = np.asarray(obs["donor_id"].astype(str))
    partition_all = (
        np.asarray(obs["partition"].astype(str))
        if "partition" in obs.columns else np.full(n_obs, "", dtype=str)
    )
    author_all = np.asarray(obs["author_cell_type"].astype(str))
    in_axis = np.array([lbl in LAYER_ORDER for lbl in labels_all])
    not_c0 = leiden_all != "0"
    mask = not_c0 & in_axis
    c0_mask = ~not_c0
    n_excl_c0 = int(c0_mask.sum())
    overlap_cm = int((c0_mask & (author_all == "Ciliary_Muscle")).sum())
    exclusions = {
        "rule": "leiden != '0' AND cell_class_4 in layer_order "
                "(PREREG-2026-08-11 amendment A)",
        "n_input_cells": int(n_obs),
        "n_excluded_cluster0": n_excl_c0,
        "n_excluded_nonlayer_or_other": int((not_c0 & ~in_axis).sum()),
        "excluded_cluster0_overlap_author_Ciliary_Muscle": overlap_cm,
        "excluded_cluster0_overlap_frac": (
            round(overlap_cm / n_excl_c0, 4) if n_excl_c0 else None
        ),
    }
    n_analysis = int(mask.sum())
    if n_analysis == 0:
        raise RuntimeError("no cells in the analysis set")
    indices = np.where(mask)[0]
    y_true = labels_all[mask]
    counts = {str(c): int((y_true == c).sum()) for c in LAYER_ORDER}
    log(f"exclusions: {exclusions}")
    log(f"n_cells={n_analysis} label counts: {counts}")

    # --- 4. forward (analysis-set cells only) --------------------------------
    try:
        cell_emb, logits4, fwd_s, n_batches, peak_gb = forward_cell_emb_logits(
            model, ds, indices, device, batch=args.batch,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log(f"FATAL: CUDA OOM during forward: {exc}")
            log("(GPU contention with the scGPT session; exit=3 -> kill/wait/"
                "relaunch)")
            return 3
        raise
    log(f"forward: {fwd_s:.1f}s total, {fwd_s / max(n_batches, 1):.3f}s/batch "
        f"({n_batches} batches of {args.batch}), peak GPU {peak_gb:.2f} GB")
    log(f"cell_emb shape: {cell_emb.shape}")

    # --- 5. rho + null (primary = L2-normed) --------------------------------
    emb_n = l2_norm_rows(cell_emb)
    rho_raw = float(marker_gradient_correlation(cell_emb, y_true, LAYER_ORDER))
    rho_normed = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
    log(f"rho(raw)=+{rho_raw:.4f} rho(L2-normed)=+{rho_normed:.4f}")
    t0 = time.time()
    null = label_shuffle_null(
        emb_n, y_true, LAYER_ORDER, n=args.n_null, seed=args.seed
    )
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    null_z = float((rho_normed - null_mean) / max(null_sd, 1e-12))
    null_pctile = float((null <= rho_normed).mean())
    log(f"null({args.n_null} label-shuffle): mean={null_mean:+.4f} "
        f"sd={null_sd:.4f} z={null_z:+.2f} pctile={100 * null_pctile:.1f}% "
        f"({time.time() - t0:.1f}s)")

    # --- 6. probe F1 (run_tier_b_eval frozen-arm口径: untrained 4-class head) --
    preds4 = np.asarray([LAYER_ORDER_4[c] for c in np.argmax(logits4, axis=1)],
                        dtype=str)
    present = sorted(set(y_true.astype(str)) & set(LAYER_ORDER))
    f1_macro = float(f1_score(y_true, preds4, labels=present, average="macro",
                              zero_division=0))
    f1_weighted = float(f1_score(y_true, preds4, labels=present,
                                 average="weighted", zero_division=0))
    per_donor = {}
    for d in np.unique(donors_all[mask]):
        dm = donors_all[mask] == d
        per_donor[str(d)] = float(f1_score(
            y_true[dm], preds4[dm], labels=present, average="macro",
            zero_division=0,
        ))
    donor_macro = float(np.mean(list(per_donor.values())))
    pred_counts = {str(c): int((preds4 == c).sum()) for c in LAYER_ORDER_4}
    probe = {
        "arm": "frozen_gf_untrained_head (run_tier_b_eval FROZEN arm)",
        "f1_macro": f1_macro,
        "f1_macro_sd": None,  # single deterministic forward; no refit repeats
        "f1_weighted": f1_weighted,
        "f1_donor_macro": donor_macro,
        "per_donor_macro": per_donor,
        "n_repeats": 1,
        "predicted_class_counts": pred_counts,
        "note": "GF probe = untrained-head F1 (reference results_frozen_gf.json "
                "0.4421); NOT the scGPT LR probe (no GF train split for van "
                "Zyl).",
    }
    log(f"probe F1 macro={f1_macro:.4f} (sd=None, deterministic) "
        f"weighted={f1_weighted:.4f} donor_macro={donor_macro:.4f} "
        f"pred_counts={pred_counts}")

    # --- 7. results json (mirrors scGPT-side structure) ----------------------
    res = {
        "meta": {
            "model": "ctheodoris/Geneformer (Geneformer-V2-104M, HF snapshot)",
            "snapshot": str(snap),
            "model_safetensors_sha256": sha,
            "model_safetensors_sha256_ok": bool(sha_ok),
            "num_labels_head": 4,
            "head": "fresh untrained 4-class head (run_tier_b_eval FROZEN arm)",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": str(device),
            "batch": args.batch,
            "n_null": args.n_null,
            "seed": args.seed,
            "rho_primary": "L2-normed cell_emb",
            "layer_order": LAYER_ORDER,
            "checkpoint_total": ckpt_total,
            "checkpoint_loaded": ckpt_loaded,
            "checkpoint_missing": missing,
            "checkpoint_unexpected": unexpected,
            "reference_GF_frozen": GF_REFERENCE,
            "reference_json": "artifacts/paper1_tier_b_eval/results_frozen_gf.json",
            "G1_gate": G1_GATE,
            "deviations": [
                "D1 dataset attrs lack leiden -> merged from locked h5ad obs "
                "by row order; donor_id sequence alignment asserted "
                f"(align_ok={align_ok})",
                "D2 probe f1_macro_sd=None (deterministic untrained head)",
            ],
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "results": {
            "vanzyl": {
                "tag": "vanzyl",
                "dataset": str(DATASET_DIR),
                "h5ad_obs": str(H5AD_OBS),
                "label_col": "cell_class_4",
                "n_cells_input": int(n_obs),
                "n_analysis": n_analysis,
                "label_counts": counts,
                "exclusions": exclusions,
                "donor_alignment_dataset_vs_h5ad": bool(align_ok),
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
                "PASS -> GF control arm confirms axis; scGPT G1 interpretation "
                "proceeds per PREREG §6/§7"
                if (rho_normed >= 0.5 and null_z >= 3.0)
                else "FAIL -> GF frozen backbone does not natively represent "
                     "the van Zyl 3-tier axis"
            ),
        },
    }
    json_path = out_dir / "gf_frozen_diag_vanzyl.json"
    json_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"saved {json_path}")

    # --- 8. per-cell embeddings npz (analysis set) ---------------------------
    npz_path = out_dir / "gf_vanzyl_cell_emb.npz"
    np.savez(
        npz_path,
        cell_emb=cell_emb.astype(np.float32),
        labels=y_true,
        donors=donors_all[mask],
        leiden=leiden_all[mask],
        partition=partition_all[mask],
        cell_class_4=y_true,
        meta=json.dumps({"tag": "vanzyl", "n_analysis": n_analysis,
                         "timestamp_utc": res["meta"]["timestamp_utc"]}),
    )
    log(f"saved {npz_path}")
    log("[GF frozen diag vanZyl] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
