#!/usr/bin/env python
"""Arm B (PREREG-2026-08-11 §9) companion evaluation: L2-normed rho +
label-shuffle null for the 1-epoch GF checkpoints on the GSE153515 LOCKED
external test.

run_tier_b_eval.py's CE arm already reports donor-level macro-F1 + bootstrap CI
+ raw marker_gradient_rank_correlation. This companion adds the frozen-
diagnosis convention: rho(raw) and rho(L2-normed) via
corneal_fm.evaluation.marker_gradient_correlation, plus a 100-shuffle
label-shuffle null on the L2-normed embeddings (primary), rng default_rng(42),
matching the G1 frozen diagnosis (gf_frozen_diag_vanzyl.py / scgpt rebuild).

Input contract mirrors run_tier_b_eval: cached tokenized external dataset
(external_tokenized/cell.dataset) + h5ad obs alignment (donor_id sequence
assert) + the final-epoch checkpoint per seed (save/caft_gf_lam0.0_seed{seed}_*).
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

CORNEAL_ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(CORNEAL_ROOT))

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

SAVE_DIR = CORNEAL_ROOT / "save"
DATA = CORNEAL_ROOT / "data"
TOKEN_CACHE = CORNEAL_ROOT / "artifacts" / "paper1_tier_b_eval" / "external_tokenized"
EXT_H5AD = DATA / "gse153515_annotated.h5ad"
OUT_DEFAULT = CORNEAL_ROOT / "artifacts" / "paper1_vanzyl" / "gf_official_1epoch"
LAYER_ORDER = ["LSC", "TAC", "basal", "differentiated"]


def log(msg: str) -> None:
    print(msg, flush=True)


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def label_shuffle_null(emb, y, layer_order, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = marker_gradient_correlation(emb, rng.permutation(y), layer_order)
    return out


def collate_right_pad(input_ids_list, lengths_list):
    max_len = max(len(ids) for ids in input_ids_list)
    padded = np.zeros((len(input_ids_list), max_len), dtype=np.int64)
    for i, ids in enumerate(input_ids_list):
        padded[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
    return (
        torch.from_numpy(padded),
        torch.from_numpy(np.asarray(lengths_list, dtype=np.int64)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--seed-null", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[armb extra] seed={args.seed} device={device}")

    # --- checkpoint discovery (mirrors run_tier_b_eval.discover_checkpoints) --
    pat = f"caft_gf_lam0.0_seed{args.seed}_*"
    save_dirs = sorted(SAVE_DIR.glob(pat))
    if not save_dirs:
        raise RuntimeError(f"no save dir for {pat}")
    save_dir = save_dirs[-1]
    ckpt_dirs = sorted((save_dir / "model").glob("checkpoint-*"))
    if not ckpt_dirs:
        raise RuntimeError(f"no checkpoints under {save_dir / 'model'}")
    ckpt_dir = ckpt_dirs[-1]
    log(f"checkpoint: {ckpt_dir} (save_dir {save_dir.name})")

    import pickle
    with open(save_dir / "prepared" / "caft_id_class_dict.pkl", "rb") as fh:
        idc = pickle.load(fh)
    num_labels = len(idc)

    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained(
        str(ckpt_dir), num_labels=num_labels, torch_dtype=torch.bfloat16,
    )
    model.to(device).eval()
    log(f"model loaded ({num_labels} classes); ckpt keys="
        f"{len(model.state_dict())}")

    # --- input: tokenized external cache + h5ad obs alignment ----------------
    from datasets import load_from_disk
    import anndata as ad

    ds = load_from_disk(str(TOKEN_CACHE / "cell.dataset"))
    log(f"tokenized external (on-disk, all cells): {len(ds)} columns "
        f"{ds.column_names}")
    a = ad.read_h5ad(str(EXT_H5AD), backed="r")
    labels_all = np.asarray(a.obs["cell_type_harmonized"].astype(str))
    mask = np.asarray(labels_all != "NON_LAYER")
    keep_idx = np.where(mask)[0]
    # run_tier_b_eval.tokenize_external applies the NON_LAYER epithelial
    # filter in-memory AFTER load_from_disk; mirror it exactly (cache on disk
    # holds all 15,827 cells).
    ds = ds.select(keep_idx)
    donors_h5ad = np.asarray(a.obs["donor_id"].astype(str).to_numpy())[mask]
    donors_attr = np.asarray(ds["donor_id"], dtype=str)
    align_ok = bool(np.array_equal(donors_attr, donors_h5ad))
    log(f"epithelial filter: {len(keep_idx)}/{len(mask)} cells; "
        f"obs alignment (donor_id) dataset-vs-h5ad: {align_ok}")
    if not align_ok:
        raise RuntimeError("external dataset <-> h5ad donor_id mismatch")
    y_true = labels_all[mask]

    # --- forward -------------------------------------------------------------
    n = len(ds)
    embs = []
    n_batches = int(np.ceil(n / args.batch))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, args.batch):
            e = min(s + args.batch, n)
            chunk = ds[s:e]
            input_ids, length = collate_right_pad(chunk["input_ids"], chunk["length"])
            input_ids = input_ids.to(device)
            length = length.to(device)
            attn_mask = (
                torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                < length.unsqueeze(1)
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                bert_out = model.bert(input_ids, attention_mask=attn_mask)
                emb = bert_out.last_hidden_state[:, 0]
            embs.append(emb.float().cpu().numpy())
            if (s // args.batch + 1) % 100 == 0 or s + args.batch >= n:
                log(f"    batch {s // args.batch + 1}/{n_batches}")
    cell_emb = np.concatenate(embs, axis=0).astype(np.float32)
    log(f"forward {time.time() - t0:.1f}s -> {cell_emb.shape}")

    # --- rho raw + L2-normed + label-shuffle null ----------------------------
    emb_n = l2_norm_rows(cell_emb)
    rho_raw = float(marker_gradient_correlation(cell_emb, y_true, LAYER_ORDER))
    rho_normed = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
    log(f"rho(raw)=+{rho_raw:.4f} rho(L2-normed)=+{rho_normed:.4f}")
    null = label_shuffle_null(emb_n, y_true, LAYER_ORDER, args.n_null, args.seed_null)
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    null_z = float((rho_normed - null_mean) / max(null_sd, 1e-12))
    null_pctile = float((null <= rho_normed).mean())
    log(f"null({args.n_null}): mean={null_mean:+.4f} sd={null_sd:.4f} "
        f"z={null_z:+.2f} pctile={100 * null_pctile:.1f}%")

    res = {
        "seed": args.seed,
        "checkpoint": str(ckpt_dir),
        "n_cells": int(n),
        "donor_alignment_ok": bool(align_ok),
        "rho_raw": rho_raw,
        "rho_normed": rho_normed,
        "null_n": args.n_null,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_z": null_z,
        "null_pctile": null_pctile,
        "convention": "L2-normed primary (G1 frozen-diagnosis); "
                      "label-shuffle rng default_rng(42)",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = out_dir / f"eval_armb_seed{args.seed}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
