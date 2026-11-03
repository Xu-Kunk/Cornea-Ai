#!/usr/bin/env python
"""CORN advantage A+D — GF (Geneformer) CORN (3 seeds) per-cell extraction.

Runs on <compute-node> (CPU) with <home>/corneal_scgpt/venv/bin/python (the
migrated checkpoint is plain HF BertForSequenceClassification — no geneformer
package needed at inference). For the locked test donor Hu0216 (20158 cells),
forwards the GSE153515-trained ordinal-GF checkpoints over the tokenized van
Zyl cells and saves per-cell threshold logits (K=4 axis -> 3 logits; the
3-tier reading uses t1=P(>=1)=not-LSC and t3=P(>=2)=differentiated),
CLS embeddings, labels and the locked van Zyl marker scores.

FROZEN protocol (PCS Stability, frozen before first run 2026-08-20):
  * test cells: gf_tokenized_vanzyl.dataset rows whose barcode is in the
    scGPT test h5ad (20158, donor Hu0216; same analysis set, same order —
    asserted at runtime)
  * checkpoint: save/ordinal_gf_lam0.0_seed{42,123,456}/model/checkpoint-7473
    migrated local -> <compute-node> (read-only copy; SHA-256 recorded in the manifest;
    locked GSE153515 CORN provenance: rho 0.8330, seeds 0.8341/0.8351/0.8299)
  * ordinal axis 4-class (LSC<TAC<basal<diff) -> 3 threshold logits;
    3-tier van Zyl reading: p1 = sigmoid(t1) (>=TAC => not-LSC => basal+diff),
    p2 = sigmoid(t3) (>=diff => differentiated). t2 (TAC->basal) is not used
    on the 3-tier axis (van Zyl has no TAC; user-confirmed 2026-08-20).
  * decode threshold = 0.5; CLS = last_hidden_state[:,0] (G1 contract);
    logits = classifier(dropout(pooler_output))
  * gate: no locked van Zyl rho for GF CORN (0.833 is GSE153515-locked) — the
    per-seed checkpoint SHA + van Zyl decode distribution + descriptive
    rho_normed are recorded; monotonicity/threshold metrics are structural.

Outputs: <out>/gf_seed<seed>.npz + .json + .DONE marker. Idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from transformers import BertForSequenceClassification

ROOT = Path("<home>/corneal_scgpt")
sys.path.insert(0, str(ROOT))
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

GF_DATASET = ROOT / "data" / "gf_tokenized_vanzyl.dataset"
RAW_COUNTS = ROOT / "data" / "vanzyl_raw_counts.h5ad"
SCGPT_TEST = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
CKPT_BASE = ROOT / "save" / "corn_advantage_gf_ckpt"
LAYER_ORDER = ["LSC", "basal", "differentiated"]
TEST_DONOR = "sanes2022_Hu0216"
BATCH = 64
N_THRESHOLDS = 3  # GF 4-class axis -> 3 threshold logits
SCORE_COLS = ["score_LSC", "score_PMC_basal", "score_TDC"]
CKPT_MAP = {
    42: "ordinal_gf_lam0.0_seed42_20260805-171141",
    123: "ordinal_gf_lam0.0_seed123_20260806-144600",
    456: "ordinal_gf_lam0.0_seed456_20260806-223909",
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_selection_indices() -> np.ndarray:
    """Positions (in gf_tokenized_vanzyl row order = raw-counts row order) of
    the 20158 locked test cells, in scGPT-test order (runtime-asserted)."""
    import anndata as ad
    from datasets import load_from_disk

    raw = ad.read_h5ad(RAW_COUNTS, backed="r")
    raw_bc = np.asarray(raw.obs_names)
    del raw
    a = ad.read_h5ad(SCGPT_TEST, backed="r")
    scgpt_bc = np.asarray(a.obs_names)
    scgpt_y = np.asarray(a.obs["cell_class_4"].astype(str))
    scores = np.stack(
        [np.asarray(a.obs[c], dtype=float) for c in SCORE_COLS], axis=1
    ).astype(np.float32)
    idx = np.where(np.isin(raw_bc, scgpt_bc))[0]
    assert len(idx) == len(scgpt_bc) == 20158
    assert (raw_bc[idx] == scgpt_bc).all(), "GF test cells not in scGPT-test order"
    ds = load_from_disk(str(GF_DATASET))
    assert len(ds) == len(raw_bc)
    y_ds = np.asarray(ds["cell_class_4"])[idx]
    assert (y_ds == scgpt_y).all(), "dataset labels disagree with scGPT test labels"
    return idx, scgpt_y, scores


def collate_right_pad(input_ids_list, lengths_list):
    max_len = max(len(ids) for ids in input_ids_list)
    padded = np.zeros((len(input_ids_list), max_len), dtype=np.int64)
    for i, ids in enumerate(input_ids_list):
        padded[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
    return (
        torch.from_numpy(padded),
        torch.from_numpy(np.asarray(lengths_list, dtype=np.int64)),
    )


def forward_cell_emb_logits(model, dataset, indices, batch: int):
    """G1 contract: CLS = last_hidden_state[:,0]; logits =
    classifier(dropout(pooler_output)); CPU fp32."""
    model.eval()
    embs, logits_list = [], []
    n = len(indices)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            chunk = dataset[indices[s:e].tolist()]
            input_ids, length = collate_right_pad(chunk["input_ids"], chunk["length"])
            attn_mask = (
                torch.arange(input_ids.shape[1]).unsqueeze(0) < length.unsqueeze(1)
            )
            bert_out = model.bert(input_ids, attention_mask=attn_mask)
            emb = bert_out.last_hidden_state[:, 0]
            logits = model.classifier(model.dropout(bert_out.pooler_output))
            embs.append(emb.float().numpy())
            logits_list.append(logits.float().numpy())
            if (s // batch + 1) % 50 == 0 or s + batch >= n:
                log(f"    batch {s // batch + 1}/{(n + batch - 1) // batch} "
                    f"({time.time() - t0:.0f}s)")
    return (
        np.concatenate(embs, axis=0).astype(np.float32),
        np.concatenate(logits_list, axis=0).astype(np.float32),
    )


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def run_seed(seed: int, out_dir: Path, threads: int) -> int:
    torch.set_num_threads(threads)
    ckpt_dir = CKPT_BASE / CKPT_MAP[seed] / "model" / "checkpoint-7473"
    assert ckpt_dir.exists(), f"migrated checkpoint missing: {ckpt_dir}"
    sha = sha256_file(ckpt_dir / "model.safetensors")

    log(f"=== CORN GF seed {seed} ===")
    from datasets import load_from_disk

    idx, y, scores = test_selection_indices()
    ds = load_from_disk(str(GF_DATASET))

    model = BertForSequenceClassification.from_pretrained(
        str(ckpt_dir), num_labels=N_THRESHOLDS, torch_dtype=torch.float32
    )
    model.eval()
    t0 = time.time()
    emb, logits = forward_cell_emb_logits(model, ds, idx, BATCH)
    log(f"forward done {time.time() - t0:.0f}s emb={emb.shape} logits={logits.shape}")

    rho_n = float(marker_gradient_correlation(l2_norm_rows(emb), y, LAYER_ORDER))
    probs = 1.0 / (1.0 + np.exp(-logits))
    k = (probs >= 0.5).sum(axis=-1)
    pred_dist = {lab: int((k == i).sum()) for i, lab in enumerate(LAYER_ORDER)}
    log(f"descriptive rho_normed={rho_n:.4f} pred_dist={pred_dist}")

    np.savez_compressed(
        out_dir / f"gf_seed{seed}.npz",
        logits=logits, emb=emb, y=_str(y), scores=scores,
    )
    (out_dir / f"gf_seed{seed}.json").write_text(
        json.dumps({
            "arm": "corn_gf", "seed": seed, "n_cells": int(len(y)),
            "threshold_logits": N_THRESHOLDS, "test_donor": TEST_DONOR,
            "axis": "4-class LSC<TAC<basal<diff (GSE153515-trained); "
                    "3-tier van Zyl reading uses t1/t3 (user-confirmed)",
            "checkpoint": str(ckpt_dir),
            "checkpoint_sha256": sha,
            "locked_provenance": {
                "GSE153515_corn_rho": 0.8330,
                "note": "paper/task-locked GF CORN (GSE153515 external test); "
                        "van Zyl application is cross-corpus by design",
            },
            "descriptive": {
                "rho_normed_van_zyl": rho_n,
                "pred_dist": pred_dist,
            },
            "gate": {
                "checkpoint_sha_recorded": True,
                "rho_locked_van_zyl": None,
                "note": "no locked van Zyl GF CORN number exists; rho recorded "
                        "descriptively, metrics A/D are structural",
            },
            "timestamp_utc": now_utc(),
        }, indent=2)
    )
    (out_dir / f"gf_seed{seed}.DONE").write_text(now_utc())
    return 0


def _str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype="<U64")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "paper1_vanzyl" / "corn_advantage")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"gf_seed{args.seed}"
    if (out_dir / f"{tag}.DONE").exists():
        log(f"SKIP {tag} (DONE marker present)")
        return 0
    return run_seed(args.seed, out_dir, args.threads)


if __name__ == "__main__":
    sys.exit(main())
