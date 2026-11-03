#!/usr/bin/env python
"""Per-cell embedding extraction for the sensitivity + UMAP tasks (p5070 CPU).

Extracts final-layer CLS cell embeddings + threshold/softmax logits for the
naive and CORN checkpoints of scGPT (van Zyl run6) and GF (van Zyl-era
GSE153515-trained), over the LOCKED test donor Hu0216 (20,158 cells). Frozen
arms and CellPLM arms are already cached (G1 npz / cellplm npz) and are NOT
forwarded here.

PCS frozen protocol (frozen before first run 2026-08-20):
  * test cells: artifacts/paper1_vanzyl/run6/scgpt_vanzyl_test.h5ad (20158,
    donor sanes2022_Hu0216, 3-tier cell_class_4) — locked
  * scGPT naive : run6/checkpoints/naive_seed{42,123,456}/model.pt
                 (30-class softmax, best-val save, 3-epoch run6 budget)
  * scGPT CORN  : run6/checkpoints/ordinal_seed{42,123,456}/model.pt
                 (K=3 -> 2 threshold logits; threshold-count decode)
  * GF naive    : save/caft_gf_lam0.0_seed42_20260818-234513/model/checkpoint-2491
                 (31-class softmax; GF per-cell extraction limited to the
                 locked PREREG primary pair seed42 — CPU cost ~12.7 h/seed)
  * GF CORN     : save/corn_advantage_gf_ckpt/ordinal_gf_lam0.0_seed42_*/model/checkpoint-7473
                 (K=4 -> 3 threshold logits; GF CORN ckpt seed42 only)
  * forward: CPU fp32, no autocast (locked corn_advantage convention);
    scGPT batch 64 / GF batch 32, threads 12
  * sanity gate: recomputed rho(L2-normed) vs the locked run6/cellplm eval
    JSONs within RHO_TOL = 2e-3 (CPU fp32 vs GPU bf16); GF gate descriptive
    (no locked van Zyl number exists — recorded for the reduce step)

Outputs (<out>/): <arch>_<arm>_seed<seed>.npz with emb, y, logits, barcode;
per-run JSON with the gate record. Idempotent via DONE markers.

Usage (p5070, CPU):
    PYTHONPATH=/home/summer/corneal_scgpt/repo/scGPT \
    /home/summer/corneal_scgpt/venv/bin/python scripts/sens_umap_extract.py \
        --arch scgpt --arm naive --seed 42 --out artifacts/paper1_vanzyl/sensitivity/_caches
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

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "repo" / "scGPT"))

import torch  # noqa: E402

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
SCPT_CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "checkpoints"
SCGPT_PRETRAINED = ROOT / "pretrained" / "scgpt_pretrained"
GF_DATASET = ROOT / "data" / "gf_tokenized_vanzyl.dataset"
RAW_COUNTS = ROOT / "data" / "vanzyl_raw_counts.h5ad"
GF_NAIVE_CKPT = (
    ROOT / "save" / "caft_gf_lam0.0_seed42_20260818-234513" / "model" / "checkpoint-2491"
)
GF_CORN_CKPT = (
    ROOT / "save" / "corn_advantage_gf_ckpt"
    / "ordinal_gf_lam0.0_seed42_20260805-171141" / "model" / "checkpoint-7473"
)
RUN6_EVAL = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval"
LAYER_ORDER = ["LSC", "basal", "differentiated"]
TEST_DONOR = "sanes2022_Hu0216"
RHO_TOL = 2e-3
SCGPT_BATCH = 64
GF_BATCH = 32
GF_THRESHOLDS = 3  # 4-class GF axis -> 3 thresholds
SCGPT_SEEDS = (42, 123, 456)
SCGPT_MAX_LEN = 4253


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def rho_normed(emb: np.ndarray, y: np.ndarray) -> float:
    return float(marker_gradient_correlation(l2_norm_rows(emb), y, LAYER_ORDER))


# ---------------------------------------------------------------------------
# scGPT forward (mirrors corn_advantage_scgpt.py / run6 eval, CPU fp32)
# ---------------------------------------------------------------------------

def forward_scgpt(seed: int, arm: str) -> dict[str, Any]:
    import scgpt_cls as sc  # noqa: E402

    ckpt_dir = SCPT_CKPT / f"{arm}_seed{seed}"
    args = json.loads((ckpt_dir / "args.json").read_text())
    n_cls = args["n_cls"]
    assert args["arm"] == arm
    max_len = args["training"]["max_len"]
    assert max_len == SCGPT_MAX_LEN

    locked = json.loads((RUN6_EVAL / f"{arm}_seed{seed}.json").read_text())
    locked_rho = locked["rho_normed"]

    vocab, pargs = sc.load_vocab(pretrained_dir=SCGPT_PRETRAINED)
    import anndata as ad

    log(f"scgpt {arm} s{seed}: loading test h5ad")
    a = ad.read_h5ad(TEST_H5AD, backed="r")
    y = np.asarray(a.obs["cell_class_4"].astype(str))
    barcode = np.asarray(a.obs_names)
    if not (a.obs["donor_id"].astype(str) == TEST_DONOR).all():
        raise RuntimeError("test h5ad is not exclusively the locked test donor")
    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    log(f"scgpt {arm} s{seed}: tokenizing {x_mat.shape}")
    tok = sc.tokenize_chunked(x_mat, gene_ids, vocab, pargs, max_len=max_len)
    log(f"scgpt {arm} s{seed}: tokenized genes={tok['genes'].shape}")

    model = sc.build_finetune_model(vocab, pargs, n_cls,
                                    pretrained_dir=SCGPT_PRETRAINED)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    log(f"scgpt {arm} s{seed}: model loaded, forward starting")
    # CPU forward: disable torch's nested-tensor fast path (prototype kernels
    # are extremely slow on CPU; values are identical — verified by the gate).
    enc = getattr(model, "transformer_encoder", None)
    if enc is not None:
        for attr in ("enable_nested_tensor", "use_nested_tensor"):
            if hasattr(enc, attr):
                setattr(enc, attr, False)

    n = tok["genes"].shape[0]
    # Sorted batching + per-batch padding trim. The tokenizer returns dense
    # arrays padded to the global max (1313); trimming each batch to its own
    # max real length makes attention cost scale with true length (~3x less
    # compute). Padded positions are masked in attention, so the position-0
    # CLS embedding is unchanged up to float error — verified by the gate.
    padded = tok["mask"].sum(dim=1).numpy()
    order = np.argsort(-padded, kind="stable")
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    genes = tok["genes"][order]
    values = tok["values"][order]
    mask = tok["mask"][order]
    embs, clss = [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, SCGPT_BATCH):
            src_full = genes[s:s + SCGPT_BATCH]
            m_full = mask[s:s + SCGPT_BATCH]
            real_max = int((~m_full).sum(dim=1).max())
            src = src_full[:, :real_max]
            vals = values[s:s + SCGPT_BATCH, :real_max]
            pad = m_full[:, :real_max]
            out = model(src, vals, src_key_padding_mask=pad, CLS=True)
            embs.append(out["cell_emb"].float().numpy())
            clss.append(out["cls_output"].float().numpy())
            bno = s // SCGPT_BATCH + 1
            if bno % 5 == 0 or s + SCGPT_BATCH >= n:
                log(f"    batch {bno}/{(n + SCGPT_BATCH - 1) // SCGPT_BATCH} "
                    f"({time.time() - t0:.0f}s)")
    emb = np.concatenate(embs, axis=0).astype(np.float32)[back]
    logits = np.concatenate(clss, axis=0).astype(np.float32)[back]
    log(f"scgpt {arm} s{seed}: forward {time.time() - t0:.0f}s "
        f"emb={emb.shape} logits={logits.shape}")

    rho_n = rho_normed(emb, y)
    gate = {
        "rho_normed_recomputed": rho_n,
        "rho_normed_locked": float(locked_rho),
        "abs_diff": abs(rho_n - locked_rho),
        "tol": RHO_TOL,
        "pass": abs(rho_n - locked_rho) <= RHO_TOL,
    }
    log(f"  gate rho_n={rho_n:.4f} locked={locked_rho:.4f} "
        f"diff={gate['abs_diff']:.2e} {'PASS' if gate['pass'] else 'FAIL'}")
    return {
        "tag": f"scgpt_{arm}_seed{seed}", "arm": arm, "seed": seed,
        "arch": "scgpt", "n_cells": int(n), "n_cls": n_cls, "max_len": max_len,
        "emb": emb, "logits": logits, "y": y, "barcode": barcode, "gate": gate,
    }


# ---------------------------------------------------------------------------
# GF forward (mirrors corn_advantage_gf.py, CPU fp32, batch 32)
# ---------------------------------------------------------------------------

def gf_test_indices():
    """Positions (in gf_tokenized_vanzyl row order) of the locked test cells,
    in scGPT-test order (runtime-asserted; identical to corn_advantage_gf)."""
    from datasets import load_from_disk

    import anndata as ad

    raw = ad.read_h5ad(RAW_COUNTS, backed="r")
    raw_bc = np.asarray(raw.obs_names)
    del raw
    a = ad.read_h5ad(TEST_H5AD, backed="r")
    scgpt_bc = np.asarray(a.obs_names)
    scgpt_y = np.asarray(a.obs["cell_class_4"].astype(str))
    idx = np.where(np.isin(raw_bc, scgpt_bc))[0]
    assert len(idx) == len(scgpt_bc) == 20158
    assert (raw_bc[idx] == scgpt_bc).all(), "GF test cells not in scGPT-test order"
    ds = load_from_disk(str(GF_DATASET))
    assert len(ds) == len(raw_bc)
    y_ds = np.asarray(ds["cell_class_4"])[idx]
    assert (y_ds == scgpt_y).all(), "dataset labels disagree with scGPT test labels"
    return idx, scgpt_y, scgpt_bc, ds


def forward_gf(arm: str) -> dict[str, Any]:
    from datasets import load_from_disk  # noqa: F401
    from transformers import BertForSequenceClassification

    ckpt_dir = GF_NAIVE_CKPT if arm == "naive" else GF_CORN_CKPT
    num_labels = 31 if arm == "naive" else GF_THRESHOLDS
    idx, y, barcode, ds = gf_test_indices()

    # Length-sorted batching (pure performance: reduces right-padding waste;
    # per-cell values are order-independent, outputs restored to h5ad order).
    lengths = np.asarray(ds["length"], dtype=np.int64)[idx]
    order = np.argsort(lengths, kind="stable")  # ascending
    idx_s = idx[order]
    back = np.empty_like(order)
    back[order] = np.arange(len(order))

    model = BertForSequenceClassification.from_pretrained(
        str(ckpt_dir), num_labels=num_labels, torch_dtype=torch.float32,
    )
    model.eval()
    embs, logits_list = [], []
    n = len(idx)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, GF_BATCH):
            e = min(s + GF_BATCH, n)
            chunk = ds[idx_s[s:e].tolist()]
            ids_list = chunk["input_ids"]
            lengths_b = np.asarray(chunk["length"], dtype=np.int64)
            ml = max(len(x) for x in ids_list)
            padded = np.zeros((len(ids_list), ml), dtype=np.int64)
            for i, ids in enumerate(ids_list):
                padded[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
            input_ids = torch.from_numpy(padded)
            length = torch.from_numpy(lengths_b)
            attn_mask = torch.arange(ml).unsqueeze(0) < length.unsqueeze(1)
            bert_out = model.bert(input_ids, attention_mask=attn_mask)
            emb = bert_out.last_hidden_state[:, 0]
            logits = model.classifier(model.dropout(bert_out.pooler_output))
            embs.append(emb.float().numpy())
            logits_list.append(logits.float().numpy())
            if (s // GF_BATCH + 1) % 50 == 0 or s + GF_BATCH >= n:
                log(f"    batch {s // GF_BATCH + 1}/{max(1, (n + GF_BATCH - 1) // GF_BATCH)} "
                    f"({time.time() - t0:.0f}s)")
    emb = np.concatenate(embs, axis=0).astype(np.float32)[back]
    logits = np.concatenate(logits_list, axis=0).astype(np.float32)[back]
    log(f"gf {arm}: forward {time.time() - t0:.0f}s emb={emb.shape} logits={logits.shape}")

    rho_n = rho_normed(emb, y)
    gate = {
        "rho_normed_recomputed": rho_n,
        "rho_normed_locked": None,  # no locked van Zyl GF number (descriptive)
        "abs_diff": None,
        "tol": RHO_TOL,
        "pass": True,
        "note": "descriptive: no locked van Zyl GF number exists (corn_advantage "
                "lane was to record it); reduced/reported by sens_umap_reduce",
    }
    log(f"  descriptive rho_normed={rho_n:.4f}")
    return {
        "tag": f"gf_{arm}_seed42", "arm": arm, "seed": 42, "arch": "gf",
        "n_cells": int(n), "n_cls": num_labels,
        "emb": emb, "logits": logits, "y": y, "barcode": barcode, "gate": gate,
    }


# ---------------------------------------------------------------------------

def _str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype="<U64")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["scgpt", "gf"], required=True)
    ap.add_argument("--arm", choices=["naive", "corn"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, int | None]] = []
    if args.arch == "scgpt":
        seeds = [args.seed] if args.seed is not None else list(SCGPT_SEEDS)
        runs = [(f"scgpt_{args.arm}_seed{s}", s) for s in seeds]
    else:
        runs = [("gf_naive_seed42", 42)] if args.arm == "naive" else [("gf_corn_seed42", 42)]

    for tag, seed in runs:
        done = out_dir / f"{tag}.DONE"
        if done.exists():
            log(f"SKIP {tag} (DONE marker)")
            continue
        if tag.startswith("scgpt"):
            res = forward_scgpt(seed, args.arm)
        else:
            res = forward_gf(args.arm)
        np.savez_compressed(
            out_dir / f"{tag}.npz",
            emb=res["emb"], logits=res["logits"], y=_str(res["y"]),
            barcode=_str(res["barcode"]),
        )
        json_path = out_dir / f"{tag}.json"
        json_path.write_text(json.dumps({
            "tag": res["tag"], "arch": res["arch"], "arm": res["arm"],
            "seed": res["seed"], "n_cells": res["n_cells"], "n_cls": res["n_cls"],
            "test_donor": TEST_DONOR, "layer_order": LAYER_ORDER,
            "threads": args.threads,
            "checkpoint": str(
                (SCPT_CKPT if res["arch"] == "scgpt" else (
                    GF_NAIVE_CKPT if res["arm"] == "naive" else GF_CORN_CKPT))
            ),
            "gate": res["gate"], "timestamp_utc": now_utc(),
        }, indent=2), encoding="utf-8")
        done.write_text(now_utc())
        log(f"saved {out_dir / f'{tag}.npz'}")
    log("extract done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
