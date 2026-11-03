#!/usr/bin/env python
"""Cornea2 scGPT naive CE fine-tuning (mirrors scgpt_cls_vanzyl_convergence.py).

Design (LOCKED, same scientific logic as the van Zyl Arm A convergence arm):
* naive CE head over the dataset's own cell_type_raw classes (DEG-enriched
  assignment from protocol_15d — the cornea2 analogue of van Zyl's 31-class
  author_cell_type head); evaluation labels = majority-map of cell_type_raw
  -> 3-tier {LSC, basal, differentiated} computed on the TRAIN split
  (mirrors the van Zyl naive_map).
* 15 epochs (= scGPT official budget), lr 5e-5, wd 1e-3, warmup 500,
  batch 8 x accum 4 (=32), bf16 (fp16 fallback), max_len 4253.
* val = 10% of the train split (stratified donor x cell_class_4,
  <ds>_val_indices.npz LOCKED; test donor never enters).
* Per epoch: val macro-F1 + val rho(L2-normed) + label-shuffle z (n=100)
  -> trajectory.jsonl; best-val (macro-F1) checkpoint is the PRIMARY
  reporting checkpoint; final-epoch is the sensitivity item.
* Collapse-start epoch = first epoch with val rho < frozen baseline - 0.15.
  The frozen baseline is computed ON THIS DATASET's test donor by
  run_cornea2_frozen_eval.py (frozen rho_frz) and passed via --frozen-rho.

Usage
-----
    repo/.venv/bin/python scripts/scgpt_cls_cornea2.py --dataset GSE186433 \
        --seed 42 --frozen-rho 0.70 --out artifacts/paper1_vanzyl/cornea2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import BuiltinVocab
from scgpt.utils import load_pretrained
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

PRETRAINED_DIR = PROJECT_ROOT / "pretrained" / "scgpt_pretrained"

MAX_LEN = 4253
EPOCHS = 15
SEEDS = [42, 123]
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4
LR = 5e-5
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 500
TOKENIZE_CHUNK = 8192
LOG_INTERVAL = 100
N_NULL = 100
NULL_SEED = 42
EVAL_CLASSES = ["LSC", "basal", "differentiated"]

DATASETS = {
    "GSE186433": {
        "train": PROJECT_ROOT / "data" / "cornea2" / "gse186433_train.h5ad",
        "val_indices": PROJECT_ROOT / "data" / "cornea2" / "gse186433_val_indices.npz",
        "val_manifest": PROJECT_ROOT / "data" / "cornea2" / "gse186433_split_val_manifest.json",
        "n_train_cells": 3491,
        "test_donor": "04",
    },
    "GSE155683": {
        "train": PROJECT_ROOT / "data" / "cornea2" / "gse155683_train.h5ad",
        "val_indices": PROJECT_ROOT / "data" / "cornea2" / "gse155683_val_indices.npz",
        "val_manifest": PROJECT_ROOT / "data" / "cornea2" / "gse155683_split_val_manifest.json",
        "n_train_cells": 6468,
        "test_donor": "donor_4",
    },
}


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Vocab / model (verbatim from the van Zyl convergence script)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_vocab(pretrained_dir: Path = PRETRAINED_DIR) -> tuple[GeneVocab, dict[str, Any]]:
    args = json.loads((pretrained_dir / "args.json").read_text(encoding="utf-8"))
    token2idx = json.loads((pretrained_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [t for t, _ in sorted(token2idx.items(), key=lambda kv: kv[1])]
    if sorted(token2idx.values()) == list(range(len(token2idx))):
        vocab = GeneVocab(BuiltinVocab(tokens))
    else:
        vocab = GeneVocab.from_file(str(pretrained_dir / "vocab.json"))
    for sp in [args["pad_token"], "<cls>", "<eoc>"]:
        if sp not in vocab:
            vocab.append_token(sp)
    if len(vocab) != args["ntoken"]:
        raise RuntimeError(f"vocab size {len(vocab)} != ntoken {args['ntoken']}")
    return vocab, args


def build_finetune_model(vocab, args, n_cls, pretrained_dir=PRETRAINED_DIR):
    model = TransformerModel(
        ntoken=args["ntoken"],
        d_model=args["embsize"],
        nhead=args["nheads"],
        d_hid=args["d_hid"],
        nlayers=args["nlayers"],
        nlayers_cls=args.get("n_layers_cls", 3),
        n_cls=n_cls,
        vocab=vocab,
        dropout=args["dropout"],
        pad_token=args["pad_token"],
        pad_value=args["pad_value"],
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=args["input_emb_style"],
        n_input_bins=args["n_bins"],
        cell_emb_style=args["cell_emb_style"],
        mvc_decoder_style=args.get("mvc_decoder_style", "inner product"),
        ecs_threshold=args.get("ecs_threshold", 0.3),
        explicit_zero_prob=args.get("explicit_zero_prob", False),
        use_fast_transformer=False,
        pre_norm=args.get("pre_norm", False),
    )
    ckpt = torch.load(
        pretrained_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    load_pretrained(model, ckpt, strict=False, verbose=False)
    return model


# ---------------------------------------------------------------------------
# Data pipeline (verbatim from the van Zyl convergence script)
# ---------------------------------------------------------------------------


def tokenize_chunked(x_mat, gene_ids, vocab, args, max_len=MAX_LEN, chunk=TOKENIZE_CHUNK):
    n = x_mat.shape[0]
    if n == 0:
        empty = torch.empty(0, max_len, dtype=torch.long)
        mask = torch.ones(0, max_len, dtype=torch.bool)
        return {"genes": empty, "values": empty.float(), "mask": mask}
    if hasattr(x_mat, "getnnz"):
        max_nnz = int(x_mat.getnnz(axis=1).max())
    else:
        max_nnz = int(np.count_nonzero(np.asarray(x_mat, dtype=bool), axis=1).max())
    width = min(max_nnz + 1, max_len)
    pad_id = vocab[args["pad_token"]]
    pad_value = args["pad_value"]

    pieces = []
    for s in range(0, n, chunk):
        block = x_mat[s : s + chunk]
        if hasattr(block, "toarray"):
            block = block.toarray()
        elif hasattr(block, "to_memory"):
            block = block.to_memory().toarray()
        block = np.asarray(block, dtype=np.float32)
        tok = tokenize_and_pad_batch(
            data=block,
            gene_ids=gene_ids,
            max_len=max_len,
            vocab=vocab,
            pad_token=args["pad_token"],
            pad_value=pad_value,
            append_cls=True,
            include_zero_gene=args.get("include_zero_gene", False),
        )
        genes_c, values_c = tok["genes"], tok["values"]
        w = genes_c.shape[1]
        if w < width:
            pad_g = torch.full((genes_c.shape[0], width - w), pad_id, dtype=genes_c.dtype)
            pad_v = torch.full(
                (values_c.shape[0], width - w), pad_value, dtype=values_c.dtype
            )
            genes_c = torch.cat([genes_c, pad_g], dim=1)
            values_c = torch.cat([values_c, pad_v], dim=1)
        pieces.append((genes_c, values_c))
    genes = torch.cat([g for g, _ in pieces], dim=0)
    values = torch.cat([v for _, v in pieces], dim=0)
    mask = genes.eq(pad_id)
    return {"genes": genes, "values": values, "mask": mask}


class CellDataset(Dataset):
    def __init__(self, genes, values, labels):
        self.genes = genes
        self.values = values
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        return self.genes[i], self.values[i], self.labels[i]


# ---------------------------------------------------------------------------
# Metrics (same implementation as the G1 rebuild / van Zyl eval lane)
# ---------------------------------------------------------------------------


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def label_shuffle_null(emb, y, layer_order, n, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = marker_gradient_correlation(emb, rng.permutation(y), layer_order)
    return out


def rho_and_null(emb: np.ndarray, labels: np.ndarray) -> dict:
    rho_raw = float(marker_gradient_correlation(emb, labels, EVAL_CLASSES))
    emb_n = l2_norm_rows(emb)
    rho_n = float(marker_gradient_correlation(emb_n, labels, EVAL_CLASSES))
    null = label_shuffle_null(emb_n, labels, EVAL_CLASSES, n=N_NULL, seed=NULL_SEED)
    z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))
    return {
        "rho_raw": rho_raw,
        "rho_normed": rho_n,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_z": z,
        "null_pctile": float((null <= rho_n).mean()),
        "n_null": N_NULL,
    }


def convergence_train(model, train_loader, val_loader, device, pad_id, epochs,
                      grad_accum, lr, weight_decay, warmup_steps, dtype,
                      run_dir, y_true_val, naive_classes, naive_map,
                      frozen_baseline_rho):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_step = 0
    t0_all = time.time()
    trajectory = []
    best_val_f1 = -1.0
    best_val_f1_epoch = None
    best_val_rho = None
    best_val_rho_epoch = None
    first_loss = None
    present = sorted(set(y_true_val.tolist()) & set(EVAL_CLASSES))
    collapse_threshold = round(frozen_baseline_rho - 0.15, 4)

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        opt.zero_grad(set_to_none=True)
        for step, (src, values, y) in enumerate(train_loader):
            src = src.to(device); values = values.to(device); y = y.to(device)
            mask = src.eq(pad_id)
            with torch.autocast(device_type=device.type, dtype=dtype):
                out = model(src, values, src_key_padding_mask=mask, CLS=True)
                loss = F.cross_entropy(out["cls_output"], y)
            (loss.float() / grad_accum).backward()
            loss_val = float(loss.detach().float())
            running.append(loss_val)
            if first_loss is None:
                first_loss = loss_val
            if (step + 1) % grad_accum == 0 or step + 1 == len(train_loader):
                opt.step()
                opt_step += 1
                if opt_step <= warmup_steps:
                    for g in opt.param_groups:
                        g["lr"] = lr * opt_step / max(warmup_steps, 1)
                else:
                    for g in opt.param_groups:
                        g["lr"] = lr
                opt.zero_grad(set_to_none=True)
        train_mean = float(np.mean(running[-min(200, len(running)):]))

        model.eval()
        embs, clss = [], []
        with torch.no_grad():
            for src, values, _y in val_loader:
                src = src.to(device); values = values.to(device)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    out = model(src, values,
                                src_key_padding_mask=src.eq(pad_id), CLS=True)
                embs.append(out["cell_emb"].float().cpu().numpy())
                clss.append(out["cls_output"].float().cpu().numpy())
        vemb = np.concatenate(embs, axis=0).astype(np.float32)
        vcls = np.concatenate(clss, axis=0).astype(np.float32)
        pred_author = np.array(
            [naive_classes[int(i)] for i in vcls.argmax(1)], dtype=str)
        pred = np.array([naive_map.get(c, "NON_LAYER") for c in pred_author],
                        dtype=str)
        val_f1 = float(f1_score(y_true_val, pred, labels=present,
                                average="macro", zero_division=0))
        rho_raw = float(marker_gradient_correlation(vemb, y_true_val, EVAL_CLASSES))
        vemb_n = l2_norm_rows(vemb)
        rho_n = float(marker_gradient_correlation(vemb_n, y_true_val, EVAL_CLASSES))
        null = label_shuffle_null(vemb_n, y_true_val, EVAL_CLASSES, n=N_NULL, seed=NULL_SEED)
        z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch
            torch.save(model.state_dict(), run_dir / "model_best_val.pt")
        if best_val_rho is None or rho_n > best_val_rho:
            best_val_rho = rho_n
            best_val_rho_epoch = epoch
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:02d}.pt")

        rec = {
            "epoch": epoch,
            "train_loss_mean": train_mean,
            "val_f1": val_f1,
            "val_rho_raw": rho_raw,
            "val_rho_normed": rho_n,
            "val_z": z,
            "best_val_f1": best_val_f1,
            "best_val_f1_epoch": best_val_f1_epoch,
            "best_val_rho": best_val_rho,
            "best_val_rho_epoch": best_val_rho_epoch,
            "wall_s": round(time.time() - t0_all, 1),
        }
        trajectory.append(rec)
        with (run_dir / "trajectory.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"epoch {epoch}/{epochs} train_mean={train_mean:.4f} "
            f"val_f1={val_f1:.4f} val_rho={rho_n:.4f} val_z={z:.1f} "
            f"best_f1={best_val_f1:.4f}@ep{best_val_f1_epoch} "
            f"({time.time() - t0_all:.0f}s)")

    collapse_epoch = None
    for epoch in range(1, epochs + 1):
        if trajectory[epoch - 1]["val_rho_normed"] < collapse_threshold:
            collapse_epoch = epoch
            break
    return {
        "first_loss": first_loss,
        "trajectory": trajectory,
        "best_val_f1": best_val_f1,
        "best_val_f1_epoch": best_val_f1_epoch,
        "best_val_rho": best_val_rho,
        "best_val_rho_epoch": best_val_rho_epoch,
        "collapse": {"epoch": collapse_epoch,
                     "threshold": collapse_threshold,
                     "frozen_baseline": frozen_baseline_rho},
        "wall_s": time.time() - t0_all,
    }


def preflight_dtype(model, batch, device, pad_id, dtype):
    src, values, y = [b.to(device) for b in batch]
    model.train()
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(src, values, src_key_padding_mask=src.eq(pad_id), CLS=True)
        loss = F.cross_entropy(out["cls_output"], y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"preflight loss not finite: {loss.item()}")
    loss.backward()
    model.zero_grad(set_to_none=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--frozen-rho", type=float, required=True,
                    help="frozen rho(L2-normed) on THIS dataset's test donor "
                         "(from run_cornea2_frozen_eval.py)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cornea2")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = 1 if args.dry_run else args.epochs

    log(f"=== scgpt_cls_cornea2 dataset={args.dataset} seed={seed} "
        f"dry_run={args.dry_run} epochs={epochs} frozen_rho={args.frozen_rho} ===")

    vocab, pargs = load_vocab()

    import anndata as ad

    t0 = time.time()
    adata = ad.read_h5ad(cfg["train"])
    max_cells = args.max_cells or adata.n_obs
    if max_cells < adata.n_obs:
        adata = adata[:max_cells].copy()
    obs = adata.obs
    if max_cells >= cfg["n_train_cells"] and adata.n_obs != cfg["n_train_cells"]:
        raise RuntimeError(
            f"train h5ad n_cells {adata.n_obs} != locked "
            f"{cfg['n_train_cells']} (split_manifest) -- refusing to train "
            "on a drifted split")
    ct_raw_all = obs["cell_type_raw"].astype(str).to_numpy()
    cc4_all = obs["cell_class_4"].astype(str).to_numpy()

    # val split (LOCKED; dry-run uses a local 90/10 split)
    if args.dry_run and max_cells < cfg["n_train_cells"]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(adata.n_obs)
        n_tr = int(adata.n_obs * 0.9)
        train_idx, val_idx_a = perm[:n_tr], perm[n_tr:]
    else:
        vd = np.load(cfg["val_indices"])
        val_idx_a = vd["val_idx"].astype(np.int64)
        val_mask = np.zeros(adata.n_obs, dtype=bool)
        val_mask[val_idx_a] = True
        train_idx = np.where(~val_mask)[0]
        vm = json.loads(cfg["val_manifest"].read_text(encoding="utf-8"))
        if int(len(train_idx)) != vm["n_train"] or int(len(val_idx_a)) != vm["n_val"]:
            raise RuntimeError(f"val split drift: {len(train_idx)}/{len(val_idx_a)} "
                               f"!= manifest {vm['n_train']}/{vm['n_val']}")
        log(f"LOCKED val split: train={len(train_idx)} val={len(val_idx_a)}")

    # naive classes + majority 3-tier mapping from the TRAIN subset
    tr_ct = ct_raw_all[train_idx]
    tr_cc4 = cc4_all[train_idx]
    naive_classes = sorted(set(tr_ct.tolist()))
    class_to_id = {c: i for i, c in enumerate(naive_classes)}
    naive_labels = np.array(
        [class_to_id.get(c, -1) for c in ct_raw_all], dtype=np.int64
    )
    naive_cont: dict[str, Counter] = {}
    for au, l in zip(tr_ct, tr_cc4):
        naive_cont.setdefault(au, Counter())[l] += 1
    naive_map = {au: max(c, key=c.get) for au, c in naive_cont.items()}
    log(f"naive mapping: {len(naive_classes)} cell_type_raw classes -> 3-tier")
    for au, m in sorted(naive_map.items()):
        log(f"  {au:>18} -> {m:<14} {dict(naive_cont[au])}")
    y_true_val = cc4_all[val_idx_a]
    labels = torch.as_tensor(naive_labels)
    log(f"data loaded ({time.time() - t0:.1f}s) n={adata.n_obs} "
        f"naive_classes={len(naive_classes)} n_val={len(val_idx_a)}")

    n_cls = len(naive_classes)
    model = build_finetune_model(vocab, pargs, n_cls)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model built n_cls={n_cls} params={n_params / 1e6:.2f}M")

    t0 = time.time()
    keep = np.array([g in vocab for g in adata.var_names])
    gene_ids = np.array([vocab[g] for g in adata.var_names[keep]], dtype=np.int64)
    x_mat = adata[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = tokenize_chunked(x_mat, gene_ids, vocab, pargs)
    log(f"tokenized ({time.time() - t0:.1f}s) genes_in_vocab={len(gene_ids)} "
        f"seq_len={tuple(tok['genes'].shape)}")

    from torch.utils.data import Subset

    ds = CellDataset(tok["genes"], tok["values"], labels)
    loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=PER_DEVICE_BATCH,
                        shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(ds, val_idx_a.tolist()), batch_size=PER_DEVICE_BATCH,
                            shuffle=False, drop_last=False)
    pad_id = vocab[pargs["pad_token"]]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fallback_used = False
    model.to(device)
    if device.type == "cuda":
        first_batch = next(iter(loader))
        if dtype == torch.bfloat16:
            try:
                preflight_dtype(model, first_batch, device, pad_id, dtype)
                log("bf16 preflight OK")
            except Exception as exc:  # noqa: BLE001
                log(f"bf16 preflight FAILED ({exc}); falling back to fp16")
                dtype = torch.float16
                fallback_used = True
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = args.out / args.dataset / f"naive_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stats = convergence_train(
        model=model, train_loader=loader, val_loader=val_loader,
        device=device, pad_id=pad_id, epochs=epochs,
        grad_accum=GRAD_ACCUM, lr=LR, weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS, dtype=dtype, run_dir=run_dir,
        y_true_val=y_true_val, naive_classes=naive_classes,
        naive_map=naive_map, frozen_baseline_rho=args.frozen_rho,
    )
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9
               if device.type == "cuda" else 0.0)
    log(f"training done ({stats['wall_s']:.1f}s) first_loss={stats['first_loss']:.4f} "
        f"best_val_f1={stats['best_val_f1']:.4f}@ep{stats['best_val_f1_epoch']} "
        f"collapse={stats['collapse']} peak_gpu={peak_gb:.2f}GB")

    args_out = {
        "dataset": args.dataset,
        "arm": "naive_ce",
        "seed": seed,
        "epochs": epochs,
        "dry_run": args.dry_run,
        "frozen_rho": args.frozen_rho,
        "dtype": str(dtype).replace("torch.", ""),
        "bf16_fallback_used": fallback_used,
        "n_cls": n_cls,
        "naive_classes": naive_classes,
        "naive_map": naive_map,
        "training": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_steps": WARMUP_STEPS,
            "batch_size": PER_DEVICE_BATCH, "grad_accum": GRAD_ACCUM,
            "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM, "max_len": MAX_LEN,
        },
        "data": {
            "n_train_cells": int(adata.n_obs),
            "n_val_cells": int(len(val_idx_a)),
            "test_donor": cfg["test_donor"],
            "val_split": "10% of train, stratified donor x cell_class_4 (LOCKED)",
        },
        "model": {k: pargs[k] for k in
                  ["ntoken", "embsize", "nheads", "d_hid", "nlayers",
                   "n_layers_cls", "dropout", "n_bins", "max_seq_len",
                   "input_emb_style", "cell_emb_style", "pad_token", "pad_value"]},
        "stats": stats,
        "peak_gpu_gb": round(peak_gb, 3),
        "timestamp_utc": _now_utc(),
    }
    (run_dir / "args.json").write_text(json.dumps(args_out, indent=2))
    with (run_dir / "CORNEATRAIN_DONE").open("w") as fh:
        fh.write(f"dataset={args.dataset} seed={seed}\n")
    log(f"saved {run_dir / 'args.json'} + CORNEATRAIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
