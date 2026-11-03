#!/usr/bin/env python
"""van Zyl head-init perturbation (basin-selection test, FROZEN 2026-08-20).

Single-variable extension of the §9 convergence arm: training seed LOCKED to
42 (the collapse reference) for every run; the ONLY perturbed variable is the
random initialization of the CLS classification head (model.cls_decoder).

Variants (frozen before run, recorded in args.json):
  * headscale_0.01 -- every Linear in cls_decoder: weight ~ N(0, 0.01), bias 0
  * headscale_1.0  -- every Linear in cls_decoder: weight ~ N(0, 1.0),  bias 0
  * headseed_999   -- DEFAULT nn.Linear init (kaiming_uniform + uniform bias)
                      re-drawn under torch.manual_seed(999) (same distribution
                      as the reference head, different random draw)

Reference (no re-init): seed42 collapse (final rho -0.091 / best-val 0.337)
vs seed123 keep (0.833/0.577) and seed456 keep (0.765/0.404) -- the §9
convergence arm. If a head-init variant flips seed-42 to keep (rho >= 0.6547,
the locked collapse threshold), the basin-selection hypothesis is confirmed.

Determinism contract (the single-variable guarantee): the torch RNG state is
snapshotted immediately AFTER build_finetune_model() (i.e. after the default
head-init draws) and RESTORED after the head re-init. Everything downstream
(data shuffle order, dropout masks, optimizer noise) therefore consumes the
identical RNG stream as the reference seed42 run -- the ONLY difference is the
cls_decoder weights. numpy.random/random are untouched by the re-init.

All other hyperparameters identical to scgpt_cls_vanzyl_convergence.py
(PREREG §9 LOCKED): 15 epochs, lr 5e-5, wd 1e-3, warmup 500, effective batch
32, bf16, max_len 4253, val split = split_val_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import BuiltinVocab
from scgpt.utils import load_pretrained
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # corneal_fm is a project package (script dir is scripts/)
from corneal_fm.caft import CELL_TYPE_TO_LAYER, NON_LAYER  # noqa: E402
from collections import Counter  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from scgpt_frozen_diag_rebuild import l2_norm_rows, label_shuffle_null  # noqa: E402

PRETRAINED_DIR = PROJECT_ROOT / "pretrained" / "scgpt_pretrained"
TRAIN_H5AD = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_train.h5ad"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "head_init" / "checkpoints"
DEFAULT_LOG = Path("/tmp/opencode/scgpt_cls_vanzyl_headinit.log")

MAX_LEN = 4253  # PREREG §4 escalation (panel 4253)
EPOCHS = 15     # scGPT official budget (§9, single EPOCHS definition)
VAL_NPZ = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence" / "val_indices.npz"
VAL_MANIFEST = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence" / "split_val_manifest.json"
FROZEN_BASELINE_RHO = 0.8047   # G1 rho_frz(L2-normed), full analysis set (§8)
COLLAPSE_THRESHOLD = round(FROZEN_BASELINE_RHO - 0.15, 4)   # 0.6547 (LOCKED §9)
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4
LR = 5e-5
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 500
TRAIN_SEED = 42   # LOCKED: the collapse reference seed (basin-selection test)
HEAD_SEED_999 = 999
TOKENIZE_CHUNK = 8192
LOG_INTERVAL = 100

VARIANTS = ["headscale_0.01", "headscale_1.0", "headseed_999"]

N_ORDINAL = 3
N_THRESHOLDS = N_ORDINAL - 1
EVAL_CLASSES = ["LSC", "basal", "differentiated"]
HARMONIZED_MAP = {
    0: "LSC",
    1: "basal",
    2: "differentiated",
    NON_LAYER: "NON_LAYER",
}
N_TRAIN_CELLS_FULL = 42_482
ORDINAL_LABEL_COL = "cell_class_4"


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with DEFAULT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


@lru_cache(maxsize=1)
def load_vocab(pretrained_dir: Path = PRETRAINED_DIR) -> tuple[GeneVocab, dict[str, Any]]:
    import json as _json
    from scgpt.tokenizer import GeneVocab as _GV

    vocab = _GV.from_file(pretrained_dir / "vocab.json")
    pargs = _json.loads((pretrained_dir / "args.json").read_text(encoding="utf-8"))
    return vocab, pargs


def build_finetune_model(
    vocab: GeneVocab,
    args: dict[str, Any],
    n_cls: int,
    pretrained_dir: Path = PRETRAINED_DIR,
) -> TransformerModel:
    """Identical to scgpt_cls_vanzyl_convergence.build_finetune_model
    (LOCKED): use_fast_transformer=False + load_pretrained(strict=False);
    the CLS head is randomly initialised by the TransformerModel constructor
    and stays fresh (no cls_decoder keys in the pretrained checkpoint)."""
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


def reinit_cls_head(model: TransformerModel, variant: str) -> None:
    """Perturb ONLY model.cls_decoder per the frozen variant spec.

    headscale_*: every Linear weight ~ N(0, std), bias zeros -- magnitude test.
    headseed_999: default nn.Linear.reset_parameters() re-drawn under
    torch.manual_seed(999) -- same distribution, different draw.

    Returns None; the caller restores the torch RNG state afterwards.
    """
    from torch import nn

    head = model.cls_decoder
    if variant.startswith("headscale_"):
        std = float(variant.split("_")[1])
        with torch.no_grad():
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    elif variant == "headseed_999":
        torch.manual_seed(HEAD_SEED_999)
        for m in head.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()  # default kaiming_uniform + uniform bias
    else:
        raise ValueError(f"unknown variant {variant}")
    log(f"head re-initialized: variant={variant}")


def author_type_to_layer(name: str) -> int:
    return CELL_TYPE_TO_LAYER.get(str(name), NON_LAYER)


def layer_to_harmonized(layer: int) -> str:
    return HARMONIZED_MAP.get(int(layer), "NON_LAYER")


def author_type_to_harmonized(name: str) -> str:
    return layer_to_harmonized(author_type_to_layer(name))


def naive_class_list(author_types: Any) -> list[str]:
    return sorted({str(t) for t in author_types})


def tokenize_chunked(
    x_mat: Any,
    gene_ids: np.ndarray,
    vocab: GeneVocab,
    args: dict[str, Any],
    max_len: int = MAX_LEN,
    chunk: int = TOKENIZE_CHUNK,
) -> dict[str, torch.Tensor]:
    """Tokenize a (n, n_genes) sparse/dense matrix in chunks.

    VERBATIM from scgpt_cls_vanzyl_convergence.tokenize_chunked (LOCKED):
    continuous log1p values, append_cls=True, max_len token cap; the padding
    mask is derived as ``genes == pad_id`` (this scgpt git-main no longer
    returns ``padding_mask``); each chunk right-padded to the global effective
    width W = min(global_max_nonzero + 1, max_len).
    """
    n = x_mat.shape[0]
    if n == 0:
        empty = torch.empty(0, max_len, dtype=torch.long)
        mask = torch.ones(0, max_len, dtype=torch.bool)
        return {"genes": empty, "values": empty.float(), "mask": mask}
    if hasattr(x_mat, "getnnz"):
        max_nnz = int(x_mat.getnnz(axis=1).max())
    else:
        max_nnz = int(np.count_nonzero(np.asarray(x_mat, dtype=bool), axis=1).max())
    width = min(max_nnz + 1, max_len)  # +1 for the leading <cls>
    pad_id = vocab[args["pad_token"]]
    pad_value = args["pad_value"]

    pieces: list[tuple[torch.Tensor, torch.Tensor]] = []
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
        if w < width:  # right-pad this chunk up to the global width
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
    def __init__(self, genes: torch.Tensor, values: torch.Tensor, labels: torch.Tensor):
        self.genes = genes
        self.values = values
        self.labels = labels

    def __len__(self) -> int:
        return self.genes.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.genes[i], self.values[i], self.labels[i]


def compute_loss(out: dict[str, Any], y: torch.Tensor, arm: str) -> torch.Tensor:
    logits = out["cls_output"]
    return F.cross_entropy(logits, y)


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def preflight_dtype(
    model: TransformerModel,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    arm: str,
    device: torch.device,
    pad_id: int,
    dtype: torch.dtype,
) -> None:
    src, values, y = batch
    src = src.to(device)
    values = values.to(device)
    y = y.to(device)
    model.train()
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(src, values, src_key_padding_mask=src.eq(pad_id), CLS=True)
        loss = compute_loss(out, y, arm)
    loss.backward()
    model.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        raise RuntimeError(f"preflight loss not finite: {loss.item()}")


def headinit_train(
    model, train_loader, val_loader, device, pad_id, epochs,
    grad_accum, lr, weight_decay, warmup_steps, dtype,
    run_dir, y_true_val, naive_classes, naive_map,
) -> dict[str, Any]:
    """Training loop IDENTICAL to scgpt_cls_vanzyl_convergence.convergence_train
    (LOCKED §9 step code): per-epoch train + val eval (macro-F1, rho raw/normed,
    label-shuffle z n=100) + per-epoch checkpoint + best-val-F1 selection."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_step = 0
    t0_all = time.time()
    trajectory: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_val_f1_epoch: int | None = None
    best_val_rho: float | None = None
    best_val_rho_epoch: int | None = None
    first_loss: float | None = None
    present = sorted(set(y_true_val.tolist()) & set(EVAL_CLASSES))

    for epoch in range(1, epochs + 1):
        model.train()
        running: list[float] = []
        opt.zero_grad(set_to_none=True)
        for step, (src, values, y) in enumerate(train_loader):
            src = src.to(device); values = values.to(device); y = y.to(device)
            mask = src.eq(pad_id)
            with torch.autocast(device_type=device.type, dtype=dtype):
                out = model(src, values, src_key_padding_mask=mask, CLS=True)
                loss = compute_loss(out, y, "naive")
            (loss.float() / grad_accum).backward()
            loss_val = float(loss.detach().float())
            running.append(loss_val)
            if first_loss is None:
                first_loss = loss_val
            if (step + 1) % grad_accum == 0 or step + 1 == len(train_loader):
                opt.step()
                opt_step += 1
                if opt_step <= warmup_steps:
                    _set_lr(opt, lr * opt_step / max(warmup_steps, 1))
                else:
                    _set_lr(opt, lr)
                opt.zero_grad(set_to_none=True)
        train_mean = float(np.mean(running[-min(200, len(running)):]))

        model.eval()
        embs: list[np.ndarray] = []
        clss: list[np.ndarray] = []
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
        null = label_shuffle_null(vemb_n, y_true_val, EVAL_CLASSES, n=100, seed=42)
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

    collapse_epoch: int | None = None
    for epoch in range(1, epochs + 1):
        if trajectory[epoch - 1]["val_rho_normed"] < COLLAPSE_THRESHOLD:
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
                     "threshold": COLLAPSE_THRESHOLD,
                     "frozen_baseline": FROZEN_BASELINE_RHO},
        "wall_s": time.time() - t0_all,
        "per_step_s": (time.time() - t0_all) / max(epochs * len(train_loader), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dry-run", action="store_true",
                    help="1 epoch, small subset, assert loss decreases")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=PER_DEVICE_BATCH)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    args = ap.parse_args()

    seed = TRAIN_SEED  # LOCKED: 42 for every variant (single-variable test)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = 1 if args.dry_run else args.epochs
    max_cells = args.max_cells or N_TRAIN_CELLS_FULL

    log(f"=== scgpt_cls_headinit variant={args.variant} train_seed={seed} "
        f"dry_run={args.dry_run} epochs={epochs} max_cells={max_cells} "
        f"dtype={args.dtype} device={device} ===")

    vocab, pargs = load_vocab()

    import anndata as ad
    t0 = time.time()
    adata = ad.read_h5ad(TRAIN_H5AD)
    if max_cells < adata.n_obs:
        adata = adata[:max_cells].copy()
    obs = adata.obs
    if max_cells >= N_TRAIN_CELLS_FULL and adata.n_obs != N_TRAIN_CELLS_FULL:
        raise RuntimeError(
            f"train h5ad n_cells {adata.n_obs} != locked {N_TRAIN_CELLS_FULL} "
            "(split_manifest.json) -- refusing to train on a drifted split"
        )
    author_all = obs["author_cell_type"].astype(str).to_numpy()
    cc4_all = obs["cell_class_4"].astype(str).to_numpy()

    if args.dry_run and max_cells < N_TRAIN_CELLS_FULL:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(adata.n_obs)
        n_tr = int(adata.n_obs * 0.9)
        train_idx, val_idx_a = perm[:n_tr], perm[n_tr:]
        log(f"SMOKE local split: train={len(train_idx)} val={len(val_idx_a)}")
    else:
        vd = np.load(VAL_NPZ)
        val_idx_a = vd["val_idx"].astype(np.int64)
        val_mask = np.zeros(adata.n_obs, dtype=bool)
        val_mask[val_idx_a] = True
        train_idx = np.where(~val_mask)[0]
        vm = json.loads(VAL_MANIFEST.read_text(encoding="utf-8"))
        if int(len(train_idx)) != vm["n_train"] or int(len(val_idx_a)) != vm["n_val"]:
            raise RuntimeError(
                f"val split drift: train={len(train_idx)} val={len(val_idx_a)} "
                f"!= manifest {vm['n_train']}/{vm['n_val']} (locked)")
        log(f"LOCKED val split: train={len(train_idx)} val={len(val_idx_a)} "
            f"(manifest ok, sha={vm['sha256']['scgpt_vanzyl_train.h5ad'][:8]})")

    tr_author = author_all[train_idx]
    tr_cc4 = cc4_all[train_idx]
    naive_classes = sorted(set(tr_author.tolist()))
    class_to_id = {c: i for i, c in enumerate(naive_classes)}
    naive_labels = np.array(
        [class_to_id.get(c, -1) for c in author_all], dtype=np.int64
    )
    naive_cont: dict[str, Counter] = {}
    for au, l in zip(tr_author, tr_cc4):
        naive_cont.setdefault(au, Counter())[l] += 1
    naive_map = {au: max(c, key=c.get) for au, c in naive_cont.items()}
    labels = torch.as_tensor(naive_labels)
    y_true_val = cc4_all[val_idx_a]
    log(f"data loaded ({time.time() - t0:.1f}s) n_cells={adata.n_obs} "
        f"naive_classes={len(naive_classes)} n_val={len(val_idx_a)} "
        f"val_class_dist={dict(Counter(y_true_val.tolist()))}")

    n_cls = len(naive_classes)
    t0 = time.time()
    model = build_finetune_model(vocab, pargs, n_cls)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model built ({time.time() - t0:.1f}s) n_cls={n_cls} "
        f"params={n_params / 1e6:.2f}M")

    # ---- THE SINGLE-VARIABLE PERTURBATION --------------------------------
    # Snapshot torch RNG (CPU + CUDA) right after the default head-init draws,
    # re-init ONLY cls_decoder, then restore -> training stream identical to
    # the reference seed42 run; the only difference is the head weights.
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
    reinit_cls_head(model, args.variant)
    torch.set_rng_state(torch_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)
    # ----------------------------------------------------------------------

    t0 = time.time()
    keep = np.array([g in vocab for g in adata.var_names])
    gene_ids = np.array([vocab[g] for g in adata.var_names[keep]], dtype=np.int64)
    x_mat = adata[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = tokenize_chunked(x_mat, gene_ids, vocab, pargs, max_len=args.max_len)
    tokenize_s = time.time() - t0
    log(f"tokenized ({tokenize_s:.1f}s) genes_in_vocab={len(gene_ids)} "
        f"seq_len={tuple(tok['genes'].shape)}")
    pad_id = vocab[pargs["pad_token"]]

    from torch.utils.data import Subset
    ds = CellDataset(tok["genes"], tok["values"], labels)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx_a.tolist())
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False)
    pad_id = vocab[pargs["pad_token"]]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fallback_used = False
    model.to(device)
    if device.type == "cuda":
        first_batch = next(iter(loader))
        if dtype == torch.bfloat16:
            try:
                preflight_dtype(model, first_batch, "naive", device, pad_id, dtype)
                log("bf16 preflight OK")
            except Exception as exc:  # noqa: BLE001 - fallback path is the contract
                log(f"bf16 preflight FAILED ({exc}); falling back to fp16")
                dtype = torch.float16
                fallback_used = True
        torch.cuda.reset_peak_memory_stats(device)

    log(f"training start (dtype={dtype}, batch={args.batch_size} x "
        f"accum={args.grad_accum}, warmup={WARMUP_STEPS})")
    run_dir = args.out / f"naive_seed{seed}_{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stats = headinit_train(
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        device=device,
        pad_id=pad_id,
        epochs=epochs,
        grad_accum=args.grad_accum,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        dtype=dtype,
        run_dir=run_dir,
        y_true_val=y_true_val,
        naive_classes=naive_classes,
        naive_map=naive_map,
    )
    stats["tokenize_s"] = tokenize_s
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )
    log(f"training done ({stats['wall_s']:.1f}s) first_loss={stats['first_loss']:.4f} "
        f"best_val_f1={stats['best_val_f1']:.4f}@ep{stats['best_val_f1_epoch']} "
        f"collapse={stats['collapse']} peak_gpu={peak_gb:.2f}GB")

    if len(stats["trajectory"]) >= 2:
        loss_decreased = bool(
            stats["trajectory"][-1]["train_loss_mean"]
            < stats["trajectory"][0]["train_loss_mean"])
    else:
        loss_decreased = bool(stats["first_loss"] is not None)
    if args.dry_run:
        log(f"DRY-RUN GUARD {'PASS' if loss_decreased else 'FAIL'}: "
            f"loss {stats['trajectory'][0]['train_loss_mean']:.4f} -> "
            f"{stats['trajectory'][-1]['train_loss_mean']:.4f} (epoch 1)")

    args_out = {
        "task": "head-init perturbation (basin-selection test), FROZEN 2026-08-20",
        "arm": "naive",
        "train_seed": seed,
        "variant": args.variant,
        "head_reinit": {
            "headscale_0.01": "Linear w ~ N(0, 0.01), b=0",
            "headscale_1.0": "Linear w ~ N(0, 1.0), b=0",
            "headseed_999": "default nn.Linear init, torch.manual_seed(999)",
        }[args.variant],
        "rng_contract": "torch RNG snapshotted after model build, restored "
                        "after head re-init -> downstream stream identical to "
                        "reference seed42 (only cls_decoder weights differ)",
        "epochs": epochs,
        "dry_run": args.dry_run,
        "max_cells": int(adata.n_obs),
        "dtype": str(dtype).replace("torch.", ""),
        "bf16_fallback_used": fallback_used,
        "n_cls": n_cls,
        "naive_classes": naive_classes,
        "model": {
            k: pargs[k]
            for k in [
                "ntoken", "embsize", "nheads", "d_hid", "nlayers",
                "n_layers_cls", "dropout", "n_bins", "max_seq_len",
                "input_emb_style", "cell_emb_style", "pad_token", "pad_value",
            ]
        },
        "training": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "max_len": args.max_len,
        },
        "data": {
            "n_train_cells_full": int(adata.n_obs),
            "n_train_cells_90pct": int(len(train_idx)),
            "n_val_cells": int(len(val_idx_a)),
            "n_genes_in_vocab": int(len(gene_ids)),
        },
        "results": {
            "first_loss": stats["first_loss"],
            "best_val_f1": stats["best_val_f1"],
            "best_val_f1_epoch": stats["best_val_f1_epoch"],
            "best_val_rho": stats["best_val_rho"],
            "best_val_rho_epoch": stats["best_val_rho_epoch"],
            "collapse": stats["collapse"],
            "per_step_s": stats["per_step_s"],
            "wall_s": stats["wall_s"],
            "peak_gpu_gb": peak_gb,
        },
        "reference": {
            "seed42_collapse": {"final_rho": -0.0908, "best_val_rho": 0.3373},
            "seed123_keep": {"final_rho": 0.8329, "best_val_rho": 0.5774},
            "seed456_keep": {"final_rho": 0.7647, "best_val_rho": 0.4039},
            "source": "convergence/final_test_eval.json (PREREG §9)",
        },
        "timestamp_utc": _now_utc(),
    }
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "args.json").write_text(json.dumps(args_out, indent=2), encoding="utf-8")
    (run_dir / "runs_status").write_text(
        "DONE\n" + _now_utc() + f"\nvariant={args.variant} seed={seed}\n", encoding="utf-8"
    )
    log(f"saved run artifacts to {run_dir}")

    if args.dry_run and not loss_decreased:
        log("DRY-RUN FAILED: loss did not decrease")
        return 1
    log(f"=== scgpt_cls_headinit done variant={args.variant} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
