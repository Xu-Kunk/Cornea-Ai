#!/usr/bin/env python
"""scGPT cross-model reproduction - fine-tuning script (naive / ordinal CORN arms).

Replicates the Paper-1 Geneformer arms (naive softmax 31-class and ordinal CORN
4-class) on the scGPT whole-human checkpoint (save/scgpt_pretrained,
perturblab/scgpt-human; input_emb_style=continuous, nlayers=12, embsize=512,
cell_emb_style=cls, max_seq_len=1200, pad_token=<pad>, pad_value=-2).

The classification head is a fresh official ClsDecoder (d_model=512, nlayers=3,
randomly initialised: the pretrained checkpoint has no CLS head, args.json
``no_cls: true``); the encoder trunk is full-fine-tuned (not frozen).

Arms:
    naive   - CrossEntropyLoss over the 31 author_cell_type classes (all cells).
    ordinal - CORN ordinal regression (Cao et al., IEEE TNNLS 2020): K-1 = 3
              threshold logits, sum of masked binary BCEs over the 27,878
              epithelial cells (layer >= 0); 4-class ordinal label comes from
              cell_type_harmonized. Mirrors corneal_fm/ordinal.py.

Training recipe: batch 8 x grad-accum 4 = effective 32, bf16 autocast (fp16
fallback with report), lr 5e-5 AdamW (weight_decay 1e-3), linear warmup over
500 optimizer steps then constant, 3 epochs, seed fixed (torch + numpy +
random). Cells are truncated to max_len=1200 tokens keeping the leading <cls>
(keep_first_n_tokens=1 semantics, inherent in tokenize_and_pad_batch with
append_cls=True).

Usage (long runs launched by the scheduler; --dry-run is the smoke path):
    python scripts/scgpt_cls.py --arm naive   --seed 42
    python scripts/scgpt_cls.py --arm ordinal --seed 456
    python scripts/scgpt_cls.py --arm naive --seed 42 --dry-run --max-cells 2000

Per-run outputs (artifacts/scgpt_repro/checkpoints/<arm>_seed<seed>/):
    model.pt  args.json  runs_status  train_log.jsonl
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

PRETRAINED_DIR = PROJECT_ROOT / "save" / "scgpt_pretrained"
TRAIN_H5AD = PROJECT_ROOT / "data" / "scgpt_train.h5ad"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "scgpt_repro" / "checkpoints"
DEFAULT_LOG = Path("/tmp/opencode/scgpt_cls.log")

MAX_LEN = 1200
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4
LR = 5e-5
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 500
EPOCHS = 3
SEEDS = [42, 123, 456]
TOKENIZE_CHUNK = 8192
LOG_INTERVAL = 500

N_ORDINAL = 4
N_THRESHOLDS = N_ORDINAL - 1
EVAL_CLASSES = ["LSC", "TAC", "basal", "differentiated"]
HARMONIZED_MAP = {
    0: "LSC",
    1: "TAC",
    2: "basal",
    3: "differentiated",  # wing
    4: "differentiated",  # squamous
    NON_LAYER: "NON_LAYER",
}
N_TRAIN_CELLS_FULL = 79_684  # data/scgpt_train.h5ad (locked)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    """Print to stdout and append to the shared run log (/tmp/opencode/scgpt_cls.log)."""
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with DEFAULT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Vocabulary / model
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_vocab(pretrained_dir: Path = PRETRAINED_DIR) -> tuple[GeneVocab, dict[str, Any]]:
    """Load the scGPT vocab (special tokens ensured) + pretrained args.json.

    Uses the O(n) BuiltinVocab list path instead of ``GeneVocab.from_file``:
    the patched scgpt git-main ``vocab_compat`` backend rebuilds the whole
    ``_stoi`` dict on every insert, so ``from_dict`` over the 60,697-token
    census vocab takes ~163 s (O(n^2)). The fast path reproduces the exact
    token<->index mapping for consecutive indices (verified against the
    reference loader); non-consecutive vocabs fall back to the slow loader.
    """
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
        raise RuntimeError(
            f"vocab size {len(vocab)} != args ntoken {args['ntoken']}"
        )
    return vocab, args


def build_finetune_model(
    vocab: GeneVocab,
    args: dict[str, Any],
    n_cls: int,
    pretrained_dir: Path = PRETRAINED_DIR,
) -> TransformerModel:
    """Build the TransformerModel and load the pretrained trunk weights.

    use_fast_transformer=False makes load_pretrained auto-rename the flash-attn
    ``Wqkv`` keys to ``in_proj_`` (max diff 0.0, verified). The CLS head
    (n_cls=31/3) is randomly initialised: the pretrained checkpoint contains no
    ``cls_decoder`` keys (no_cls: true), so load_pretrained skips nothing and
    the head stays fresh.
    """
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
# Label mappings (locked: corneal_fm/caft.py CELL_TYPE_TO_LAYER +
# scripts/run_tier_b_eval.py HARMONIZED_MAP)
# ---------------------------------------------------------------------------


def author_type_to_layer(name: str) -> int:
    """31-class author_cell_type name -> 5-layer index {0..4} or NON_LAYER."""
    return CELL_TYPE_TO_LAYER.get(str(name), NON_LAYER)


def layer_to_harmonized(layer: int) -> str:
    """5-layer index -> 4-class harmonized label or NON_LAYER."""
    return HARMONIZED_MAP.get(int(layer), "NON_LAYER")


def author_type_to_harmonized(name: str) -> str:
    """31-class name -> 4-class harmonized label (composition of the above)."""
    return layer_to_harmonized(author_type_to_layer(name))


def naive_class_list(author_types: Any) -> list[str]:
    """Sorted unique author_cell_type names = the naive head's class order."""
    return sorted({str(t) for t in author_types})


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def tokenize_chunked(
    x_mat: Any,
    gene_ids: np.ndarray,
    vocab: GeneVocab,
    args: dict[str, Any],
    max_len: int = MAX_LEN,
    chunk: int = TOKENIZE_CHUNK,
) -> dict[str, torch.Tensor]:
    """Tokenize a (n, n_genes) sparse/dense matrix in chunks.

    Mirrors the official annotation-tutorial pipeline: continuous log1p values,
    append_cls=True (leading <cls> kept by the internal truncation),
    max_len token cap; the padding mask is derived as ``genes == pad_id``
    because this scgpt git-main no longer returns ``padding_mask`` from
    ``tokenize_and_pad_batch``.

    ``pad_batch`` pads each call to its own batch max (never up to max_len),
    so every chunk is right-padded here to the global effective width
    W = min(global_max_nonzero + 1, max_len) - identical to a single-shot call
    - before concatenation.
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
    """(genes, values, label) triples; label is the naive id or ordinal id (-1
    = non-epithelial, masked out of the ordinal loss)."""

    def __init__(self, genes: torch.Tensor, values: torch.Tensor, labels: torch.Tensor):
        self.genes = genes
        self.values = values
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.genes[i], self.values[i], self.labels[i]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def corn_loss(
    logits: torch.Tensor,
    ordinal_labels: torch.Tensor,
    valid: torch.Tensor,
    n_ordinal: int = N_ORDINAL,
) -> torch.Tensor:
    """CORN ordinal loss: K-1 binary BCEs, masked to valid (epithelial) cells.

    Numerically identical to corneal_fm/ordinal.py's ``_ordinal_targets`` +
    ``F.binary_cross_entropy_with_logits(..., reduction="mean")``; the
    all-invalid fallback keeps a grad-carrying zero (mirrors the reference).
    """
    device = logits.device
    if bool(valid.any()):
        y = ordinal_labels[valid]
        thresholds = torch.arange(1, n_ordinal, device=device)  # {1..K-1}
        targets = (y[:, None] >= thresholds[None, :]).float()
        return F.binary_cross_entropy_with_logits(
            logits[valid], targets, reduction="mean"
        )
    return logits.sum() * 0.0


def compute_loss(
    out: dict[str, torch.Tensor], labels: torch.Tensor, arm: str
) -> torch.Tensor:
    logits = out["cls_output"]
    if arm == "naive":
        return F.cross_entropy(logits, labels)
    return corn_loss(logits, labels, labels >= 0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
    """One forward+backward in the chosen dtype; raises if unsupported/NaN."""
    src, values, y = [b.to(device) for b in batch]
    model.train()
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(src, values, src_key_padding_mask=src.eq(pad_id), CLS=True)
        loss = compute_loss(out, y, arm)
    if not torch.isfinite(loss):
        raise RuntimeError(f"preflight loss not finite: {loss.item()}")
    loss.backward()
    model.zero_grad(set_to_none=True)


def train(
    model: TransformerModel,
    loader: DataLoader,
    arm: str,
    device: torch.device,
    pad_id: int,
    epochs: int,
    grad_accum: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    dtype: torch.dtype,
    log_path: Path,
) -> dict[str, Any]:
    """Full fine-tuning loop; returns loss/timing statistics."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_step = 0
    losses: list[float] = []
    first_loss: float | None = None
    last_mean: dict[str, float | None] = {"start": None, "end": None}
    micro_total = len(loader) * epochs
    t0_all = time.time()
    t_step = 0.0
    n_steps_timed = 0

    for epoch in range(epochs):
        model.train()
        running: list[float] = []
        opt.zero_grad(set_to_none=True)
        for step, (src, values, y) in enumerate(loader):
            src = src.to(device)
            values = values.to(device)
            y = y.to(device)
            mask = src.eq(pad_id)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type=device.type, dtype=dtype):
                out = model(src, values, src_key_padding_mask=mask, CLS=True)
                loss = compute_loss(out, y, arm)
            loss_scaled = loss.float() / grad_accum
            loss_scaled.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_step += time.time() - t0
            n_steps_timed += 1
            loss_val = float(loss.detach().float())
            losses.append(loss_val)
            running.append(loss_val)
            if first_loss is None:
                first_loss = loss_val
                last_mean["start"] = loss_val

            if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
                opt.step()
                opt_step += 1
                if opt_step <= warmup_steps:
                    _set_lr(opt, lr * opt_step / max(warmup_steps, 1))
                else:
                    _set_lr(opt, lr)
                opt.zero_grad(set_to_none=True)

            gstep = epoch * len(loader) + step
            if gstep % LOG_INTERVAL == 0 or gstep == micro_total - 1:
                mean500 = float(np.mean(running[-LOG_INTERVAL:]))
                cur_lr = opt.param_groups[0]["lr"]
                line = (
                    f"epoch={epoch + 1}/{epochs} step={gstep + 1}/{micro_total} "
                    f"loss_mean={mean500:.4f} first_loss={first_loss:.4f} "
                    f"lr={cur_lr:.2e} dt={time.time() - t0_all:.1f}s"
                )
                log(line)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "gstep": gstep,
                                "loss_mean": mean500,
                                "first_loss": first_loss,
                                "lr": cur_lr,
                                "wall_s": round(time.time() - t0_all, 2),
                            }
                        )
                        + "\n"
                    )
        last_mean["end"] = float(np.mean(running[-min(100, len(running)) :]))

    wall_s = time.time() - t0_all
    per_step = t_step / max(n_steps_timed, 1)
    return {
        "first_loss": first_loss,
        "losses": losses,
        "loss_start_mean": last_mean["start"],
        "loss_end_mean": last_mean["end"],
        "per_step_s": per_step,
        "wall_s": wall_s,
        "opt_steps": opt_step,
    }


def _dry_run_projection(
    stats: dict[str, Any], n_train_cells: int, max_len: int, batch_size: int
) -> dict[str, float]:
    """Project full 3-epoch, 79,684-cell runtime from the dry-run per-step time."""
    n_micro_full = int(np.ceil(N_TRAIN_CELLS_FULL / batch_size))
    total_micro = n_micro_full * EPOCHS
    per_run_s = stats["per_step_s"] * total_micro
    # tokenization cost scales linearly with cell count (dense chunk -> tokenize)
    per_cell_tok = stats.get("tokenize_s", 0.0) / max(n_train_cells, 1)
    tok_s = per_cell_tok * N_TRAIN_CELLS_FULL
    return {
        "micro_steps_per_epoch_full": n_micro_full,
        "total_micro_steps_3ep": total_micro,
        "per_run_train_s": per_run_s,
        "per_run_tokenize_s": tok_s,
        "per_run_total_s": per_run_s + tok_s,
        "six_runs_total_s": 6.0 * (per_run_s + tok_s),
        "six_runs_total_min": 6.0 * (per_run_s + tok_s) / 60.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["naive", "ordinal"], required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dry-run", action="store_true", help="1 epoch, small subset, assert loss decreases")
    ap.add_argument("--max-cells", type=int, default=None, help="cap training cells (dry-run)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=PER_DEVICE_BATCH)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    args = ap.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = 1 if args.dry_run else args.epochs
    max_cells = args.max_cells or N_TRAIN_CELLS_FULL

    log(f"=== scgpt_cls start arm={args.arm} seed={seed} dry_run={args.dry_run} "
        f"epochs={epochs} max_cells={max_cells} dtype={args.dtype} device={device} ===")

    # 1. vocab + model
    vocab, pargs = load_vocab()
    n_cls = 31 if args.arm == "naive" else N_THRESHOLDS
    t0 = time.time()
    model = build_finetune_model(vocab, pargs, n_cls)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model built ({time.time() - t0:.1f}s) n_cls={n_cls} params={n_params / 1e6:.2f}M")

    # 2. data
    import anndata as ad

    t0 = time.time()
    adata = ad.read_h5ad(TRAIN_H5AD)
    if max_cells < adata.n_obs:
        adata = adata[:max_cells].copy()
    obs = adata.obs
    naive_classes = naive_class_list(obs["author_cell_type"])
    class_to_id = {c: i for i, c in enumerate(naive_classes)}
    naive_labels = np.array(
        [class_to_id[c] for c in obs["author_cell_type"].astype(str)], dtype=np.int64
    )
    ordinal_labels = np.full(adata.n_obs, -1, dtype=np.int64)
    harm = obs["cell_type_harmonized"].astype(str).to_numpy()
    for name, k in zip(EVAL_CLASSES, range(N_ORDINAL)):
        ordinal_labels[harm == name] = k
    n_ordinal_valid = int((ordinal_labels >= 0).sum())
    labels = torch.as_tensor(naive_labels if args.arm == "naive" else ordinal_labels)
    log(f"data loaded ({time.time() - t0:.1f}s) n_cells={adata.n_obs} "
        f"naive_classes={len(naive_classes)} ordinal_valid={n_ordinal_valid}")

    # 3. tokenize (single-pass pipeline; genes filtered to vocab)
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

    # 4. loader
    ds = CellDataset(tok["genes"], tok["values"], labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    pad_id = vocab[pargs["pad_token"]]

    # 5. dtype preflight (bf16 -> fp16 fallback, reported)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fallback_used = False
    model.to(device)
    if device.type == "cuda":
        first_batch = next(iter(loader))
        if dtype == torch.bfloat16:
            try:
                preflight_dtype(model, first_batch, args.arm, device, pad_id, dtype)
                log("bf16 preflight OK")
            except Exception as exc:  # noqa: BLE001 - fallback path is the contract
                log(f"bf16 preflight FAILED ({exc}); falling back to fp16")
                dtype = torch.float16
                fallback_used = True
        torch.cuda.reset_peak_memory_stats(device)

    # 6. train
    log(f"training start (dtype={dtype}, batch={args.batch_size} x "
        f"accum={args.grad_accum}, warmup={WARMUP_STEPS})")
    run_dir = args.out / f"{args.arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    stats = train(
        model=model,
        loader=loader,
        arm=args.arm,
        device=device,
        pad_id=pad_id,
        epochs=epochs,
        grad_accum=args.grad_accum,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        dtype=dtype,
        log_path=log_path,
    )
    stats["tokenize_s"] = tokenize_s
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )
    log(f"training done ({stats['wall_s']:.1f}s) first_loss={stats['first_loss']:.4f} "
        f"end_mean={stats['loss_end_mean']:.4f} per_step={stats['per_step_s'] * 1e3:.1f}ms "
        f"peak_gpu={peak_gb:.2f}GB")

    # 7. dry-run correctness guard: loss must decrease over the epoch
    loss_decreased = bool(stats["loss_end_mean"] < stats["loss_start_mean"])
    if args.dry_run:
        if loss_decreased:
            log(f"DRY-RUN GUARD PASS: loss {stats['loss_start_mean']:.4f} -> "
                f"{stats['loss_end_mean']:.4f} (decreasing)")
        else:
            log(f"DRY-RUN GUARD FAIL: loss {stats['loss_start_mean']:.4f} -> "
                f"{stats['loss_end_mean']:.4f} (not decreasing)")

    # 8. 6-run full projection
    proj = _dry_run_projection(stats, adata.n_obs, args.max_len, args.batch_size)
    log("6-RUN PROJECTION (full 79,684 cells, 3 epochs):")
    for k, v in proj.items():
        log(f"    {k} = {v:.1f}" if isinstance(v, float) else f"    {k} = {v}")

    # 9. save artifacts
    args_out = {
        "arm": args.arm,
        "seed": seed,
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
            "n_train_cells": int(adata.n_obs),
            "n_genes_in_vocab": int(len(gene_ids)),
            "n_ordinal_valid": int(n_ordinal_valid),
        },
        "results": {
            "first_loss": stats["first_loss"],
            "loss_start_mean": stats["loss_start_mean"],
            "loss_end_mean": stats["loss_end_mean"],
            "loss_decreased": loss_decreased,
            "per_step_s": stats["per_step_s"],
            "wall_s": stats["wall_s"],
            "peak_gpu_gb": peak_gb,
            "projection": proj,
        },
        "timestamp_utc": _now_utc(),
    }
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "args.json").write_text(json.dumps(args_out, indent=2), encoding="utf-8")
    (run_dir / "runs_status").write_text(
        "DONE\n" + _now_utc() + f"\narm={args.arm} seed={seed}\n", encoding="utf-8"
    )
    log(f"saved run artifacts to {run_dir}")

    if args.dry_run and not loss_decreased:
        log("DRY-RUN FAILED: loss did not decrease")
        return 1
    log(f"=== scgpt_cls done arm={args.arm} seed={seed} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
