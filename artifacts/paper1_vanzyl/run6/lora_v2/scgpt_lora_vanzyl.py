#!/usr/bin/env python
"""E4: scGPT LoRA fine-tuning control (van Zyl same-corpus, user-approved).

Purpose: test whether a lightweight parameter-efficient adapter (LoRA)
preserves the differentiation gradient, as a pressure test on the CORN
uniqueness claim (scPEFT framing). scGPT repo has no built-in LoRA; this
script implements QKV LoRA (r=8, alpha=16) by replacing each encoder
layer's nn.MultiheadAttention with a semantically identical wrapper whose
q/k/v projections carry low-rank adapters; with B=0 initialization the
wrapper output is identical to the base attention (verified in smoke).

Protocol mirrors the §9 convergence arm: naive CE head over the train-90%
author classes, 15 epochs, locked 90/10 stratified validation partition,
per-epoch checkpoint + validation macro-F1 / rho / label-shuffle z
trajectory, best-validation (macro-F1) checkpoint as primary and
final-epoch as sensitivity, both evaluated inline on the locked test donor
(Hu0216, 20,158 cells) with the same metric implementations
(macro_f1_ci/rho_and_null from the 6-run eval suite). Only the LoRA
adapters and the random-init classification head are trained; the base
trunk is frozen. Seeds: 42, 123, 456 (deterministic choice, recorded).
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scgpt_cls  # noqa: E402
from scgpt_frozen_diag_rebuild import l2_norm_rows, label_shuffle_null  # noqa: E402
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

SCGPT_PRETRAINED = ROOT / "pretrained" / "scgpt_pretrained"
TRAIN_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_train.h5ad"
TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
VAL_NPZ = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence" / "val_indices.npz"
VAL_MANIFEST = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence" / "split_val_manifest.json"
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "lora"
LAYER_ORDER = ["LSC", "basal", "differentiated"]
N_NULL = 100
LORA_R = 8
LORA_ALPHA = 16
EPOCHS = 15
SEEDS_E4 = [42, 123, 456]
LR = 5e-5
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-3
N_TRAIN_CELLS_FULL = 42_482


def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class LoRAAttention(nn.Module):
    """Semantically identical replacement for nn.MultiheadAttention(batch_first)
    with LoRA on the q/k/v projections (B=0 init -> identical to base)."""

    def __init__(self, mha: nn.MultiheadAttention, r: int = LORA_R,
                 alpha: int = LORA_ALPHA):
        super().__init__()
        d = mha.embed_dim
        self.embed_dim = d
        self.num_heads = mha.num_heads
        self.head_dim = d // mha.num_heads
        self.dropout = mha.dropout
        self.scaling = alpha / r
        W = mha.in_proj_weight.detach()
        b = mha.in_proj_bias.detach()
        self.batch_first = True  # TransformerEncoderLayer queries this attribute
        self.in_proj_weight = W  # informational (torch layer checks in_proj_bias)
        self.in_proj_bias = b
        # remaining nn.MultiheadAttention attributes queried by torch internals
        self._qkv_same_embed_dim = True
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.q.weight.data.copy_(W[:d]); self.q.bias.data.copy_(b[:d])
        self.k.weight.data.copy_(W[d:2 * d]); self.k.bias.data.copy_(b[d:2 * d])
        self.v.weight.data.copy_(W[2 * d:]); self.v.bias.data.copy_(b[2 * d:])
        self.out_proj = mha.out_proj
        for mod in (self.q, self.k, self.v, self.out_proj):
            for p in mod.parameters():
                p.requires_grad_(False)
        for name in ("q", "k", "v"):
            setattr(self, f"lora_A_{name}", nn.Parameter(torch.zeros(r, d)))
            setattr(self, f"lora_B_{name}", nn.Parameter(torch.zeros(d, r)))
            getattr(self, f"lora_A_{name}").data.normal_(0.0, 0.02)
            getattr(self, f"lora_B_{name}").data.zero_()

    def _delta(self, name: str) -> torch.Tensor:
        return self.scaling * (getattr(self, f"lora_B_{name}")
                               @ getattr(self, f"lora_A_{name}"))

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        q = F.linear(query, self.q.weight + self._delta("q"), self.q.bias)
        k = F.linear(key, self.k.weight + self._delta("k"), self.k.bias)
        v = F.linear(value, self.v.weight + self._delta("v"), self.v.bias)
        B, S, D = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        m = None
        if key_padding_mask is not None:
            m = key_padding_mask[:, None, None, :].expand(B, self.num_heads, S, S)
        if attn_mask is not None:
            m = attn_mask if m is None else (m + attn_mask)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out), None


class LoRALayer(nn.Module):
    """Exact replica of nn.TransformerEncoderLayer.forward (batch_first,
    norm_first) that calls the LoRA-wrapped self_attn explicitly, bypassing
    torch's C++ fast-path dispatch (which would use the base packed weights
    and silently ignore the adapters at eval time)."""

    def __init__(self, layer: nn.TransformerEncoderLayer, r: int = LORA_R,
                 alpha: int = LORA_ALPHA):
        super().__init__()
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.dropout = layer.dropout
        self.dropout1 = layer.dropout1
        self.dropout2 = layer.dropout2
        self.activation = layer.activation
        self.norm_first = layer.norm_first
        self.self_attn = LoRAAttention(layer.self_attn, r=r, alpha=alpha)

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask,
                                              src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def wrap_lora(model, r: int = LORA_R, alpha: int = LORA_ALPHA) -> list[str]:
    """Replace every encoder-layer self_attn with LoRAAttention; freeze the
    base trunk; return the names of trainable parameters (head + adapters)."""
    wrapped = []
    # disable nested-tensor packing: the LoRA attention works on padded
    # (B, S, D) tensors; nesting would otherwise be applied by the encoder
    model.transformer_encoder.enable_nested_tensor = False
    model.transformer_encoder.use_nested_tensor = False  # forward consults this
    for i, layer in enumerate(model.transformer_encoder.layers):
        model.transformer_encoder.layers[i] = LoRALayer(layer, r=r, alpha=alpha)
        wrapped.append(i)
    model.requires_grad_(False)
    for p in model.cls_decoder.parameters():
        p.requires_grad_(True)
    # CRITICAL FIX (2026-08-20): requires_grad_(False) above also froze the
    # freshly created LoRA adapters; the first E4 run silently trained the
    # head only (val rho constant = frozen level). Re-enable the adapters.
    n_adapt = 0
    for name, p in model.named_parameters():
        if "lora_A_" in name or "lora_B_" in name:
            p.requires_grad_(True)
            n_adapt += p.numel()
    n_head = sum(p.numel() for p in model.cls_decoder.parameters()
                 if p.requires_grad)
    log(f"LoRA wrapped layers={wrapped} (r={r}, alpha={alpha}); "
        f"trainable = head {n_head / 1e6:.3f}M + adapters {n_adapt / 1e6:.3f}M")
    return [n for n, p in model.named_parameters() if p.requires_grad]


def macro_f1_ci(y_true, y_pred, n_boot=1000, seed=42) -> dict:
    present = sorted(set(y_true.tolist()) & set(LAYER_ORDER))
    m = float(f1_score(y_true, y_pred, labels=present, average="macro",
                       zero_division=0))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = f1_score(y_true[b], y_pred[b], labels=present,
                            average="macro", zero_division=0)
    return {"macro_f1": m, "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5))}


def rho_and_null(emb, labels) -> dict:
    rho_raw = float(marker_gradient_correlation(emb, labels, LAYER_ORDER))
    emb_n = l2_norm_rows(emb)
    rho_n = float(marker_gradient_correlation(emb_n, labels, LAYER_ORDER))
    null = label_shuffle_null(emb_n, labels, LAYER_ORDER, n=N_NULL, seed=42)
    return {"rho_raw": rho_raw, "rho_normed": rho_n,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_z": float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12)),
            "null_pctile": float((null <= rho_n).mean())}


def decode_and_eval(model, genes, values, mask, device, y_true,
                    naive_classes, naive_map, batch=32) -> dict:
    """Forward a tokenized set, decode naive preds, return F1/CI + rho/null."""
    embs, clss = [], []
    with torch.no_grad():
        for s in range(0, genes.shape[0], batch):
            src = genes[s:s + batch].to(device)
            vals = values[s:s + batch].to(device)
            pad = mask[s:s + batch].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(src, vals, src_key_padding_mask=pad, CLS=True)
            embs.append(out["cell_emb"].float().cpu().numpy())
            clss.append(out["cls_output"].float().cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    cls = np.concatenate(clss, axis=0).astype(np.float32)
    pred = np.array([naive_map.get(naive_classes[int(i)], "NON_LAYER")
                     for i in cls.argmax(1)], dtype=str)
    res = {"macro_f1": macro_f1_ci(y_true, pred), **rho_and_null(emb, y_true),
           "pred_dist": dict(Counter(pred.tolist()))}
    return res


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=SEEDS_E4, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--identity-check", action="store_true",
                    help="verify B=0 LoRA output == base model output")
    args = ap.parse_args()
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab, pargs = scgpt_cls.load_vocab(pretrained_dir=SCGPT_PRETRAINED)
    import anndata as ad
    a = ad.read_h5ad(TRAIN_H5AD)
    author_all = a.obs["author_cell_type"].astype(str).to_numpy()
    cc4_all = a.obs["cell_class_4"].astype(str).to_numpy()
    vd = np.load(VAL_NPZ)
    val_mask = np.zeros(a.n_obs, dtype=bool)
    val_mask[vd["val_idx"].astype(np.int64)] = True
    train_idx = np.where(~val_mask)[0]
    val_idx_a = np.where(val_mask)[0]
    vm = json.loads(VAL_MANIFEST.read_text(encoding="utf-8"))
    assert len(train_idx) == vm["n_train"] and len(val_idx_a) == vm["n_val"]
    tr_author, tr_cc4 = author_all[train_idx], cc4_all[train_idx]
    naive_classes = sorted(set(tr_author.tolist()))
    class_to_id = {c: i for i, c in enumerate(naive_classes)}
    naive_labels = np.array([class_to_id.get(c, -1) for c in author_all],
                            dtype=np.int64)
    cont: dict[str, Counter] = {}
    for au, l in zip(tr_author, tr_cc4):
        cont.setdefault(au, Counter())[l] += 1
    naive_map = {au: max(c, key=c.get) for au, c in cont.items()}
    log(f"seed={seed} train={len(train_idx)} val={len(val_idx_a)} "
        f"n_cls={len(naive_classes)}")

    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = scgpt_cls.tokenize_chunked(x_mat, gene_ids, vocab, pargs, max_len=4253)
    labels = torch.as_tensor(naive_labels)
    pad_id = vocab[pargs["pad_token"]]

    model = scgpt_cls.build_finetune_model(vocab, pargs, len(naive_classes),
                                           pretrained_dir=SCGPT_PRETRAINED)
    if args.identity_check:
        # fp32 (no autocast): B=0 -> delta = 0 exactly, so any residual diff is
        # purely the attention-kernel path; threshold 1e-4 on the final cell_emb
        base = scgpt_cls.build_finetune_model(vocab, pargs, len(naive_classes),
                                              pretrained_dir=SCGPT_PRETRAINED)
        base.to(device).eval()
        with torch.no_grad():
            src = tok["genes"][:8].to(device); vals = tok["values"][:8].to(device)
            pad = tok["mask"][:8].to(device)
            out_base = base(src, vals, src_key_padding_mask=pad, CLS=False)
        wrap_lora(model)
        model.to(device).eval()
        with torch.no_grad():
            src = tok["genes"][:8].to(device); vals = tok["values"][:8].to(device)
            pad = tok["mask"][:8].to(device)
            out_lora = model(src, vals, src_key_padding_mask=pad, CLS=False)
        diff = (out_base["cell_emb"] - out_lora["cell_emb"]).abs().max().item()
        log(f"IDENTITY CHECK (fp32): B=0 LoRA vs base max|diff| = {diff:.2e}")
        if diff > 1e-4:
            log("IDENTITY CHECK FAILED")
            return 1
        log("IDENTITY CHECK PASS (B=0 init preserves the base model)")
        return 0

    wrap_lora(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    run_dir = args.out / "checkpoints" / f"lora_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    y_true_val = cc4_all[val_idx_a]
    present = sorted(set(y_true_val.tolist()) & set(LAYER_ORDER))
    opt_step = 0
    best_val_f1, best_val_f1_ep = -1.0, None
    best_val_rho, best_val_rho_ep = None, None
    traj_path = run_dir / "trajectory.jsonl"
    t_all = time.time()
    epochs = 1 if args.dry_run else EPOCHS

    # data loaders
    from torch.utils.data import Dataset, DataLoader, Subset
    ds = scgpt_cls.CellDataset(tok["genes"], tok["values"], labels)
    tr_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx_a.tolist())
    loader = DataLoader(tr_ds, batch_size=PER_DEVICE_BATCH, shuffle=True,
                        drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        opt.zero_grad(set_to_none=True)
        for step, (src, values, y) in enumerate(loader):
            src = src.to(device); values = values.to(device); y = y.to(device)
            mask = src.eq(pad_id)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(src, values, src_key_padding_mask=mask, CLS=True)
                loss = F.cross_entropy(out["cls_output"], y)
            (loss.float() / GRAD_ACCUM).backward()
            running.append(float(loss.detach().float()))
            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
                opt.step(); opt_step += 1
                if opt_step <= WARMUP_STEPS:
                    for g in opt.param_groups:
                        g["lr"] = LR * opt_step / max(WARMUP_STEPS, 1)
                else:
                    for g in opt.param_groups:
                        g["lr"] = LR
                opt.zero_grad(set_to_none=True)
        train_mean = float(np.mean(running[-min(200, len(running)):]))

        # val eval
        model.eval()
        vemb, vcls = [], []
        with torch.no_grad():
            for src, values, _y in val_loader:
                src = src.to(device); values = values.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model(src, values, src_key_padding_mask=src.eq(pad_id),
                                CLS=True)
                vemb.append(out["cell_emb"].float().cpu().numpy())
                vcls.append(out["cls_output"].float().cpu().numpy())
        vemb = np.concatenate(vemb, axis=0).astype(np.float32)
        vcls = np.concatenate(vcls, axis=0).astype(np.float32)
        vpred = np.array([naive_map.get(naive_classes[int(i)], "NON_LAYER")
                          for i in vcls.argmax(1)], dtype=str)
        val_f1 = float(f1_score(y_true_val, vpred, labels=present,
                                average="macro", zero_division=0))
        rh = rho_and_null(vemb, y_true_val)
        if val_f1 > best_val_f1:
            best_val_f1, best_val_f1_ep = val_f1, epoch
            torch.save(model.state_dict(), run_dir / "model_best_val.pt")
        if best_val_rho is None or rh["rho_normed"] > best_val_rho:
            best_val_rho, best_val_rho_ep = rh["rho_normed"], epoch
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:02d}.pt")
        ad_norm = 0.0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B_" in name:
                    ad_norm = max(ad_norm, float(p.abs().max()))
        rec = {"epoch": epoch, "train_loss_mean": train_mean, "val_f1": val_f1,
               "val_rho_normed": rh["rho_normed"], "val_z": rh["null_z"],
               "best_val_f1": best_val_f1, "best_val_f1_epoch": best_val_f1_ep,
               "max_lora_B_abs": ad_norm,
               "wall_s": round(time.time() - t_all, 1)}
        with traj_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"epoch {epoch}/{epochs} train_mean={train_mean:.4f} "
            f"val_f1={val_f1:.4f} val_rho={rh['rho_normed']:.4f} "
            f"val_z={rh['null_z']:.1f} ({time.time() - t_all:.0f}s)")

    torch.save(model.state_dict(), run_dir / "model.pt")

    # inline final test-donor evaluation (best-val + final-epoch)
    import anndata as ad2
    te = ad2.read_h5ad(TEST_H5AD, backed="r")
    y_test = np.asarray(te.obs["cell_class_4"].astype(str))
    keep_t = np.array([g in vocab for g in te.var_names])
    gids = np.array([vocab[g] for g in te.var_names[keep_t]], dtype=np.int64)
    xt = te[:, keep_t].X
    if hasattr(xt, "to_memory"):
        xt = xt.to_memory()
    tok_t = scgpt_cls.tokenize_chunked(xt, gids, vocab, pargs, max_len=4253)
    ev = {}
    for ctype, fname in [("best_val", "model_best_val.pt"),
                         ("final_epoch", "model_epoch15.pt")]:
        ck = run_dir / fname
        if not ck.is_file():
            log(f"SKIP {ctype}: {fname} missing")
            continue
        st = torch.load(ck, map_location="cpu", weights_only=True)
        m2 = scgpt_cls.build_finetune_model(vocab, pargs, len(naive_classes),
                                            pretrained_dir=SCGPT_PRETRAINED)
        wrap_lora(m2)
        m2.load_state_dict(st, strict=True)
        m2.to(device).eval()
        ev[ctype] = decode_and_eval(m2, tok_t["genes"], tok_t["values"],
                                    tok_t["mask"], device, y_test,
                                    naive_classes, naive_map)
        log(f"  test {ctype}: F1={ev[ctype]['macro_f1']['macro_f1']:.4f} "
            f"rho={ev[ctype]['rho_normed']:.4f} z={ev[ctype]['null_z']:.1f}")

    summary = {
        "arm": "LoRA naive CE (QKV r=8 alpha=16; head + adapters trained; "
               "base trunk frozen)", "seed": seed, "epochs": epochs,
        "n_cls": len(naive_classes), "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
        "val_split": {"n_train": int(len(train_idx)), "n_val": int(len(val_idx_a))},
        "best_val_f1": best_val_f1, "best_val_f1_epoch": best_val_f1_ep,
        "best_val_rho": best_val_rho, "best_val_rho_epoch": best_val_rho_ep,
        "test_donor_eval": ev,
        "wall_s": time.time() - t_all,
        "timestamp_utc": now_utc(),
    }
    (run_dir / "args.json").write_text(json.dumps(summary, indent=2),
                                       encoding="utf-8")
    (run_dir / "runs_status").write_text(f"DONE\n{now_utc()}\n", encoding="utf-8")
    log("saved " + str(run_dir))
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
