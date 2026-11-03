#!/usr/bin/env python
"""CORN advantage A+D — scGPT CORN (3 seeds) + LoRA (2 seeds) per-cell extraction.

Runs on p5070 (CPU). For the locked test donor Hu0216 (20158 cells), forwards
the final-epoch checkpoints and saves per-cell threshold logits / softmax
logits + CLS embeddings + labels + marker scores, so the downstream reduction
(corn_advantage_reduce.py) can compute:

  * Experiment A (monotonicity):
      - literal cumulative violation  P(class>=1) < P(class>=2)
      - decision-consistency violation (PRIMARY): model decision class vs
        the CORN-style threshold-count rule over its own cumulative
        probabilities (softmax arm only; CORN decision IS the count).
  * Experiment D (threshold interpretability): differentiation-axis positions
    where P(class>=k) crosses 0.5, with the locked van Zyl marker panel.

FROZEN protocol (PCS Stability, frozen before first run 2026-08-20):
  * test cells: run6/scgpt_vanzyl_test.h5ad (20158, donor Hu0216) — locked
  * CORN arm: checkpoint run6/checkpoints/ordinal_seed{42,123,456}/model.pt
    (K=3 -> 2 threshold logits; p1=P(>=1), p2=P(>=2))
  * LoRA arm: checkpoint run6/lora_v2/checkpoints/lora_seed{42,123}/model.pt
    (final epoch, 30-class softmax; matches the locked final-epoch rho 0.834)
  * decode threshold = 0.5 (ordinal_decode contract, unchanged)
  * LoRA 30-class -> 3-tier mapping = data-driven majority cell_class_4 per
    author class in the TRAIN h5ad (scgpt_eval_vanzyl.naive_mapping contract)
  * sanity gate: recompute L2-normed rho on the test donor vs the locked eval
    JSONs (GPU bf16 vs CPU fp32 tolerance 2e-3, recorded)
  * seeds: CORN 42/123/456; LoRA 42/123 (2-seed, as trained)

Outputs (per arm/seed): <out>/<arm>_seed<seed>.npz with logits, emb, y,
barcode, author (LoRA), scores (3-marker panel) — plus seed.json (gates).
Idempotent per seed via DONE markers.

Usage (p5070, CPU):
    PYTHONPATH=/home/summer/corneal_scgpt/repo/scGPT \
    /home/summer/corneal_scgpt/repo/.venv/bin/python \
        scripts/corn_advantage_scgpt.py --arm ordinal --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "repo" / "scGPT"))

import scgpt_cls  # noqa: E402  (p5070 scripts/scgpt_cls.py, locked)

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
TRAIN_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_train.h5ad"
CKPT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "checkpoints"
LORA_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "lora_v2" / "checkpoints"
EVAL_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval"
SCGPT_PRETRAINED = ROOT / "save" / "scgpt_pretrained"

LAYER_ORDER = ["LSC", "basal", "differentiated"]
TEST_DONOR = "sanes2022_Hu0216"
N_ORDINAL = 3
N_THRESHOLDS = 2
MAX_LEN = 4253  # PREREG §4 escalation (locked)
BATCH = 64
LORA_R = 8
LORA_ALPHA = 16
RHO_TOL = 2e-3  # CPU fp32 vs GPU bf16 locked value

SCORE_COLS = ["score_LSC", "score_PMC_basal", "score_TDC"]
DIRECT_MARKERS = ["ABCG2", "KRT14", "KRT12", "KRT3", "GJA1", "TP63", "SOX17", "COL17A1"]

# ─────────────────────────────────────────────────────────────────────────────
# LoRA attention wrapper (exact replica of the archived E4 implementation;
# needed to load the lora checkpoints whose attention modules carry adapters)
# ─────────────────────────────────────────────────────────────────────────────
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402, N812  (mirrors corneal_fm/caft.py style)


class _LoRAAttention(nn.Module):
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
        w = mha.in_proj_weight.detach()
        b = mha.in_proj_bias.detach()
        self.batch_first = True
        self.in_proj_weight = w
        self.in_proj_bias = b
        self._qkv_same_embed_dim = True
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.q.weight.data.copy_(w[:d])
        self.q.bias.data.copy_(b[:d])
        self.k.weight.data.copy_(w[d:2 * d])
        self.k.bias.data.copy_(b[d:2 * d])
        self.v.weight.data.copy_(w[2 * d:])
        self.v.bias.data.copy_(b[2 * d:])
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
        bsz, seq, dim = q.shape
        q = q.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        m = None
        if key_padding_mask is not None:
            m = key_padding_mask[:, None, None, :].expand(bsz, self.num_heads, seq, seq)
        if attn_mask is not None:
            m = attn_mask if m is None else (m + attn_mask)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal)
        out = out.transpose(1, 2).reshape(bsz, seq, dim)
        return self.out_proj(out), None


class _LoRALayer(nn.Module):
    """Exact replica of nn.TransformerEncoderLayer.forward (batch_first,
    norm_first) that calls the LoRA-wrapped self_attn explicitly, bypassing
    torch's C++ fast-path (which would silently use base weights)."""

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
        self.self_attn = _LoRAAttention(layer.self_attn, r=r, alpha=alpha)

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


def wrap_lora(model: nn.Module) -> None:
    """Replace every encoder-layer self_attn with the LoRA wrapper (E4 layout)."""
    model.transformer_encoder.enable_nested_tensor = False
    model.transformer_encoder.use_nested_tensor = False
    for i, layer in enumerate(model.transformer_encoder.layers):
        model.transformer_encoder.layers[i] = _LoRALayer(layer)


# ─────────────────────────────────────────────────────────────────────────────
# data + forward
# ─────────────────────────────────────────────────────────────────────────────


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def naive_majority_map() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Data-driven majority cell_class_4 per author class in the TRAIN h5ad
    (scgpt_eval_vanzyl.naive_mapping contract — LoRA decode rule)."""
    import anndata as ad

    a = ad.read_h5ad(TRAIN_H5AD, backed="r")
    author = a.obs["author_cell_type"].astype(str).to_numpy()
    cc4 = a.obs["cell_class_4"].astype(str).to_numpy()
    cont: dict[str, Counter] = {}
    for au, lab in zip(author, cc4):
        cont.setdefault(au, Counter())[lab] += 1
    mapping = {au: max(c, key=c.get) for au, c in cont.items()}
    return mapping, {au: dict(c) for au, c in cont.items()}


def load_test() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import anndata as ad

    a = ad.read_h5ad(TEST_H5AD, backed="r")
    y = np.asarray(a.obs["cell_class_4"].astype(str))
    barcode = np.asarray(a.obs_names)
    author = np.asarray(a.obs["author_cell_type"].astype(str))
    scores = np.stack(
        [np.asarray(a.obs[c], dtype=float) for c in SCORE_COLS], axis=1
    )
    genes = np.zeros((a.n_obs, len(DIRECT_MARKERS)), dtype=np.float32)
    var_set = set(a.var_names)
    for j, g in enumerate(DIRECT_MARKERS):
        if g in var_set:
            genes[:, j] = np.asarray(a[:, g].X).ravel()
    if not (a.obs["donor_id"].astype(str) == TEST_DONOR).all():
        raise RuntimeError("test h5ad is not exclusively the locked test donor")
    return a, y, barcode, author, scores, genes


def forward_model(model: nn.Module, tok: dict, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """CPU fp32 forward over the tokenized test cells -> (cell_emb, cls_logits)."""
    n = tok["genes"].shape[0]
    embs, clss = [], []
    with torch.no_grad():
        for s in range(0, n, BATCH):
            src = tok["genes"][s:s + BATCH]
            vals = tok["values"][s:s + BATCH]
            pad = tok["mask"][s:s + BATCH]
            out = model(src, vals, src_key_padding_mask=pad, CLS=True)
            embs.append(out["cell_emb"].float().numpy())
            clss.append(out["cls_output"].float().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32), \
        np.concatenate(clss, axis=0).astype(np.float32)


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def rho_and_gate(emb: np.ndarray, y_true: np.ndarray,
                 locked_rho: float | None) -> dict[str, Any]:
    rho_n = float(marker_gradient_correlation(l2_norm_rows(emb), y_true, LAYER_ORDER))
    return {
        "rho_normed_recomputed": rho_n,
        "rho_normed_locked": locked_rho,
        "abs_diff": None if locked_rho is None else abs(rho_n - locked_rho),
        "tol": RHO_TOL,
        "pass": locked_rho is None or abs(rho_n - locked_rho) <= RHO_TOL,
    }


def run_corn_seed(seed: int, out_dir: Path) -> int:
    ckpt = CKPT_DIR / f"ordinal_seed{seed}" / "model.pt"
    args = json.loads((CKPT_DIR / f"ordinal_seed{seed}" / "args.json").read_text())
    assert args["arm"] == "ordinal" and args["n_cls"] == N_THRESHOLDS
    locked = json.loads((EVAL_DIR / f"ordinal_seed{seed}.json").read_text())
    locked_rho = locked["rho_normed"]

    log(f"=== CORN scGPT seed {seed} ===")
    vocab, pargs = scgpt_cls.load_vocab(SCGPT_PRETRAINED)
    a, y, barcode, _author, scores, genes = load_test()
    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = scgpt_cls.tokenize_chunked(x_mat, gene_ids, vocab, pargs, max_len=MAX_LEN)

    model = scgpt_cls.build_finetune_model(vocab, pargs, N_THRESHOLDS,
                                           pretrained_dir=SCGPT_PRETRAINED)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    t0 = time.time()
    emb, logits = forward_model(model, tok, vocab[pargs["pad_token"]])
    log(f"forward done {time.time() - t0:.0f}s emb={emb.shape} logits={logits.shape}")

    gate = rho_and_gate(emb, y, locked_rho)
    log(f"rho recomputed {gate['rho_normed_recomputed']:.4f} "
        f"locked {locked_rho:.4f} diff={gate['abs_diff']:.2e} "
        f"{'PASS' if gate['pass'] else 'FAIL'}")
    np.savez_compressed(
        out_dir / f"ordinal_seed{seed}.npz",
        logits=logits, emb=emb, y=_str(y), barcode=_str(barcode),
        scores=scores.astype(np.float32), genes=genes,
    )
    (out_dir / f"ordinal_seed{seed}.json").write_text(
        json.dumps({
            "arm": "corn_scgpt", "seed": seed, "n_cells": int(len(y)),
            "threshold_logits": N_THRESHOLDS, "test_donor": TEST_DONOR,
            "checkpoint": str(ckpt), "gate": gate, "timestamp_utc": now_utc(),
        }, indent=2)
    )
    (out_dir / f"ordinal_seed{seed}.DONE").write_text(now_utc())
    return 0


def run_lora_seed(seed: int, out_dir: Path) -> int:
    ckpt = LORA_DIR / f"lora_seed{seed}" / "model.pt"
    args = json.loads((LORA_DIR / f"lora_seed{seed}" / "args.json").read_text())
    assert args["n_cls"] == 30
    locked_rho = args["test_donor_eval"]["final_epoch"]["rho_normed"]
    naive_map, cont = naive_majority_map()

    log(f"=== LoRA scGPT seed {seed} ===")
    vocab, pargs = scgpt_cls.load_vocab(SCGPT_PRETRAINED)
    a, y, barcode, author, scores, genes = load_test()
    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = scgpt_cls.tokenize_chunked(x_mat, gene_ids, vocab, pargs, max_len=MAX_LEN)

    naive_classes = args.get("naive_classes") or sorted({str(t) for t in naive_map})
    model = scgpt_cls.build_finetune_model(vocab, pargs, len(naive_classes),
                                           pretrained_dir=SCGPT_PRETRAINED)
    wrap_lora(model)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    t0 = time.time()
    emb, logits = forward_model(model, tok, vocab[pargs["pad_token"]])
    log(f"forward done {time.time() - t0:.0f}s emb={emb.shape} logits={logits.shape}")

    gate = rho_and_gate(emb, y, locked_rho)
    log(f"rho recomputed {gate['rho_normed_recomputed']:.4f} "
        f"locked {locked_rho:.4f} diff={gate['abs_diff']:.2e} "
        f"{'PASS' if gate['pass'] else 'FAIL'}")
    np.savez_compressed(
        out_dir / f"lora_seed{seed}.npz",
        logits=logits, emb=emb, y=_str(y), barcode=_str(barcode),
        author=_str(author), scores=scores.astype(np.float32), genes=genes,
    )
    (out_dir / f"lora_seed{seed}.json").write_text(
        json.dumps({
            "arm": "lora_scgpt", "seed": seed, "n_cells": int(len(y)),
            "n_classes": len(naive_classes), "test_donor": TEST_DONOR,
            "checkpoint": str(ckpt), "gate": gate,
            "naive_classes": naive_classes, "naive_map": naive_map,
            "naive_mapping_contingency": cont, "timestamp_utc": now_utc(),
        }, indent=2)
    )
    (out_dir / f"lora_seed{seed}.DONE").write_text(now_utc())
    return 0


def _str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype="<U64")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["ordinal", "lora"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "paper1_vanzyl" / "corn_advantage")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ordinal_seed{args.seed}" if args.arm == "ordinal" else f"lora_seed{args.seed}"
    if (out_dir / f"{tag}.DONE").exists():
        log(f"SKIP {tag} (DONE marker present)")
        return 0
    torch.set_num_threads(args.threads)
    if args.arm == "ordinal":
        return run_corn_seed(args.seed, out_dir)
    return run_lora_seed(args.seed, out_dir)


if __name__ == "__main__":
    sys.exit(main())
