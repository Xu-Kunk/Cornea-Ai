#!/usr/bin/env python
"""Isolated per-batch timing on REAL tokenized scGPT data (diagnostic)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("<home>/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "repo" / "scGPT"))

import scgpt_cls as sc  # noqa: E402

torch.set_num_threads(12)
TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "checkpoints" / "naive_seed42"

vocab, pargs = sc.load_vocab(pretrained_dir=ROOT / "pretrained" / "scgpt_pretrained")
args = json.loads((CKPT / "args.json").read_text())
import anndata as ad

a = ad.read_h5ad(TEST_H5AD, backed="r")
keep = np.array([g in vocab for g in a.var_names])
gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
x_mat = a[:, keep].X
if hasattr(x_mat, "to_memory"):
    x_mat = x_mat.to_memory()
tok = sc.tokenize_chunked(x_mat, gene_ids, vocab, pargs, max_len=args["training"]["max_len"])
padded = tok["mask"].sum(dim=1).numpy()
order = np.argsort(-padded, kind="stable")
genes, values, mask = tok["genes"][order], tok["values"][order], tok["mask"][order]

model = sc.build_finetune_model(vocab, pargs, args["n_cls"],
                                pretrained_dir=ROOT / "pretrained" / "scgpt_pretrained")
state = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
enc = model.transformer_encoder
enc.enable_nested_tensor = False
enc.use_nested_tensor = False

for b in (1, 26, 100, 200, 300):
    s = (b - 1) * 64
    src = genes[s:s + 64]
    vl = values[s:s + 64]
    m = mask[s:s + 64]
    print(f"batch {b}: seq_max={src.shape[1]} real_len_max={(~m).sum(1).max().item()}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(src, vl, src_key_padding_mask=m, CLS=True)
    print(f"   time {time.time() - t0:.1f}s", flush=True)
print("DONE", flush=True)
