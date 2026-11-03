#!/usr/bin/env python
"""Quick CPU forward benchmark for GF (104M) and scGPT (25M) on p5070.

Benchmarks only the transformer forward (no tokenization) over a fixed
window of test cells, across a couple of (threads, batch) configs, and
prints seconds-per-batch + peak RSS so the sensitivity/UMAP driver can
pick a stable configuration.  Read-only: loads checkpoints + dataset,
forwards a few batches, saves nothing.

Usage (p5070, CPU):
    /home/summer/corneal_scgpt/venv/bin/python scripts/bench_cpu_forward.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "repo" / "scGPT"))

N_BATCHES = 4  # batches per config (keep small: we share the box)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def bench_gf(threads: int, batch: int) -> dict:
    from datasets import load_from_disk
    from transformers import BertForSequenceClassification

    torch.set_num_threads(threads)
    ds = load_from_disk(str(ROOT / "data" / "gf_tokenized_vanzyl.dataset"))
    idx = np.arange(0, batch * N_BATCHES)  # first rows (any cells: timing only)
    model = BertForSequenceClassification.from_pretrained(
        str(ROOT / "save" / "corn_advantage_gf_ckpt"
            / "ordinal_gf_lam0.0_seed42_20260805-171141" / "model" / "checkpoint-7473"),
        num_labels=3, torch_dtype=torch.float32,
    )
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            e = min(s + batch, len(idx))
            chunk = ds[idx[s:e].tolist()]
            ids_list = chunk["input_ids"]
            ml = max(len(x) for x in ids_list)
            padded = np.zeros((len(ids_list), ml), dtype=np.int64)
            for i, ids in enumerate(ids_list):
                padded[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
            ids_t = torch.from_numpy(padded)
            length = torch.from_numpy(np.asarray(chunk["length"], dtype=np.int64))
            attn = torch.arange(ml).unsqueeze(0) < length.unsqueeze(1)
            model.bert(ids_t, attention_mask=attn)
    wall = time.time() - t0
    return {
        "model": "gf", "threads": threads, "batch": batch,
        "n_cells": len(idx), "wall_s": round(wall, 1),
        "s_per_batch": round(wall / N_BATCHES, 1),
        "rss_gb": round(rss_gb(), 1),
    }


def bench_scgpt(threads: int, batch: int) -> dict:
    import scgpt_cls as sc

    torch.set_num_threads(threads)
    vocab, pargs = sc.load_vocab(pretrained_dir=str(ROOT / "pretrained" / "scgpt_pretrained"))
    ckpt = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "checkpoints" / "naive_seed42"
    args = json.loads((ckpt / "args.json").read_text())
    state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    model = sc.build_finetune_model(vocab, pargs, args["n_cls"],
                                    pretrained_dir=str(ROOT / "pretrained" / "scgpt_pretrained"))
    model.load_state_dict(state, strict=True)
    model.eval()
    # synthetic tokenized chunk (timing only)
    n = batch * N_BATCHES
    genes = torch.randint(0, len(vocab), (n, 512), dtype=torch.long)
    vals = torch.rand((n, 512), dtype=torch.float32)
    mask = torch.zeros((n, 512), dtype=torch.bool)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch):
            model(genes[s:s + batch], vals[s:s + batch],
                  src_key_padding_mask=mask[s:s + batch], CLS=True)
    wall = time.time() - t0
    return {
        "model": "scgpt", "threads": threads, "batch": batch,
        "n_cells": n, "wall_s": round(wall, 1),
        "s_per_batch": round(wall / N_BATCHES, 1),
        "rss_gb": round(rss_gb(), 1),
    }


def main() -> int:
    results = []
    for cfg in [(12, 16), (12, 24), (8, 16)]:
        t, b = cfg
        results.append(bench_gf(t, b))
        print(results[-1], flush=True)
    for cfg in [(12, 32), (12, 64)]:
        t, b = cfg
        results.append(bench_scgpt(t, b))
        print(results[-1], flush=True)
    print("ALL", flush=True)
    print(json.dumps(results, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
