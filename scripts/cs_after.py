#!/usr/bin/env python
"""CS_after / CS_before computation on Geneformer embeddings (Methods §4.4 + §4.6).

Recomputes the Continuity Score on the [CLS] embedding so results are directly
comparable to the Windows baseline (Method A marker-gradient ≈ 0.6338,
Method B author-annotation ≈ 0.5477) and feed the Gradient Preservation Ratio
GPR = CS_after / CS_before.

Cell -> 5-layer assignment uses the §4.4 Method B mapping (author_cell_type ->
LSC/TAC/basal/wing/squamous), injected as a 'layer' column during tokenization
(the same mechanism caft_gf.py uses).

Default (no --checkpoint): the pretrained Geneformer-V2-104M base model ->
CS_before. With --checkpoint: a fine-tuned checkpoint -> CS_after, and GPR when
--cs_before is given.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corneal_fm.caft import (  # noqa: E402
    LAYER_ORDER,
    NON_LAYER,
    author_cell_type_to_layer,
)

_HF_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DEFAULT = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))


def _resolve_checkpoint_or_base(checkpoint):
    if checkpoint is None or checkpoint == "base":
        hits = glob.glob(_HF_SNAP_GLOB)
        if not hits:
            raise FileNotFoundError("Geneformer-V2-104M base weights not found in HF cache")
        return hits[0], "base"
    p = Path(checkpoint)
    if p.is_dir():
        cands = sorted(p.glob("checkpoint-*"))
        ckpt = str(cands[-1]) if cands else str(p)
    else:
        ckpt = str(p)
    return ckpt, f"checkpoint:{checkpoint}"


def tokenize_with_layer(src_h5ad: Path, out_dir: Path) -> str:
    """Tokenize src_h5ad, adding a 'layer' column derived from author_cell_type."""
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer

    print(f"[tokenize] loading {src_h5ad} ...", flush=True)
    a = ad.read_h5ad(str(src_h5ad))
    if a.raw is not None:
        a.X = a.raw.X.copy()
        a.raw = None
    a.obs = a.obs.reset_index(drop=True)
    var = a.var.copy()
    var = var.reset_index(drop=True)
    a.var = var
    a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    a.obs["layer"] = a.obs["author_cell_type"].map(author_cell_type_to_layer).astype("int64")
    subset_h5ad = out_dir / "src_layer.h5ad"
    a.write_h5ad(str(subset_h5ad), compression="gzip")
    ld = a.obs["layer"].tolist()
    layer_dist = {LAYER_ORDER[i]: ld.count(i) for i in range(5)} | {"nonEpi": ld.count(-1)}
    print(f"[tokenize] layer dist: {layer_dist}", flush=True)

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"cell_type": "cell_type", "disease": "disease",
                               "tissue": "tissue", "donor_id": "donor_id", "layer": "layer"},
        nproc=4, chunk_size=512, model_version="V2",
        use_h5ad_index=True, collapse_gene_ids=True,
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="cs_tk_"))
    shutil.copy2(str(subset_h5ad), str(tmp_dir / subset_h5ad.name))
    tk_dir = out_dir / "tk"
    tk_dir.mkdir(parents=True, exist_ok=True)
    tk.tokenize_data(data_directory=str(tmp_dir), output_directory=str(tk_dir),
                     output_prefix="cell", file_format="h5ad", target_sum=10000)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    ds_path = str(tk_dir / "cell.dataset")
    n = len(load_from_disk(ds_path))
    print(f"[tokenize] {n} cells -> {ds_path}", flush=True)
    return ds_path, layer_dist


def extract_cls_embeddings(model, dataset, pad_id=0, batch=256) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out, n = [], len(dataset)
    with torch.no_grad():
        for s in range(0, n, batch):
            b = dataset[s:s + batch]
            ids = torch.tensor(b["input_ids"], device=device)
            length = b.get("length")
            length = torch.as_tensor(length if length is not None
                                     else ids.ne(pad_id).sum(1).tolist(), device=device)
            attn = torch.arange(ids.shape[1], device=device)[None, :] < length[:, None]
            z = model(input_ids=ids, attention_mask=attn,
                      output_hidden_states=True).hidden_states[-1][:, 0]
            out.append(F.normalize(z, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def centroids_and_cs(emb, layers, n_layers=5):
    centroids = np.zeros((n_layers, emb.shape[1]), dtype=np.float32)
    present = []
    for lyr in range(n_layers):
        m = layers == lyr
        if m.any():
            c = emb[m].mean(0)
            norm = np.linalg.norm(c)
            centroids[lyr] = c / norm if norm > 0 else c
            present.append(lyr)
    sims, detail = [], {}
    for i in range(len(present) - 1):
        a, bb = present[i], present[i + 1]
        s = float(np.clip(np.dot(centroids[a], centroids[bb]), -1, 1))
        sims.append(s)
        detail[f"{LAYER_ORDER[a]}->{LAYER_ORDER[bb]}"] = round(s, 4)
    cs = float(np.mean(sims)) if sims else 0.0
    return centroids, present, cs, detail


def layer_laa(centroids, present):
    """LAA = fraction of adjacent pairs closer than the median non-adjacent pair."""
    if len(present) < 3:
        return float("nan")
    d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    adj = np.array([d[present[i], present[i + 1]] for i in range(len(present) - 1)])
    nonadj = np.array([d[present[i], present[j]]
                       for i in range(len(present)) for j in range(i + 2, len(present))])
    if nonadj.size == 0:
        return float("nan")
    return float((adj < np.median(nonadj)).mean())


def bootstrap_cs(emb, layers, n_layers=5, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    present = [l for l in range(n_layers) if (layers == l).any()]
    n = len(layers)
    css = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c, p, cs, _ = centroids_and_cs(emb[idx], layers[idx], n_layers)
        bp = [l for l in present if (layers[idx] == l).any()]
        css.append(centroids_and_cs(emb[idx], layers[idx], n_layers)[2] if bp else 0.0)
    css = np.array([x for x in css if x])
    return {"mean": float(css.mean()), "ci_low": float(np.percentile(css, 2.5)),
            "ci_high": float(np.percentile(css, 97.5))}


def main():
    ap = argparse.ArgumentParser(description="CS_before/CS_after on Geneformer embeddings (§4.4)")
    ap.add_argument("--data_file", default="corneal_20k_donor_split.h5ad",
                    help="source h5ad under CORNEAL_FM_DATA (needs author_cell_type)")
    ap.add_argument("--checkpoint", default=None,
                    help="fine-tuned checkpoint -> CS_after; omit for base model CS_before")
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--cs_before", type=float, default=None, help="for GPR")
    ap.add_argument("--workdir", default=None, help="dir for tokenized scratch")
    ap.add_argument("--out", default=None, help="output JSON")
    args = ap.parse_args()

    workdir = Path(args.workdir) if args.workdir else DATA_DEFAULT / "cs_work"
    workdir.mkdir(parents=True, exist_ok=True)

    from transformers import BertForSequenceClassification
    from datasets import load_from_disk

    ckpt, src_tag = _resolve_checkpoint_or_base(args.checkpoint)
    print(f"[load] model from {ckpt} ({src_tag})", flush=True)
    model = BertForSequenceClassification.from_pretrained(ckpt, num_labels=args.num_classes)

    ds_path, layer_dist = tokenize_with_layer(DATA_DEFAULT / args.data_file, workdir)
    dataset = load_from_disk(ds_path)

    print("[embed] extracting [CLS] embeddings ...", flush=True)
    t0 = time.time()
    emb = extract_cls_embeddings(model, dataset)
    layers = np.array(dataset["layer"])
    print(f"[embed] {emb.shape} in {time.time() - t0:.1f}s", flush=True)

    _, present, cs, detail = centroids_and_cs(emb, layers)
    laa = layer_laa(centroids_and_cs(emb, layers)[0], present)
    boot = bootstrap_cs(emb, layers, n_boot=args.n_boot)

    result = {
        "n_cells": int(len(layers)),
        "model": src_tag,
        "data_file": args.data_file,
        "embedding": "[CLS] (last hidden state)",
        "layers_present": [LAYER_ORDER[l] for l in present],
        "per_layer_counts": {LAYER_ORDER[l]: int((layers == l).sum()) for l in range(5)},
        "continuity_score": round(cs, 4),
        "pair_similarity": detail,
        "layer_adjacency_accuracy": round(laa, 4) if not np.isnan(laa) else None,
        "bootstrap_cs_95ci": boot,
        "layer_dist_at_tokenize": {k: int(v) for k, v in layer_dist.items()},
    }
    if args.cs_before is not None:
        result["gpr"] = {"cs_before": args.cs_before, "cs_after": round(cs, 4),
                         "gpr": round(cs / args.cs_before, 4),
                         "interpretation": "preserved/improved" if cs / args.cs_before >= 1 else "degraded"}

    out = args.out or str(workdir / ("cs_before.json" if args.checkpoint is None else "cs_after.json"))
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("=" * 60)
    print(f"CS ({src_tag}): {result['continuity_score']}")
    print(f"  pair similarity: {detail}")
    print(f"  LAA: {result['layer_adjacency_accuracy']}")
    print(f"  bootstrap 95% CI: {boot}")
    if "gpr" in result:
        print(f"GPR = {result['gpr']['gpr']} ({result['gpr']['interpretation']})")
    print(f"saved -> {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
