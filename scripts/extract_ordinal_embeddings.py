#!/usr/bin/env python
"""Extract layer-12 CLS per-cell embeddings for the ordinal (CORN) checkpoints
on the GSE153515 external test — CPU-free, GPU inference.

The Tier-B ordinal eval (run_ordinal_eval.py) saves only logits; the
biological-validation section (CNS-anchor marker-program monotonicity along
the differentiation axis) requires per-cell hidden-state-12 embeddings for the
ordinal arms. This script reuses run_ordinal_eval's locked loading/collate/
inference code and persists layer-12 CLS embeddings + labels per checkpoint as
artifacts/paper1_bio_validation/ordinal_emb_lam<lam>_seed<seed>.npz.

Usage:
    /home/dq/venv-ds/bin/python3 scripts/extract_ordinal_embeddings.py [--runs 0.0:42,0.5:42,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from run_ordinal_eval import (  # noqa: E402
    BATCH_SIZE,
    LAMBDAS,
    SEEDS,
    _collate_batch,
    _detect_device,
    _load_external_cells,
    discover_checkpoints,
)


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_bio_validation"


def extract_layer12(
    model,
    tokenized_dataset,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
) -> np.ndarray:
    """Per-cell layer-12 CLS embeddings (n, 768), CLS at position 0."""
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_batch,
    )
    parts: list[np.ndarray] = []
    n_batches = len(loader)
    with torch.no_grad():
        for i, (input_ids, length) in enumerate(loader):
            input_ids = input_ids.to(device)
            length = length.to(device)
            attn_mask = (
                torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                < length.unsqueeze(1)
            )
            bert_out = model.bert(
                input_ids, attention_mask=attn_mask, output_hidden_states=True
            )
            h12 = bert_out.hidden_states[12]
            parts.append(h12[:, 0].float().cpu().numpy())  # CLS
            if (i + 1) % 20 == 0 or i + 1 == n_batches:
                print(f"{_ts()}    batch {i + 1}/{n_batches}")
    return np.concatenate(parts, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=None,
                        help="Comma-separated LAM:SEED pairs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for token in (args.runs or "").split(","):
        if token.strip():
            lam_s, seed_s = token.strip().split(":")
            runs.append((float(lam_s), int(seed_s)))
    if not runs:
        runs = [(lam, seed) for lam in LAMBDAS for seed in SEEDS]

    checkpoints = discover_checkpoints(runs)
    if not checkpoints:
        print(f"{_ts()} ERROR: no ordinal checkpoints found for {runs}")
        return 1

    device = _detect_device()
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"{_ts()} Device: {device}; runs: {runs}")

    dataset, labels, external_adata = _load_external_cells()
    print(f"{_ts()} External eval cells: {len(dataset)}; labels {labels.shape}")
    y_true_str = np.asarray(labels, dtype=str)

    for ck in checkpoints:
        lam, seed = ck["lam"], ck["seed"]
        out_path = OUT_DIR / f"ordinal_emb_lam{lam}_seed{seed}.npz"
        if out_path.exists():
            print(f"{_ts()} SKIP (exists): {out_path.name}")
            continue
        from transformers import BertForSequenceClassification

        print(f"{_ts()} === ordinal lam={lam} seed={seed} "
              f"({ck['checkpoint_path']}) ===")
        model = BertForSequenceClassification.from_pretrained(
            str(ck["checkpoint_path"]), num_labels=3,
            torch_dtype=torch_dtype,
        )
        model = model.to(device)
        emb = extract_layer12(model, dataset, batch_size=args.batch_size, device=device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        np.savez(out_path, emb=emb, labels=y_true_str)
        meta = {
            "arm": "ordinal",
            "lambda": lam,
            "seed": seed,
            "checkpoint": str(ck["checkpoint_path"]),
            "n_cells": int(emb.shape[0]),
            "emb_dim": int(emb.shape[1]),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        with open(out_path.with_suffix(".npz.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{_ts()} saved {out_path.name} ({emb.shape})")
    print(f"{_ts()} DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
