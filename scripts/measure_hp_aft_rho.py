#!/usr/bin/env python
"""Layer-wise marker-gradient rho (rho) for HPA-FT checkpoints (Paper 1 main method).

Mirror of scripts/measure_layerwise_rho.py for the HPA-FT arms. The locked
rho script globs `caft_gf_lam*_seed*_*` save dirs only, so HPA-FT runs
(save/hp_aft_gf_<mode>_lam<λ>_seed<s>_*/) need this counterpart. It reuses the
locked measurement machinery READ-ONLY:
    - corneal_fm.evaluation.marker_gradient_correlation (unchanged)
    - measure_layerwise_rho.collect_layerwise_cls / _load_external_cells /
      _measure_rhos (imported, not modified)
and adds an HPA-FT checkpoint discovery + per-arm outputs, so the 13-layer
rho curves of the HPA-FT arms are directly comparable with the locked
caft/naive/frozen curves in artifacts/paper1_layerwise_rho/summary.json.

13 layers (Paper 1 diagnostic): hidden_states[0] = embedding output,
[1..12] = transformer layers, [12] == last_hidden_state. L1-L4 retain,
L5-L12 collapse for naive FT; HPA-FT aims to keep the whole curve up.

Outputs (artifacts/paper1_hp_aft/layerwise_rho/):
    layerwise_rho_<mode>_lam<λ>_s<seed>.json   summary.json (all arms x 13 layers)

Usage:
    python3 scripts/measure_hp_aft_rho.py \
        [--runs lwc:0.5:42,headfreeze:0.5:42] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json

# GPU / environment settings (mirror the eval/training pipelines).
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts/ imports

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Read-only reuse of the locked layer-wise rho machinery.
from measure_layerwise_rho import (  # noqa: E402
    LAYER_ORDER_4,
    N_HIDDEN_STATES,
    _load_external_cells,
    _measure_rhos,
    collect_layerwise_cls,
)

SAVE_DIR = PROJECT_ROOT / "save"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_hp_aft" / "layerwise_rho"
N_FT_CLASSES = 31
BATCH_SIZE = 32

# mode:lambda:seed triples; default = the queued grid (both arms x 3 seeds).
MODES = ["lwc", "headfreeze"]
LAMBDAS = [0.5]
SEEDS = [42, 123, 456]


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def discover_checkpoints(runs: list[tuple[str, float, int]]) -> list[dict]:
    """Latest checkpoint per (mode, lambda, seed); mirrors the locked script."""
    found = []
    for mode, lam, seed in runs:
        pattern = f"hp_aft_gf_{mode}_lam{lam}_seed{seed}_*"
        save_dirs = sorted(SAVE_DIR.glob(pattern))
        if not save_dirs:
            print(f"{_ts()}  WARN: no save dir for {mode} lam={lam} seed={seed} "
                  f"(pattern {pattern}); skipping.")
            continue
        save_dir = save_dirs[-1]
        ckpt_dirs = sorted((save_dir / "model").glob("checkpoint-*"))
        if not ckpt_dirs:
            print(f"{_ts()}  WARN: no checkpoints under {save_dir / 'model'}; skipping.")
            continue
        ckpt_dir = ckpt_dirs[-1]
        step_match = re.search(r"checkpoint-(\d+)", ckpt_dir.name)
        step = int(step_match.group(1)) if step_match else 0
        found.append({
            "mode": mode, "lam": lam, "seed": seed,
            "save_dir": save_dir, "checkpoint_path": ckpt_dir, "step": step,
        })
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", default=None,
        help="Comma-separated MODE:LAM:SEED triples "
             "(default: all {MODES} x {LAMBDAS} x {SEEDS})",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{_ts()}  Device: {device}")
    if device != "cuda":
        print(f"{_ts()}  WARN: CUDA unavailable; measurement skipped (needs GPU).")
        return 3
    torch_dtype = torch.bfloat16

    runs = (
        [(t.split(":")[0], float(t.split(":")[1]), int(t.split(":")[2]))
         for t in args.runs.split(",")]
        if args.runs
        else [(m, lam, s) for m in MODES for lam in LAMBDAS for s in SEEDS]
    )
    print(f"{_ts()}  HPA-FT runs requested: {runs}")
    checkpoints = discover_checkpoints(runs)
    if not checkpoints:
        print(f"{_ts()}  ERROR: no checkpoints found.")
        return 1

    dataset, labels = _load_external_cells()
    n_cells = len(dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rhos: dict[str, list[float]] = {}
    for ck in checkpoints:
        mode, lam, seed = ck["mode"], ck["lam"], ck["seed"]
        tag = f"{mode}_lam{lam}_s{seed}"
        print(f"\n{_ts()}  === {tag} step={ck['step']} ({ck['checkpoint_path']}) ===")

        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(
            str(ck["checkpoint_path"]), num_labels=N_FT_CLASSES,
            torch_dtype=torch_dtype,
        ).to(device)
        t0 = time.time()
        layer_cls = collect_layerwise_cls(
            model, dataset, device=device, batch_size=args.batch_size,
            l2_normalize=False, fp16_autocast=False,
        )
        del model
        torch.cuda.empty_cache()

        rhos = _measure_rhos(layer_cls, labels)
        if len(rhos) != N_HIDDEN_STATES or not all(np.isfinite(rhos)):
            print(f"{_ts()}  FAIL structural gate: rhos={rhos}")
            return 2
        print(f"{_ts()}  rho: " + " ".join(f"{x:+.4f}" for x in rhos))
        print(f"{_ts()}  L12 rho={rhos[-1]:.4f}  (elapsed {time.time()-t0:.0f}s)")
        summary_rhos[tag] = rhos

        out_json = OUT_DIR / f"layerwise_rho_{tag}.json"
        with open(out_json, "w") as f:
            json.dump({
                "model": tag, "mode": mode, "lambda": lam, "seed": seed,
                "checkpoint_path": str(ck["checkpoint_path"]),
                "n_layers": N_HIDDEN_STATES,
                "layer_labels": ["embedding_output"] + [f"transformer_layer_{k}" for k in range(1, 13)],
                "n_cells": int(n_cells),
                "layer_order": LAYER_ORDER_4,
                "l2_normalized": False,
                "dtype": "bfloat16",
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "marker_gradient_rho": rhos,
                "structural_gate": "PASS",
            }, f, indent=2)
        print(f"{_ts()}  Saved {out_json}")

    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_models": len(summary_rhos),
        "n_layers": N_HIDDEN_STATES,
        "layer_labels": ["embedding_output"] + [f"transformer_layer_{k}" for k in range(1, 13)],
        "layer_order": LAYER_ORDER_4,
        "n_cells": int(n_cells),
        "reference": "artifacts/paper1_layerwise_rho/summary.json (locked caft/naive/frozen)",
        "models": summary_rhos,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{_ts()}  Saved {OUT_DIR / 'summary.json'}")
    print(f"{_ts()}  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
