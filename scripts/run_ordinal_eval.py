#!/usr/bin/env python
"""Tier-B evaluation for the Paper 1 D5 ordinal arms (runs AFTER ordinal training).

Evaluates each ordinal checkpoint (lambda x seed) on the donor-disjoint
external test (GSE153515). The ordinal checkpoint is a
BertForSequenceClassification with K-1=3 threshold logits (CORN); predictions
are recovered by counting thresholds with sigmoid >= 0.5 (the locked 4-class
scheme), and class probabilities are the CORN posterior P(y=k)=p_k-p_{k+1}
(used for ECE). Every Tier-B metric is computed by the UNMODIFIED
corneal_fm.evaluation functions (macro-F1 + donor bootstrap CI, LAA,
marker-gradient rho, ECE) plus a 13-layer marker-gradient rho profile
(mirroring scripts/measure_layerwise_rho.py, same metric function).

Pipeline per checkpoint:
    1. Load the cached tokenized external test cells (reused read-only from
       artifacts/paper1_tier_b_eval/external_tokenized/).
    2. Load the ordinal fine-tuned checkpoint.
    3. Extract CLS embeddings of ALL 13 hidden states + 3 threshold logits
       in batches (bf16 autocast).
    4. CORN-decode to {LSC, TAC, basal, differentiated}.
    5. evaluate_on_external_test() (static + leakage gates inside).
    6. Save results_{lambda}_{seed}.json, layerwise_rho_{arm}_s{seed}.json,
       summary.csv, layerwise_summary.json to artifacts/paper1_ordinal_eval/.

SANITY GATE (fail-closed): rho at hidden_states[-1] (layer 12) computed from
the layerwise pass must equal the marker_gradient_rho inside the Tier-B
results JSON within 1e-6; otherwise the script aborts.

Usage:
    /home/dq/venv-ds/bin/python3 scripts/run_ordinal_eval.py [--runs 0.0:42,...] [--skip-tokenize]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# --- GPU / environment settings (mirror the eval/training pipelines) ---------
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from corneal_fm.evaluation import (  # noqa: E402
    evaluate_on_external_test,
    marker_gradient_correlation,
)
from corneal_fm.ordinal import ordinal_class_probs, ordinal_decode  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (Paper 1 locked evaluation contract)
# ---------------------------------------------------------------------------

SAVE_DIR = PROJECT_ROOT / "save"
EXT_DATA = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
TRAIN_DATA = PROJECT_ROOT / "data" / "corneal_100k_donor_split.h5ad"
CONFIG_PATH = PROJECT_ROOT / "config" / "external_test_gse153515.yml"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "paper1_fair_split" / "manifest.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_ordinal_eval"
# Read-only reuse of the cached tokenized external dataset (built by
# run_tier_b_eval.py from the locked config + annotation).
TOKEN_DATASET = (
    PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval" / "external_tokenized" / "cell.dataset"
)

EVAL_CLASSES = ["LSC", "TAC", "basal", "differentiated"]
LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]

N_HIDDEN_STATES = 13  # [0]=embedding output, [1..12]=transformer layers
LAYER_LABELS = ["embedding_output"] + [f"transformer_layer_{k}" for k in range(1, 13)]
N_THRESHOLDS = 3  # K-1 for the 4-class ordinal axis

N_BOOT = 1000
BOOT_SEED = 42
BATCH_SIZE = 32

LAMBDAS = [0.0, 0.5]
SEEDS = [42, 123, 456]

SANITY_TOL = 1e-6


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Checkpoint discovery (mirrors run_tier_b_eval.discover_checkpoints)
# ---------------------------------------------------------------------------

def discover_checkpoints(runs: list[tuple[float, int]]) -> list[dict]:
    found = []
    for lam, seed in runs:
        pattern = f"ordinal_gf_lam{lam}_seed{seed}_*"
        save_dirs = sorted(SAVE_DIR.glob(pattern))
        if not save_dirs:
            print(f"{_ts()}  WARN: no save dir for lam={lam} seed={seed} "
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
            "lam": lam,
            "seed": seed,
            "save_dir": save_dir,
            "checkpoint_path": ckpt_dir,
            "step": step,
        })
    return found


# ---------------------------------------------------------------------------
# External test cells (cached tokenized dataset, read-only reuse)
# ---------------------------------------------------------------------------

def _load_external_cells() -> tuple[object, np.ndarray, object]:
    """Tokenized epithelial cells + harmonized labels + filtered adata.

    Mirrors measure_layerwise_rho._load_external_cells / run_tier_b_eval:
    NON_LAYER cells are excluded and the dataset/adata are aligned cell-wise.
    """
    import anndata as ad
    from datasets import load_from_disk

    if not TOKEN_DATASET.is_dir():
        raise FileNotFoundError(
            f"cached tokenized external dataset not found: {TOKEN_DATASET} "
            "(run scripts/run_tier_b_eval.py once to build it)"
        )
    dataset = load_from_disk(str(TOKEN_DATASET))
    adata = ad.read_h5ad(str(EXT_DATA))
    labels = adata.obs["cell_type_harmonized"].astype(str)
    mask = np.asarray(labels != "NON_LAYER")
    keep_idx = np.where(mask)[0]
    dataset = dataset.select(keep_idx)
    adata_epi = adata[mask].copy()
    labels_epi = np.asarray(labels[mask].to_numpy(), dtype=str)
    print(f"{_ts()}  Epithelial filter: {len(keep_idx)}/{labels.shape[0]} cells "
          f"(NON_LAYER excluded); labels {labels_epi.shape}")
    return dataset, labels_epi, adata_epi


def _collate_batch(batch: list[dict]):
    """Right-pad input_ids within a batch (identical to run_tier_b_eval)."""
    max_len = max(len(b["input_ids"]) for b in batch)
    padded = np.zeros((len(batch), max_len), dtype=np.int64)
    lengths = np.zeros(len(batch), dtype=np.int64)
    for i, b in enumerate(batch):
        ids = np.asarray(b["input_ids"], dtype=np.int64)
        padded[i, : len(ids)] = ids
        lengths[i] = int(b["length"])
    return torch.from_numpy(padded), torch.from_numpy(lengths)


# ---------------------------------------------------------------------------
# Inference: all hidden-state CLS + ordinal threshold logits
# ---------------------------------------------------------------------------

def run_ordinal_inference(
    model,
    tokenized_dataset,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """(13, n, 768) layer-wise CLS embeddings and (n, 3) threshold logits."""
    model.eval()
    loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_batch,
    )
    layer_cls: list[list[np.ndarray]] = [[] for _ in range(N_HIDDEN_STATES)]
    logits_list: list[np.ndarray] = []
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
            hidden = bert_out.hidden_states
            if len(hidden) != N_HIDDEN_STATES:
                raise RuntimeError(
                    f"expected {N_HIDDEN_STATES} hidden states, got {len(hidden)}"
                )
            for li, h in enumerate(hidden):
                layer_cls[li].append(h[:, 0].float().cpu().numpy())  # CLS at position 0
            logits = model.classifier(model.dropout(bert_out.pooler_output))
            logits_list.append(logits.float().cpu().numpy())
            if (i + 1) % 10 == 0 or i + 1 == n_batches:
                print(f"{_ts()}    inference batch {i + 1}/{n_batches}")
    cls_stack = np.stack([np.concatenate(part, axis=0) for part in layer_cls], axis=0)
    return cls_stack, np.concatenate(logits_list)


# ---------------------------------------------------------------------------
# CORN decode -> 4-class labels + probabilities
# ---------------------------------------------------------------------------

def decode_ordinal(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,3) threshold logits -> (n,) string labels + (n,4) class probs."""
    preds_ord, thr_probs = ordinal_decode(logits)
    predictions = np.asarray([EVAL_CLASSES[c] for c in preds_ord], dtype=object)
    probabilities = ordinal_class_probs(thr_probs)
    return predictions, probabilities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_runs(runs_arg: str | None) -> list[tuple[float, int]]:
    if not runs_arg:
        return [(lam, seed) for lam in LAMBDAS for seed in SEEDS]
    runs = []
    for token in runs_arg.split(","):
        lam_s, seed_s = token.strip().split(":")
        runs.append((float(lam_s), int(seed_s)))
    return runs


def _detect_device() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", default=None,
        help="Comma-separated LAM:SEED pairs (default: all 6 of "
             f"{LAMBDAS} x {SEEDS})",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = _parse_runs(args.runs)
    print(f"{_ts()}  Runs requested: {runs}")
    checkpoints = discover_checkpoints(runs)
    if not checkpoints:
        print(f"{_ts()}  ERROR: no ordinal checkpoints found for any requested run.")
        return 1

    device = _detect_device()
    print(f"{_ts()}  Device: {device}")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    dataset, labels, external_adata = _load_external_cells()
    print(f"{_ts()}  External eval cells: {len(dataset)}")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(MANIFEST_PATH) as f:
        split_manifest = json.load(f)

    import anndata as ad
    train_adata = ad.read_h5ad(str(TRAIN_DATA), backed="r")
    print(f"{_ts()}  Training data (leakage checks): {train_adata.n_obs} cells")

    summary_rows = []
    layerwise_summary: dict[str, list[float]] = {}
    gate_failed = False
    gate_results: dict[str, dict] = {}

    for ck in checkpoints:
        lam, seed = ck["lam"], ck["seed"]
        arm = "ordinal_naive" if lam == 0.0 else "ordinal_caft"
        ckpt_path = ck["checkpoint_path"]
        print(f"\n{_ts()}  === {arm} lam={lam} seed={seed} step={ck['step']} "
              f"({ckpt_path}) ===")

        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            str(ckpt_path), num_labels=N_THRESHOLDS, torch_dtype=torch_dtype,
        )
        model = model.to(device)
        layer_cls, logits = run_ordinal_inference(
            model, dataset, batch_size=args.batch_size, device=device,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        predictions, probabilities = decode_ordinal(logits)
        print(f"{_ts()}  ordinal class counts: "
              f"{np.unique(predictions, return_counts=True)[0].tolist()} "
              f"{np.unique(predictions, return_counts=True)[1].tolist()}")

        # -- layer-wise marker-gradient rho (all 13 hidden states) --------
        layer_rhos = [
            marker_gradient_correlation(layer_cls[i], labels, layer_order=LAYER_ORDER_4)
            for i in range(N_HIDDEN_STATES)
        ]
        print(f"{_ts()}  layerwise rho: L12={layer_rhos[12]:+.4f} "
              f"L6={layer_rhos[6]:+.4f}")

        # -- Tier-B evaluation (primary + secondary endpoints) ------------
        results = evaluate_on_external_test(
            train_adata=train_adata,
            external_adata=external_adata,
            embeddings=layer_cls[12],
            predictions=predictions,
            probabilities=probabilities,
            config=config,
            split_manifest=split_manifest,
            classes=EVAL_CLASSES,
            layer_order=LAYER_ORDER_4,
            n_boot=N_BOOT,
            seed=BOOT_SEED,
        )
        # Sanity gate: rho from the layerwise pass (layer 12) must match the
        # Tier-B marker-gradient rho (same metric, same cells).
        ref_rho = float(results["secondary_endpoints"]["marker_gradient_rank_correlation"])
        diff = abs(layer_rhos[12] - ref_rho)
        passed = bool(diff <= SANITY_TOL)
        gate_failed |= not passed
        print(f"{_ts()}  SANITY lam={lam} seed={seed}: rho_layer12={layer_rhos[12]:.8f} "
              f"ref={ref_rho:.8f} diff={diff:.2e} -> {'PASS' if passed else 'FAIL'}")
        gate_results[f"lam{lam}_seed{seed}"] = {
            "layer12_rho": layer_rhos[12], "reference_rho": ref_rho,
            "abs_diff": diff, "passed": passed,
        }

        out_json = OUT_DIR / f"results_{lam}_{seed}.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{_ts()}  Saved {out_json}")

        layerwise_out = OUT_DIR / f"layerwise_rho_{arm}_s{seed}.json"
        with open(layerwise_out, "w") as f:
            json.dump({
                "model": f"lam{lam}_seed{seed}",
                "arm": arm,
                "loss": "CORN ordinal",
                "lambda": lam,
                "seed": seed,
                "checkpoint_path": str(ckpt_path),
                "n_layers": N_HIDDEN_STATES,
                "layer_labels": LAYER_LABELS,
                "n_cells": int(len(dataset)),
                "layer_order": LAYER_ORDER_4,
                "l2_normalized": False,
                "dtype": "bfloat16" if device == "cuda" else "float32",
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "marker_gradient_rho": layer_rhos,
                "sanity_gate": gate_results[f"lam{lam}_seed{seed}"],
            }, f, indent=2)
        print(f"{_ts()}  Saved {layerwise_out}")

        pe = results["primary_endpoint"]
        se = results["secondary_endpoints"]
        summary_rows.append({
            "lambda": lam,
            "seed": seed,
            "checkpoint": ckpt_path.name,
            "macro_f1": pe["macro_f1"],
            "ci_lower": pe["ci_lower"],
            "ci_upper": pe["ci_upper"],
            "LAA": se["layer_adjacency_accuracy"],
            "marker_gradient_rho": se["marker_gradient_rank_correlation"],
            "rho_L6": layer_rhos[6],
            "rho_L12": layer_rhos[12],
            "ECE": se["expected_calibration_error"],
        })
        layerwise_summary[f"lam{lam}_seed{seed}"] = layer_rhos

    df = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{_ts()}  Saved {csv_path}")
    print(df.to_string(index=False))

    lw_summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_models": len(layerwise_summary),
        "n_layers": N_HIDDEN_STATES,
        "layer_labels": LAYER_LABELS,
        "layer_order": LAYER_ORDER_4,
        "sanity_tolerance": SANITY_TOL,
        "sanity_gate_all_passed": not gate_failed,
        "sanity_gate": gate_results,
        "models": {k: v for k, v in layerwise_summary.items()},
    }
    with open(OUT_DIR / "layerwise_summary.json", "w") as f:
        json.dump(lw_summary, f, indent=2)
    print(f"{_ts()}  Saved {OUT_DIR / 'layerwise_summary.json'}")

    if gate_failed:
        print(f"\n{_ts()}  SANITY GATE FAILED — refusing to report success.")
        for name, g in gate_results.items():
            if not g["passed"]:
                print(f"    {name}: rho_layer12={g['layer12_rho']:.8f} "
                      f"ref={g['reference_rho']:.8f} diff={g['abs_diff']:.3e}")
        return 2
    print(f"\n{_ts()}  SANITY GATE PASSED for all {len(gate_results)} models "
          f"(tol {SANITY_TOL:.0e}).")
    print(f"{_ts()}  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
