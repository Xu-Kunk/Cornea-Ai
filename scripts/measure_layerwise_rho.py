#!/usr/bin/env python
"""Layer-wise marker-gradient rho (rho) measurement across Geneformer-V2-104M.

Diagnostic for the Paper 1 observation that fine-tuning collapses the
marker-differentiation gradient of the FINAL layer CLS embedding
(pretrained rho 0.806 -> naive fine-tune ~ -0.042). This script measures
rho at EVERY hidden state (index 0 = embedding output, 1..12 = transformer
layers, 12 == last_hidden_state) to localise the collapse.

Models measured (7 total):
  - frozen Geneformer-V2-104M (HF snapshot, L2-normalised CLS, fp16 autocast)
  - 6 CAFT fine-tuned checkpoints: lambda in {0.0 (naive), 0.5 (CAFT)} x
    seeds {42, 123, 456} (bf16, raw CLS)

Inference + label construction mirror scripts/run_tier_b_eval.py /
scripts/run_tier_b_baselines.py exactly; the metric reuses
corneal_fm.evaluation.marker_gradient_correlation UNMODIFIED on each layer.

SANITY GATE (fail-closed): rho at hidden_states[-1] (layer 12) must equal the
pre-existing artifacts/paper1_tier_b_eval/*.json value within 1e-6 for every
model; otherwise the script aborts and reports the discrepancy.

Outputs (artifacts/paper1_layerwise_rho/):
  layerwise_rho_naive_s{seed}.json   layerwise_rho_caft_s{seed}.json
  layerwise_rho_frozen_gf.json       summary.json (7 models x 13 layers)

Usage:
    <venv>/bin/python3 scripts/measure_layerwise_rho.py \
        [--runs 0.0:42,0.5:42] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
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
import torch  # noqa: E402
import torch.nn.functional as functional  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Local project import (read-only reuse; the function is NOT modified).
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (Paper 1 locked evaluation contract)
# ---------------------------------------------------------------------------

SAVE_DIR = PROJECT_ROOT / "save"
EXT_DATA = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_layerwise_rho"
REF_JSON_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
TOKEN_DATASET = REF_JSON_DIR / "external_tokenized" / "cell.dataset"

# Geneformer-V2-104M frozen weights (HF cache snapshot, as in baselines).
PRETRAINED_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--ctheodoris--Geneformer"
    / "snapshots"
    / "04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
    / "Geneformer-V2-104M"
)

LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]
EVAL_CLASSES = ["LSC", "TAC", "basal", "differentiated"]

# hidden_states: [0] = embedding output, [1..12] = transformer layers,
# [12] == last_hidden_state (final layer).
N_HIDDEN_STATES = 13
LAYER_LABELS = ["embedding_output"] + [f"transformer_layer_{k}" for k in range(1, 13)]

LAMBDAS = [0.0, 0.5]
SEEDS = [42, 123, 456]
N_FT_CLASSES = 31  # 31 author_cell_type classes (num_labels for the head)
BATCH_SIZE = 32

SANITY_TOL = 1e-6


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Checkpoint discovery (identical logic to run_tier_b_eval.py)
# ---------------------------------------------------------------------------

def discover_checkpoints(runs: list[tuple[float, int]]) -> list[dict]:
    """Latest checkpoint per (lambda, seed); mirrors run_tier_b_eval."""
    found = []
    for lam, seed in runs:
        pattern = f"caft_gf_lam{lam}_seed{seed}_*"
        save_dirs = sorted(SAVE_DIR.glob(pattern))
        if not save_dirs:
            print(f"{_ts()}  WARN: no save dir for lam={lam} seed={seed} "
                  f"(pattern {pattern}); skipping.")
            continue
        save_dir = save_dirs[-1]  # latest timestamp (lexicographic sort)
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
# External test data (mirror run_tier_b_eval.tokenize_external, cached input)
# ---------------------------------------------------------------------------

def _load_external_cells() -> tuple[object, np.ndarray]:
    """Tokenized epithelial cells (14,775) + harmonized layer labels in order."""
    import anndata as ad
    from datasets import load_from_disk

    dataset = load_from_disk(str(TOKEN_DATASET))
    adata = ad.read_h5ad(EXT_DATA)
    labels = adata.obs["cell_type_harmonized"].astype(str)
    mask = np.asarray(labels != "NON_LAYER")
    keep_idx = np.where(mask)[0]
    dataset = dataset.select(keep_idx)
    labels_epi = np.asarray(labels[mask].to_numpy(), dtype=str)
    print(f"{_ts()}  Epithelial filter: {len(keep_idx)}/{labels.shape[0]} cells "
          f"(NON_LAYER excluded); labels {labels_epi.shape}")
    return dataset, labels_epi


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
# Layer-wise CLS collection
# ---------------------------------------------------------------------------

def collect_layerwise_cls(
    model,
    tokenized_dataset,
    *,
    device: str,
    batch_size: int = BATCH_SIZE,
    l2_normalize: bool,
    fp16_autocast: bool,
) -> np.ndarray:
    """CLS (position 0) of every hidden state for all cells.

    Returns (13, n_cells, 768) float32 numpy. For frozen Geneformer the CLS
    is L2-normalised (arm protocol in run_tier_b_baselines); for fine-tuned
    checkpoints it is the raw CLS (arm protocol in run_tier_b_eval).
    """
    model.eval()
    loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_batch,
    )
    layer_cls: list[list[np.ndarray]] = [[] for _ in range(N_HIDDEN_STATES)]
    n_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for i, (input_ids, length) in enumerate(loader):
            input_ids = input_ids.to(device)
            length = length.to(device)
            attn_mask = (
                torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                < length.unsqueeze(1)
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=fp16_autocast and device == "cuda",
            ):
                if hasattr(model, "bert"):
                    bert_out = model.bert(
                        input_ids, attention_mask=attn_mask,
                        output_hidden_states=True,
                    )
                else:
                    bert_out = model(
                        input_ids=input_ids, attention_mask=attn_mask,
                        output_hidden_states=True,
                    )
            hidden = bert_out.hidden_states
            if len(hidden) != N_HIDDEN_STATES:
                raise RuntimeError(
                    f"expected {N_HIDDEN_STATES} hidden states, got {len(hidden)}"
                )
            for li, h in enumerate(hidden):
                cls = h[:, 0].float()  # CLS at position 0
                if l2_normalize:
                    cls = functional.normalize(cls, dim=-1)
                layer_cls[li].append(cls.cpu().numpy())
            if (i + 1) % 10 == 0 or i + 1 == n_batches:
                print(f"{_ts()}    batch {i + 1}/{n_batches} "
                      f"({time.time() - t0:.0f}s)")
    return np.stack(
        [np.concatenate(part, axis=0) for part in layer_cls], axis=0
    )


# ---------------------------------------------------------------------------
# Sanity gate + outputs
# ---------------------------------------------------------------------------

def _reference_rho(lam: float | None, seed: int | None) -> float:
    """Existing Tier-B rho for the model's LAST hidden state (layer 12)."""
    if lam is None:  # frozen arm
        ref_path = REF_JSON_DIR / "results_frozen_gf.json"
    else:
        ref_path = REF_JSON_DIR / f"results_{lam}_{seed}.json"
    with open(ref_path) as f:
        ref = json.load(f)
    return float(ref["secondary_endpoints"]["marker_gradient_rank_correlation"])


def _measure_rhos(layer_cls: np.ndarray, labels: np.ndarray) -> list[float]:
    return [
        marker_gradient_correlation(
            layer_cls[i], labels, layer_order=LAYER_ORDER_4
        )
        for i in range(N_HIDDEN_STATES)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", default=None,
        help="Comma-separated LAM:SEED pairs (default: all 6 of "
             f"{LAMBDAS} x {SEEDS}); frozen GF always measured",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--skip-frozen", action="store_true",
        help="Skip re-measuring the frozen GF arm (already in "
             "layerwise_rho_frozen_gf.json)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{_ts()}  Device: {device}")
    if device != "cuda":
        print(f"{_ts()}  WARN: CUDA unavailable; frozen GF arm skipped "
              f"(needs GPU, as in run_tier_b_baselines).")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    dataset, labels = _load_external_cells()
    n_cells = len(dataset)

    runs = (
        [(float(t.split(":")[0]), int(t.split(":")[1])) for t in args.runs.split(",")]
        if args.runs
        else [(lam, seed) for lam in LAMBDAS for seed in SEEDS]
    )
    print(f"{_ts()}  Fine-tuned runs requested: {runs}")
    checkpoints = discover_checkpoints(runs)
    if not checkpoints:
        print(f"{_ts()}  ERROR: no checkpoints found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rhos: dict[str, list[float]] = {}
    gate_results: dict[str, dict] = {}
    gate_failed = False

    # --- Frozen Geneformer arm -------------------------------------------
    if device == "cuda" and not args.skip_frozen:
        print(f"\n{_ts()}  === frozen_gf (Geneformer-V2-104M) ===")
        from transformers import BertModel

        model = BertModel.from_pretrained(str(PRETRAINED_DIR)).to(device)
        layer_cls = collect_layerwise_cls(
            model, dataset, device=device, batch_size=args.batch_size,
            l2_normalize=True, fp16_autocast=True,
        )
        del model
        torch.cuda.empty_cache()

        rhos = _measure_rhos(layer_cls, labels)
        ref = _reference_rho(None, None)
        diff = abs(rhos[-1] - ref)
        passed = diff <= SANITY_TOL
        gate_failed |= not passed
        print(f"{_ts()}  SANITY frozen_gf: rho[-1]={rhos[-1]:.8f} "
              f"ref={ref:.8f} diff={diff:.2e} -> {'PASS' if passed else 'FAIL'}")
        gate_results["frozen_gf"] = {
            "layer12_rho": rhos[-1], "reference_rho": ref,
            "abs_diff": diff, "passed": bool(passed),
        }
        summary_rhos["frozen_gf"] = rhos

        out_json = OUT_DIR / "layerwise_rho_frozen_gf.json"
        with open(out_json, "w") as f:
            json.dump({
                "model": "frozen_gf",
                "arm": "frozen",
                "checkpoint_path": str(PRETRAINED_DIR),
                "n_layers": N_HIDDEN_STATES,
                "layer_labels": LAYER_LABELS,
                "n_cells": int(n_cells),
                "layer_order": LAYER_ORDER_4,
                "l2_normalized": True,
                "dtype": "fp16 autocast",
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "marker_gradient_rho": rhos,
                "sanity_gate": gate_results["frozen_gf"],
            }, f, indent=2)
        print(f"{_ts()}  Saved {out_json}")

    # --- Fine-tuned CAFT checkpoints -------------------------------------
    for ck in checkpoints:
        lam, seed = ck["lam"], ck["seed"]
        arm = "naive" if lam == 0.0 else "caft"
        ckpt_path = ck["checkpoint_path"]
        print(f"\n{_ts()}  === {arm} lam={lam} seed={seed} step={ck['step']} "
              f"({ckpt_path}) ===")

        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(
            str(ckpt_path), num_labels=N_FT_CLASSES, torch_dtype=torch_dtype,
        )
        model = model.to(device)
        layer_cls = collect_layerwise_cls(
            model, dataset, device=device, batch_size=args.batch_size,
            l2_normalize=False, fp16_autocast=False,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        rhos = _measure_rhos(layer_cls, labels)
        ref = _reference_rho(lam, seed)
        diff = abs(rhos[-1] - ref)
        passed = diff <= SANITY_TOL
        gate_failed |= not passed
        print(f"{_ts()}  SANITY lam={lam} seed={seed}: rho[-1]={rhos[-1]:.8f} "
              f"ref={ref:.8f} diff={diff:.2e} -> {'PASS' if passed else 'FAIL'}")
        gate_results[f"lam{lam}_seed{seed}"] = {
            "layer12_rho": rhos[-1], "reference_rho": ref,
            "abs_diff": diff, "passed": bool(passed),
        }
        summary_rhos[f"lam{lam}_seed{seed}"] = rhos

        out_json = OUT_DIR / f"layerwise_rho_{arm}_s{seed}.json"
        with open(out_json, "w") as f:
            json.dump({
                "model": f"lam{lam}_seed{seed}",
                "arm": arm,
                "lambda": lam,
                "seed": seed,
                "checkpoint_path": str(ckpt_path),
                "n_layers": N_HIDDEN_STATES,
                "layer_labels": LAYER_LABELS,
                "n_cells": int(n_cells),
                "layer_order": LAYER_ORDER_4,
                "l2_normalized": False,
                "dtype": "bfloat16" if device == "cuda" else "float32",
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "marker_gradient_rho": rhos,
                "sanity_gate": gate_results[f"lam{lam}_seed{seed}"],
            }, f, indent=2)
        print(f"{_ts()}  Saved {out_json}")

    # --- Summary + fail-closed gate --------------------------------------
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_models": len(summary_rhos),
        "n_layers": N_HIDDEN_STATES,
        "layer_labels": LAYER_LABELS,
        "layer_order": LAYER_ORDER_4,
        "n_cells": int(n_cells),
        "sanity_tolerance": SANITY_TOL,
        "sanity_gate_all_passed": not gate_failed,
        "sanity_gate": gate_results,
        "models": {k: v for k, v in summary_rhos.items()},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{_ts()}  Saved {OUT_DIR / 'summary.json'}")

    if gate_failed:
        print(f"\n{_ts()}  SANITY GATE FAILED — refusing to report success.")
        for name, g in gate_results.items():
            if not g["passed"]:
                print(f"    {name}: rho[-1]={g['layer12_rho']:.8f} "
                      f"ref={g['reference_rho']:.8f} diff={g['abs_diff']:.3e}")
        return 2

    print(f"\n{_ts()}  SANITY GATE PASSED for all {len(gate_results)} models "
          f"(tol {SANITY_TOL:.0e}).")
    print(f"{_ts()}  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
