#!/usr/bin/env python
"""Paper 1 — per-cell ordinal-head (CORN) logits on the external test set.

Extracts, for one ordinal checkpoint (arm x seed), the per-cell 3 threshold
logits of the K-1=3 CORN binary heads on the donor-disjoint external test
(GSE153515), plus the CORN-decode class probabilities (P(y=k) = p_k - p_{k+1})
and the cell labels (cell_type_harmonized / layer / donor_id).

The loading / evaluation path MIRRORS scripts/run_ordinal_eval.py exactly
(read-only reuse of the cached tokenized external dataset at
artifacts/paper1_tier_b_eval/external_tokenized/cell.dataset; NON_LAYER cells
excluded; 4-class locked scheme LSC/TAC/basal/differentiated; right-padded
batches; forward pass = model(input_ids, attention_mask).logits, i.e. the same
classifier(dropout(pooler_output)) head used by the eval script). CORN decode
reuses corneal_fm/ordinal.py (ordinal_decode / ordinal_class_probs) — no
re-implementation.

Per-cell monotonic consistency (the D5 claim that the head learns ORDERED
boundaries): a cell is consistent when its cumulative threshold probabilities
satisfy P(y>=1) >= P(y>=2) >= P(y>=3). The summary JSON reports the
consistency proportion overall and per differentiation stage, plus a seeded
within-cell permutation null (random assignment of the 3 logits to the 3
threshold slots; expected consistency under an UNORDERED head) so the figure
can contrast observed vs chance.

Outputs (artifacts/paper1_corn_boundary/):
  percell_<arm>_<seed>.npz     logits (n,3) | thr_probs (n,3) | probs (n,4)
                               preds (n,) | labels (n,) | layer (n,) | donor_id (n,)
  percell_<arm>_<seed>.json    summary (counts, monotonic consistency, null)

Usage:
    <venv>/bin/python3 scripts/extract_ordinal_logits.py \
        [--arm lam0.0|lam0.5] [--seed 42|123|456] [--checkpoint PATH]
Default: --arm lam0.0 --seed 42 and the terminal (final, epoch-3)
checkpoint of that arm/seed (checkpoint-7473 for all 6 runs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
from datasets import load_from_disk  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import BertForSequenceClassification  # noqa: E402

from corneal_fm.ordinal import ordinal_class_probs, ordinal_decode  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (Paper 1 locked evaluation contract — same as run_ordinal_eval.py)
# ---------------------------------------------------------------------------

SAVE_DIR = PROJECT_ROOT / "save"
EXT_DATA = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
# Read-only reuse of the cached tokenized external dataset (built by
# run_tier_b_eval.py from the locked config + annotation).
TOKEN_DATASET = (
    PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval" / "external_tokenized" / "cell.dataset"
)
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_corn_boundary"

EVAL_CLASSES = ["LSC", "TAC", "basal", "differentiated"]
N_THRESHOLDS = 3  # K-1 for the 4-class ordinal axis
BATCH_SIZE = 32

N_PERM_DRAWS = 1000
PERM_SEED = 42


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


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


def resolve_checkpoint(arm: str, seed: int, checkpoint_arg: str | None) -> Path:
    """Resolve the checkpoint directory: explicit path or terminal (last)."""
    if checkpoint_arg:
        ckpt = Path(checkpoint_arg).expanduser().resolve()
        if not ckpt.is_dir():
            raise FileNotFoundError(f"checkpoint dir not found: {ckpt}")
        return ckpt
    lam = float(arm.split("lam")[-1])
    pattern = f"ordinal_gf_lam{lam}_seed{seed}_*"
    save_dirs = sorted(SAVE_DIR.glob(pattern))
    if not save_dirs:
        raise FileNotFoundError(
            f"no save dir for arm={arm} seed={seed} (pattern {pattern})"
        )
    ckpt_dirs = sorted((save_dirs[-1] / "model").glob("checkpoint-*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"no checkpoints under {save_dirs[-1] / 'model'}")
    return ckpt_dirs[-1]


def _checkpoint_step(ckpt: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", ckpt.name)
    return int(m.group(1)) if m else 0


def _checkpoint_epoch(ckpt: Path) -> float | None:
    """Final epoch from trainer_state.json (provenance)."""
    try:
        state = json.loads((ckpt / "trainer_state.json").read_text(encoding="utf-8"))
        epochs = [
            float(e["epoch"])
            for e in state.get("log_history", [])
            if isinstance(e.get("epoch"), (int, float))
        ]
        return max(epochs) if epochs else float(state.get("epoch", 0.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# External test cells (cached tokenized dataset, read-only reuse)
# ---------------------------------------------------------------------------

def _load_external_cells():
    """Tokenized epithelial cells + harmonized labels / layer / donor_id.

    Mirrors run_ordinal_eval._load_external_cells: NON_LAYER cells are excluded
    and the dataset/adata are aligned cell-wise (same index order).
    """
    import anndata as ad

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
    labels_epi = np.asarray(labels[mask].to_numpy(), dtype=str)
    layer_epi = np.asarray(adata.obs["layer"].to_numpy(), dtype=np.int64)[mask]
    donor_epi = np.asarray(adata.obs["donor_id"].astype(str).to_numpy(), dtype=str)[mask]
    print(f"{_ts()}  Epithelial filter: {len(keep_idx)}/{labels.shape[0]} cells "
          f"(NON_LAYER excluded)")
    return dataset, labels_epi, layer_epi, donor_epi


def _collate_batch(batch: list[dict]):
    """Right-pad input_ids within a batch (identical to run_ordinal_eval)."""
    max_len = max(len(b["input_ids"]) for b in batch)
    padded = np.zeros((len(batch), max_len), dtype=np.int64)
    lengths = np.zeros(len(batch), dtype=np.int64)
    for i, b in enumerate(batch):
        ids = np.asarray(b["input_ids"], dtype=np.int64)
        padded[i, : len(ids)] = ids
        lengths[i] = int(b["length"])
    return torch.from_numpy(padded), torch.from_numpy(lengths)


# ---------------------------------------------------------------------------
# Inference: 3 threshold logits per cell
# ---------------------------------------------------------------------------

def run_logit_inference(
    model, tokenized_dataset, batch_size: int = BATCH_SIZE, device: str = "cpu",
) -> np.ndarray:
    """(n, 3) CORN threshold logits (model forward = classifier(dropout(pooler)))."""
    model.eval()
    loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_batch,
    )
    logits_list: list[np.ndarray] = []
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
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            logits_list.append(out.logits.float().cpu().numpy())
            if (i + 1) % 25 == 0 or i + 1 == n_batches:
                elapsed = time.time() - t0
                print(f"{_ts()}    inference batch {i + 1}/{n_batches} "
                      f"({elapsed:.0f}s elapsed)")
    return np.concatenate(logits_list)


# ---------------------------------------------------------------------------
# Monotonic consistency + permutation null
# ---------------------------------------------------------------------------

def monotone_consistency(thr_probs: np.ndarray) -> np.ndarray:
    """Per-cell P(y>=1) >= P(y>=2) >= P(y>=3) (boolean, n,)."""
    return (thr_probs[:, 0] >= thr_probs[:, 1]) & (thr_probs[:, 1] >= thr_probs[:, 2])


def permutation_null(thr_probs: np.ndarray, n_draws: int = N_PERM_DRAWS,
                     seed: int = PERM_SEED) -> float:
    """Expected consistency under an UNORDERED head.

    Within-cell random permutation of the 3 threshold slots (the same 3 values
    reassigned to the 3 thresholds), seeded. Ties handled naturally. For 3
    distinct values the chance consistency is exactly 1/6.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        perm = rng.permuted(thr_probs, axis=1)  # per-row column permutation
        draws[d] = monotone_consistency(perm).mean()
    return float(draws.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["lam0.0", "lam0.5"], default="lam0.0")
    parser.add_argument("--seed", type=int, choices=[42, 123, 456], default=42)
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint dir; default = terminal (epoch-3) "
                             "checkpoint of the arm/seed")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = resolve_checkpoint(args.arm, args.seed, args.checkpoint)
    lam = float(args.arm.split("lam")[-1])
    step = _checkpoint_step(ckpt)
    epoch = _checkpoint_epoch(ckpt)
    print(f"{_ts()}  arm={args.arm} seed={args.seed} checkpoint={ckpt} "
          f"(step={step}, epoch={epoch if epoch is None else round(epoch, 4)})")

    device = _detect_device()
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"{_ts()}  Device: {device} dtype: {torch_dtype}")

    dataset, labels, layer, donor = _load_external_cells()
    n = len(dataset)
    print(f"{_ts()}  External eval cells: {n}")
    if n != 14775:
        print(f"{_ts()}  WARN: expected 14,775 epithelial cells, got {n}")

    t0 = time.time()
    model = BertForSequenceClassification.from_pretrained(
        str(ckpt), num_labels=N_THRESHOLDS, torch_dtype=torch_dtype,
    )
    model = model.to(device)
    logits = run_logit_inference(model, dataset, batch_size=BATCH_SIZE, device=device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    infer_s = time.time() - t0
    print(f"{_ts()}  logits shape {logits.shape}; inference {infer_s:.0f}s")

    preds_ord, thr_probs = ordinal_decode(logits)
    probs = ordinal_class_probs(thr_probs)
    preds_str = np.asarray([EVAL_CLASSES[c] for c in preds_ord], dtype=object)
    print(f"{_ts()}  ordinal class counts: "
          f"{np.unique(preds_str, return_counts=True)[0].tolist()} "
          f"{np.unique(preds_str, return_counts=True)[1].tolist()}")

    # --- monotonic consistency (observed) + permutation null -------------
    cons = monotone_consistency(thr_probs)
    overall = float(cons.mean())
    per_stage: dict[str, dict] = {}
    for cls in EVAL_CLASSES:
        m = labels == cls
        per_stage[cls] = {
            "n": int(m.sum()),
            "monotone_consistent": float(cons[m].mean()) if m.any() else None,
        }
    null = permutation_null(thr_probs)
    print(f"{_ts()}  monotone consistency overall={overall:.4f} "
          f"(permutation null={null:.4f})")

    # --- save -------------------------------------------------------------
    stem = f"percell_{args.arm}_{args.seed}"
    npz_path = OUT_DIR / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        logits=logits.astype(np.float32),
        thr_probs=thr_probs.astype(np.float32),
        probs=probs.astype(np.float32),
        preds=preds_ord.astype(np.int64),
        labels=labels,
        layer=layer,
        donor_id=donor,
        eval_classes=np.asarray(EVAL_CLASSES, dtype=object),
    )
    print(f"{_ts()}  Saved {npz_path}")

    summary = {
        "model": {
            "arm": args.arm,
            "lambda": lam,
            "seed": args.seed,
            "checkpoint": str(ckpt),
            "step": step,
            "epoch": epoch,
            "n_thresholds": N_THRESHOLDS,
        },
        "external_test": {
            "accession": "GSE153515",
            "file": str(EXT_DATA),
            "n_cells": n,
            "n_donors": int(len(np.unique(donor))),
            "non_layer_excluded": True,
        },
        "eval_classes": EVAL_CLASSES,
        "monotone_consistency": {
            "definition": "P(y>=1) >= P(y>=2) >= P(y>=3) per cell "
                          "(threshold sigmoid probabilities)",
            "overall": overall,
            "per_stage": per_stage,
        },
        "permutation_null": {
            "definition": "within-cell random permutation of the 3 threshold "
                          "slots (unordered-head expectation)",
            "n_draws": N_PERM_DRAWS,
            "seed": PERM_SEED,
            "expected_consistency": null,
        },
        "timing_seconds": round(time.time() - t0, 1),
        "device": device,
        "dtype": str(torch_dtype),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    json_path = OUT_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"{_ts()}  Saved {json_path}")
    print(f"{_ts()}  Done ({time.time() - t0:.0f}s total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
