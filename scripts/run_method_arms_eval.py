#!/usr/bin/env python
"""Paper 1 method-arm external-test evaluation (runs AFTER run_method_arms.py).

Evaluates each completed method-arm checkpoint (axis_ft / soft_spearman /
flow x lambda x seed) on the donor-disjoint external test (GSE153515) using
the same conventions as scripts/run_ordinal_eval.py:

    * same external test data (data/gse153515_annotated.h5ad), locked config
      (config/external_test_gse153515.yml) and split manifest
      (artifacts/paper1_fair_split/manifest.json) -> the static + leakage
      gates inside corneal_fm.evaluation.evaluate_on_external_test apply;
    * same cached tokenized external cells (read-only reuse of
      artifacts/paper1_tier_b_eval/external_tokenized/cell.dataset);
    * seed 42 / n_boot 1000 for the donor-level bootstrap CI; the ECE and
      datetime fixes are already in evaluation.py.

The method-arm head is the plain 4-class CE head (num_labels=4): predictions
= argmax of the 4 class logits, probabilities = softmax of the 4 logits (the
ordinal eval's CORN decode does NOT apply — these are not threshold heads).

Pipeline per checkpoint:
    1. resolve the LATEST save/method_<arm>_lam<lam>_seed<seed>_* dir that
       carries the METHOD_ARM_DONE marker (a newer dir without the marker,
       e.g. a crashed retrain, is ignored);
    2. pick model/checkpoint-7473 (the locked final step), falling back to
       the highest checkpoint step;
    3. run evaluate_on_external_test(...) on data/gse153515_annotated.h5ad;
    4. write artifacts/paper1_method_arms/results_<arm>_<lam>_<seed>.json
       (same schema as the ordinal results_*.json — primary_endpoint.
       donor_level_macro_f1 + secondary_endpoints.layer_adjacency_accuracy /
       marker_gradient_rank_correlation / expected_calibration_error);
    5. flip artifacts/paper1_method_arms/runs_status.json eval -> "OK" for
       that run.

Idempotent: runs whose results JSON already exists are skipped (logged per
run). --dry-run prints the resolved checkpoint paths + skip decisions WITHOUT
running evaluation (no dataset/model loading, CPU-only).

Usage:
    /home/dq/venv-ds/bin/python3 scripts/run_method_arms_eval.py [--runs arm:lam:seed,...] [--dry-run]
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.special import softmax  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from corneal_fm.evaluation import evaluate_on_external_test  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (Paper 1 locked evaluation contract, mirrors run_ordinal_eval.py)
# ---------------------------------------------------------------------------

SAVE_DIR = PROJECT_ROOT / "save"
EXT_DATA = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
TRAIN_DATA = PROJECT_ROOT / "data" / "corneal_100k_donor_split.h5ad"
CONFIG_PATH = PROJECT_ROOT / "config" / "external_test_gse153515.yml"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "paper1_fair_split" / "manifest.json"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "paper1_method_arms"
STATUS_JSON = RESULTS_DIR / "runs_status.json"
# Read-only reuse of the cached tokenized external dataset (built by
# run_tier_b_eval.py from the locked config + annotation).
TOKEN_DATASET = (
    PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval" / "external_tokenized" / "cell.dataset"
)

EVAL_CLASSES = ["LSC", "TAC", "basal", "differentiated"]
LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]
# Method-arm head: the plain 4-class CE head (N_CE_CLASSES = N_ORDINAL = 4).
N_CE_CLASSES = 4

N_HIDDEN_STATES = 13  # [0]=embedding output, [1..12]=transformer layers
FINAL_CKPT = "checkpoint-7473"  # locked final step (3 epochs, V2-104M)

N_BOOT = 1000
BOOT_SEED = 42
BATCH_SIZE = 32

VALID_ARMS = ("axis_ft", "soft_spearman", "flow")
# Phase A feasibility grid (seed 42) — the default run set.
DEFAULT_RUNS = [
    ("axis_ft", 0.5, 42),
    ("axis_ft", 1.0, 42),
    ("soft_spearman", 0.5, 42),
    ("soft_spearman", 1.0, 42),
    ("flow", 0.5, 42),
]


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Checkpoint discovery (mirrors run_method_arms.find_save_dir / is_done)
# ---------------------------------------------------------------------------

def find_done_save_dir(arm: str, lam: float, seed: int) -> Path | None:
    """Latest save dir for the run that carries the METHOD_ARM_DONE marker."""
    dirs = sorted(SAVE_DIR.glob(f"method_{arm}_lam{lam}_seed{seed}_*"))
    for d in reversed(dirs):
        if (d / "METHOD_ARM_DONE").is_file():
            return d
    return None


def pick_checkpoint(save_dir: Path) -> tuple[Path, int] | None:
    """(checkpoint_path, step): prefer the locked checkpoint-7473, else the
    highest numeric step under save_dir/model/."""
    model_dir = save_dir / "model"
    if not model_dir.is_dir():
        return None
    exact = model_dir / FINAL_CKPT
    if exact.is_dir():
        step = re.search(r"checkpoint-(\d+)$", exact.name)
        return exact, int(step.group(1)) if step else 0
    steps: list[tuple[int, Path]] = []
    for p in sorted(model_dir.glob("checkpoint-*")):
        if not p.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)$", p.name)
        if m:
            steps.append((int(m.group(1)), p))
    if not steps:
        return None
    step, path = max(steps, key=lambda t: t[0])
    return path, step


def discover_run(arm: str, lam: float, seed: int) -> dict | None:
    """Resolve one run to its final checkpoint; None when not eval-able."""
    save_dir = find_done_save_dir(arm, lam, seed)
    if save_dir is None:
        return None
    ckpt = pick_checkpoint(save_dir)
    if ckpt is None:
        return None
    ckpt_path, step = ckpt
    return {
        "arm": arm,
        "lambda": lam,
        "seed": seed,
        "save_dir": save_dir,
        "checkpoint_path": ckpt_path,
        "step": step,
    }


def result_json(arm: str, lam: float, seed: int) -> Path:
    return RESULTS_DIR / f"results_{arm}_{lam}_{seed}.json"


# ---------------------------------------------------------------------------
# External test cells (cached tokenized dataset, read-only reuse — identical
# to run_ordinal_eval._load_external_cells)
# ---------------------------------------------------------------------------

def _load_external_cells() -> tuple[object, np.ndarray, object]:
    """Tokenized epithelial cells + harmonized labels + filtered adata.

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
# Inference: layer-12 CLS embeddings + 4-class CE-head logits
# ---------------------------------------------------------------------------

def run_method_inference(
    model,
    tokenized_dataset,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """(n, 768) layer-12 CLS embeddings and (n, 4) class logits."""
    model.eval()
    loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_batch,
    )
    cls_list: list[np.ndarray] = []
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
            # CLS at position 0 of the final transformer layer (== layer 12).
            cls_list.append(hidden[-1][:, 0].float().cpu().numpy())
            logits = model.classifier(model.dropout(bert_out.pooler_output))
            logits_list.append(logits.float().cpu().numpy())
            if (i + 1) % 10 == 0 or i + 1 == n_batches:
                print(f"{_ts()}    inference batch {i + 1}/{n_batches}")
    return np.concatenate(cls_list), np.concatenate(logits_list)


# ---------------------------------------------------------------------------
# CE-head decode -> 4-class labels + probabilities (no CORN here)
# ---------------------------------------------------------------------------

def decode_ce_head(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 4) CE-head logits -> (n,) string labels + (n, 4) class probs."""
    probs = softmax(logits, axis=1)
    predictions = np.asarray(
        [EVAL_CLASSES[c] for c in np.argmax(logits, axis=1)], dtype=object
    )
    return predictions, probs


# ---------------------------------------------------------------------------
# runs_status.json bookkeeping
# ---------------------------------------------------------------------------

def mark_status_eval_ok(arm: str, lam: float, seed: int, save_dir: Path) -> None:
    """Flip the run's eval field to "OK" in runs_status.json (idempotent).

    Appends the run entry when absent (e.g. status file written by an older
    launcher or a partial grid); preserves every other key.
    """
    payload: dict = {"grid": [], "mode": "train", "runs": []}
    if STATUS_JSON.is_file():
        try:
            payload = json.loads(STATUS_JSON.read_text())
        except (json.JSONDecodeError, OSError, TypeError):
            payload = {"grid": [], "mode": "train", "runs": []}
    runs = payload.setdefault("runs", [])
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in runs:
        if (
            str(r.get("arm")) == arm
            and float(r.get("lambda")) == float(lam)
            and int(r.get("seed")) == int(seed)
        ):
            r["eval"] = "OK"
            r["updated_utc"] = stamp
            break
    else:
        runs.append({
            "arm": arm, "lambda": lam, "seed": seed,
            "train": "OK", "eval": "OK",
            "save_dir": str(save_dir), "updated_utc": stamp,
        })
    STATUS_JSON.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_runs(runs_arg: str | None) -> list[tuple[str, float, int]]:
    if not runs_arg:
        return list(DEFAULT_RUNS)
    runs = []
    for token in runs_arg.split(","):
        parts = token.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"bad run spec {token!r} (expected arm:lam:seed)")
        arm, lam_s, seed_s = parts
        if arm not in VALID_ARMS:
            raise ValueError(f"unknown arm {arm!r}; valid: {VALID_ARMS}")
        runs.append((arm, float(lam_s), int(seed_s)))
    return runs


def _detect_device() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # nvidia-smi missing / timeout
        pass
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", default=None,
        help="Comma-separated ARM:LAM:SEED triples (default: all 5 Phase A runs "
             f"{[(a, lam, s) for a, lam, s in DEFAULT_RUNS]})",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve checkpoint paths + skip decisions only; "
                             "do NOT run evaluation (CPU-only).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        runs = _parse_runs(args.runs)
    except ValueError as exc:
        print(f"{_ts()}  ERROR: {exc}")
        return 1
    print(f"{_ts()}  Runs requested: {runs}")

    # -- resolution + skip decisions (no torch/dataset work) ----------------
    resolved: list[dict] = []
    n_would_eval = 0
    n_skipped = 0
    n_missing = 0
    for arm, lam, seed in runs:
        res_json = result_json(arm, lam, seed)
        ck = discover_run(arm, lam, seed)
        if ck is None:
            n_missing += 1
            print(f"{_ts()}  {arm} λ={lam} seed={seed}  MISSING: no "
                  f"METHOD_ARM_DONE save dir / checkpoint — not eval-able")
            continue
        if res_json.is_file():
            n_skipped += 1
            decision = "SKIP (results JSON exists)"
        else:
            n_would_eval += 1
            decision = "WOULD EVAL"
        print(f"{_ts()}  {arm} λ={lam} seed={seed} step={ck['step']}  "
              f"[{decision}]  → {ck['checkpoint_path']}")
        if not args.dry_run and not res_json.is_file():
            resolved.append(ck)

    if args.dry_run:
        print(f"\n{_ts()}  dry-run: {n_would_eval} would evaluate, "
              f"{n_skipped} already have results, {n_missing} missing; "
              f"no evaluation run.")
        return 0

    if not resolved:
        print(f"{_ts()}  Nothing to evaluate (all requested runs missing or "
              f"already evaluated).")
        return 0

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

    n_ok = 0
    for ck in resolved:
        arm, lam, seed = ck["arm"], ck["lambda"], ck["seed"]
        ckpt_path = ck["checkpoint_path"]
        print(f"\n{_ts()}  === {arm} lam={lam} seed={seed} step={ck['step']} "
              f"({ckpt_path}) ===")

        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            str(ckpt_path), num_labels=N_CE_CLASSES, torch_dtype=torch_dtype,
        )
        model = model.to(device)
        embeddings, logits = run_method_inference(
            model, dataset, batch_size=args.batch_size, device=device,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        predictions, probabilities = decode_ce_head(logits)
        print(f"{_ts()}  class counts: "
              f"{np.unique(predictions, return_counts=True)[0].tolist()} "
              f"{np.unique(predictions, return_counts=True)[1].tolist()}")

        results = evaluate_on_external_test(
            train_adata=train_adata,
            external_adata=external_adata,
            embeddings=embeddings,
            predictions=predictions,
            probabilities=probabilities,
            config=config,
            split_manifest=split_manifest,
            classes=EVAL_CLASSES,
            layer_order=LAYER_ORDER_4,
            n_boot=N_BOOT,
            seed=BOOT_SEED,
        )

        out_json = result_json(arm, lam, seed)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{_ts()}  Saved {out_json}")

        mark_status_eval_ok(arm, lam, seed, ck["save_dir"])
        print(f"{_ts()}  runs_status.json eval -> OK for {arm} λ={lam} seed={seed}")

        pe = results["primary_endpoint"]
        se = results["secondary_endpoints"]
        print(f"{_ts()}  macro_f1={pe['macro_f1']:.4f} "
              f"(CI {pe['ci_lower']:.4f}-{pe['ci_upper']:.4f})  "
              f"LAA={se['layer_adjacency_accuracy']:.4f}  "
              f"rho={se['marker_gradient_rank_correlation']:.4f}  "
              f"ECE={se['expected_calibration_error']}")
        n_ok += 1

    print(f"\n{_ts()}  Done: {n_ok}/{len(resolved)} runs evaluated "
          f"-> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
