#!/usr/bin/env python
"""Tier-B evaluation on an external test set (runs AFTER CAFT training completes).

Evaluates each trained checkpoint (arm x lambda x seed) on a donor-disjoint
external test dataset and produces the Paper1 primary endpoint (donor-level
macro-F1 with bootstrap CI) plus secondary endpoints (LAA, marker-gradient
rank correlation, ECE) via corneal_fm.evaluation.

Supported external test sets (selected with --accession, default gse153515):

* GSE153515 — Ocul Surf 2021 (doi:10.1016/j.jtos.2020.12.004, PMID 33388438);
  2 donors, legacy default paths (behavior-identical when no args are given).
* GSE299152 — PNAS Nexus 2025 (doi:10.1093/pnasnexus/pgaf348, PMID 41245088);
  3 donors (Donor2046/2206/3748), post-cutoff true external set, evaluated
  under the same locked protocol (config/external_test_gse299152.yml).

Checkpoint discovery is arm-aware: a run token may be ARM:LAM:SEED where ARM is
a save-dir prefix such as ``caft_gf`` or ``ordinal_gf`` (2-part LAM:SEED tokens
use the default arm caft_gf, preserving the original behavior).

Pipeline per checkpoint:
    1. Load tokenized external test cells (HF Dataset; cached, --skip-tokenize).
    2. Load the fine-tuned BertForSequenceClassification checkpoint.
    3. Extract CLS embeddings + 31-class logits in batches (bf16 autocast).
    4. Map 31 author_cell_type classes -> 4 harmonized layers
       ({LSC, TAC, basal, differentiated}; NON_LAYER excluded from evaluation).
    5. Call evaluate_on_external_test() (static + leakage gates inside).
    6. Save results_{[arm_]lambda}_{seed}.json and a combined summary.csv.

Reference: Li D-Q, Kim S, et al. Ocul Surf. 2021;20:20-32.
doi:10.1016/j.jtos.2020.12.004  (PMID: 33388438)

Usage:
    <venv>/bin/python3 scripts/run_tier_b_eval.py \
        [--accession gse153515|gse299152] [--runs ARM:]LAM:SEED,... \
        [--skip-tokenize]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# --- GPU / environment settings (mirror the training pipeline) ---------------
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

# Local project imports
from corneal_fm.caft import CELL_TYPE_TO_LAYER, NON_LAYER  # noqa: E402
from corneal_fm.evaluation import (  # noqa: E402
    evaluate_on_external_test,
    run_static_checks,
)
from corneal_fm.ordinal import (  # noqa: E402  CORN ordinal head (K-1 thresholds) decode
    N_THRESHOLDS,
    ordinal_class_probs,
    ordinal_decode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default external test set (legacy behavior: no --accession == gse153515).
DEFAULT_ACCESSION = "gse153515"
# Locked external-test configs per accession (relative to PROJECT_ROOT).
ACCESSION_CONFIGS = {
    "gse153515": "config/external_test_gse153515.yml",
    "gse299152": "config/external_test_gse299152.yml",
}
SUPPORTED_ACCESSION = sorted(ACCESSION_CONFIGS)
# Default checkpoint save-dir arm prefix (2-part LAM:SEED run tokens use this).
DEFAULT_ARM = "caft_gf"
# CORN ordinal arm: K-1=3 binary-threshold head (ordinal_gf.py loads the model
# with num_classes=N_THRESHOLDS); the ordinal decode is the number of
# thresholds with sigmoid >= 0.5 (corneal_fm.ordinal.ordinal_decode).
ORDINAL_ARM = "ordinal_gf"
# Frozen-reference pseudo-arm (no save dir; the HF Geneformer-V2-104M snapshot
# with a fresh 4-class head — used for the frozen gradient baseline).
FROZEN_ARM = "frozen"
_HF_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M"
)

# Legacy default-accession paths (unchanged when --accession is not given).
SAVE_DIR = PROJECT_ROOT / "save"
EXT_DATA = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
TRAIN_DATA = PROJECT_ROOT / "data" / "corneal_100k_donor_split.h5ad"
CONFIG_PATH = PROJECT_ROOT / "config" / "external_test_gse153515.yml"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "paper1_fair_split" / "manifest.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
TOKEN_CACHE = OUT_DIR / "external_tokenized"
TOKEN_DATASET = TOKEN_CACHE / "cell.dataset"

EVAL_CLASSES = ["LSC", "TAC", "basal", "differentiated"]
LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]
# 5-class CAFT layer index -> 4-class harmonized evaluation label.
HARMONIZED_MAP = {
    0: "LSC",
    1: "TAC",
    2: "basal",
    3: "differentiated",  # wing
    4: "differentiated",  # squamous
    NON_LAYER: "NON_LAYER",
}

N_BOOT = 1000
BOOT_SEED = 42
BATCH_SIZE = 32

LAMBDAS = [0.0, 0.5]
SEEDS = [42, 123, 456]


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Accession / run-path resolution
# ---------------------------------------------------------------------------

def resolve_accession(accession: str | None) -> str:
    """Normalize and validate an --accession value (default gse153515)."""
    acc = (accession or DEFAULT_ACCESSION).strip().lower()
    if acc not in ACCESSION_CONFIGS:
        raise ValueError(
            f"unsupported --accession {accession!r}; "
            f"supported: {', '.join(SUPPORTED_ACCESSION)}"
        )
    return acc


def resolve_run_paths(accession: str | None = None) -> dict:
    """Resolve config / data / output / token-cache paths for an accession.

    The default accession (gse153515) resolves to exactly the legacy paths
    (config/external_test_gse153515.yml, data/gse153515_annotated.h5ad,
    artifacts/paper1_tier_b_eval) so that running the script with no arguments
    is behavior-identical to the pre-parameterization version.

    Any other accession (gse299152) uses its own locked config, the annotated
    file declared in config['external_test']['annotated_file'], and a dedicated
    output + token-cache directory (artifacts/paper1_tier_b_eval_<accession>)
    so results never mix with the GSE153515 run.
    """
    acc = resolve_accession(accession)
    config_path = PROJECT_ROOT / ACCESSION_CONFIGS[acc]
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ext_rel = (cfg.get("external_test") or {}).get("annotated_file")
    ext_path = Path(str(ext_rel))
    ext_data = ext_path if ext_path.is_absolute() else PROJECT_ROOT / ext_path
    if acc == DEFAULT_ACCESSION:
        out_dir = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
    else:
        out_dir = PROJECT_ROOT / "artifacts" / f"paper1_tier_b_eval_{acc}"
    token_cache = out_dir / "external_tokenized"
    return {
        "accession": acc,
        "config_path": config_path,
        "ext_data": ext_data,
        "out_dir": out_dir,
        "token_cache": token_cache,
        "token_dataset": token_cache / "cell.dataset",
    }


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _hf_snapshot() -> Path:
    """Path of the Geneformer-V2-104M HF snapshot (for the frozen arm)."""
    import glob as _glob

    hits = sorted(_glob.glob(_HF_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError(f"Geneformer-V2-104M snapshot not found ({_HF_SNAP_GLOB})")
    return Path(hits[0])


def discover_checkpoints(runs: list[tuple[str, float, int]]) -> list[dict]:
    """Find the latest checkpoint per (arm, lambda, seed) combo.

    Returns list of dicts: {arm, lam, seed, save_dir, checkpoint_path, step}.
    Missing combos are warned and skipped. ``arm`` is the save-dir prefix
    (caft_gf / ordinal_gf / ...), so multiple arms can be evaluated in one run.
    The ``frozen`` pseudo-arm resolves to the HF snapshot (no save dir).
    """
    found = []
    for arm, lam, seed in runs:
        if arm == FROZEN_ARM:
            found.append({
                "arm": arm, "lam": lam, "seed": seed,
                "save_dir": Path("frozen"), "checkpoint_path": _hf_snapshot(), "step": 0,
            })
            continue
        pattern = f"{arm}_lam{lam}_seed{seed}_*"
        save_dirs = sorted(SAVE_DIR.glob(pattern))
        if not save_dirs:
            print(f"{_ts()}  WARN: no save dir for arm={arm} lam={lam} seed={seed} "
                  f"(pattern {pattern}); skipping.")
            continue
        # Latest timestamp dir (YYYYMMDD-HHMMSS sorts lexicographically).
        save_dir = save_dirs[-1]
        ckpt_dirs = sorted((save_dir / "model").glob("checkpoint-*"))
        if not ckpt_dirs:
            print(f"{_ts()}  WARN: no checkpoints under {save_dir / 'model'}; skipping.")
            continue
        ckpt_dir = ckpt_dirs[-1]
        step_match = re.search(r"checkpoint-(\d+)$", ckpt_dir.name)
        step = int(step_match.group(1)) if step_match else 0
        found.append({
            "arm": arm,
            "lam": lam,
            "seed": seed,
            "save_dir": save_dir,
            "checkpoint_path": ckpt_dir,
            "step": step,
        })
    return found


# ---------------------------------------------------------------------------
# Tokenization of the external test data
# ---------------------------------------------------------------------------

def _promote_counts(adata) -> None:
    """Ensure adata.X holds raw counts (annotated files store counts in a layer)."""
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        print(f"{_ts()}  Promoted layers['counts'] -> X (scaled matrix found in X).")


def tokenize_external(
    ext_data: Path,
    token_cache: Path,
    token_dataset: Path,
    accession: str,
    skip_tokenize: bool = False,
):
    """Tokenize an annotated external-test h5ad with the Geneformer V2 tokenizer.

    ``accession`` is the canonical study label (e.g. GSE299152) used to declare
    the source in ``obs['study']`` for the leakage gate. Returns
    (dataset, adata) where `adata` is filtered to epithelial cells
    (cell_type_harmonized != 'NON_LAYER') in the SAME order as the dataset.
    The tokenized HF Dataset is cached under ``token_cache`` and reused when
    ``--skip-tokenize`` is given.
    """
    import anndata as ad
    from geneformer import TranscriptomeTokenizer

    from corneal_fm import gf_compat
    gf_compat.install()

    token_cache.mkdir(parents=True, exist_ok=True)
    if skip_tokenize and token_dataset.is_file():
        print(f"{_ts()}  Loading cached tokenized dataset: {token_dataset}")
        from datasets import load_from_disk
        dataset = load_from_disk(str(token_dataset))
    else:
        print(f"{_ts()}  Reading {ext_data.name} ...")
        adata = ad.read_h5ad(ext_data)
        _promote_counts(adata)

        obs = adata.obs.copy().reset_index(drop=True)
        var = adata.var.copy()
        # Geneformer V2 keys its gene dictionary by Ensembl ID. GSE153515
        # already stores Ensembl IDs in var.index; GSE299152 stores gene
        # symbols with the Ensembl IDs in var['gene_ids'] (annotation pipeline
        # deviation #2), so remap var.index before tokenization.
        if not str(var.index[0]).startswith("ENSG") and "gene_ids" in var.columns:
            print(f"{_ts()}  var.index holds gene symbols; remapping to "
                  f"Ensembl IDs (var['gene_ids']) for tokenization.")
            var = var.copy()
            var.index = var["gene_ids"].values.astype(str)
        # Tokenizer requires n_counts in obs.
        obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        # Layer index from the harmonized cell type (author_cell_type absent here).
        layer_index = {"LSC": 0, "TAC": 1, "basal": 2, "differentiated": 3,
                       "NON_LAYER": NON_LAYER}
        obs["layer"] = obs["cell_type_harmonized"].map(layer_index).astype("int64")
        # Dummy values for custom attrs the tokenizer copies into the dataset.
        for col in ("cell_type", "author_cell_type", "disease", "tissue"):
            if col not in obs.columns:
                obs[col] = "unknown"

        # Keep Ensembl-ID var.index (use_h5ad_index=True) and expose collapsed id.
        if "ensembl_id_collapsed" not in var.columns and "ensembl_id" in var.columns:
            var["ensembl_id_collapsed"] = var["ensembl_id"].values

        adata.obs = obs
        adata.var = var

        tmp_dir = Path(tempfile.mkdtemp(prefix="tierb_tokenize_"))
        tmp_h5ad = tmp_dir / "external.h5ad"
        adata.write_h5ad(tmp_h5ad)

        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={
                "cell_type": "cell_type",
                "author_cell_type": "author_cell_type",
                "disease": "disease",
                "tissue": "tissue",
                "donor_id": "donor_id",
                "layer": "layer",
            },
            nproc=4,
            chunk_size=512,
            model_version="V2",
            use_h5ad_index=True,
            collapse_gene_ids=True,
        )
        tk.tokenize_data(
            data_directory=tmp_dir,
            output_directory=str(token_cache),
            output_prefix="cell",
            file_format="h5ad",
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        from datasets import load_from_disk
        dataset = load_from_disk(str(token_dataset))
        print(f"{_ts()}  Tokenized {len(dataset)} cells -> {token_dataset}")

    # Align adata (epithelial only) with the tokenized cell order.
    adata = ad.read_h5ad(ext_data)
    labels = adata.obs["cell_type_harmonized"].astype(str)
    mask = np.asarray(labels != "NON_LAYER")
    keep_idx = np.where(mask)[0]
    dataset = dataset.select(keep_idx)
    adata = adata[mask].copy()
    # Declare the study source explicitly: the leakage gate derives the external
    # accession from obs['study'] when present (else the GSE153515 module
    # default). GSE299152 has no 'study' column; for gse153515 this equals the
    # module default, so behavior is unchanged.
    adata.obs["study"] = accession
    print(f"{_ts()}  Epithelial filter: {len(keep_idx)}/{labels.shape[0]} cells "
          f"(NON_LAYER excluded).")
    return dataset, adata


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _collate_batch(batch: list[dict]):
    """Right-pad input_ids within a batch to the batch max length."""
    max_len = max(len(b["input_ids"]) for b in batch)
    padded = np.zeros((len(batch), max_len), dtype=np.int64)
    lengths = np.zeros(len(batch), dtype=np.int64)
    for i, b in enumerate(batch):
        ids = np.asarray(b["input_ids"], dtype=np.int64)
        padded[i, : len(ids)] = ids
        lengths[i] = int(b["length"])
    return torch.from_numpy(padded), torch.from_numpy(lengths)


def run_inference(
    model,
    tokenized_dataset,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """CLS embeddings (n, 768) and 31-class logits (n, 31) for all cells."""
    model.eval()
    loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_batch,
    )
    embeddings: list[np.ndarray] = []
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
            bert_out = model.bert(input_ids, attention_mask=attn_mask)
            emb = bert_out.last_hidden_state[:, 0]  # CLS at position 0
            logits = model.classifier(model.dropout(bert_out.pooler_output))
            embeddings.append(emb.float().cpu().numpy())
            logits_list.append(logits.float().cpu().numpy())
            if (i + 1) % 10 == 0 or i + 1 == n_batches:
                print(f"{_ts()}    inference batch {i + 1}/{n_batches}")
    return np.concatenate(embeddings), np.concatenate(logits_list)


# ---------------------------------------------------------------------------
# Prediction mapping: 31 author_cell_type classes -> 4 harmonized layers
# ---------------------------------------------------------------------------

def _load_id2label(save_dir: Path, checkpoint_path: Path) -> dict[int, str]:
    """id -> author_cell_type name. Prefers caft_id_class_dict.pkl; the
    checkpoint config.json id2label is numeric-string only (useless)."""
    pkl_candidates = sorted(
        (save_dir / "prepared").glob("caft_id_class_dict.pkl")
    ) + sorted(save_dir.glob("**/caft_id_class_dict.pkl"))
    if pkl_candidates:
        with open(pkl_candidates[0], "rb") as f:
            idc = pickle.load(f)
        return {int(k): str(v) for k, v in idc.items()}

    cfg_path = checkpoint_path / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        id2label = cfg.get("id2label")
        if id2label and not all(v.strip().isdigit() for v in id2label.values()):
            return {int(k): v for k, v in id2label.items()}

    raise FileNotFoundError(
        f"Could not locate a real id->class map for {save_dir} "
        f"(looked for caft_id_class_dict.pkl / non-numeric id2label)."
    )


def map_predictions(
    logits_31: np.ndarray,
    id2label: dict[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map 31-class logits to harmonized labels + aggregated probabilities.

    Returns (predictions, probabilities):
      predictions   (n,) string labels in {LSC,TAC,basal,differentiated,NON_LAYER}
      probabilities (n, 4) softmax summed over the classes mapping to each
      harmonized class (columns ordered as EVAL_CLASSES).
    """
    probs = np.zeros((logits_31.shape[0], len(LAYER_ORDER_4)))
    predictions = np.empty(logits_31.shape[0], dtype=object)

    # Precompute the harmonized bucket for each of the 31 classes.
    class_bucket: dict[int, str] = {}
    for cls_id, name in id2label.items():
        layer = CELL_TYPE_TO_LAYER.get(name, NON_LAYER)
        class_bucket[cls_id] = HARMONIZED_MAP.get(layer, "NON_LAYER")

    bucket_index = {c: i for i, c in enumerate(LAYER_ORDER_4)}
    for i in range(logits_31.shape[0]):
        row = logits_31[i]
        soft = np.exp(row - np.max(row))
        soft /= soft.sum()
        argmax_id = int(np.argmax(row))
        predictions[i] = class_bucket.get(argmax_id, "NON_LAYER")
        for cls_id, bucket in class_bucket.items():
            if bucket in bucket_index:
                probs[i, bucket_index[bucket]] += soft[cls_id]

    return predictions, probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_runs(
    runs_arg: str | None,
    default_arm: str = DEFAULT_ARM,
) -> list[tuple[str, float, int]]:
    """Parse --runs into (arm, lambda, seed) tuples.

    A token may be ``ARM:LAM:SEED`` (e.g. ordinal_gf:0.5:42) or the legacy
    ``LAM:SEED`` form, which uses ``default_arm`` (caft_gf — unchanged default).
    """
    if not runs_arg:
        return [(default_arm, lam, seed) for lam in LAMBDAS for seed in SEEDS]
    runs = []
    for token in runs_arg.split(","):
        parts = token.strip().split(":")
        if len(parts) == 3:
            arm_s, lam_s, seed_s = parts
        elif len(parts) == 2:
            arm_s, lam_s, seed_s = default_arm, parts[0], parts[1]
        else:
            raise ValueError(
                f"invalid run token {token!r} "
                f"(expected [ARM:]LAM:SEED, e.g. 0.0:42 or ordinal_gf:0.5:42)"
            )
        runs.append((arm_s.strip(), float(lam_s), int(seed_s)))
    return runs


def _detect_device() -> str:
    """CUDA if a GPU is visible (nvidia-smi), else CPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"{_ts()}  GPU detected: {out.stdout.strip().splitlines()[0]}")
            return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # nvidia-smi missing / timeout
        print(f"{_ts()}  WARN: nvidia-smi failed ({exc}); using CPU.")
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accession", default=None,
        help=f"External test set accession (default: {DEFAULT_ACCESSION}; "
             f"supported: {', '.join(SUPPORTED_ACCESSION)})",
    )
    parser.add_argument(
        "--runs", default=None,
        help="Comma-separated [ARM:]LAM:SEED pairs (default: all 6 of "
             f"{LAMBDAS} x {SEEDS} under the default arm {DEFAULT_ARM}); "
             "ARM is a checkpoint save-dir prefix such as caft_gf or ordinal_gf",
    )
    parser.add_argument(
        "--skip-tokenize", action="store_true",
        help="Reuse the cached tokenized external dataset",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Resolve per-accession paths (default gse153515 -> legacy paths).
    paths = resolve_run_paths(args.accession)
    accession = paths["accession"]
    config_path = paths["config_path"]
    ext_data = paths["ext_data"]
    out_dir = paths["out_dir"]
    token_cache = paths["token_cache"]
    token_dataset = paths["token_dataset"]
    print(f"{_ts()}  External test accession: {accession}")
    print(f"{_ts()}  Config: {config_path}")
    print(f"{_ts()}  External data: {ext_data}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Fail-closed static gate (locked config + split manifest).
    static_results = run_static_checks(
        config_path=str(config_path),
        split_manifest_path=str(MANIFEST_PATH),
    )
    print(f"{_ts()}  Static checks: {static_results}")

    runs = _parse_runs(args.runs)
    print(f"{_ts()}  Runs requested: {runs}")
    checkpoints = discover_checkpoints(runs)
    if not checkpoints:
        print(f"{_ts()}  ERROR: no checkpoints found for any requested run.")
        return 1

    device = _detect_device()
    print(f"{_ts()}  Device: {device}")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(MANIFEST_PATH) as f:
        split_manifest = json.load(f)
    study_accession = (config.get("external_test") or {}).get("accession", accession.upper())

    dataset, external_adata = tokenize_external(
        ext_data,
        token_cache,
        token_dataset,
        study_accession,
        skip_tokenize=args.skip_tokenize,
    )
    print(f"{_ts()}  External eval cells: {len(dataset)}")

    # Training obs for leakage checks (metadata only, backed read).
    import anndata as ad
    train_adata = ad.read_h5ad(str(TRAIN_DATA), backed="r")
    print(f"{_ts()}  Training data (leakage checks): {train_adata.n_obs} cells")

    summary_rows = []
    for ck in checkpoints:
        arm, lam, seed = ck["arm"], ck["lam"], ck["seed"]
        ckpt_path = ck["checkpoint_path"]
        print(f"\n{_ts()}  === arm={arm} lam={lam} seed={seed} step={ck['step']} "
              f"({ckpt_path}) ===")

        if arm == FROZEN_ARM:
            # Frozen reference: the HF snapshot with a fresh 4-class head (no
            # fine-tuning). The embeddings-only secondary metrics (rho/LAA) are
            # the point; the F1/ECE from the untrained head are near-chance.
            num_labels = len(LAYER_ORDER_4)
        elif arm == ORDINAL_ARM:
            # CORN ordinal head: K-1 = 3 binary thresholds ("y >= k").
            num_labels = N_THRESHOLDS
        else:
            id2label = _load_id2label(ck["save_dir"], ckpt_path)
            print(f"{_ts()}  id->class map: {len(id2label)} classes "
                  f"(e.g. 3 -> {id2label.get(3)})")
            num_labels = len(id2label)

        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            str(ckpt_path), num_labels=num_labels, torch_dtype=torch_dtype,
        )
        model = model.to(device)
        model.eval()

        embeddings, logits = run_inference(
            model, dataset, batch_size=args.batch_size, device=device,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        if arm == ORDINAL_ARM:
            # CORN inference: predicted class = number of thresholds with
            # sigmoid >= 0.5; class probabilities from the threshold masses.
            preds_int, thresh_probs = ordinal_decode(logits)
            predictions = np.asarray(
                [LAYER_ORDER_4[c] for c in preds_int], dtype=object
            )
            probabilities = ordinal_class_probs(thresh_probs)
        elif arm == FROZEN_ARM:
            soft = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = soft / soft.sum(axis=1, keepdims=True)
            predictions = np.asarray(
                [LAYER_ORDER_4[c] for c in np.argmax(logits, axis=1)], dtype=object
            )
        else:
            predictions, probabilities = map_predictions(logits, id2label)
        n_epi = int((predictions != "NON_LAYER").sum())
        print(f"{_ts()}  Predicted epithelial: {n_epi}/{len(predictions)}")

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

        # Non-default arms get an arm-qualified results file so a multi-arm run
        # never collides; the default arm keeps the legacy filename; the frozen
        # reference keeps the frozen_gf name (mirrors the GSE153515 artifact).
        if arm == FROZEN_ARM:
            out_json = out_dir / "results_frozen_gf.json"
        elif arm == DEFAULT_ARM:
            out_json = out_dir / f"results_{lam}_{seed}.json"
        else:
            out_json = out_dir / f"results_{arm}_{lam}_{seed}.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{_ts()}  Saved {out_json}")

        pe = results["primary_endpoint"]
        se = results["secondary_endpoints"]
        summary_rows.append({
            "arm": arm,
            "lambda": lam,
            "seed": seed,
            "checkpoint": ckpt_path.name,
            "macro_f1": pe["macro_f1"],
            "ci_lower": pe["ci_lower"],
            "ci_upper": pe["ci_upper"],
            "LAA": se["layer_adjacency_accuracy"],
            "marker_gradient_rho": se["marker_gradient_rank_correlation"],
            "ECE": se["expected_calibration_error"],
        })

    df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{_ts()}  Saved {csv_path}")
    print(df.to_string(index=False))
    print(f"{_ts()}  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
