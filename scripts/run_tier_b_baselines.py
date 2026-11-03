#!/usr/bin/env python
"""Tier-B external-test BASELINE metrics (HVG/PCA and frozen Geneformer) for this study.

Computes the two baselines that accompany the fine-tuned CAFT checkpoints on the
donor-disjoint external test (GSE153515, 14,775 epithelial cells, NON_LAYER
excluded), filling the "—" cells of the figure set external-baseline table:

  Arm 1 — HVG/PCA (CPU):
      LOCKED preprocessing (config/external_test_gse153515.yml §evaluation_preprocessing):
      normalize_total(1e4) -> log1p -> HVG flavor='seurat' n_top=2000 (fit on TRAIN only)
      -> scale(max_value=10, mean/std fit on TRAIN, transformed onto external)
      -> PCA n_comps=30 (fit on train, transform external)
      -> 31-class logistic-regression probe (author_cell_type labels).

  Arm 2 — Frozen Geneformer (GPU):
      Geneformer-V2-104M frozen [CLS] embeddings (L2-normalised) for the tokenized
      donor-split train (99,606 cells) and the tokenized external test, then the
      same 31-class logistic-regression probe. Both tokenized datasets already
      exist on disk and were verified to be row-aligned with their source h5ads.

Both arms map the 31 author_cell_type classes -> 4 harmonized layers
({LSC, TAC, basal, differentiated}; NON_LAYER excluded from evaluation) by
summing softmax probability mass over the CELL_TYPE_TO_LAYER buckets, then score
with corneal_fm.evaluation (donor-level macro-F1 + donor-level bootstrap 95% CI,
per-class precision/recall/F1, layer-adjacency accuracy, marker-gradient rank
correlation, ECE) — the exact pipeline used for the fine-tuned checkpoints.
Results are written in the same JSON schema as results_{lam}_{seed}.json.

References: Li D-Q, Kim S, et al. Ocul Surf. 2021;20:20-32.
doi:10.1016/j.jtos.2020.12.004 (PMID: 33388438).

Usage:
    <venv>/bin/python3 scripts/run_tier_b_baselines.py [--arm 1|2|both]
        [--batch-size 32] [--skip-tokenize]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- GPU / environment settings (mirror the training/eval pipelines) ---------
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import numpy as np
import scanpy as sc
import torch
import yaml
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Local project imports
from corneal_fm.caft import CELL_TYPE_TO_LAYER, NON_LAYER
from corneal_fm.evaluation import evaluate_on_external_test, run_static_checks

# ---------------------------------------------------------------------------
# Constants (this study locked evaluation contract)
# ---------------------------------------------------------------------------

TRAIN_DATA = PROJECT_ROOT / "data" / "corneal_100k_donor_split.h5ad"
EXT_DATA = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
CONFIG_PATH = PROJECT_ROOT / "config" / "external_test_gse153515.yml"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "paper1_fair_split" / "manifest.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"

# Geneformer-V2-104M frozen weights (HF cache snapshot).
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

# Already-tokenized, row-aligned datasets (verified against source h5ads).
TOKENIZED_TRAIN = (
    PROJECT_ROOT
    / "save"
    / "caft_gf_lam0.5_seed42_20260801-200953"
    / "corneal_layer_subset_tk"
    / "cell.dataset"
)
TOKENIZED_EXT = OUT_DIR / "external_tokenized" / "cell.dataset"

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
N_HVG = 2000
N_PC = 30
SEED = 42
PROBE_MAX_ITER = 5_000
BATCH_SIZE = 32

# Internal (cell-level macro-F1 on the internal donor-disjoint test) baseline
# values from the figure set, for the end-of-run comparison table only.
INTERNAL_BASELINES = {"HVG/PCA": 0.7787, "Frozen Geneformer": 0.8834}


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Harmonization: 31 author_cell_type classes -> 4 evaluation classes
# ---------------------------------------------------------------------------

def _class_bucket_map(class_names: list[str]) -> dict[str, str]:
    """author_cell_type name -> harmonized evaluation class (NON_LAYER for out)."""
    bucket: dict[str, str] = {}
    for name in class_names:
        layer = CELL_TYPE_TO_LAYER.get(name, NON_LAYER)
        bucket[name] = HARMONIZED_MAP.get(layer, "NON_LAYER")
    return bucket


def harmonize_probe_outputs(
    proba_31: np.ndarray,
    predicted_31: np.ndarray,
    classes_31: list[str],
    class_bucket: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate 31-class probe probabilities into 4 harmonized buckets.

    Returns (predictions, probabilities):
      predictions   (n,) strings in {LSC,TAC,basal,differentiated,NON_LAYER}
                     (argmax of the 31-class logits mapped to its bucket)
      probabilities (n, 4) softmax mass summed over classes per bucket
                     (columns ordered as EVAL_CLASSES).
    """
    n = proba_31.shape[0]
    probs = np.zeros((n, len(LAYER_ORDER_4)), dtype=float)
    predictions = np.empty(n, dtype=object)
    bucket_index = {c: i for i, c in enumerate(LAYER_ORDER_4)}
    class_id = {name: i for i, name in enumerate(classes_31)}
    for i in range(n):
        # Argmax class -> harmonized bucket.
        pred_name = predicted_31[i]
        predictions[i] = class_bucket.get(pred_name, "NON_LAYER")
        # Sum probability mass per bucket (matches run_tier_b_eval.map_predictions).
        row = proba_31[i]
        for name, bucket in class_bucket.items():
            if bucket in bucket_index:
                probs[i, bucket_index[bucket]] += row[class_id[name]]
    return predictions, probs


# ---------------------------------------------------------------------------
# Arm 1 — HVG/PCA (CPU)
# ---------------------------------------------------------------------------

def _matrix_to_dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _load_train_counts() -> tuple[ad.AnnData, np.ndarray]:
    """Donor-split train AnnData (raw counts in .X) + 31-class author labels."""
    a = ad.read_h5ad(TRAIN_DATA)
    if a.raw is not None:
        a.X = a.raw.X.copy()
        a.raw = None
    y_train = a.obs["author_cell_type"].astype(str).to_numpy()
    return a, y_train


def _prepare_expression(adata) -> None:
    """In-place normalize_total(1e4) -> log1p (LOCKED preprocessing)."""
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)


def arm1_hvg_pca(config: dict, split_manifest: dict, out_json: Path) -> None:
    """HVG/PCA baseline: train-only HVG+scale+PCA, 31-class probe, Tier-B eval.

    HVG is fitted on TRAIN expression statistics only, but on the TRAIN ∩
    EXTERNAL gene universe: GSE153515 is a targeted limbal-epithelium panel
    (10,605 genes vs 35,477 in the training atlas), so full-atlas HVGs largely
    do not transfer. Restricting the universe keeps the LOCKED seurat/n_top=2000
    protocol while guaranteeing all selected HVGs exist in the external panel.
    """
    print(f"{_ts()}  [Arm 1] loading train counts ...")
    train, y_train = _load_train_counts()
    _prepare_expression(train)
    # External gene universe first (backed read — cheap, no full load).
    ext_b = ad.read_h5ad(EXT_DATA, backed="r")
    shared = sorted(set(train.var_names) & set(ext_b.var_names))
    n_ext_genes = ext_b.n_vars
    del ext_b
    print(f"{_ts()}  [Arm 1] gene universe restricted to train∩external: "
          f"{len(shared)}/{train.n_vars} (external panel {n_ext_genes} genes)")
    train = train[:, shared].copy()
    # HVG fit on TRAIN only (LOCKED: flavor='seurat', n_top_genes=2000).
    sc.pp.highly_variable_genes(
        train, n_top_genes=N_HVG, flavor="seurat"
    )
    hvg = train.var_names[train.var["highly_variable"].to_numpy()].tolist()
    print(f"{_ts()}  [Arm 1] train HVGs (shared universe): {len(hvg)}")

    Xtr = _matrix_to_dense(train[:, hvg].X)
    del train
    # scale(max_value=10): mean/std fit on TRAIN only, applied to external.
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0
    Xtr = np.clip((Xtr - mean) / std, -10.0, 10.0)
    pca = PCA(n_components=N_PC, random_state=SEED)
    pca.fit(Xtr)
    train_pca = pca.transform(Xtr)
    print(f"{_ts()}  [Arm 1] PCA fit on train -> {train_pca.shape} "
          f"(explained var {pca.explained_variance_ratio_.sum():.3f})")

    print(f"{_ts()}  [Arm 1] loading external test ...")
    ext = ad.read_h5ad(EXT_DATA)
    labels = ext.obs["cell_type_harmonized"].astype(str)
    mask = np.asarray(labels != "NON_LAYER")
    ext = ext[mask].copy()  # 14,775 epithelial cells
    # Raw counts -> X (GSE153515 stores counts in layers['counts']).
    ext.X = ext.layers["counts"].copy()
    _prepare_expression(ext)
    missing = [g for g in hvg if g not in ext.var_names]
    if missing:
        raise RuntimeError(
            f"{len(missing)} HVGs missing from external (universe bug): "
            f"{missing[:5]}"
        )
    Xte = _matrix_to_dense(ext[:, hvg].X)
    Xte = np.clip((Xte - mean) / std, -10.0, 10.0)
    ext_pca = pca.transform(Xte)
    print(f"{_ts()}  [Arm 1] external features -> {ext_pca.shape}")

    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER, random_state=SEED, solver="lbfgs"
    )
    probe.fit(train_pca, y_train)
    proba_31 = probe.predict_proba(ext_pca)
    pred_31 = probe.predict(ext_pca)
    print(f"{_ts()}  [Arm 1] probe: {len(probe.classes_)} classes, "
          f"train macro-F1@31classes (cell): "
          f"{float(_macro_f1_cell(train_pca, y_train, probe)):.4f}")

    class_bucket = _class_bucket_map(list(probe.classes_))
    predictions, probabilities = harmonize_probe_outputs(
        proba_31, pred_31, list(probe.classes_), class_bucket
    )
    return _finish_arm(
        label="HVG/PCA baseline",
        train_adata=ad.read_h5ad(TRAIN_DATA, backed="r"),
        external_adata=ext,
        embeddings=ext_pca,
        predictions=predictions,
        probabilities=probabilities,
        config=config,
        split_manifest=split_manifest,
        out_json=out_json,
    )


def _macro_f1_cell(X_train, y_train, probe) -> float:
    """Cell-level macro-F1 of the probe on its training cells (sanity only)."""
    from sklearn.metrics import f1_score
    y_pred = probe.predict(X_train)
    labels = np.unique(np.concatenate((y_train, y_pred)))
    return f1_score(y_train, y_pred, labels=labels, average="macro",
                    zero_division=0)


# ---------------------------------------------------------------------------
# Arm 2 — Frozen Geneformer (GPU)
# ---------------------------------------------------------------------------

def _geneformer_pad_id() -> int:
    try:
        from geneformer.classifier import TOKEN_DICTIONARY_FILE
    except Exception:
        return 0
    try:
        with open(TOKEN_DICTIONARY_FILE, "rb") as fh:
            td = pickle.load(fh)
        return int(td["<pad>"]) if "<pad>" in td else 0
    except Exception:
        return 0


def _frozen_embeddings(
    model: Any, dataset: Any, batch_size: int = BATCH_SIZE, pad_token_id: int = 0
) -> np.ndarray:
    """Frozen [CLS] embeddings (L2-normalised), right-padded batches.

    Mirrors scripts/e0_axis1_baseline.py `_frozen_embeddings` (fp16 autocast on
    CUDA; position-0 [CLS] attends only to real tokens via the length mask).
    """
    import torch
    import torch.nn.functional as functional

    model.eval()
    device = next(model.parameters()).device
    embeddings: list[np.ndarray] = []
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    t0 = time.time()
    with torch.no_grad():
        for b, start in enumerate(range(0, len(dataset), batch_size)):
            batch = dataset[start : start + batch_size]
            seqs = [list(ids) for ids in batch["input_ids"]]
            lengths = torch.as_tensor(batch["length"], device=device)
            max_len = int(lengths.max().item()) if seqs else 0
            input_ids = torch.full(
                (len(seqs), max_len), pad_token_id, dtype=torch.long, device=device
            )
            for row, ids in enumerate(seqs):
                input_ids[row, : len(ids)] = torch.as_tensor(
                    ids, dtype=torch.long, device=device
                )
            attention_mask = (
                torch.arange(max_len, device=device)[None, :] < lengths[:, None]
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(
                functional.normalize(
                    out.last_hidden_state[:, 0].float(), dim=-1
                ).cpu().numpy()
            )
            if (b + 1) % 200 == 0 or (b + 1) == n_batches:
                print(
                    f"{_ts()}    emb {min(start + batch_size, len(dataset))}/"
                    f"{len(dataset)} cells (batch {b + 1}/{n_batches}, "
                    f"{time.time() - t0:.0f}s)",
                    flush=True,
                )
    return np.concatenate(embeddings, axis=0)


def arm2_frozen_gf(config: dict, split_manifest: dict, out_json: Path,
                   batch_size: int, skip_tokenize: bool) -> None:
    """Frozen Geneformer baseline: CLS embeddings, 31-class probe, Tier-B eval."""
    from datasets import load_from_disk
    from transformers import BertModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{_ts()}  [Arm 2] device: {device}")

    # Reuse the already-tokenized, row-aligned datasets.
    if not TOKENIZED_TRAIN.is_dir():
        raise FileNotFoundError(
            f"tokenized train dataset not found: {TOKENIZED_TRAIN} "
            "(re-tokenize with caft_gf.py make_layer_tokenized first)"
        )
    if not TOKENIZED_EXT.is_dir():
        raise FileNotFoundError(
            f"tokenized external dataset not found: {TOKENIZED_EXT} "
            "(run scripts/run_tier_b_eval.py --skip-tokenize first)"
        )
    train_ds = load_from_disk(str(TOKENIZED_TRAIN))
    ext_ds = load_from_disk(str(TOKENIZED_EXT))
    print(f"{_ts()}  [Arm 2] tokenized train {len(train_ds)} cells, "
          f"external {len(ext_ds)} cells")

    # Epithelial filter on the external side (same keep_idx rule as run_tier_b_eval).
    ext_adata = ad.read_h5ad(EXT_DATA)
    labels = ext_adata.obs["cell_type_harmonized"].astype(str)
    mask = np.asarray(labels != "NON_LAYER")
    keep_idx = np.where(mask)[0]
    ext_ds = ext_ds.select(keep_idx)
    ext_adata = ext_adata[mask].copy()
    print(f"{_ts()}  [Arm 2] external epithelial cells: {len(ext_ds)}")

    print(f"{_ts()}  [Arm 2] loading Geneformer-V2-104M ...")
    model = BertModel.from_pretrained(str(PRETRAINED_DIR)).to(device)
    pad_token_id = _geneformer_pad_id()
    print(f"{_ts()}  [Arm 2] pad_token_id={pad_token_id}")

    print(f"{_ts()}  [Arm 2] frozen CLS embeddings: train ...")
    train_emb = _frozen_embeddings(
        model, train_ds, batch_size=batch_size, pad_token_id=pad_token_id
    )
    print(f"{_ts()}  [Arm 2] frozen CLS embeddings: external ...")
    ext_emb = _frozen_embeddings(
        model, ext_ds, batch_size=batch_size, pad_token_id=pad_token_id
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    y_train = np.asarray(train_ds["author_cell_type"], dtype=str)
    print(f"{_ts()}  [Arm 2] train labels: {len(np.unique(y_train))} classes "
          f"({train_emb.shape})")

    # Same probe as the internal frozen-Geneformer baseline (e0_axis1_baseline):
    # StandardScaler on the (already L2-normalised) embeddings, then logreg.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_emb)
    Xte = scaler.transform(ext_emb)
    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER, random_state=SEED, solver="lbfgs"
    )
    probe.fit(Xtr, y_train)
    proba_31 = probe.predict_proba(Xte)
    pred_31 = probe.predict(Xte)
    print(f"{_ts()}  [Arm 2] probe: {len(probe.classes_)} classes")

    class_bucket = _class_bucket_map(list(probe.classes_))
    predictions, probabilities = harmonize_probe_outputs(
        proba_31, pred_31, list(probe.classes_), class_bucket
    )
    return _finish_arm(
        label="Frozen Geneformer V2-104M baseline",
        train_adata=ad.read_h5ad(TRAIN_DATA, backed="r"),
        external_adata=ext_adata,
        embeddings=ext_emb,
        predictions=predictions,
        probabilities=probabilities,
        config=config,
        split_manifest=split_manifest,
        out_json=out_json,
    )


# ---------------------------------------------------------------------------
# Shared evaluation / save / summary
# ---------------------------------------------------------------------------

def _finish_arm(
    *,
    label: str,
    train_adata,
    external_adata,
    embeddings: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    config: dict,
    split_manifest: dict,
    out_json: Path,
) -> dict:
    """Run evaluate_on_external_test and save the results JSON (Tier-B schema)."""
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
    results["model"] = label
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    pe = results["primary_endpoint"]
    se = results["secondary_endpoints"]
    print(f"{_ts()}  {label}: donor-level macro-F1 = {pe['macro_f1']:.4f} "
          f"(95% CI {pe['ci_lower']:.4f}-{pe['ci_upper']:.4f}), "
          f"LAA={se['layer_adjacency_accuracy']:.4f}, "
          f"marker_rho={se['marker_gradient_rank_correlation']:.4f}, "
          f"ECE={se['expected_calibration_error']:.4f}")
    print(f"{_ts()}  Saved {out_json}")
    return results


def _print_comparison(rows: list[dict]) -> None:
    """Internal (cell-level) vs external (donor-level) baseline summary table."""
    print("\n" + "=" * 92)
    print("BASELINE COMPARISON — internal (cell-level) vs external GSE153515 (donor-level)")
    print("=" * 92)
    header = ["Baseline", "Internal\nmacro-F1*", "External\nmacro-F1",
              "95% CI", "LAA", "marker_rho", "ECE"]
    print(f"{header[0]:<28}{header[1]:>12}{header[2]:>14}{header[3]:>22}"
          f"{header[4]:>8}{header[5]:>12}{header[6]:>8}")
    print("-" * 92)
    for r in rows:
        ci = "[{:.4f}, {:.4f}]".format(r["ci_lower"], r["ci_upper"])
        print(f"{r['name']:<28}{r['internal']:>12.4f}{r['external']:>14.4f}"
              f"{ci:>22}"
              f"{r['laa']:>8.4f}{r['rho']:>12.4f}{r['ece']:>8.4f}")
    print("-" * 92)
    print("* Internal values are cell-level macro-F1 on the internal donor-disjoint "
          "test (figure set);")
    print("  external values are donor-level macro-F1 on GSE153515 — not directly "
          "comparable (different")
    print("  granularity and test set). TAC (n=171) contributes ~0 F1 on the external "
          "set by design.")


# ---------------------------------------------------------------------------
# GPU availability gate
# ---------------------------------------------------------------------------

def _gpu_free_for_compute() -> tuple[bool, str]:
    """True when no non-GUI compute process (vLLM/training) holds the GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return False, f"nvidia-smi compute query failed: {out.stderr.strip()}"
        compute = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
        if compute:
            return False, "GPU busy with compute processes:\n  " + "\n  ".join(compute)
        return True, "no compute processes on GPU (GUI processes only)"
    except Exception as exc:
        return False, f"nvidia-smi unavailable: {exc}"


def _detect_device() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"{_ts()}  GPU detected: {out.stdout.strip().splitlines()[0]}")
            return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["1", "2", "both"], default="both",
                        help="Which baseline arm to run (1=HVG/PCA, 2=frozen GF).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Reuse cached tokenized datasets (default).")
    args = parser.parse_args()

    # Fail-closed static gate (locked config + split manifest).
    static = run_static_checks(str(CONFIG_PATH), str(MANIFEST_PATH))
    print(f"{_ts()}  Static checks: {static}")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(MANIFEST_PATH) as f:
        split_manifest = json.load(f)

    device = _detect_device()
    print(f"{_ts()}  Device: {device}")

    rows: list[dict] = []
    if args.arm in ("1", "both"):
        t0 = time.time()
        r = arm1_hvg_pca(
            config, split_manifest, OUT_DIR / "results_hvg_pca.json"
        )
        rows.append({
            "name": "HVG/PCA",
            "internal": INTERNAL_BASELINES["HVG/PCA"],
            "external": r["primary_endpoint"]["macro_f1"],
            "ci_lower": r["primary_endpoint"]["ci_lower"],
            "ci_upper": r["primary_endpoint"]["ci_upper"],
            "laa": r["secondary_endpoints"]["layer_adjacency_accuracy"],
            "rho": r["secondary_endpoints"]["marker_gradient_rank_correlation"],
            "ece": r["secondary_endpoints"]["expected_calibration_error"],
        })
        print(f"{_ts()}  Arm 1 done in {time.time() - t0:.0f}s")

    if args.arm in ("2", "both"):
        if device != "cuda":
            print(f"{_ts()}  WARN: no CUDA device; skipping Arm 2 (frozen GF needs GPU).")
        else:
            free, msg = _gpu_free_for_compute()
            if not free:
                print(f"{_ts()}  ABORT Arm 2: {msg}")
                print(f"{_ts()}  Arm 2 requires the GPU; not killing other processes.")
            else:
                print(f"{_ts()}  GPU gate: {msg}")
                t0 = time.time()
                r = arm2_frozen_gf(
                    config, split_manifest, OUT_DIR / "results_frozen_gf.json",
                    args.batch_size, args.skip_tokenize,
                )
                rows.append({
                    "name": "Frozen Geneformer",
                    "internal": INTERNAL_BASELINES["Frozen Geneformer"],
                    "external": r["primary_endpoint"]["macro_f1"],
                    "ci_lower": r["primary_endpoint"]["ci_lower"],
                    "ci_upper": r["primary_endpoint"]["ci_upper"],
                    "laa": r["secondary_endpoints"]["layer_adjacency_accuracy"],
                    "rho": r["secondary_endpoints"]["marker_gradient_rank_correlation"],
                    "ece": r["secondary_endpoints"]["expected_calibration_error"],
                })
                print(f"{_ts()}  Arm 2 done in {time.time() - t0:.0f}s")

    if rows:
        _print_comparison(rows)
    else:
        print(f"{_ts()}  Nothing ran; check --arm and GPU availability.")
    print(f"{_ts()}  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
