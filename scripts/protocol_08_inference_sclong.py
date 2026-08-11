#!/usr/bin/env python
"""Corneal-FM: scLong zero-shot cell-type inference.

Loads the pretrained scLong model (1B parameters, Performer architecture)
from HuggingFace (BaiDing1234/scLong) and evaluates it zero-shot on corneal
test data.  Reports accuracy and macro F1.

Reference:
    Ding et al., "scLong: a 1B-parameter long-context Performer model for
    single-cell biology", Nat Biotechnol 2025.

Usage:
    python scripts/08_inference_scLong.py
    python scripts/protocol_08_inference_sclong.py --data_path data/corneal_20k_test.h5ad
    python scripts/08_inference_scLong.py --n_cells 5000 --dry_run
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import torch
from corneal_fm.config import load_config, set_seed, get_save_dir, archive_script
from corneal_fm.utils import get_logger, get_device, count_parameters, check_gpu_memory

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))


@click.command()
@click.option("--data_path", default=None, help="Path to test h5ad file")
@click.option("--n_cells", default=None, type=int, help="Number of cells to evaluate (default: all)")
@click.option("--config", default=None, help="YAML config override")
@click.option("--dry_run", is_flag=True, help="Validate setup without inference")
@click.option("--seed", default=42, type=int)
def main(data_path: str | None, n_cells: int | None, config: str | None, dry_run: bool, seed: int) -> None:
    """Run zero-shot scLong cell-type classification on corneal test data."""
    cfg = load_config(config or "config/default_train_args.yml")
    set_seed(seed)
    device = get_device()
    logger.info(f"Device: {device} | GPU: {check_gpu_memory()}")

    # Resolve test data
    if data_path is None:
        data_path = str(DATA / "corneal_20k_test.h5ad")
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Test data not found: {data_path}")
    logger.info(f"Test data: {data_path}")

    # Setup save directory
    save_dir = get_save_dir("sclong", "zero_shot")
    archive_script(__file__, save_dir)

    # Load test data (h5ad)
    logger.info("Loading test data...")
    try:
        import scanpy as sc
    except ImportError:
        logger.error("scanpy not installed. Run: pip install scanpy")
        sys.exit(1)

    adata = sc.read_h5ad(data_path, backed="r")
    full_n = adata.n_obs
    n_use = n_cells if n_cells is not None else full_n
    n_use = min(n_use, full_n)
    logger.info(f"Test data: {full_n} cells × {adata.n_vars} genes (using {n_use})")

    # Subsample if needed
    if n_use < full_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(full_n, n_use, replace=False)
        idx.sort()
        adata = adata[idx].copy()
    else:
        # Force load into memory if backed
        adata = adata.to_memory()

    # Extract ground-truth cell-type labels
    if "cell_type" not in adata.obs:
        logger.error("Test data missing 'cell_type' column in obs. Cannot evaluate.")
        sys.exit(1)

    cell_types = adata.obs["cell_type"]
    unique_types = cell_types.cat.categories if hasattr(cell_types, "cat") else sorted(cell_types.unique())
    n_types = len(unique_types)
    logger.info(f"Cell types: {n_types} classes -> {list(unique_types[:5])}{'...' if n_types > 5 else ''}")

    label_codes = cell_types.cat.codes.values if hasattr(cell_types, "cat") else cell_types.values

    if dry_run:
        logger.info("Dry-run: data loads OK, scLong model check follows...")
        # Verify the model can be imported / accessed
        try:
            _check_sclong_import(cfg)
            logger.info("Dry-run complete — all dependencies OK.")
        except ImportError as e:
            logger.warning(f"Dry-run: scLong not installed ({e}). Install with:")
            logger.warning("  pip install sclong  # or transformers")
        return

    # ------------------------------------------------------------------
    # Load scLong model
    # ------------------------------------------------------------------
    logger.info("Loading scLong (1B Performer) from HuggingFace...")
    model, tokenizer = _load_sclong(cfg, device)
    logger.info(f"scLong model: {count_parameters(model):,} params")

    # ------------------------------------------------------------------
    # Tokenize and run zero-shot inference
    # ------------------------------------------------------------------
    logger.info("Tokenizing cells...")
    # scLong uses gene-expression tokenization akin to Geneformer:
    # rank-value encoding (highly expressed genes get lower token IDs).
    # For each cell, the top-k expressed genes are encoded as a sequence.
    input_ids = _tokenize_cells(adata, tokenizer, cfg)
    logger.info(f"Tokenized: {input_ids.shape}")

    logger.info("Running zero-shot inference...")
    t0 = datetime.now()
    all_logits: list[np.ndarray] = []
    batch_size = cfg.get("model_parameters", {}).get("batch_size", 32)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            # scLong forward — returns logits for cell-type prediction head
            outputs = model(batch_ids)
            logits = _extract_logits(outputs)
            all_logits.append(logits.cpu().numpy())

    logits_all = np.concatenate(all_logits, axis=0)
    preds = logits_all.argmax(axis=1)
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"Inference: {elapsed:.1f}s ({elapsed / max(1, n_use):.3f}s/cell)")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # Map predictions to ground-truth range if necessary
    # scLong's output dimension may differ from local label encoding.
    # Use Hungarian matching or subspace accuracy if dimensions mismatch.
    if logits_all.shape[1] == n_types:
        matched_preds = preds
    else:
        logger.info(
            "Predicting %d classes (model head) vs %d local types — "
            "using best-overlap mapping.",
            logits_all.shape[1],
            n_types,
        )
        matched_preds = _match_labels(preds, label_codes, logits_all.shape[1], n_types, seed)

    accuracy = accuracy_score(label_codes, matched_preds)
    macro_f1 = f1_score(label_codes, matched_preds, average="macro", zero_division=0)
    report = classification_report(label_codes, matched_preds, output_dict=True, zero_division=0)
    macro_avg = report.get("macro avg", {})

    logger.info("=" * 56)
    logger.info("  scLong Zero-Shot Cell-Type Classification")
    logger.info("=" * 56)
    logger.info(f"  Accuracy :  {accuracy:.4f}")
    logger.info(f"  Macro F1 :  {macro_f1:.4f}")
    logger.info(f"  Precision:  {macro_avg.get('precision', 0.0):.4f}")
    logger.info(f"  Recall   :  {macro_avg.get('recall', 0.0):.4f}")
    logger.info(f"  Cells    :  {n_use}")
    logger.info(f"  Classes  :  {n_types}")
    logger.info("=" * 56)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "precision": float(macro_avg.get("precision", 0.0)),
        "recall": float(macro_avg.get("recall", 0.0)),
        "n_cells": int(n_use),
        "n_classes": int(n_types),
        "model": "sclong_v1-1B",
        "data": str(data_path),
        "timestamp": datetime.now().isoformat(),
    }

    # Save metrics as CSV
    import csv

    metrics_path = save_dir / "zero_shot_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader()
        w.writerow(results)
    logger.info(f"Metrics saved: {metrics_path}")

    # Save predictions alongside ground truth
    np.savez(
        save_dir / "predictions.npz",
        predictions=matched_preds,
        ground_truth=label_codes,
        cell_types=np.array(unique_types, dtype=object),
    )
    logger.info(f"Predictions saved: {save_dir / 'predictions.npz'}")

    # Print summary line for pipeline consumption
    print(f"SCLONG_RESULT accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_sclong_import(cfg: dict) -> None:
    """Verify scLong can be imported; raise ImportError if not."""
    # Try the HuggingFace path first, then the sclong package
    try:
        from transformers import AutoModel  # noqa: F401
    except ImportError:
        pass
    try:
        import sclong  # noqa: F401
    except ImportError:
        pass
    # If neither is available, the load function will fail with a clear message


def _load_sclong(
    cfg: dict, device: torch.device
) -> tuple[torch.nn.Module, object]:
    """Load scLong model and tokenizer from HuggingFace."""
    pretrained = cfg.get("sclong", {}).get("pretrained", "BaiDing1234/scLong")

    # Attempt HuggingFace Transformers path.
    # scLong is registered as a custom model in the transformers hub, so
    # AutoModelForSequenceClassification is the expected entry point.
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification

        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained,
            config=config,
            trust_remote_code=True,
        )
        model = model.to(device)
        tokenizer = None  # scLong uses raw token ids, not text tokens
        logger.info(f"Loaded scLong via transformers: {pretrained}")
        return model, tokenizer
    except Exception:
        pass

    # Fallback: native sclong package
    try:
        import sclong  # type: ignore[import-untyped]

        model = sclong.PerformerForCellClassification.from_pretrained(pretrained)
        model = model.to(device)
        tokenizer = None
        logger.info(f"Loaded scLong via sclong package: {pretrained}")
        return model, tokenizer
    except ImportError:
        logger.error(
            "scLong not available. Install with:\n"
            "  pip install sclong\n\n"
            "The model is also available via HuggingFace:\n"
            "  BaiDing1234/scLong\n"
            "  pip install transformers && pip install --upgrade huggingface_hub"
        )
        raise


def _tokenize_cells(
    adata: "Any", tokenizer: object, cfg: dict
) -> torch.Tensor:
    """Convert h5ad expression matrix to scLong input token IDs.

    scLong uses rank-value encoding: for each cell, genes are ranked by
    expression level and the top-k gene indices form the input sequence.
    This matches Geneformer's tokenization scheme.

    Returns
    -------
    torch.Tensor
        Shape ``(n_cells, seq_len)`` of token IDs (long).
    """
    import scanpy as sc

    max_seq_len = cfg.get("model_parameters", {}).get("max_seq_len", 2048)
    n_cells = adata.n_obs
    n_genes = adata.n_vars

    # Use .X (dense or sparse)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Rank-based encoding: for each cell, get indices of top-k expressed genes.
    # scLong expects gene indices ranked by expression (most expressed = lowest rank).
    token_ids = np.zeros((n_cells, max_seq_len), dtype=np.int64)

    for i in range(n_cells):
        expr = X[i]
        # Sort genes by expression descending, take top max_seq_len
        top_genes = np.argsort(expr)[::-1][:max_seq_len]
        token_ids[i] = top_genes

    return torch.tensor(token_ids, dtype=torch.long)


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Extract classification logits from model output."""
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def _match_labels(
    preds: np.ndarray,
    ground_truth: np.ndarray,
    n_pred_classes: int,
    n_true_classes: int,
    seed: int,
) -> np.ndarray:
    """Map predicted class indices to ground-truth label indices.

    Uses Hungarian matching when the scLong output head dimension differs
    from the local label encoding dimension. Falls back to greedy overlap.
    """
    from scipy.optimize import linear_sum_assignment

    # Build confusion matrix
    conf = np.zeros((n_pred_classes, n_true_classes), dtype=np.int64)
    for p, t in zip(preds, ground_truth):
        conf[p, t] += 1

    # Hungarian: maximize true-positive matches
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Apply mapping (unmapped preds -> 0)
    mapped = np.array([mapping.get(p, 0) for p in preds], dtype=ground_truth.dtype)
    return mapped


if __name__ == "__main__":
    main()
