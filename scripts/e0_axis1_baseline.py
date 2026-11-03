#!/usr/bin/env python
"""E0 Axis-1 GO/NO-GO gate for cell-type discrimination (Methods/design §11).

This minimal four-axis-framework comparator asks whether frozen Geneformer
embeddings outperform train-only HVG/PCA features on the supplied,
donor-disjoint held-out split. It does not fine-tune a model or run CAFT.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import click
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class ProbeEvaluation:
    predictions: np.ndarray
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: dict[str, Any]


def validate_no_donor_leakage(
    train_obs: pd.DataFrame, test_obs: pd.DataFrame, donor_key: str
) -> dict[str, int | bool]:
    if donor_key not in train_obs or donor_key not in test_obs:
        raise KeyError(f"Missing donor key {donor_key!r} in one or both splits.")
    train_donors = set(train_obs[donor_key].dropna().astype(str))
    test_donors = set(test_obs[donor_key].dropna().astype(str))
    overlap = train_donors & test_donors
    if overlap:
        raise ValueError(f"Donor overlap detected: {sorted(overlap)!r}")
    return {
        "train_donors": len(train_donors),
        "test_donors": len(test_donors),
        "donor_disjoint": True,
    }


def select_hvg_from_train(train_adata: ad.AnnData, n_top: int) -> list[str]:
    """Mark and return HVGs fitted solely from the supplied training cells."""
    try:
        sc.pp.highly_variable_genes(train_adata, n_top_genes=n_top, flavor="seurat_v3")
    except (ImportError, ModuleNotFoundError):
        sc.pp.highly_variable_genes(train_adata, n_top_genes=n_top, flavor="seurat")
    return train_adata.var_names[train_adata.var["highly_variable"].to_numpy()].tolist()


def _matrix_to_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def fit_eval_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> ProbeEvaluation:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(X_train)
    test_features = scaler.transform(X_test)
    probe = LogisticRegression(max_iter=5_000, random_state=seed, solver="lbfgs")
    probe.fit(train_features, y_train)
    predictions = probe.predict(test_features)
    labels = np.unique(np.concatenate((y_train, y_test)))
    per_class = f1_score(y_test, predictions, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    return ProbeEvaluation(
        predictions=predictions,
        macro_f1=float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        per_class_f1={str(label): float(score) for label, score in zip(labels, per_class, strict=True)},
        confusion={"labels": [str(label) for label in labels], "matrix": matrix.tolist()},
    )


def go_no_go(scfm_macro_f1: float, baseline_macro_f1: float) -> str:
    return "GO" if scfm_macro_f1 > baseline_macro_f1 else "NO_GO"


def build_result(
    *,
    label: str,
    seed: int,
    train_path: str,
    test_path: str,
    split: dict[str, int | bool],
    baseline: dict[str, Any],
    scfm: dict[str, Any],
    class_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    baseline_score = float(baseline["macro_f1"])
    scfm_score = float(scfm["macro_f1"])
    return {
        "task": "E0 Axis-1 frozen-scFM versus HVG/PCA linear-probe comparator",
        "label": label,
        "seed": seed,
        "split": {"train_path": train_path, "test_path": test_path, **split},
        "baseline_hvg_pca": baseline,
        "scfm_frozen": scfm,
        "go_no_go": go_no_go(scfm_score, baseline_score),
        "rule": "GO iff frozen scFM held-out macro-F1 is strictly greater than HVG/PCA macro-F1.",
        "n_classes": len(set(class_counts["train"]) | set(class_counts["test"])),
        "class_counts": class_counts,
    }


def write_result(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _prepare_expression(adata: ad.AnnData) -> ad.AnnData:
    counts = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    sc.pp.normalize_total(counts, target_sum=10_000)
    sc.pp.log1p(counts)
    return counts


def _baseline_embeddings(
    train_adata: ad.AnnData, test_adata: ad.AnnData, n_hvg: int, n_pc: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    train = _prepare_expression(train_adata)
    test = _prepare_expression(test_adata)
    hvg_names = select_hvg_from_train(train, n_hvg)
    missing = set(hvg_names) - set(test.var_names)
    if missing:
        raise ValueError(f"Held-out split is missing {len(missing)} train-selected HVGs.")
    train_hvg = _matrix_to_dense(train[:, hvg_names].X)
    test_hvg = _matrix_to_dense(test[:, hvg_names].X)
    components = min(n_pc, train_hvg.shape[0], train_hvg.shape[1])
    if components < 1:
        raise ValueError("PCA requires at least one training cell and one selected HVG.")
    pca = PCA(n_components=components, random_state=seed)
    return pca.fit_transform(train_hvg), pca.transform(test_hvg), len(hvg_names), components


def _prepare_tokenizer_adata(adata: ad.AnnData) -> ad.AnnData:
    tokenizable = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    tokenizable.obs = tokenizable.obs.reset_index(drop=True)
    # Keep the Ensembl-ID var.index for TranscriptomeTokenizer(use_h5ad_index=True);
    # resetting it breaks gene matching (pair with the pandas2 shim installed in
    # _tokenize_adata via corneal_fm.gf_compat).
    if "ensembl_id_collapsed" not in tokenizable.var and "ensembl_id" in tokenizable.var:
        tokenizable.var["ensembl_id_collapsed"] = tokenizable.var["ensembl_id"].to_numpy()
    tokenizable.obs["n_counts"] = np.asarray(tokenizable.X.sum(axis=1)).ravel()
    return tokenizable


def _tokenize_adata(adata: ad.AnnData, workspace: Path, prefix: str, label: str, donor_key: str):
    import sys

    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corneal_fm import gf_compat

    gf_compat.install()

    source = _prepare_tokenizer_adata(adata)
    # tokenize_data globs every h5ad in its input dir, so each split needs its own
    # directory; a shared dir made the later split tokenize both files (wrong N).
    input_dir = workspace / f"{prefix}_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_path = input_dir / f"{prefix}.h5ad"
    output_dir = workspace / f"{prefix}_tokenized"
    source.write_h5ad(source_path)
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={label: label, donor_key: donor_key},
        nproc=1,
        chunk_size=512,
        model_version="V2",
        use_h5ad_index=True,
        collapse_gene_ids=True,
    )
    tokenizer.tokenize_data(
        data_directory=str(input_dir),
        output_directory=str(output_dir),
        output_prefix=prefix,
        file_format="h5ad",
    )
    return load_from_disk(str(output_dir / f"{prefix}.dataset"))


def _frozen_embeddings(
    model: Any, dataset: Any, batch_size: int = 32, pad_token_id: int = 0
) -> np.ndarray:
    import time

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
            # Right-pad the variable-length tokenized cells to a rectangular tensor and
            # mask padded positions so the [CLS] at position 0 only attends to real tokens.
            # Keep the batch modest: attention memory is O(batch*seq^2) and large batches
            # froze the <compute-node>'s unified memory. last_hidden_state[:,0] == hidden_states[-1][:0]
            # but avoids materializing every layer's output.
            max_len = int(lengths.max().item()) if seqs else 0
            input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long, device=device)
            for row, ids in enumerate(seqs):
                input_ids[row, : len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=device)
            attention_mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(functional.normalize(out.last_hidden_state[:, 0].float(), dim=-1).cpu().numpy())
            if (b + 1) % 100 == 0 or (b + 1) == n_batches:
                print(
                    f"[emb] {min(start + batch_size, len(dataset))}/{len(dataset)} cells "
                    f"(batch {b + 1}/{n_batches}, {time.time() - t0:.0f}s)",
                    flush=True,
                )
    return np.concatenate(embeddings, axis=0)


def _geneformer_pad_id() -> int:
    import pickle

    try:
        from geneformer.classifier import TOKEN_DICTIONARY_FILE
    except Exception:
        return 0
    try:
        with open(TOKEN_DICTIONARY_FILE, "rb") as fh:
            token_dictionary = pickle.load(fh)
        return int(token_dictionary["<pad>"]) if "<pad>" in token_dictionary else 0
    except Exception:
        return 0


def _scfm_embeddings(
    train_adata: ad.AnnData, test_adata: ad.AnnData, pretrained: Path, label: str, donor_key: str
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import BertModel

    workspace = Path(tempfile.mkdtemp(prefix="e0_geneformer_"))
    try:
        train_dataset = _tokenize_adata(train_adata, workspace, "train", label, donor_key)
        test_dataset = _tokenize_adata(test_adata, workspace, "test", label, donor_key)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertModel.from_pretrained(str(pretrained)).to(device)
        pad_token_id = _geneformer_pad_id()
        return (
            _frozen_embeddings(model, train_dataset, pad_token_id=pad_token_id),
            _frozen_embeddings(model, test_dataset, pad_token_id=pad_token_id),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


@click.command()
@click.option("--train", "train_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--test", "test_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--pretrained", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--label", default="cell_type", show_default=True)
@click.option("--donor-key", default="donor_id", show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--n-hvg", default=2_000, type=int, show_default=True)
@click.option("--n-pc", default=50, type=int, show_default=True)
@click.option("--out", "output", type=click.Path(path_type=Path), required=True)
def main(
    train_path: Path, test_path: Path, pretrained: Path, label: str, donor_key: str, seed: int,
    n_hvg: int, n_pc: int, output: Path,
) -> None:
    """Compare the frozen scFM and HVG/PCA probes on a fixed donor split."""
    train_adata = ad.read_h5ad(train_path)
    test_adata = ad.read_h5ad(test_path)
    if label not in train_adata.obs or label not in test_adata.obs:
        raise KeyError(f"Missing label {label!r} in one or both splits.")
    split = validate_no_donor_leakage(train_adata.obs, test_adata.obs, donor_key)
    y_train = train_adata.obs[label].astype(str).to_numpy()
    y_test = test_adata.obs[label].astype(str).to_numpy()
    unseen = set(y_test) - set(y_train)
    if unseen:
        raise ValueError(f"Held-out labels absent from training split: {sorted(unseen)!r}")
    baseline_train, baseline_test, hvg_count, pc_count = _baseline_embeddings(
        train_adata, test_adata, n_hvg, n_pc, seed
    )
    baseline = fit_eval_logreg(baseline_train, y_train, baseline_test, y_test, seed=seed)
    scfm_train, scfm_test = _scfm_embeddings(train_adata, test_adata, pretrained, label, donor_key)
    scfm = fit_eval_logreg(scfm_train, y_train, scfm_test, y_test, seed=seed)
    result = build_result(
        label=label,
        seed=seed,
        train_path=str(train_path),
        test_path=str(test_path),
        split=split,
        baseline={
            "macro_f1": baseline.macro_f1,
            "n_hvg": hvg_count,
            "n_pcs": pc_count,
            "per_class_f1": baseline.per_class_f1,
            "confusion": baseline.confusion,
        },
        scfm={
            "macro_f1": scfm.macro_f1,
            "model": str(pretrained),
            "per_class_f1": scfm.per_class_f1,
            "confusion": scfm.confusion,
        },
        class_counts={"train": _class_counts(y_train), "test": _class_counts(y_test)},
    )
    write_result(result, output)
    click.echo(
        f"{result['go_no_go']}: HVG/PCA macro-F1={baseline.macro_f1:.4f}; "
        f"frozen scFM macro-F1={scfm.macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
