"""
Geneformer TranscriptomeTokenizer for corneal subsampled data.

Tokenizes corneal_100k_subsample.h5ad and corneal_20k_test.h5ad
using Geneformer V2 rank-value encoding.

Pre-processing:
  - Sets .X = .raw.X (raw UMI counts required by Geneformer)
  - Computes n_counts per cell
  - Uses use_h5ad_index=True (ensembl_id in var.index)
  - Saves preprocessed copies to temp dirs to preserve originals
"""

import sys
import os
import shutil
import time
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
from geneformer import TranscriptomeTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === CONFIG ===
DATA_DIR = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
SCRIPT_DIR = PROJECT_ROOT / "scripts"
TRAIN_INPUT = DATA_DIR / "corneal_100k_subsample.h5ad"
TEST_INPUT = DATA_DIR / "corneal_20k_test.h5ad"
TRAIN_OUT = DATA_DIR / "gf_tokenized_train"
TEST_OUT = DATA_DIR / "gf_tokenized_test"

CUSTOM_ATTRS = {
    "cell_type": "cell_type",
    "disease": "disease",
    "tissue": "tissue",
    "donor_id": "donor_id",
}
NPROC = 4
CHUNK_SIZE = 512


def preprocess_h5ad(src_path: Path, dst_path: Path) -> dict:
    """
    Load h5ad, promote .raw.X to .X, compute n_counts, save preprocessed copy.
    Returns summary stats.
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing: {src_path.name}")
    print(f"{'='*60}")

    adata = ad.read_h5ad(str(src_path))

    # --- Check metadata columns ---
    for col in CUSTOM_ATTRS:
        assert col in adata.obs.columns, (
            f"Column '{col}' missing from obs columns: {list(adata.obs.columns)}"
        )

    # --- Promote raw counts to .X ---
    if adata.raw is not None:
        print(f"  Promoting .raw.X (raw counts) to .X")
        print(f"  .X before: type={type(adata.X)}, sum of cell 0 = {adata.X[0].sum():.1f}")
        adata.X = adata.raw.X.copy()
        adata.raw = None  # drop raw ref to avoid confusion
        print(f"  .X after:  type={type(adata.X)}, sum of cell 0 = {adata.X[0].sum():.1f}")
    else:
        print(f"  No .raw found; using existing .X as-is")

    # --- Compute n_counts ---
    sums = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs["n_counts"] = sums.astype(np.float32)
    print(f"  n_counts: min={sums.min():.0f}, max={sums.max():.0f}, "
          f"mean={sums.mean():.0f}, median={np.median(sums):.0f}")

    # Verify n_counts presence
    assert "n_counts" in adata.obs.columns, "n_counts not in obs after computation"

    # --- Save preprocessed copy ---
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(dst_path), compression="gzip")
    file_size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"  Saved preprocessed copy: {dst_path.name} ({file_size_mb:.1f} MB)")

    return {
        "cells": adata.shape[0],
        "genes": adata.shape[1],
        "mean_n_counts": float(sums.mean()),
        "file_size_mb": file_size_mb,
    }


def tokenize_dataset(input_h5ad: Path, output_dir: Path, prefix: str) -> dict:
    """
    Tokenize a single preprocessed h5ad file using TranscriptomeTokenizer.
    Uses a temp directory approach since tokenize_data reads from a directory.
    """
    print(f"\n{'='*60}")
    print(f"Tokenizing: {input_h5ad.name} → {output_dir} ({prefix})")
    print(f"{'='*60}")

    # Create temp dir with just this h5ad
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gf_tk_{prefix}_"))
    tmp_h5ad = tmp_dir / input_h5ad.name
    shutil.copy2(str(input_h5ad), str(tmp_h5ad))
    print(f"  Copied to temp dir: {tmp_dir}")

    # Initialize tokenizer
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=CUSTOM_ATTRS,
        nproc=NPROC,
        chunk_size=CHUNK_SIZE,
        model_version="V2",
        use_h5ad_index=True,
        collapse_gene_ids=True,
    )
    print(f"  Tokenizer initialized: model_version=V2, nproc={NPROC}, chunk_size={CHUNK_SIZE}")

    # Tokenize
    t_start = time.time()
    tk.tokenize_data(
        data_directory=str(tmp_dir),
        output_directory=str(output_dir),
        output_prefix=prefix,
        file_format="h5ad",
    )
    elapsed = time.time() - t_start

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Temp dir cleaned up")

    # Find output
    dataset_path = output_dir / f"{prefix}.dataset"
    assert dataset_path.exists(), f"Output {dataset_path} not found!"

    # Quick stats from the saved dataset
    from datasets import load_from_disk
    ds = load_from_disk(str(dataset_path))
    input_ids = ds["input_ids"]
    lengths = [len(ids) for ids in input_ids]
    token_count = sum(lengths)

    stats = {
        "cells": len(ds),
        "avg_tokens_per_cell": token_count / len(ds),
        "median_tokens_per_cell": float(np.median(lengths)),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "total_tokens": token_count,
        "elapsed_min": elapsed / 60,
        "dataset_path": str(dataset_path),
    }

    print(f"\n  ✅ Tokenized {stats['cells']} cells in {stats['elapsed_min']:.1f} min")
    print(f"     Avg tokens/cell: {stats['avg_tokens_per_cell']:.1f}")
    print(f"     Median tokens/cell: {stats['median_tokens_per_cell']:.1f}")
    print(f"     Total tokens: {stats['total_tokens']:,}")

    return stats


def verify_metadata_integrity(ds, name: str):
    """Verify all custom metadata columns exist and have no missing values."""
    print(f"\n  Metadata check for {name}:")
    for col in CUSTOM_ATTRS.values():
        values = ds[col]
        n_missing = sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))
        n_unique = len(set(v for v in values if v is not None))
        print(f"    {col}: {len(values)} entries, {n_unique} unique, {n_missing} missing")


def verify_dataset(dataset_path: Path, name: str):
    """Verify tokenized dataset integrity."""
    from datasets import load_from_disk

    print(f"\n{'='*60}")
    print(f"VERIFICATION: {name}")
    print(f"{'='*60}")

    ds = load_from_disk(str(dataset_path))
    print(f"  Dataset path: {dataset_path}")
    print(f"  Number of cells: {len(ds)}")
    print(f"  Features: {list(ds.features.keys())}")
    print(f"  Column names: {ds.column_names}")

    # Token stats
    input_ids = ds["input_ids"]
    lengths = np.array([len(ids) for ids in input_ids])
    print(f"  Token stats per cell:")
    print(f"    Min:     {lengths.min()}")
    print(f"    Max:     {lengths.max()}")
    print(f"    Mean:    {lengths.mean():.1f}")
    print(f"    Median:  {np.median(lengths):.1f}")
    print(f"    Std:     {lengths.std():.1f}")

    # Verify CLS and EOS tokens for V2
    cls_count = sum(1 for ids in input_ids if len(ids) > 0 and ids[0] == 1)
    eos_count = sum(1 for ids in input_ids if len(ids) > 1 and ids[-1] == 2)
    print(f"  Cells with CLS token (id=1): {cls_count}/{len(ds)}")
    print(f"  Cells with EOS token (id=2): {eos_count}/{len(ds)}")

    # Metadata verification
    verify_metadata_integrity(ds, name)

    return {"cells": len(ds), "mean_length": float(lengths.mean()), "total_tokens": int(lengths.sum())}


def main():
    print("=" * 60)
    print("GEFORMER TRANSCRIPTOME TOKENIZER — CORNEAL DATA")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Train input: {TRAIN_INPUT}")
    print(f"Test input:  {TEST_INPUT}")
    print(f"Train out:   {TRAIN_OUT}")
    print(f"Test out:    {TEST_OUT}")
    print(f"Nproc:       {NPROC}")
    print(f"Chunk size:  {CHUNK_SIZE}")
    print(f"Custom attrs: {CUSTOM_ATTRS}")
    print()

    # Validate inputs
    for f in [TRAIN_INPUT, TEST_INPUT]:
        assert f.exists(), f"Input file not found: {f}"
        assert f.stat().st_size > 0, f"Input file empty: {f}"

    # === Step 1: Preprocess train ===
    train_prep = DATA_DIR / "tmp_prep_train" / TRAIN_INPUT.name
    prep_stats_train = preprocess_h5ad(TRAIN_INPUT, train_prep)
    print(f"  Preprocessed: {prep_stats_train['cells']} cells, "
          f"{prep_stats_train['genes']} genes")

    # === Step 2: Tokenize train ===
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    train_stats = tokenize_dataset(train_prep, TRAIN_OUT, "corneal_100k")

    # === Step 3: Preprocess test ===
    test_prep = DATA_DIR / "tmp_prep_test" / TEST_INPUT.name
    prep_stats_test = preprocess_h5ad(TEST_INPUT, test_prep)
    print(f"  Preprocessed: {prep_stats_test['cells']} cells, "
          f"{prep_stats_test['genes']} genes")

    # === Step 4: Tokenize test ===
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    test_stats = tokenize_dataset(test_prep, TEST_OUT, "corneal_20k")

    # === Step 5: Clean up temp preprocessed files ===
    for p in [DATA_DIR / "tmp_prep_train", DATA_DIR / "tmp_prep_test"]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"  Cleaned up: {p}")

    # === Step 6: Verify ===
    train_v = verify_dataset(TRAIN_OUT / "corneal_100k.dataset", "TRAIN")
    test_v = verify_dataset(TEST_OUT / "corneal_20k.dataset", "TEST")

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {train_v['cells']} cells, "
          f"{train_v['mean_length']:.1f} avg tokens/cell, "
          f"{train_v['total_tokens']:,} total tokens")
    print(f"  Test:  {test_v['cells']} cells, "
          f"{test_v['mean_length']:.1f} avg tokens/cell, "
          f"{test_v['total_tokens']:,} total tokens")
    print(f"  Train path: {TRAIN_OUT / 'corneal_100k.dataset'}")
    print(f"  Test path:  {TEST_OUT / 'corneal_20k.dataset'}")
    print(f"  Metadata: {', '.join(CUSTOM_ATTRS.values())}")
    print(f"  Elapsed: {train_stats['elapsed_min'] + test_stats['elapsed_min']:.1f} min total")
    print("✅ DONE")


if __name__ == "__main__":
    main()
