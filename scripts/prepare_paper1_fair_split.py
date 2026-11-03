#!/usr/bin/env python
"""Prepare the metadata-only this study fairness contract.

This command deliberately does *not* write an H5AD file or tokenize cells.  It
only reads ``obs`` from the two donor-split H5ADs in backed mode and writes a
small, auditable contract for subsequent training/evaluation commands.

The output directory is create-once by default.  This is intentional: a
manifest is provenance, not a cache that can be silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = PROJECT_ROOT / "data/corneal_100k_donor_split.h5ad"
DEFAULT_TEST = PROJECT_ROOT / "data/corneal_20k_donor_split.h5ad"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/paper1_fair_split"
DEFAULT_SEED = 42


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_mask(values: Any) -> Any:
    """Return a pandas-compatible mask for missing donor identifiers."""
    import pandas as pd

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isin(("", "nan", "none", "null", "<na>"))


def _donor_set(obs: Any) -> set[str]:
    """Normalized donor identifiers (strip + lower) for overlap detection.

    Mirrors the ``_missing_mask`` normalization so case/whitespace-only donor
    differences are still detected as overlap (F2): ``"Donor_01"`` and
    ``" donor_01 "`` are the same donor and must trip the disjointness guard.
    """
    import pandas as pd

    values = obs["donor_id"]
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    normalized = series.astype("string").str.strip().str.lower().dropna()
    return set(normalized.tolist())


def _stable_counts(values: Any) -> dict[str, int]:
    import pandas as pd

    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return {str(key): int(value) for key, value in series.astype(str).value_counts().sort_index().items()}


def _coverage(train_obs: Any, test_obs: Any, column: str) -> dict[str, Any]:
    train_counts = _stable_counts(train_obs[column])
    test_counts = _stable_counts(test_obs[column])
    labels = sorted(set(train_counts) | set(test_counts))
    missing_train = [label for label in labels if train_counts.get(label, 0) == 0]
    missing_test = [label for label in labels if test_counts.get(label, 0) == 0]
    return {
        "column": column,
        "n_labels": len(labels),
        "labels": labels,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "missing_from_train": missing_train,
        "missing_from_test": missing_test,
        "complete": not missing_train and not missing_test,
    }


def validate_metadata(
    train_obs: Any,
    test_obs: Any,
    *,
    allow_missing_test_labels: bool = False,
) -> dict[str, Any]:
    """Validate donor and label metadata without touching an expression matrix.

    ``train_obs`` and ``test_obs`` are intentionally accepted as DataFrames so
    this contract can be tested with tiny metadata fixtures.

    Donor identifiers are normalized (strip + lower) before the disjointness
    check, so case/whitespace-only differences cannot hide a shared donor (F2).

    ``allow_missing_test_labels`` (F6) controls the label-coverage policy.  By
    default (``False``) it is a hard error for any label present in one split to
    be absent from the other.  Passing ``True`` relaxes this *only* for labels
    missing from the test split (e.g. a rare class deliberately held out of the
    held-out-batch test); labels missing from the train split remain a hard
    error because they can never be learned.  The strict default is intentional
    and must not be changed silently.
    """
    required = "donor_id"
    for name, obs in (("train", train_obs), ("test", test_obs)):
        if required not in obs.columns:
            raise ValueError(f"{name} metadata must contain '{required}'")
        missing = int(_missing_mask(obs[required]).sum())
        if missing:
            raise ValueError(f"{name} metadata has {missing} missing donor_id value(s)")

    label_columns = [column for column in ("cell_type", "author_cell_type")
                     if column in train_obs.columns and column in test_obs.columns]
    if not label_columns:
        raise ValueError("both metadata tables must contain cell_type or author_cell_type")

    train_donors = _donor_set(train_obs)
    test_donors = _donor_set(test_obs)
    overlap = sorted(train_donors & test_donors)
    if overlap:
        raise ValueError(f"train/test donor overlap is non-zero: {overlap[:10]}")

    coverage = {column: _coverage(train_obs, test_obs, column) for column in label_columns}
    if allow_missing_test_labels:
        incomplete = [column for column, result in coverage.items() if result["missing_from_train"]]
    else:
        incomplete = [column for column, result in coverage.items() if not result["complete"]]
    if incomplete:
        details = {column: coverage[column] for column in incomplete}
        raise ValueError(f"cell type coverage is incomplete: {details}")

    return {
        "donor_overlap": 0,
        "donor_overlap_count": 0,
        "train_donors": len(train_donors),
        "test_donors": len(test_donors),
        "cell_type_coverage": coverage,
        "label_columns": label_columns,
    }


def _layer_mapping(author_cell_types: list[str]) -> dict[str, Any]:
    # Import lazily: metadata preparation must remain usable without importing
    # the training stack until it is actually needed.
    try:
        from corneal_fm.caft import CELL_TYPE_TO_LAYER, LAYER_ORDER, NON_LAYER
    except ImportError as exc:  # pragma: no cover - only for incomplete installs
        raise RuntimeError("corneal_fm.caft is required to record the layer mapping") from exc
    mapping = {name: int(CELL_TYPE_TO_LAYER.get(name, NON_LAYER)) for name in author_cell_types}
    return {
        "source": "corneal_fm.caft.CELL_TYPE_TO_LAYER",
        "layer_order": list(LAYER_ORDER),
        "non_layer": int(NON_LAYER),
        "author_cell_type_to_layer": mapping,
    }


def _read_obs(path: Path) -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - dependency is in requirements-core
        raise RuntimeError("anndata is required to inspect H5AD metadata") from exc
    if not path.is_file():
        raise FileNotFoundError(f"input H5AD not found: {path}")
    handle = ad.read_h5ad(str(path), backed="r")
    try:
        # Copy obs only.  No X/raw layer is loaded or written by this command.
        return handle.obs.copy()
    finally:
        handle.file.close()


def build_manifest(train_h5ad: Path, test_h5ad: Path, seed: int = DEFAULT_SEED) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build manifest/check data from metadata and input file provenance."""
    train_h5ad = train_h5ad.resolve()
    test_h5ad = test_h5ad.resolve()
    train_obs = _read_obs(train_h5ad)
    test_obs = _read_obs(test_h5ad)
    checks = validate_metadata(train_obs, test_obs)

    author_types = sorted(set(train_obs.get("author_cell_type", train_obs["cell_type"]).astype(str)) |
                          set(test_obs.get("author_cell_type", test_obs["cell_type"]).astype(str)))
    input_paths = {"train_h5ad": str(train_h5ad), "test_h5ad": str(test_h5ad)}
    input_checksums = {key: _sha256(Path(value)) for key, value in input_paths.items()}
    counts = {
        "train_cells": int(len(train_obs)),
        "test_cells": int(len(test_obs)),
        "total_cells": int(len(train_obs) + len(test_obs)),
        "train_donors": checks["train_donors"],
        "test_donors": checks["test_donors"],
    }
    donor_check = {
        "train_donors": checks["train_donors"],
        "test_donors": checks["test_donors"],
        "overlap_count": 0,
        "overlap": [],
        "passed": True,
    }
    manifest = {
        "protocol": "paper1_fair",
        "manifest_version": 1,
        "mode": "strict_fair",
        "seed": int(seed),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": input_paths,
        "input_checksums": input_checksums,
        # Keep a short, obvious alias for consumers that use the generic term.
        "checksums": input_checksums,
        "counts": counts,
        "donor_overlap": 0,
        "donor_overlap_count": 0,
        "cell_type_coverage": checks["cell_type_coverage"],
        "layer_mapping": _layer_mapping(author_types),
        "metadata_schema": {
            "donor_id": "donor_id",
            "label_columns": checks["label_columns"],
            "matrix_access": "metadata-only/backed; no tokenization",
        },
    }
    return manifest, donor_check


def _write_json(path: Path, payload: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare(train_h5ad: Path, test_h5ad: Path, output_dir: Path, seed: int, force: bool = False) -> None:
    if output_dir.exists() and not force:
        raise FileExistsError(f"output directory already exists; use --force explicitly: {output_dir}")
    manifest, donor_check = build_manifest(train_h5ad, test_h5ad, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", manifest, force)
    _write_json(output_dir / "donor_overlap_check.json", donor_check, force)
    print(f"Wrote metadata-only this study fair manifest: {output_dir / 'manifest.json'}")
    print("No H5AD was modified and no tokenization was performed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the this study donor-disjoint fairness manifest")
    parser.add_argument("--train_h5ad", "--train-h5ad", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test_h5ad", "--test-h5ad", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Allow replacing manifest JSON in output_dir")
    args = parser.parse_args(argv)
    try:
        prepare(args.train_h5ad, args.test_h5ad, args.output_dir, args.seed, args.force)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
