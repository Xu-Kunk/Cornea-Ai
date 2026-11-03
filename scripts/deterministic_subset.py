#!/usr/bin/env python
"""Deterministic 100-cell subset generator: 20 cells × 5 corneal epithelial layers.

Derives 'layer' from author_cell_type via the ratified CELL_TYPE_TO_LAYER mapping
(corneal_fm/caft.py §4.4 Method B), samples deterministically with seed=42, and
preserves author_cell_type, raw counts, n_counts, and all tokenization prerequisites.

Usage:
    python scripts/deterministic_subset.py \
        --src artifacts/pipeline-20260723/S6/corneal_30k_layer_subsample.h5ad \
        --out /tmp/smoke_100cells.h5ad \
        [--seed 42]
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corneal_fm.caft import CELL_TYPE_TO_LAYER, LAYER_ORDER, NON_LAYER


def generate_subset(src_path: str, out_path: str, seed: int = 42) -> dict:
    """Generate a deterministic 100-cell subset: 20 cells per epithelial layer.

    Returns provenance dict with per-layer cell counts.
    """
    rng = np.random.default_rng(seed)

    print(f"Loading {src_path} ...", flush=True)
    a = ad.read_h5ad(src_path, backed="r")
    print(f"  Shape: {a.n_obs} cells × {a.n_vars} genes", flush=True)

    if "author_cell_type" not in a.obs.columns:
        raise ValueError("source h5ad must have 'author_cell_type' column")

    # Map each cell to layer index
    cell_types = a.obs["author_cell_type"].astype(str).values
    layers = np.array(
        [CELL_TYPE_TO_LAYER.get(ct, NON_LAYER) for ct in cell_types],
        dtype=np.int64,
    )

    # For each LAYER_ORDER index, sample exactly 20 cells
    selected_indices: list[int] = []
    per_layer_counts: dict[str, int] = {}

    for i, name in enumerate(LAYER_ORDER):
        candidate_indices = np.where(layers == i)[0]
        n_available = len(candidate_indices)
        if n_available < 20:
            raise ValueError(
                f"Layer {name} ({i}) has only {n_available} cells; need ≥20"
            )
        chosen = rng.choice(candidate_indices, size=20, replace=False)
        selected_indices.extend(chosen.tolist())
        per_layer_counts[name] = 20
        print(
            f"  {name}: sampled 20 from {n_available} available cells",
            flush=True,
        )

    selected_indices = sorted(selected_indices)
    assert len(selected_indices) == 100, (
        f"Expected 100 cells, got {len(selected_indices)}"
    )

    # Materialise the subset (backed→memory)
    print(f"Materialising {len(selected_indices)} cells ...", flush=True)
    sub = a[selected_indices].to_memory()

    # Preserve raw counts
    if sub.raw is not None:
        sub.X = sub.raw.X.copy()
        sub.raw = None
        print("  Using raw counts (sub.raw → sub.X)", flush=True)

    # Ensure n_counts column (tokenization prerequisite)
    if "n_counts" not in sub.obs.columns:
        sub.obs["n_counts"] = np.asarray(sub.X.sum(axis=1)).ravel()

    # Verify prerequisites
    assert sub.n_obs == 100, f"Output has {sub.n_obs} cells, expected 100"
    assert "author_cell_type" in sub.obs.columns
    assert "n_counts" in sub.obs.columns
    for name in LAYER_ORDER:
        count = int(sub.obs["author_cell_type"].map(
            lambda ct: 1 if CELL_TYPE_TO_LAYER.get(ct, NON_LAYER) == LAYER_ORDER.index(name) else 0
        ).sum())
        assert count == 20, f"Layer {name} has {count} cells, expected 20"

    # Write output
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(str(out), compression="gzip")
    print(f"Wrote {out} ({sub.n_obs} cells × {sub.n_vars} genes)", flush=True)

    provenance = {
        "src": str(src_path),
        "out": str(out),
        "cells": 100,
        "genes": int(sub.n_vars),
        "seed": seed,
        "per_layer_counts": per_layer_counts,
        "layer_order": LAYER_ORDER,
    }
    return provenance


@click.command()
@click.option("--src", required=True, type=click.Path(exists=True),
              help="Source h5ad with author_cell_type column.")
@click.option("--out", required=True, type=click.Path(),
              help="Output h5ad path (will be created).")
@click.option("--seed", default=42, type=int, help="Random seed.")
def main(src: str, out: str, seed: int) -> None:
    """Generate a deterministic 100-cell subset for smoke validation."""
    provenance = generate_subset(src, out, seed)
    print(f"Provenance: {provenance}")


if __name__ == "__main__":
    main()
