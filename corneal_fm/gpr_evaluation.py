"""Provenance-bearing single and paired GPR evaluation interfaces."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from corneal_fm.gpr_embeddings import (
    dataset_identity,
    extract_embeddings,
    load_from_disk,
    load_model,
    require_layers,
    resolve_checkpoint,
)
from corneal_fm.gpr_contract import (
    EMBEDDING_SOURCE,
    comparison_contract,
    provenance,
    single_contract,
)
from corneal_fm.gpr_metrics import (
    CONTINUITY_FORMULA,
    LAYER_NAMES,
    NORMALIZATION_CONTRACT,
    N_LAYERS,
    MetricProvenance,
    continuity_score,
    layer_adjacency_accuracy,
    layer_centroids,
    measure_continuity,
    point_comparison,
)
from corneal_fm.gpr_resampling import (
    anisotropy_diagnostics,
    bootstrap_cs,
    paired_stratified_bootstrap,
)
from corneal_fm.gpr_schema import (
    ComparisonDiagnostics,
    ComparisonRequest,
    ComparisonResult,
    MetricContext,
    PrimaryResult,
    SensitivityResults,
    SingleResult,
)

type EmbeddingViews = tuple[NDArray[np.float32], NDArray[np.float32]]


def _summarize(
    embeddings: NDArray[np.float32],
    layers: NDArray[np.int64],
    provenance: MetricProvenance,
) -> SingleResult:
    centroids, masks = layer_centroids(embeddings, layers)
    present = sorted(masks)
    score, pair_detail, _ = continuity_score(centroids, present)
    adjacency = layer_adjacency_accuracy(centroids, present)
    adjacency_score = adjacency[0] if isinstance(adjacency, tuple) else adjacency
    return {
        "mode": "checkpoint"
        if provenance.embedding_source == EMBEDDING_SOURCE
        else "embedding",
        "source": provenance.checkpoint_path,
        "n_cells": int(len(layers)),
        "embedding_dim": int(embeddings.shape[1]),
        "layers_present": [LAYER_NAMES[layer] for layer in present],
        "per_layer_counts": {
            LAYER_NAMES[layer]: int((layers == layer).sum()) for layer in range(N_LAYERS)
        },
        "continuity_score": round(score, 4),
        "pair_similarity": pair_detail,
        "layer_adjacency_accuracy": round(adjacency_score, 4),
        "bootstrap_cs_95ci": bootstrap_cs(
            embeddings,
            layers,
            N_LAYERS,
            n_boot=provenance.n_boot,
            seed=provenance.seed,
        ),
        "contract": single_contract(provenance),
        "diagnostics": {"anisotropy": anisotropy_diagnostics(embeddings)},
    }


def evaluate_checkpoint(
    checkpoint: str,
    tokenized_path: str,
    num_classes: int = 20,
    n_boot: int = 1000,
    use_cls: bool = True,
    seed: int = 42,
) -> SingleResult:
    resolved_checkpoint = resolve_checkpoint(checkpoint)
    model = load_model(resolved_checkpoint, num_classes=num_classes)
    dataset = load_from_disk(tokenized_path)
    layers = require_layers(dataset, tokenized_path)
    embeddings = extract_embeddings(model, dataset, use_cls=use_cls)
    dataset_path, fingerprint = dataset_identity(dataset, tokenized_path)
    context = MetricContext(dataset_path, fingerprint, seed, n_boot)
    pooling = "raw_cls" if use_cls else "mean_pool"
    return _summarize(embeddings, layers, provenance(context, resolved_checkpoint, pooling))


def evaluate_embeddings(
    embeddings_path: str,
    layers_path: str,
    n_boot: int = 1000,
    label: str = "embedding",
) -> SingleResult:
    embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)
    layers = np.asarray(np.load(layers_path), dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(layers.tobytes())
    digest.update(str(embeddings.shape).encode())
    provenance = MetricProvenance(
        embedding_source="precomputed_npy_unverified",
        pooling="unknown_precomputed",
        formula=CONTINUITY_FORMULA,
        normalization="unverified",
        layer_order=tuple(LAYER_NAMES),
        dataset_fingerprint=f"sha256:{digest.hexdigest()}",
        dataset_path=str(Path(layers_path).expanduser().resolve()),
        checkpoint_path=label,
        seed=42,
        n_boot=n_boot,
    )
    return _summarize(embeddings, layers, provenance)


def _extract_checkpoint_views(checkpoint: str, dataset, num_classes: int) -> EmbeddingViews:
    model = load_model(checkpoint, num_classes=num_classes)
    raw_cls = extract_embeddings(model, dataset, use_cls=True)
    mean_pool = extract_embeddings(model, dataset, use_cls=False)
    return raw_cls, mean_pool


def compare_checkpoints(request: ComparisonRequest) -> ComparisonResult:
    dataset = load_from_disk(request.tokenized_path)
    layers = require_layers(dataset, request.tokenized_path)
    dataset_path, fingerprint = dataset_identity(dataset, request.tokenized_path)
    context = MetricContext(dataset_path, fingerprint, request.seed, request.n_boot)
    before_checkpoint = resolve_checkpoint(request.before_checkpoint)
    after_checkpoint = resolve_checkpoint(request.after_checkpoint)
    before_raw, before_mean = _extract_checkpoint_views(
        before_checkpoint,
        dataset,
        request.num_classes,
    )
    after_raw, after_mean = _extract_checkpoint_views(
        after_checkpoint,
        dataset,
        request.num_classes,
    )
    before_raw_provenance = provenance(context, before_checkpoint, "raw_cls")
    after_raw_provenance = provenance(context, after_checkpoint, "raw_cls")
    primary: PrimaryResult = {
        "metric_name": "raw_cls_same_space_gpr",
        "ceiling_sensitive": True,
        **point_comparison(
            measure_continuity(after_raw, layers, after_raw_provenance),
            measure_continuity(before_raw, layers, before_raw_provenance),
        ),
        "paired_stratified_bootstrap_95ci": paired_stratified_bootstrap(
            before_raw,
            after_raw,
            layers,
            n_boot=request.n_boot,
            seed=request.seed,
        ),
    }
    before_mean_provenance = provenance(context, before_checkpoint, "mean_pool")
    after_mean_provenance = provenance(context, after_checkpoint, "mean_pool")
    sensitivity: SensitivityResults = {
        "mean_pool": {
            "metric_name": "mean_pool_same_space_gpr_sensitivity",
            **point_comparison(
                measure_continuity(after_mean, layers, after_mean_provenance),
                measure_continuity(before_mean, layers, before_mean_provenance),
            ),
        },
    }
    diagnostics: ComparisonDiagnostics = {
        "anisotropy": {
            "raw_cls": {
                "before": anisotropy_diagnostics(before_raw),
                "after": anisotropy_diagnostics(after_raw),
            },
            "mean_pool": {
                "before": anisotropy_diagnostics(before_mean),
                "after": anisotropy_diagnostics(after_mean),
            },
        },
    }
    return {
        "contract": comparison_contract(before_raw_provenance, after_raw_provenance),
        "primary": primary,
        "sensitivity": sensitivity,
        "diagnostics": diagnostics,
    }
