"""Bootstrap uncertainty and linear-memory anisotropy diagnostics for GPR."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from corneal_fm.gpr_metrics import (
    N_LAYERS,
    IncompatibleMetricSpaceError,
    continuity_score,
    layer_centroids,
)


class ConfidenceInterval(TypedDict):
    ci_low: float
    ci_high: float


class PairedBootstrapResult(TypedDict):
    cs_before: ConfidenceInterval
    cs_after: ConfidenceInterval
    gpr: ConfidenceInterval
    absolute_delta: ConfidenceInterval


def bootstrap_cs(
    embeddings: NDArray[np.float32],
    layers: NDArray[np.int64],
    n_layers: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    present = [layer for layer in range(n_layers) if (layers == layer).any()]
    samples = np.empty(n_boot, dtype=np.float64)
    for bootstrap_index in range(n_boot):
        sampled_indices = rng.integers(0, len(layers), size=len(layers))
        sampled_layers = layers[sampled_indices]
        centroids, _ = layer_centroids(embeddings[sampled_indices], sampled_layers, n_layers)
        sampled_present = [layer for layer in present if (sampled_layers == layer).any()]
        samples[bootstrap_index] = continuity_score(centroids, sampled_present)[0]
    return {
        "mean": float(samples.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def paired_stratified_bootstrap(
    before_embeddings: NDArray[np.float32],
    after_embeddings: NDArray[np.float32],
    layers: NDArray[np.int64],
    *,
    n_boot: int,
    seed: int,
) -> PairedBootstrapResult:
    if before_embeddings.shape != after_embeddings.shape:
        raise IncompatibleMetricSpaceError("before/after embedding shapes differ")
    rng = np.random.default_rng(seed)
    layer_indices = [np.flatnonzero(layers == layer) for layer in range(N_LAYERS)]
    layer_indices = [indices for indices in layer_indices if indices.size]
    samples = np.empty((n_boot, 4), dtype=np.float64)
    for bootstrap_index in range(n_boot):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in layer_indices],
        )
        before_centroids, before_masks = layer_centroids(before_embeddings[sampled], layers[sampled])
        after_centroids, after_masks = layer_centroids(after_embeddings[sampled], layers[sampled])
        before_score = continuity_score(before_centroids, sorted(before_masks))[0]
        after_score = continuity_score(after_centroids, sorted(after_masks))[0]
        samples[bootstrap_index] = (
            before_score,
            after_score,
            after_score / before_score,
            after_score - before_score,
        )

    def interval(values: NDArray[np.float64]) -> ConfidenceInterval:
        return {
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }

    return {
        "cs_before": interval(samples[:, 0]),
        "cs_after": interval(samples[:, 1]),
        "gpr": interval(samples[:, 2]),
        "absolute_delta": interval(samples[:, 3]),
    }


def anisotropy_diagnostics(
    embeddings: NDArray[np.float32],
) -> dict[str, float]:
    norms = np.linalg.norm(embeddings, axis=1)
    valid = norms > 0
    unit_embeddings = embeddings[valid] / norms[valid, None]
    count = len(unit_embeddings)
    if count == 0:
        return {"mean_resultant_length": float("nan"), "mean_pairwise_cosine": float("nan")}
    vector_sum = unit_embeddings.sum(axis=0)
    resultant_length = float(np.linalg.norm(vector_sum) / count)
    pairwise_cosine = (
        float((np.dot(vector_sum, vector_sum) - count) / (count * (count - 1)))
        if count > 1
        else float("nan")
    )
    return {
        "mean_resultant_length": resultant_length,
        "mean_pairwise_cosine": float(np.clip(pairwise_cosine, -1.0, 1.0)),
    }
