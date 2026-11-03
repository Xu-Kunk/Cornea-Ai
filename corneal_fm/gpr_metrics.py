"""Typed same-space contracts and CPU-only metrics for GPR comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, assert_never

import numpy as np
from numpy.typing import NDArray

LAYER_NAMES = ["LSC", "TAC", "basal", "wing", "squamous"]
N_LAYERS = len(LAYER_NAMES)
CONTINUITY_FORMULA = (
    "CS=mean(cos(unit(mean(unit(cell_embeddings[layer_i]))),"
    "unit(mean(unit(cell_embeddings[layer_i+1])))));GPR=CS_after/CS_before"
)
NORMALIZATION_CONTRACT = "per_cell_l2_then_layer_centroid_mean_then_l2"


class GprResult(TypedDict):
    cs_before: float
    cs_after: float
    gpr: float
    absolute_delta: float
    interpretation: str


class PointComparisonResult(GprResult):
    pair_similarity_before: dict[str, float]
    pair_similarity_after: dict[str, float]
    per_adjacent_pair_delta: dict[str, float]


@dataclass(frozen=True, slots=True)
class MetricProvenance:
    embedding_source: str
    pooling: str
    formula: str
    normalization: str
    layer_order: tuple[str, ...]
    dataset_fingerprint: str
    dataset_path: str
    checkpoint_path: str
    seed: int
    n_boot: int


@dataclass(frozen=True, slots=True)
class ContinuityMeasurement:
    score: float
    pair_similarity: tuple[tuple[str, float], ...]
    provenance: MetricProvenance


@dataclass(frozen=True, slots=True)
class IncompatibleMetricSpaceError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


type LegacyArtifactValue = str | int | float | bool | None
type GprInput = ContinuityMeasurement | int | float | Mapping[str, LegacyArtifactValue]


def layer_centroids(
    embeddings: NDArray[np.float32],
    layers: NDArray[np.int64],
    n_layers: int = N_LAYERS,
) -> tuple[NDArray[np.float32], dict[int, NDArray[np.bool_]]]:
    dimension = embeddings.shape[1]
    centroids = np.zeros((n_layers, dimension), dtype=np.float32)
    masks: dict[int, NDArray[np.bool_]] = {}
    for layer in range(n_layers):
        mask = layers == layer
        if mask.any():
            centroid = embeddings[mask].mean(0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[layer] = centroid
            masks[layer] = mask
    return centroids, masks


def continuity_score(
    centroids: NDArray[np.float32],
    present: list[int],
) -> tuple[float, dict[str, float], list[float]]:
    ordered = sorted(present)
    similarities: list[float] = []
    pair_detail: dict[str, float] = {}
    for index in range(len(ordered) - 1):
        first, second = ordered[index], ordered[index + 1]
        similarity = float(np.clip(np.dot(centroids[first], centroids[second]), -1.0, 1.0))
        similarities.append(similarity)
        pair_detail[f"{LAYER_NAMES[first]}->{LAYER_NAMES[second]}"] = round(similarity, 4)
    score = float(np.mean(similarities)) if similarities else 0.0
    return score, pair_detail, similarities


def layer_adjacency_accuracy(
    centroids: NDArray[np.float32],
    present: list[int],
) -> float | tuple[float, float, list[float]]:
    ordered = sorted(present)
    if len(ordered) < 3:
        return float("nan")
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    adjacent = np.array(
        [distances[ordered[index], ordered[index + 1]] for index in range(len(ordered) - 1)],
    )
    non_adjacent = np.array(
        [
            distances[ordered[first], ordered[second]]
            for first in range(len(ordered))
            for second in range(first + 2, len(ordered))
        ],
    )
    if non_adjacent.size == 0:
        return float("nan")
    median_non_adjacent = float(np.median(non_adjacent))
    return (
        float((adjacent < median_non_adjacent).mean()),
        median_non_adjacent,
        adjacent.tolist(),
    )


def measure_continuity(
    embeddings: NDArray[np.float32],
    layers: NDArray[np.int64],
    provenance: MetricProvenance,
) -> ContinuityMeasurement:
    centroids, masks = layer_centroids(embeddings, layers, len(provenance.layer_order))
    score, pairs, _ = continuity_score(centroids, sorted(masks))
    return ContinuityMeasurement(
        score=score,
        pair_similarity=tuple(pairs.items()),
        provenance=provenance,
    )


def _require_measurement(value: GprInput, role: str) -> ContinuityMeasurement:
    match value:
        case ContinuityMeasurement():
            return value
        case bool() | int() | float():
            raise IncompatibleMetricSpaceError(
                f"{role} is a bare scalar; use paired checkpoint comparison with provenance",
            )
        case Mapping() as artifact:
            embedding = str(artifact.get("embedding", "unknown"))
            detail = "PCA artifact" if embedding.upper() == "PCA" else "untyped JSON artifact"
            raise IncompatibleMetricSpaceError(
                f"{role} is a legacy {detail}; GPR requires typed same-space measurements",
            )
        case unreachable:
            assert_never(unreachable)


def _space_mismatches(
    after: MetricProvenance,
    before: MetricProvenance,
) -> list[str]:
    mismatches: list[str] = []
    if after.embedding_source != before.embedding_source:
        mismatches.append("embedding_source")
    if after.pooling != before.pooling:
        mismatches.append("pooling")
    if after.formula != before.formula:
        mismatches.append("formula")
    if after.normalization != before.normalization:
        mismatches.append("normalization")
    if after.layer_order != before.layer_order:
        mismatches.append("layer_order")
    if after.dataset_fingerprint != before.dataset_fingerprint:
        mismatches.append("dataset_fingerprint")
    if after.dataset_path != before.dataset_path:
        mismatches.append("dataset_path")
    return mismatches


def compute_gpr(cs_after: GprInput, cs_before: GprInput) -> GprResult:
    before = _require_measurement(cs_before, "CS_before")
    after = _require_measurement(cs_after, "CS_after")
    mismatches = _space_mismatches(after.provenance, before.provenance)
    if mismatches:
        raise IncompatibleMetricSpaceError(
            f"same-space contract mismatch: {', '.join(mismatches)}",
        )
    if before.score == 0:
        raise IncompatibleMetricSpaceError("CS_before is zero; GPR is undefined")
    ratio = after.score / before.score
    return {
        "cs_before": before.score,
        "cs_after": after.score,
        "gpr": ratio,
        "absolute_delta": after.score - before.score,
        "interpretation": "preserved/improved" if ratio >= 1.0 else "degraded",
    }


def point_comparison(
    after: ContinuityMeasurement,
    before: ContinuityMeasurement,
) -> PointComparisonResult:
    result = compute_gpr(after, before)
    before_pairs = dict(before.pair_similarity)
    after_pairs = dict(after.pair_similarity)
    if before_pairs.keys() != after_pairs.keys():
        raise IncompatibleMetricSpaceError("adjacent layer pairs differ between measurements")
    return {
        "cs_before": result["cs_before"],
        "cs_after": result["cs_after"],
        "gpr": result["gpr"],
        "absolute_delta": result["absolute_delta"],
        "interpretation": result["interpretation"],
        "pair_similarity_before": before_pairs,
        "pair_similarity_after": after_pairs,
        "per_adjacent_pair_delta": {
            pair: after_pairs[pair] - before_pairs[pair] for pair in before_pairs
        },
    }
