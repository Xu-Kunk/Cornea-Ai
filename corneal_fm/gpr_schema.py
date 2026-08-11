"""JSON-compatible result schemas for GPR measurement and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from corneal_fm.gpr_metrics import PointComparisonResult
from corneal_fm.gpr_resampling import PairedBootstrapResult


@dataclass(frozen=True, slots=True)
class ComparisonRequest:
    before_checkpoint: str
    after_checkpoint: str
    tokenized_path: str
    num_classes: int = 20
    seed: int = 42
    n_boot: int = 1000


@dataclass(frozen=True, slots=True)
class MetricContext:
    dataset_path: str
    dataset_fingerprint: str
    seed: int
    n_boot: int


class EmbeddingContract(TypedDict):
    source: str
    pooling: str
    normalization: str


class MetricContract(TypedDict):
    formula: str
    layer_order: list[str]


class DatasetContract(TypedDict):
    path: str
    fingerprint: str


class SingleResamplingContract(TypedDict):
    seed: int
    n_boot: int


class PairedResamplingContract(SingleResamplingContract):
    method: str


class SingleContract(TypedDict):
    embedding: EmbeddingContract
    metric: MetricContract
    dataset: DatasetContract
    checkpoints: dict[str, str]
    resampling: SingleResamplingContract


class ComparisonContract(TypedDict):
    embedding: EmbeddingContract
    metric: MetricContract
    dataset: DatasetContract
    checkpoints: dict[str, str]
    resampling: PairedResamplingContract


class SingleDiagnostics(TypedDict):
    anisotropy: dict[str, float]


class SingleResult(TypedDict):
    mode: str
    source: str
    n_cells: int
    embedding_dim: int
    layers_present: list[str]
    per_layer_counts: dict[str, int]
    continuity_score: float
    pair_similarity: dict[str, float]
    layer_adjacency_accuracy: float
    bootstrap_cs_95ci: dict[str, float]
    contract: SingleContract
    diagnostics: SingleDiagnostics


class PrimaryResult(PointComparisonResult):
    metric_name: str
    ceiling_sensitive: bool
    paired_stratified_bootstrap_95ci: PairedBootstrapResult


class SensitivityResult(PointComparisonResult):
    metric_name: str


class SensitivityResults(TypedDict):
    mean_pool: SensitivityResult


class AnisotropyPair(TypedDict):
    before: dict[str, float]
    after: dict[str, float]


class ComparisonAnisotropy(TypedDict):
    raw_cls: AnisotropyPair
    mean_pool: AnisotropyPair


class ComparisonDiagnostics(TypedDict):
    anisotropy: ComparisonAnisotropy


class ComparisonResult(TypedDict):
    contract: ComparisonContract
    primary: PrimaryResult
    sensitivity: SensitivityResults
    diagnostics: ComparisonDiagnostics
