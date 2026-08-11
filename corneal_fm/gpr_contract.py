"""Provenance factory and JSON-serialisable contract builders for GPR results."""

from __future__ import annotations

from corneal_fm.gpr_metrics import (
    CONTINUITY_FORMULA,
    LAYER_NAMES,
    NORMALIZATION_CONTRACT,
    MetricProvenance,
)
from corneal_fm.gpr_schema import (
    ComparisonContract,
    MetricContext,
    SingleContract,
)

EMBEDDING_SOURCE = "geneformer.last_hidden_state"


def provenance(
    context: MetricContext,
    checkpoint_path: str,
    pooling: str,
) -> MetricProvenance:
    return MetricProvenance(
        embedding_source=EMBEDDING_SOURCE,
        pooling=pooling,
        formula=CONTINUITY_FORMULA,
        normalization=NORMALIZATION_CONTRACT,
        layer_order=tuple(LAYER_NAMES),
        dataset_fingerprint=context.dataset_fingerprint,
        dataset_path=context.dataset_path,
        checkpoint_path=checkpoint_path,
        seed=context.seed,
        n_boot=context.n_boot,
    )


def single_contract(provenance: MetricProvenance) -> SingleContract:
    return {
        "embedding": {
            "source": provenance.embedding_source,
            "pooling": provenance.pooling,
            "normalization": provenance.normalization,
        },
        "metric": {
            "formula": provenance.formula,
            "layer_order": list(provenance.layer_order),
        },
        "dataset": {
            "path": provenance.dataset_path,
            "fingerprint": provenance.dataset_fingerprint,
        },
        "checkpoints": {"measured": provenance.checkpoint_path},
        "resampling": {"seed": provenance.seed, "n_boot": provenance.n_boot},
    }


def comparison_contract(
    before: MetricProvenance,
    after: MetricProvenance,
) -> ComparisonContract:
    return {
        "embedding": {
            "source": before.embedding_source,
            "pooling": before.pooling,
            "normalization": before.normalization,
        },
        "metric": {
            "formula": before.formula,
            "layer_order": list(before.layer_order),
        },
        "dataset": {
            "path": before.dataset_path,
            "fingerprint": before.dataset_fingerprint,
        },
        "checkpoints": {
            "before": before.checkpoint_path,
            "after": after.checkpoint_path,
        },
        "resampling": {
            "method": "paired_stratified_percentile",
            "seed": before.seed,
            "n_boot": before.n_boot,
        },
    }
