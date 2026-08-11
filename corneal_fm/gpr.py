"""GPR / Continuity-Score measurement for fine-tuned scFMs (Methods §4.4).

Loads a fine-tuned Geneformer checkpoint, extracts the [CLS]-equivalent cell
embedding, assigns cells to differentiation layers, and computes:

  CS  = mean( cos(centroid_layer_i, centroid_layer_{i+1}) )        i in [1, N-1]
  LAA = fraction of adjacent-layer pairs whose centroid distance is smaller
        than the median distance to random non-adjacent pairs
  GPR = CS_after / CS_before            (>1 => preservation/improvement)

Bootstrap 95% CI is obtained by resampling cells within each layer (1000
resamples, percentile method), per Methods §4.6.

Two usage modes:
  * model mode:  --checkpoint <dir> --tokenized <labeled.dataset>
                 (extracts embeddings from a fine-tuned checkpoint)
  * embedding mode: --embeddings <.npy> --layers <.npy>
                 (reuses a precomputed [N, D] embedding matrix + per-cell layer ids)
"""

from __future__ import annotations

from corneal_fm.gpr_embeddings import extract_embeddings, load_from_disk, load_model
from corneal_fm.gpr_evaluation import (
    compare_checkpoints,
    evaluate_checkpoint,
    evaluate_embeddings,
)
from corneal_fm.gpr_metrics import (
    CONTINUITY_FORMULA,
    LAYER_NAMES,
    NORMALIZATION_CONTRACT,
    IncompatibleMetricSpaceError,
    MetricProvenance,
    compute_gpr,
    continuity_score,
    layer_adjacency_accuracy,
    layer_centroids,
    measure_continuity,
)
from corneal_fm.gpr_resampling import bootstrap_cs
from corneal_fm.gpr_schema import ComparisonRequest

__all__ = [
    "CONTINUITY_FORMULA",
    "LAYER_NAMES",
    "NORMALIZATION_CONTRACT",
    "ComparisonRequest",
    "IncompatibleMetricSpaceError",
    "MetricProvenance",
    "bootstrap_cs",
    "compare_checkpoints",
    "compute_gpr",
    "continuity_score",
    "evaluate_checkpoint",
    "evaluate_embeddings",
    "extract_embeddings",
    "layer_adjacency_accuracy",
    "layer_centroids",
    "load_from_disk",
    "load_model",
    "measure_continuity",
]


def main() -> None:
    from corneal_fm.gpr_cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
