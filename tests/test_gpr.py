"""Known-answer contract tests for GPR continuity measurements."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from corneal_fm import gpr


@pytest.fixture(autouse=True)
def _injected_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED harness: enabled only by explicit subprocess commands used for evidence."""
    fault = os.environ.get("GPR_INJECT_FAULT")
    if fault == "continuity":
        monkeypatch.setattr(gpr, "continuity_score", lambda centroids, present: (0.0, {}, [0.0]))
    elif fault == "normalization":
        real_centroids = gpr.layer_centroids
        monkeypatch.setattr(gpr, "layer_centroids", lambda *args, **kwargs: (real_centroids(*args, **kwargs)[0] * 2, real_centroids(*args, **kwargs)[1]))
    elif fault == "ratio":
        monkeypatch.setattr(gpr, "compute_gpr", lambda after, before: {"gpr": 1.0})


def test_continuity_score_matches_known_adjacent_cosines() -> None:
    angles = np.deg2rad([0, 60, 120, 180, 240])
    centroids = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    result: object = gpr.continuity_score(centroids, [0, 1, 2, 3, 4])
    assert isinstance(result, tuple)
    score, detail, similarities = result

    assert score == pytest.approx(0.5, abs=1e-7)
    assert similarities == pytest.approx([0.5] * 4, abs=1e-7)
    assert set(detail) == {"LSC->TAC", "TAC->basal", "basal->wing", "wing->squamous"}


def test_layer_centroids_normalize_present_layers_and_leave_absent_zero() -> None:
    embeddings = np.array([[3, 0], [1, 0], [0, 5], [0, 2]], dtype=np.float32)
    layers = np.array([0, 0, 2, 2])
    centroids, masks = gpr.layer_centroids(embeddings, layers)

    assert np.linalg.norm(centroids[0]) == pytest.approx(1.0)
    assert np.linalg.norm(centroids[2]) == pytest.approx(1.0)
    assert np.array_equal(centroids[[1, 3, 4]], np.zeros((3, 2), dtype=np.float32))
    assert set(masks) == {0, 2}


def _provenance(
    *,
    pooling: str = "raw_cls",
    checkpoint: str = "/models/before",
) -> gpr.MetricProvenance:
    return gpr.MetricProvenance(
        embedding_source="geneformer.last_hidden_state",
        pooling=pooling,
        formula=gpr.CONTINUITY_FORMULA,
        normalization=gpr.NORMALIZATION_CONTRACT,
        layer_order=tuple(gpr.LAYER_NAMES),
        dataset_fingerprint="huggingface:fixture-v1",
        dataset_path="/data/cells.dataset",
        checkpoint_path=checkpoint,
        seed=17,
        n_boot=20,
    )


@pytest.mark.parametrize(
    ("legacy_before", "message"),
    [
        (0.5477, "bare scalar"),
        (
            {"embedding": "PCA", "n_pcs": 50, "continuity_score": 0.5477},
            "PCA",
        ),
    ],
)
def test_compute_gpr_rejects_legacy_baselines_before_ratio(
    legacy_before: float | dict[str, str | int | float],
    message: str,
) -> None:
    with pytest.raises(gpr.IncompatibleMetricSpaceError, match=message):
        gpr.compute_gpr(0.75, legacy_before)


def test_compute_gpr_rejects_pooling_mismatch() -> None:
    embeddings = np.repeat(np.eye(5, dtype=np.float32), 2, axis=0)
    layers = np.repeat(np.arange(5), 2)
    before = gpr.measure_continuity(embeddings, layers, _provenance())
    after = gpr.measure_continuity(
        embeddings,
        layers,
        replace(_provenance(), pooling="mean_pool", checkpoint_path="/models/after"),
    )

    with pytest.raises(gpr.IncompatibleMetricSpaceError, match="pooling"):
        gpr.compute_gpr(after, before)


def test_bootstrap_cs_is_finite_ordered_and_seed_deterministic() -> None:
    rng = np.random.default_rng(13)
    directions = np.column_stack((np.cos(np.linspace(0, np.pi / 2, 5)), np.sin(np.linspace(0, np.pi / 2, 5))))
    embeddings = np.vstack([direction + rng.normal(scale=0.01, size=(4, 2)) for direction in directions]).astype(np.float32)
    layers = np.repeat(np.arange(5), 4)
    first = gpr.bootstrap_cs(embeddings, layers, n_layers=5, n_boot=100, seed=91)
    second = gpr.bootstrap_cs(embeddings, layers, n_layers=5, n_boot=100, seed=91)

    assert first == second
    assert np.isfinite(list(first.values())).all()
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_adjacency_accuracy_is_higher_for_ordered_axis_than_anti_ordered_axis() -> None:
    ordered = np.arange(5, dtype=np.float32)[:, None]
    anti_ordered = np.array([0, 4, 1, 3, 2], dtype=np.float32)[:, None]
    ordered_result: object = gpr.layer_adjacency_accuracy(ordered, [0, 1, 2, 3, 4])
    anti_result: object = gpr.layer_adjacency_accuracy(anti_ordered, [0, 1, 2, 3, 4])
    assert isinstance(ordered_result, tuple)
    assert isinstance(anti_result, tuple)
    ordered_score, _, _ = ordered_result
    anti_score, _, _ = anti_result

    assert ordered_score > 0.5
    assert anti_score < ordered_score
