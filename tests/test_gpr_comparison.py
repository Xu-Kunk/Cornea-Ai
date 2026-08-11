"""Contract tests for paired checkpoint GPR evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from corneal_fm import gpr
from corneal_fm import gpr_evaluation as evaluation


class _TokenizedDataset:
    column_names = ["input_ids", "length", "layer"]
    _fingerprint = "fixture-tokenized-v1"

    def __init__(self, layers: NDArray[np.int64]) -> None:
        self._layers = layers

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, key: str) -> list[int]:
        if key != "layer":
            raise KeyError(key)
        return self._layers.tolist()


def _layer_directions(
    degrees: list[float],
    cells_per_layer: int = 3,
) -> NDArray[np.float32]:
    radians = np.deg2rad(degrees)
    directions = np.column_stack((np.cos(radians), np.sin(radians))).astype(np.float32)
    return np.repeat(directions, cells_per_layer, axis=0)


def test_compare_checkpoints_emits_same_space_primary_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before_checkpoint = tmp_path / "before"
    after_checkpoint = tmp_path / "after"
    tokenized_path = tmp_path / "cells.dataset"
    before_checkpoint.mkdir()
    after_checkpoint.mkdir()
    tokenized_path.mkdir()
    layers = np.repeat(np.arange(5), 3)
    dataset = _TokenizedDataset(layers)
    loaded_paths: list[str] = []

    embedding_views: dict[tuple[str, bool], NDArray[np.float32]] = {
        (str(before_checkpoint.resolve()), True): _layer_directions([0, 30, 60, 90, 120]),
        (str(after_checkpoint.resolve()), True): _layer_directions([0, 20, 40, 60, 80]),
        (str(before_checkpoint.resolve()), False): _layer_directions([0, 40, 80, 120, 160]),
        (str(after_checkpoint.resolve()), False): _layer_directions([0, 35, 70, 105, 140]),
    }

    def fake_load_dataset(path: str) -> _TokenizedDataset:
        loaded_paths.append(path)
        return dataset

    def fake_load_model(checkpoint: str, num_classes: int = 20) -> str:
        del num_classes
        return checkpoint

    def fake_extract_embeddings(
        model: str,
        tokenized: _TokenizedDataset,
        pad_id: int = 0,
        batch: int = 32,
        use_cls: bool = True,
    ) -> NDArray[np.float32]:
        del tokenized, pad_id, batch
        return embedding_views[(model, use_cls)]

    monkeypatch.setattr(evaluation, "load_from_disk", fake_load_dataset)
    monkeypatch.setattr(evaluation, "load_model", fake_load_model)
    monkeypatch.setattr(evaluation, "extract_embeddings", fake_extract_embeddings)
    request = gpr.ComparisonRequest(
        before_checkpoint=str(before_checkpoint),
        after_checkpoint=str(after_checkpoint),
        tokenized_path=str(tokenized_path),
        num_classes=31,
        seed=23,
        n_boot=25,
    )

    first = gpr.compare_checkpoints(request)
    second = gpr.compare_checkpoints(request)

    assert loaded_paths == [str(tokenized_path)] * 2
    assert first == second
    assert set(first) == {"contract", "primary", "sensitivity", "diagnostics"}
    primary = first["primary"]
    assert primary["metric_name"] == "raw_cls_same_space_gpr"
    assert primary["ceiling_sensitive"] is True
    assert primary["gpr"] == pytest.approx(primary["cs_after"] / primary["cs_before"])
    assert primary["absolute_delta"] == pytest.approx(
        primary["cs_after"] - primary["cs_before"],
    )
    assert set(primary["per_adjacent_pair_delta"]) == {
        "LSC->TAC",
        "TAC->basal",
        "basal->wing",
        "wing->squamous",
    }
    assert set(primary["paired_stratified_bootstrap_95ci"]) == {
        "cs_before",
        "cs_after",
        "gpr",
        "absolute_delta",
    }
    assert set(first["sensitivity"]) == {"mean_pool"}
    assert "mean_pool" not in first["primary"]
    assert set(first["diagnostics"]["anisotropy"]) == {"raw_cls", "mean_pool"}

    contract = first["contract"]
    assert contract["embedding"] == {
        "source": "geneformer.last_hidden_state",
        "pooling": "raw_cls",
        "normalization": gpr.NORMALIZATION_CONTRACT,
    }
    assert contract["metric"] == {
        "formula": gpr.CONTINUITY_FORMULA,
        "layer_order": gpr.LAYER_NAMES,
    }
    assert contract["dataset"] == {
        "path": str(tokenized_path.resolve()),
        "fingerprint": "huggingface:fixture-tokenized-v1",
    }
    assert contract["checkpoints"] == {
        "before": str(before_checkpoint.resolve()),
        "after": str(after_checkpoint.resolve()),
    }
    assert contract["resampling"] == {
        "method": "paired_stratified_percentile",
        "seed": 23,
        "n_boot": 25,
    }


def test_single_checkpoint_measurement_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    tokenized_path = tmp_path / "cells.dataset"
    checkpoint.mkdir()
    tokenized_path.mkdir()
    layers = np.repeat(np.arange(5), 2)
    embeddings = _layer_directions([0, 30, 60, 90, 120], cells_per_layer=2)
    dataset = _TokenizedDataset(layers)

    monkeypatch.setattr(evaluation, "load_from_disk", lambda path: dataset)
    monkeypatch.setattr(
        evaluation,
        "load_model",
        lambda checkpoint, num_classes=20: checkpoint,
    )
    monkeypatch.setattr(
        evaluation,
        "extract_embeddings",
        lambda model, tokenized, use_cls=True: embeddings,
    )
    result = gpr.evaluate_checkpoint(
        str(checkpoint),
        str(tokenized_path),
        n_boot=10,
        use_cls=True,
        seed=29,
    )

    assert result["continuity_score"] == pytest.approx(np.cos(np.deg2rad(30)), abs=1e-4)
    assert result["contract"]["embedding"]["pooling"] == "raw_cls"
    assert result["contract"]["dataset"] == {
        "path": str(tokenized_path.resolve()),
        "fingerprint": "huggingface:fixture-tokenized-v1",
    }
    assert result["contract"]["checkpoints"] == {
        "measured": str(checkpoint.resolve()),
    }
    assert result["contract"]["resampling"] == {"seed": 29, "n_boot": 10}
    assert set(result["diagnostics"]["anisotropy"]) == {
        "mean_resultant_length",
        "mean_pairwise_cosine",
    }
