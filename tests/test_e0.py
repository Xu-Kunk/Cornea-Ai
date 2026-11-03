"""Unit tests for the leakage-safe E0 Axis-1 comparator.

These tests deliberately use CPU-only synthetic AnnData and never load a
foundation model or project data.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("click", reason="analysis stack not installed in this environment")
pytest.importorskip("datasets", reason="analysis stack not installed in this environment")

from scripts.e0_axis1_baseline import (
    build_result,
    fit_eval_logreg,
    go_no_go,
    select_hvg_from_train,
    validate_no_donor_leakage,
    write_result,
)


def _adata_with_variable_gene(gene_index: int) -> ad.AnnData:
    """Create a fixed synthetic split whose selected HVGs are controllable."""
    rng = np.random.default_rng(7 + gene_index)
    matrix = rng.poisson(2, size=(48, 8)).astype(np.float32)
    matrix[:, gene_index] = np.arange(48, dtype=np.float32) * 10
    matrix = np.log1p(matrix)
    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"donor_id": [f"d{i // 6}" for i in range(48)]}),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(8)]),
    )


def test_select_hvg_uses_only_the_train_adata() -> None:
    """Given distinct train/test objects, selection only mutates and reflects train."""
    train_a = _adata_with_variable_gene(0)
    train_b = _adata_with_variable_gene(7)
    held_out = _adata_with_variable_gene(3)

    hvg_a = select_hvg_from_train(train_a, n_top=1)
    hvg_b = select_hvg_from_train(train_b, n_top=1)

    assert "highly_variable" in train_a.var
    assert "highly_variable" not in held_out.var
    assert hvg_a != hvg_b


def test_validate_no_donor_leakage_rejects_overlap() -> None:
    """Given a donor in both splits, the guard fails before model fitting."""
    train_obs = pd.DataFrame({"donor_id": ["train_1", "shared"]})
    test_obs = pd.DataFrame({"donor_id": ["shared", "test_1"]})

    with pytest.raises(ValueError, match="overlap"):
        validate_no_donor_leakage(train_obs, test_obs, "donor_id")


def test_validate_no_donor_leakage_accepts_disjoint_sets() -> None:
    """Given disjoint donors, the guard returns donor provenance."""
    provenance = validate_no_donor_leakage(
        pd.DataFrame({"donor_id": ["train_1", "train_2"]}),
        pd.DataFrame({"donor_id": ["test_1"]}),
        "donor_id",
    )

    assert provenance == {"train_donors": 2, "test_donors": 1, "donor_disjoint": True}


def test_fit_eval_logreg_scores_separable_data_and_random_labels() -> None:
    """Given separable versus random labels, macro-F1 follows the expected direction."""
    rng = np.random.default_rng(11)
    train_x = np.vstack((rng.normal(-3, 0.1, (30, 2)), rng.normal(3, 0.1, (30, 2))))
    test_x = np.vstack((rng.normal(-3, 0.1, (10, 2)), rng.normal(3, 0.1, (10, 2))))
    train_y = np.array(["a"] * 30 + ["b"] * 30)
    test_y = np.array(["a"] * 10 + ["b"] * 10)

    separable = fit_eval_logreg(train_x, train_y, test_x, test_y, seed=42)
    random_train_y = rng.integers(0, 2, size=60).astype(str)
    random_test_y = rng.integers(0, 2, size=20).astype(str)
    random_labels = fit_eval_logreg(train_x, random_train_y, test_x, random_test_y, seed=42)

    assert separable.predictions.shape == test_y.shape
    assert 0.0 <= separable.macro_f1 <= 1.0
    assert separable.macro_f1 == pytest.approx(1.0)
    assert random_labels.macro_f1 < separable.macro_f1


def test_fit_eval_logreg_is_reproducible_for_a_fixed_seed() -> None:
    """Given identical inputs and seed, logistic-probe macro-F1 is identical."""
    rng = np.random.default_rng(12)
    train_x = rng.normal(size=(40, 4))
    test_x = rng.normal(size=(16, 4))
    train_y = np.array(["a", "b"] * 20)
    test_y = np.array(["a", "b"] * 8)

    first = fit_eval_logreg(train_x, train_y, test_x, test_y, seed=42)
    second = fit_eval_logreg(train_x, train_y, test_x, test_y, seed=42)

    assert first.macro_f1 == second.macro_f1
    assert np.array_equal(first.predictions, second.predictions)


def test_write_result_has_required_json_schema_and_mechanical_gate(tmp_path) -> None:
    """Given scores, writer emits the E0 schema and GO iff scFM strictly wins."""
    result = build_result(
        label="cell_type",
        seed=42,
        train_path="train.h5ad",
        test_path="test.h5ad",
        split={"train_donors": 2, "test_donors": 1, "donor_disjoint": True},
        baseline={"macro_f1": 0.7, "n_hvg": 2, "n_pcs": 2, "per_class_f1": {"a": 0.7}},
        scfm={"macro_f1": 0.8, "model": "Geneformer-V2-104M", "per_class_f1": {"a": 0.8}},
        class_counts={"train": {"a": 4}, "test": {"a": 2}},
    )
    output = tmp_path / "e0.json"
    write_result(result, output)
    saved = json.loads(output.read_text())

    required = {
        "task",
        "label",
        "seed",
        "split",
        "baseline_hvg_pca",
        "scfm_frozen",
        "go_no_go",
        "rule",
        "n_classes",
        "class_counts",
    }
    assert required <= saved.keys()
    assert isinstance(saved["seed"], int)
    assert isinstance(saved["split"]["donor_disjoint"], bool)
    assert isinstance(saved["baseline_hvg_pca"]["macro_f1"], float)
    assert isinstance(saved["baseline_hvg_pca"]["n_hvg"], int)
    assert isinstance(saved["baseline_hvg_pca"]["per_class_f1"], dict)
    assert isinstance(saved["scfm_frozen"]["model"], str)
    assert saved["go_no_go"] == "GO"
    assert go_no_go(0.8, 0.7) == "GO"
    assert go_no_go(0.7, 0.7) == "NO_GO"
