"""Unit tests for the Tier-B evaluator (corneal_fm.evaluation).

Covers the seven required gates:

* static checks: pass / reject-wrong-epochs
* leakage checks: pass (disjoint donors) / reject-overlap
* donor-level macro-F1 on known labels
* layer adjacency accuracy (LAA)
* expected calibration error (ECE)

Tests are self-contained (tmp_path fixtures); no real data files are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("transformers", reason="analysis stack not installed in this environment")
pytest.importorskip("geneformer", reason="analysis stack not installed in this environment")
from corneal_fm.evaluation import (
    donor_level_macro_f1,
    expected_calibration_error,
    layer_adjacency_accuracy,
    marker_gradient_correlation,
    run_leakage_checks,
    run_static_checks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, *, epochs: int = 3, locked: bool = True) -> Path:
    cfg = {
        "external_test": {
            "accession": "GSE153515",
            "annotated_file": "data/gse153515_annotated.h5ad",
            "source_disjoint": True,
            "donor_disjoint": True,
        },
        "fair_protocol": {"epochs": epochs},
    }
    cfg_path = tmp_path / "config" / "external_test_gse153515.yml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Status: LOCKED 2026-07-31\n" if locked else "# Status: draft\n"
    cfg_path.write_text(header + yaml.safe_dump(cfg), encoding="utf-8")

    # Create the referenced data files so existence checks pass.
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for name in ("gse153515_annotated.h5ad", "corneal_100k_donor_split.h5ad"):
        (data / name).write_bytes(b"\x89HDF\r\n\x1a\nplaceholder")
    return cfg_path


def _write_manifest(tmp_path: Path, *, donor_overlap: int = 0) -> Path:
    manifest = {
        "donor_overlap": donor_overlap,
        "donor_overlap_count": donor_overlap,
        "counts": {"train_cells": 99606, "test_cells": 20394},
        "input_paths": {
            "train_h5ad": str(tmp_path / "data" / "corneal_100k_donor_split.h5ad"),
            "test_h5ad": str(tmp_path / "data" / "corneal_20k_donor_split.h5ad"),
        },
        "mode": "strict_fair",
    }
    mani_path = tmp_path / "manifest.json"
    mani_path.write_text(json.dumps(manifest), encoding="utf-8")
    return mani_path


def _obs_frame(donors: list[str], studies: list[str] | None = None,
               harmonized: bool = True) -> dict[str, list[str]]:
    df = {"donor_id": donors}
    if studies is not None:
        df["study"] = studies
    if harmonized:
        df["cell_type_harmonized"] = ["basal"] * len(donors)
    else:
        df["author_cell_type"] = ["Limbus_Epi"] * len(donors)
    return df


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def test_static_checks_pass(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, epochs=3)
    mani_path = _write_manifest(tmp_path, donor_overlap=0)
    results = run_static_checks(str(cfg_path), str(mani_path))
    assert results["passed"] is True
    assert results["config_locked"] is True
    assert results["fair_protocol_epochs"] == 3
    assert results["manifest_donor_overlap"] == 0
    assert results["external_test_exists"] is True
    assert results["train_data_exists"] is True
    assert results["legacy_data_refs"] == []


def test_static_checks_reject_wrong_epochs(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, epochs=10)
    mani_path = _write_manifest(tmp_path, donor_overlap=0)
    with pytest.raises(ValueError, match="epochs=10"):
        run_static_checks(str(cfg_path), str(mani_path))


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

def test_leakage_checks_pass(tmp_path: Path) -> None:
    manifest = json.loads(_write_manifest(tmp_path, donor_overlap=0).read_text())
    import pandas as pd

    train_obs = pd.DataFrame(
        _obs_frame(
            donors=["donor_A", "donor_B"],
            studies=["chen_limbus", "shi_GSE157474"],
        )
    )
    ext_obs = pd.DataFrame(
        _obs_frame(donors=["donor_1", "donor_2"], harmonized=True)
    )
    results = run_leakage_checks(train_obs, ext_obs, manifest)
    assert results["passed"] is True
    assert results["donor_disjoint"] is True
    assert results["source_disjoint"] is True
    assert results["external_accession"] == "GSE153515"


def test_leakage_checks_reject_overlap(tmp_path: Path) -> None:
    manifest = json.loads(_write_manifest(tmp_path, donor_overlap=0).read_text())
    import pandas as pd

    # Shared donor between training and external test.
    train_obs = pd.DataFrame(_obs_frame(donors=["donor_1", "donor_B"]))
    ext_obs = pd.DataFrame(_obs_frame(donors=["donor_1", "donor_2"]))
    with pytest.raises(ValueError, match="donor overlap"):
        run_leakage_checks(train_obs, ext_obs, manifest)

    # Source overlap: a training study contains GSE153515.
    train_obs2 = pd.DataFrame(
        _obs_frame(donors=["donor_A", "donor_B"], studies=["chen_limbus", "GSE153515"])
    )
    ext_obs2 = pd.DataFrame(_obs_frame(donors=["donor_1", "donor_2"]))
    with pytest.raises(ValueError, match="GSE153515"):
        run_leakage_checks(train_obs2, ext_obs2, manifest)


# ---------------------------------------------------------------------------
# Donor-level macro-F1
# ---------------------------------------------------------------------------

def test_donor_level_macro_f1_basic() -> None:
    # Single donor covering all four classes, perfectly classified → macro-F1 1.
    y_true = np.array(["LSC", "LSC", "TAC", "TAC", "basal", "differentiated"])
    y_pred = np.array(["LSC", "LSC", "TAC", "TAC", "basal", "differentiated"])
    donors = np.array(["d1"] * 6)
    classes = ["LSC", "TAC", "basal", "differentiated"]

    res = donor_level_macro_f1(y_true, y_pred, donors, classes, n_boot=100, seed=42)
    assert res["macro_f1"] == pytest.approx(1.0)
    assert res["per_donor"]["d1"] == pytest.approx(1.0)
    assert 0.0 <= res["ci_lower"] <= res["ci_upper"] <= 1.0
    assert set(res["per_class"].keys()) == set(classes)
    assert res["per_class"]["LSC"]["f1"] == pytest.approx(1.0)
    assert res["per_class"]["basal"]["support"] == 1
    assert len(res["confusion"]) == len(classes)

    # Donor_2 all wrong: donor-level macro-F1 must be the mean of (1.0, 0.0).
    # Each donor covers all four classes so macro-F1 is over the full label set.
    y_true2 = np.array(
        ["LSC", "TAC", "basal", "differentiated", "LSC", "TAC",
         "LSC", "TAC", "basal", "differentiated", "basal", "differentiated"]
    )
    y_pred2 = np.array(
        ["LSC", "TAC", "basal", "differentiated", "LSC", "TAC",
         "basal", "basal", "LSC", "LSC", "TAC", "TAC"]
    )
    donors2 = np.array(["d1"] * 6 + ["d2"] * 6)
    res2 = donor_level_macro_f1(y_true2, y_pred2, donors2, classes, n_boot=100, seed=1)
    assert res2["per_donor"]["d1"] == pytest.approx(1.0)
    assert res2["per_donor"]["d2"] == pytest.approx(0.0)
    assert res2["macro_f1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LAA / marker gradient / ECE
# ---------------------------------------------------------------------------

def test_laa_basic() -> None:
    # Embeddings well separated along a line; nearest neighbors share labels.
    labels = np.array(["LSC", "LSC", "TAC", "TAC", "basal", "basal",
                       "differentiated", "differentiated"])
    rng = np.random.default_rng(0)
    emb = np.stack([np.linspace(0, 1, 8), np.zeros(8)], axis=1) + rng.normal(
        0, 0.01, (8, 2)
    )
    laa = layer_adjacency_accuracy(labels, emb)
    assert 0.0 <= laa <= 1.0
    # Perfectly clustered adjacent pairs → LAA close to 1.
    assert laa >= 0.75

    # NON_LAYER cells are excluded, not scored.
    labels2 = np.array(["LSC", "TAC", "NON_LAYER"])
    emb2 = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    assert 0.0 <= layer_adjacency_accuracy(labels2, emb2) <= 1.0


def test_marker_gradient_monotonic() -> None:
    # One cell per layer, embeddings ordered along axis 0 by layer ordinal →
    # exact monotone sequence (no tied ranks) → rho exactly 1.
    labels = np.array(["LSC", "TAC", "basal", "differentiated"])
    emb = np.stack([np.linspace(0, 1, 4), np.zeros(4)], axis=1)
    rho = marker_gradient_correlation(emb, labels)
    assert rho == pytest.approx(1.0)
    assert np.isfinite(rho)

    # The pseudotime axis is oriented LSC -> differentiated, so a mirrored
    # coordinate orientation is invariant: rho stays +1.
    emb_rev = np.stack([np.linspace(1, 0, 4), np.zeros(4)], axis=1)
    assert marker_gradient_correlation(emb_rev, labels) == pytest.approx(1.0)

    # A non-monotone arrangement along the axis clearly fails the gradient.
    emb_bad = np.array([[0.0, 0.0], [0.8, 0.0], [0.2, 0.0], [0.6, 0.0]])
    rho_bad = marker_gradient_correlation(emb_bad, labels)
    assert rho_bad < 0.9


def test_ece_basic() -> None:
    # Perfectly confident and accurate model → ECE 0.
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.eye(3)[y_true]
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)

    # Uniform random confidence, perfect accuracy on the argmax → high ECE.
    y_true2 = np.array([0, 1, 2, 0, 1, 2])
    y_prob2 = np.full((6, 3), 1 / 3)
    y_prob2[range(6), y_true2] = 0.6
    ece = expected_calibration_error(y_true2, y_prob2)
    assert ece > 0.0
    assert ece < 1.0

    # Binary (1-D) probabilities are supported.
    y_true3 = np.array([0, 1, 0, 1])
    y_prob3 = np.array([0.9, 0.9, 0.8, 0.8])
    assert 0.0 <= expected_calibration_error(y_true3, y_prob3) <= 1.0
