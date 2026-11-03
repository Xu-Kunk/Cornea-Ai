"""Parameterization tests for scripts/run_tier_b_eval.py (--accession).

Covers the accession-parameterization contract:

* default accession (gse153515) resolves to exactly the legacy paths — no-arg
  behavior unchanged,
* --accession gse299152 resolves the new locked config + external data file and
  a dedicated output/token-cache directory,
* arm-aware checkpoint discovery + run parsing ([ARM:]LAM:SEED),
* the GSE299152 locked config content (accession / donors / harmonize map /
  epochs / annotated file),
* CPU smoke on real files: run_static_checks passes for the new config and the
  real GSE299152 obs contract (donor values, empty TAC) plus the leakage gate.

No GPU work and no full evaluation is executed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
pytestmark = pytest.mark.requires_data  # needs the distributed-less data / full environment
import pytest
import yaml

from scripts import run_tier_b_eval as rtbe

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Accession / path resolution
# ---------------------------------------------------------------------------

def test_default_accession_resolves_legacy_paths() -> None:
    p = rtbe.resolve_run_paths(None)
    assert p["accession"] == "gse153515"
    assert p["config_path"] == rtbe.CONFIG_PATH
    assert p["ext_data"] == rtbe.EXT_DATA
    assert p["out_dir"] == rtbe.OUT_DIR
    assert p["token_dataset"] == rtbe.TOKEN_DATASET

    # Explicit default, case-insensitive → identical result.
    assert rtbe.resolve_run_paths("GSE153515") == p


def test_gse299152_accession_resolves_new_paths() -> None:
    p = rtbe.resolve_run_paths("gse299152")
    assert p["accession"] == "gse299152"
    assert p["config_path"] == PROJECT_ROOT / "config" / "external_test_gse299152.yml"
    assert p["ext_data"] == PROJECT_ROOT / "data" / "gse299152" / "gse299152_annotated.h5ad"
    assert p["out_dir"] == PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval_gse299152"
    assert p["token_cache"] == p["out_dir"] / "external_tokenized"
    assert p["token_dataset"] == p["token_cache"] / "cell.dataset"
    # Dedicated output dir must not collide with the GSE153515 dir.
    assert p["out_dir"] != rtbe.OUT_DIR


def test_unsupported_accession_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        rtbe.resolve_run_paths("gse12345")


# ---------------------------------------------------------------------------
# Run parsing + arm-aware checkpoint discovery
# ---------------------------------------------------------------------------

def test_parse_runs_default_and_arm_forms() -> None:
    # No --runs → all 6 combos under the default arm (unchanged).
    runs = rtbe._parse_runs(None)
    assert len(runs) == 6
    assert all(arm == "caft_gf" for arm, _, _ in runs)
    assert sorted((lam, seed) for _, lam, seed in runs) == sorted(
        (lam, seed) for lam in rtbe.LAMBDAS for seed in rtbe.SEEDS
    )

    # Legacy 2-part token → default arm.
    assert rtbe._parse_runs("0.0:42") == [("caft_gf", 0.0, 42)]

    # 3-part token → explicit arm.
    assert rtbe._parse_runs("ordinal_gf:0.5:456") == [("ordinal_gf", 0.5, 456)]

    # Mixed list.
    mixed = rtbe._parse_runs("0.0:42,ordinal_gf:0.0:123,ordinal_gf:0.5:456")
    assert mixed == [
        ("caft_gf", 0.0, 42),
        ("ordinal_gf", 0.0, 123),
        ("ordinal_gf", 0.5, 456),
    ]


def test_parse_runs_malformed_raises() -> None:
    with pytest.raises(ValueError, match="invalid run token"):
        rtbe._parse_runs("0.0")  # missing seed
    with pytest.raises(ValueError, match="invalid run token"):
        rtbe._parse_runs("a:b:c:d")  # too many parts


def test_discover_checkpoints_arm_aware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save = tmp_path / "save"
    (save / "caft_gf_lam0.0_seed42_20260731-211824" / "model" / "checkpoint-1000").mkdir(parents=True)
    (save / "ordinal_gf_lam0.5_seed456_20260807-091349" / "model" / "checkpoint-500").mkdir(parents=True)
    monkeypatch.setattr(rtbe, "SAVE_DIR", save)

    runs = [
        ("caft_gf", 0.0, 42),
        ("ordinal_gf", 0.5, 456),
        ("caft_gf", 9.9, 999),  # missing → skipped with a warning
    ]
    found = rtbe.discover_checkpoints(runs)
    assert len(found) == 2
    by_arm = {ck["arm"]: ck for ck in found}
    assert by_arm["caft_gf"]["step"] == 1000
    assert by_arm["ordinal_gf"]["step"] == 500
    assert by_arm["ordinal_gf"]["checkpoint_path"].name == "checkpoint-500"


# ---------------------------------------------------------------------------
# GSE299152 locked config
# ---------------------------------------------------------------------------

def test_gse299152_config_contract() -> None:
    cfg_path = PROJECT_ROOT / "config" / "external_test_gse299152.yml"
    raw = cfg_path.read_text(encoding="utf-8")
    assert "LOCKED" in raw
    cfg = yaml.safe_load(raw)

    ext = cfg["external_test"]
    assert ext["accession"] == "GSE299152"
    assert ext["annotated_file"] == "data/gse299152/gse299152_annotated.h5ad"
    assert ext["source_disjoint"] is True
    assert ext["donor_disjoint"] is True
    assert set(ext["donors"]) == {"Donor2046", "Donor2206", "Donor3748"}
    assert ext["annotated_shape"] == [23970, 11274]
    # DOI must be verifiable, not a placeholder.
    assert ext["references"][0]["doi"] == "10.1093/pnasnexus/pgaf348"
    assert ext["references"][0]["pmid"] == "41245088"

    assert cfg["fair_protocol"]["epochs"] == 3
    assert cfg["fair_protocol"]["seeds"] == [42, 123, 456]
    assert cfg["primary_endpoint"]["classes"] == ["LSC", "TAC", "basal", "differentiated"]

    hm = cfg["annotation"]["harmonize_map"]
    assert hm["LPC"] == "LSC"
    assert hm["TAC"] == "TAC"
    assert hm["conjunctival"] == "NON_LAYER"
    assert hm["endothelial"] == "NON_LAYER"
    assert hm["stromal"] == "NON_LAYER"
    # Annotated file must exist (evaluation reads it).
    assert (PROJECT_ROOT / ext["annotated_file"]).is_file()


def test_static_checks_pass_on_gse299152_config() -> None:
    from corneal_fm.evaluation import run_static_checks

    res = run_static_checks(
        str(PROJECT_ROOT / "config" / "external_test_gse299152.yml"),
        str(PROJECT_ROOT / "artifacts" / "paper1_fair_split" / "manifest.json"),
    )
    assert res["passed"] is True
    assert res["config_locked"] is True
    assert res["fair_protocol_epochs"] == 3
    assert res["manifest_donor_overlap"] == 0
    assert res["external_test_exists"] is True
    assert res["train_data_exists"] is True
    assert res["legacy_data_refs"] == []


# ---------------------------------------------------------------------------
# Real-data CPU smoke: obs contract + leakage gate (no GPU, no evaluation)
# ---------------------------------------------------------------------------

def test_gse299152_obs_donor_contract() -> None:
    import anndata as ad

    ext = ad.read_h5ad(
        str(PROJECT_ROOT / "data" / "gse299152" / "gse299152_annotated.h5ad"),
        backed="r",
    )
    assert ext.n_obs == 23970
    assert ext.n_vars == 11274

    donors = sorted(ext.obs["donor"].astype(str).unique())
    assert donors == ["Donor2046", "Donor2206", "Donor3748"]
    # Donor is pre-assigned and duplicated into the evaluation key donor_id.
    assert (ext.obs["donor"].values == ext.obs["donor_id"].values).all()
    assert sorted(ext.obs["donor_id"].astype(str).unique()) == donors

    harm = set(ext.obs["cell_type_harmonized"].astype(str).unique())
    assert harm <= {"LSC", "TAC", "basal", "differentiated", "NON_LAYER"}
    # Documented caveat: TAC (layer 1) is empty at the chosen resolution.
    assert "TAC" not in harm
    assert set(ext.obs["layer"].astype(str).unique()) <= {"0", "1", "2", "3", "-1"}
    assert int((ext.obs["cell_type_harmonized"].astype(str) != "NON_LAYER").sum()) > 0


def test_gse299152_real_leakage_gate() -> None:
    import anndata as ad

    from corneal_fm.evaluation import run_leakage_checks

    manifest = json.loads(
        (PROJECT_ROOT / "artifacts" / "paper1_fair_split" / "manifest.json").read_text()
    )
    train_obs = ad.read_h5ad(
        str(PROJECT_ROOT / "data" / "corneal_100k_donor_split.h5ad"), backed="r"
    ).obs
    ext = ad.read_h5ad(
        str(PROJECT_ROOT / "data" / "gse299152" / "gse299152_annotated.h5ad"),
        backed="r",
    )
    ext_obs = ext.obs.copy()
    # tokenize_external() now declares the study source explicitly.
    ext_obs["study"] = "GSE299152"

    res = run_leakage_checks(train_obs, ext_obs, manifest)
    assert res["passed"] is True
    assert res["donor_disjoint"] is True
    assert res["source_disjoint"] is True
    assert res["external_accession"] == "GSE299152"
    assert sorted(res["external_donors"]) == ["donor2046", "donor2206", "donor3748"]


# ---------------------------------------------------------------------------
# evaluate_on_external_test accepts the GSE299152 config (invariant gate)
# ---------------------------------------------------------------------------

def _tiny_eval_inputs() -> tuple:
    """Minimal in-memory adata + arrays for evaluate_on_external_test."""
    import anndata as ad

    train_adata = ad.AnnData(
        X=np.zeros((4, 4)),
        obs=pd.DataFrame({"donor_id": ["train_a", "train_a", "train_b", "train_b"]}),
    )
    labels = ["LSC", "TAC", "basal", "differentiated",
              "LSC", "TAC", "basal", "differentiated"]
    ext_adata = ad.AnnData(
        X=np.zeros((8, 4)),
        obs=pd.DataFrame({
            "donor_id": ["ext_1", "ext_1", "ext_1", "ext_1", "ext_2", "ext_2", "ext_2", "ext_2"],
            "cell_type_harmonized": labels,
            "study": ["GSE299152"] * 8,
        }),
    )
    predictions = np.array(labels, dtype=object)
    embeddings = np.zeros((8, 8))
    order = ["LSC", "TAC", "basal", "differentiated"]
    prob = np.zeros((8, 4))
    for i, lbl in enumerate(labels):
        prob[i, order.index(lbl)] = 1.0
    return train_adata, ext_adata, embeddings, predictions, prob


def test_eval_accepts_gse299152_config() -> None:
    from corneal_fm.evaluation import evaluate_on_external_test

    train_adata, ext_adata, embeddings, predictions, prob = _tiny_eval_inputs()
    config = {
        "external_test": {"accession": "GSE299152", "annotated_file": "data/gse299152/gse299152_annotated.h5ad"},
        "fair_protocol": {"epochs": 3},
    }
    manifest = {
        "donor_overlap": 0,
        "donor_overlap_count": 0,
        "input_paths": {
            "train_h5ad": "data/corneal_100k_donor_split.h5ad",
            "test_h5ad": "data/corneal_20k_donor_split.h5ad",
        },
    }
    results = evaluate_on_external_test(
        train_adata, ext_adata, embeddings, predictions, prob,
        config, manifest,
    )
    assert results["provenance"]["accession"] == "GSE299152"
    assert results["provenance"]["n_eval_donors"] == 2
    assert results["primary_endpoint"]["macro_f1"] == pytest.approx(1.0)
    assert results["leakage"]["external_accession"] == "GSE299152"


def test_eval_rejects_missing_accession() -> None:
    from corneal_fm.evaluation import evaluate_on_external_test

    train_adata, ext_adata, embeddings, predictions, prob = _tiny_eval_inputs()
    config = {
        "external_test": {"annotated_file": "data/gse299152/gse299152_annotated.h5ad"},  # no accession
        "fair_protocol": {"epochs": 3},
    }
    manifest = {"donor_overlap": 0, "donor_overlap_count": 0}
    with pytest.raises(ValueError, match="accession"):
        evaluate_on_external_test(
            train_adata, ext_adata, embeddings, predictions, prob,
            config, manifest,
        )
