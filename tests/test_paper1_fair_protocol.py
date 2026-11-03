"""Contract tests for the this study fair-protocol scaffold (metadata paths only).

Covers the pure validator / metadata paths of the three this study entry scripts:

* ``scripts/prepare_paper1_fair_split.py`` — donor-disjoint manifest contract
* ``scripts/run_post_grid.py`` — strict-fair CLI gate (fail-closed, F1)
* ``scripts/caft_gf.py`` — ``validate_strict_fair_args`` guard

No GPU work, no real H5ADs: tiny synthetic pandas ``obs`` frames and temp JSON
manifests only; file readers (``_read_obs``/``_sha256``) are monkeypatched.

The fair evaluator itself (macro-F1/GPR/LAA/marker-gradient on the donor-disjoint
external test) is deliberately OUT OF SCOPE — the strict gate must fail closed
with ``NotImplementedError`` before any training, which these tests lock in.

``prepare_paper1_fair_split`` and ``run_post_grid`` import only the stdlib at top
level, so they import directly.  ``caft_gf`` pulls in anndata/datasets/transformers
at import time; the ``caft_gf`` fixture below stubs those module-scopedly so the
pure validator tests still RUN in a light environment (and never leak elsewhere).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import prepare_paper1_fair_split as prep
from scripts import run_post_grid as rpg

# Canonical layer table, mirrored from corneal_fm/caft.py.  Used to stub the
# module when transformers is unavailable; the cross-check test below verifies
# these values against the real table whenever transformers is installed.
CANONICAL_LAYER_ORDER = ["LSC", "TAC", "basal", "wing", "squamous"]
CANONICAL_NON_LAYER = -1
CANONICAL_CELL_TYPE_TO_LAYER = {
    "LPC": 0,
    "Fetal_Epi/LSC/LPC": 0,
    "Limbus_Epi-C1": 1,
    "Limbus_Epi-C2": 1,
    "Epi-PMC": 2,
    "Epi-TDC/PMC": 3,
    "Epi-TDC": 4,
    "Unknown Epithelium": 4,
    "Conj-Epi": -1,
    "Unknown-DCN+": -1,
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _obs(donors: list[str], labels: list[str], label_column: str = "cell_type") -> pd.DataFrame:
    return pd.DataFrame({"donor_id": list(donors), label_column: list(labels)})


def _unavailable(what: str):
    def _raise(*args, **kwargs):
        raise RuntimeError(f"{what} is stubbed out in tests")

    return _raise


def _patch_readers(monkeypatch: pytest.MonkeyPatch, train_obs: pd.DataFrame, test_obs: pd.DataFrame) -> None:
    """Replace H5AD readers with synthetic obs; checksums become name-derived."""
    monkeypatch.setattr(
        prep, "_read_obs", lambda path: train_obs if Path(path).name == "train.h5ad" else test_obs
    )
    monkeypatch.setattr(prep, "_sha256", lambda path: f"sha256-{Path(path).name}")


def _write_h5ad_placeholders(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.h5ad"
    test = tmp_path / "test.h5ad"
    train.write_bytes(b"train-bytes")
    test.write_bytes(b"test-bytes")
    return train, test


def _fair_manifest(
    tmp_path: Path,
    *,
    overlap: object = 0,
    train: Path | str | None = "sentinel",
    test: Path | str | None = None,
    mode: str = "strict_fair",
    protocol: str = "paper1_fair",
    name: str = "manifest.json",
) -> Path:
    payload = {
        "protocol": protocol,
        "mode": mode,
        "donor_overlap": overlap,
        "input_paths": {
            "train_h5ad": None if train is None else str(train),
            "test_h5ad": None if test is None else str(test),
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture()
def stub_caft(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Provide corneal_fm.caft for _layer_mapping (stub if transformers missing)."""
    try:
        import corneal_fm.caft as real  # noqa: F401  (needs transformers)

        return real
    except Exception:
        pass
    mod = ModuleType("corneal_fm.caft")
    setattr(mod, "CELL_TYPE_TO_LAYER", dict(CANONICAL_CELL_TYPE_TO_LAYER))
    setattr(mod, "LAYER_ORDER", list(CANONICAL_LAYER_ORDER))
    setattr(mod, "NON_LAYER", CANONICAL_NON_LAYER)
    monkeypatch.setitem(sys.modules, "corneal_fm.caft", mod)
    return mod


@pytest.fixture()
def caft_gf(monkeypatch: pytest.MonkeyPatch):
    """Import scripts.caft_gf with anndata/datasets/corneal_fm.caft stubbed.

    The real module imports anndata, datasets and transformers (via
    corneal_fm.caft) at top level, none of which the pure validator needs.
    Stubs are fixture-scoped via monkeypatch so they never leak to other tests.
    """
    for name in ("anndata", "datasets"):
        if name not in sys.modules:
            stub = ModuleType(name)
            setattr(stub, "read_h5ad", _unavailable(f"{name}.read_h5ad"))
            setattr(stub, "load_from_disk", _unavailable(f"{name}.load_from_disk"))
            monkeypatch.setitem(sys.modules, name, stub)
    caft_stub = ModuleType("corneal_fm.caft")
    setattr(caft_stub, "LAYER_ORDER", list(CANONICAL_LAYER_ORDER))
    setattr(caft_stub, "NON_LAYER", CANONICAL_NON_LAYER)
    setattr(caft_stub, "CELL_TYPE_TO_LAYER", dict(CANONICAL_CELL_TYPE_TO_LAYER))
    setattr(caft_stub, "author_cell_type_to_layer",
            lambda n: CANONICAL_CELL_TYPE_TO_LAYER.get(n, CANONICAL_NON_LAYER))
    setattr(caft_stub, "build_caft_trainer", _unavailable("corneal_fm.caft.build_caft_trainer"))
    setattr(caft_stub, "load_pretrained_classifier", _unavailable("corneal_fm.caft.load_pretrained_classifier"))
    monkeypatch.setitem(sys.modules, "corneal_fm.caft", caft_stub)
    monkeypatch.delitem(sys.modules, "scripts.caft_gf", raising=False)
    import scripts.caft_gf as mod

    return mod


# ── prepare_paper1_fair_split: validate_metadata ─────────────────────────


def test_validate_metadata_disjoint_passes() -> None:
    train = _obs(["D1", "D1", "D2", "D2"], ["A", "B", "A", "B"])
    test = _obs(["D3", "D3", "D4"], ["A", "B", "A"])

    result = prep.validate_metadata(train, test)

    assert result["donor_overlap"] == 0
    assert result["donor_overlap_count"] == 0
    assert result["train_donors"] == 2
    assert result["test_donors"] == 2
    assert result["label_columns"]  # non-empty
    assert "cell_type" in result["label_columns"]


def test_validate_metadata_raises_on_donor_overlap() -> None:
    train = _obs(["D1", "D2"], ["A", "B"])
    test = _obs(["D2", "D3"], ["A", "B"])

    with pytest.raises(ValueError, match="donor overlap is non-zero"):
        prep.validate_metadata(train, test)


def test_validate_metadata_normalization_robust_overlap() -> None:
    """F2 lock: case/whitespace-only donor differences are still overlap."""
    train = _obs(["Donor_01", "D2"], ["A", "B"])
    test = _obs([" donor_01 ", "D3"], ["A", "B"])

    with pytest.raises(ValueError, match="donor overlap is non-zero"):
        prep.validate_metadata(train, test)


@pytest.mark.parametrize("bad_donor", [None, np.nan, "", "none", "null", "<NA>"])
def test_validate_metadata_rejects_missing_or_placeholder_donor(bad_donor) -> None:
    train = pd.DataFrame({"donor_id": ["D1", bad_donor], "cell_type": ["A", "B"]})
    test = _obs(["D3", "D3"], ["A", "B"])

    with pytest.raises(ValueError, match="missing donor_id"):
        prep.validate_metadata(train, test)


def test_validate_metadata_raises_when_donor_column_absent() -> None:
    train = pd.DataFrame({"cell_type": ["A", "B"]})
    test = _obs(["D3"], ["A"])

    with pytest.raises(ValueError, match="must contain 'donor_id'"):
        prep.validate_metadata(train, test)


def test_validate_metadata_raises_when_no_shared_label_column() -> None:
    train = pd.DataFrame({"donor_id": ["D1"], "cell_type": ["A"]})
    test = pd.DataFrame({"donor_id": ["D2"], "author_cell_type": ["A"]})

    with pytest.raises(ValueError, match="cell_type or author_cell_type"):
        prep.validate_metadata(train, test)


def test_validate_metadata_coverage_hard_error_by_default() -> None:
    """F6: label present in train but absent from test is a hard error by default."""
    train = _obs(["D1", "D2"], ["A", "B"])
    test = _obs(["D3", "D4"], ["A", "A"])

    with pytest.raises(ValueError, match="coverage is incomplete"):
        prep.validate_metadata(train, test)


def test_validate_metadata_allow_missing_test_labels_overrides() -> None:
    """F6: the explicit flag tolerates labels missing only from the test split."""
    train = _obs(["D1", "D2"], ["A", "B"])
    test = _obs(["D3", "D4"], ["A", "A"])

    result = prep.validate_metadata(train, test, allow_missing_test_labels=True)

    assert result["donor_overlap"] == 0


def test_validate_metadata_allow_missing_test_labels_still_rejects_missing_train() -> None:
    """F6: a label missing from train stays a hard error even with the flag."""
    train = _obs(["D1", "D2"], ["A", "A"])
    test = _obs(["D3", "D4"], ["A", "B"])

    with pytest.raises(ValueError, match="coverage is incomplete"):
        prep.validate_metadata(train, test, allow_missing_test_labels=True)


# ── prepare_paper1_fair_split: _layer_mapping ────────────────────────────


def test_layer_mapping_uses_canonical_table(stub_caft: ModuleType) -> None:
    names = ["LPC", "Limbus_Epi-C1", "Epi-PMC", "Epi-TDC/PMC", "Epi-TDC", "TotallyUnknownType"]

    result = prep._layer_mapping(names)
    mapping = result["author_cell_type_to_layer"]

    assert mapping["LPC"] == 0
    assert mapping["Limbus_Epi-C1"] == 1
    assert mapping["Epi-PMC"] == 2
    assert mapping["Epi-TDC/PMC"] == 3
    assert mapping["Epi-TDC"] == 4
    assert mapping["TotallyUnknownType"] == -1
    assert result["layer_order"] == CANONICAL_LAYER_ORDER
    assert result["non_layer"] == -1
    assert result["source"] == "corneal_fm.caft.CELL_TYPE_TO_LAYER"


def test_layer_mapping_cross_checks_real_caft_table() -> None:
    """Guard the mirrored canonical table against drift in corneal_fm.caft."""
    pytest.importorskip("transformers")
    from corneal_fm.caft import CELL_TYPE_TO_LAYER, LAYER_ORDER, NON_LAYER

    assert CELL_TYPE_TO_LAYER == CANONICAL_CELL_TYPE_TO_LAYER
    assert list(LAYER_ORDER) == CANONICAL_LAYER_ORDER
    assert NON_LAYER == CANONICAL_NON_LAYER


# ── prepare_paper1_fair_split: build_manifest / prepare / main ───────────


def test_build_manifest_schema_and_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_caft: ModuleType) -> None:
    train_obs = pd.DataFrame(
        {
            "donor_id": ["D1", "D1", "D2", "D2"],
            "cell_type": ["A", "B", "A", "B"],
            "author_cell_type": ["LPC", "Epi-TDC", "WeirdType", "LPC"],
        }
    )
    test_obs = pd.DataFrame(
        {
            "donor_id": ["D3", "D3", "D4"],
            "cell_type": ["A", "B", "A"],
            "author_cell_type": ["LPC", "Epi-TDC", "WeirdType"],
        }
    )
    _patch_readers(monkeypatch, train_obs, test_obs)
    train, test = _write_h5ad_placeholders(tmp_path)

    manifest, donor_check = prep.build_manifest(train, test)

    assert manifest["protocol"] == "paper1_fair"
    assert manifest["mode"] == "strict_fair"
    assert manifest["seed"] == 42
    assert manifest["donor_overlap"] == 0
    assert manifest["donor_overlap_count"] == 0
    assert manifest["input_paths"] == {
        "train_h5ad": str(train.resolve()),
        "test_h5ad": str(test.resolve()),
    }
    assert manifest["input_checksums"] == {
        "train_h5ad": "sha256-train.h5ad",
        "test_h5ad": "sha256-test.h5ad",
    }
    assert manifest["checksums"] == manifest["input_checksums"]
    assert manifest["counts"]["train_cells"] == 4
    assert manifest["counts"]["test_cells"] == 3
    layer_map = manifest["layer_mapping"]["author_cell_type_to_layer"]
    assert layer_map["LPC"] == 0  # known epithelial -> 0-4
    assert layer_map["Epi-TDC"] == 4
    assert layer_map["WeirdType"] == -1  # unknown -> -1
    assert donor_check["passed"] is True
    assert donor_check["overlap_count"] == 0

    # Deterministic provenance: identical except for the generation timestamp.
    again, _ = prep.build_manifest(train, test)
    manifest.pop("generated_at_utc")
    again.pop("generated_at_utc")
    assert manifest == again


def test_prepare_create_once_then_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_caft: ModuleType) -> None:
    train_obs = _obs(["D1", "D2"], ["A", "B"])
    test_obs = _obs(["D3", "D4"], ["A", "B"])
    _patch_readers(monkeypatch, train_obs, test_obs)
    train, test = _write_h5ad_placeholders(tmp_path)
    out = tmp_path / "out"

    prep.prepare(train, test, out, seed=42)
    assert (out / "manifest.json").is_file()
    assert (out / "donor_overlap_check.json").is_file()

    with pytest.raises(FileExistsError):
        prep.prepare(train, test, out, seed=42)

    prep.prepare(train, test, out, seed=42, force=True)  # succeeds with force


def test_main_exits_nonzero_on_donor_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_obs = _obs(["D1", "D2"], ["A", "B"])
    test_obs = _obs(["D2", "D3"], ["A", "B"])  # D2 overlaps
    _patch_readers(monkeypatch, train_obs, test_obs)
    train, test = _write_h5ad_placeholders(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        prep.main(
            ["--train_h5ad", str(train), "--test_h5ad", str(test), "--output_dir", str(tmp_path / "out")]
        )

    assert excinfo.value.code != 0


# ── run_post_grid: _manifest_overlap / helpers ───────────────────────────


def test_manifest_overlap_variants() -> None:
    assert rpg._manifest_overlap({"donor_overlap": 0}) == 0
    assert rpg._manifest_overlap({"donor_overlap_count": 3}) == 3
    assert rpg._manifest_overlap({"donor_overlap": {"count": 2}}) == 2
    assert rpg._manifest_overlap({"donor_overlap": {"overlap_count": 5}}) == 5
    assert rpg._manifest_overlap({}) is None


def test_legacy_test_names_deduped() -> None:
    """F3: the legacy test-name set is clean (no duplicate literal)."""
    assert rpg.LEGACY_TEST_NAMES == {"corneal_20k_test.h5ad"}
    assert len(rpg.LEGACY_TEST_NAMES) == 1


def test_find_checkpoint_picks_highest(tmp_path: Path) -> None:
    model = tmp_path / "model"
    for step in (100, 200, 50):  # 50 proves numeric (not lexicographic) ordering
        (model / f"checkpoint-{step}").mkdir(parents=True)

    assert rpg._find_checkpoint(tmp_path) == model / "checkpoint-200"
    assert rpg._find_checkpoint(tmp_path / "does-not-exist") is None


def test_done_set_and_load_tsv_round_trip(tmp_path: Path) -> None:
    tsv = tmp_path / "results.tsv"
    tsv.write_text(
        rpg._GPR_HDR
        + "0.0\t0.90\t0.75\t0.70\t0.80\t1.50\tsave_a\t20260101-000000\n"
        + "0.5\t0.91\t0.80\t\t\t2.00\tsave_b\t20260101-000001\n"
    )

    assert rpg._done_set(tsv) == {0.0, 0.5}
    rows = rpg._load_tsv(tsv)
    assert [row["lam"] for row in rows] == [0.0, 0.5]
    assert rows[0]["gpr"] == 0.75
    assert rows[0]["gpr_ci_low"] == 0.70
    assert rows[0]["save_dir_name"] == "save_a"
    assert rows[1]["gpr_ci_low"] is None  # empty CI field round-trips to None
    assert rpg._load_tsv(tmp_path / "missing.tsv") == []
    assert rpg._done_set(tmp_path / "missing.tsv") == set()


# ── run_post_grid: validate_strict_fair_cli ──────────────────────────────


def test_cli_rejects_missing_manifest() -> None:
    with pytest.raises(ValueError, match="requires --split_manifest"):
        rpg.validate_strict_fair_cli(None, "external.h5ad", 3)


def test_cli_rejects_missing_external_test(tmp_path: Path) -> None:
    manifest = _fair_manifest(tmp_path)
    with pytest.raises(ValueError, match="requires --external_test_path"):
        rpg.validate_strict_fair_cli(str(manifest), None, 3)


@pytest.mark.parametrize("epochs", [None, 10, 1])
def test_cli_rejects_wrong_epochs(epochs) -> None:
    with pytest.raises(ValueError, match="requires explicit --epochs 3"):
        rpg.validate_strict_fair_cli("manifest.json", "external.h5ad", epochs)


def test_cli_rejects_missing_manifest_file(tmp_path: Path) -> None:
    external = tmp_path / "external.h5ad"
    external.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        rpg.validate_strict_fair_cli(str(tmp_path / "missing.json"), str(external), 3)


def test_cli_rejects_legacy_test_name(tmp_path: Path) -> None:
    external = tmp_path / "corneal_20k_test.h5ad"
    external.write_bytes(b"x")
    manifest = _fair_manifest(tmp_path, test=external)
    with pytest.raises(ValueError, match="refuses the legacy cell-level test"):
        rpg.validate_strict_fair_cli(str(manifest), str(external), 3)


def test_cli_rejects_nonzero_overlap(tmp_path: Path) -> None:
    external = tmp_path / "external.h5ad"
    external.write_bytes(b"x")
    manifest = _fair_manifest(tmp_path, overlap=1, test=external)
    with pytest.raises(ValueError, match="donor_overlap must be 0"):
        rpg.validate_strict_fair_cli(str(manifest), str(external), 3)


def test_cli_rejects_mismatched_external_path(tmp_path: Path) -> None:
    recorded = tmp_path / "recorded_test.h5ad"
    recorded.write_bytes(b"a")
    other = tmp_path / "other_test.h5ad"
    other.write_bytes(b"b")
    manifest = _fair_manifest(tmp_path, test=recorded)
    with pytest.raises(ValueError, match="does not match the test path"):
        rpg.validate_strict_fair_cli(str(manifest), str(other), 3)


def test_cli_happy_path_returns_manifest(tmp_path: Path) -> None:
    external = tmp_path / "external.h5ad"
    external.write_bytes(b"x")
    manifest_path = _fair_manifest(tmp_path, test=external)

    manifest = rpg.validate_strict_fair_cli(str(manifest_path), str(external), 3)

    assert manifest["mode"] == "strict_fair"
    assert manifest["donor_overlap"] == 0


# ── run_post_grid: main strict entry + fail-closed gate (F1) ─────────────


def test_main_forwards_strict_args_to_phase1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rpg, "preflight", lambda: None)
    monkeypatch.setattr(rpg, "RESULTS_DIR", tmp_path / "results")
    captured: dict[str, object] = {}
    monkeypatch.setattr(rpg, "phase1_lam0_baseline", lambda **kwargs: captured.update(kwargs))
    phase2_called = []
    monkeypatch.setattr(rpg, "phase2_macro_f1", lambda: phase2_called.append(True))
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    external = tmp_path / "t.h5ad"
    external.write_bytes(b"x")

    # Must not SystemExit on unrecognized args; forwards every strict kwarg.
    rpg.main(
        [
            "--phase", "1",
            "--strict-fair",
            "--split_manifest", str(manifest),
            "--external_test_path", str(external),
            "--epochs", "3",
            "--prepared_train", "prepared.dataset",
        ]
    )

    assert captured["strict_fair"] is True
    assert captured["split_manifest"] == str(manifest)
    assert captured["external_test_path"] == str(external)
    assert captured["epochs"] == 3
    assert captured["prepared_train"] == "prepared.dataset"
    assert phase2_called == []


def test_phase1_strict_fair_fails_closed_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external = tmp_path / "external_test.h5ad"
    external.write_bytes(b"x")
    manifest_path = _fair_manifest(tmp_path, train=tmp_path / "train.h5ad", test=external)

    gpr_calls: list[object] = []
    subprocess_runs: list[object] = []
    monkeypatch.setattr(rpg, "_run_gpr", lambda *args, **kwargs: gpr_calls.append(args))
    monkeypatch.setattr(rpg.subprocess, "run", lambda *args, **kwargs: subprocess_runs.append(args))

    with pytest.raises(NotImplementedError, match="evaluator not wired"):
        rpg.phase1_lam0_baseline(
            strict_fair=True,
            split_manifest=str(manifest_path),
            external_test_path=str(external),
            epochs=3,
        )

    assert gpr_calls == []  # legacy GPR never runs under strict mode
    assert subprocess_runs == []  # no training subprocess is launched


# ── caft_gf: validate_strict_fair_args (fixture-stubbed import) ──────────


def test_caft_requires_split_manifest(caft_gf) -> None:
    with pytest.raises(ValueError, match="requires --split_manifest"):
        caft_gf.validate_strict_fair_args(None, "foo.h5ad", 3)


def test_caft_requires_manifest_file(caft_gf, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        caft_gf.validate_strict_fair_args(str(tmp_path / "missing.json"), "foo.h5ad", 3)


def test_caft_rejects_invalid_json(caft_gf, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(ValueError, match="invalid strict fair manifest"):
        caft_gf.validate_strict_fair_args(str(bad), "foo.h5ad", 3)


def test_caft_rejects_nonzero_overlap_int_and_dict(caft_gf, tmp_path: Path) -> None:
    int_manifest = _fair_manifest(tmp_path, overlap=2, name="int.json")
    with pytest.raises(ValueError, match="donor overlap must be 0"):
        caft_gf.validate_strict_fair_args(str(int_manifest), "foo.h5ad", 3)

    dict_manifest = tmp_path / "dict.json"
    dict_manifest.write_text(
        json.dumps({"protocol": "paper1_fair", "mode": "strict_fair",
                    "donor_overlap": {"count": 1}, "input_paths": {}})
    )
    with pytest.raises(ValueError, match="donor overlap must be 0"):
        caft_gf.validate_strict_fair_args(str(dict_manifest), "foo.h5ad", 3)


def test_caft_rejects_legacy_data_file(caft_gf, tmp_path: Path) -> None:
    manifest = _fair_manifest(tmp_path)
    with pytest.raises(ValueError, match="corneal_100k_subsample.h5ad"):
        caft_gf.validate_strict_fair_args(str(manifest), "corneal_100k_subsample.h5ad", 3)


@pytest.mark.parametrize(
    ("epochs", "match"),
    [(None, "requires explicit --epochs"), (10, "requires --epochs 3")],
)
def test_caft_requires_epochs_three(caft_gf, tmp_path: Path, epochs, match) -> None:
    manifest = _fair_manifest(tmp_path, train=None)  # train=None skips path-match check
    with pytest.raises(ValueError, match=match):
        caft_gf.validate_strict_fair_args(str(manifest), "donor_train.h5ad", epochs)


def test_caft_data_file_must_match_manifest_train(caft_gf, tmp_path: Path) -> None:
    train = tmp_path / "donor_train.h5ad"
    train.write_bytes(b"train")
    other = tmp_path / "other.h5ad"
    other.write_bytes(b"other")
    manifest = _fair_manifest(tmp_path, train=train)
    with pytest.raises(ValueError, match="does not match the donor-disjoint train path"):
        caft_gf.validate_strict_fair_args(str(manifest), str(other), 3)


def test_caft_rejects_wrong_mode_and_protocol(caft_gf, tmp_path: Path) -> None:
    """F4: the manifest must declare strict_fair mode and paper1_fair protocol."""
    wrong_mode = _fair_manifest(tmp_path, mode="legacy", name="mode.json")
    with pytest.raises(ValueError, match="mode must be 'strict_fair'"):
        caft_gf.validate_strict_fair_args(str(wrong_mode), "donor_train.h5ad", 3)

    wrong_protocol = _fair_manifest(tmp_path, protocol="other", name="protocol.json")
    with pytest.raises(ValueError, match="protocol must be 'paper1_fair'"):
        caft_gf.validate_strict_fair_args(str(wrong_protocol), "donor_train.h5ad", 3)


def test_caft_happy_path_returns_manifest_and_sha(caft_gf, tmp_path: Path) -> None:
    train = tmp_path / "donor_train.h5ad"
    train.write_bytes(b"train")
    manifest_path = _fair_manifest(tmp_path, train=train)

    manifest, sha = caft_gf.validate_strict_fair_args(str(manifest_path), str(train), 3)

    assert manifest["mode"] == "strict_fair"
    assert manifest["protocol"] == "paper1_fair"
    assert sha == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


# ── Unbreakable invariant: donor-disjointness by exception everywhere ────


def test_invariant_donor_overlap_raises_across_all_validators(
    caft_gf, tmp_path: Path
) -> None:
    # 1) prepare_paper1_fair_split.validate_metadata
    train = _obs(["D1", "D2"], ["A", "B"])
    test = _obs(["D2", "D3"], ["A", "B"])
    with pytest.raises(ValueError, match="overlap"):
        prep.validate_metadata(train, test)

    # 2) caft_gf.validate_strict_fair_args
    manifest = _fair_manifest(tmp_path, overlap=1, train=tmp_path / "t.h5ad", name="inv1.json")
    with pytest.raises(ValueError, match="overlap"):
        caft_gf.validate_strict_fair_args(str(manifest), "nonlegacy.h5ad", 3)

    # 3) run_post_grid.validate_strict_fair_cli
    external = tmp_path / "external.h5ad"
    external.write_bytes(b"x")
    cli_manifest = tmp_path / "inv2.json"
    cli_manifest.write_text(json.dumps({"donor_overlap": 1, "input_paths": {"test_h5ad": str(external)}}))
    with pytest.raises(ValueError, match="overlap"):
        rpg.validate_strict_fair_cli(str(cli_manifest), str(external), 3)
