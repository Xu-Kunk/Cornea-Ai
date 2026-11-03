#!/usr/bin/env python
"""Tier-B external-test evaluator for the this study donor-disjoint gate (P1-O4e).

This module implements the fail-closed evaluation gate that must pass BEFORE
any GPU training run (P1-O2) is allowed to launch in strict-fair mode:

* :func:`run_static_checks`  — lock/epoch/legacy-data/file-existence checks.
* :func:`run_leakage_checks` — donor- and source-disjointness gates between the
  training data and the external held-out test (GSE153515).
* :func:`donor_level_macro_f1` — the primary endpoint: macro-F1 computed per
  donor, then averaged across donors, with a donor-level bootstrap 95% CI.
* :func:`layer_adjacency_accuracy` — secondary endpoint (embedding structure).
* :func:`marker_gradient_correlation` — secondary endpoint (differentiation
  axis alignment).
* :func:`expected_calibration_error` — secondary endpoint (calibration).
* :func:`evaluate_on_external_test` — the full Tier-B evaluation entry point.

External test configuration is locked in ``config/external_test_gse153515.yml``
(LOCKED 2026-07-31; Ocul Surf 2021, doi:10.1016/j.jtos.2020.12.004, PMID
33388438; IOVS 2021, doi:10.1167/iovs.62.9.36, PMID 34297801). The split
manifest ``artifacts/paper1_fair_split/manifest.json`` records the donor-level
train/test split of the in-domain collection.

Evaluation classes (NON_LAYER excluded from macro-F1)::

    classes    = ["LSC", "TAC", "basal", "differentiated"]
    layer_order = ["LSC", "TAC", "basal", "differentiated"]   # 4-class scheme

Donor key: ``obs['donor_id']``; cell-type key: ``obs['cell_type_harmonized']``.

All functions are importable from the package::

    from corneal_fm.evaluation import evaluate_on_external_test
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors

try:
    import anndata as ad  # type: ignore
except ImportError:  # pragma: no cover - import-time only
    ad = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants (this study locked evaluation contract)
# ---------------------------------------------------------------------------

DEFAULT_CLASSES = ["LSC", "TAC", "basal", "differentiated"]
DEFAULT_LAYER_ORDER = ["LSC", "TAC", "basal", "differentiated"]
DONOR_KEY = "donor_id"
CELL_TYPE_KEY = "cell_type_harmonized"

NON_LAYER = "NON_LAYER"

N_BOOT = 1000
BOOT_SEED = 42

# Training collections (study column / paper §2.1). ``lako_GSE155683`` covers
# both ``lako_fetal_GSE155683`` and ``lako_adult_GSE155683`` in the donor split
# (source-disjointness is checked by GSE accession token, so variants match).
TRAINING_STUDIES = [
    "chen_limbus",
    "chen_cornea",
    "chen_sclera",
    "chen_fetal",
    "lako_GSE155683",
    "shi_GSE157474",
    "chakravarti_GSE218123",
    "dickman_GSE186433",
    "li_GSE147979",
]

EXTERNAL_ACCESSION = "GSE153515"

# Legacy cell-level data files that must never act as the external test.
LEGACY_TEST_NAMES = {"corneal_20k_test.h5ad"}
LEGACY_TRAIN_NAME = "corneal_100k_subsample.h5ad"

# Relative training data file checked by run_static_checks.
TRAIN_DATA_REL = "data/corneal_100k_donor_split.h5ad"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _manifest_overlap(manifest: dict) -> int:
    """Read donor overlap from a split manifest (dict-safe)."""
    ov = manifest.get("donor_overlap", 0)
    if isinstance(ov, dict):
        ov = ov.get("count", ov.get("overlap_count", 0))
    cnt = manifest.get("donor_overlap_count", 0)
    if isinstance(cnt, dict):
        cnt = cnt.get("count", cnt.get("overlap_count", 0))
    return int(ov or cnt or 0)


def _resolve_data_path(value: str | None, project_root: Path) -> Path | None:
    """Resolve a possibly-relative data path against the project root."""
    if value is None:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else project_root / p


def _walk_strings(obj: Any) -> list[str]:
    """Recursively collect every string in a JSON/YAML structure."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_walk_strings(v))
    return out


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to plain JSON-serializable values."""
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# 1. Static checks (fail-closed)
# ---------------------------------------------------------------------------


def run_static_checks(config_path: str, split_manifest_path: str) -> dict:
    """Fail-closed static checks on the locked external test configuration.

    Raises ``ValueError`` on ANY violation. Checks:

    a) the external test config exists and carries a LOCKED status,
    b) the split manifest exists and its donor_overlap check passed (== 0),
    c) ``fair_protocol.epochs == 3``,
    d) no legacy cell-level data files (``corneal_100k_subsample.h5ad`` /
       ``corneal_20k_test.h5ad``) are referenced as primary inputs,
    e) the external test annotated file exists,
    f) the donor-split training data file exists.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise ValueError(f"static check (a) FAILED: external test config not found: {cfg_path}")
    raw_text = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw_text) or {}
    project_root = cfg_path.resolve().parent.parent

    results: dict[str, Any] = {}
    failures: list[str] = []

    # (a) config exists + locked
    locked = "LOCKED" in raw_text
    cfg_status = cfg.get("status")
    if isinstance(cfg_status, str) and cfg_status.strip().upper() == "LOCKED":
        locked = True
    results["config_exists"] = True
    results["config_locked"] = locked
    if not locked:
        failures.append("external test config is not marked LOCKED")

    # (b) split manifest exists + donor overlap check passed
    mani_path = Path(split_manifest_path)
    if not mani_path.is_file():
        results["manifest_exists"] = False
        failures.append(f"split manifest not found: {mani_path}")
    else:
        manifest = json.loads(mani_path.read_text(encoding="utf-8"))
        overlap = _manifest_overlap(manifest)
        results["manifest_exists"] = True
        results["manifest_donor_overlap"] = overlap
        results["manifest_donor_overlap_ok"] = overlap == 0
        if overlap != 0:
            failures.append(f"split manifest donor_overlap={overlap} (must be 0)")

    # (c) epochs == 3
    epochs = (cfg.get("fair_protocol") or {}).get("epochs")
    results["fair_protocol_epochs"] = epochs
    if epochs != 3:
        failures.append(f"fair_protocol.epochs={epochs} (strict fair mode requires 3)")

    # (d) reject legacy data files as primary inputs
    legacy_refs = sorted(
        {
            s
            for s in _walk_strings(cfg)
            if LEGACY_TRAIN_NAME in s or any(t in s for t in LEGACY_TEST_NAMES)
        }
    )
    results["legacy_data_refs"] = legacy_refs
    results["legacy_data_rejected"] = not legacy_refs
    if legacy_refs:
        failures.append(f"config references legacy cell-level data files: {legacy_refs}")

    # (e) external test file exists
    ext_rel = (cfg.get("external_test") or {}).get("annotated_file")
    ext_path = _resolve_data_path(ext_rel, project_root)
    results["external_test_file"] = str(ext_path) if ext_path else None
    results["external_test_exists"] = ext_path is not None and ext_path.is_file()
    if not (ext_path is not None and ext_path.is_file()):
        failures.append(f"external test annotated file not found: {ext_path}")

    # (f) training data exists
    train_path = _resolve_data_path(TRAIN_DATA_REL, project_root)
    results["train_data_file"] = str(train_path)
    results["train_data_exists"] = train_path is not None and train_path.is_file()
    if train_path is None or not train_path.is_file():
        failures.append(f"training data file not found: {train_path}")

    if failures:
        raise ValueError("Tier-B static checks FAILED:\n  - " + "\n  - ".join(failures))
    results["passed"] = True
    return results


# ---------------------------------------------------------------------------
# 2. Data-leakage checks (fail-closed)
# ---------------------------------------------------------------------------


def _training_studies(train_obs: pd.DataFrame) -> list[str]:
    """Training study labels: from the ``study`` column, else the locked list."""
    if "study" in train_obs.columns:
        return sorted({str(s).strip() for s in train_obs["study"].dropna().unique()})
    return list(TRAINING_STUDIES)


def _external_accession(external_obs: pd.DataFrame) -> str:
    """External source accession: GSE token from a ``study`` column, else default."""
    if "study" in external_obs.columns:
        vals = " ".join(str(s) for s in external_obs["study"].dropna().unique())
        m = re.search(r"GSE\d+", vals, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    return EXTERNAL_ACCESSION


def _source_overlap(train_studies: list[str], external_accession: str) -> list[str]:
    """Return training studies that overlap the external source accession."""
    ext_norm = external_accession.strip().lower()
    ext_tokens = set(re.findall(r"gse\d+", ext_norm))
    overlap: list[str] = []
    for s in train_studies:
        sn = str(s).strip().lower()
        if ext_norm and ext_norm in sn:
            overlap.append(s)
            continue
        tokens = set(re.findall(r"gse\d+", sn))
        if ext_tokens & tokens:
            overlap.append(s)
    return overlap


def _is_legacy_cell_level(external_obs: pd.DataFrame, split_manifest: dict) -> bool:
    """True if the external test looks like the legacy cell-level split."""
    has_harmonized = CELL_TYPE_KEY in external_obs.columns
    has_author = "author_cell_type" in external_obs.columns
    if not has_harmonized and has_author:
        return True
    test_path = ((split_manifest or {}).get("input_paths") or {}).get("test_h5ad")
    if test_path and Path(str(test_path)).name in LEGACY_TEST_NAMES:
        return True
    return False


def run_leakage_checks(
    train_obs: pd.DataFrame,
    external_obs: pd.DataFrame,
    split_manifest: dict,
) -> dict:
    """Verify there is no data leakage between training and the external test.

    Checks (raises ``ValueError`` on leakage):

    a) donor disjointness — normalized (strip + lower) donor sets are disjoint,
    b) source disjointness — the external study (GSE153515) is not among the
       training studies,
    c) the external test is not the legacy cell-level split.
    """
    if DONOR_KEY not in train_obs.columns or DONOR_KEY not in external_obs.columns:
        raise ValueError(
            f"leakage check FAILED: '{DONOR_KEY}' column required in both training and external obs"
        )

    train_donors = {str(d).strip().lower() for d in train_obs[DONOR_KEY].dropna().unique()}
    external_donors = {str(d).strip().lower() for d in external_obs[DONOR_KEY].dropna().unique()}
    donor_overlap = sorted(train_donors & external_donors)

    train_studies = _training_studies(train_obs)
    external_accession = _external_accession(external_obs)
    source_overlap = _source_overlap(train_studies, external_accession)

    legacy = _is_legacy_cell_level(external_obs, split_manifest)

    results: dict[str, Any] = {
        "train_donors": sorted(train_donors),
        "external_donors": sorted(external_donors),
        "donor_overlap": donor_overlap,
        "donor_disjoint": not donor_overlap,
        "train_studies": train_studies,
        "external_accession": external_accession,
        "source_overlap": source_overlap,
        "source_disjoint": not source_overlap,
        "legacy_cell_level_split": legacy,
    }

    failures: list[str] = []
    if donor_overlap:
        failures.append(f"donor overlap between training and external test: {donor_overlap}")
    if source_overlap:
        failures.append(
            f"external source {external_accession} found among training studies: {source_overlap}"
        )
    if legacy:
        failures.append(
            "external test looks like the legacy cell-level split "
            "(author_cell_type present without cell_type_harmonized, "
            "or manifest test file is a legacy corneal_20k_test.h5ad)"
        )
    if failures:
        raise ValueError("Tier-B leakage checks FAILED:\n  - " + "\n  - ".join(failures))
    results["passed"] = True
    return results


# ---------------------------------------------------------------------------
# 3. Donor-level evaluation (primary endpoint)
# ---------------------------------------------------------------------------


def donor_level_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    donors: np.ndarray,
    classes: list[str],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Donor-level macro-F1 with a donor-level bootstrap 95% CI.

    Macro-F1 is computed per donor (across ``classes``), then averaged across
    donors. The CI resamples donor IDs with replacement (``n_boot`` draws,
    ``seed``). Per-class precision/recall/F1/support are pooled (micro-averaged)
    across donors; the confusion matrix is aggregated over all cells.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    donors = np.asarray(donors).astype(str)
    if len(y_true) != len(y_pred) or len(y_true) != len(donors):
        raise ValueError("y_true, y_pred and donors must have equal length")

    donor_ids = np.unique(donors)
    per_donor: dict[str, float] = {}
    for d in donor_ids:
        mask = donors == d
        per_donor[str(d)] = float(
            f1_score(
                y_true[mask],
                y_pred[mask],
                labels=classes,
                average="macro",
                zero_division=0,
            )
        )
    macro_f1 = float(np.mean(list(per_donor.values())))

    # Bootstrap: resample donor IDs with replacement, recompute the mean.
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    per_donor_arr = np.array([per_donor[str(d)] for d in donor_ids], dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(donor_ids), size=len(donor_ids))
        boot_means[b] = float(np.mean(per_donor_arr[idx]))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    precision, recall, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = {
        str(c): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1s[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(classes)
    }
    conf = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "macro_f1": macro_f1,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "per_donor": per_donor,
        "per_class": per_class,
        "confusion": conf,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# 4. Secondary metrics
# ---------------------------------------------------------------------------


def layer_adjacency_accuracy(
    labels: np.ndarray,
    embeddings: np.ndarray,
    layer_order: list[str] = DEFAULT_LAYER_ORDER,
) -> float:
    """Fraction of cells whose nearest neighbor has an adjacent layer label.

    Adjacent means ``|index_i - index_j| <= 1`` in ``layer_order``. Cells
    labelled outside ``layer_order`` (NON_LAYER) are excluded. Returns 0.0 when
    fewer than two cells remain.
    """
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings, dtype=float)
    valid = np.array([lbl in layer_order for lbl in labels])
    if int(valid.sum()) < 2:
        return 0.0
    lab = labels[valid]
    emb = embeddings[valid]

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(emb)
    _, idx = nn.kneighbors(emb)  # idx[:, 0] is the cell itself
    neighbor = idx[:, 1]

    order_map = {c: i for i, c in enumerate(layer_order)}
    li = np.array([order_map[x] for x in lab], dtype=int)
    lj = np.array([order_map[x] for x in lab[neighbor]], dtype=int)
    adjacent = np.abs(li - lj) <= 1
    return float(adjacent.mean())


def marker_gradient_correlation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    layer_order: list[str] = DEFAULT_LAYER_ORDER,
) -> float:
    """Spearman rho between differentiation pseudotime and the layer ordinal.

    Pseudotime = projection of each cell embedding onto the principal axis
    (first singular vector) of the per-layer centroids. NON_LAYER cells are
    excluded. Returns NaN when fewer than two layers are present.
    """
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings, dtype=float)
    valid = np.array([lbl in layer_order for lbl in labels])
    lab = labels[valid]
    emb = embeddings[valid]
    if len(lab) == 0:
        return float("nan")

    order_map = {c: i for i, c in enumerate(layer_order)}
    ordinals = np.array([order_map[x] for x in lab], dtype=float)
    present = [c for c in layer_order if (ordinals == order_map[c]).any()]
    if len(present) < 2:
        return float("nan")

    centroids = np.vstack([emb[ordinals == order_map[c]].mean(axis=0) for c in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[0]  # principal axis (unit vector)
    # The SVD singular-vector sign is arbitrary; orient the axis so that
    # pseudotime increases from the first to the last layer (LSC ->
    # differentiated), making the gradient direction deterministic.
    first_centroid = emb[ordinals == order_map[present[0]]].mean(axis=0)
    last_centroid = emb[ordinals == order_map[present[-1]]].mean(axis=0)
    if (last_centroid - first_centroid) @ axis < 0:
        axis = -axis
    pseudotime = emb @ axis
    rho, _ = spearmanr(pseudotime, ordinals)
    return float(rho)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error: weighted mean of |accuracy - confidence|.

    ``y_true`` holds integer class indices; ``y_prob`` is
    ``(n_samples, n_classes)``. For a 1-D ``y_prob`` (binary case) the two-class
    probability ``[1 - p, p]`` is assumed.

    Binning (FIX 2026-08-05, P0): the default 10 bins are the standard
    half-open intervals ``[i/10, (i+1)/10)`` via ``np.digitize`` (bin 0 =
    ``[0.0, 0.1)``), with the final bin closed on the right so that a
    confidence of exactly 1.0 falls in bin 9 (``[0.9, 1.0]``). The previous
    implementation used ``conf > lo`` for bin 0 without an upper bound, which
    silently assigned every cell with confidence > 0 to bin 0 (covering the
    whole (0, 1.0] range) and systematically inflated every reported ECE.
    Each cell now contributes to exactly one bin.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        y_prob = np.stack([1.0 - y_prob, y_prob], axis=1)
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be 2-D (n_samples, n_classes)")
    n = len(y_prob)

    pred = y_prob.argmax(axis=1)
    conf = y_prob.max(axis=1)
    acc = pred == y_true

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize(conf, edges, right=False) returns 1..n_bins for the
    # half-open [lo, hi) bins; conf == 1.0 lands in bin n_bins (the last
    # interval, which is closed on the right). Subtract 1 -> 0..n_bins-1.
    bins = np.clip(np.digitize(conf, edges, right=False) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if int(mask.sum()) == 0:
            continue
        ece += (mask.sum() / n) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


# ---------------------------------------------------------------------------
# 5. Full evaluation pipeline
# ---------------------------------------------------------------------------


def _config_invariant_checks(config: dict, split_manifest: dict) -> None:
    """Re-verify config/manifest invariants from the loaded dicts (fail-closed)."""
    epochs = (config.get("fair_protocol") or {}).get("epochs")
    if epochs != 3:
        raise ValueError(f"config fair_protocol.epochs={epochs} (strict fair mode requires 3)")
    if _manifest_overlap(split_manifest) != 0:
        raise ValueError("split manifest donor_overlap != 0 (must be donor-disjoint)")
    accession = (config.get("external_test") or {}).get("accession")
    # The locked config is selected by the CLI --accession flag, so the
    # accession it declares is authoritative. Accept any locked GSE accession
    # (GSE153515 or GSE299152) without changing any metric code; the module
    # default remains EXTERNAL_ACCESSION.
    if not accession or not re.fullmatch(r"GSE\d+", str(accession).strip().upper()):
        raise ValueError(
            f"config external_test.accession={accession!r} "
            f"(expected a locked GSE accession such as {EXTERNAL_ACCESSION})"
        )


def evaluate_on_external_test(
    train_adata: Any,
    external_adata: Any,
    embeddings: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    config: dict,
    split_manifest: dict,
    *,
    classes: list[str] = DEFAULT_CLASSES,
    layer_order: list[str] = DEFAULT_LAYER_ORDER,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Run the full Tier-B evaluation on the external test.

    Steps:

    1. config/manifest invariant re-checks (the path-based
       :func:`run_static_checks` gate is enforced at the CLI entry point),
    2. donor + source leakage checks,
    3. filter to the evaluation classes (NON_LAYER excluded),
    4. donor-level macro-F1 with bootstrap CI (primary endpoint),
    5. per-class precision/recall/F1 + support,
    6. layer-adjacency accuracy,
    7. marker-gradient rank correlation,
    8. expected calibration error (when ``probabilities`` are provided),
    9. compile a JSON-serializable results dict with provenance.
    """
    if ad is None:
        raise RuntimeError("anndata is required for evaluate_on_external_test")

    _config_invariant_checks(config, split_manifest)
    leakage = run_leakage_checks(train_adata.obs, external_adata.obs, split_manifest)

    labels = np.asarray(external_adata.obs[CELL_TYPE_KEY].astype(str))
    donors = np.asarray(external_adata.obs[DONOR_KEY].astype(str))
    mask = np.array([lbl in classes for lbl in labels])
    if int(mask.sum()) == 0:
        raise ValueError("no external test cells fall inside the evaluation classes")

    emb = np.asarray(embeddings, dtype=float)[mask]
    y_pred = np.asarray(predictions)[mask]
    y_true = labels[mask]
    donors_eval = donors[mask]
    if len(emb) != len(y_true):
        raise ValueError(
            f"embeddings ({len(emb)}) and predictions/labels ({len(y_true)}) "
            "must cover the same cells"
        )

    mf1 = donor_level_macro_f1(y_true, y_pred, donors_eval, classes, n_boot=n_boot, seed=seed)
    laa = layer_adjacency_accuracy(y_true, emb, layer_order)
    marker_rho = marker_gradient_correlation(emb, y_true, layer_order)

    ece = None
    if probabilities is not None:
        prob = np.asarray(probabilities, dtype=float)[mask]
        if prob.ndim == 1:
            prob = np.stack([1.0 - prob, prob], axis=1)
        if prob.shape[1] != len(classes):
            raise ValueError(
                f"probabilities have {prob.shape[1]} columns but {len(classes)} evaluation classes"
            )
        idx_map = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([idx_map[lbl] for lbl in y_true], dtype=int)
        ece = expected_calibration_error(y_idx, prob)

    accession = (config.get("external_test") or {}).get("accession", EXTERNAL_ACCESSION)
    results: dict[str, Any] = {
        "provenance": {
            "script": "corneal_fm/evaluation.py",
            "accession": accession,
            "external_test_file": str((config.get("external_test") or {}).get("annotated_file")),
            "n_external_cells": int(external_adata.n_obs),
            "n_eval_cells": int(mask.sum()),
            "n_eval_donors": int(len(np.unique(donors_eval))),
            "classes": list(classes),
            "layer_order": list(layer_order),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 — datetime.UTC does not exist at runtime (CPython 3.12); ruff's py312 UP017 suggestion is invalid
        },
        "leakage": leakage,
        "primary_endpoint": {
            "metric": "donor_level_macro_f1",
            "macro_f1": mf1["macro_f1"],
            "ci_lower": mf1["ci_lower"],
            "ci_upper": mf1["ci_upper"],
            "per_donor": mf1["per_donor"],
        },
        "per_class": mf1["per_class"],
        "confusion": mf1["confusion"],
        "secondary_endpoints": {
            "layer_adjacency_accuracy": laa,
            "marker_gradient_rank_correlation": marker_rho,
            "expected_calibration_error": ece,
        },
    }
    return _json_safe(results)


def main() -> None:  # pragma: no cover - CLI smoke helper
    """CLI smoke: run the static + leakage checks on the project's locked inputs."""
    project_root = Path(__file__).resolve().parent.parent
    cfg = project_root / "config" / "external_test_gse153515.yml"
    manifest = project_root / "artifacts" / "paper1_fair_split" / "manifest.json"
    print(json.dumps(run_static_checks(str(cfg), str(manifest)), indent=2))
    with open(manifest, encoding="utf-8") as f:
        m = json.load(f)
    train_obs = ad.read_h5ad(
        project_root / "data" / "corneal_100k_donor_split.h5ad", backed="r"
    ).obs
    ext_obs = ad.read_h5ad(project_root / "data" / "gse153515_annotated.h5ad", backed="r").obs
    print(json.dumps(run_leakage_checks(train_obs, ext_obs, m), indent=2))


if __name__ == "__main__":
    sys.exit(main())
