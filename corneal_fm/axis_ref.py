"""Frozen differentiation-axis for the this study method arms (ARM-1: axis_ref).

The three new fine-tuning arms (AA-FT, SS, flow-consistency) all need a FIXED
reference for "the epithelial differentiation axis" that the paper's
differentiation-gradient measurement (Spearman rho between the CLS-embedding
axis projection and the ordinal order LSC < TAC < basal < differentiated)
tracks. This module computes that axis ONCE from the *pretrained, frozen*
Geneformer V2-104M on the training cells, caches it to
``artifacts/paper1_method_arms/axis_ref.json`` before any training starts,
and never recomputes it inside a training loop.

Two axis definitions; the NEW PRIMARY is the per-layer-centroid PCA axis
(``mode="centroid_pca"``), the marker-score axis (``mode="marker_score"``) is
retained as the secondary/reference definition. PROTOCOL DEVIATION
2026-08-10 (user sign-off): the AA-FT / soft-Spearman failure diagnosis found
the marker-score training axis only ``cos 0.470`` from the eval-metric
direction; a per-layer-centroid-PCA axis fit on the training cells reaches
external alignment ``rho(proj, ordinal) ~ 0.777``, ``cos(a_int, a_ext) ~
0.818``, so the primary axis definition was switched. ``mode`` is recorded
on every :class:`AxisResult` and baked into the cache key (schema v2), so
the retrain consumes the new axis automatically and never loads the old
marker-score cache record.

Primary definition (``mode="centroid_pca"``, the default) — the eval
metric's training-side analog:

1. Forward the FROZEN pretrained model over ALL epithelial training cells,
   collect the L2-normalised CLS embeddings (final hidden state).
2. ``w`` = the first singular vector of the centred per-layer centroids
   (one centroid = mean CLS embedding per ordinal layer
   LSC < TAC < basal < differentiated, layers present only), oriented so the
   projection increases from the first to the last ordinal layer. This is
   EXACTLY the axis the paper's differentiation-gradient measurement
   (``corneal_fm.evaluation.marker_gradient_correlation``) is defined on —
   the direction that measurement tracks on the training cells. The
   objective value of this axis is reported as ``rho_marker``:
   ``spearmanr(proj_w, ordinal)`` over the axis cells (~0.71 on the frozen
   training cells, per the diagnostic).
3. Reference: ``pca_w`` = the marker-score axis (the OLD primary, computed
   exactly as in the marker_score mode below on the ``<= max_cells``
   subsample) — the marker axis is thereby retained as a reference inside
   the new cache record; ``rho_pca`` is its marker-score Spearman.

Secondary definition (``mode="marker_score"``, the original, kept for
comparison):

1. Marker score per cell (spec)::

       s = mean(zscore(differentiated markers KRT12, KRT3))
           - mean(zscore(basal markers TP63, KRT15, KRT14))

   where zscore is computed across cells and "expression" is the presence of
   the marker token in the cell's Geneformer ``input_ids`` (Geneformer
   collapses duplicate gene ids, so presence is the natural expression
   proxy). Gene symbols are resolved to token ranks through the Geneformer
   ``gene_name_id_dict`` (symbol -> Ensembl id) and ``token_dictionary``
   (Ensembl id -> token rank), both shipped in the model snapshot. Markers
   absent from the vocabulary are skipped with a warning.

2. ``w`` = the unit vector in frozen CLS-embedding space maximising
   Spearman(rho) between ``proj_w(embedding)`` and ``s``. Implemented as
   50 steps of projected gradient ascent on the differentiable soft-rank
   Spearman (``corneal_fm.soft_rank.soft_spearman``), initialised from the
   PCA-1 axis; if the refinement is non-finite or fails to beat PCA-1, the
   PCA-1 axis is returned as the default (``method="pca_fallback"``).

3. Robustness reference: ``pca_w`` = the first PC of the frozen embeddings
   (oriented by correlation with ``s``) — the axis used by the existing
   ``marker_gradient_correlation`` diagnostics, returned alongside ``w``.

Conventions (documented choices):

* The axis is fit on EPITHELIAL training cells only (layer != -1), matching
  the paper's epi-only rho measurement and the epi-only masking used by all
  three arms; non-epithelial cells (~65% of the set) are not structured
  along the differentiation axis and would dilute it.
* The embeddings are L2-normalised CLS vectors, matching the paper's
  layer-wise rho diagnostic (scripts/measure_layerwise_rho.py uses
  ``l2_normalize=True``) and the cached per-cell embeddings
  (artifacts/paper1_tier_b_eval/percell_*.npz are already unit-norm);
  ``w`` therefore lives in normalised embedding space and the arms project
  NORMALISED CLS embeddings onto it. The centroid-PCA axis is invariant to
  this normalisation (verified: raw- vs norm-fit axes agree to 1e-12).
* The centroid-PCA axis is fit on ALL epithelial cells (deterministic, no
  subsample — the diagnostic's a_int used the full internal set); the
  marker-score reference keeps the original ``<= max_cells`` (default 2048)
  stratified subsample because its soft-rank step is O(n^2). The full
  epithelial set is always used for the ids-only marker presence pass,
  which needs no forward at all. Forward batches are length-sorted and
  memory-budgeted (GB10 IOMMU vmalloc OOM protection).

The JSON cache record carries the vector, the mode, the method, provenance
and a ``cache_key`` (built by the training script from the prepared-dataset
path, the model path, the marker-gene set, the axis ``mode`` and the schema
version — ``extra="v2"``) so a rerun with identical inputs loads the cached
axis instead of recomputing it, and a definition change never reuses an
older record.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)
from scipy.stats import spearmanr

from corneal_fm.caft import NON_LAYER
from corneal_fm.soft_rank import soft_spearman

# Marker genes: differentiated (KRT12, KRT3) vs basal (TP63, KRT15, KRT14).
MARKER_GENES: dict[str, list[str]] = {
    "differentiated": ["KRT12", "KRT3"],
    "basal": ["TP63", "KRT15", "KRT14"],
}
_MARKER_GROUPS = list(MARKER_GENES)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "paper1_method_arms"
AXIS_CACHE_FILE = DEFAULT_ARTIFACT_DIR / "axis_ref.json"
# v2 (2026-08-10 protocol deviation, Option A): the PRIMARY axis definition
# changed from the marker-score axis to the per-layer-centroid PCA axis, and
# AxisResult gained a ``mode`` field; the schema bump guarantees an old v1
# record (marker-score cache) is never loaded for the new definition.
SCHEMA = "paper1-method-arms/axis-ref/v2"

# Axis definitions. ``centroid_pca`` (NEW PRIMARY) is the eval metric's
# training-side analog; ``marker_score`` is the original definition, kept as
# the secondary/reference mode.
AXIS_MODES = ("centroid_pca", "marker_score")
DEFAULT_AXIS_MODE = "centroid_pca"

# Axis optimisation hyper-parameters (validated in tests/test_method_arms.py).
OPT_STEPS = 50
OPT_LR = 1.0
OPT_TAU_SCALE = 0.05
MAX_AXIS_CELLS = 2048

_HF_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _load_dict(path_or_dict: str | Path | dict) -> dict:
    """Return the dict itself, or load a pickled dict from ``path_or_dict``."""
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    with open(Path(path_or_dict), "rb") as fh:
        obj = __import__("pickle").load(fh)
    if not isinstance(obj, dict):
        raise TypeError(f"expected a pickled dict at {path_or_dict}; got {type(obj)}")
    return obj


def default_geneformer_dicts() -> tuple[str, str]:
    """Locate the Geneformer snapshot's token + gene-name dicts in the HF cache.

    Returns ``(token_dictionary_file, gene_name_id_dict_file)`` — the
    Ensembl-id -> token-rank and gene-symbol -> Ensembl-id dictionaries
    shipped with the V2-104M snapshot.
    """
    hits = sorted(glob(_HF_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError(
            "Geneformer snapshot not found in HF cache; pass dicts explicitly "
            f"(glob {_HF_SNAP_GLOB})"
        )
    snap = hits[0]
    td = glob(os.path.join(snap, "geneformer", "token_dictionary*.pkl"))
    gd = glob(os.path.join(snap, "geneformer", "gene_name_id_dict*.pkl"))
    if not td or not gd:
        raise FileNotFoundError(f"token/gene-name dicts not found under {snap}/geneformer/")
    return td[0], gd[0]


def marker_token_ranks(
    token_dictionary: str | Path | dict,
    gene_name_id_dictionary: str | Path | dict,
    marker_genes: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, int | None]]:
    """Resolve marker gene symbols to token ranks.

    Returns ``{"differentiated": {"KRT12": rank_or_None, ...}, "basal": {...}}``.
    A gene that is missing from either dictionary resolves to ``None`` (the
    caller skips it with a warning). The Geneformer token dictionary is keyed
    by Ensembl id, so symbols are mapped symbol -> Ensembl id -> token rank.
    """
    genes = marker_genes if marker_genes is not None else MARKER_GENES
    td = _load_dict(token_dictionary)
    gd = _load_dict(gene_name_id_dictionary)
    out: dict[str, dict[str, int | None]] = {}
    for group, syms in genes.items():
        out[group] = {}
        for sym in syms:
            ens = gd.get(sym)
            rank = td.get(ens) if ens is not None else None
            out[group][sym] = None if rank is None else int(rank)
    return out


def batch_presence(ids: torch.Tensor, ranks: dict[str, int | None]) -> dict[str, torch.Tensor]:
    """Per-batch marker presence: ``{gene: (b,) bool}`` for the given ranks.

    ``ids`` is the padded ``(b, seq)`` token tensor; padding is token 0 which
    never equals a marker rank, so presence in the padded row equals presence
    in the original row.
    """
    out: dict[str, torch.Tensor] = {}
    for gene, rank in ranks.items():
        if rank is None:
            continue
        out[gene] = (ids == rank).any(dim=1)
    return out


def marker_scores_from_presence(
    presence: dict[str, np.ndarray],
    marker_genes: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Marker score ``s = mean(z_diff) - mean(z_basal)`` from presence arrays.

    Each ``presence[gene]`` is an ``(n,)`` 0/1 array over the same cells;
    every gene present in ``presence`` is zscored across cells (std == 0 ->
    z = 0). The score is the mean z of the differentiated group minus the
    mean z of the basal group. If one group is entirely missing the contrast
    falls back to the other group (with a warning); if both are missing a
    ``ValueError`` is raised.
    """
    genes = marker_genes if marker_genes is not None else MARKER_GENES
    if not presence:
        raise ValueError("marker presence is empty")
    n = len(next(iter(presence.values())))
    zs: dict[str, np.ndarray] = {}
    for gene, p in presence.items():
        p = np.asarray(p, dtype=float).reshape(-1)
        if p.shape[0] != n:
            raise ValueError(f"presence length mismatch for {gene}: {p.shape[0]} != {n}")
        sd = float(p.std())
        zs[gene] = np.zeros(n, dtype=float) if sd == 0 else (p - p.mean()) / sd
    group_z: dict[str, np.ndarray | None] = {}
    for group, syms in genes.items():
        vals = [zs[g] for g in syms if g in zs]
        group_z[group] = np.mean(vals, axis=0) if vals else None
    if all(v is None for v in group_z.values()):
        raise ValueError(f"no marker gene could be resolved to a token rank (got {list(zs)})")
    score = np.zeros(n, dtype=float)
    for group in _MARKER_GROUPS:
        gz = group_z[group]
        if gz is None:
            continue
        score = score + (-1.0 if group == "basal" else 1.0) * gz
    return score


def pca_unit_axis(embeddings: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA-1 axis: first singular vector of the centred embeddings.

    The singular-vector sign is arbitrary; it is oriented so that the Pearson
    correlation of ``embeddings @ w`` with ``scores`` is positive (the score
    increases along the differentiation axis LSC -> differentiated). Returns
    ``(w, rho)`` with ``rho = spearmanr(proj, scores)``.
    """
    emb = np.asarray(embeddings, dtype=float)
    s = np.asarray(scores, dtype=float)
    if emb.shape[0] < 2:
        raise ValueError("need >= 2 cells for a PCA axis")
    centered = emb - emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    w = vt[0]
    proj = emb @ w
    if np.corrcoef(proj, s)[0, 1] < 0:
        w = -w
        proj = -proj
    rho = float(spearmanr(proj, s).statistic)
    return w, rho


def centroid_pca_axis(embeddings: np.ndarray, ordinals: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-layer-centroid PCA axis — the eval metric's training-side analog.

    Mirrors ``corneal_fm.evaluation.marker_gradient_correlation`` EXACTLY
    (the paper's differentiation-gradient measurement): per-layer centroids
    over the ordinal layers present (LSC < TAC < basal < differentiated),
    centred, first singular vector, oriented so the projection increases
    from the first to the last ordinal layer. Returns ``(w, rho)`` with
    ``rho = spearmanr(embeddings @ w, ordinals)`` — the
    ``marker_gradient_correlation`` objective value of this axis on these
    cells (the number the AA-FT/SS diagnostic predicts as ~0.71 on the
    frozen training cells). ``ordinals`` carries the ordinal layer label of
    every row (epithelial only; NON_LAYER rows must be excluded by the
    caller).
    """
    emb = np.asarray(embeddings, dtype=float)
    ords = np.asarray(ordinals, dtype=float).reshape(-1)
    if emb.shape[0] < 2:
        raise ValueError("need >= 2 cells for a centroid-PCA axis")
    if emb.shape[0] != ords.shape[0]:
        raise ValueError(f"embedding/ordinal length mismatch: {emb.shape[0]} != {ords.shape[0]}")
    present = np.unique(ords).astype(int)
    if len(present) < 2:
        raise ValueError("need >= 2 ordinal layers for a centroid-PCA axis")
    centroids = np.vstack([emb[ords == k].mean(axis=0) for k in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    w = vt[0]
    # Orient LSC -> differentiated (arbitrary SVD sign; identical rule to
    # evaluation.marker_gradient_correlation).
    first_centroid = emb[ords == present[0]].mean(axis=0)
    last_centroid = emb[ords == present[-1]].mean(axis=0)
    if (last_centroid - first_centroid) @ w < 0:
        w = -w
    rho = float(spearmanr(emb @ w, ords).statistic)
    return w, rho


def optimize_axis(
    embeddings: np.ndarray,
    scores: np.ndarray,
    *,
    steps: int = OPT_STEPS,
    lr: float = OPT_LR,
    tau_scale: float = OPT_TAU_SCALE,
    seed: int = 0,
) -> tuple[np.ndarray, str, float]:
    """Refine ``w`` by projected gradient ascent on the soft-rank Spearman.

    Initialised from the PCA-1 axis (fast, well-conditioned start); 50 steps
    of ``w <- normalize(w + lr * grad rho_soft)`` where
    ``rho_soft = soft_spearman(embeddings @ w, scores)``. Falls back to the
    PCA-1 axis if the soft-rank step is undefined or the gradient becomes
    non-finite. Returns ``(w, method, rho_final)`` with ``method`` in
    ``{"soft_spearman_refined", "pca_fallback"}`` and ``rho_final`` the
    exact (scipy) Spearman on the subsample.
    """
    emb = np.asarray(embeddings, dtype=float)
    s = np.asarray(scores, dtype=float)
    w0, rho0 = pca_unit_axis(emb, s)
    emb_t = torch.as_tensor(emb, dtype=torch.float32)
    scores_t = torch.as_tensor(s, dtype=torch.float32)
    w = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
    for _ in range(int(steps)):
        with torch.no_grad():
            w.data = w.data / (w.data.norm() + 1e-12)
        rho, ok = soft_spearman(emb_t @ w, scores_t, tau_scale=tau_scale)
        if not ok:
            return w0, "pca_fallback", rho0
        rho.backward()
        g = w.grad
        if g is None or not bool(torch.isfinite(g).all()):
            return w0, "pca_fallback", rho0
        with torch.no_grad():
            w.data = w.data + lr * g
            w.grad = None
    with torch.no_grad():
        w.data = w.data / (w.data.norm() + 1e-12)
    w_final = w.detach().numpy()
    rho_final = float(spearmanr(emb @ w_final, s).statistic)
    method = "soft_spearman_refined" if rho_final >= rho0 - 1e-6 else "pca_fallback"
    return w_final, method, rho_final


# ---------------------------------------------------------------------------
# Axis result + cache
# ---------------------------------------------------------------------------


@dataclass
class AxisResult:
    """The frozen differentiation axis + reference (mode-dependent).

    ``mode`` records which axis definition produced ``w``:
    ``"centroid_pca"`` (primary, default) or ``"marker_score"`` (reference).

    Field semantics per mode:

    * ``centroid_pca``: ``w`` = per-layer-centroid PCA axis (fit on ALL
      epithelial cells); ``pca_w`` = the marker-score axis (the old primary,
      fit on the subsample — retained as reference); ``rho_marker`` =
      spearmanr(proj_w, ordinal) over the axis cells (the
      marker_gradient_correlation objective on the training cells);
      ``rho_pca`` = spearmanr(proj_pca_w, marker score) on the subsample.
    * ``marker_score``: ``w`` = soft-rank-optimised marker axis (subsample);
      ``pca_w`` = PCA-1 robustness reference; ``rho_marker``/``rho_pca`` =
      spearmanr against the marker score (unchanged from v1).
    """

    w: np.ndarray  # (d,) unit vector, the PRIMARY axis (mode-dependent)
    pca_w: np.ndarray  # (d,) unit vector, the reference axis (mode-dependent)
    mode: str  # "centroid_pca" (primary) | "marker_score" (reference)
    method: str  # "centroid_pca" | "soft_spearman_refined" | "pca_fallback"
    rho_marker: float  # primary axis's objective value (see class docstring)
    rho_pca: float  # reference axis's objective value (see class docstring)
    n_cells: int  # epithelial cells used for the marker scores
    n_axis_cells: int  # cells used to fit the PRIMARY axis
    marker_status: dict[str, dict[str, int | None]]  # gene -> token rank | None
    cache_key: str  # provenance key (dataset + model + markers + mode + schema)
    timestamp: str  # ISO-8601 UTC

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["w"] = np.asarray(self.w, dtype=float).tolist()
        d["pca_w"] = np.asarray(self.pca_w, dtype=float).tolist()
        d["schema"] = SCHEMA
        return d


def _axis_from_dict(d: dict[str, Any]) -> AxisResult:
    return AxisResult(
        w=np.asarray(d["w"], dtype=float),
        pca_w=np.asarray(d["pca_w"], dtype=float),
        mode=str(d.get("mode", "marker_score")),
        method=str(d.get("method", "pca_fallback")),
        rho_marker=float(d.get("rho_marker", float("nan"))),
        rho_pca=float(d.get("rho_pca", float("nan"))),
        n_cells=int(d.get("n_cells", 0)),
        n_axis_cells=int(d.get("n_axis_cells", 0)),
        marker_status=dict(d.get("marker_status", {})),
        cache_key=str(d.get("cache_key", "")),
        timestamp=str(d.get("timestamp", "")),
    )


def cache_axis(axis: AxisResult, cache_path: str | Path) -> Path:
    """Write the axis to ``cache_path`` (JSON); returns the path."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(axis.as_dict(), fh, indent=2, default=str)
    return path


def load_cached_axis(cache_path: str | Path, cache_key: str) -> AxisResult | None:
    """Load the cached axis iff it exists and matches ``cache_key``."""
    path = Path(cache_path)
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if d.get("schema") != SCHEMA or d.get("cache_key") != cache_key:
        return None
    return _axis_from_dict(d)


def axis_cache_key(
    source_data: str | Path,
    model_dir: str | Path,
    marker_genes: dict[str, list[str]] | None = None,
    extra: str = "v2",
    mode: str = DEFAULT_AXIS_MODE,
    source_checksum: str = "",
) -> str:
    """Stable cache key: source dataset + model + marker set + schema + axis mode.

    ``source_data`` is the SOURCE training h5ad (seed-independent), NOT the
    run-local prepared dataset: the frozen axis is fixed by the frozen model
    plus the source cells, so it must be reused across seeds/arms instead of
    recomputed per run. ``source_checksum`` (from the fair-split manifest)
    pins the source content so a changed source never reuses a stale axis.

    ``mode`` is baked into the key (and the schema was bumped to v2 for the
    new primary definition) so a changed axis definition NEVER loads an older
    record — the retrain consumes the fresh axis automatically.
    """
    if mode not in AXIS_MODES:
        raise ValueError(f"unknown axis mode {mode!r}; valid: {AXIS_MODES}")
    genes = marker_genes if marker_genes is not None else MARKER_GENES
    flat = ",".join(f"{g}:{','.join(v)}" for g, v in genes.items())
    payload = (
        f"{Path(source_data).resolve()}|{source_checksum}|{Path(model_dir).resolve()}"
        f"|{flat}|{SCHEMA}|{extra}|mode={mode}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Full computation
# ---------------------------------------------------------------------------


def _stratified_subsample_indices(
    n: int,
    ordinals: np.ndarray | None,
    max_cells: int,
    seed: int,
) -> np.ndarray:
    """Indices of the stratified subsample (see ``_stratified_subsample``).

    Splitting the index selection from the array indexing lets the axis
    computation subsample BEFORE any model forward pass: the axis is fit on
    ``<= max_cells`` cells only, so forwarding the other ~93% of the
    epithelial set would be pure GPU waste for identical results.
    """
    if n <= max_cells:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    if ordinals is None or len(np.unique(ordinals)) < 2:
        return np.asarray(rng.choice(n, size=max_cells, replace=False))
    present = np.unique(ordinals)
    per = max(1, max_cells // len(present))
    chosen: list[np.ndarray] = []
    for k in present:
        pool = np.where(ordinals == k)[0]
        take = min(per, len(pool))
        chosen.append(np.asarray(rng.choice(pool, size=take, replace=False)))
    idx = np.concatenate(chosen)
    if len(idx) > max_cells:
        idx = rng.choice(idx, size=max_cells, replace=False)
    return idx


def _stratified_subsample(
    embeddings: np.ndarray,
    scores: np.ndarray,
    ordinals: np.ndarray | None,
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = embeddings.shape[0]
    idx = _stratified_subsample_indices(n, ordinals, max_cells, seed)
    return embeddings[idx], scores[idx], idx


def compute_axis(
    model,
    train_data,
    class_to_layer: dict,
    token_dictionary: str | Path | dict,
    gene_name_id_dictionary: str | Path | dict,
    *,
    class_to_ordinal: dict | None = None,
    cache_path: str | Path | None = None,
    cache_key: str | None = None,
    max_cells: int = MAX_AXIS_CELLS,
    centroid_batch: int = 256,
    seed: int = 0,
    verbose: bool = True,
    mode: str = DEFAULT_AXIS_MODE,
) -> AxisResult:
    """Compute (or load, if cached) the frozen differentiation axis.

    ``mode`` selects the axis definition (see the module docstring):

    * ``"centroid_pca"`` (PRIMARY, default): forwards the FROZEN ``model``
      over ALL epithelial cells of ``train_data`` (ragged-safe right-padded
      batches, attention mask from ``length``, memory-budgeted batch
      shrinking — mirrors ``ordinal._precompute_centroids``), collects
      L2-normalised CLS embeddings, and fits ``w`` as the first singular
      vector of the centred per-layer centroids, oriented LSC ->
      differentiated (bit-identical to
      ``corneal_fm.evaluation.marker_gradient_correlation``). Marker scores
      (ids-only presence pass, no forward) over the full epithelial set feed
      the REFERENCE ``pca_w`` = marker-score axis (the old primary), fit on
      the ``<= max_cells`` stratified subsample.
    * ``"marker_score"`` (reference/secondary): the original definition —
      soft-rank-optimised marker axis on the subsample (see the v1 docs).

    When ``cache_path`` and ``cache_key`` are given, an existing matching
    record is returned without any forward pass (the axis is fixed before
    training and never recomputed inside training loops). The cache key
    encodes the mode (schema v2), so a definition change never reuses an
    older record.
    """
    if mode not in AXIS_MODES:
        raise ValueError(f"unknown axis mode {mode!r}; valid: {AXIS_MODES}")
    if cache_path is not None and cache_key is not None:
        cached = load_cached_axis(cache_path, cache_key)
        if cached is not None:
            if verbose:
                print(
                    f"[axis_ref] loaded cached axis from {cache_path} "
                    f"(mode={cached.mode}, method={cached.method}, "
                    f"rho_marker={cached.rho_marker:.4f})",
                    flush=True,
                )
            return cached

    ds = train_data
    n = len(ds)
    if n == 0:
        raise ValueError("cannot compute the axis on an empty training set")
    epi_rows = [
        i
        for i, c in enumerate(list(ds["label"]))
        if class_to_layer.get(int(c), NON_LAYER) != NON_LAYER
    ]
    if len(epi_rows) < 2:
        raise ValueError(f"need >= 2 epithelial training cells for the axis; got {len(epi_rows)}")

    device = model.device
    model.eval()
    rank_info = marker_token_ranks(token_dictionary, gene_name_id_dictionary)
    missing = {g for group in rank_info.values() for g, r in group.items() if r is None}
    if missing:
        print(
            f"[axis_ref] WARN marker genes missing from the Geneformer vocab "
            f"(skipped): {sorted(missing)}",
            flush=True,
        )

    # presence accumulation per gene (genes with rank None are skipped)
    active = {g: r for group in rank_info.values() for g, r in group.items() if r is not None}

    def _presence_scores() -> np.ndarray:
        """Marker scores over ALL epi cells (ids-only pass, NO model forward).

        The marker scores are zscored across the full epithelial set, which
        needs only the token rows, not any embedding. Right-pads each ragged
        slice in ONE C++-level op (pad_sequence) — the old per-row Python
        assignment loop was O(rows) interpreter work per batch and dominated
        with small batches over large epi sets.
        """
        pres_list: dict[str, list[np.ndarray]] = {g: [] for g in active}
        with torch.no_grad():
            for s in range(0, len(epi_rows), centroid_batch):
                rows = epi_rows[s : s + centroid_batch]
                batch = ds[rows]
                ids_list = batch["input_ids"]
                ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(row, dtype=torch.long) for row in ids_list],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                for g, r in active.items():
                    pres_list[g].append((ids == r).any(dim=1).cpu().numpy())
        presence = {g: np.concatenate(v) for g, v in pres_list.items() if v}
        return marker_scores_from_presence(presence)

    def _length_order(rows: list[int]) -> np.ndarray | None:
        """Permutation sorting ``rows`` by ascending length (padding tightness).

        ``None`` when the dataset has no ``length`` column. Attention cost
        grows ~quadratically with the padded length, so sorting tightens
        per-batch padding; batch order cannot change the result (each
        embedding is indexed by its own row).
        """
        if "length" not in ds.column_names:
            return None
        return np.asarray(
            sorted(range(len(rows)), key=lambda j: int(ds["length"][rows[j]])),
            dtype=int,
        )

    def _forward_cls(rows: list[int], tag: str = "") -> np.ndarray:
        """(n, d) fp32 L2-normalised CLS embeddings for ``rows``, in order.

        Ragged-safe right-padded batches with the attention mask derived from
        ``length`` (mirrors ``ordinal._precompute_centroids``). Memory
        batching: on the GB10 unified-memory system an oversized per-batch
        attention tensor can exhaust the NVIDIA driver's IOMMU vmalloc
        mappings mid-forward (observed: kernel ``oom_reaper`` SIGKILL of the
        run, rc!=0, no traceback). Keep the peak per-batch footprint bounded
        by shrinking the batch when long cells are present — a pure batching
        change (length-sorted rows => the slice's longest cell sits at its
        end), identical results.
        """
        cls_list: list[torch.Tensor] = []
        has_len_col = "length" in ds.column_names
        budget = max(64, int(centroid_batch) * 2300)
        s = 0
        batch_no = 0
        with torch.no_grad():
            while s < len(rows):
                k = min(centroid_batch, len(rows) - s)
                if has_len_col:
                    while k > 32 and int(ds["length"][rows[s + k - 1]]) * k > budget:
                        k = max(32, budget // int(ds["length"][rows[s + k - 1]]))
                batch = ds[rows[s : s + k]]
                ids_list = batch["input_ids"]
                lengths = batch.get("length")
                if lengths is None:
                    lengths = [len(row) for row in ids_list]
                ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(row, dtype=torch.long) for row in ids_list],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                length = torch.as_tensor(lengths, device=device)
                max_len = ids.shape[1]
                out = model.bert(
                    input_ids=ids,
                    attention_mask=(
                        torch.arange(max_len, device=device)[None, :] < length[:, None]
                    ),
                )
                cls_list.append(F.normalize(out.last_hidden_state[:, 0], dim=-1).float())
                batch_no += 1
                if verbose and (k < centroid_batch or batch_no % 10 == 0):
                    print(
                        f"[axis_ref]{tag} embeddings {s + len(rows)}/{len(rows)}",
                        flush=True,
                    )
                s += k
        return torch.cat(cls_list, dim=0).cpu().numpy()  # (n, d) fp32

    ordinals = None
    if class_to_ordinal is not None:
        labels_epi = [int(ds["label"][i]) for i in epi_rows]
        ordinals = np.asarray([class_to_ordinal.get(c, NON_LAYER) for c in labels_epi], dtype=int)

    if mode == "centroid_pca":
        # --- PRIMARY: per-layer-centroid PCA axis on ALL epithelial cells ---
        if class_to_ordinal is None or ordinals is None:
            raise ValueError(
                "centroid_pca axis mode requires class_to_ordinal "
                "(per-layer centroids need the ordinal labels)"
            )
        if verbose:
            print(
                f"[axis_ref] forward(centroid): device={device} "
                f"epi_cells={len(epi_rows)} centroid_batch={centroid_batch}",
                flush=True,
            )
        rows_all = list(epi_rows)
        order_all = _length_order(rows_all)
        if order_all is not None:
            rows_all = [rows_all[j] for j in order_all]
        emb_all = _forward_cls(rows_all, tag="[centroid]")
        # Ordinals aligned to the (possibly length-sorted) rows: each row's
        # own label, so no index bookkeeping.
        ord_all = np.asarray(
            [class_to_ordinal.get(int(ds["label"][i]), NON_LAYER) for i in rows_all],
            dtype=int,
        )
        if len(emb_all) != len(ord_all):
            raise RuntimeError("embedding/ordinal length mismatch")
        w, rho_ord = centroid_pca_axis(emb_all, ord_all)

        # --- REFERENCE: the marker-score axis (the old primary), fit on the
        # stratified subsample exactly as the marker_score mode does below,
        # so the new cache record carries the marker axis for comparison. ---
        scores = _presence_scores()
        sub_idx = _stratified_subsample_indices(len(epi_rows), ordinals, int(max_cells), seed=seed)
        sub_rows = [epi_rows[i] for i in sub_idx]
        order = _length_order(sub_rows)
        if order is not None:
            sub_rows = [sub_rows[j] for j in order]
        sub_emb = _forward_cls(sub_rows, tag="[marker-ref]")
        sub_scores = scores[sub_idx]
        if order is not None:
            sub_scores = sub_scores[order]
        if len(sub_emb) != len(sub_scores):
            raise RuntimeError("embedding/marker-score length mismatch")
        pca_w, _marker_method, rho_pca = optimize_axis(sub_emb, sub_scores, seed=seed)
        axis = AxisResult(
            w=w,
            pca_w=pca_w,
            mode="centroid_pca",
            method="centroid_pca",
            rho_marker=rho_ord,
            rho_pca=rho_pca,
            n_cells=len(epi_rows),
            n_axis_cells=len(emb_all),
            marker_status=rank_info,
            cache_key=cache_key or "",
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    else:
        # --- marker_score mode: the original definition, unchanged ---------
        scores = _presence_scores()
        # Subsample the axis-optimisation cells BEFORE any forward pass: the
        # axis (``w``, ``pca_w``, rho) is fit on the ``<= max_cells``
        # subsample only, so forwarding the other ~93% of epithelial cells
        # would be pure GPU waste for bit-identical results (same seeded RNG,
        # same cells, same scores). The subsample rows are sorted by
        # ascending length to tighten per-batch padding.
        sub_idx = _stratified_subsample_indices(len(epi_rows), ordinals, int(max_cells), seed=seed)
        sub_rows = [epi_rows[i] for i in sub_idx]
        order = _length_order(sub_rows)
        if order is not None:
            # Apply the length sort to the INDICES too, so sub_emb[i] stays
            # aligned with scores[sub_idx[i]] (embedding/score pairs must
            # match).
            sub_rows = [sub_rows[j] for j in order]
        if verbose:
            print(
                f"[axis_ref] forward: device={device} epi_cells={len(epi_rows)} "
                f"axis_cells={len(sub_rows)} centroid_batch={centroid_batch} "
                f"max_cells={max_cells}",
                flush=True,
            )
        sub_emb = _forward_cls(sub_rows)
        sub_scores = scores[sub_idx]
        if order is not None:
            sub_scores = sub_scores[order]
        if len(sub_emb) != len(sub_scores):
            raise RuntimeError("embedding/marker-score length mismatch")
        w, method, rho_marker = optimize_axis(sub_emb, sub_scores, seed=seed)
        pca_w, rho_pca = pca_unit_axis(sub_emb, sub_scores)
        axis = AxisResult(
            w=w,
            pca_w=pca_w,
            mode="marker_score",
            method=method,
            rho_marker=rho_marker,
            rho_pca=rho_pca,
            n_cells=len(epi_rows),
            n_axis_cells=len(sub_emb),
            marker_status=rank_info,
            cache_key=cache_key or "",
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    model.train()

    if cache_path is not None:
        cache_axis(axis, cache_path)
        if verbose:
            print(
                f"[axis_ref] cached axis -> {cache_path} "
                f"(mode={axis.mode}, method={axis.method}, "
                f"rho_marker={axis.rho_marker:.4f}, rho_pca={axis.rho_pca:.4f})",
                flush=True,
            )
    return axis


def axis_projection_matrix(w: np.ndarray) -> np.ndarray:
    """The rank-1 projection matrix ``P_A = w w^T`` (frozen, used by AA-FT).

    ``P_A`` is a ``(d, d)`` idempotent orthogonal projector onto the axis
    direction; ``||P_A v||^2 = (w . v)^2`` for unit ``w``, which is how the
    AA-FT anchor term is actually evaluated (see ``corneal_fm/axis_ft.py``).
    """
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    return w @ w.T
