"""Unit tests for the Paper 1 method arms (axis_ref / axis_ft / soft_spearman / flow).

CPU-only tiny-model tests (mirroring the hpa_callback_feasibility pattern:
tiny Bert + ragged dataset):

(a) axis_ref computes a finite unit vector for BOTH axis definitions
    (centroid_pca — the new primary — and marker_score — the retained
    reference), caches/reloads per mode, and keys the cache by mode,
(b) axis_ft lambda=0 == pure CE loss; lambda>0 reduces ||P_A(h - h0)||,
(c) soft_spearman matches scipy.stats.spearmanr within tolerance + gradients
    flow,
(d) flow term is finite + gradients flow,
(e) all three arms train 2 steps on the tiny model without NaN,
(f) the launcher's Phase A summary aggregation writes phase_a_summary.csv.

No GPU, no real data files: a tiny 2-layer BertForSequenceClassification, a
synthetic ragged cell dataset (4 ordinal classes + non-epithelial), and
synthetic Geneformer gene-name/token dictionaries.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corneal_fm.axis_ft import attn_mask, build_axis_ft_trainer  # noqa: E402
from corneal_fm.axis_ref import (  # noqa: E402
    axis_cache_key,
    axis_projection_matrix,
    centroid_pca_axis,
    compute_axis,
    load_cached_axis,
    marker_token_ranks,
)
from corneal_fm.caft import class_to_layer_map  # noqa: E402
from corneal_fm.flow_consistency import build_flow_trainer  # noqa: E402
from corneal_fm.ordinal import class_to_ordinal_map  # noqa: E402
from corneal_fm.soft_rank import soft_spearman  # noqa: E402
from corneal_fm.soft_spearman import build_ss_trainer  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixtures (self-contained; no geneformer dependency)
# ---------------------------------------------------------------------------

# 31-class stand-in: 4 epithelial ordinal classes + 2 non-epithelial.
IDC = {
    0: "LPC",
    1: "Limbus_Epi-C1",
    2: "Epi-PMC",
    3: "Epi-TDC",
    4: "Conj-Epi",
    5: "Unknown-DCN+",
}
N_EPI_CLASSES = 4
CELLS_PER_CLASS = 8
N_NON_EPI = 8

# Synthetic Geneformer dicts: symbol -> Ensembl id -> token rank.
TOKEN_DICT = {
    "<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3,
    "ENSG_KRT12": 10, "ENSG_KRT3": 11, "ENSG_TP63": 12,
    "ENSG_KRT15": 13, "ENSG_KRT14": 14, "ENSG_OTHER": 15,
}
GENE_DICT = {
    "KRT12": "ENSG_KRT12", "KRT3": "ENSG_KRT3", "TP63": "ENSG_TP63",
    "KRT15": "ENSG_KRT15", "KRT14": "ENSG_KRT14",
}
COLLATOR_DICT = {"<pad>": 0, "<mask>": 1, "<cls>": 2}

HIDDEN = 32
N_LAYERS = 2
ANCHOR_LAYER = N_LAYERS  # final hidden-state index for the tiny model


def tiny_model() -> torch.nn.Module:
    """Fresh tiny BertForSequenceClassification (4-class head), CPU fp32."""
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=4,
        pad_token_id=0,
    )
    return BertForSequenceClassification(cfg)


def _cell_row(cls: int, i: int) -> dict:
    """One ragged cell row; marker tokens vary by ordinal class."""
    marker = {0: [], 1: [13], 2: [12, 13, 14], 3: [10, 11]}.get(cls, [])
    tokens = [2] + marker + [100 + i % 40, 101 + i % 13, 3]
    return {"input_ids": tokens, "length": len(tokens), "label": cls, "idx": i}


def tiny_rows() -> list[dict]:
    rows: list[dict] = []
    i = 0
    for cls in range(N_EPI_CLASSES):
        for _ in range(CELLS_PER_CLASS):
            rows.append(_cell_row(cls, i))
            i += 1
    for _ in range(N_NON_EPI):  # Conj-Epi (non-epithelial)
        rows.append(_cell_row(4, i))
        i += 1
    return rows


def tiny_dataset():
    from datasets import Dataset

    return Dataset.from_list(tiny_rows())


def collator():
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    return DataCollatorForCellClassification(token_dictionary=COLLATOR_DICT)


def collate(rows: list[dict]) -> dict:
    """Collate a list of rows into the batch dict compute_loss expects."""
    return collator()(rows)


def minimal_targs(output_dir: Path) -> dict:
    return {
        "num_train_epochs": 1,
        "max_steps": 2,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "weight_decay": 0.001,
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        "fp16": False,
        "bf16": False,
        "use_cpu": True,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "report_to": [],
    }


def unit_vector(d: int = HIDDEN, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def axis_w_for(model) -> np.ndarray:
    """Compute a real axis from the tiny model (used by all arm tests)."""
    axis = compute_axis(
        model, tiny_dataset(), class_to_layer_map(IDC),
        TOKEN_DICT, GENE_DICT, class_to_ordinal=class_to_ordinal_map(IDC),
        cache_path=None, max_cells=2000, seed=0,
    )
    return axis.w


# ---------------------------------------------------------------------------
# (a) axis_ref: finite unit vector (both modes), cache round-trip, P_A
# ---------------------------------------------------------------------------

def test_axis_ref_centroid_pca_primary(tmp_path: Path) -> None:
    """The NEW primary mode: centroid-PCA axis, finite/unit, cached."""
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    ds = tiny_dataset()
    cache = tmp_path / "axis_ref.json"
    key = "test-key-centroid"

    axis = compute_axis(
        model, ds, class_to_layer_map(IDC), TOKEN_DICT, GENE_DICT,
        class_to_ordinal=class_to_ordinal_map(IDC),
        cache_path=cache, cache_key=key, max_cells=2000, seed=0,
    )
    assert axis.mode == "centroid_pca"
    assert axis.method == "centroid_pca"
    assert np.isfinite(axis.w).all()
    assert np.isfinite(axis.pca_w).all()
    assert abs(np.linalg.norm(axis.w) - 1.0) < 1e-3
    assert abs(np.linalg.norm(axis.pca_w) - 1.0) < 1e-3
    assert axis.n_cells == N_EPI_CLASSES * CELLS_PER_CLASS  # epi-only
    assert axis.n_axis_cells == axis.n_cells  # centroid axis uses ALL epi cells
    assert np.isfinite(axis.rho_marker)  # spearmanr(proj_w, ordinal)
    assert np.isfinite(axis.rho_pca)
    assert cache.is_file()

    # cache round-trip: reload returns the identical axis (no recompute)
    axis2 = compute_axis(
        model, ds, class_to_layer_map(IDC), TOKEN_DICT, GENE_DICT,
        class_to_ordinal=class_to_ordinal_map(IDC),
        cache_path=cache, cache_key=key, max_cells=2000, seed=0,
    )
    assert np.allclose(axis.w, axis2.w)
    assert axis2.method == axis.method
    assert axis2.mode == "centroid_pca"


def test_centroid_pca_axis_recovers_known_direction() -> None:
    """centroid_pca_axis finds the first PC of the per-layer centroids.

    Synthetic embeddings whose per-class centroids lie along a known unit
    direction v (with in-class noise) must yield w == +/- v, oriented so the
    projection increases from ordinal 0 to ordinal 3, and rho == 1.0.
    """
    rng = np.random.default_rng(0)
    d = 16
    v = rng.normal(size=d)
    v = v / np.linalg.norm(v)
    ords = np.repeat(np.arange(4, dtype=float), 25)
    # centroids advance monotonically along v: 0, 1, 2, 3 * v
    emb = np.vstack(
        [ords[:, None] * v + rng.normal(scale=0.05, size=(len(ords), d))]
    )
    w, rho = centroid_pca_axis(emb, ords)
    assert abs(float(np.clip(w @ v, -1, 1))) > 0.999
    # oriented LSC -> differentiated: proj increases with the ordinal
    assert float(np.corrcoef(emb @ w, ords)[0, 1]) > 0.9
    assert rho > 0.9  # nearly-perfect ordinal Spearman (in-class noise only)


def test_axis_ref_marker_score_reference(tmp_path: Path) -> None:
    """The retained reference mode (original definition) still works."""
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    ds = tiny_dataset()
    cache = tmp_path / "axis_ref.json"

    axis = compute_axis(
        model, ds, class_to_layer_map(IDC), TOKEN_DICT, GENE_DICT,
        class_to_ordinal=class_to_ordinal_map(IDC),
        cache_path=cache, cache_key="test-key-marker", max_cells=2000, seed=0,
        mode="marker_score",
    )
    assert axis.mode == "marker_score"
    assert axis.method in {"soft_spearman_refined", "pca_fallback"}
    assert np.isfinite(axis.w).all()
    assert abs(np.linalg.norm(axis.w) - 1.0) < 1e-3
    assert axis.n_axis_cells <= 2000  # marker mode fits on the subsample
    assert cache.is_file()

    # reload under the marker key returns the marker axis
    axis2 = compute_axis(
        model, ds, class_to_layer_map(IDC), TOKEN_DICT, GENE_DICT,
        class_to_ordinal=class_to_ordinal_map(IDC),
        cache_path=cache, cache_key="test-key-marker", max_cells=2000, seed=0,
        mode="marker_score",
    )
    assert np.allclose(axis.w, axis2.w)
    # a different key must NOT hit the cache
    assert load_cached_axis(cache, "other-key") is None


def test_axis_cache_key_encodes_mode(tmp_path: Path) -> None:
    """The cache key MUST differ between axis definitions (schema v2)."""
    ds_path = str(tmp_path / "prepared")
    model_dir = str(tmp_path / "model")
    k_centroid = axis_cache_key(ds_path, model_dir)
    k_marker = axis_cache_key(ds_path, model_dir, mode="marker_score")
    assert k_centroid != k_marker
    # the new primary's default key differs from the old v1 key space
    k_old_style = axis_cache_key(ds_path, model_dir, extra="v1", mode="marker_score")
    assert k_old_style != k_centroid


def test_axis_ref_rank_resolution_and_projector() -> None:
    """Marker rank resolution + P_A = w w^T projector (mode-independent)."""
    ranks = marker_token_ranks(TOKEN_DICT, GENE_DICT)
    assert ranks["differentiated"]["KRT12"] == 10
    assert ranks["basal"]["KRT14"] == 14

    w = unit_vector(seed=3)
    proj_mat = axis_projection_matrix(w)
    assert proj_mat.shape == (HIDDEN, HIDDEN)
    assert np.allclose(proj_mat @ w, w, atol=1e-6)
    assert np.allclose(proj_mat @ proj_mat, proj_mat, atol=1e-6)


# ---------------------------------------------------------------------------
# (b) axis_ft: lambda=0 == CE; lambda>0 reduces axis drift
# ---------------------------------------------------------------------------

def _reference_ce(model, batch, class_to_ordinal) -> torch.Tensor:
    input_ids = batch["input_ids"]
    length = batch["length"]
    labels = batch["labels"]
    out = model.bert(input_ids=input_ids, attention_mask=attn_mask(input_ids, length))
    logits = model.classifier(model.dropout(out.pooler_output))
    ordinals = torch.tensor(
        [class_to_ordinal.get(int(c), -1) for c in labels.tolist()]
    )
    valid = ordinals != -1
    return F.cross_entropy(logits[valid], ordinals[valid])


def _axis_drift(model, w: np.ndarray, batch, h0) -> float:
    """mean over epi cells of (w . (h - h0))^2 = ||P_A (h - h0)||^2 (unit w)."""
    input_ids = batch["input_ids"]
    length = batch["length"]
    labels = batch["labels"]
    class_to_ordinal = class_to_ordinal_map(IDC)
    with torch.no_grad():
        out = model.bert(
            input_ids=input_ids,
            attention_mask=attn_mask(input_ids, length),
            output_hidden_states=True,
        )
        h = out.hidden_states[ANCHOR_LAYER][:, 0].float()
    w_t = torch.tensor(w, dtype=torch.float32)
    proj = (h - h0) @ w_t
    ordinals = torch.tensor(
        [class_to_ordinal.get(int(c), -1) for c in labels.tolist()]
    )
    valid = ordinals != -1
    return float((proj[valid] ** 2).mean())


def _frozen_cls(model, batch) -> torch.Tensor:
    """CLS at the anchor layer from a (pristine) model, detached."""
    input_ids = batch["input_ids"]
    with torch.no_grad():
        out = model.bert(
            input_ids=input_ids,
            attention_mask=attn_mask(input_ids, batch["length"]),
            output_hidden_states=True,
        )
        return out.hidden_states[ANCHOR_LAYER][:, 0].float().clone()


def test_axis_ft_lambda_zero_equals_ce(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = tiny_model()
    model.eval()  # deterministic (no dropout) so both forwards agree exactly
    w = unit_vector(seed=3)
    batch = collate(tiny_rows()[:16])  # classes 0..1 + a couple of non-epi

    trainer = build_axis_ft_trainer(
        model, tiny_dataset(), IDC, w, axis_lambda=0.0,
        training_args_kwargs=minimal_targs(tmp_path / "out0"),
        output_directory=str(tmp_path / "out0"),
        token_dictionary_file=None,
    )
    loss = trainer.compute_loss(model, {k: v.clone() for k, v in batch.items()})
    ref = _reference_ce(model, batch, class_to_ordinal_map(IDC))
    assert float(loss.detach()) == pytest.approx(float(ref.detach()), rel=1e-5)


def test_axis_ft_reduces_axis_drift(tmp_path: Path) -> None:
    torch.manual_seed(2)
    m0 = tiny_model()
    m1 = tiny_model()  # identical init (same seed)
    w = axis_w_for(m0)
    rows = tiny_rows()
    # Training + holdout batches that span ALL ordinal classes (the anchor
    # constrains every class, so its effect transfers to the holdout).
    train_mix = rows[0:4] + rows[8:12] + rows[16:20] + rows[24:28] + rows[32:34]
    eval_mix = rows[4:8] + rows[12:16] + rows[20:24] + rows[28:32] + rows[34:36]
    train_batch = collate(train_mix)
    eval_batch = collate(eval_mix)

    t0 = build_axis_ft_trainer(
        m0, tiny_dataset(), IDC, w, axis_lambda=0.0,
        training_args_kwargs=minimal_targs(tmp_path / "lam0"),
        output_directory=str(tmp_path / "lam0"),
    )
    t1 = build_axis_ft_trainer(
        m1, tiny_dataset(), IDC, w, axis_lambda=0.5,
        training_args_kwargs=minimal_targs(tmp_path / "lam05"),
        output_directory=str(tmp_path / "lam05"),
    )
    # Capture the frozen h0 from the PRISTINE weights before any step.
    t0._precompute_frozen_embeddings()
    t1._precompute_frozen_embeddings()
    h0_eval = t1.frozen_emb[eval_batch["idx"].long()]

    opt0 = torch.optim.SGD(m0.parameters(), lr=1e-2)
    opt1 = torch.optim.SGD(m1.parameters(), lr=1e-2)
    for _ in range(5):
        for trainer, opt in ((t0, opt0), (t1, opt1)):
            loss = trainer.compute_loss(model=trainer.model, inputs={k: v.clone() for k, v in train_batch.items()})
            loss.backward()
            opt.step()
            opt.zero_grad()

    d0 = _axis_drift(m0, w, eval_batch, h0_eval)
    d1 = _axis_drift(m1, w, eval_batch, h0_eval)
    assert d0 > 0.0  # the unanchored model actually drifts
    assert d1 < d0  # the anchor pulls the axis projection back


# ---------------------------------------------------------------------------
# (c) soft_spearman: matches scipy within tolerance + gradients flow
# ---------------------------------------------------------------------------

def test_soft_spearman_matches_scipy_and_gradients() -> None:
    rng = np.random.default_rng(0)
    for trial in range(5):
        a = rng.normal(size=120)
        b = a * 0.7 + rng.normal(size=120) * 0.5
        x = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        y = torch.tensor(b, dtype=torch.float64)
        rho, ok = soft_spearman(x, y)
        assert ok
        rho_sc = spearmanr(a, b).statistic
        assert abs(float(rho.detach()) - rho_sc) < 0.05
        rho.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert float(x.grad.abs().max()) > 0.0  # backprop flows

    # discrete ordinal y {0,1,2,3} (the actual SS use case)
    rng2 = np.random.default_rng(7)
    xv = np.linspace(-1, 1, 40)
    yv = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    perm = rng2.permutation(40)
    xv, yv = xv[perm], yv[perm]
    xt = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(yv, dtype=torch.float64)
    rho2, ok2 = soft_spearman(xt, yt)
    assert ok2
    assert abs(float(rho2.detach()) - spearmanr(xv, yv).statistic) < 0.05
    rho2.backward()
    assert xt.grad is not None and float(xt.grad.abs().max()) > 0.0

    # degenerate: single cell -> valid False, grad-carrying zero
    xd = torch.tensor([1.0], requires_grad=True)
    yd = torch.tensor([0.0])
    rho3, ok3 = soft_spearman(xd, yd)
    assert not ok3
    assert float(rho3.detach()) == 0.0


# ---------------------------------------------------------------------------
# (d) flow: term finite + gradients flow
# ---------------------------------------------------------------------------

def test_flow_term_finite_and_gradients_flow(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = tiny_model()
    model.eval()
    w = unit_vector(seed=5)
    rows = tiny_rows()
    # a batch spanning adjacent classes: 4 cells per epi class + 2 non-epi
    pick = rows[0:4] + rows[8:12] + rows[16:20] + rows[24:28] + rows[32:34]
    batch = collate(pick)
    assert int((batch["labels"] == 4).sum()) == 2  # sanity: non-epi present

    trainer = build_flow_trainer(
        model, tiny_dataset(), IDC, w, flow_lambda=0.5,
        training_args_kwargs=minimal_targs(tmp_path / "flow"),
        output_directory=str(tmp_path / "flow"),
        k_pairs=32, min_cells_per_class=2,
    )
    loss = trainer.compute_loss(model, {k: v.clone() for k, v in batch.items()})
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0  # CE + a non-zero flow term
    loss.backward()
    grads = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert max(float(g.abs().max()) for g in grads) > 0.0  # gradients flow

    # lambda=0 -> pure CE
    t0 = build_flow_trainer(
        model, tiny_dataset(), IDC, w, flow_lambda=0.0,
        training_args_kwargs=minimal_targs(tmp_path / "flow0"),
        output_directory=str(tmp_path / "flow0"),
    )
    loss0 = t0.compute_loss(model, {k: v.clone() for k, v in batch.items()})
    assert float(loss0.detach()) == pytest.approx(float(_reference_ce(model, batch, class_to_ordinal_map(IDC)).detach()), rel=1e-5)


# ---------------------------------------------------------------------------
# (d2) dtype hazard regression: fp32 axis w against bf16 embeddings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["axis_ft", "soft_spearman", "flow"])
def test_arm_loss_dtype_alignment_bf16(tmp_path: Path, arm: str) -> None:
    """The frozen axis vector is fp32 (from axis_ref.json) but at real scale
    the model runs bf16, so ``z`` is bf16 and ``z @ w`` raised
    "addmv input tensors must have the same dtype, but got Float, BFloat16"
    (soft_spearman.py:101 / flow_consistency.py:109). Every arm's loss must
    dtype-align the projection. Tiny CPU fp32 tests missed this; here the tiny
    model's weights are bf16 so the embeddings are bf16, exactly like the
    real-scale run. The fp32 path stays covered by the tests above.
    """
    torch.manual_seed(21)
    model = tiny_model().to(torch.bfloat16)  # bf16 weights -> bf16 CLS embeddings
    w = unit_vector(seed=22)  # np float64 -> stored fp32 by each trainer
    rows = tiny_rows()
    # a batch spanning adjacent classes: 4 cells per epi class + 2 non-epi
    pick = rows[0:4] + rows[8:12] + rows[16:20] + rows[24:28] + rows[32:34]
    batch = collate(pick)
    out = tmp_path / arm
    if arm == "axis_ft":
        trainer = build_axis_ft_trainer(
            model, tiny_dataset(), IDC, w, axis_lambda=0.5,
            training_args_kwargs=minimal_targs(out), output_directory=str(out),
        )
    elif arm == "soft_spearman":
        trainer = build_ss_trainer(
            model, tiny_dataset(), IDC, w, ss_lambda=0.5,
            training_args_kwargs=minimal_targs(out), output_directory=str(out),
        )
    else:
        trainer = build_flow_trainer(
            model, tiny_dataset(), IDC, w, flow_lambda=0.5,
            training_args_kwargs=minimal_targs(out), output_directory=str(out),
            k_pairs=32, min_cells_per_class=2,
        )
    loss = trainer.compute_loss(model, {k: v.clone() for k, v in batch.items()})
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0  # CE + a non-zero arm term
    loss.backward()
    grads = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert max(float(g.abs().max()) for g in grads) > 0.0  # gradients flow


# ---------------------------------------------------------------------------
# (e) all three arms train 2 steps without NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["axis_ft", "soft_spearman", "flow"])
def test_arm_trains_two_steps_no_nan(tmp_path: Path, arm: str) -> None:
    torch.manual_seed(11)
    model = tiny_model()
    w = unit_vector(seed=12)
    ds = tiny_dataset()
    out = tmp_path / arm
    kwargs: dict = {"training_args_kwargs": minimal_targs(out), "output_directory": str(out)}
    if arm == "axis_ft":
        trainer = build_axis_ft_trainer(model, ds, IDC, w, axis_lambda=0.5, **kwargs)
    elif arm == "soft_spearman":
        trainer = build_ss_trainer(model, ds, IDC, w, ss_lambda=0.5, **kwargs)
    else:
        trainer = build_flow_trainer(
            model, ds, IDC, w, flow_lambda=0.5,
            k_pairs=32, min_cells_per_class=2, **kwargs,
        )
    trainer.train()
    losses = [float(h["loss"]) for h in trainer.state.log_history if "loss" in h]
    assert losses, f"{arm}: no loss entries in log_history"
    assert all(math.isfinite(v) for v in losses), f"{arm}: NaN in {losses}"


# ---------------------------------------------------------------------------
# (f) launcher Phase A summary aggregation
# ---------------------------------------------------------------------------

def test_phase_a_summary_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_method_arms as launcher

    monkeypatch.setattr(launcher, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(launcher, "STATUS_JSON", tmp_path / "runs_status.json")
    monkeypatch.setattr(launcher, "SUMMARY_CSV", tmp_path / "phase_a_summary.csv")

    # one evaluated run + one trained-but-not-evaluated run
    (tmp_path / "results_axis_ft_0.5_42.json").write_text(json.dumps({
        "primary_endpoint": {"macro_f1": 0.75},
        "secondary_endpoints": {
            "marker_gradient_rank_correlation": 0.61,
            "expected_calibration_error": 0.09,
        },
    }))
    (tmp_path / "runs_status.json").write_text(json.dumps({
        "runs": [{"arm": "flow", "lambda": 0.5, "seed": 42,
                  "train": "OK", "eval": "pending"}],
    }))
    import pandas as pd

    out = launcher.build_phase_a_summary()
    df = pd.read_csv(out)
    row = df[(df["arm"] == "axis_ft") & (df["lambda"] == 0.5) & (df["seed"] == 42)].iloc[0]
    assert row["macro_f1"] == pytest.approx(0.75)
    assert row["rho_L12"] == pytest.approx(0.61)
    assert row["axis_conservation"] == pytest.approx(0.61)
    assert row["ECE"] == pytest.approx(0.09)
    flow_row = df[(df["arm"] == "flow") & (df["seed"] == 42)].iloc[0]
    assert flow_row["status"] == "train:OK eval:pending"
    assert math.isnan(flow_row["macro_f1"])
    # the full Phase A grid is present
    assert len(df) == sum(len(v) for v in launcher.DEFAULT_PHASE_A_LAMBDAS.values())
