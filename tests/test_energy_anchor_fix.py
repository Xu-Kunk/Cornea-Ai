"""CPU smoke tests for the energy-anchor arm family fix (run_energy_anchor.py +
corneal_fm/energy_anchor.py).

Mirrors tests/test_method_arms.py: a tiny 2-layer BertForSequenceClassification
and a synthetic ragged cell dataset (4 ordinal classes + non-epithelial),
CPU-only. Verifies:

(a) the trainer builds and the init-time reference precompute completes on the
    tiny model — finite (N_STAGES, hidden) centroids, all 4 stage counts > 0,
    finite positive median-heuristic sigma,
(b) one training step runs without NaN (CE + the MMD anchor term),
(c) anchor_lambda=0 skips the precompute entirely (plain-CE control).

The tiny model's vocab (256) covers the geneformer collator's pad token 0, so
the real DataCollatorForCellClassification can pad the synthetic rows (same
pattern as test_method_arms.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corneal_fm.energy_anchor import N_STAGES, build_energy_anchor_trainer  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixtures (self-contained; no real data, no GPU)
# ---------------------------------------------------------------------------

# 6-class stand-in: 4 epithelial ordinal classes + 2 non-epithelial.
IDC = {
    0: "LPC",
    1: "Limbus_Epi-C1",
    2: "Epi-PMC",
    3: "Epi-TDC",
    4: "Conj-Epi",
    5: "Unknown-DCN+",
}
CELLS_PER_CLASS = 8
N_NON_EPI = 8

HIDDEN = 32
N_LAYERS = 2


def tiny_model() -> torch.nn.Module:
    """Fresh tiny BertForSequenceClassification (6-class head), CPU fp32.

    num_labels=6 covers ALL dataset classes (4 epithelial + 2 non-epithelial):
    energy-anchor keeps plain CE over every class (non-epi cells are masked
    only from the anchor term), exactly like the real 31-class Geneformer head.
    """
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=6,
        pad_token_id=0,
    )
    return BertForSequenceClassification(cfg)


def _cell_row(cls: int, i: int) -> dict:
    # Ragged: lengths vary across cells (3-5 tokens after <cls>/<eos>).
    pad = [101 + i % 13] * (i % 3)
    tokens = [2] + pad + [100 + i % 40, 3]
    return {"input_ids": tokens, "length": len(tokens), "label": cls}


def tiny_rows() -> list[dict]:
    rows: list[dict] = []
    i = 0
    for cls in range(4):
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


def _build(tmp_path: Path, anchor_lambda: float, name: str):
    model = tiny_model()
    ds = tiny_dataset()
    out = tmp_path / name
    return build_energy_anchor_trainer(
        model, ds, IDC,
        anchor_lambda=anchor_lambda,
        anchor_hidden=N_LAYERS,  # FINAL hidden-state index of the tiny model (2)
        training_args_kwargs=minimal_targs(out),
        output_directory=str(out),
    )


# ---------------------------------------------------------------------------
# (a) trainer builds + reference precompute completes (finite centroids)
# ---------------------------------------------------------------------------


def test_energy_anchor_precompute_finite_centroids(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tr = _build(tmp_path, anchor_lambda=0.5, name="precompute")
    c = tr.stage_centroids
    assert c is not None
    assert c.shape == (N_STAGES, HIDDEN)
    assert torch.isfinite(c).all()
    # every epithelial stage contributed cells (4 classes x 8 cells); the
    # non-epithelial rows are masked out
    assert tr._epi_rows is not None and len(tr._epi_rows) == 4 * CELLS_PER_CLASS
    # reference is frozen (stop-gradient)
    assert c.requires_grad is False
    # median-heuristic sigma is finite and positive
    assert tr.kernel_bandwidth is not None
    assert math.isfinite(tr.kernel_bandwidth) and tr.kernel_bandwidth > 0.0


# ---------------------------------------------------------------------------
# (b) one training step runs without NaN (CE + MMD anchor)
# ---------------------------------------------------------------------------


def test_energy_anchor_one_step_no_nan(tmp_path: Path) -> None:
    torch.manual_seed(1)
    tr = _build(tmp_path, anchor_lambda=0.5, name="step")
    tr.train()
    losses = [float(h["loss"]) for h in tr.state.log_history if "loss" in h]
    assert losses, "no loss entries in log_history"
    assert all(math.isfinite(v) for v in losses), f"NaN in {losses}"


# ---------------------------------------------------------------------------
# (c) lambda=0 skips the precompute (plain-CE control)
# ---------------------------------------------------------------------------


def test_energy_anchor_lambda_zero_skips_precompute(tmp_path: Path) -> None:
    tr = _build(tmp_path, anchor_lambda=0.0, name="lam0")
    assert tr.stage_centroids is None
    assert tr.kernel_bandwidth is None
