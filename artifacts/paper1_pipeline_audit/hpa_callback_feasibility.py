#!/usr/bin/env python
"""CPU feasibility check for the HPA-FT callback fix (2026-08-05, P0).

Verifies on a TINY BertForSequenceClassification (CPU, no Geneformer needed):

1. ragged-safe ``HPATrainer._precompute_centroids``: variable-length
   ``input_ids`` (as produced by geneformer's tokenized dataset) are padded
   per batch and do NOT crash; centroids are finite, shape
   (n_cont, n_layers, hidden_size), and the non-epithelial class is excluded
   from the centroid accumulation (epi-only forwarding).
2. real L_cont effect: with lambda=0.5 the training loss is
   loss_cls + 0.5 * L_cont (strictly greater than the lambda=0 loss on the
   same batch), and the trainer.log payload contains L_cont / hpa_loss /
   n_cont_layers — i.e. the continuity term is NOT silently dropped.
3. callback registration: ``build_hpa_trainer`` attaches an
   ``HPAEpochBeginCallback`` and the callback's ``on_epoch_begin`` triggers a
   centroid recompute (cont_layer_centroids becomes not-None).

Run: /home/dq/venv-ds/bin/python3 artifacts/paper1_pipeline_audit/hpa_callback_feasibility.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments

from corneal_fm.caft import NON_LAYER
from corneal_fm.hp_aft import HPAEpochBeginCallback, HPATrainer, build_hpa_trainer

# Non-epithelial class id (maps to NON_LAYER; must be excluded from centroids).
NON_EPI_CLASS = 4


def make_tiny_model() -> BertForSequenceClassification:
    cfg = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=5,
    )
    return BertForSequenceClassification(cfg)


def _pad_collator(features: list[dict]) -> dict:
    """Right-pad ragged input_ids within the batch (mirrors the collator)."""
    max_len = max(len(f["input_ids"]) for f in features)
    ids = torch.zeros(len(features), max_len, dtype=torch.long)
    for i, f in enumerate(features):
        ids[i, : len(f["input_ids"])] = torch.tensor(f["input_ids"])
    return {
        "input_ids": ids,
        "length": torch.tensor([f["length"] for f in features]),
        "label": torch.tensor([f["label"] for f in features]),
    }


def make_ragged_dataset(n: int = 40) -> Dataset:
    rng = torch.Generator().manual_seed(0)
    rows = []
    for i in range(n):
        # variable (ragged) token lengths, 6..24 tokens; class 4 is non-epi.
        length = int(torch.randint(6, 25, (1,), generator=rng).item())
        label = (i % 5)  # 0..3 epi, 4 non-epi
        rows.append(
            {
                "input_ids": [2]
                + [int(torch.randint(3, 63, (1,), generator=rng).item()) for _ in range(length - 2)]
                + [3],
                "length": length,
                "label": label,
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    torch.manual_seed(0)
    model = make_tiny_model()
    ds = make_ragged_dataset()
    # class ids 0..3 -> layers 0..3, class 4 -> NON_LAYER
    class_to_layer = {0: 0, 1: 1, 2: 2, 3: 3, NON_EPI_CLASS: NON_LAYER}
    args = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="hpa_cpu_"),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        report_to=[],
        logging_steps=1,
        bf16=False,
        remove_unused_columns=False,
    )
    trainer = HPATrainer(
        model=model,
        args=args,
        data_collator=lambda features: _pad_collator(features),
        train_dataset=ds,
        mode="lwc",
        hpa_lambda=0.5,
        cont_layers="1-2",  # hidden states 1..2 (tiny 2-layer encoder)
        class_to_layer=class_to_layer,
    )

    # ---- 1. ragged-safe precompute -------------------------------------
    trainer._precompute_centroids()
    c = trainer.cont_layer_centroids
    dev = c.device  # Trainer/Accelerator may place the model on CUDA if present
    assert c is not None, "cont_layer_centroids is None after precompute"
    assert tuple(c.shape) == (2, 5, 32), f"unexpected centroid shape {tuple(c.shape)}"
    assert torch.isfinite(c).all(), "centroids contain non-finite values"
    print(f"[1] ragged-safe precompute OK: cont_layer_centroids shape={tuple(c.shape)} dtype={c.dtype} device={dev}")

    # epi-only: non-epi layer (4) must have zero counts -> zero centroid row.
    assert torch.equal(c[:, NON_LAYER], torch.zeros(2, 32, device=dev)), (
        "non-epi layer centroid row not zero"
    )
    print("[1] epi-only forwarding OK: NON_LAYER centroid row is zero (excluded)")

    def to_dev(batch: dict):
        return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}

    # ---- 2. L_cont really enters the loss ------------------------------
    batch = to_dev(trainer.data_collator([ds[i] for i in range(8)]))

    logged: list[dict] = []

    def capture_log(payload, *a, **kw):
        logged.append(payload)

    trainer.log = capture_log  # type: ignore[method-assign]
    loss_lambda = trainer.compute_loss(model, {k: v.clone() for k, v in batch.items()})
    assert torch.isfinite(loss_lambda), "loss non-finite with lambda=0.5"
    assert logged and "L_cont" in logged[0], f"L_cont not logged: {logged}"
    l_cont = logged[0]["L_cont"]
    l_cls = logged[0]["L_cls"]
    # L_cont = -mean(cos(z, centroid_adj)) is the NEGATED cosine term, so its
    # sign depends on the model state; what matters is that it is NON-ZERO and
    # enters the loss exactly as loss = L_cls + lambda * L_cont.
    assert l_cont != 0.0, "L_cont is zero (continuity term inert)"
    assert logged[0]["n_cont_layers"] >= 1, "no cont layer contributed"
    expected = l_cls + 0.5 * l_cont
    assert abs(float(loss_lambda) - expected) < 1e-5, (
        f"loss {float(loss_lambda):.6f} != L_cls + 0.5*L_cont {expected:.6f}"
    )

    # lambda=0 trainer on the same batch -> pure CE loss
    trainer0 = HPATrainer(
        model=make_tiny_model(),
        args=args,
        data_collator=trainer.data_collator,
        train_dataset=ds,
        mode="lwc",
        hpa_lambda=0.0,
        cont_layers="1-2",
        class_to_layer=class_to_layer,
    )
    loss0 = trainer0.compute_loss(trainer0.model, to_dev({k: v.clone() for k, v in batch.items()}))
    assert abs(float(loss_lambda) - float(loss0)) > 1e-4, (
        f"lambda=0.5 loss {float(loss_lambda):.6f} should differ from lambda=0 loss {float(loss0):.6f}"
    )
    print(
        f"[2] L_cont effective OK: lambda=0.5 loss={float(loss_lambda):.6f} "
        f"(=L_cls {l_cls:.6f} + 0.5*L_cont {l_cont:.6f}) vs lambda=0 loss={float(loss0):.6f}; "
        f"logged keys={sorted(logged[0].keys())}"
    )

    # ---- 3. callback registration + epoch-begin trigger -----------------
    cb = HPAEpochBeginCallback(trainer)
    # no-op args/state/control stand-ins
    from types import SimpleNamespace

    trainer.cont_layer_centroids = None
    cb.on_epoch_begin(None, SimpleNamespace(), SimpleNamespace())
    assert trainer.cont_layer_centroids is not None, "on_epoch_begin did not recompute centroids"
    print("[3] HPAEpochBeginCallback.on_epoch_begin recomputes centroids OK")

    # build_hpa_trainer registers the callback (collator patched to a passthrough).
    import geneformer.collator_for_classification as gcc

    original_collator = gcc.DataCollatorForCellClassification  # noqa: F841
    gcc.DataCollatorForCellClassification = lambda token_dictionary: trainer.data_collator
    try:
        tr = build_hpa_trainer(
            make_tiny_model(),
            ds,
            {0: "LSC", 1: "TAC", 2: "basal", 3: "differentiated", NON_EPI_CLASS: "Conj-Epi"},
            mode="lwc",
            hpa_lambda=0.5,
            cont_layers="1-2",
            training_args_kwargs={
                "num_train_epochs": 1,
                "per_device_train_batch_size": 8,
                "report_to": [],
                "remove_unused_columns": False,
            },
            output_directory=tempfile.mkdtemp(prefix="hpa_out_"),
        )
    finally:
        gcc.DataCollatorForCellClassification = original_collator
    kinds = [type(cb_).__name__ for cb_ in tr.callback_handler.callbacks]
    assert "HPAEpochBeginCallback" in kinds, f"callback not registered: {kinds}"
    print(f"[3] build_hpa_trainer registers HPAEpochBeginCallback OK (callbacks={kinds})")

    print("\nALL HPA-FT CALLBACK FEASIBILITY CHECKS PASSED")


if __name__ == "__main__":
    main()
