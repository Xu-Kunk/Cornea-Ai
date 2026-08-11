"""Contract tests for CAFT's detached, layer-aware continuity loss."""

from __future__ import annotations

import os
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from corneal_fm.caft import (
    CAFTTrainer,
    CELL_TYPE_TO_LAYER,
    LAYER_ORDER,
    NON_LAYER,
    author_cell_type_to_layer,
    class_to_layer_map,
)


class _SliceDataset:
    def __init__(self, rows: dict[str, list[list[int]] | list[int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows["label"])

    def __getitem__(self, index: slice) -> dict[str, list[list[int]] | list[int]]:
        return {key: value[index] for key, value in self.rows.items()}


def _model(num_labels: int = 8):
    torch.manual_seed(7)
    transformers = import_module("transformers")
    model = transformers.BertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=32,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=num_labels,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    model.eval()
    return model


def _trainer(model, caft_lambda: float, class_map: dict[int, int]) -> CAFTTrainer:
    trainer = object.__new__(CAFTTrainer)
    trainer.model = model
    trainer.caft_lambda = caft_lambda
    trainer.class_to_layer = class_map
    trainer.n_layers = len(LAYER_ORDER)
    trainer.layer_centroids = None
    trainer.pad_id = 0
    trainer.centroid_batch = 2
    trainer.state = SimpleNamespace(global_step=1)
    return trainer


def _inputs(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 5, 6, 0], [2, 7, 8, 9]], dtype=torch.long),
        "length": torch.tensor([3, 4], dtype=torch.long),
        "label": labels,
    }


@pytest.fixture(autouse=True)
def _injected_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED harness: enabled only by explicit subprocess commands used for evidence."""
    fault = os.environ.get("CAFT_INJECT_FAULT")
    if fault == "lambda":
        real_compute_loss = CAFTTrainer.compute_loss

        def wrong_lambda(self: CAFTTrainer, model, inputs, **kwargs):
            original = self.caft_lambda
            self.caft_lambda = 0.5
            try:
                return real_compute_loss(self, model, inputs, **kwargs)
            finally:
                self.caft_lambda = original

        monkeypatch.setattr(CAFTTrainer, "compute_loss", wrong_lambda)
    elif fault == "non_layer":
        real_compute_loss = CAFTTrainer.compute_loss

        def include_non_layer(self: CAFTTrainer, model, inputs, **kwargs):
            original = self.class_to_layer
            self.class_to_layer = {key: 0 for key in original}
            try:
                return real_compute_loss(self, model, inputs, **kwargs)
            finally:
                self.class_to_layer = original

        monkeypatch.setattr(CAFTTrainer, "compute_loss", include_non_layer)
    elif fault == "continuity":
        real_compute_loss = CAFTTrainer.compute_loss

        def wrong_continuity(self: CAFTTrainer, model, inputs, **kwargs):
            result = real_compute_loss(self, model, inputs, **kwargs)
            return result + 0.125 if isinstance(result, torch.Tensor) else (result[0] + 0.125, result[1])

        monkeypatch.setattr(CAFTTrainer, "compute_loss", wrong_continuity)
    elif fault == "stop_gradient":
        real_compute_loss = CAFTTrainer.compute_loss

        def attached_centroids(self: CAFTTrainer, model, inputs, **kwargs):
            assert self.layer_centroids is not None
            self.layer_centroids.requires_grad_(True)
            return real_compute_loss(self, model, inputs, **kwargs)

        monkeypatch.setattr(CAFTTrainer, "compute_loss", attached_centroids)


def _classification_loss(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["length"][:, None] > torch.arange(inputs["input_ids"].shape[1]),
        output_hidden_states=True,
    )
    return F.cross_entropy(outputs.logits, inputs["label"])


def test_lambda_zero_is_exact_classification_loss() -> None:
    model = _model()
    trainer = _trainer(model, caft_lambda=0.0, class_map={0: 1, 1: 1})
    trainer.layer_centroids = F.normalize(torch.arange(160, dtype=torch.float32).reshape(5, 32) + 1, dim=-1)
    inputs = _inputs(torch.tensor([0, 1]))

    loss = trainer.compute_loss(model, inputs.copy())
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, _classification_loss(model, inputs), rtol=0, atol=0)


def test_non_layer_cells_are_excluded_from_continuity() -> None:
    model = _model()
    trainer = _trainer(model, caft_lambda=0.7, class_map={7: NON_LAYER})
    trainer.layer_centroids = F.normalize(torch.arange(160, dtype=torch.float32).reshape(5, 32) + 1, dim=-1)
    inputs = _inputs(torch.tensor([7, 7]))

    loss = trainer.compute_loss(model, inputs.copy())
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, _classification_loss(model, inputs), rtol=0, atol=0)


def test_valid_layers_add_hand_computed_adjacent_centroid_continuity() -> None:
    model = _model()
    trainer = _trainer(model, caft_lambda=0.25, class_map={0: 1, 1: 3})
    centroids = torch.zeros(5, 32)
    centroids[0, 0], centroids[2, 1], centroids[4, 2] = 1.0, 1.0, 1.0
    trainer.layer_centroids = centroids.detach()
    inputs = _inputs(torch.tensor([0, 1]))
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=trainer._attn_mask(inputs["input_ids"], inputs["length"]),
        output_hidden_states=True,
    )
    z = F.normalize(outputs.hidden_states[-1][:, 0], dim=-1)
    expected_continuity = -torch.stack(((z[0] @ centroids[[0, 2]].T).mean(), (z[1] @ centroids[[2, 4]].T).mean())).mean()
    expected = F.cross_entropy(outputs.logits, inputs["label"]) + trainer.caft_lambda * expected_continuity

    loss = trainer.compute_loss(model, inputs.copy())
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)


def test_centroids_are_stop_gradient_while_model_receives_gradients() -> None:
    model = _model()
    trainer = _trainer(model, caft_lambda=0.25, class_map={0: 1, 1: 3})
    source = torch.randn(5, 32, requires_grad=True)
    trainer.layer_centroids = F.normalize(source, dim=-1).detach()
    loss = trainer.compute_loss(model, _inputs(torch.tensor([0, 1])))
    assert isinstance(loss, torch.Tensor)
    loss.backward()

    assert trainer.layer_centroids.requires_grad is False
    assert trainer.layer_centroids.grad_fn is None
    assert source.grad is None
    assert model.bert.embeddings.word_embeddings.weight.grad is not None


def test_embed_returns_last_hidden_cls_and_attention_mask_uses_length() -> None:
    model = _model()
    trainer = _trainer(model, caft_lambda=0.0, class_map={})
    ids = torch.tensor([[2, 5, 0, 0], [2, 6, 7, 0]])
    length = torch.tensor([2, 3])
    mask = trainer._attn_mask(ids, length)
    direct = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1][:, 0]

    assert torch.equal(mask, torch.tensor([[True, True, False, False], [True, True, True, False]]))
    assert trainer._embed(ids, length).shape == (2, 32)
    assert torch.allclose(trainer._embed(ids, length), direct, rtol=0, atol=0)


def test_layer_mapping_is_total_deterministic_and_returns_ints() -> None:
    # Current names are a pre-ratification fixture; do not encode final 31→5 assignments here.
    id_class_dict = {str(index): name for index, name in enumerate(CELL_TYPE_TO_LAYER)} | {"99": "unseen"}
    mapped = class_to_layer_map(id_class_dict)

    assert set(mapped) == {int(key) for key in id_class_dict}
    assert all(isinstance(value, int) for value in mapped.values())
    assert author_cell_type_to_layer("unseen") == NON_LAYER
    assert author_cell_type_to_layer("unseen") == author_cell_type_to_layer("unseen")
    assert mapped[99] == NON_LAYER


def test_precompute_centroids_normalizes_nonempty_rows_and_handles_empty_layers() -> None:
    model = _model()
    trainer = _trainer(model, caft_lambda=0.25, class_map={0: 0, 1: 2})
    trainer.train_dataset = _SliceDataset(
        {"input_ids": [[2, 5, 0, 0], [2, 6, 7, 0], [2, 8, 0, 0]], "length": [2, 3, 2], "label": [0, 1, 0]}
    )
    trainer._precompute_centroids()
    assert trainer.layer_centroids is not None
    norms = torch.linalg.vector_norm(trainer.layer_centroids, dim=1)

    assert trainer.layer_centroids.shape == (5, 32)
    assert torch.allclose(norms[[0, 2]], torch.ones(2), atol=1e-6)
    assert torch.equal(trainer.layer_centroids[[1, 3, 4]], torch.zeros(3, 32))
    assert torch.isfinite(trainer.layer_centroids).all()
    loss = trainer.compute_loss(model, _inputs(torch.tensor([0, 1])))
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
