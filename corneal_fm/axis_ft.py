"""AA-FT — Axis-Anchored Fine-Tuning of Geneformer V2-104M (Paper 1 arm 1).

Loss (axis-anchored fine-tuning)::

    L = L_CE + lambda_axis * || P_A ( h_theta(x) - h_theta0(x) ) ||^2

* ``h_theta(x)`` — the CLS embedding of cell x at a configurable hidden-state
  layer (default the FINAL hidden state; the term is designed for the
  head-proximal band L5-L12 where the paper's layer-wise diagnostic localises
  the differentiation-gradient collapse, so ``--anchor-layer`` accepts
  hidden-state indices 5..12 on the 13-layer V2-104M).
* ``h_theta0(x)`` — the FROZEN pretrained CLS embedding of the SAME cell,
  captured once from the pretrained (non-fine-tuned) weights before the
  first optimizer step and never recomputed inside the training loop.
* ``P_A = w w^T`` — the frozen rank-1 projector onto the differentiation
  axis ``w`` from ``corneal_fm/axis_ref.py`` (cached to
  ``artifacts/paper1_method_arms/axis_ref.json`` before training). Because
  ``w`` is unit, ``||P_A v||^2 = (w . v)^2``, which is how the term is
  actually evaluated (O(d) instead of O(d^2), exact).

The arm uses the SAME 4-class CE head as the paper's naive arm (ordinal
labels LSC(0) < TAC(1) < basal(2) < differentiated(3)); non-epithelial cells
(layer == -1) carry no ordinal label and are masked out of BOTH the CE term
and the anchor term (mirroring how ordinal.py masks its ordinal loss and
CAFT masks its continuity term). They still pass through the encoder.

Frozen-embedding cache (documented choice): ``h_theta0`` is precomputed ONCE
over the full training set (epi-only forward, ragged-safe, mirrors
``ordinal._precompute_centroids`` — non-epi cells are ~65% of the set and
contribute nothing to the anchor, so skipping their forwards cuts the cost
by ~2/3) and stored detached in fp32 as a ``(n_train, hidden)`` tensor
(~150 MB for 100k x 768 fp32, bounded). Per-batch indexing uses an ``idx``
column that the training script adds to the dataset (the geneformer cell
collator preserves extra columns as tensors, verified empirically), so the
loss incurs ZERO extra forwards. The precompute is dispatched by
:class:`AxisFTInitCallback` at ``on_train_begin`` (transformers 4.57
dispatches ``on_epoch_begin``/``on_train_begin`` only to registered
TrainerCallback objects, never to Trainer methods — same fix as ordinal.py),
with a lazy fallback inside ``compute_loss`` for direct-loss tests.

Sanity contract (tested in tests/test_method_arms.py): lambda_axis == 0 must
reproduce pure CE exactly; lambda_axis > 0 must reduce ``||P_A(h - h0)||``
relative to lambda == 0 after gradient descent.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)
from transformers import Trainer, TrainerCallback, TrainingArguments

from corneal_fm.caft import NON_LAYER, class_to_layer_map
from corneal_fm.ordinal import N_ORDINAL, class_to_ordinal_map


def attn_mask(input_ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """(b, s) boolean attention mask from the ``length`` column (right-pad)."""
    device = input_ids.device
    return torch.arange(input_ids.shape[1], device=device)[None, :] < length[:, None]


def ordinal_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    class_to_ordinal: dict,
    n_ordinal: int = N_ORDINAL,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """4-class CE on the ordinal labels, masked to epithelial cells.

    Maps each cell's class id through ``class_to_ordinal`` (0..3 for
    epithelial, NON_LAYER=-1 otherwise) and returns
    ``(loss_ce, ordinals, valid)``. Cells with no ordinal label are masked
    out of the CE; an all-non-epi batch yields a grad-carrying zero loss
    (``logits.sum() * 0.0``, the ordinal.py pattern) so ``Trainer.backward``
    never sees a constant. ``labels is None`` returns the same zero loss with
    ``(None, None)``.
    """
    if labels is None:
        return logits.sum() * 0.0, None, None
    device = labels.device
    ordinals = torch.tensor(
        [class_to_ordinal.get(int(c), NON_LAYER) for c in labels.tolist()],
        device=device,
    )
    valid = ordinals != NON_LAYER
    if not bool(valid.any()):
        return logits.sum() * 0.0, ordinals, valid
    loss = F.cross_entropy(logits[valid], ordinals[valid])
    return loss, ordinals, valid


class AxisFTTrainer(Trainer):
    """HF Trainer whose classification loss is CE + the axis anchor term.

    ``h_theta0`` is captured from the PRETRAINED weights: :meth:`_precompute_
    frozen_embeddings` must run before the first optimizer step (guaranteed
    by :class:`AxisFTInitCallback` / the lazy call in ``compute_loss`` before
    any training happens).
    """

    def __init__(
        self,
        axis_lambda: float,
        class_to_layer: dict,
        class_to_ordinal: dict,
        axis_w,
        *,
        anchor_layer: int | None = None,
        token_dictionary_file: str | Path | None = None,
        centroid_batch: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis_lambda = float(axis_lambda)
        self.class_to_layer = class_to_layer
        self.class_to_ordinal = class_to_ordinal
        w = np.asarray(axis_w, dtype=float).reshape(-1)
        if w.shape[0] != int(self.model.config.hidden_size):
            raise ValueError(
                f"axis_w dim {w.shape[0]} != hidden_size {self.model.config.hidden_size}"
            )
        self._w = torch.tensor(w, dtype=torch.float32)
        self.anchor_layer = (
            int(self.model.config.num_hidden_layers) if anchor_layer is None else int(anchor_layer)
        )
        n_hidden = int(self.model.config.num_hidden_layers)
        if not (1 <= self.anchor_layer <= n_hidden):
            raise ValueError(
                f"anchor_layer must be a hidden-state index in 1..{n_hidden}; "
                f"got {self.anchor_layer}"
            )
        self.frozen_emb: torch.Tensor | None = None
        self._frozen_rows: list[int] | None = None
        self.centroid_batch = centroid_batch

    # -- frozen h0 precompute (ragged-safe, epi-only, once) ----------------

    def _precompute_frozen_embeddings(self) -> None:
        """Capture h_theta0 for every cell (epi-only forward, stop-gradient).

        Mirrors ``ordinal.OrdinalTrainer._precompute_centroids``: ragged
        slices are right-padded to their own max length, the attention mask
        derives from ``length``, and only EPITHELIAL cells are forwarded
        (non-epi contribute nothing to the anchor term). CLS embeddings at
        ``anchor_layer`` are stored detached in fp32 in a full-size
        ``(n_train, hidden)`` tensor indexed by the dataset ``idx`` column.
        """
        ds = self.train_dataset
        n = len(ds)
        if n == 0:
            return
        if self._frozen_rows is None:
            self._frozen_rows = [
                i
                for i, c in enumerate(list(ds["label"]))
                if self.class_to_layer.get(int(c), NON_LAYER) != NON_LAYER
            ]
            # Sort by ascending length once: attention cost grows ~quadratically
            # with the padded length, and the per-row writes (emb[idx] = cls)
            # make the result order-independent — mirrors axis_ref.compute_axis.
            if "length" in ds.column_names:
                self._frozen_rows = sorted(self._frozen_rows, key=lambda i: int(ds["length"][i]))
        device = self.model.device
        hidden = int(self.model.config.hidden_size)
        n_hidden = int(self.model.config.num_hidden_layers)
        # The final hidden state is already exposed as ``last_hidden_state``:
        # skip ``output_hidden_states`` (which retains ALL 13 layers) unless an
        # interior anchor layer is requested — halves the per-batch footprint.
        need_all_layers = self.anchor_layer < n_hidden
        self.model.eval()
        emb = torch.zeros(n, hidden, dtype=torch.float32, device=device)
        rows = self._frozen_rows
        has_len_col = "length" in ds.column_names
        # Same bounded-footprint batching as axis_ref.compute_axis: shrink the
        # batch when long cells are present so the per-batch attention peak
        # stays well inside the GB10 unified-memory pool (kernel oom_reaper
        # kills on oversized IOMMU mappings / exhausted vmalloc page arrays).
        # Length-sorted rows => the slice's longest cell sits at its end. Pure
        # batching change: emb[idx] writes per dataset row, so results are
        # order-independent.
        budget = max(64, int(self.centroid_batch) * 1500)
        with torch.no_grad():
            s = 0
            batch_no = 0
            while s < len(rows):
                k = min(self.centroid_batch, len(rows) - s)
                if has_len_col:
                    while k > 32 and int(ds["length"][rows[s + k - 1]]) * k > budget:
                        k = max(32, budget // int(ds["length"][rows[s + k - 1]]))
                idx = rows[s : s + k]
                batch = ds[idx]
                ids_list = batch["input_ids"]
                lengths = batch.get("length")
                if lengths is None:
                    lengths = [len(row) for row in ids_list]
                # Right-pad the ragged slice in ONE C++-level op (pad_sequence);
                # mirrors the axis_ref.compute_axis fix — the old per-row Python
                # assignment loop was O(rows) interpreter work per batch.
                ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(row, dtype=torch.long) for row in ids_list],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                length = torch.as_tensor(lengths, device=device)
                out = self.model.bert(
                    input_ids=ids,
                    attention_mask=attn_mask(ids, length),
                    output_hidden_states=need_all_layers,
                )
                if need_all_layers:
                    cls = out.hidden_states[self.anchor_layer][:, 0].float()
                else:
                    cls = out.last_hidden_state[:, 0].float()
                emb[idx] = cls
                del out
                s += k
                batch_no += 1
                # Release the caching allocator's reserved pool periodically so
                # the unified-memory driver mappings are returned to the kernel
                # (GB10 OOM-reaper fragility under vmalloc pressure).
                if batch_no % 20 == 0:
                    torch.cuda.empty_cache()
        self.frozen_emb = emb
        self.model.train()

    # -- loss --------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("label", None)
        if labels is None:
            labels = inputs.pop("labels", None)
        input_ids = inputs["input_ids"]
        length = inputs.get("length")
        device = input_ids.device
        attn = attn_mask(input_ids, length) if isinstance(length, torch.Tensor) else None
        need_hidden = bool(self.axis_lambda != 0 and labels is not None)
        bert_out = model.bert(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=need_hidden,
        )
        logits = model.classifier(model.dropout(bert_out.pooler_output))
        loss_ce, ordinals, valid = ordinal_ce_loss(logits, labels, self.class_to_ordinal)
        if not getattr(self, "_first_loss_diag_done", False):
            self._first_loss_diag_done = True
            print(
                "[axis_ft] loss diag: labels_present="
                f"{labels is not None} batch_keys={sorted(inputs)} "
                f"logging_steps={getattr(self.state, 'logging_steps', None)}",
                flush=True,
            )

        if self.axis_lambda == 0 or labels is None:
            loss = loss_ce
            l_anchor = None
        else:
            if self.frozen_emb is None:
                # Lazy capture of the pretrained h0 (model is still at init
                # weights — safe before the first optimizer step).
                self._precompute_frozen_embeddings()
            idx = inputs.get("idx")
            if idx is None:
                raise RuntimeError(
                    "axis-anchored term requires an 'idx' column on the training "
                    "dataset (see scripts/method_arm_gf.py)"
                )
            w = self._w.to(device)
            h = bert_out.hidden_states[self.anchor_layer][:, 0]  # raw CLS
            h0 = self.frozen_emb[idx.long()]
            # ||P_A (h - h0)||^2 == (w . (h - h0))^2 for unit w (P_A = w w^T).
            l_anchor = (w @ (h.float() - h0).t()).pow(2)
            if valid is not None and bool(valid.any()):
                l_anchor = l_anchor[valid].mean()
            else:
                l_anchor = logits.sum() * 0.0
            loss = loss_ce + self.axis_lambda * l_anchor
            if self.state.global_step % 50 == 0:
                self.log(
                    {
                        "L_ce": float(loss_ce.detach()),
                        "L_anchor": float(l_anchor.detach()),
                        "axis_ft_loss": float(loss.detach()),
                    }
                )
        # Explicit per-run log lines (independent of the Trainer callback
        # machinery, which may not render metrics into a piped stdout log).
        if self.state.global_step % 10 == 0:
            print(
                f"[axis_ft] step={self.state.global_step} "
                f"L_ce={float(loss_ce.detach()):.4f} "
                f"L_anchor={(0.0 if l_anchor is None else float(l_anchor.detach())):.4f} "
                f"axis_ft_loss={float(loss.detach()):.4f}",
                flush=True,
            )
        return (loss, bert_out) if return_outputs else loss


class AxisFTInitCallback(TrainerCallback):
    """Capture the frozen h0 embeddings ONCE, before the first optimizer step.

    transformers 4.57 dispatches ``on_train_begin`` only to registered
    TrainerCallback OBJECTS (never to a Trainer method), so this callback is
    the mechanism that guarantees ``h_theta0`` comes from the pretrained
    weights and never drifts with training (mirrors the ordinal.py fix).
    """

    def __init__(self, trainer: AxisFTTrainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if self.trainer.axis_lambda != 0 and self.trainer.frozen_emb is None:
            self.trainer._precompute_frozen_embeddings()


def build_axis_ft_trainer(
    model,
    train_data,
    id_class_dict: dict,
    axis_w,
    axis_lambda: float,
    training_args_kwargs: dict,
    output_directory: str | Path,
    *,
    anchor_layer: int | None = None,
    token_dictionary_file: str | Path | None = None,
    eval_data=None,
) -> AxisFTTrainer:
    """Build the AxisFTTrainer (same collator as build_caft_trainer).

    Registers :class:`AxisFTInitCallback` so the frozen h0 embeddings are
    captured before training begins (without it the anchor term would
    silently start from whatever weights ``compute_loss`` first sees).
    """
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    if token_dictionary_file is None:
        from geneformer.classifier import (
            TOKEN_DICTIONARY_FILE as token_dictionary_file,  # noqa: N811 (mirrors caft.py)
        )
    with open(token_dictionary_file, "rb") as _fh:
        token_dictionary = pickle.load(_fh)
    collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    args = TrainingArguments(output_dir=str(output_directory), **training_args_kwargs)
    trainer = AxisFTTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        axis_lambda=axis_lambda,
        axis_w=axis_w,
        anchor_layer=anchor_layer,
        class_to_layer=class_to_layer_map(id_class_dict),
        class_to_ordinal=class_to_ordinal_map(id_class_dict),
        token_dictionary_file=token_dictionary_file,
        compute_metrics=None,
    )
    trainer.add_callback(AxisFTInitCallback(trainer))
    return trainer
