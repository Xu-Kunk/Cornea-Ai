"""Flow-consistency fine-tuning of Geneformer V2-104M (Paper 1 arm 3, exploratory).

Loss (adjacent-class flow consistency, [SF]^2M "no energy barrier" spirit)::

    L = L_CE + lambda_f * mean_{adjacent pairs (ci, ci+1)} || h_proj(mid) - linear_interp(h_proj(ci), h_proj(ci+1)) ||^2

For every adjacent ordinal pair (0,1), (1,2), (2,3) we sample K paired cells
(``a`` in class ci, ``b`` in class ci+1) from the batch's EPITHELIAL cells
(masked sampling; pairs whose classes have too few cells are skipped), then
penalise the squared deviation of the MID-POINT projection from the
straight-line interpolant of the two endpoint projections::

    z_a = normalize(h(a))   z_b = normalize(h(b))
    mid = normalize((z_a + z_b) / 2)          # spherical mid-point embedding
    term = ( w . mid - (w . z_a + w . z_b) / 2 )^2

with ``w`` the frozen differentiation axis from ``corneal_fm/axis_ref.py``.
Because the projection is LINEAR but the L2 normalisation is NOT, the
mid-point projection is exactly the linear interpolant only when the two
endpoint embeddings are already collinear with the axis; any "energy
barrier" (curvature of the projection profile between adjacent classes)
shows up as a positive term that this loss squeezes out — the straight-line
transport of the differentiation coordinate between adjacent layers.

Implementation notes (documented choices):

* Paired cells are sampled per batch with replacement (``K=256`` pairs,
  ``--k-pairs``), mirroring the batch-stochastic nature of the other two
  arms; adjacent pairs whose classes have fewer than
  ``--min-cells-per-class`` (default 2) cells in the batch are skipped, and
  a batch with no valid adjacent pair contributes a grad-carrying zero
  (ordinal.py pattern) so the flow term never goes NaN.
* The endpoint embeddings are L2-normalised CLS vectors, matching the
  paper's rho_L12 measurement convention and making the term non-degenerate
  (with raw embeddings the term is identically zero by linearity).
* Feasibility-gated arm (Phase A, seed 42 only): the launcher's
  ``--summary`` / flow gate extends it to seeds {123, 456} only when the
  seed-42 run's loss history has no NaN and (once eval results exist) the
  axis is preserved (rho > 0.5 at lambda_f = 0.5).

Sanity contract (tested in tests/test_method_arms.py): lambda_f == 0 gives
pure CE; lambda_f > 0 gives a finite, non-NaN term with gradients flowing.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)
from transformers import Trainer, TrainingArguments

from corneal_fm.axis_ft import ordinal_ce_loss
from corneal_fm.caft import class_to_layer_map
from corneal_fm.ordinal import N_ORDINAL, class_to_ordinal_map


class FlowConsistencyTrainer(Trainer):
    """HF Trainer whose classification loss is CE + the flow-consistency term."""

    def __init__(
        self,
        flow_lambda: float,
        class_to_layer: dict,
        class_to_ordinal: dict,
        axis_w,
        *,
        k_pairs: int = 256,
        min_cells_per_class: int = 2,
        token_dictionary_file: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flow_lambda = float(flow_lambda)
        self.class_to_layer = class_to_layer
        self.class_to_ordinal = class_to_ordinal
        w = np.asarray(axis_w, dtype=float).reshape(-1)
        if w.shape[0] != int(self.model.config.hidden_size):
            raise ValueError(
                f"axis_w dim {w.shape[0]} != hidden_size {self.model.config.hidden_size}"
            )
        self._w = torch.tensor(w, dtype=torch.float32)
        self.k_pairs = int(k_pairs)
        self.min_cells_per_class = int(min_cells_per_class)

    def _adjacent_pair_terms(
        self,
        z: torch.Tensor,
        ordinals: torch.Tensor,
        valid: torch.Tensor,
        w: torch.Tensor,
        device,
    ) -> list[torch.Tensor]:
        """Per-adjacent-pair squared mid-point deviations (empty if none valid)."""
        # Defensive dtype alignment (callers already cast, but a fp32 ``w``
        # against bf16 embeddings would raise on the matmuls below).
        w = w.to(z.dtype)
        terms: list[torch.Tensor] = []
        for k in range(N_ORDINAL - 1):
            ia = torch.nonzero(valid & (ordinals == k), as_tuple=False).squeeze(1)
            ib = torch.nonzero(valid & (ordinals == k + 1), as_tuple=False).squeeze(1)
            if ia.numel() < self.min_cells_per_class or ib.numel() < self.min_cells_per_class:
                continue
            ga = torch.randint(0, ia.numel(), (self.k_pairs,), device=device)
            gb = torch.randint(0, ib.numel(), (self.k_pairs,), device=device)
            za = z[ia[ga]]
            zb = z[ib[gb]]
            mid = F.normalize((za + zb) / 2.0, dim=-1)
            interp = (za @ w + zb @ w) / 2.0
            terms.append(((mid @ w - interp).pow(2)).mean())
        return terms

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("label", None)
        if labels is None:
            labels = inputs.pop("labels", None)
        input_ids = inputs["input_ids"]
        length = inputs.get("length")
        device = input_ids.device
        if isinstance(length, torch.Tensor):
            attn = torch.arange(input_ids.shape[1], device=device)[None, :] < length[:, None]
        else:
            attn = None
        bert_out = model.bert(input_ids=input_ids, attention_mask=attn)
        logits = model.classifier(model.dropout(bert_out.pooler_output))
        loss_ce, ordinals, valid = ordinal_ce_loss(logits, labels, self.class_to_ordinal)

        if self.flow_lambda == 0 or labels is None:
            loss = loss_ce
        else:
            z = F.normalize(bert_out.last_hidden_state[:, 0], dim=-1)
            # Align the frozen fp32 axis with the bf16 embedding dtype before
            # the projections (Float @ BFloat16 raises at real scale); no-op
            # in fp32, so the loss math is unchanged (mirrors axis_ft).
            w = self._w.to(device).to(z.dtype)
            terms = self._adjacent_pair_terms(z, ordinals, valid, w, device)
            if terms:
                l_flow = torch.stack(terms).mean()
                loss = loss_ce + self.flow_lambda * l_flow
                if self.state.global_step % 50 == 0:
                    self.log(
                        {
                            "L_ce": float(loss_ce.detach()),
                            "L_flow": float(l_flow.detach()),
                            "flow_loss": float(loss.detach()),
                        }
                    )
            else:
                loss = loss_ce
        return (loss, bert_out) if return_outputs else loss


def build_flow_trainer(
    model,
    train_data,
    id_class_dict: dict,
    axis_w,
    flow_lambda: float,
    training_args_kwargs: dict,
    output_directory: str | Path,
    *,
    k_pairs: int = 256,
    min_cells_per_class: int = 2,
    token_dictionary_file: str | Path | None = None,
    eval_data=None,
) -> FlowConsistencyTrainer:
    """Build the FlowConsistencyTrainer; mirrors build_ordinal_trainer."""
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    if token_dictionary_file is None:
        from geneformer.classifier import (
            TOKEN_DICTIONARY_FILE as token_dictionary_file,  # noqa: N811 (mirrors caft.py)
        )
    with open(token_dictionary_file, "rb") as _fh:
        token_dictionary = pickle.load(_fh)
    collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    args = TrainingArguments(output_dir=str(output_directory), **training_args_kwargs)
    return FlowConsistencyTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        flow_lambda=flow_lambda,
        axis_w=axis_w,
        k_pairs=k_pairs,
        min_cells_per_class=min_cells_per_class,
        class_to_layer=class_to_layer_map(id_class_dict),
        class_to_ordinal=class_to_ordinal_map(id_class_dict),
        token_dictionary_file=token_dictionary_file,
        compute_metrics=None,
    )
