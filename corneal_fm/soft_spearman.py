"""SS — Soft-Spearman fine-tuning of Geneformer V2-104M (this study arm 2).

Loss (soft-Spearman ordinal alignment)::

    L = L_CE + lambda_ss * (1 - rho_soft(proj_A(h_theta(x)), y_ordinal))

* ``proj_A(h) = w . normalize(h)`` — the scalar projection of the L2-
  normalised CLS embedding (final hidden state) onto the frozen
  differentiation axis ``w`` from ``corneal_fm/axis_ref.py``. Normalisation
  matches the paper's rho_L12 measurement convention
  (scripts/measure_layerwise_rho.py uses ``l2_normalize=True``).
* ``rho_soft`` — the differentiable Spearman rho of
  ``corneal_fm/soft_rank.soft_spearman``: Pearson correlation of the
  logistic-kernel soft ranks of the two inputs. The term is minimised by
  making the axis projection of each cell monotonically ordered with its
  ordinal label — i.e. the loss directly pushes the differentiation
  gradient back into the embedding, at matched classification accuracy via
  the CE term.
* ``y_ordinal`` in {0, 1, 2, 3} for EPITHELIAL cells only — non-epithelial
  cells (layer == -1) are masked out of BOTH the CE term and the rho term
  (mirroring ordinal.py); they still pass through the encoder.

Degenerate-batch handling (mirrors ordinal.py): a batch with fewer than 2
epithelial cells or with zero ordinal spread (all same class) carries no
rank information, so the rho term is a grad-carrying zero
(``logits.sum() * 0.0``) rather than a constant 1 that would spuriously push
``rho -> 1``.

Sanity contract (tested in tests/test_method_arms.py): ``rho_soft`` on a
tiny tensor matches ``scipy.stats.spearmanr`` within ~0.05, the gradient
w.r.t. the input is non-zero (backprop flows), and lambda_ss == 0 reproduces
pure CE exactly.
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
from corneal_fm.ordinal import class_to_ordinal_map
from corneal_fm.soft_rank import soft_spearman


class SoftSpearmanTrainer(Trainer):
    """HF Trainer whose classification loss is CE + the soft-Spearman term."""

    def __init__(
        self,
        ss_lambda: float,
        class_to_layer: dict,
        class_to_ordinal: dict,
        axis_w,
        *,
        tau_scale: float = 0.1,
        token_dictionary_file: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ss_lambda = float(ss_lambda)
        self.class_to_layer = class_to_layer
        self.class_to_ordinal = class_to_ordinal
        w = np.asarray(axis_w, dtype=float).reshape(-1)
        if w.shape[0] != int(self.model.config.hidden_size):
            raise ValueError(
                f"axis_w dim {w.shape[0]} != hidden_size {self.model.config.hidden_size}"
            )
        self._w = torch.tensor(w, dtype=torch.float32)
        self.tau_scale = float(tau_scale)

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

        if self.ss_lambda == 0 or labels is None:
            loss = loss_ce
        else:
            z = F.normalize(bert_out.last_hidden_state[:, 0], dim=-1)
            # Align the frozen fp32 axis with the embedding dtype: at real
            # scale the model runs bf16, and Float @ BFloat16 raises
            # ("addmv input tensors must have the same dtype"). No-op in fp32,
            # so the loss math is unchanged (mirrors axis_ft's alignment).
            w = self._w.to(device).to(z.dtype)
            proj = z @ w  # (b,) scalar axis projection
            if valid is not None and int(valid.sum()) >= 2:
                rho, rho_ok = soft_spearman(
                    proj[valid], ordinals[valid].float(), tau_scale=self.tau_scale
                )
                if rho_ok:
                    l_ss = 1.0 - rho
                else:
                    l_ss = logits.sum() * 0.0
            else:
                l_ss = logits.sum() * 0.0
            loss = loss_ce + self.ss_lambda * l_ss
            if self.state.global_step % 50 == 0:
                self.log(
                    {
                        "L_ce": float(loss_ce.detach()),
                        "L_ss": float(l_ss.detach()),
                        "ss_loss": float(loss.detach()),
                    }
                )
        return (loss, bert_out) if return_outputs else loss


def build_ss_trainer(
    model,
    train_data,
    id_class_dict: dict,
    axis_w,
    ss_lambda: float,
    training_args_kwargs: dict,
    output_directory: str | Path,
    *,
    tau_scale: float = 0.1,
    token_dictionary_file: str | Path | None = None,
    eval_data=None,
) -> SoftSpearmanTrainer:
    """Build the SoftSpearmanTrainer; mirrors build_ordinal_trainer."""
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    if token_dictionary_file is None:
        from geneformer.classifier import (
            TOKEN_DICTIONARY_FILE as token_dictionary_file,  # noqa: N811 (mirrors caft.py)
        )
    with open(token_dictionary_file, "rb") as _fh:
        token_dictionary = pickle.load(_fh)
    collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    args = TrainingArguments(output_dir=str(output_directory), **training_args_kwargs)
    return SoftSpearmanTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        ss_lambda=ss_lambda,
        axis_w=axis_w,
        tau_scale=tau_scale,
        class_to_layer=class_to_layer_map(id_class_dict),
        class_to_ordinal=class_to_ordinal_map(id_class_dict),
        token_dictionary_file=token_dictionary_file,
        compute_metrics=None,
    )
