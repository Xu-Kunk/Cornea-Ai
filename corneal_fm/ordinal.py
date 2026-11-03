"""Ordinal (CORN-style) fine-tuning of Geneformer V2-104M — this study D5 loss arm.

Motivation (D5): this study observed that softmax classification fine-tuning
collapses the epithelial differentiation gradient of the CLS embedding
(Spearman rho 0.82 -> ~ -0.04 for naive fine-tune; CAFT partially restores it
to ~ +0.21). The D5 hypothesis is that treating the 4 differentiation layers
(LSC < TAC < basal < differentiated) as *unordered* softmax labels is the
causal mechanism of the rho collapse; encoding the SAME axis as ordered
thresholds should preserve layer continuity.

Scheme A (chosen) — CORN (Cao et al., "Rank Consistent Ordinal Regression",
IEEE TNNLS 2020): for K=4 ordinal classes we train K-1=3 binary heads, where
head k predicts P(y >= k). The loss is the sum of the 3 binary
cross-entropies (masked to epithelial cells); at inference the predicted
class is the number of thresholds with sigmoid >= 0.5. Chosen over the binned
ordinal alternative because it makes no equidistance assumption on the
thresholds and yields class probabilities directly for calibration metrics
(P(y=k) = p_k - p_{k+1}, with p_0 = 1, p_K = 0).

Continuity regularization (ordinal-CAFT) reuses the CAFT layer-centroid cosine
adjacency + stop gradient (5-layer centroids precomputed at each epoch start)
so that lambda=0.5 applies the SAME regularization in the CAFT and ordinal-CAFT
arms — the only difference between the arms is the classification head/loss (the
D5 variable).

IMPORTANT implementation note (divergence from read-only corneal_fm/caft.py):
transformers 4.57 dispatches ``on_epoch_begin`` to TrainerCallback OBJECTS and
never calls a method named ``on_epoch_begin`` on the Trainer itself.
``CAFTTrainer.on_epoch_begin`` (the centroid precompute hook) is therefore dead
code, and the inherited ``_precompute_centroids`` also crashes on the ragged
variable-length ``input_ids`` of the geneformer-prepared dataset (verified
empirically: a full CAFT train() completes with ``layer_centroids is None`` and
the λ=0 / λ=0.5 runs have identical loss history). OrdinalTrainer fixes BOTH by
(a) overriding ``_precompute_centroids`` with a ragged-safe, per-batch padded
implementation and (b) registering a real TrainerCallback
(:class:`OrdinalEpochBeginCallback`) that calls it at the start of every epoch.
Without this fix ordinal-CAFT would silently train WITHOUT continuity — exactly
what the existing CAFT λ=0.5 baseline numbers were produced from.

Arms:
    ordinal-naive : ordinal_lambda = 0.0  (ordinal loss only)
    ordinal-CAFT  : ordinal_lambda = 0.5  (ordinal loss + lambda * L_cont)

Non-epithelial cells (layer == -1, e.g. Conj-Epi) carry no ordinal label and
are masked out of the ordinal loss (mirroring how CAFT masks its continuity
term); they still pass through the encoder in every batch.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)
from transformers import TrainerCallback, TrainingArguments

from corneal_fm.caft import NON_LAYER, CAFTTrainer, class_to_layer_map

# Ordinal axis: LSC(0) < TAC(1) < basal(2) < differentiated(3).
N_ORDINAL = 4
# K-1 binary thresholds: "is the cell >= k" for k in {1, 2, 3}.
N_THRESHOLDS = N_ORDINAL - 1


def class_to_ordinal_map(id_class_dict: dict) -> dict:
    """Map numeric class id -> 4-class ordinal target {-1, 0, 1, 2, 3}.

    Built from the CAFT 5-layer map: LSC=0, TAC=1, basal=2, wing(3) and
    squamous(4) both collapse onto differentiated=3 (the locked 4-class
    harmonization of config/external_test_gse153515.yml); non-epithelial
    classes map to -1 and are masked out of the ordinal loss.
    """
    layer_map = class_to_layer_map(id_class_dict)
    out: dict[int, int] = {}
    for cls_id, layer in layer_map.items():
        out[int(cls_id)] = NON_LAYER if layer == NON_LAYER else min(int(layer), N_ORDINAL - 1)
    return out


def ordinal_decode(
    logits: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """CORN inference: (n, K-1) threshold logits -> (n,) class 0..K-1.

    Predicted class = number of thresholds with sigmoid >= ``threshold``.
    Also returns the sigmoid threshold probabilities (n, K-1).
    """
    logits = np.asarray(logits, dtype=float)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).sum(axis=-1).astype(np.int64)
    return preds, probs


def ordinal_class_probs(threshold_probs: np.ndarray) -> np.ndarray:
    """CORN threshold probabilities (n, K-1) -> class probabilities (n, K).

    P(y=0) = 1 - p_1, P(y=k) = p_k - p_{k+1}, P(y=K-1) = p_{K-1}. Negative
    mass (non-monotone thresholds) is clipped to 0 and each row is
    renormalized so the result is a valid probability distribution (used for
    ECE; hard predictions use :func:`ordinal_decode`).
    """
    probs = np.asarray(threshold_probs, dtype=float)
    n = probs.shape[0]
    k = probs.shape[1] + 1
    diffs = np.zeros((n, k), dtype=float)
    diffs[:, 0] = 1.0 - probs[:, 0]
    for i in range(1, k - 1):
        diffs[:, i] = probs[:, i - 1] - probs[:, i]
    diffs[:, k - 1] = probs[:, k - 2]
    diffs = np.clip(diffs, 0.0, None)
    row_sum = diffs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return diffs / row_sum


class OrdinalTrainer(CAFTTrainer):
    """HF Trainer whose classification loss is the CORN ordinal loss.

    Reuses CAFTTrainer's epoch-begin layer-centroid precomputation, the
    stop-gradient continuity term and the attention/pad machinery unchanged;
    only the head supervision differs (K-1 BCEs instead of K-way softmax CE).
    ``ordinal_lambda`` is stored as ``caft_lambda`` on the base class so the
    inherited continuity block runs with identical semantics.
    """

    def __init__(
        self,
        ordinal_lambda: float,
        class_to_layer: dict,
        class_to_ordinal: dict,
        n_layers: int = 5,
        token_dictionary_file: str | Path | None = None,
        centroid_batch: int = 512,
        **kwargs,
    ):
        super().__init__(
            caft_lambda=ordinal_lambda,
            class_to_layer=class_to_layer,
            n_layers=n_layers,
            token_dictionary_file=token_dictionary_file,
            centroid_batch=centroid_batch,
            **kwargs,
        )
        self.class_to_ordinal = class_to_ordinal
        self.n_ordinal = N_ORDINAL

    def _precompute_centroids(self):
        """Ragged-safe layer-centroid precompute over the full training set.

        Overrides CAFTTrainer._precompute_centroids (which crashes on the
        variable-length geneformer-prepared dataset: ``torch.tensor`` on ragged
        ``input_ids`` raises). Here each slice is padded to its own max length
        (right-pad with pad_id, matching the collator), the attention mask is
        derived from the ``length`` column as in :meth:`_embed`, and the CLS
        embeddings are pooled per layer (detached, stop-gradient).

        Cost guard: only EPITHELIAL cells are forwarded. Non-epithelial cells
        (label -> layer -1) contribute nothing to any centroid, so skipping
        their forward passes yields identical centroids while cutting the
        precompute cost by ~2/3 (non-epi are ~65% of the training set).
        """
        ds = self.train_dataset
        n = len(ds)
        if n == 0:
            return
        if getattr(self, "_epi_rows", None) is None:
            labels_all = list(ds["label"])
            self._epi_rows = [
                i
                for i, c in enumerate(labels_all)
                if self.class_to_layer.get(int(c), NON_LAYER) != NON_LAYER
            ]
        device = self.model.device
        dtype = next(self.model.parameters()).dtype  # bf16 weights -> bf16 centroids
        self.model.eval()
        sums = torch.zeros(self.n_layers, self.model.config.hidden_size, device=device, dtype=dtype)
        counts = torch.zeros(self.n_layers, device=device)
        epi = self._epi_rows
        with torch.no_grad():
            for s in range(0, len(epi), self.centroid_batch):
                rows = epi[s : s + self.centroid_batch]
                batch = ds[rows]
                ids_list = batch["input_ids"]
                lengths = batch.get("length")
                if lengths is None:
                    lengths = [len(row) for row in ids_list]
                max_len = int(max(lengths)) if lengths else 0
                ids = torch.zeros(len(ids_list), max_len, dtype=torch.long, device=device)
                for i, row in enumerate(ids_list):
                    ids[i, : len(row)] = torch.as_tensor(row, dtype=torch.long)
                length = torch.as_tensor(lengths, device=device)
                labels = torch.as_tensor(batch["label"], device=device)
                z = F.normalize(self._embed(ids, length), dim=-1)
                for cls_id in labels.unique().tolist():
                    lyr = self.class_to_layer.get(int(cls_id), NON_LAYER)
                    if lyr == NON_LAYER:
                        continue
                    mask = labels == cls_id
                    sums[lyr] += z[mask].sum(0)
                    counts[lyr] += mask.sum()
        valid = counts > 0
        centroids = torch.zeros_like(sums)
        centroids[valid] = (sums[valid] / counts[valid].unsqueeze(1)).to(dtype)
        self.layer_centroids = F.normalize(centroids, dim=-1).detach()
        self.model.train()

    def _ordinal_targets(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(n_valid,) ordinal labels, (n_valid, K-1) BCE targets, valid mask."""
        device = labels.device
        ordinals = torch.tensor(
            [self.class_to_ordinal.get(int(c), NON_LAYER) for c in labels.tolist()],
            device=device,
        )
        valid = ordinals != NON_LAYER
        if not bool(valid.any()):
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, torch.zeros(0, N_THRESHOLDS, dtype=torch.float32, device=device), valid
        y = ordinals[valid]
        thresholds = torch.arange(1, self.n_ordinal, device=device)  # {1..K-1}
        targets = (y[:, None] >= thresholds[None, :]).float()
        return y, targets, valid

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("label", None)
        if labels is None:
            labels = inputs.pop("labels", None)
        input_ids = inputs["input_ids"]
        length = inputs.get("length")
        device = input_ids.device
        if isinstance(length, torch.Tensor):
            attn = self._attn_mask(input_ids, length)
        else:
            attn = None
        bert_out = model.bert(input_ids=input_ids, attention_mask=attn)
        logits = model.classifier(model.dropout(bert_out.pooler_output))  # (n, K-1)

        # ── CORN ordinal loss (masked to epithelial cells) ──────────────
        if labels is not None:
            _, targets, valid = self._ordinal_targets(labels)
            if bool(valid.any()):
                loss_ord = F.binary_cross_entropy_with_logits(
                    logits[valid], targets, reduction="mean"
                )
            else:
                # No epithelial cells in this batch (non-epi are ~2/3 of the
                # training set, so all-non-epi batches occur regularly). A
                # bare zero constant would break Trainer.backward() ("does not
                # require grad"); multiplying a grad-carrying tensor by zero
                # yields zero gradients while keeping a valid grad_fn.
                loss_ord = logits.sum() * 0.0
        else:
            loss_ord = logits.sum() * 0.0

        # ── Continuity term (inherited CAFT semantics, stop-gradient) ───
        if self.caft_lambda == 0 or self.layer_centroids is None or labels is None:
            loss = loss_ord
        else:
            z_cls = F.normalize(bert_out.last_hidden_state[:, 0], dim=-1)
            layers = torch.tensor(
                [self.class_to_layer.get(int(c), NON_LAYER) for c in labels.tolist()],
                device=device,
            )
            valid_l = layers != NON_LAYER
            if not bool(valid_l.any()):
                loss = loss_ord
            else:
                zv = z_cls[valid_l]
                lv = layers[valid_l]
                adj_pairs = []
                for lyr in lv.tolist():
                    cands = [i for i in (lyr - 1, lyr + 1) if 0 <= i < self.n_layers]
                    if cands:
                        adj_pairs.append(cands)
                if adj_pairs:
                    sims = []
                    for zi, cands in zip(zv, adj_pairs):
                        c = self.layer_centroids[cands]
                        sims.append((zi @ c.t()).mean())
                    l_cont = -torch.stack(sims).mean()
                    loss = loss_ord + self.caft_lambda * l_cont
                    if self.state.global_step % 50 == 0:
                        self.log(
                            {
                                "L_ord": float(loss_ord),
                                "L_cont": float(l_cont),
                                "ordinal_loss": float(loss),
                            }
                        )
                else:
                    loss = loss_ord
        return (loss, bert_out) if return_outputs else loss


class OrdinalEpochBeginCallback(TrainerCallback):
    """Recompute layer centroids at the start of every epoch (real callback).

    transformers 4.57 calls ``on_epoch_begin`` on registered TrainerCallback
    OBJECTS (not on the Trainer). CAFTTrainer's hook was written as a Trainer
    method and is therefore never dispatched — registering this callback is the
    fix that makes the ordinal-CAFT continuity term actually run. It holds a
    reference to the trainer and invokes its (ragged-safe)
    ``_precompute_centroids`` before each epoch, exactly as the CAFT design
    intended ("pre-computed over the full training set at the start of each
    epoch").
    """

    def __init__(self, trainer: OrdinalTrainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        # ordinal-naive (lambda=0) never consumes the centroids; skipping the
        # ~11 min full-epithelial forward pass keeps the naive arm's runtime
        # identical to the existing CAFT naive arm.
        if self.trainer.caft_lambda != 0:
            self.trainer._precompute_centroids()


def build_ordinal_trainer(
    model,
    train_data,
    id_class_dict: dict,
    ordinal_lambda: float,
    training_args_kwargs: dict,
    output_directory: str | Path,
    token_dictionary_file: str | Path | None = None,
    eval_data=None,
) -> OrdinalTrainer:
    """Build the OrdinalTrainer; mirrors build_caft_trainer (same collator).

    Registers :class:`OrdinalEpochBeginCallback` so the continuity centroids
    are recomputed each epoch (see the note in the module docstring).
    """
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    if token_dictionary_file is None:
        from geneformer.classifier import (
            TOKEN_DICTIONARY_FILE as token_dictionary_file,  # noqa: N811  (mirrors caft.py)
        )
    with open(token_dictionary_file, "rb") as _fh:
        token_dictionary = pickle.load(_fh)
    collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    args = TrainingArguments(output_dir=str(output_directory), **training_args_kwargs)
    trainer = OrdinalTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        ordinal_lambda=ordinal_lambda,
        class_to_layer=class_to_layer_map(id_class_dict),
        class_to_ordinal=class_to_ordinal_map(id_class_dict),
        token_dictionary_file=token_dictionary_file,
        compute_metrics=None,
    )
    trainer.add_callback(OrdinalEpochBeginCallback(trainer))
    return trainer
