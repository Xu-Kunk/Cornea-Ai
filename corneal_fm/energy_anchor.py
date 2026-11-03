"""EnergyAnchor-FT — Energy-model A1 arm: MMD anchor to frozen pretrained
differentiation-stage centroids (Paper 1 energy-model extension).

Motivation (lib-6 energy-model survey, user decision 2026-08-05): free-energy
(JEM-style) fine-tuning is the WRONG tool for the corneal epithelial task
because training an EBM on the classifier logits distorts the ordinal
differentiation axis (rho). The *anchored-energy* alternative keeps the
standard softmax classifier and constrains the representation geometry
instead: pull the fine-tuned penultimate-layer [CLS] embedding back toward
the FROZEN pretrained geometry, grouped by differentiation stage.

    L = L_cls + lambda_e * mean_s  MMD^2( z_ft^{(s)},  c_ref^{(s)} )

  - z_ft^{(s)}  : fine-tuned penultimate hidden-state [CLS] embeddings of the
                  batch cells in differentiation stage s (hidden_states index
                  11 by default — the layer immediately below the head, i.e.
                  the most head-proximal layer whose rho collapses under
                  naive FT; constraining HERE mirrors the head-proximal
                  diagnostic instead of the CAFT final-representation anchor).
  - c_ref^{(s)} : FROZEN reference centroid of stage s, precomputed ONCE from
                  the freshly-loaded PRETRAINED Geneformer (identical weights
                  to the training model at init, before any optimizer step)
                  over the full training set penultimate [CLS] embeddings.
                  Detached + stop-gradient: the constraint never pulls the
                  reference.
  - MMD^2 with an RBF kernel exp(-||x-y||^2 / (2 sigma^2)); sigma = median
                  pairwise distance of a subsample of the frozen reference
                  embeddings (median heuristic) unless --kernel-bandwidth is
                  given. For a point-mass reference the MMD decomposes as
                  MMD^2 = mean_{i,j} k(z_i,z_j) + k(c,c) - 2 mean_i k(z_i,c);
                  k(c,c)=1 is a constant (frozen c) kept only for reporting.

Stage grouping reuses the locked 4-class harmonization (LSC(0) < TAC(1) <
basal(2) < differentiated(3), wing/squamous -> differentiated, non-epithelial
-> -1 masked out), identical to the D5 ordinal arm (corneal_fm.ordinal).

Orthogonality to HPA-FT (LWC): HPA-FT's layer-wise continuity constrains
INTER-layer adjacency (z^(l) vs centroids of l-1, l+1) over mid-body layers
L3-L10 — a relative, intra-model constraint. A1 constrains the ABSOLUTE
distance of ONE head-proximal layer (penultimate, L11) to the frozen
pretrained reference — an absolute, cross-model anchor. Different mechanisms,
independently sweepable (lambda_e vs hpa_lambda); combining them is future
work, not done here.

Everything else mirrors the locked arms: same tokenizer / donor split /
Classifier.prepare_data / HF training arguments / bf16 weights / group_by_length.
Only the loss placement differs (corneal_fm.energy_anchor.EnergyAnchorTrainer
instead of CAFTTrainer/HPATrainer).

NOTE on init-time reference precompute: the reference centroids are computed
from self.model at __init__, which is valid ONLY when the trainer is built on
a freshly-loaded pretrained checkpoint (the run script does this). Resuming
training from an already fine-tuned checkpoint would pollute the reference;
the run scripts do not resume.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)
from transformers import TrainingArguments

from corneal_fm.caft import NON_LAYER, CAFTTrainer, class_to_layer_map
from corneal_fm.ordinal import class_to_ordinal_map

# hidden_states index 0 = embedding output, 1..12 = transformer layers,
# 12 == last_hidden_state. Same indexing as scripts/measure_layerwise_rho.py.
N_HIDDEN_STATES = 13
FINAL_HIDDEN = N_HIDDEN_STATES - 1  # 12
# Penultimate hidden state = output of transformer layer 10 (0-indexed),
# the layer directly below the head (head-proximal collapse band L5-L12).
DEFAULT_ANCHOR_HIDDEN = FINAL_HIDDEN - 1  # 11

N_STAGES = 4  # LSC(0) < TAC(1) < basal(2) < differentiated(3)
STAGE_NAMES = ["LSC", "TAC", "basal", "differentiated"]


def _rbf_kernel_pairwise(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel matrix K_zz = exp(-||z_i - z_j||^2 / (2 sigma^2)).

    ``z`` is (n, H) float32. Computed with the |z_i|^2 + |z_j|^2 - 2 z_i.z_j
    identity (O(n^2) memory is fine: batch-size cells, n <= 8).
    """
    sq = (z * z).sum(1, keepdim=True)
    d2 = sq + sq.t() - 2.0 * (z @ z.t())
    d2 = d2.clamp(min=0.0)
    return torch.exp(-d2 / (2.0 * sigma * sigma))


def _rbf_kernel_to_centroid(z: torch.Tensor, c: torch.Tensor, sigma: float) -> torch.Tensor:
    """k(z_i, c) = exp(-||z_i - c||^2 / (2 sigma^2)) for frozen centroid c."""
    d2 = ((z - c) ** 2).sum(1)
    return torch.exp(-d2 / (2.0 * sigma * sigma))


class EnergyAnchorTrainer(CAFTTrainer):
    """HF Trainer implementing the A1 anchored-energy loss.

    Inherits CAFTTrainer's pad-id resolution / attention-mask / class-to-layer
    machinery; adds (a) a ONE-TIME init-time precompute of the frozen per-stage
    penultimate-layer reference centroids + median-heuristic RBF bandwidth and
    (b) the MMD anchor loss in compute_loss. No epoch-begin hook is needed:
    the reference is static (frozen pretrained model), unlike CAFT's per-epoch
    centroids.
    """

    def __init__(
        self,
        anchor_lambda: float,
        class_to_layer: dict,
        class_to_stage: dict,
        anchor_hidden: int = DEFAULT_ANCHOR_HIDDEN,
        kernel_bandwidth: float | None = None,
        centroid_batch: int = 512,
        ref_bandwidth_cells: int = 2048,
        token_dictionary_file: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(
            caft_lambda=0.0,  # no CAFT continuity in this arm
            class_to_layer=class_to_layer,
            n_layers=N_STAGES,
            token_dictionary_file=token_dictionary_file,
            centroid_batch=centroid_batch,
            **kwargs,
        )
        self.anchor_lambda = float(anchor_lambda)
        self.class_to_stage = {int(k): int(v) for k, v in class_to_stage.items()}
        if not (1 <= anchor_hidden <= FINAL_HIDDEN):
            raise ValueError(f"anchor_hidden must be in 1..{FINAL_HIDDEN}; got {anchor_hidden}")
        self.anchor_hidden = int(anchor_hidden)
        self.ref_bandwidth_cells = int(ref_bandwidth_cells)
        self.stage_centroids: torch.Tensor | None = None  # (N_STAGES, H) float32
        self.kernel_bandwidth: float | None = (
            float(kernel_bandwidth) if kernel_bandwidth is not None else None
        )
        self._epi_rows: list[int] | None = None
        if self.anchor_lambda != 0:
            self._precompute_reference()
        else:
            print(
                "[EnergyAnchor] anchor_lambda=0 -> reference centroids skipped "
                "(plain softmax CE arm).",
                flush=True,
            )

    # -- one-time frozen reference precompute (init-time) -------------------

    def _precompute_reference(self) -> None:
        """Frozen per-stage penultimate [CLS] centroids + median-heuristic sigma.

        Runs over the FULL training set (epithelial cells only, exactly like
        the ordinal arm's cost guard: non-epithelial cells contribute nothing
        to any stage centroid) with the model in eval/no_grad. The model is
        the freshly-loaded PRETRAINED Geneformer at __init__ (see module note);
        the run script moves it to cuda BEFORE building the trainer (the
        family fix from method_arm_gf.py — a CPU-resident model here ran the
        full-epi forward on CPU for hours and died under GB10 unified-memory
        pressure with a driver-level SIGKILL, no traceback).
        Ragged-safe and memory-bounded: each slice is right-padded with ONE
        vectorised pad_sequence op to its own max length (the raw prepared
        dataset has variable-length input_ids, and the CAFT base precompute
        crashes on ragged tensors — see corneal_fm.ordinal docstring), the epi
        rows are length-sorted and the batch is shrunk when long cells are
        present (budget = centroid_batch x 2300 tokens, mirrors
        axis_ref.compute_axis / axis_ft._precompute_frozen_embeddings).
        Centroids are float32 + detached (frozen); sigma defaults to the
        median pairwise distance over a deterministic subsample
        (ref_bandwidth_cells) of the reference embeddings.
        """
        ds = self.train_dataset
        n = len(ds)
        if n == 0:
            return
        labels_all = list(ds["label"])
        self._epi_rows = [
            i
            for i, c in enumerate(labels_all)
            if self.class_to_stage.get(int(c), NON_LAYER) != NON_LAYER
        ]
        epi = self._epi_rows
        device = self.model.device
        if torch.cuda.is_available() and device.type != "cuda":
            # Guard (family fix, mirrors method_arm_gf.py): the run script
            # moves the freshly-loaded pretrained model to cuda BEFORE
            # building the trainer, so this fires only for direct construction
            # with a CPU-resident model on a CUDA box. The full-epi reference
            # forward would otherwise run on CPU for hours — the root cause of
            # this arm's silent SIGKILL deaths (GB10 driver OOM under the
            # concurrent CPU + unified-memory load).
            print(
                "[EnergyAnchor] WARN model on cpu but CUDA is available; the "
                "reference forward will run on CPU (slow on the full set) — "
                "move the model to cuda before building the trainer.",
                flush=True,
            )
        self.model.eval()
        hidden_size = self.model.config.hidden_size
        n_hidden = int(self.model.config.num_hidden_layers)
        # The FINAL hidden state is already exposed as last_hidden_state:
        # skip output_hidden_states (which retains ALL 13 layers) unless an
        # interior anchor layer is requested — halves the per-batch footprint
        # (mirrors axis_ft._precompute_frozen_embeddings).
        need_all_layers = self.anchor_hidden < n_hidden
        sums = torch.zeros(N_STAGES, hidden_size, device=device, dtype=torch.float32)
        counts = torch.zeros(N_STAGES, device=device, dtype=torch.float32)
        # deterministic bandwidth subsample (evenly spaced indices) — taken
        # from the UN-SORTED epi ordering so the length sort for batching
        # below cannot change which cells enter the sigma heuristic
        bw_idx = set(
            epi[:: max(1, len(epi) // self.ref_bandwidth_cells)][: self.ref_bandwidth_cells]
        )
        # Sort epi rows by ascending length ONCE: attention cost grows
        # ~quadratically with the padded length, and sums/counts plus the bw
        # subsample are order-independent (mirrors axis_ref.compute_axis and
        # axis_ft._precompute_frozen_embeddings).
        has_len_col = "length" in ds.column_names
        if has_len_col:
            epi = sorted(epi, key=lambda i: int(ds["length"][i]))
        bw_emb: list[torch.Tensor] = []
        # Memory-bounded batching (mirrors axis_ref.compute_axis / axis_ft):
        # shrink the batch when long cells are present so the per-batch
        # attention peak stays inside the GB10 unified-memory pool — the
        # kernel oom_reaper SIGKILLs runs on oversized IOMMU/vmalloc mappings
        # (observed NVRM NV_ERR_NO_MEMORY, memdesc.c:1359). Length-sorted rows
        # => the slice's longest cell sits at its end. Pure batching change:
        # sums/counts are row-wise, so results are identical.
        budget = max(64, int(self.centroid_batch) * 2300)
        with torch.no_grad():
            s = 0
            batch_no = 0
            while s < len(epi):
                k = min(self.centroid_batch, len(epi) - s)
                if has_len_col:
                    while k > 32 and int(ds["length"][epi[s + k - 1]]) * k > budget:
                        k = max(32, budget // int(ds["length"][epi[s + k - 1]]))
                rows = epi[s : s + k]
                batch = ds[rows]
                ids_list = batch["input_ids"]
                lengths = batch.get("length")
                if lengths is None:
                    lengths = [len(row) for row in ids_list]
                # Right-pad the ragged slice in ONE C++-level op (pad_sequence);
                # the old per-row Python assignment loop
                # (``ids[i, :len(row)] = ...``) was O(rows) interpreter work per
                # batch and dominated the precompute (axis_ref family fix).
                ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(row, dtype=torch.long) for row in ids_list],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                length = torch.as_tensor(lengths, device=device)
                labels = torch.as_tensor(batch["label"], device=device)
                out = self.model.bert(
                    input_ids=ids,
                    attention_mask=self._attn_mask(ids, length),
                    output_hidden_states=need_all_layers,
                )
                if need_all_layers:
                    z = out.hidden_states[self.anchor_hidden][:, 0].float()
                else:
                    z = out.last_hidden_state[:, 0].float()
                for cls_id in labels.unique().tolist():
                    st = self.class_to_stage.get(int(cls_id), NON_LAYER)
                    if st == NON_LAYER:
                        continue
                    mask = labels == cls_id
                    sums[st] += z[mask].sum(0)
                    counts[st] += mask.sum()
                # collect the bandwidth subsample rows present in this slice
                for i, row in enumerate(rows):
                    if row in bw_idx:
                        bw_emb.append(z[i])
                del out
                s += k
                batch_no += 1
                # Release the caching allocator's reserved pool periodically so
                # the unified-memory driver mappings are returned to the kernel
                # (GB10 OOM-reaper fragility under vmalloc pressure).
                if batch_no % 20 == 0:
                    torch.cuda.empty_cache()
        valid = counts > 0
        centroids = torch.zeros_like(sums)
        centroids[valid] = sums[valid] / counts[valid].unsqueeze(1)
        self.stage_centroids = centroids.detach().clone()

        # -- median-heuristic bandwidth ------------------------------------
        if self.kernel_bandwidth is None:
            if len(bw_emb) >= 2:
                z_stack = torch.stack(bw_emb).detach()  # (m, hidden_size)
                # chunked pairwise squared distances -> median
                meds: list[torch.Tensor] = []
                m = z_stack.shape[0]
                chunk = 256
                sq = (z_stack * z_stack).sum(1)
                for i0 in range(0, m, chunk):
                    zi = z_stack[i0 : i0 + chunk]
                    si = sq[i0 : i0 + chunk]
                    for j0 in range(0, m, chunk):
                        zj = z_stack[j0 : j0 + chunk]
                        d2 = si[:, None] + sq[j0 : j0 + chunk][None, :] - 2.0 * (zi @ zj.t())
                        meds.append(d2.clamp(min=0.0).sqrt().flatten())
                all_d = torch.cat(meds)
                sigma = float(all_d.median())
            else:
                sigma = float("nan")
            if not torch.isfinite(torch.as_tensor(sigma)) or sigma <= 0:
                sigma = float(
                    torch.sqrt(torch.as_tensor(float(hidden_size), dtype=torch.float32)) * 0.1
                )
                print(
                    f"[EnergyAnchor] WARN degenerate bandwidth subsample; "
                    f"fallback sigma={sigma:.4f} (sqrt(H)*0.1).",
                    flush=True,
                )
            self.kernel_bandwidth = sigma
        self.model.train()
        counts_np = counts.cpu().numpy().tolist()
        print(
            f"[EnergyAnchor] frozen reference @ hidden {self.anchor_hidden}: "
            f"stage centroids={[int(c) for c in counts_np]} "
            f"sigma={self.kernel_bandwidth:.4f}",
            flush=True,
        )

    # -- anchored-energy loss ----------------------------------------------

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
        need_hidden = bool(
            self.anchor_lambda != 0 and self.stage_centroids is not None and labels is not None
        )
        bert_out = model.bert(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=need_hidden,
        )
        logits = model.classifier(model.dropout(bert_out.pooler_output))
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
        else:
            loss_cls = F.cross_entropy(
                logits, torch.zeros(input_ids.size(0), dtype=torch.long, device=device)
            )
        if not need_hidden:
            loss = loss_cls
        else:
            z = bert_out.hidden_states[self.anchor_hidden][:, 0].float()
            stages = torch.tensor(
                [self.class_to_stage.get(int(c), NON_LAYER) for c in labels.tolist()],
                device=device,
            )
            valid = stages != NON_LAYER
            terms: list[torch.Tensor] = []
            weights: list[float] = []
            if bool(valid.any()):
                sigma = float(self.kernel_bandwidth)
                for st in range(N_STAGES):
                    sel = (stages == st) & valid
                    ns = int(sel.sum())
                    if ns == 0:
                        continue
                    zs = z[sel]
                    c = self.stage_centroids[st]
                    k_zz = _rbf_kernel_pairwise(zs, sigma).mean()
                    k_zc = _rbf_kernel_to_centroid(zs, c, sigma).mean()
                    # k(c,c) = exp(0) = 1: constant, kept for a true MMD^2 value
                    terms.append(k_zz + 1.0 - 2.0 * k_zc)
                    weights.append(float(ns))
            if terms:
                w = torch.tensor(weights, device=device, dtype=torch.float32)
                l_anchor = torch.stack(terms) @ (w / w.sum())
                loss = loss_cls + self.anchor_lambda * l_anchor
                if self.state.global_step % 50 == 0:
                    self.log(
                        {
                            "L_cls": float(loss_cls.detach()),
                            "L_anchor_mmd2": float(l_anchor.detach()),
                            "energy_anchor_loss": float(loss.detach()),
                            "anchor_sigma": sigma,
                            "n_anchor_cells": int(valid.sum()),
                        }
                    )
            else:
                loss = loss_cls
        return (loss, bert_out) if return_outputs else loss


def build_energy_anchor_trainer(
    model,
    train_data,
    id_class_dict: dict,
    anchor_lambda: float,
    training_args_kwargs: dict,
    output_directory: str | Path,
    anchor_hidden: int = DEFAULT_ANCHOR_HIDDEN,
    kernel_bandwidth: float | None = None,
    token_dictionary_file: str | Path | None = None,
    eval_data=None,
) -> EnergyAnchorTrainer:
    """Build the EnergyAnchorTrainer; mirrors build_caft_trainer (same collator)."""
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    if token_dictionary_file is None:
        from geneformer.classifier import (
            TOKEN_DICTIONARY_FILE as token_dictionary_file,  # noqa: N811  (mirrors caft.py)
        )
    with open(token_dictionary_file, "rb") as _fh:
        token_dictionary = pickle.load(_fh)
    collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    args = TrainingArguments(output_dir=str(output_directory), **training_args_kwargs)
    return EnergyAnchorTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        anchor_lambda=anchor_lambda,
        anchor_hidden=anchor_hidden,
        kernel_bandwidth=kernel_bandwidth,
        class_to_layer=class_to_layer_map(id_class_dict),
        class_to_stage=class_to_ordinal_map(id_class_dict),
        token_dictionary_file=token_dictionary_file,
        compute_metrics=None,
    )
